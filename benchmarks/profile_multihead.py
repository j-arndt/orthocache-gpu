#!/usr/bin/env python3
"""OrthoCache Multi-Head Benchmark — Hero Figure Generator.

Profiles the ACTUAL inference scenario: multi-head attention (32 heads)
with Split-K V2 distributing all heads × splits across 24 SMs.

This is the benchmark that produces the real speedup numbers, because:
  - Dense: processes 32 heads sequentially (or SDPA batched)
  - Split-K V2: grid=(num_heads, num_splits) → all SMs saturated

Usage:
    python benchmarks/profile_multihead.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orthocache_gpu.triton_kernels.fwht_fused_prototype import (
    triton_fwht_eviction,
    generate_walsh_matrix,
    TILE_SIZE,
    BAND_LOW_64,
    BAND_HIGH_64,
)
from orthocache_gpu.triton_kernels.fused_eviction import (
    fused_orthocache_attention_v2,
)


# ---------------------------------------------------------------------------
# Configuration — matches LLM inference decode step
# ---------------------------------------------------------------------------

NUM_HEADS = 32          # Standard transformer (e.g., LLaMA-7B, Mistral)
HEAD_DIM = 128          # Standard head dimension
SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768]
EVICTION_RATE = 0.50    # 50% eviction — the standard comparison point
NUM_WARMUP = 10
NUM_ITERS = 25          # More iterations for tighter statistics


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_multihead_kv(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    eviction_rate: float,
    device: torch.device,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Generate multi-head Q, K, V with controllable eviction rate.

    Returns:
        (q, keys, values, zeta_max) where:
        - q: (num_heads, head_dim)
        - keys: (num_heads, seq_len, head_dim)
        - values: (num_heads, seq_len, head_dim)
        - zeta_max: calibrated threshold for target eviction rate
    """
    torch.manual_seed(seed)
    num_tiles = seq_len // TILE_SIZE

    q = torch.randn(num_heads, head_dim, device=device, dtype=torch.float32)

    num_noise = max(0, int(num_tiles * eviction_rate))
    num_semantic = num_tiles - num_noise

    W = generate_walsh_matrix(TILE_SIZE).to(device)

    all_keys = []
    for h in range(num_heads):
        tiles = []
        # Semantic tiles (low ζ)
        for _ in range(num_semantic):
            coeffs = torch.zeros(TILE_SIZE, head_dim, device=device)
            coeffs[:8, :] = torch.randn(8, head_dim, device=device) * 2.0
            tiles.append(W.T @ coeffs)
        # Noise tiles (high ζ)
        for _ in range(num_noise):
            tiles.append(torch.randn(TILE_SIZE, head_dim, device=device) * 0.1)
        # Shuffle deterministically per head
        torch.manual_seed(seed + h + 1)
        perm = torch.randperm(num_tiles, device=device)
        tiles = [tiles[i] for i in perm.tolist()]
        all_keys.append(torch.cat(tiles, dim=0))

    keys = torch.stack(all_keys)  # (num_heads, seq_len, head_dim)
    values = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=torch.float32)

    # Calibrate zeta_max using first head
    with torch.no_grad():
        _, zeta_vals = triton_fwht_eviction(keys[0], zeta_max=1e9, return_zeta=True)
        if zeta_vals is not None:
            sorted_z, _ = torch.sort(zeta_vals)
            keep_idx = min(num_semantic, num_tiles - 1)
            zeta_max = float(
                (sorted_z[keep_idx] + sorted_z[min(keep_idx + 1, num_tiles - 1)]) / 2
            )
        else:
            zeta_max = 1.0

    return q, keys, values, zeta_max


# ---------------------------------------------------------------------------
# Attention implementations
# ---------------------------------------------------------------------------

def dense_multihead_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Standard dense multi-head attention (no eviction).

    q: (num_heads, head_dim)
    keys: (num_heads, seq_len, head_dim)
    values: (num_heads, seq_len, head_dim)
    """
    head_dim = q.shape[-1]
    scale = 1.0 / (head_dim ** 0.5)

    # (num_heads, 1, head_dim) × (num_heads, head_dim, seq_len) → (num_heads, 1, seq_len)
    logits = torch.bmm(q.unsqueeze(1).float(), keys.float().transpose(1, 2)) * scale
    weights = F.softmax(logits, dim=-1)
    # (num_heads, 1, seq_len) × (num_heads, seq_len, head_dim) → (num_heads, 1, head_dim)
    out = torch.bmm(weights, values.float())
    return out.squeeze(1)  # (num_heads, head_dim)


def splitk_multihead_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    zeta_max: float,
) -> torch.Tensor:
    """Split-K V2 multi-head attention with spectral eviction."""
    out, _ = fused_orthocache_attention_v2(q, keys, values, zeta_max=zeta_max)
    return out


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def time_fn(fn, device, num_warmup=NUM_WARMUP, num_iters=NUM_ITERS):
    """CUDA Events timing with proper warmup."""
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = fn()
            torch.cuda.synchronize()

        times = []
        for _ in range(num_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            _ = fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    arr = np.array(times)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "median_ms": float(np.median(arr)),
        "p5_ms": float(np.percentile(arr, 5)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("ERROR: CUDA required for this benchmark.")
        sys.exit(1)

    props = torch.cuda.get_device_properties(device)
    print("=" * 72)
    print("  OrthoCache Multi-Head Benchmark (Hero Figure)")
    print("=" * 72)
    print(f"  GPU        : {props.name}")
    print(f"  SMs        : {props.multi_processor_count}")
    print(f"  VRAM       : {props.total_memory / (1024**3):.1f} GB")
    print(f"  Heads      : {NUM_HEADS}")
    print(f"  Head dim   : {HEAD_DIM}")
    print(f"  Eviction   : {EVICTION_RATE:.0%}")
    print(f"  Tile size  : {TILE_SIZE}")
    print(f"  Warmup     : {NUM_WARMUP}")
    print(f"  Iterations : {NUM_ITERS}")
    print()

    results = []

    for seq_len in SEQ_LENS:
        num_tiles = seq_len // TILE_SIZE
        print(f"--- seq_len={seq_len:>6,} ({num_tiles} tiles) ---")

        q, keys, values, zeta_max = generate_multihead_kv(
            seq_len, NUM_HEADS, HEAD_DIM, EVICTION_RATE, device
        )

        # Dense multi-head
        print(f"  Dense (32-head)               ... ", end="", flush=True)
        dense_stats = time_fn(
            lambda: dense_multihead_attention(q, keys, values),
            device,
        )
        print(f"mean={dense_stats['mean_ms']:.4f} ± {dense_stats['std_ms']:.4f} ms")

        # Split-K V2 multi-head
        print(f"  Split-K V2 (32-head)          ... ", end="", flush=True)
        splitk_stats = time_fn(
            lambda: splitk_multihead_attention(q, keys, values, zeta_max),
            device,
        )
        print(f"mean={splitk_stats['mean_ms']:.4f} ± {splitk_stats['std_ms']:.4f} ms")

        speedup = dense_stats["mean_ms"] / splitk_stats["mean_ms"]
        print(f"  → Speedup: {speedup:.2f}×")
        print()

        results.append({
            "seq_len": seq_len,
            "num_tiles": num_tiles,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "eviction_rate": EVICTION_RATE,
            "dense_mean_ms": dense_stats["mean_ms"],
            "dense_std_ms": dense_stats["std_ms"],
            "dense_median_ms": dense_stats["median_ms"],
            "dense_p5_ms": dense_stats["p5_ms"],
            "dense_p95_ms": dense_stats["p95_ms"],
            "splitk_mean_ms": splitk_stats["mean_ms"],
            "splitk_std_ms": splitk_stats["std_ms"],
            "splitk_median_ms": splitk_stats["median_ms"],
            "splitk_p5_ms": splitk_stats["p5_ms"],
            "splitk_p95_ms": splitk_stats["p95_ms"],
            "speedup": speedup,
            "gpu_name": props.name,
            "sm_count": props.multi_processor_count,
        })

        del q, keys, values
        torch.cuda.empty_cache()

    # Save results
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "multihead_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Summary table
    print()
    print("=" * 72)
    print(f"  {'SeqLen':>8} {'Dense (ms)':>12} {'Split-K (ms)':>13} {'Speedup':>9}")
    print("-" * 72)
    for r in results:
        marker = " ★" if r["speedup"] > 5 else ""
        print(
            f"  {r['seq_len']:>8,} "
            f"{r['dense_mean_ms']:>12.4f} "
            f"{r['splitk_mean_ms']:>13.4f} "
            f"{r['speedup']:>8.2f}×{marker}"
        )
    print()


if __name__ == "__main__":
    main()
