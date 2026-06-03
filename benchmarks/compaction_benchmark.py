"""OrthoCache GPU Stream Compaction Benchmark.

Measures stream_compact and stream_decompact timing at various eviction rates
and sequence lengths. Uses the real compaction primitives from
orthocache_gpu.compaction.

Sweep parameters:
  - Eviction rates: [0.25, 0.50, 0.625, 0.75, 0.875]
  - Sequence lengths: [4096, 8192, 16384, 32768]

Outputs
-------
* Timing comparison table on stdout
* JSON results → benchmarks/results/compaction_results.json

Usage
-----
    python benchmarks/compaction_benchmark.py
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

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orthocache_gpu.compaction import stream_compact, stream_decompact


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BLOCK_SIZE = 512
EVICTION_RATES = [0.25, 0.50, 0.625, 0.75, 0.875]
SEQ_LENS = [4096, 8192, 16384, 32768]
NUM_HEADS = 8
HEAD_DIM = 128
NUM_WARMUP = 3
NUM_ITERS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_block_mask(
    num_blocks: int,
    num_heads: int,
    eviction_rate: float,
    device: torch.device,
    seed: int = 42,
) -> torch.Tensor:
    """Generate a random block mask with approximately the target eviction rate.

    Args:
        num_blocks: Number of blocks.
        num_heads: Number of attention heads.
        eviction_rate: Fraction of blocks to evict (0.0 = keep all, 1.0 = evict all).
        device: Target device.
        seed: Random seed.

    Returns:
        Boolean mask of shape (num_blocks, num_heads). True = retained.
    """
    torch.manual_seed(seed)
    # Generate per-block retention probability
    retain_prob = 1.0 - eviction_rate
    mask = torch.rand(num_blocks, num_heads, device=device) < retain_prob
    # Ensure at least one block is retained
    if not mask.any():
        mask[0, :] = True
    return mask


def time_compact(
    keys: torch.Tensor,
    values: torch.Tensor,
    block_mask: torch.Tensor,
    device: torch.device,
    num_warmup: int,
    num_iters: int,
) -> dict[str, float]:
    """Time the stream_compact operation."""
    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            _ = stream_compact(keys, values, block_mask, BLOCK_SIZE)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Measured
        times = []
        for _ in range(num_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            compact_k, compact_v, active_idx, num_active = stream_compact(
                keys, values, block_mask, BLOCK_SIZE
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    arr = np.array(times) * 1000
    return {
        "compact_mean_ms": float(np.mean(arr)),
        "compact_std_ms": float(np.std(arr)),
        "num_active": int(num_active),
    }


def time_decompact(
    compact_output: torch.Tensor,
    active_indices: torch.Tensor,
    num_active: torch.Tensor,
    num_blocks: int,
    device: torch.device,
    num_warmup: int,
    num_iters: int,
) -> dict[str, float]:
    """Time the stream_decompact operation."""
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = stream_decompact(compact_output, active_indices, num_active, num_blocks, BLOCK_SIZE)
            if device.type == "cuda":
                torch.cuda.synchronize()

        times = []
        for _ in range(num_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = stream_decompact(compact_output, active_indices, num_active, num_blocks, BLOCK_SIZE)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    arr = np.array(times) * 1000
    return {
        "decompact_mean_ms": float(np.mean(arr)),
        "decompact_std_ms": float(np.std(arr)),
    }


def time_attention_on_compact(
    q: torch.Tensor,
    compact_keys: torch.Tensor,
    compact_values: torch.Tensor,
    num_active: int,
    num_blocks: int,
    device: torch.device,
    num_warmup: int,
    num_iters: int,
) -> dict[str, float]:
    """Time attention on the compacted tensor."""
    head_dim = q.shape[-1]
    num_heads = q.shape[1]
    active_tokens = num_active * BLOCK_SIZE

    with torch.no_grad():
        k_flat = compact_keys.reshape(num_blocks * BLOCK_SIZE, num_heads, head_dim)[:active_tokens]
        v_flat = compact_values.reshape(num_blocks * BLOCK_SIZE, num_heads, head_dim)[:active_tokens]

        def run_attention():
            scale = torch.sqrt(torch.tensor(float(head_dim), device=q.device))
            logits = torch.einsum("qhd,khd->qkh", q.float(), k_flat.float()) / scale
            weights = F.softmax(logits, dim=1)
            return torch.einsum("qkh,khd->qhd", weights, v_flat.float())

        # Warmup
        for _ in range(num_warmup):
            _ = run_attention()
            if device.type == "cuda":
                torch.cuda.synchronize()

        times = []
        for _ in range(num_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = run_attention()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    arr = np.array(times) * 1000
    return {
        "attention_mean_ms": float(np.mean(arr)),
        "attention_std_ms": float(np.std(arr)),
    }


def time_dense_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    device: torch.device,
    num_warmup: int,
    num_iters: int,
) -> dict[str, float]:
    """Time full dense attention (baseline)."""
    head_dim = q.shape[-1]

    with torch.no_grad():
        def run():
            scale = torch.sqrt(torch.tensor(float(head_dim), device=q.device))
            logits = torch.einsum("qhd,khd->qkh", q.float(), keys.float()) / scale
            weights = F.softmax(logits, dim=1)
            return torch.einsum("qkh,khd->qhd", weights, values.float())

        for _ in range(num_warmup):
            _ = run()
            if device.type == "cuda":
                torch.cuda.synchronize()

        times = []
        for _ in range(num_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = run()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    arr = np.array(times) * 1000
    return {
        "dense_mean_ms": float(np.mean(arr)),
        "dense_std_ms": float(np.std(arr)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_compaction_benchmark() -> list[dict]:
    """Run the complete compaction benchmark sweep."""
    device = get_device()
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  OrthoCache GPU Stream Compaction Benchmark")
    print("=" * 70)
    print(f"  Device         : {device}")
    if device.type == "cuda":
        print(f"  GPU            : {torch.cuda.get_device_name()}")
    print(f"  Eviction rates : {EVICTION_RATES}")
    print(f"  Seq lengths    : {SEQ_LENS}")
    print(f"  Block size     : {BLOCK_SIZE}")
    print(f"  Warmup         : {NUM_WARMUP}")
    print(f"  Iterations     : {NUM_ITERS}")
    if device.type == "cpu":
        print("  NOTE: Running on CPU fallback")
    print()

    all_results: list[dict] = []

    for seq_len in SEQ_LENS:
        num_blocks = seq_len // BLOCK_SIZE
        print(f"--- Sequence length: {seq_len} ({num_blocks} blocks) ---")

        # Generate synthetic data
        torch.manual_seed(42)
        keys = torch.randn(seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
        values = torch.randn(seq_len, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
        queries = torch.randn(16, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)

        # Dense baseline
        dense_stats = time_dense_attention(queries, keys, values, device, NUM_WARMUP, NUM_ITERS)
        dense_ms = dense_stats["dense_mean_ms"]
        print(f"  Dense baseline: {dense_ms:.3f} ms")

        for eviction_rate in EVICTION_RATES:
            print(f"  Eviction {eviction_rate:.1%}: ", end="", flush=True)

            # Generate mask
            mask = generate_block_mask(num_blocks, NUM_HEADS, eviction_rate, device)

            # Time compaction
            compact_stats = time_compact(keys, values, mask, device, NUM_WARMUP, NUM_ITERS)

            # Run compaction once to get outputs for decompact + attention timing
            with torch.no_grad():
                ck, cv, ai, na = stream_compact(keys, values, mask, BLOCK_SIZE)

            # Time decompaction
            decompact_stats = time_decompact(ck, ai, na, num_blocks, device, NUM_WARMUP, NUM_ITERS)

            # Time attention on compact
            attn_stats = time_attention_on_compact(
                queries, ck, cv, int(na), num_blocks, device, NUM_WARMUP, NUM_ITERS
            )

            total_ms = compact_stats["compact_mean_ms"] + attn_stats["attention_mean_ms"]
            speedup = dense_ms / total_ms if total_ms > 0 else float("inf")

            print(
                f"compact={compact_stats['compact_mean_ms']:.3f}ms "
                f"+ attn={attn_stats['attention_mean_ms']:.3f}ms "
                f"= {total_ms:.3f}ms "
                f"(speedup={speedup:.2f}x, active={compact_stats['num_active']}/{num_blocks})"
            )

            result = {
                "seq_len": seq_len,
                "num_blocks": num_blocks,
                "eviction_rate": eviction_rate,
                "num_active": compact_stats["num_active"],
                **compact_stats,
                **decompact_stats,
                **attn_stats,
                **dense_stats,
                "total_orthocache_ms": total_ms,
                "speedup": speedup,
            }
            all_results.append(result)
        print()

    # Save results
    json_path = output_dir / "compaction_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results written to {json_path.resolve()}")
    return all_results


if __name__ == "__main__":
    run_compaction_benchmark()
