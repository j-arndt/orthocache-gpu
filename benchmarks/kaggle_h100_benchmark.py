#!/usr/bin/env python3
"""OrthoCache GPU Benchmark — Datacenter Cost & Memory Efficiency.

Runs the REAL OrthoCache pipeline (orthocache_forward) with two-gate eviction:
  Gate 1: Query-aware logit bound >= tau
  Gate 2: Spectral decay ratio zeta <= zeta_max

Uses structured synthetic data with realistic ζ distribution matching
the paper's measurements on Gemma 4 31B (Table II):
  - Global layers: ζ_mean=5.45, σ=0.40
  - Sliding layers: ζ_mean=5.65, σ=1.07

Measures:
  - TV distance & reconstruction error (quality)
  - VRAM savings (memory efficiency)
  - Concurrent sequences per GPU (throughput)
  - Fleet-level cost/energy projection

Usage (Kaggle notebook cell):
    from kaggle_h100_benchmark import run_kaggle_benchmark
    results = run_kaggle_benchmark()
"""

from __future__ import annotations

import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

try:
    import orthocache_gpu
except ImportError:
    _pkg_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_pkg_root / "src"))
    import orthocache_gpu

import gc
import json
import math
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

# The actual OrthoCache pipeline — two-gate eviction, auto-tau, the works
from orthocache_gpu.pipeline import orthocache_forward


# ═════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════

BLOCK_SIZE = 512
NUM_HEADS = 4       # Matches paper: Gemma 4 has 4 global KV heads
HEAD_DIM = 128      # Matches paper
QUERY_LEN = 16      # Matches paper benchmarks

SEQ_LENS = [4096, 8192, 16384, 32768, 65536, 131072]

# zeta_max sweep — the paper uses 5.0 as default, but we test a range
# to show the quality-efficiency tradeoff
ZETA_MAX_VALUES = [3.0, 5.0, 8.0, 12.0]

NUM_WARMUP = 3
NUM_MEASURED = 10
SEED = 42

# Datacenter cost assumptions (from paper Table VII methodology)
GPU_HOUR_COST = {'H100': 3.50, 'A100': 2.20, 'RTX PRO 6000': 1.80, 'L40S': 1.50}
WATTS_PER_GPU = {'H100': 700, 'A100': 400, 'RTX PRO 6000': 300, 'L40S': 350}
COST_PER_KWH = 0.08


# ═════════════════════════════════════════════════════════════════════
# Structured Synthetic Data Generator
# ═════════════════════════════════════════════════════════════════════

def generate_structured_kv(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int = 512,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate KV-cache data with realistic spectral structure.

    Mimics the ζ distribution measured on Gemma 4 (Table II of the paper):
    some blocks are semantically coherent (low ζ, strong low-frequency energy)
    and others are noise-dominated (high ζ, strong high-frequency energy).

    This ensures the two-gate eviction has meaningful work to do,
    unlike random data where all blocks have identical ζ ≈ 4.0.

    Strategy:
      - ~30% of blocks: "semantic" — smooth low-frequency signal (ζ < 2)
      - ~40% of blocks: "mixed"    — balanced spectrum (ζ ≈ 3-6)
      - ~20% of blocks: "noise"    — high-frequency dominated (ζ > 8)
      - ~10% of blocks: "anchor"   — very strong coherent signal (ζ < 0.5)
        These are the attention sinks / critical context blocks.

    The query is constructed to attend strongly to anchor+semantic blocks,
    weakly to mixed blocks, and negligibly to noise blocks — mimicking
    real attention patterns where most tokens attend to a few key positions.
    """
    torch.manual_seed(seed)
    num_blocks = seq_len // block_size

    keys = torch.zeros(seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
    values = torch.zeros(seq_len, num_heads, head_dim, device=device, dtype=torch.float32)

    # Assign block types
    block_types = []
    for b in range(num_blocks):
        r = (b * 7 + seed) % 10  # deterministic but varied assignment
        if r < 1:       # 10% anchor
            block_types.append('anchor')
        elif r < 4:     # 30% semantic
            block_types.append('semantic')
        elif r < 8:     # 40% mixed
            block_types.append('mixed')
        else:           # 20% noise
            block_types.append('noise')

    for b, btype in enumerate(block_types):
        start = b * block_size
        end = start + block_size
        t = torch.linspace(0, 1, block_size, device=device).unsqueeze(-1).unsqueeze(-1)
        # t: (block_size, 1, 1) — position within block

        if btype == 'anchor':
            # Very strong, smooth signal — low-frequency dominated
            # Creates a clear DC component + gentle low-freq oscillation
            base = torch.randn(1, num_heads, head_dim, device=device) * 2.0
            keys[start:end] = base + 0.3 * torch.sin(2 * math.pi * t * 2) * \
                torch.randn(1, num_heads, head_dim, device=device) * 0.5
            values[start:end] = base + 0.2 * torch.randn(block_size, num_heads, head_dim, device=device)

        elif btype == 'semantic':
            # Coherent signal with moderate variation — low ζ
            base = torch.randn(1, num_heads, head_dim, device=device) * 1.0
            smooth = torch.zeros(block_size, num_heads, head_dim, device=device)
            # Add a few low-frequency components
            for freq in [1, 2, 3, 5]:
                smooth += torch.sin(2 * math.pi * t * freq) * \
                    torch.randn(1, num_heads, head_dim, device=device) * (0.4 / freq)
            keys[start:end] = base + smooth + \
                torch.randn(block_size, num_heads, head_dim, device=device) * 0.1
            values[start:end] = base + smooth * 0.5 + \
                torch.randn(block_size, num_heads, head_dim, device=device) * 0.15

        elif btype == 'mixed':
            # Balanced spectrum — moderate ζ (3-6)
            base = torch.randn(1, num_heads, head_dim, device=device) * 0.5
            keys[start:end] = base + \
                torch.randn(block_size, num_heads, head_dim, device=device) * 0.5
            values[start:end] = base + \
                torch.randn(block_size, num_heads, head_dim, device=device) * 0.5

        else:  # noise
            # High-frequency dominated — high ζ
            # Alternating signs create high-sequency energy
            noise = torch.randn(block_size, num_heads, head_dim, device=device)
            sign_flip = torch.ones(block_size, 1, 1, device=device)
            sign_flip[1::2] = -1  # alternating +/- creates max-sequency pattern
            keys[start:end] = noise * 0.3 + sign_flip * \
                torch.randn(block_size, num_heads, head_dim, device=device) * 0.4
            values[start:end] = torch.randn(block_size, num_heads, head_dim, device=device) * 0.3

    # Query: attends to anchor/semantic blocks (those with strong DC)
    # This creates realistic peaked attention rather than uniform
    q_base = torch.zeros(QUERY_LEN, num_heads, head_dim, device=device)
    anchor_count = 0
    for b, btype in enumerate(block_types):
        if btype in ('anchor', 'semantic'):
            start = b * block_size
            # Query is a noisy average of anchor/semantic block means
            block_mean = keys[start:start+block_size].mean(dim=0, keepdim=True)
            q_base += block_mean * (2.0 if btype == 'anchor' else 0.5)
            anchor_count += 1
    if anchor_count > 0:
        q_base /= anchor_count
    q = q_base + torch.randn(QUERY_LEN, num_heads, head_dim, device=device) * 0.3

    return keys.to(dtype), values.to(dtype), q.to(dtype)


# ═════════════════════════════════════════════════════════════════════
# Hardware Detection
# ═════════════════════════════════════════════════════════════════════

def detect_hardware() -> dict:
    """Detect GPU hardware."""
    info: dict = {}
    if not torch.cuda.is_available():
        info["gpu_name"] = "none"
        return info

    props = torch.cuda.get_device_properties(0)
    vram_free, vram_total = torch.cuda.mem_get_info(0)

    _bw = {'H100': 3350, 'A100': 2039, 'H200': 4800, 'RTX PRO 6000': 960,
            'RTX 6000 Ada': 960, 'RTX 4090': 1008, 'L40S': 864, 'L4': 300}
    hbm_bw = next((bw for p, bw in _bw.items() if p.lower() in props.name.lower()), 0)

    try:
        import triton
        triton_ver = triton.__version__
    except ImportError:
        triton_ver = "n/a"

    info.update({
        "gpu_name": props.name,
        "sm_version": f"{props.major}.{props.minor}",
        "vram_total_gb": round(vram_total / (1024**3), 2),
        "vram_free_gb": round(vram_free / (1024**3), 2),
        "vram_total_bytes": vram_total,
        "num_sms": props.multi_processor_count,
        "hbm_bandwidth_gbps": hbm_bw,
        "cuda_version": torch.version.cuda or "n/a",
        "triton_version": triton_ver,
        "pytorch_version": torch.__version__,
    })

    print("=" * 72)
    print("  OrthoCache GPU Benchmark — Datacenter Cost & Memory")
    print("=" * 72)
    print(f"  GPU            : {info['gpu_name']}")
    print(f"  VRAM           : {info['vram_total_gb']:.1f} GB total, {info['vram_free_gb']:.1f} GB free")
    print(f"  SMs            : {info['num_sms']} | SM {info['sm_version']}")
    print(f"  HBM BW         : ~{info['hbm_bandwidth_gbps']} GB/s")
    print(f"  CUDA {info['cuda_version']} | Triton {info['triton_version']} | PyTorch {info['pytorch_version']}")
    print("=" * 72)
    print()
    return info


# ═════════════════════════════════════════════════════════════════════
# TV Distance
# ═════════════════════════════════════════════════════════════════════

def compute_tv_distance(alpha: torch.Tensor, alpha_hat: torch.Tensor) -> float:
    """Total Variation distance between two attention distributions."""
    return 0.5 * float(torch.sum(torch.abs(alpha - alpha_hat)).item())


# ═════════════════════════════════════════════════════════════════════
# Memory Accounting
# ═════════════════════════════════════════════════════════════════════

def kv_cache_bytes(seq_len: int, num_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
    return 2 * seq_len * num_heads * head_dim * dtype_bytes


def max_concurrent_seqs(vram_bytes: int, seq_len: int, num_heads: int,
                         head_dim: int, overhead_gb: float = 2.0) -> int:
    usable = vram_bytes - int(overhead_gb * 1024**3)
    per_seq = kv_cache_bytes(seq_len, num_heads, head_dim)
    return max(0, usable // per_seq) if per_seq > 0 else 0


# ═════════════════════════════════════════════════════════════════════
# GPU Timer
# ═════════════════════════════════════════════════════════════════════

def gpu_timer(fn, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    arr = np.array(times)
    return float(np.mean(arr)), float(np.std(arr))


# ═════════════════════════════════════════════════════════════════════
# Core Evaluation
# ═════════════════════════════════════════════════════════════════════

def evaluate_config(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    zeta_max: float,
    vram_total_bytes: int,
) -> dict:
    """Run OrthoCache pipeline and measure quality + efficiency metrics."""
    device = q.device
    seq_len_k = keys.shape[0]

    # --- Dense baseline ---
    with torch.no_grad():
        dense_out, dense_meta = orthocache_forward(
            q.float(), keys.float(), values.float(),
            block_size=BLOCK_SIZE, mode='dense'
        )

    # --- OrthoCache with two-gate eviction ---
    with torch.no_grad():
        ortho_out, ortho_meta = orthocache_forward(
            q.float(), keys.float(), values.float(),
            block_size=BLOCK_SIZE, zeta_max=zeta_max, mode='compact'
        )

    # --- Quality metrics ---
    # Reconstruction error (Frobenius, relative)
    recon_err = float(
        torch.linalg.norm(dense_out - ortho_out) /
        torch.linalg.norm(dense_out)
    )

    # TV distance: compute full attention distributions and compare
    head_dim = q.shape[-1]
    scale = math.sqrt(head_dim)
    with torch.no_grad():
        # Dense attention weights
        logits = torch.einsum('qhd,khd->qkh', q.float(), keys.float()) / scale
        dense_weights = F.softmax(logits, dim=1)

        # Sparse: mask out evicted blocks, recompute softmax
        unified_mask = torch.any(
            torch.zeros(ortho_meta['num_blocks'], NUM_HEADS, dtype=torch.bool, device=device),
            dim=-1
        )
        # Reconstruct the block mask from metadata
        # We need to re-run the mask computation
        from orthocache_gpu.spectral_energy import compute_multiband_mask
        from orthocache_gpu.pipeline import _compute_auto_tau
        tau = _compute_auto_tau(q.float(), keys.float(), BLOCK_SIZE)
        block_mask = compute_multiband_mask(q.float(), keys.float(), float(tau), zeta_max, BLOCK_SIZE)
        # Expand to token level
        unified_block = torch.any(block_mask, dim=-1)  # (num_blocks,)
        token_mask = unified_block.repeat_interleave(BLOCK_SIZE)  # (seq_len_k,)
        mask_broad = token_mask[None, :, None].expand_as(logits)
        logits_masked = torch.where(mask_broad, logits, torch.tensor(-1e9, device=device))
        sparse_weights = F.softmax(logits_masked, dim=1)

        # TV distance averaged over query positions and heads
        tv_total = 0.0
        seq_len_q, _, num_h = dense_weights.shape
        for qi in range(seq_len_q):
            for hi in range(num_h):
                tv_total += compute_tv_distance(
                    dense_weights[qi, :, hi],
                    sparse_weights[qi, :, hi]
                )
        tv_mean = tv_total / (seq_len_q * num_h)

    # --- Truncation Bound verification ---
    # TV <= |S^c| * exp(tau - z_max)
    bound_violations = 0
    with torch.no_grad():
        logits_np = logits.cpu().numpy()
        token_mask_np = token_mask.cpu().numpy()
        for qi in range(seq_len_q):
            for hi in range(num_h):
                retained = logits_np[qi, token_mask_np, hi]
                evicted = logits_np[qi, ~token_mask_np, hi]
                if evicted.size == 0 or retained.size == 0:
                    continue
                z_max = float(np.max(retained))
                s_c = evicted.size
                bound = s_c * np.exp(float(tau) - z_max)
                measured = compute_tv_distance(
                    dense_weights[qi, :, hi].cpu(),
                    sparse_weights[qi, :, hi].cpu()
                )
                if measured > bound + 1e-6:
                    bound_violations += 1

    # --- Memory efficiency ---
    eviction_rate = ortho_meta['eviction_rate']
    retained_tokens = ortho_meta['blocks_retained'] * BLOCK_SIZE
    full_bytes = kv_cache_bytes(seq_len_k, NUM_HEADS, HEAD_DIM)
    compact_bytes = kv_cache_bytes(retained_tokens, NUM_HEADS, HEAD_DIM)
    saved_bytes = full_bytes - compact_bytes
    memory_reduction = saved_bytes / full_bytes * 100 if full_bytes > 0 else 0

    # Concurrent sequence capacity
    dense_conc = max_concurrent_seqs(vram_total_bytes, seq_len_k, NUM_HEADS, HEAD_DIM)
    ortho_conc = max_concurrent_seqs(vram_total_bytes, retained_tokens, NUM_HEADS, HEAD_DIM)
    throughput_gain = ortho_conc / max(1, dense_conc)

    return {
        "seq_len": seq_len_k,
        "zeta_max": zeta_max,
        "tau": round(float(tau), 4),
        "eviction_rate": round(eviction_rate, 4),
        "blocks_retained": ortho_meta['blocks_retained'],
        "blocks_evicted": ortho_meta['blocks_evicted'],
        "num_blocks": ortho_meta['num_blocks'],
        # Quality
        "tv_distance": round(tv_mean, 6),
        "reconstruction_error": round(recon_err, 6),
        "bound_violations": bound_violations,
        # ζ distribution
        "zeta_mean": round(ortho_meta.get('zeta_mean', 0), 4),
        "zeta_std": round(ortho_meta.get('zeta_std', 0), 4),
        "zeta_min": round(ortho_meta.get('zeta_min', 0), 4),
        "zeta_max_observed": round(ortho_meta.get('zeta_max_observed', 0), 4),
        # Perfect eviction
        "perfect_eviction_blocks": ortho_meta.get('perfect_eviction_blocks', 0),
        "perfect_eviction_rate": round(ortho_meta.get('perfect_eviction_rate', 0), 4),
        # Memory
        "full_kv_bytes": full_bytes,
        "compact_kv_bytes": compact_bytes,
        "vram_saved_mb": round(saved_bytes / (1024**2), 1),
        "memory_reduction_pct": round(memory_reduction, 1),
        "dense_concurrent_seqs": dense_conc,
        "ortho_concurrent_seqs": ortho_conc,
        "throughput_gain_x": round(throughput_gain, 2),
        # Timing
        "spectral_ms": round(ortho_meta.get('spectral_ms', 0), 3),
        "attention_ms": round(ortho_meta.get('attention_ms', 0), 3),
        "total_ms": round(ortho_meta.get('total_ms', 0), 3),
        "dense_ms": round(dense_meta.get('latency_ms', 0), 3),
    }


# ═════════════════════════════════════════════════════════════════════
# Cost Projection (from paper Table VII methodology)
# ═════════════════════════════════════════════════════════════════════

def compute_cost_projection(gpu_name: str, results: list[dict]) -> dict:
    gpu_key = next((k for k in GPU_HOUR_COST if k.lower() in gpu_name.lower()), None)
    if not gpu_key:
        return {"note": "GPU not in cost table"}

    hourly = GPU_HOUR_COST[gpu_key]
    watts = WATTS_PER_GPU[gpu_key]
    projections = []

    for r in results:
        d, o = r['dense_concurrent_seqs'], r['ortho_concurrent_seqs']
        if d == 0 or o == 0:
            continue
        target = 1000
        gpus_d = math.ceil(target / d)
        gpus_o = math.ceil(target / o)
        monthly_d = gpus_d * hourly * 730
        monthly_o = gpus_o * hourly * 730
        kwh_d = gpus_d * watts * 730 / 1000
        kwh_o = gpus_o * watts * 730 / 1000
        projections.append({
            "seq_len": r["seq_len"], "zeta_max": r["zeta_max"],
            "eviction_rate": r["eviction_rate"],
            "dense_per_gpu": d, "ortho_per_gpu": o,
            "throughput_gain_x": r["throughput_gain_x"],
            "gpus_dense": gpus_d, "gpus_ortho": gpus_o,
            "gpus_saved": gpus_d - gpus_o,
            "monthly_dense": round(monthly_d), "monthly_ortho": round(monthly_o),
            "monthly_savings": round(monthly_d - monthly_o),
            "annual_savings": round((monthly_d - monthly_o) * 12),
            "kwh_saved_monthly": round(kwh_d - kwh_o),
        })

    return {"gpu": gpu_key, "hourly": hourly, "watts": watts, "projections": projections}


# ═════════════════════════════════════════════════════════════════════
# Output
# ═════════════════════════════════════════════════════════════════════

def get_output_path() -> Path:
    kaggle = Path("/kaggle/working")
    if kaggle.exists():
        return kaggle / "orthocache_results.json"
    local = Path(__file__).resolve().parent / "results"
    local.mkdir(parents=True, exist_ok=True)
    return local / "orthocache_results.json"


# ═════════════════════════════════════════════════════════════════════
# Summary Tables
# ═════════════════════════════════════════════════════════════════════

def print_quality_table(results: list[dict]) -> None:
    print()
    print("=" * 105)
    print("  QUALITY — TV Distance & Reconstruction Error")
    print("  (Paper Table VI reference: 50% evict → TV≈0.50, recon_err≈0.018)")
    print("=" * 105)
    print(f"{'SeqLen':>8} {'ζ_max':>6} {'Evict%':>7} {'TV Dist':>9} {'ReconErr':>9} "
          f"{'Bound✓':>7} {'ζ_mean':>7} {'ζ_std':>7} {'PE%':>5} {'Retained':>10}")
    print("-" * 105)
    for r in results:
        bv = "✓" if r['bound_violations'] == 0 else f"✗{r['bound_violations']}"
        print(
            f"{r['seq_len']:>8} {r['zeta_max']:>6.1f} {r['eviction_rate']:>6.1%} "
            f"{r['tv_distance']:>9.6f} {r['reconstruction_error']:>9.6f} "
            f"{bv:>7} {r['zeta_mean']:>7.2f} {r['zeta_std']:>7.2f} "
            f"{r['perfect_eviction_rate']:>4.0%} "
            f"{r['blocks_retained']:>4}/{r['num_blocks']:<4}"
        )
    print("=" * 105)


def print_memory_table(results: list[dict]) -> None:
    print()
    print("=" * 100)
    print("  MEMORY EFFICIENCY & THROUGHPUT")
    print("=" * 100)
    print(f"{'SeqLen':>8} {'ζ_max':>6} {'Evict%':>7} {'FullKV':>8} {'CompKV':>8} "
          f"{'Saved':>8} {'Reduct%':>8} {'Dense/GPU':>10} {'Ortho/GPU':>10} {'Gain':>6}")
    print("-" * 100)
    for r in results:
        print(
            f"{r['seq_len']:>8} {r['zeta_max']:>6.1f} {r['eviction_rate']:>6.1%} "
            f"{r['full_kv_bytes']/(1024**2):>7.1f}M {r['compact_kv_bytes']/(1024**2):>7.1f}M "
            f"{r['vram_saved_mb']:>7.1f}M {r['memory_reduction_pct']:>7.1f}% "
            f"{r['dense_concurrent_seqs']:>10} {r['ortho_concurrent_seqs']:>10} "
            f"{r['throughput_gain_x']:>5.1f}x"
        )
    print("=" * 100)


def print_cost_table(cost: dict) -> None:
    if "note" in cost:
        print(f"  Cost: {cost['note']}")
        return
    projs = cost["projections"]
    if not projs:
        return
    print()
    print("=" * 115)
    print(f"  FLEET COST PROJECTION — {cost['gpu']} @ ${cost['hourly']}/hr, {cost['watts']}W")
    print(f"  (Paper Table VII reference: 50% sparsity → $65.6M/yr savings)")
    print("=" * 115)
    print(f"{'SeqLen':>8} {'ζ_max':>6} {'Evict%':>7} {'GPUs_D':>7} {'GPUs_O':>7} "
          f"{'Saved':>6} {'$/mo Dense':>11} {'$/mo Ortho':>11} {'$/mo Save':>10} {'$/yr Save':>11}")
    print("-" * 115)
    for p in projs:
        print(
            f"{p['seq_len']:>8} {p['zeta_max']:>6.1f} {p['eviction_rate']:>6.1%} "
            f"{p['gpus_dense']:>7} {p['gpus_ortho']:>7} {p['gpus_saved']:>6} "
            f"${p['monthly_dense']:>10,} ${p['monthly_ortho']:>10,} "
            f"${p['monthly_savings']:>9,} ${p['annual_savings']:>10,}"
        )
    print("=" * 115)

    best = max(projs, key=lambda p: p['annual_savings'], default=None)
    if best and best['annual_savings'] > 0:
        print(f"\n  Best: seq={best['seq_len']}, ζ_max={best['zeta_max']}, "
              f"{best['eviction_rate']:.0%} evict → "
              f"{best['throughput_gain_x']:.1f}x throughput, "
              f"${best['annual_savings']:,}/yr savings")
    print()


# ═════════════════════════════════════════════════════════════════════
# Main Benchmark Runner
# ═════════════════════════════════════════════════════════════════════

def run_kaggle_benchmark() -> list[dict]:
    """Run the full OrthoCache GPU benchmark."""

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available.")
        return []

    device = torch.device("cuda")
    hw = detect_hardware()
    output_path = get_output_path()
    vram_bytes = hw.get("vram_total_bytes", 0)

    print(f"  Output         : {output_path}")
    print(f"  Seq lengths    : {SEQ_LENS}")
    print(f"  ζ_max values   : {ZETA_MAX_VALUES}")
    print(f"  Config         : {NUM_HEADS}H x {HEAD_DIM}D, Q={QUERY_LEN}, block={BLOCK_SIZE}")
    print(f"  Pipeline       : orthocache_forward (two-gate: logit + ζ)")
    print()

    all_results: list[dict] = []
    ts = datetime.now(timezone.utc).isoformat()

    for seq_len in SEQ_LENS:
        num_blocks = seq_len // BLOCK_SIZE
        print(f"{'='*72}")
        print(f"  Sequence Length: {seq_len:,} ({num_blocks} blocks)")
        print(f"{'='*72}")

        try:
            keys, values, q = generate_structured_kv(
                seq_len, NUM_HEADS, HEAD_DIM, BLOCK_SIZE, device, torch.bfloat16, SEED
            )
            torch.cuda.synchronize()
            kv_mb = kv_cache_bytes(seq_len, NUM_HEADS, HEAD_DIM) / (1024**2)
            print(f"  KV cache: {kv_mb:.1f} MB | Data: structured (anchor/semantic/mixed/noise)")
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] Skipping")
            gc.collect(); torch.cuda.empty_cache()
            continue

        for zeta_max in ZETA_MAX_VALUES:
            print(f"  ζ_max={zeta_max:.1f}: ", end="", flush=True)

            try:
                record = evaluate_config(q, keys, values, zeta_max, vram_bytes)
                record["timestamp"] = ts
                record["gpu_name"] = hw.get("gpu_name", "unknown")
                all_results.append(record)

                print(
                    f"evict={record['eviction_rate']:.0%}  "
                    f"TV={record['tv_distance']:.6f}  "
                    f"err={record['reconstruction_error']:.6f}  "
                    f"bound={'✓' if record['bound_violations']==0 else '✗'}  "
                    f"saved={record['vram_saved_mb']:.0f}MB  "
                    f"seqs={record['dense_concurrent_seqs']}->{record['ortho_concurrent_seqs']} "
                    f"({record['throughput_gain_x']:.1f}x)"
                )

                _write_results(output_path, all_results, hw)

            except torch.cuda.OutOfMemoryError:
                print("[OOM]")
                torch.cuda.empty_cache(); gc.collect()
            except Exception as exc:
                print(f"[ERROR] {exc}")
                import traceback; traceback.print_exc()

        del keys, values, q
        gc.collect(); torch.cuda.empty_cache()
        print()

    if all_results:
        print_quality_table(all_results)
        print_memory_table(all_results)
        cost = compute_cost_projection(hw.get("gpu_name", ""), all_results)
        print_cost_table(cost)
        _write_results(output_path, all_results, hw, cost)
        print(f"Results saved to: {output_path}")
    else:
        print("[WARN] No results.")

    return all_results


def _write_results(path: Path, results: list[dict], hw: dict,
                    cost: dict | None = None) -> None:
    payload = {
        "benchmark": "orthocache_gpu_datacenter",
        "version": orthocache_gpu.__version__,
        "hardware": hw,
        "config": {
            "block_size": BLOCK_SIZE, "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM, "query_len": QUERY_LEN,
            "seq_lens": SEQ_LENS, "zeta_max_values": ZETA_MAX_VALUES,
            "data_type": "structured_synthetic",
        },
        "results": results,
    }
    if cost:
        payload["cost_projection"] = cost
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(str(tmp), str(path))


if __name__ == "__main__":
    run_kaggle_benchmark()
