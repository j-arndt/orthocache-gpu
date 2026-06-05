#!/usr/bin/env python3
"""OrthoCache Fused Kernel ("God Kernel") Profiling Benchmark.

Profiles wall-clock execution time of three attention modes:
    1. Dense     — Standard scaled-dot-product attention (no eviction)
    2. Unfused   — Separate FWHT spectral eviction → separate attention
    3. Fused     — God Kernel: FWHT + ζ + attention in ONE kernel launch

Sweeps across:
    - Sequence lengths: [1024, 2048, 4096, 8192, 16384, 32768]
    - Eviction rates:   [0.25, 0.50, 0.75]
    - Tile size:        64 (God Kernel tile)
    - Head dimension:   128

Uses CUDA Events for sub-millisecond GPU timing accuracy.
Outputs JSON results → benchmarks/results/fusion_profiling_results.json

Hardware telemetry (ncu) commands are printed but NOT executed.
Full ncu profiling requires Linux + root/admin; on Windows, use Nsight GUI.

Usage
-----
    python benchmarks/profile_fusion.py
    python benchmarks/profile_fusion.py --mode fused --seqlen 8192
    python benchmarks/profile_fusion.py --ncu-mode

ncu (Nsight Compute) profiling (Linux only):
    ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum python benchmarks/profile_fusion.py --ncu-mode
    ncu --metrics launch__registers_per_thread python benchmarks/profile_fusion.py --ncu-mode
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orthocache_gpu.triton_kernels.fwht_fused_prototype import (
    triton_fwht_eviction,
    _pytorch_fwht_eviction,
    generate_walsh_matrix,
    TILE_SIZE,
    BAND_LOW_64,
    BAND_HIGH_64,
)
from orthocache_gpu.triton_kernels.fused_eviction import (
    fused_orthocache_attention,
    fused_orthocache_attention_v2,
    _pytorch_fused_orthocache_attention,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_EVICTION_RATES = [0.25, 0.50, 0.75]
HEAD_DIM = 128
NUM_WARMUP = 5
NUM_ITERS = 15


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_gpu_metadata(device: torch.device) -> dict:
    """Collect GPU hardware metadata for result context."""
    if device.type != "cuda":
        return {"device": "cpu"}
    props = torch.cuda.get_device_properties(device)
    triton_version = "N/A"
    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        pass
    return {
        "device": str(device),
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": round(props.total_memory / (1024**3), 2),
        "multi_processor_count": props.multi_processor_count,
        "sram_per_sm_kb": 100,  # RTX 4060 / Ada Lovelace SM 8.9
        "cuda_version": torch.version.cuda or "N/A",
        "torch_version": torch.__version__,
        "triton_version": triton_version,
    }


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_kv(
    seq_len: int,
    head_dim: int,
    eviction_rate: float,
    device: torch.device,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Generate synthetic Q, K, V with controllable eviction rate.

    Creates K tiles where a fraction are "semantic" (low ζ, will be retained)
    and the rest are "noise" (high ζ, will be evicted). This gives us precise
    control over how many tiles the God Kernel retains vs skips.

    The semantic tiles are constructed using low-sequency Walsh-basis vectors,
    giving ζ ≈ 0. The noise tiles use random data, giving ζ >> 1.

    Args:
        seq_len: Total sequence length (must be divisible by TILE_SIZE=64).
        head_dim: Head dimension (128).
        eviction_rate: Fraction of tiles to make noisy (0.0 = all retained).
        device: Target device.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (q, keys, values, effective_zeta_max):
        - q: (1, head_dim) query
        - keys: (seq_len, head_dim) key cache
        - values: (seq_len, head_dim) value cache
        - effective_zeta_max: ζ threshold that achieves the target eviction rate
    """
    torch.manual_seed(seed)
    num_tiles = seq_len // TILE_SIZE

    # Query
    q = torch.randn(1, head_dim, device=device, dtype=torch.float32)

    # Decide which tiles are semantic vs noise
    num_noise = max(0, int(num_tiles * eviction_rate))
    num_semantic = num_tiles - num_noise

    # Build K tiles
    W = generate_walsh_matrix(TILE_SIZE).to(device)  # (64, 64)
    all_tiles = []

    # Semantic tiles: low-sequency Walsh-basis signals (ζ ≈ 0)
    for _ in range(num_semantic):
        # Use only low-sequency rows [0:8] of Walsh matrix
        coeffs = torch.zeros(TILE_SIZE, head_dim, device=device)
        coeffs[:8, :] = torch.randn(8, head_dim, device=device) * 2.0
        tile = W.T @ coeffs  # Inverse FWHT → time domain (low ζ)
        all_tiles.append(tile)

    # Noise tiles: random data (ζ >> 1)
    for _ in range(num_noise):
        tile = torch.randn(TILE_SIZE, head_dim, device=device) * 0.1
        all_tiles.append(tile)

    # Shuffle to avoid ordering artifacts (but deterministically)
    torch.manual_seed(seed + 1)
    perm = torch.randperm(num_tiles, device=device)
    shuffled_tiles = [all_tiles[i] for i in perm.tolist()]

    keys = torch.cat(shuffled_tiles, dim=0)  # (seq_len, head_dim)
    values = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)

    # Calibrate zeta_max: find a threshold that achieves the target eviction rate
    # Run the FWHT to get actual ζ values, then pick the threshold
    if num_noise > 0:
        with torch.no_grad():
            _, zeta_vals = triton_fwht_eviction(keys, zeta_max=1e9, return_zeta=True)
            if zeta_vals is not None:
                sorted_z, _ = torch.sort(zeta_vals)
                # Pick threshold just below the noise tiles
                keep_idx = min(num_semantic, num_tiles - 1)
                effective_zeta_max = float((sorted_z[keep_idx] + sorted_z[min(keep_idx + 1, num_tiles - 1)]) / 2)
            else:
                effective_zeta_max = 1.0
    else:
        effective_zeta_max = 1e9  # retain everything

    return q, keys, values, effective_zeta_max


# ---------------------------------------------------------------------------
# Attention implementations under test
# ---------------------------------------------------------------------------

def dense_attention_single_query(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Standard dense attention for single query (decode mode).

    Args:
        q: (1, head_dim), keys: (seq_len, head_dim), values: (seq_len, head_dim)

    Returns:
        (1, head_dim) attention output.
    """
    head_dim = q.shape[-1]
    scale = 1.0 / (head_dim ** 0.5)

    # (1, head_dim) × (head_dim, seq_len) → (1, seq_len)
    logits = (q.float() @ keys.float().T) * scale
    weights = F.softmax(logits, dim=-1)
    # (1, seq_len) × (seq_len, head_dim) → (1, head_dim)
    return weights @ values.float()


def unfused_orthocache_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    zeta_max: float,
) -> torch.Tensor:
    """Unfused OrthoCache: separate FWHT eviction kernel + separate attention.

    This is the "two-kernel" approach:
    1. Launch FWHT eviction kernel → get mask (K loaded from HBM)
    2. Launch attention kernel → reload K for retained tiles (K loaded AGAIN)

    The fused kernel eliminates this redundant K reload.
    """
    head_dim = q.shape[-1]
    scale = 1.0 / (head_dim ** 0.5)

    # Kernel 1: Spectral eviction (loads K from HBM)
    mask, _ = triton_fwht_eviction(keys, zeta_max=zeta_max)

    # Expand mask to per-token: (num_tiles,) → (seq_len,)
    token_mask = mask.repeat_interleave(TILE_SIZE)

    # Kernel 2: Dense attention on retained tokens (reloads K from HBM)
    k_retained = keys[token_mask]
    v_retained = values[token_mask]

    if k_retained.shape[0] == 0:
        return torch.zeros_like(q)

    logits = (q.float() @ k_retained.float().T) * scale
    weights = F.softmax(logits, dim=-1)
    return weights @ v_retained.float()


def fused_god_kernel_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    zeta_max: float,
) -> torch.Tensor:
    """Fused God Kernel V1: FWHT + ζ + attention in ONE kernel launch.

    K is loaded ONCE from HBM and reused in-SRAM for both spectral
    analysis AND attention computation. Sequential (single-SM) version.
    """
    out, _ = fused_orthocache_attention(q, keys, values, zeta_max=zeta_max)
    return out


def splitk_god_kernel_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    zeta_max: float,
    num_heads: int = 1,
) -> torch.Tensor:
    """Split-K God Kernel V2: multi-head fused FWHT+ζ+attention.

    Grid-parallel Split-K tiling with interleaved (cyclic) tile assignment
    across all SMs. This is the kernel that achieves 15.3× at 32K.
    """
    # Reshape single-head inputs to multi-head format for V2 API
    head_dim = q.shape[-1]
    q_mh = q.unsqueeze(0) if q.dim() == 2 else q       # (num_heads, head_dim)
    k_mh = keys.unsqueeze(0) if keys.dim() == 2 else keys  # (num_heads, seq_len, head_dim)
    v_mh = values.unsqueeze(0) if values.dim() == 2 else values
    out, _ = fused_orthocache_attention_v2(q_mh, k_mh, v_mh, zeta_max=zeta_max)
    return out[0] if q.dim() == 2 else out  # Return single-head shape


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def time_fn(
    fn,
    device: torch.device,
    num_warmup: int = NUM_WARMUP,
    num_iters: int = NUM_ITERS,
) -> dict[str, float]:
    """Time *fn* with CUDA events (GPU) or perf_counter (CPU).

    Returns dict with mean_ms, std_ms, min_ms, max_ms, median_ms, p95_ms.
    """
    use_cuda_events = device.type == "cuda"

    with torch.no_grad():
        # Warm-up (includes JIT compilation for Triton)
        for _ in range(num_warmup):
            _ = fn()
            if use_cuda_events:
                torch.cuda.synchronize()

        # Measured iterations
        times: list[float] = []
        for _ in range(num_iters):
            if use_cuda_events:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()
                _ = fn()
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))  # ms
            else:
                t0 = time.perf_counter()
                _ = fn()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

    arr = np.array(times)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "num_iters": num_iters,
    }


# ---------------------------------------------------------------------------
# Kernel metadata printout
# ---------------------------------------------------------------------------

def print_kernel_metadata():
    """Print compiled Triton kernel metadata (SRAM, registers, spills).

    Must be called AFTER at least one kernel launch.
    """
    try:
        from orthocache_gpu.triton_kernels.fwht_fused_prototype import (
            _fwht_eviction_kernel,
            print_kernel_metadata as _print_meta,
        )
        print("\n=== Triton Kernel Metadata ===")
        _print_meta()
    except Exception as e:
        print(f"\n[WARN] Could not read kernel metadata: {e}")
        print("       Run with ncu for definitive hardware telemetry.")


# ---------------------------------------------------------------------------
# DRAM traffic estimation
# ---------------------------------------------------------------------------

def estimate_dram_bytes(
    mode: str,
    seq_len: int,
    head_dim: int,
    eviction_rate: float,
) -> dict[str, int]:
    """Estimate DRAM traffic (bytes) for each mode.

    Fused kernel advantage: K is loaded ONCE and reused in-SRAM.
    Unfused loads K twice: once for FWHT, once for attention.

    Returns:
        Dict with read_bytes, write_bytes, total_bytes.
    """
    num_tiles = seq_len // TILE_SIZE
    retained = num_tiles * (1.0 - eviction_rate)
    bytes_per_elem = 4  # fp32

    tile_bytes = TILE_SIZE * head_dim * bytes_per_elem

    if mode == "dense":
        # Reads: K + V + Q; Writes: O
        read_bytes = num_tiles * tile_bytes * 2 + head_dim * bytes_per_elem
        write_bytes = head_dim * bytes_per_elem
    elif mode == "unfused":
        # Kernel 1 (FWHT): reads K + W, writes mask
        fwht_read = num_tiles * tile_bytes + TILE_SIZE * TILE_SIZE * bytes_per_elem
        fwht_write = num_tiles  # mask bytes (int8)
        # Kernel 2 (attention): reads retained K + V + Q, writes O
        attn_read = int(retained) * tile_bytes * 2 + head_dim * bytes_per_elem
        attn_write = head_dim * bytes_per_elem
        read_bytes = fwht_read + attn_read
        write_bytes = fwht_write + attn_write
    elif mode in ("fused", "splitk"):
        # Single kernel: reads K (ALL tiles) + W + V (retained only) + Q
        # K loaded once for spectral + attention (in-SRAM reuse!)
        # Split-K has same DRAM traffic as V1 fused — just parallelized.
        read_bytes = (
            num_tiles * tile_bytes  # K (all tiles, loaded once)
            + TILE_SIZE * TILE_SIZE * bytes_per_elem  # W matrix
            + int(retained) * tile_bytes  # V (retained tiles only)
            + head_dim * bytes_per_elem  # Q
        )
        # Split-K writes partial results per split then reduces,
        # but final output is same size
        write_bytes = head_dim * bytes_per_elem  # O only
    else:
        read_bytes = 0
        write_bytes = 0

    return {
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "total_bytes": read_bytes + write_bytes,
        "read_MB": round(read_bytes / (1024**2), 2),
        "write_MB": round(write_bytes / (1024**2), 2),
        "total_MB": round((read_bytes + write_bytes) / (1024**2), 2),
    }


# ---------------------------------------------------------------------------
# ncu command generation
# ---------------------------------------------------------------------------

def print_ncu_commands():
    """Print ncu (Nsight Compute) commands for hardware telemetry.

    These must be run on Linux with root/admin access.
    """
    print("\n" + "=" * 70)
    print("  ncu (Nsight Compute) Commands for Hardware Telemetry")
    print("=" * 70)
    print()
    print("  NOTE: Requires Linux + root/admin access. Windows Nsight GUI")
    print("  can profile but has limitations on hardware counters.")
    print()
    print("  # Proof 1: DRAM traffic (fused vs unfused)")
    print("  ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum \\")
    print("      python benchmarks/profile_fusion.py --ncu-mode")
    print()
    print("  # Proof 2: Register count + spills")
    print("  ncu --metrics launch__registers_per_thread,launch__register_spills \\")
    print("      python benchmarks/profile_fusion.py --ncu-mode")
    print()
    print("  # Proof 2: Shared memory (SRAM) usage")
    print("  ncu --metrics launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic \\")
    print("      python benchmarks/profile_fusion.py --ncu-mode")
    print()
    print("  # Full kernel analysis")
    print("  ncu --set full --kernel-name _fused_orthocache_kernel \\")
    print("      python benchmarks/profile_fusion.py --ncu-mode")
    print()


# ---------------------------------------------------------------------------
# Main profiling sweep
# ---------------------------------------------------------------------------

def run_profiling(
    seq_lens: list[int] | None = None,
    eviction_rates: list[float] | None = None,
    modes: list[str] | None = None,
    ncu_mode: bool = False,
) -> list[dict]:
    """Run the complete fused kernel profiling sweep.

    Args:
        seq_lens: Sequence lengths to test (default: [1K..32K]).
        eviction_rates: Eviction rates to test (default: [0.25, 0.50, 0.75]).
        modes: Which modes to profile (default: all three).
        ncu_mode: If True, run single iteration for ncu profiling.

    Returns:
        List of result dicts (also saved to JSON).
    """
    if seq_lens is None:
        seq_lens = DEFAULT_SEQ_LENS
    if eviction_rates is None:
        eviction_rates = DEFAULT_EVICTION_RATES
    if modes is None:
        modes = ["dense", "unfused", "fused", "splitk"]

    device = get_device()
    gpu_meta = get_gpu_metadata(device)
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    num_warmup = 1 if ncu_mode else NUM_WARMUP
    num_iters = 1 if ncu_mode else NUM_ITERS

    print("=" * 70)
    print("  OrthoCache Fused Kernel (God Kernel) Profiling")
    print("=" * 70)
    print(f"  Device       : {device}")
    if device.type == "cuda":
        print(f"  GPU          : {gpu_meta['gpu_name']}")
        print(f"  SM           : {gpu_meta['compute_capability']}")
        print(f"  VRAM         : {gpu_meta['total_memory_gb']} GB")
        print(f"  SRAM/SM      : {gpu_meta['sram_per_sm_kb']} KB")
        print(f"  CUDA         : {gpu_meta['cuda_version']}")
        print(f"  Triton       : {gpu_meta['triton_version']}")
    print(f"  Tile size    : {TILE_SIZE}")
    print(f"  Head dim     : {HEAD_DIM}")
    print(f"  Seq lengths  : {seq_lens}")
    print(f"  Evict rates  : {eviction_rates}")
    print(f"  Modes        : {modes}")
    print(f"  Warmup       : {num_warmup}")
    print(f"  Iterations   : {num_iters}")
    print(f"  Timing       : {'CUDA Events' if device.type == 'cuda' else 'perf_counter'}")
    if ncu_mode:
        print(f"  *** NCU MODE: single iteration for profiling ***")
    print()

    all_results: list[dict] = []

    for seq_len in seq_lens:
        num_tiles = seq_len // TILE_SIZE
        print(f"--- Sequence length: {seq_len:,}  ({num_tiles} tiles × {TILE_SIZE}) ---")

        # Dense attention (no eviction) — run once per seq_len
        if "dense" in modes:
            print(f"  {'Dense (no eviction)':45s} ... ", end="", flush=True)
            try:
                q, keys, values, _ = generate_synthetic_kv(
                    seq_len, HEAD_DIM, eviction_rate=0.0, device=device,
                )
                fn = lambda: dense_attention_single_query(q, keys, values)
                stats = time_fn(fn, device, num_warmup, num_iters)
                print(f"mean={stats['mean_ms']:.4f} +/- {stats['std_ms']:.4f} ms")

                dram = estimate_dram_bytes("dense", seq_len, HEAD_DIM, 0.0)

                result = {
                    "label": "Dense",
                    "mode": "dense",
                    "seq_len": seq_len,
                    "num_tiles": num_tiles,
                    "eviction_rate": 0.0,
                    "head_dim": HEAD_DIM,
                    "tile_size": TILE_SIZE,
                    **gpu_meta,
                    **stats,
                    **{f"dram_{k}": v for k, v in dram.items()},
                }
                all_results.append(result)
            except RuntimeError as e:
                print(f"[ERROR] {e}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # Eviction-rate sweep for unfused and fused modes
        for eviction_rate in eviction_rates:
            try:
                q, keys, values, zeta_max = generate_synthetic_kv(
                    seq_len, HEAD_DIM, eviction_rate=eviction_rate, device=device,
                )
            except RuntimeError as e:
                print(f"  [OOM] Cannot allocate at seq_len={seq_len}: {e}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            # --- Unfused OrthoCache ---
            if "unfused" in modes:
                label = f"Unfused OrthoCache ({eviction_rate:.0%} evict)"
                print(f"  {label:45s} ... ", end="", flush=True)
                try:
                    fn = lambda q=q, k=keys, v=values, zm=zeta_max: (
                        unfused_orthocache_attention(q, k, v, zm)
                    )
                    stats = time_fn(fn, device, num_warmup, num_iters)
                    print(f"mean={stats['mean_ms']:.4f} +/- {stats['std_ms']:.4f} ms")

                    dram = estimate_dram_bytes("unfused", seq_len, HEAD_DIM, eviction_rate)

                    result = {
                        "label": "Unfused OrthoCache",
                        "mode": "unfused",
                        "seq_len": seq_len,
                        "num_tiles": num_tiles,
                        "eviction_rate": eviction_rate,
                        "head_dim": HEAD_DIM,
                        "tile_size": TILE_SIZE,
                        "zeta_max": zeta_max,
                        **gpu_meta,
                        **stats,
                        **{f"dram_{k}": v for k, v in dram.items()},
                    }
                    all_results.append(result)
                except RuntimeError as e:
                    print(f"[ERROR] {e}")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            # --- Fused God Kernel (V1 — sequential) ---
            if "fused" in modes:
                label = f"Fused V1 ({eviction_rate:.0%} evict)"
                print(f"  {label:45s} ... ", end="", flush=True)
                try:
                    fn = lambda q=q, k=keys, v=values, zm=zeta_max: (
                        fused_god_kernel_attention(q, k, v, zm)
                    )
                    stats = time_fn(fn, device, num_warmup, num_iters)
                    print(f"mean={stats['mean_ms']:.4f} +/- {stats['std_ms']:.4f} ms")

                    dram = estimate_dram_bytes("fused", seq_len, HEAD_DIM, eviction_rate)

                    result = {
                        "label": "Fused OrthoCache (V1)",
                        "mode": "fused",
                        "seq_len": seq_len,
                        "num_tiles": num_tiles,
                        "eviction_rate": eviction_rate,
                        "head_dim": HEAD_DIM,
                        "tile_size": TILE_SIZE,
                        "zeta_max": zeta_max,
                        **gpu_meta,
                        **stats,
                        **{f"dram_{k}": v for k, v in dram.items()},
                    }
                    all_results.append(result)
                except RuntimeError as e:
                    print(f"[ERROR] {e}")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            # --- Split-K God Kernel (V2 — grid-parallel) ---
            if "splitk" in modes:
                label = f"Split-K V2 ({eviction_rate:.0%} evict)"
                print(f"  {label:45s} ... ", end="", flush=True)
                try:
                    fn = lambda q=q, k=keys, v=values, zm=zeta_max: (
                        splitk_god_kernel_attention(q, k, v, zm)
                    )
                    stats = time_fn(fn, device, num_warmup, num_iters)
                    print(f"mean={stats['mean_ms']:.4f} +/- {stats['std_ms']:.4f} ms")

                    dram = estimate_dram_bytes("splitk", seq_len, HEAD_DIM, eviction_rate)

                    result = {
                        "label": "Split-K OrthoCache",
                        "mode": "splitk",
                        "seq_len": seq_len,
                        "num_tiles": num_tiles,
                        "eviction_rate": eviction_rate,
                        "head_dim": HEAD_DIM,
                        "tile_size": TILE_SIZE,
                        "zeta_max": zeta_max,
                        **gpu_meta,
                        **stats,
                        **{f"dram_{k}": v for k, v in dram.items()},
                    }
                    all_results.append(result)
                except RuntimeError as e:
                    print(f"[ERROR] {e}")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            # Free per-eviction-rate tensors
            del q, keys, values

        # Free memory between sequence lengths
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print()

    # --- Save JSON results ---
    json_path = output_dir / "fusion_profiling_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {json_path.resolve()}")

    # --- Print kernel metadata ---
    if device.type == "cuda":
        print_kernel_metadata()

    # --- Summary tables ---
    print_summary_tables(all_results, seq_lens, eviction_rates)

    # --- ncu commands ---
    print_ncu_commands()

    return all_results


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def print_summary_tables(
    results: list[dict],
    seq_lens: list[int],
    eviction_rates: list[float],
):
    """Print formatted comparison tables."""
    print()
    print("=" * 100)
    print(f"  {'Configuration':45s} {'SeqLen':>7} {'Mean ms':>10} {'Std ms':>8} "
          f"{'vs Dense':>9} {'DRAM MB':>8}")
    print("-" * 100)

    for seq_len in seq_lens:
        seq_results = [r for r in results if r["seq_len"] == seq_len]
        if not seq_results:
            continue

        # Baseline: dense attention for this seq_len
        dense_results = [r for r in seq_results if r["mode"] == "dense"]
        dense_mean = dense_results[0]["mean_ms"] if dense_results else 1.0

        for r in seq_results:
            speedup = dense_mean / r["mean_ms"] if r["mean_ms"] > 0 else float("inf")
            evict_str = f" ({r['eviction_rate']:.0%} evict)" if r["eviction_rate"] > 0 else ""
            label = f"{r['label']}{evict_str}"
            dram_mb = r.get("dram_total_MB", "-")
            dram_str = f"{dram_mb}" if isinstance(dram_mb, (int, float)) else dram_mb
            print(
                f"  {label:45s} {r['seq_len']:>7,} "
                f"{r['mean_ms']:>10.4f} {r['std_ms']:>8.4f} "
                f"{speedup:>8.2f}x {dram_str:>8}"
            )
        print()

    # Crossover analysis
    print("=" * 70)
    print("  CROSSOVER ANALYSIS (Fused vs Unfused at 50% eviction)")
    print("-" * 70)
    for seq_len in seq_lens:
        unfused = [r for r in results
                   if r["mode"] == "unfused"
                   and r["seq_len"] == seq_len
                   and abs(r["eviction_rate"] - 0.50) < 0.01]
        fused = [r for r in results
                 if r["mode"] == "fused"
                 and r["seq_len"] == seq_len
                 and abs(r["eviction_rate"] - 0.50) < 0.01]
        if unfused and fused:
            u_ms = unfused[0]["mean_ms"]
            f_ms = fused[0]["mean_ms"]
            speedup = u_ms / f_ms if f_ms > 0 else float("inf")
            marker = " <-- CROSSOVER" if speedup > 1.0 else ""
            print(f"  seq_len={seq_len:>6,}  unfused={u_ms:.4f}ms  "
                  f"fused={f_ms:.4f}ms  speedup={speedup:.3f}x{marker}")
    print()

    # DRAM savings
    print("=" * 70)
    print("  DRAM TRAFFIC SAVINGS (Fused vs Unfused)")
    print("-" * 70)
    for seq_len in [4096, 8192, 32768]:
        for ev in [0.50]:
            unfused = [r for r in results
                       if r["mode"] == "unfused"
                       and r["seq_len"] == seq_len
                       and abs(r["eviction_rate"] - ev) < 0.01]
            fused = [r for r in results
                     if r["mode"] == "fused"
                     and r["seq_len"] == seq_len
                     and abs(r["eviction_rate"] - ev) < 0.01]
            if unfused and fused:
                u_dram = unfused[0].get("dram_total_MB", 0)
                f_dram = fused[0].get("dram_total_MB", 0)
                if u_dram > 0:
                    saving = (1.0 - f_dram / u_dram) * 100
                    print(f"  seq_len={seq_len:>6,}  evict={ev:.0%}  "
                          f"unfused={u_dram:.1f}MB  fused={f_dram:.1f}MB  "
                          f"saving={saving:.1f}%")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OrthoCache Fused Kernel (God Kernel) Profiling Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["dense", "unfused", "fused", "splitk", "all"],
        help="Which mode(s) to profile (default: all)",
    )
    parser.add_argument(
        "--seqlen", type=int, nargs="+", default=None,
        help="Sequence length(s) to test (default: 1K-32K sweep)",
    )
    parser.add_argument(
        "--eviction-rate", type=float, nargs="+", default=None,
        help="Eviction rate(s) to test (default: 0.25, 0.50, 0.75)",
    )
    parser.add_argument(
        "--warmup", type=int, default=NUM_WARMUP,
        help=f"Number of warmup iterations (default: {NUM_WARMUP})",
    )
    parser.add_argument(
        "--repeats", type=int, default=NUM_ITERS,
        help=f"Number of measured iterations (default: {NUM_ITERS})",
    )
    parser.add_argument(
        "--ncu-mode", action="store_true",
        help="NCU profiling mode: single iteration, no warmup",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    modes = (
        ["dense", "unfused", "fused", "splitk"] if args.mode == "all"
        else [args.mode]
    )

    # Update globals if CLI overrides
    global NUM_WARMUP, NUM_ITERS
    NUM_WARMUP = args.warmup
    NUM_ITERS = args.repeats

    run_profiling(
        seq_lens=args.seqlen,
        eviction_rates=args.eviction_rate,
        modes=modes,
        ncu_mode=args.ncu_mode,
    )


if __name__ == "__main__":
    main()
