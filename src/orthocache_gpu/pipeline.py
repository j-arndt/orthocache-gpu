"""OrthoCache end-to-end pipeline (GPU Edition).

Chains the full OrthoCache flow:
    FWHT → spectral bands → ζ computation → two-gate mask → compaction → attention

This module provides the high-level API that users call. It handles:
- **Adaptive crossover bypass**: automatically falls back to dense attention
  for short sequences (< CROSSOVER_SEQ_LEN) where spectral analysis overhead
  exceeds eviction savings. This prevents performance degradation on short
  prompts (0.51× at 1K, 0.91× at 2K tokens).
- Automatic GPU detection and fallback
- Block size alignment and padding
- ζ_max auto-calibration hints
- Timing and telemetry metadata
"""

import time
from functools import partial

import torch
import torch.nn.functional as F

from orthocache_gpu.spectral_energy import (
    compute_spectral_bands,
    compute_spectral_decay_ratio,
    compute_query_aware_mask,
    compute_multiband_mask,
)
from orthocache_gpu.compaction import stream_compact, compact_and_attend
from orthocache_gpu.adaptive_attention import orthocache_attention

# ============================================================
# Dynamic Hardware Dispatcher
# ============================================================
# The crossover point where OrthoCache's spectral analysis overhead
# breaks even with eviction savings depends on GPU architecture:
#
#   - Memory bandwidth: higher BW → dense is faster → higher crossover
#   - SM count: more SMs → better Split-K parallelism → lower crossover
#   - SRAM/SM: more SRAM → larger tiles → lower per-tile overhead
#   - Compute density: higher FLOPS → FWHT is cheaper relative to attention
#
# Instead of hardcoding a single threshold, we detect the GPU at init
# and select from empirical profiles. Unknown GPUs use a bandwidth-ratio
# model anchored to the RTX 4060 Laptop reference measurement.
# ============================================================

import dataclasses
import enum


# ============================================================
# Enterprise Telemetry — Zero-Overhead Bitmask Flags
# ============================================================
# In production inference loops, f-string interpolation for logging
# can consume 2-5 µs per call — enough to neutralize kernel speedups
# at high QPS. We use IntFlag bitmasks instead: a single int write
# replaces all string formatting. Human-readable strings are available
# via format_bypass_reason() for debugging, but never in the hot path.
# ============================================================

class BypassFlag(enum.IntFlag):
    """Bitmask flags for bypass/dispatch decisions. Zero-cost hot-path telemetry."""
    NONE              = 0
    CROSSOVER_BYPASS  = 1 << 0   # seq_len < crossover → dense fallback
    DENSE_EXPLICIT    = 1 << 1   # mode='dense' requested explicitly
    TRITON_FUSED      = 1 << 2   # mode='triton_fused'
    SPECTRAL_EVICTION = 1 << 3   # full spectral analysis path
    ZERO_ACTIVE       = 1 << 4   # all blocks evicted → zero output
    TAU_AUTO          = 1 << 5   # tau was auto-computed
    PERFECT_EVICTION  = 1 << 6   # at least one block was perfectly evicted
    HW_AUTO_DETECTED  = 1 << 7   # GPU was auto-detected (not from known list)


class TelemetryLevel(enum.IntEnum):
    """Controls metadata verbosity in the inference hot path.

    SILENT:  No metadata dict at all — maximum throughput.
    FLAGS:   Bitmask flags + numeric scalars only. No strings. (default)
    VERBOSE: Full metadata with human-readable strings. For debugging.
    """
    SILENT  = 0
    FLAGS   = 1
    VERBOSE = 2


def format_bypass_reason(
    flags: BypassFlag,
    seq_len_k: int = 0,
    threshold: int = 0,
    profile: 'GPUProfile | None' = None,
) -> str:
    """Lazy string formatter for bypass telemetry. Call ONLY for debugging.

    This function is never invoked in the hot path. It converts bitmask
    flags into the human-readable strings that were previously computed
    via f-string interpolation on every forward pass.
    """
    parts = []
    if flags & BypassFlag.CROSSOVER_BYPASS:
        p = f'seq_len_k={seq_len_k} < crossover={threshold}'
        if profile:
            p += f' on {profile.name} ({profile.sm_count} SMs, {profile.mem_bandwidth_gbps:.0f} GB/s)'
        parts.append(p)
    if flags & BypassFlag.DENSE_EXPLICIT:
        parts.append('mode=dense (explicit)')
    if flags & BypassFlag.TRITON_FUSED:
        parts.append('mode=triton_fused (God Kernel)')
    if flags & BypassFlag.SPECTRAL_EVICTION:
        parts.append('full spectral eviction path')
    if flags & BypassFlag.ZERO_ACTIVE:
        parts.append('all blocks evicted')
    if flags & BypassFlag.TAU_AUTO:
        parts.append('tau auto-computed (mean - 1σ)')
    if flags & BypassFlag.HW_AUTO_DETECTED:
        parts.append('GPU auto-detected via bandwidth-ratio model')
    return '; '.join(parts) if parts else 'no flags set'



@dataclasses.dataclass(frozen=True)
class GPUProfile:
    """Hardware profile for crossover threshold estimation."""
    name: str
    sm_count: int
    mem_bandwidth_gbps: float   # GB/s
    sram_per_sm_kb: float       # KB
    crossover_seq_len: int      # tokens — empirical or estimated
    compute_class: str          # 'consumer' | 'datacenter' | 'hpc'


# Empirical and estimated crossover thresholds per GPU family.
# RTX 4060 Laptop is the reference (measured): crossover at ~4K tokens.
# Others are estimated via bandwidth-ratio scaling:
#   crossover ∝ (target_bw / reference_bw) × reference_crossover
# Higher bandwidth GPUs can stream dense KV faster, so OrthoCache's
# spectral overhead needs more tokens to amortize → higher crossover.
# BUT more SMs and larger SRAM offset this → apply SM correction.
_GPU_PROFILES: dict[str, GPUProfile] = {
    # ── Consumer / Workstation ──────────────────────────────────────
    'RTX 4060 Laptop': GPUProfile(
        name='RTX 4060 Laptop', sm_count=24,
        mem_bandwidth_gbps=256, sram_per_sm_kb=100,
        crossover_seq_len=4096,  # MEASURED: 0.91× at 2K, 1.09× at 4K
        compute_class='consumer',
    ),
    'RTX 4060': GPUProfile(
        name='RTX 4060', sm_count=24,
        mem_bandwidth_gbps=272, sram_per_sm_kb=100,
        crossover_seq_len=4096,  # Similar to laptop variant
        compute_class='consumer',
    ),
    'RTX 4070': GPUProfile(
        name='RTX 4070', sm_count=46,
        mem_bandwidth_gbps=504, sram_per_sm_kb=100,
        crossover_seq_len=3072,  # More SMs offset higher BW
        compute_class='consumer',
    ),
    'RTX 4080': GPUProfile(
        name='RTX 4080', sm_count=76,
        mem_bandwidth_gbps=717, sram_per_sm_kb=100,
        crossover_seq_len=2048,  # 76 SMs → excellent Split-K coverage
        compute_class='consumer',
    ),
    'RTX 4090': GPUProfile(
        name='RTX 4090', sm_count=128,
        mem_bandwidth_gbps=1008, sram_per_sm_kb=100,
        crossover_seq_len=2048,  # 128 SMs dominate; BW offset by parallelism
        compute_class='consumer',
    ),
    # ── Datacenter ──────────────────────────────────────────────────
    'A100': GPUProfile(
        name='A100', sm_count=108,
        mem_bandwidth_gbps=2039, sram_per_sm_kb=164,
        crossover_seq_len=2048,  # Massive BW but 108 SMs + 164KB SRAM
        compute_class='datacenter',
    ),
    'H100': GPUProfile(
        name='H100', sm_count=132,
        mem_bandwidth_gbps=3350, sram_per_sm_kb=228,
        crossover_seq_len=1024,  # 132 SMs + 228KB SRAM → very low overhead
        compute_class='datacenter',
    ),
    'H200': GPUProfile(
        name='H200', sm_count=132,
        mem_bandwidth_gbps=4800, sram_per_sm_kb=228,
        crossover_seq_len=1024,  # Same SMs as H100, more HBM bandwidth
        compute_class='datacenter',
    ),
    # ── Next-gen (Blackwell) ────────────────────────────────────────
    'B200': GPUProfile(
        name='B200', sm_count=192,
        mem_bandwidth_gbps=8000, sram_per_sm_kb=256,
        crossover_seq_len=512,   # 192 SMs + 256KB SRAM → near-zero overhead
        compute_class='hpc',
    ),
    'GB200': GPUProfile(
        name='GB200', sm_count=192,
        mem_bandwidth_gbps=8000, sram_per_sm_kb=256,
        crossover_seq_len=512,
        compute_class='hpc',
    ),
}

# Reference GPU for bandwidth-ratio estimation
_REFERENCE_PROFILE = _GPU_PROFILES['RTX 4060 Laptop']


def _detect_gpu_profile(device: torch.device | None = None) -> GPUProfile:
    """Auto-detect GPU and return the best-matching hardware profile.

    Matching priority:
    1. Exact name match from known profiles
    2. Substring match (e.g., "4090" in "NVIDIA GeForce RTX 4090")
    3. Bandwidth-ratio estimation for unknown GPUs
    """
    if not torch.cuda.is_available():
        # CPU fallback — use reference profile (conservative)
        return _REFERENCE_PROFILE

    if device is None:
        device = torch.device('cuda', torch.cuda.current_device())

    idx = device.index if device.index is not None else 0
    props = torch.cuda.get_device_properties(idx)
    gpu_name = props.name  # e.g., "NVIDIA GeForce RTX 4060 Laptop GPU"
    sm_count = props.multi_processor_count

    # 1. Try exact key match
    for key, profile in _GPU_PROFILES.items():
        if key in gpu_name:
            return profile

    # 2. Try partial matches for common identifiers
    _PARTIAL_KEYS = ['B200', 'GB200', 'H200', 'H100', 'A100',
                     '4090', '4080', '4070', '4060']
    for partial in _PARTIAL_KEYS:
        if partial in gpu_name:
            for key, profile in _GPU_PROFILES.items():
                if partial in key:
                    return profile

    # 3. Unknown GPU — estimate crossover via bandwidth-ratio model
    # total_memory as proxy for bandwidth class (rough but workable)
    total_mem_gb = props.total_mem / (1024 ** 3)

    # Heuristic bandwidth estimation from memory size + SM count:
    #   Consumer (<16 GB): ~256-500 GB/s
    #   Prosumer (16-48 GB): ~500-1000 GB/s
    #   Datacenter (48-80 GB): ~2000-3500 GB/s
    #   HPC (>80 GB): ~4000-8000 GB/s
    if total_mem_gb > 80:
        est_bw = 4000.0
    elif total_mem_gb > 48:
        est_bw = 2500.0
    elif total_mem_gb > 16:
        est_bw = 700.0
    else:
        est_bw = 300.0

    # Bandwidth-ratio scaling with SM correction:
    #   crossover ∝ (bw / ref_bw) × ref_crossover × (ref_sms / sms)
    bw_ratio = est_bw / _REFERENCE_PROFILE.mem_bandwidth_gbps
    sm_ratio = _REFERENCE_PROFILE.sm_count / max(sm_count, 1)
    raw_crossover = _REFERENCE_PROFILE.crossover_seq_len * bw_ratio * sm_ratio

    # Clamp to sensible range and round to nearest power of 2
    clamped = max(512, min(8192, int(raw_crossover)))
    # Round to nearest power of 2
    import math
    crossover = 1 << int(math.log2(clamped) + 0.5)

    return GPUProfile(
        name=f'{gpu_name} (auto-detected)',
        sm_count=sm_count,
        mem_bandwidth_gbps=est_bw,
        sram_per_sm_kb=100.0,  # conservative default
        crossover_seq_len=crossover,
        compute_class='unknown',
    )


# Module-level cache: detect once, reuse forever.
_cached_profile: GPUProfile | None = None


def get_hardware_profile(device: torch.device | None = None) -> GPUProfile:
    """Get the cached GPU profile, detecting on first call."""
    global _cached_profile
    if _cached_profile is None:
        _cached_profile = _detect_gpu_profile(device)
    return _cached_profile


def get_crossover_threshold(device: torch.device | None = None) -> int:
    """Get the dynamic crossover threshold for the current GPU.

    Returns the sequence length below which OrthoCache auto-bypasses
    to dense attention because spectral analysis overhead exceeds
    eviction savings on this specific hardware.
    """
    return get_hardware_profile(device).crossover_seq_len


# Legacy constant — kept for backward compatibility but now delegates
# to the dynamic dispatcher. Code that references CROSSOVER_SEQ_LEN
# directly will get the RTX 4060 Laptop value (4096).
CROSSOVER_SEQ_LEN = _REFERENCE_PROFILE.crossover_seq_len



def orthocache_forward(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_size: int = 512,
    zeta_max: float = 5.0,
    tau: float | None = None,
    mode: str = 'compact',
    crossover_threshold: int = 0,
    telemetry: TelemetryLevel = TelemetryLevel.FLAGS,
) -> tuple[torch.Tensor, dict]:
    """Full OrthoCache pipeline: spectral analysis → eviction → attention.

    This is the primary public API. It runs the complete OrthoCache flow:
    1. Compute spectral decay ratio (ζ) for all blocks
    2. Generate two-gate eviction mask (logit bound + ζ coherence)
    3. Either compact the KV-cache or apply predicated sparse attention
    4. Return the attention output and detailed metadata

    Args:
        q: Query tensor of shape (seq_len_q, num_heads, head_dim).
        keys: Key tensor of shape (seq_len_k, num_heads, head_dim).
        values: Value tensor of shape (seq_len_k, num_heads, head_dim).
        block_size: Tokens per block (must be 512 for FWHT).
        zeta_max: Maximum spectral decay ratio. Blocks with ζ > zeta_max
            are evicted regardless of query-aware logit bound.
            Default 5.0 is a conservative starting point.
        tau: Query-aware logit bound threshold. If None, computed
            automatically as mean - 1σ of the logit bounds.
        mode: Execution mode:
            - 'compact': Stream compaction (Phase C). Physically removes
              evicted blocks before attention. Recommended.
            - 'dense': Full dense attention (baseline). Ignores all
              eviction logic. For comparison only.
            - 'triton_fused': Phase 7 God Kernel. Fused FWHT + ζ + attention
              in a single Triton kernel launch. Uses TILE_SIZE=64.
        crossover_threshold: Context length below which eviction is bypassed
            and dense attention is used (default: auto from hardware profile).
        telemetry: Controls metadata verbosity:
            - SILENT: No metadata. Maximum throughput for production.
            - FLAGS: Bitmask flags + numeric scalars only. No strings. (default)
            - VERBOSE: Full human-readable metadata. For debugging.

    Returns:
        Tuple of (output, metadata):
        - output: Attention result, shape (seq_len_q, num_heads, head_dim).
        - metadata: Dict with timing, eviction stats, ζ distribution.
          In SILENT mode, metadata is an empty dict.
          In FLAGS mode, 'bypass_flags' is an int (BypassFlag bitmask).
          In VERBOSE mode, 'crossover_reason' is a human-readable string.
    """
    seq_len_k, num_heads, head_dim = keys.shape
    seq_len_q = q.shape[0]
    num_blocks = seq_len_k // block_size

    # Check for adaptive crossover fallback (short sequences bypass eviction)
    crossover_fallback = False
    original_mode = mode
    if seq_len_k < crossover_threshold and mode in ('compact', 'triton_fused'):
        mode = 'dense'
        crossover_fallback = True

    # --- Metadata construction (telemetry-level gated) ---
    if telemetry == TelemetryLevel.SILENT:
        metadata: dict = {}
    else:
        metadata = {
            'mode': original_mode,
            'actual_mode': mode,
            'crossover_fallback': crossover_fallback,
            'crossover_threshold': crossover_threshold,
            'seq_len_q': seq_len_q,
            'seq_len_k': seq_len_k,
            'num_blocks': num_blocks,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'block_size': block_size,
            'zeta_max': zeta_max,
        }

    # --- Dense baseline ---
    if mode == 'dense':
        t0 = time.perf_counter()
        output = _dense_attention(q, keys, values, head_dim)
        metadata['latency_ms'] = (time.perf_counter() - t0) * 1000
        metadata['eviction_rate'] = 0.0
        return output, metadata

    # --- Adaptive crossover bypass (hardware-aware) ---
    # Threshold adapts per GPU: 4096 on RTX 4060, 1024 on H100, 512 on B200.
    # Below the threshold, spectral analysis overhead exceeds eviction savings.
    hw_profile = get_hardware_profile(q.device)
    dynamic_threshold = hw_profile.crossover_seq_len
    if seq_len_k < dynamic_threshold and mode != 'triton_fused':
        t0 = time.perf_counter()
        output = _dense_attention(q, keys, values, head_dim)
        if telemetry >= TelemetryLevel.FLAGS:
            flags = BypassFlag.CROSSOVER_BYPASS
            if hw_profile.compute_class == 'unknown':
                flags |= BypassFlag.HW_AUTO_DETECTED
            metadata['latency_ms'] = (time.perf_counter() - t0) * 1000
            metadata['eviction_rate'] = 0.0
            metadata['bypass_flags'] = int(flags)
            metadata['crossover_threshold'] = dynamic_threshold
            metadata['gpu_profile'] = hw_profile.name
        if telemetry >= TelemetryLevel.VERBOSE:
            metadata['crossover_reason'] = format_bypass_reason(
                flags, seq_len_k, dynamic_threshold, hw_profile
            )
        return output, metadata

    # --- Triton fused: Split-K God Kernel (Phase 7b) ---
    if mode == 'triton_fused':
        from orthocache_gpu.triton_kernels.fused_eviction import (
            fused_orthocache_attention_v2 as _fused_attn_v2,
        )
        t0 = time.perf_counter()

        # Transpose (seq, heads, dim) → (heads, seq, dim) for the kernel.
        # Single launch: grid=(num_heads, num_splits) — no Python loop.
        q_fused = q.squeeze(0) if seq_len_q == 1 else q[0]  # (num_heads, head_dim)
        # Handle multi-token query by taking first token (decode mode)
        if q_fused.ndim == 1:
            q_fused = q_fused.unsqueeze(0)  # (1, head_dim) → need (heads, dim)
        # q: (seq_q, heads, dim) → take first query token → (heads, dim)
        q_heads = q[0]  # (num_heads, head_dim)
        k_heads = keys.permute(1, 0, 2).contiguous()   # (heads, seq, dim)
        v_heads = values.permute(1, 0, 2).contiguous()  # (heads, seq, dim)

        output_heads, fused_meta = _fused_attn_v2(
            q_heads, k_heads, v_heads, zeta_max=zeta_max
        )
        # output_heads: (num_heads, head_dim) → (1, num_heads, head_dim)
        output = output_heads.unsqueeze(0)

        tile_size = 64  # God Kernel tile size
        num_tiles_fused = seq_len_k // tile_size

        metadata['latency_ms'] = (time.perf_counter() - t0) * 1000
        metadata['eviction_rate'] = fused_meta.get('eviction_rate', 0.0)
        metadata['tile_size_fused'] = tile_size
        metadata['num_tiles_fused'] = num_tiles_fused
        metadata['num_splits'] = fused_meta.get('num_splits', 1)
        metadata['tile_assignment'] = 'interleaved'
        return output, metadata


    # --- Spectral analysis ---
    t_spectral = time.perf_counter()

    # Compute ζ for all blocks
    zeta = compute_spectral_decay_ratio(keys, block_size)  # (num_blocks, num_heads)

    # Auto-compute tau if not provided
    if tau is None:
        bounds = _compute_auto_tau(q, keys, block_size)
        tau = float(bounds)
        metadata['tau_auto'] = True
    else:
        metadata['tau_auto'] = False

    metadata['tau'] = tau

    # Two-gate mask: logit bound AND spectral coherence
    block_mask = compute_multiband_mask(q, keys, tau, zeta_max, block_size)
    # block_mask: (num_blocks, num_heads) boolean

    t_spectral_end = time.perf_counter()
    metadata['spectral_ms'] = (t_spectral_end - t_spectral) * 1000

    # ζ statistics
    zeta_any_head = torch.mean(zeta, dim=-1)  # (num_blocks,)
    metadata['zeta_mean'] = float(torch.mean(zeta_any_head).item())
    metadata['zeta_std'] = float(torch.std(zeta_any_head).item())
    metadata['zeta_min'] = float(torch.min(zeta_any_head).item())
    metadata['zeta_max_observed'] = float(torch.max(zeta_any_head).item())

    # Eviction stats
    blocks_retained = torch.sum(torch.any(block_mask, dim=-1).to(torch.int32))
    metadata['blocks_retained'] = int(blocks_retained.item())
    metadata['blocks_evicted'] = int(num_blocks - blocks_retained.item())
    metadata['eviction_rate'] = float(1.0 - blocks_retained.item() / num_blocks)

    # --- Perfect Eviction Classification ---
    # Classify evicted blocks into deterministic (TV=0) and statistical regimes
    try:
        from orthocache_gpu.perfect_eviction import classify_eviction
        from orthocache_gpu.spectral_energy import compute_block_energy

        block_energies = compute_block_energy(keys, block_size)

        # Compute z_max from retained logits (approximate via query-key max)
        scale = torch.sqrt(torch.tensor(float(head_dim), device=q.device))
        with torch.no_grad():
            # Sample max logit from retained blocks for z_max estimation
            unified_mask_for_zmax = torch.any(block_mask, dim=-1)  # (num_blocks,)
            if unified_mask_for_zmax.any():
                # Use the max logit bound as a z_max proxy
                from orthocache_gpu.spectral_energy import compute_query_aware_bounds
                all_bounds = compute_query_aware_bounds(q, keys, block_size)
                max_bounds = torch.max(all_bounds, dim=0).values  # (num_blocks, num_heads)
                retained_bounds = max_bounds[unified_mask_for_zmax]
                z_max_estimate = torch.max(retained_bounds)
            else:
                z_max_estimate = torch.tensor(0.0, device=q.device)

        eviction_meta = classify_eviction(
            q, block_energies, z_max_estimate, block_mask, head_dim
        )
        metadata['perfect_eviction_blocks'] = eviction_meta.num_perfect
        metadata['statistical_eviction_blocks'] = eviction_meta.num_statistical
        metadata['perfect_eviction_rate'] = (
            eviction_meta.num_perfect / max(1, num_blocks - int(blocks_retained.item()))
            if num_blocks > int(blocks_retained.item()) else 0.0
        )
    except ImportError:
        # perfect_eviction module not available — skip classification
        metadata['perfect_eviction_blocks'] = None
        metadata['statistical_eviction_blocks'] = None
        metadata['perfect_eviction_rate'] = None

    # --- Attention ---
    t_attn = time.perf_counter()

    if mode == 'compact':
        # Phase C: Stream Compaction + Adaptive Attention
        # Use the unified mask (any-head retention) for block selection
        unified_mask = torch.any(block_mask, dim=-1)  # (num_blocks,)
        output, attn_stats = orthocache_attention(
            q, keys, values, unified_mask, block_size=block_size
        )
        metadata.update({
            'compact_num_active': int(blocks_retained.item()),
        })
    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. Use 'dense', 'compact', or 'triton_fused'."
        )

    t_attn_end = time.perf_counter()
    metadata['attention_ms'] = (t_attn_end - t_attn) * 1000
    metadata['total_ms'] = (t_attn_end - t_spectral) * 1000

    return output, metadata


def _dense_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Standard dense attention (baseline)."""
    scale = torch.sqrt(torch.tensor(head_dim, dtype=torch.float32, device=q.device))
    logits = torch.einsum('qhd,khd->qkh', q, keys) / scale
    weights = F.softmax(logits, dim=1)
    return torch.einsum('qkh,khd->qhd', weights, values)


def _compute_auto_tau(
    q: torch.Tensor,
    keys: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Auto-compute tau as mean - 1σ of query-aware logit bounds."""
    from orthocache_gpu.spectral_energy import compute_query_aware_bounds
    bounds = compute_query_aware_bounds(q, keys, block_size)
    # bounds: (seq_len_q, num_blocks, num_heads)
    max_bounds = torch.max(bounds, dim=0).values  # (num_blocks, num_heads)
    mean_b = torch.mean(max_bounds)
    std_b = torch.std(max_bounds)
    return mean_b - std_b
