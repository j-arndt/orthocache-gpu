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

# Empirically measured crossover point: OrthoCache is slower than dense
# attention below this sequence length due to spectral analysis overhead.
# Measured on RTX 4060 Laptop (Ada Lovelace): 0.51× at 1K, 0.91× at 2K,
# 1.09× at 4K. The crossover is between 2K-4K tokens; we use 4K as the
# conservative threshold to ensure OrthoCache is always a net win.
CROSSOVER_SEQ_LEN = 4096


def orthocache_forward(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_size: int = 512,
    zeta_max: float = 5.0,
    tau: float | None = None,
    mode: str = 'compact',
    crossover_threshold: int = 0,
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
            and dense attention is used (default: 4096).

    Returns:
        Tuple of (output, metadata):
        - output: Attention result, shape (seq_len_q, num_heads, head_dim).
        - metadata: Dict with timing, eviction stats, ζ distribution.
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

    # --- Adaptive crossover bypass ---
    # Below CROSSOVER_SEQ_LEN, spectral analysis overhead exceeds eviction
    # savings (0.51× at 1K, 0.91× at 2K). Auto-bypass to dense attention.
    if seq_len_k < CROSSOVER_SEQ_LEN and mode != 'triton_fused':
        t0 = time.perf_counter()
        output = _dense_attention(q, keys, values, head_dim)
        metadata['latency_ms'] = (time.perf_counter() - t0) * 1000
        metadata['eviction_rate'] = 0.0
        metadata['crossover_bypass'] = True
        metadata['crossover_reason'] = (
            f'seq_len_k={seq_len_k} < CROSSOVER_SEQ_LEN={CROSSOVER_SEQ_LEN}; '
            f'spectral analysis overhead would degrade performance'
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
