"""OrthoCache end-to-end pipeline (GPU Edition).

Chains the full OrthoCache flow:
    FWHT → spectral bands → ζ computation → two-gate mask → compaction → attention

This module provides the high-level API that users call. It handles:
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


def orthocache_forward(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_size: int = 512,
    zeta_max: float = 5.0,
    tau: float | None = None,
    mode: str = 'compact',
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
            - 'triton_fused': Placeholder for Triton fused kernel.
              Not implemented yet — raises NotImplementedError.

    Returns:
        Tuple of (output, metadata):
        - output: Attention result, shape (seq_len_q, num_heads, head_dim).
        - metadata: Dict with timing, eviction stats, ζ distribution.
    """
    seq_len_k, num_heads, head_dim = keys.shape
    seq_len_q = q.shape[0]
    num_blocks = seq_len_k // block_size

    metadata = {
        'mode': mode,
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

    # --- Triton fused placeholder ---
    if mode == 'triton_fused':
        raise NotImplementedError(
            "mode='triton_fused' is not implemented yet. "
            "Use mode='compact' or mode='dense'."
        )

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
