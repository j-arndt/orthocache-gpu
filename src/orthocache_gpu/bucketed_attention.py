"""Bucketed compaction + attention (GPU Edition).

Instead of a dynamic while_loop (high per-iteration overhead) or a
predicated unrolled loop (zero speedup), this module:

1. Stream-compacts the KV-cache (active blocks first)
2. Pulls num_active to host (single scalar sync)
3. Rounds up to nearest bucket size (power of 2)
4. Calls the lean attention backend with bucket blocks instead of all blocks

On the GPU edition, this uses lean_bucketed_attention as a placeholder
until Triton fused attention kernels are implemented. torch.compile
auto-caches compiled graphs per bucket size, matching the JAX/Pallas
per-bucket caching behavior.
"""

import torch
from functools import partial

from .compaction import stream_compact
from .lean_attention import lean_bucketed_attention


# Bucket sizes: powers of 2 from 1 to 512.
# Covers up to 256K context at block_size=512.
BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def _next_bucket(n: int) -> int:
    """Round up to the nearest bucket size."""
    for b in BUCKETS:
        if b >= n:
            return b
    return n  # Fallback: use exact count if > max bucket


def bucketed_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int = 512,
) -> tuple[torch.Tensor, dict]:
    """Compacted attention using bucketed dispatch.
    
    This is the Phase D kernel that achieves both:
    - Efficient GPU utilization (torch.compile-optimized)
    - Proportional Δτ scaling with eviction rate
    
    NOTE: Currently wired to lean_bucketed_attention as a placeholder.
    Will be replaced with Triton fused attention kernels for full
    GPU memory hierarchy exploitation.
    
    Args:
        q: Query (seq_len_q, num_heads, head_dim).
        keys: Keys (seq_len_k, num_heads, head_dim).
        values: Values (seq_len_k, num_heads, head_dim).
        block_mask: Boolean (num_blocks, num_heads).
        block_size: Tokens per block.
        
    Returns:
        (output, metadata) where output is (seq_len_q, num_heads, head_dim).
    """
    seq_len_k, num_heads, head_dim = keys.shape
    num_blocks = seq_len_k // block_size
    
    # Step 1: Stream compact — active blocks first, padded with zeros
    compact_keys, compact_values, active_indices, num_active = stream_compact(
        keys, values, block_mask, block_size
    )
    # compact_keys: (num_blocks, block_size, num_heads, head_dim)
    # num_active: scalar int32 on device
    
    # Step 2: Pull num_active to host (single scalar transfer)
    # This is the only sync point. ~microseconds for one int32.
    n_active = int(num_active)
    
    # Step 3: Determine bucket
    if n_active == 0:
        # All blocks evicted — return zeros
        return torch.zeros_like(q), {
            'num_active': 0, 'bucket': 0, 'num_blocks': num_blocks,
            'eviction_rate': 1.0,
        }
    
    bucket = _next_bucket(n_active)
    
    # Step 4: Slice compact tensor to bucket size
    # compact_keys[:bucket] gives us the first `bucket` blocks
    ck = compact_keys[:bucket]  # (bucket, block_size, num_heads, head_dim)
    cv = compact_values[:bucket]
    
    # Reshape to (bucket * block_size, num_heads, head_dim)
    ck_flat = ck.reshape(bucket * block_size, num_heads, head_dim)
    cv_flat = cv.reshape(bucket * block_size, num_heads, head_dim)
    
    # All-true mask (every block in the compact tensor is active)
    bucket_mask = torch.ones((bucket, num_heads), dtype=torch.bool, device=keys.device)
    
    # Step 5: Attention with REDUCED block count
    # =========================================================================
    # TODO(triton): Replace lean_bucketed_attention with Triton fused kernel.
    # The Triton kernel should be dispatched per-bucket for optimal occupancy.
    # torch.compile will cache the compiled graph per bucket size.
    # =========================================================================
    output, _ = lean_bucketed_attention(
        q, ck_flat, cv_flat, bucket_mask, block_size
    )
    
    metadata = {
        'num_active': n_active,
        'bucket': bucket,
        'num_blocks': num_blocks,
        'eviction_rate': 1.0 - n_active / num_blocks,
    }
    
    return output, metadata
