"""Stream compaction for OrthoCache block-sparse attention (GPU Edition).

Implements the user-space stream compaction primitive that transforms a
block-masked KV-cache into a dense, compacted tensor containing only
retained blocks. This eliminates the need for predication in the attention
kernel — the loop iterates over only active blocks.

Architecture:
    1. Prefix sum on boolean block mask → cumulative index array
    2. Gather active block indices
    3. Gather active key/value blocks into compacted tensors
    4. Return compacted tensors + indirection table

Design decision: We use the "pad to max" strategy for torch.compile compatibility.
The output tensors are always shaped [num_blocks, block_size, heads, dim],
but only the first num_active blocks contain real data. The rest are zeros.
This avoids dynamic shapes while letting the attention kernel loop over
[0, num_active) instead of [0, num_blocks).

See docs/xla_pass_design.md §2 for the compiler-level version of this.
"""

import torch
from functools import partial
from .lean_attention import lean_bucketed_attention


def stream_compact(
    keys: torch.Tensor,
    values: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact KV-cache blocks, retaining only active (non-evicted) blocks.
    
    Takes the full KV-cache and a boolean block mask, returns compacted
    tensors containing only the retained blocks in a contiguous layout.
    
    Args:
        keys: Key tensor of shape (seq_len, num_heads, head_dim).
        values: Value tensor of shape (seq_len, num_heads, head_dim).
        block_mask: Boolean tensor of shape (num_blocks, num_heads).
            A block is retained if mask is True for ANY head (logical OR).
        block_size: Token count per block (default: 512).
        
    Returns:
        Tuple of (compact_keys, compact_values, active_indices, num_active):
        - compact_keys: Shape (num_blocks, block_size, num_heads, head_dim).
            First num_active blocks are real data, rest are zeros.
        - compact_values: Same shape and layout as compact_keys.
        - active_indices: Shape (num_blocks,) int32 tensor. 
            active_indices[i] = original block index for compacted position i.
            Valid for i < num_active; undefined for i >= num_active.
        - num_active: Scalar int32. Number of retained blocks.
    """
    seq_len, num_heads, head_dim = keys.shape
    num_blocks = seq_len // block_size
    
    # Reshape into blocks: (num_blocks, block_size, num_heads, head_dim)
    keys_blocked = keys.reshape(num_blocks, block_size, num_heads, head_dim)
    values_blocked = values.reshape(num_blocks, block_size, num_heads, head_dim)
    
    # Reduce mask across heads: retain block if ANY head says retain
    # block_mask: (num_blocks, num_heads) → (num_blocks,)
    block_active = torch.any(block_mask, dim=-1)  # (num_blocks,) boolean
    
    # --- Stream Compaction via Prefix Sum ---
    # 
    # Convert boolean mask to int for prefix sum:
    #   active = [1, 0, 1, 1, 0, 1, 0, 0]
    #   prefix = [1, 1, 2, 3, 3, 4, 4, 4]  (inclusive prefix sum)
    # 
    # Active indices are found by sorting: blocks with mask=True
    # get low indices, blocks with mask=False get high indices.
    
    active_int = block_active.to(torch.int32)  # (num_blocks,)
    num_active = torch.sum(active_int)  # scalar
    
    # Use argsort on the negated mask to push active blocks to the front.
    # -active_int: active blocks get -1 (sorts first), inactive get 0 (sorts last).
    # stable sort preserves relative order within active blocks.
    sort_order = torch.argsort(-active_int, stable=True)  # (num_blocks,)
    
    # sort_order[0:num_active] are the original indices of active blocks, in order.
    # sort_order[num_active:] are the original indices of inactive blocks.
    active_indices = sort_order  # (num_blocks,) — valid positions [0, num_active)
    
    # Gather active blocks into compacted layout
    compact_keys = keys_blocked[active_indices]    # (num_blocks, block_size, num_heads, head_dim)
    compact_values = values_blocked[active_indices]  # same
    
    # Zero out inactive positions to prevent data leakage
    # Create a mask: (num_blocks,) where position i is True if i < num_active
    position_mask = torch.arange(num_blocks, device=keys.device) < num_active  # (num_blocks,)
    position_mask_4d = position_mask[:, None, None, None]  # (num_blocks, 1, 1, 1)
    
    compact_keys = torch.where(position_mask_4d, compact_keys, torch.zeros_like(compact_keys))
    compact_values = torch.where(position_mask_4d, compact_values, torch.zeros_like(compact_values))
    
    return compact_keys, compact_values, active_indices, num_active


def stream_decompact(
    compact_output: torch.Tensor,
    active_indices: torch.Tensor,
    num_active: torch.Tensor,
    num_blocks_original: int,
    block_size: int = 512,
) -> torch.Tensor:
    """Reverse the compaction: scatter compacted blocks back to original positions.
    
    Useful for verification (compact → decompact should recover the original
    masked tensor).
    
    Args:
        compact_output: Shape (num_blocks, block_size, num_heads, head_dim).
        active_indices: Shape (num_blocks,) from stream_compact.
        num_active: Scalar from stream_compact.
        num_blocks_original: Original number of blocks before compaction.
        block_size: Token count per block.
        
    Returns:
        Scattered tensor of shape (num_blocks_original, block_size, num_heads, head_dim)
        with active blocks placed at their original positions, inactive blocks zeroed.
    """
    _, bs, nh, hd = compact_output.shape
    
    # Initialize output with zeros
    output = torch.zeros((num_blocks_original, bs, nh, hd), dtype=compact_output.dtype, device=compact_output.device)
    
    # Scatter: for each compacted position i < num_active,
    # place compact_output[i] at position active_indices[i]
    position_mask = torch.arange(num_blocks_original, device=compact_output.device) < num_active
    
    # active_indices tells us WHERE each compacted block came from
    # We need to scatter back: output[active_indices[i]] = compact_output[i]
    output[active_indices] = compact_output
    
    # Zero out positions beyond num_active (they got scattered to wrong places)
    # Reconstruct the is_active mask from active_indices + position_mask
    is_active = torch.zeros(num_blocks_original, dtype=torch.bool, device=compact_output.device)
    is_active[active_indices] = position_mask
    
    output = torch.where(is_active[:, None, None, None], output, torch.zeros_like(output))
    
    return output


@torch.compile
def compact_and_attend(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int = 512,
) -> tuple[torch.Tensor, dict]:
    """Full compaction + attention pipeline (torch.compile-optimized).
    
    This is the user-space compaction path (Phase C). It:
    1. Compacts the KV-cache using stream_compact
    2. Runs attention over only the active blocks
    3. Returns the output + metadata
    
    NOTE: This uses lean_bucketed_attention as the attention backend.
    Triton kernel integration point is marked below for future replacement.
    
    Args:
        q: Query tensor (seq_len_q, num_heads, head_dim).
        keys: Key tensor (seq_len_k, num_heads, head_dim).
        values: Value tensor (seq_len_k, num_heads, head_dim).
        block_mask: Boolean mask (num_blocks, num_heads).
        block_size: Tokens per block.
        
    Returns:
        Tuple of (output, metadata):
        - output: Attention output (seq_len_q, num_heads, head_dim).
        - metadata: Dict with num_active, num_blocks, eviction_rate.
    """
    seq_len_k, num_heads, head_dim = keys.shape
    num_blocks = seq_len_k // block_size
    
    # Step 1: Compact
    compact_keys, compact_values, active_indices, num_active = stream_compact(
        keys, values, block_mask, block_size
    )
    
    # Step 2: Flatten compact tensors back to (seq_len, heads, dim) for attention
    # The compact tensors are (num_blocks, block_size, heads, dim)
    # We need (num_blocks * block_size, heads, dim) but only first
    # num_active * block_size tokens are real
    compact_keys_flat = compact_keys.reshape(num_blocks * block_size, num_heads, head_dim)
    compact_values_flat = compact_values.reshape(num_blocks * block_size, num_heads, head_dim)
    
    # Step 3: Create an all-True mask for the compact tensor
    # (the compaction already filtered out inactive blocks)
    compact_mask = torch.arange(num_blocks, device=keys.device) < num_active  # (num_blocks,)
    compact_mask_heads = compact_mask[:, None].expand(num_blocks, num_heads)  # (num_blocks, num_heads)
    
    # Step 4: Run attention on the compact tensor
    # =========================================================================
    # TODO(triton): Replace lean_bucketed_attention with Triton fused attention
    # kernel from .triton_kernels once implemented. The Triton kernel should:
    #   - Fuse the gather + QK^T + softmax + V multiply into a single kernel
    #   - Use shared memory tiling for the block-sparse access pattern
    #   - Target H100/B200 SM occupancy
    # See: orthocache_gpu/triton_kernels/__init__.py
    # =========================================================================
    output, meta = lean_bucketed_attention(
        q, compact_keys_flat, compact_values_flat,
        compact_mask_heads, block_size
    )
    
    # Metadata
    eviction_rate = 1.0 - (int(num_active) / num_blocks)
    
    return output, {
        'num_active': int(num_active),
        'num_blocks': num_blocks,
        'eviction_rate': eviction_rate,
    }
