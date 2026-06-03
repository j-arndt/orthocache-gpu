"""Dynamic-bound attention kernel using Python while loop (GPU Edition).

This replaces the Pallas unrolled-loop kernel with a PyTorch kernel
that uses a Python while loop for truly dynamic iteration. torch.compile
optimizes the loop into efficient GPU operations whose condition is
checked at runtime — meaning the loop terminates when i >= num_active.

This is the Phase D kernel. It runs on NVIDIA GPUs without any custom
CUDA or compiler modifications.

Key difference from sparse_attention.py:
  - sparse_attention.py: `for b in range(num_blocks):` → Python unrolls
    at trace time → compiler sees N sequential blocks → predication
  - This file: `while i < num_active:` → dynamic termination →
    true loop elision
"""

import torch
import torch.nn.functional as F
from functools import partial


@torch.compile
def dynamic_compact_attention(
    q: torch.Tensor,
    compact_keys: torch.Tensor,
    compact_values: torch.Tensor,
    num_active: torch.Tensor,
    block_size: int = 512,
) -> torch.Tensor:
    """Attention over compacted KV-cache with dynamic loop bound.

    Uses a Python while loop so torch.compile can optimize the
    dynamic termination, achieving true loop elision.

    Args:
        q: Query tensor (seq_len_q, head_dim). Single head.
        compact_keys: Compacted key blocks (num_blocks * block_size, head_dim).
            First num_active * block_size entries are real data.
        compact_values: Same shape as compact_keys.
        num_active: Scalar int32. Number of active (non-evicted) blocks.
        block_size: Tokens per block.

    Returns:
        Attention output (seq_len_q, head_dim).
    """
    seq_len_q, head_dim = q.shape
    scale = 1.0 / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32, device=q.device))

    def single_query_attention(q_vec: torch.Tensor) -> torch.Tensor:
        """Process one query vector against all active blocks."""
        # State: loop_var, running_max, running_sum, running_output
        i = 0
        r_max = torch.tensor(-1e9, dtype=torch.float32, device=q.device)
        r_sum = torch.tensor(0.0, dtype=torch.float32, device=q.device)
        r_out = torch.zeros((head_dim,), dtype=torch.float32, device=q.device)

        num_active_val = num_active.item() if isinstance(num_active, torch.Tensor) else int(num_active)

        while i < num_active_val:
            # Load block i using tensor slicing
            block_start = i * block_size
            k_block = compact_keys[block_start:block_start + block_size, :]  # (block_size, head_dim)
            v_block = compact_values[block_start:block_start + block_size, :]  # (block_size, head_dim)

            # Compute logits: q_vec . k_block^T
            logits = torch.mv(k_block, q_vec) * scale  # (block_size,)

            # Online softmax: find local max, update running stats
            local_max = torch.max(logits)
            new_max = torch.maximum(r_max, local_max)

            exp_logits = torch.exp(logits - new_max)       # (block_size,)
            scale_old = torch.exp(r_max - new_max)          # scalar

            r_sum = r_sum * scale_old + torch.sum(exp_logits)
            r_out = r_out * scale_old + torch.mv(v_block.T, exp_logits)  # (head_dim,)
            r_max = new_max

            i += 1

        # Normalize
        return r_out / torch.maximum(r_sum, torch.tensor(1e-9, device=q.device))

    # vmap over query positions
    q_f32 = q.to(torch.float32)
    output = torch.vmap(single_query_attention)(q_f32)

    return output


@torch.compile
def dynamic_multihead_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int = 512,
) -> torch.Tensor:
    """Full multi-head attention with dynamic compaction and while loop.

    Uses a UNIFIED num_active across all heads (any-head retention) so
    we can vmap the while-loop kernel across heads for parallel execution.

    Args:
        q: Query (seq_len_q, num_heads, head_dim).
        keys: Keys (seq_len_k, num_heads, head_dim).
        values: Values (seq_len_k, num_heads, head_dim).
        block_mask: Boolean mask (num_blocks, num_heads).
        block_size: Tokens per block.

    Returns:
        Attention output (seq_len_q, num_heads, head_dim).
    """
    seq_len_q, num_heads, head_dim = q.shape
    seq_len_k = keys.shape[0]
    num_blocks = seq_len_k // block_size

    # Unified mask: retain block if ANY head says retain
    block_active = torch.any(block_mask, dim=-1)  # (num_blocks,)
    active_int = block_active.to(torch.int32)
    num_active = torch.sum(active_int)

    # Sort order: active blocks first (stable preserves relative order)
    sort_order = torch.argsort(-active_int, stable=True)  # (num_blocks,)

    # Compact ALL heads at once using the unified sort order
    # keys: (seq_len_k, num_heads, head_dim) → (num_blocks, block_size, num_heads, head_dim)
    k_blocked = keys.reshape(num_blocks, block_size, num_heads, head_dim)
    v_blocked = values.reshape(num_blocks, block_size, num_heads, head_dim)

    compact_k = k_blocked[sort_order]  # (num_blocks, block_size, num_heads, head_dim)
    compact_v = v_blocked[sort_order]

    # Transpose to (num_heads, num_blocks * block_size, head_dim) for vmap
    compact_k_flat = compact_k.reshape(num_blocks * block_size, num_heads, head_dim)
    compact_v_flat = compact_v.reshape(num_blocks * block_size, num_heads, head_dim)

    # (num_heads, seq_len_k, head_dim)
    compact_k_heads = compact_k_flat.permute(1, 0, 2)
    compact_v_heads = compact_v_flat.permute(1, 0, 2)

    # q: (seq_len_q, num_heads, head_dim) → (num_heads, seq_len_q, head_dim)
    q_heads = q.permute(1, 0, 2)

    # vmap dynamic_compact_attention over the head dimension
    # Each head gets the same num_active (unified mask)
    vmapped_attn = torch.vmap(
        lambda q_h, k_h, v_h: dynamic_compact_attention(
            q_h, k_h, v_h, num_active, block_size=block_size
        )
    )

    # (num_heads, seq_len_q, head_dim)
    out_heads = vmapped_attn(q_heads, compact_k_heads, compact_v_heads)

    # Transpose back: (num_heads, seq_len_q, head_dim) → (seq_len_q, num_heads, head_dim)
    return out_heads.permute(1, 0, 2)
