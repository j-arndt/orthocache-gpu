"""Lean bucketed attention: no physical compaction, just indirect gather + reduced attention.

The expensive part of bucketed_attention.py was stream_compact copying
512MB of KV data. This version:
1. Computes active_indices from the mask (argsort on 64 elements — microseconds)
2. Gathers ONLY the active blocks using advanced indexing  
3. Runs dense einsum attention on the reduced set

The gather + einsum gets fused by torch.compile into an optimized CUDA graph.
"""

import torch
import torch.nn.functional as F
from functools import partial

BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

def _next_bucket(n):
    for b in BUCKETS:
        if b >= n:
            return b
    return n


def lean_bucketed_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int = 512,
) -> tuple[torch.Tensor, dict]:
    """Lean bucketed attention: gather active blocks + dense attention.
    
    No stream_compact. No full-tensor copy. Just index the blocks we need.
    """
    seq_len_k, num_heads, head_dim = keys.shape
    num_blocks = seq_len_k // block_size
    
    # Step 1: Compute active indices (CHEAP — operates on tiny mask array)
    block_active = torch.any(block_mask, dim=-1)  # (num_blocks,)
    active_int = block_active.to(torch.int32)
    num_active_dev = torch.sum(active_int)
    sort_order = torch.argsort(-active_int, stable=True)  # (num_blocks,)
    
    # Step 2: Pull num_active to host for bucketing (single int32 sync)
    n_active = int(num_active_dev)
    
    if n_active == 0:
        return torch.zeros_like(q), {'num_active': 0, 'bucket': 0, 'eviction_rate': 1.0}
    
    bucket = _next_bucket(n_active)
    active_idx = sort_order[:bucket]  # (bucket,) — just indices, no data copy yet
    
    # Step 3: Gather active blocks (torch.compile fuses this with the attention einsum)
    k_blocked = keys.reshape(num_blocks, block_size, num_heads, head_dim)
    v_blocked = values.reshape(num_blocks, block_size, num_heads, head_dim)
    
    k_active = k_blocked[active_idx]  # (bucket, block_size, num_heads, head_dim)
    v_active = v_blocked[active_idx]
    
    # Flatten: (bucket * block_size, num_heads, head_dim)
    k_flat = k_active.reshape(bucket * block_size, num_heads, head_dim)
    v_flat = v_active.reshape(bucket * block_size, num_heads, head_dim)
    
    # Step 4: Dense attention on reduced set
    scale = torch.sqrt(torch.tensor(float(head_dim), dtype=torch.float32))
    q_f32 = q.to(torch.float32)
    k_f32 = k_flat.to(torch.float32)
    v_f32 = v_flat.to(torch.float32)
    
    logits = torch.einsum('qhd,khd->qkh', q_f32, k_f32) / scale
    weights = F.softmax(logits, dim=1)
    output = torch.einsum('qkh,khd->qhd', weights, v_f32)
    
    return output, {
        'num_active': n_active,
        'bucket': bucket,
        'num_blocks': num_blocks,
        'eviction_rate': 1.0 - n_active / num_blocks,
    }
