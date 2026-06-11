"""Unit tests for PagedAttention support in OrthoCache."""

import pytest
import torch
import numpy as np
from typing import Tuple

from orthocache_gpu.triton_kernels.fused_eviction import (
    fused_orthocache_attention_v2,
)
from orthocache_gpu.triton_kernels.paged_eviction import (
    fused_orthocache_attention_paged,
)

# ── Quantization Helper ──
def quantize_to_fp8(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Helper to quantize a tensor to float8_e4m3fn and return its scale factor."""
    max_val = tensor.abs().max().item()
    scale = max_val / 448.0 if max_val > 0 else 1.0
    try:
        quantized = (tensor / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    except (TypeError, RuntimeError):
        quantized = (tensor / scale).clamp(-448.0, 448.0)
    return quantized, scale


@pytest.mark.parametrize("device", ["cuda"] if torch.cuda.is_available() else ["cpu"])
def test_paged_attention_correctness(device):
    """Verify paged attention matches contiguous attention output exactly."""
    torch.manual_seed(42)
    device = torch.device(device)
    
    num_seqs = 2
    num_heads = 4
    head_dim = 64
    tile_size = 64
    block_size = 32  # 2 blocks per tile
    num_tiles = 4
    seq_len = num_tiles * tile_size  # 256 tokens, 8 blocks per sequence
    max_blocks_per_seq = seq_len // block_size  # 8
    
    # 1. Generate random contiguous query, keys, and values
    q = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=torch.float32)
    keys_contig = torch.randn(num_seqs, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    values_contig = torch.randn(num_seqs, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    
    # 2. Build block tables and non-contiguous physical caches
    # Allocate more physical blocks than needed to simulate real memory pool fragmentation
    total_physical_blocks = num_seqs * max_blocks_per_seq + 5
    # Shuffle block assignment to ensure they are physically non-contiguous
    shuffled_blocks = torch.randperm(total_physical_blocks)
    
    block_tables = torch.empty((num_seqs, max_blocks_per_seq), dtype=torch.int32, device=device)
    for s in range(num_seqs):
        for b in range(max_blocks_per_seq):
            block_tables[s, b] = shuffled_blocks[s * max_blocks_per_seq + b]
            
    # Cache shape: (num_physical_blocks, num_kv_heads, block_size, head_dim)
    k_cache = torch.zeros((total_physical_blocks, num_heads, block_size, head_dim), device=device, dtype=torch.float32)
    v_cache = torch.zeros((total_physical_blocks, num_heads, block_size, head_dim), device=device, dtype=torch.float32)
    
    # Populate cache
    for s in range(num_seqs):
        for b in range(max_blocks_per_seq):
            phys_block = block_tables[s, b].item()
            token_start = b * block_size
            token_end = token_start + block_size
            # Contiguous keys has shape (num_seqs, num_heads, seq_len, head_dim)
            k_cache[phys_block] = keys_contig[s, :, token_start:token_end]
            v_cache[phys_block] = values_contig[s, :, token_start:token_end]

            
    zeta_max = 999.0  # Keep all tiles
    
    # 3. Compute baseline contiguous attention (separately per sequence since v2 is seq-by-seq)
    ref_outputs = []
    for s in range(num_seqs):
        ref_out, _ = fused_orthocache_attention_v2(
            q[s], keys_contig[s], values_contig[s], zeta_max=zeta_max, tile_size=tile_size
        )
        ref_outputs.append(ref_out)
    out_ref = torch.stack(ref_outputs, dim=0)  # (num_seqs, num_heads, head_dim)
    
    # 4. Compute paged attention
    out_paged, _ = fused_orthocache_attention_paged(
        q, k_cache, v_cache, block_tables, zeta_max=zeta_max, tile_size=tile_size
    )
    
    # 5. Assert equality
    torch.testing.assert_close(out_ref, out_paged, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda"] if torch.cuda.is_available() else ["cpu"])
def test_paged_attention_fp8(device):
    """Verify paged attention with FP8 quantized cache works and is close to baseline."""
    torch.manual_seed(42)
    device = torch.device(device)
    
    num_seqs = 1
    num_heads = 2
    head_dim = 64
    tile_size = 64
    block_size = 64
    num_tiles = 2
    seq_len = num_tiles * tile_size
    max_blocks_per_seq = seq_len // block_size
    
    q = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=torch.float32)
    keys_contig = torch.randn(num_seqs, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    values_contig = torch.randn(num_seqs, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    
    block_tables = torch.arange(max_blocks_per_seq, dtype=torch.int32, device=device).unsqueeze(0)
    
    k_cache = torch.zeros((max_blocks_per_seq, num_heads, block_size, head_dim), device=device, dtype=torch.float32)
    v_cache = torch.zeros((max_blocks_per_seq, num_heads, block_size, head_dim), device=device, dtype=torch.float32)
    
    for b in range(max_blocks_per_seq):
        k_cache[b] = keys_contig[0, :, b*block_size:(b+1)*block_size]
        v_cache[b] = values_contig[0, :, b*block_size:(b+1)*block_size]

        
    # Quantize key cache
    k_cache_fp8, k_scale = quantize_to_fp8(k_cache)
    
    # Run float32 baseline
    out_ref, _ = fused_orthocache_attention_paged(
        q, k_cache, v_cache, block_tables, zeta_max=999.0, tile_size=tile_size
    )
    
    # Run FP8 quantized
    out_fp8, _ = fused_orthocache_attention_paged(
        q, k_cache_fp8, v_cache, block_tables, zeta_max=999.0, tile_size=tile_size, k_scale=k_scale
    )
    
    cos_sim = torch.nn.functional.cosine_similarity(out_ref, out_fp8, dim=-1).mean().item()
    assert cos_sim > 0.99, f"FP8 paged attention output has low similarity: {cos_sim}"


def test_paged_attention_robustness_assertions():
    """Verify size, power-of-2, and alignment checks in paged attention."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    q = torch.randn(1, 2, 64, device=device)
    k_cache = torch.randn(4, 2, 32, 64, device=device)
    v_cache = torch.randn(4, 2, 32, 64, device=device)
    block_tables = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    
    # 1. Invalid head_dim
    q_bad = torch.randn(1, 2, 63, device=device)
    k_bad = torch.randn(4, 2, 32, 63, device=device)
    v_bad = torch.randn(4, 2, 32, 63, device=device)
    with pytest.raises(AssertionError, match="head_dim.*power of 2"):
        fused_orthocache_attention_paged(q_bad, k_bad, v_bad, block_tables, zeta_max=5.0)
        
    # 2. Invalid block_size (not power of 2)
    k_bad_block = torch.randn(4, 2, 31, 64, device=device)
    v_bad_block = torch.randn(4, 2, 31, 64, device=device)
    with pytest.raises(AssertionError, match="block_size.*power of 2"):
        fused_orthocache_attention_paged(q, k_bad_block, v_bad_block, block_tables, zeta_max=5.0)
