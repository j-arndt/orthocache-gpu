"""Unit tests for FP8 quantization and robustness assertions in OrthoCache."""

import pytest
import torch
import numpy as np
from typing import Tuple

from orthocache_gpu.triton_kernels.fused_eviction import (
    fused_orthocache_attention,
    fused_orthocache_attention_v2,
)
from orthocache_gpu.triton_kernels.gqa_eviction import (
    fused_orthocache_attention_v3_gqa,
)

# ── Quantization Helper ──
def quantize_to_fp8(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Helper to quantize a tensor to float8_e4m3fn and return its scale factor."""
    max_val = tensor.abs().max().item()
    scale = max_val / 448.0 if max_val > 0 else 1.0
    # On CPU, casting to float8_e4m3fn is supported in newer PyTorch versions.
    # We use a fallback to float32 representation if the platform lacks full float8 support.
    try:
        quantized = (tensor / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    except (TypeError, RuntimeError):
        # Fallback if float8 dtype isn't fully supported on the platform
        quantized = (tensor / scale).clamp(-448.0, 448.0)
    return quantized, scale


# ── Correctness Tests ──
@pytest.mark.parametrize("device", ["cuda"] if torch.cuda.is_available() else ["cpu"])
def test_fused_attention_fp8_correctness(device):
    """Verify single-head attention with FP8 keys matches unquantized float32/float16 baseline."""
    torch.manual_seed(42)
    device = torch.device(device)
    
    head_dim = 128
    seq_len = 256
    tile_size = 64
    zeta_max = 999.0  # Keep all tiles
    
    q = torch.randn(1, head_dim, device=device, dtype=torch.float32)
    keys_f32 = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
    values = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
    
    # Quantize keys
    keys_fp8, k_scale = quantize_to_fp8(keys_f32)
    
    # Run unquantized baseline
    out_ref, _ = fused_orthocache_attention(
        q, keys_f32, values, zeta_max=zeta_max, tile_size=tile_size
    )
    
    # Run FP8 quantized version
    out_fp8, _ = fused_orthocache_attention(
        q, keys_fp8, values, zeta_max=zeta_max, tile_size=tile_size, k_scale=k_scale
    )
    
    # Due to float8 quantization error, we check for a reasonable cosine similarity or absolute tolerance.
    cos_sim = torch.nn.functional.cosine_similarity(out_ref, out_fp8, dim=-1).mean().item()
    assert cos_sim > 0.99, f"Cosine similarity between FP8 and reference is too low: {cos_sim}"


@pytest.mark.parametrize("device", ["cuda"] if torch.cuda.is_available() else ["cpu"])
def test_fused_attention_v2_fp8_correctness(device):
    """Verify multi-head attention v2 with FP8 keys matches baseline."""
    torch.manual_seed(42)
    device = torch.device(device)
    
    num_heads = 4
    head_dim = 64
    seq_len = 128
    tile_size = 64
    zeta_max = 999.0
    
    q = torch.randn(num_heads, head_dim, device=device, dtype=torch.float32)
    keys_f32 = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    values = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    
    keys_fp8, k_scale = quantize_to_fp8(keys_f32)
    
    out_ref, _ = fused_orthocache_attention_v2(
        q, keys_f32, values, zeta_max=zeta_max, tile_size=tile_size
    )
    
    out_fp8, _ = fused_orthocache_attention_v2(
        q, keys_fp8, values, zeta_max=zeta_max, tile_size=tile_size, k_scale=k_scale
    )
    
    cos_sim = torch.nn.functional.cosine_similarity(out_ref, out_fp8, dim=-1).mean().item()
    assert cos_sim > 0.99, f"Cosine similarity between FP8 and reference is too low: {cos_sim}"


@pytest.mark.parametrize("device", ["cuda"] if torch.cuda.is_available() else ["cpu"])
def test_fused_attention_v3_gqa_fp8_correctness(device):
    """Verify GQA attention v3 with FP8 keys matches baseline."""
    torch.manual_seed(42)
    device = torch.device(device)
    
    num_kv_heads = 2
    num_query_groups = 4
    num_query_heads = num_kv_heads * num_query_groups
    head_dim = 128
    seq_len = 128
    tile_size = 64
    tau = 999.0
    
    q = torch.randn(num_query_heads, head_dim, device=device, dtype=torch.float32)
    keys_f32 = torch.randn(num_kv_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    values = torch.randn(num_kv_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    
    keys_fp8, k_scale = quantize_to_fp8(keys_f32)
    
    out_ref, _ = fused_orthocache_attention_v3_gqa(
        q, keys_f32, values, tau=tau, num_query_groups=num_query_groups, tile_size=tile_size
    )
    
    out_fp8, _ = fused_orthocache_attention_v3_gqa(
        q, keys_fp8, values, tau=tau, num_query_groups=num_query_groups, tile_size=tile_size, k_scale=k_scale
    )
    
    cos_sim = torch.nn.functional.cosine_similarity(out_ref, out_fp8, dim=-1).mean().item()
    assert cos_sim > 0.99, f"Cosine similarity between FP8 and reference is too low: {cos_sim}"


# ── Robustness Assertion Tests ──
def test_robustness_assertions():
    """Verify that improper dimensions, alignments, or split counts correctly raise assertions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    q = torch.randn(1, 128, device=device)
    keys = torch.randn(256, 128, device=device)
    values = torch.randn(256, 128, device=device)
    
    # 1. Non-power-of-2 head_dim
    q_bad_dim = torch.randn(1, 100, device=device)
    keys_bad_dim = torch.randn(256, 100, device=device)
    values_bad_dim = torch.randn(256, 100, device=device)
    with pytest.raises(AssertionError, match="head_dim.*must be a power of 2"):
        fused_orthocache_attention(q_bad_dim, keys_bad_dim, values_bad_dim, zeta_max=5.0)
        
    # 2. Non-power-of-2 tile_size
    with pytest.raises(AssertionError, match="tile_size.*must be a power of 2"):
        fused_orthocache_attention(q, keys, values, zeta_max=5.0, tile_size=50)
        
    # 3. Sequence length not divisible by tile_size
    keys_unaligned = torch.randn(250, 128, device=device)
    values_unaligned = torch.randn(250, 128, device=device)
    with pytest.raises(AssertionError, match="not divisible by tile_size"):
        fused_orthocache_attention(q, keys_unaligned, values_unaligned, zeta_max=5.0, tile_size=64)
        
    # 4. Invalid num_splits <= 0 in V2
    q_v2 = torch.randn(2, 128, device=device)
    keys_v2 = torch.randn(2, 256, 128, device=device)
    values_v2 = torch.randn(2, 256, 128, device=device)
    with pytest.raises(AssertionError, match="num_splits.*greater than 0"):
        fused_orthocache_attention_v2(q_v2, keys_v2, values_v2, zeta_max=5.0, num_splits=0)
