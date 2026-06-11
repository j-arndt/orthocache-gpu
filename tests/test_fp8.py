import pytest
import torch
import numpy as np

from orthocache_gpu.triton_kernels.fused_eviction import (
    fused_orthocache_attention,
    fused_orthocache_attention_v2,
)
from orthocache_gpu.triton_kernels.gqa_eviction import (
    fused_orthocache_attention_v3_gqa,
)

# Skip tests if float8_e4m3fn is not supported by this PyTorch version
try:
    fp8_type = torch.float8_e4m3fn
    HAS_FP8 = True
except AttributeError:
    HAS_FP8 = False


def quantize_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Quantize a float tensor to float8_e4m3fn with a scalar scale factor."""
    # float8_e4m3fn has a max representable value of 448.0
    finfo = torch.finfo(torch.float8_e4m3fn)
    max_val = finfo.max
    
    # Calculate scale factor
    abs_max = x.abs().max().item()
    scale = max(1e-8, abs_max / max_val)
    
    # Scale and cast
    x_scaled = (x / scale).clamp(min=-max_val, max=max_val)
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    
    return x_fp8, scale


@pytest.mark.skipif(not HAS_FP8, reason="float8_e4m3fn not supported by PyTorch version")
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_fp8_single_head_attention(device):
    """Test FP8 quantization on the single-head attention kernel."""
    torch.manual_seed(42)
    
    tile_size = 64
    num_tiles = 4
    seq_len = num_tiles * tile_size
    head_dim = 64
    zeta_max = 5.0
    
    # Create random inputs
    q = torch.randn(1, head_dim, device=device, dtype=torch.float32)
    keys = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
    values = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
    
    # Get baseline reference output (unquantized)
    out_ref, meta_ref = fused_orthocache_attention(
        q, keys, values, zeta_max=zeta_max, tile_size=tile_size, return_mask=True
    )
    
    # Quantize keys to FP8
    keys_fp8, k_scale = quantize_to_fp8(keys)
    
    # Run FP8 attention
    out_fp8, meta_fp8 = fused_orthocache_attention(
        q, keys_fp8, values, zeta_max=zeta_max, tile_size=tile_size, return_mask=True, k_scale=k_scale
    )
    
    # Verify outputs are close (allow small absolute tolerance for quantization noise)
    # The max absolute difference should be small
    diff = torch.abs(out_ref - out_fp8).max().item()
    assert diff < 0.15, f"FP8 attention output deviated too much from baseline. Max diff: {diff}"
    
    # Verify metadata masks match or are highly similar
    mask_ref = meta_ref['eviction_mask']
    mask_fp8 = meta_fp8['eviction_mask']
    assert torch.equal(mask_ref, mask_fp8), "Eviction masks between float32 and FP8 mismatched"


@pytest.mark.skipif(not HAS_FP8, reason="float8_e4m3fn not supported by PyTorch version")
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_fp8_multi_head_attention_v2(device):
    """Test FP8 quantization on the multi-head Split-K attention kernel (v2)."""
    torch.manual_seed(123)
    
    tile_size = 64
    num_tiles = 4
    seq_len = num_tiles * tile_size
    head_dim = 128
    num_heads = 4
    zeta_max = 5.0
    
    q = torch.randn(num_heads, head_dim, device=device, dtype=torch.float32)
    keys = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    values = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    
    # Reference
    out_ref, _ = fused_orthocache_attention_v2(
        q, keys, values, zeta_max=zeta_max, tile_size=tile_size
    )
    
    # Quantize keys to FP8 per-tensor
    keys_fp8, k_scale = quantize_to_fp8(keys)
    
    # FP8
    out_fp8, _ = fused_orthocache_attention_v2(
        q, keys_fp8, values, zeta_max=zeta_max, tile_size=tile_size, k_scale=k_scale
    )
    
    diff = torch.abs(out_ref - out_fp8).max().item()
    assert diff < 0.15, f"FP8 multi-head attention output deviated too much. Max diff: {diff}"


@pytest.mark.skipif(not HAS_FP8, reason="float8_e4m3fn not supported by PyTorch version")
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_fp8_gqa_attention_v3(device):
    """Test FP8 quantization on GQA consensus attention kernel (v3)."""
    torch.manual_seed(456)
    
    tile_size = 64
    num_tiles = 4
    seq_len = num_tiles * tile_size
    head_dim = 64
    num_kv_heads = 2
    num_query_groups = 4  # G = 4
    num_query_heads = num_kv_heads * num_query_groups
    tau = 2.0
    
    q = torch.randn(num_query_heads, head_dim, device=device, dtype=torch.float32)
    keys = torch.randn(num_kv_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    values = torch.randn(num_kv_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    
    # Reference
    out_ref, _ = fused_orthocache_attention_v3_gqa(
        q, keys, values, tau=tau, num_query_groups=num_query_groups, tile_size=tile_size
    )
    
    # Quantize keys to FP8
    keys_fp8, k_scale = quantize_to_fp8(keys)
    
    # FP8
    out_fp8, _ = fused_orthocache_attention_v3_gqa(
        q, keys_fp8, values, tau=tau, num_query_groups=num_query_groups, tile_size=tile_size, k_scale=k_scale
    )
    
    diff = torch.abs(out_ref - out_fp8).max().item()
    assert diff < 0.15, f"FP8 GQA attention output deviated too much. Max diff: {diff}"


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_robustness_assertions(device):
    """Verify that Red Team Audit power-of-2 and validity assertions raise Errors."""
    q = torch.randn(1, 64, device=device)
    keys = torch.randn(128, 64, device=device)
    values = torch.randn(128, 64, device=device)
    
    # 1. Non-power-of-2 head_dim should fail
    q_bad = torch.randn(1, 63, device=device)
    keys_bad = torch.randn(128, 63, device=device)
    values_bad = torch.randn(128, 63, device=device)
    with pytest.raises(AssertionError, match="head_dim.*must be a power of 2"):
        fused_orthocache_attention(q_bad, keys_bad, values_bad, zeta_max=5.0)
        
    # 2. Non-power-of-2 tile_size should fail
    keys_div = torch.randn(126, 64, device=device)
    values_div = torch.randn(126, 64, device=device)
    with pytest.raises(AssertionError, match="tile_size.*must be a power of 2"):
        fused_orthocache_attention(q, keys_div, values_div, zeta_max=5.0, tile_size=63)
        
    # 3. Invalid num_splits should fail
    q_v2 = torch.randn(4, 64, device=device)
    keys_v2 = torch.randn(4, 128, 64, device=device)
    values_v2 = torch.randn(4, 128, 64, device=device)
    with pytest.raises(AssertionError, match="num_splits.*must be positive"):
        fused_orthocache_attention_v2(q_v2, keys_v2, values_v2, zeta_max=5.0, num_splits=-1)
