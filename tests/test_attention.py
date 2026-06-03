"""Tests for orthocache_gpu.lean_attention — lean bucketed attention (GPU Edition).

Validates:
1. lean_bucketed_attention matches dense attention at 0% eviction
2. Sparse attention output differs from dense (eviction has effect)
3. Various eviction rates: 0%, 50%, 75%
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from orthocache_gpu.lean_attention import lean_bucketed_attention


# ── Dense Reference ─────────────────────────────────────────────────────────


def _dense_attention(q, keys, values, head_dim):
    """Reference dense multi-head attention in float32."""
    scale = torch.sqrt(torch.tensor(float(head_dim)))
    logits = torch.einsum('qhd,khd->qkh', q, keys) / scale
    weights = F.softmax(logits, dim=1)
    return torch.einsum('qkh,khd->qhd', weights, values)


# ── Lean Bucketed Attention ──────────────────────────────────────────────────


class TestLeanBucketedAttention:
    """Tests for lean_bucketed_attention."""

    def test_matches_dense_at_zero_eviction(self):
        """With all blocks retained, lean_bucketed_attention should match dense attention."""
        torch.manual_seed(42)
        seq_len_q = 8
        seq_len_k = 1024
        num_heads = 2
        head_dim = 64
        block_size = 512
        num_blocks = seq_len_k // block_size

        q = torch.randn(seq_len_q, num_heads, head_dim, dtype=torch.float32)
        keys = torch.randn(seq_len_k, num_heads, head_dim, dtype=torch.float32)
        values = torch.randn(seq_len_k, num_heads, head_dim, dtype=torch.float32)

        # All blocks retained
        block_mask = torch.ones(num_blocks, num_heads, dtype=torch.bool)

        output, meta = lean_bucketed_attention(q, keys, values, block_mask, block_size)
        dense_out = _dense_attention(q, keys, values, head_dim)

        assert output.shape == (seq_len_q, num_heads, head_dim)

        torch.testing.assert_close(
            output.to(torch.float32), dense_out, atol=1e-4, rtol=1e-4,
            msg="Lean bucketed attention should match dense at 0% eviction",
        )

    def test_sparse_differs_from_dense(self):
        """With some blocks evicted, output should differ from dense."""
        torch.manual_seed(42)
        seq_len_q = 8
        seq_len_k = 1024
        num_heads = 2
        head_dim = 64
        block_size = 512
        num_blocks = seq_len_k // block_size

        q = torch.randn(seq_len_q, num_heads, head_dim, dtype=torch.float32)
        keys = torch.randn(seq_len_k, num_heads, head_dim, dtype=torch.float32)
        values = torch.randn(seq_len_k, num_heads, head_dim, dtype=torch.float32)

        # Evict block 1
        block_mask = torch.tensor([[True, True], [False, False]], dtype=torch.bool)

        sparse_out, _ = lean_bucketed_attention(q, keys, values, block_mask, block_size)
        dense_out = _dense_attention(q, keys, values, head_dim)

        diff = torch.max(torch.abs(dense_out - sparse_out.to(torch.float32))).item()
        assert diff > 0.0, "Sparse and dense should differ when blocks are evicted"


class TestEvictionRates:
    """Test lean_bucketed_attention at various eviction rates."""

    @pytest.mark.parametrize("eviction_label,mask_pattern", [
        ("0pct", [True, True, True, True]),
        ("50pct", [True, False, True, False]),
        ("75pct", [True, False, False, False]),
    ])
    def test_eviction_rate(self, eviction_label, mask_pattern):
        """Test output validity at different eviction rates."""
        torch.manual_seed(42)
        num_blocks = 4
        seq_len_k = num_blocks * 512
        seq_len_q = 4
        num_heads = 2
        head_dim = 64
        block_size = 512

        q = torch.randn(seq_len_q, num_heads, head_dim, dtype=torch.float32)
        keys = torch.randn(seq_len_k, num_heads, head_dim, dtype=torch.float32)
        values = torch.randn(seq_len_k, num_heads, head_dim, dtype=torch.float32)

        # Build block mask (broadcast across heads)
        mask_1d = torch.tensor(mask_pattern, dtype=torch.bool)
        block_mask = mask_1d[:, None].expand(num_blocks, num_heads).contiguous()

        output, meta = lean_bucketed_attention(q, keys, values, block_mask, block_size)

        # Output shape should always be correct
        assert output.shape == (seq_len_q, num_heads, head_dim)

        # Output should not contain NaN
        assert not torch.any(torch.isnan(output)), f"NaN in output at {eviction_label}"

        # Metadata should report the correct number of active blocks
        n_active = sum(mask_pattern)
        assert meta['num_active'] == n_active

    def test_full_eviction_returns_zeros(self):
        """At 100% eviction, output should be all zeros."""
        torch.manual_seed(42)
        num_blocks = 4
        seq_len_k = num_blocks * 512
        seq_len_q = 4
        num_heads = 2
        head_dim = 64
        block_size = 512

        q = torch.randn(seq_len_q, num_heads, head_dim, dtype=torch.float32)
        keys = torch.randn(seq_len_k, num_heads, head_dim, dtype=torch.float32)
        values = torch.randn(seq_len_k, num_heads, head_dim, dtype=torch.float32)

        block_mask = torch.zeros(num_blocks, num_heads, dtype=torch.bool)

        output, meta = lean_bucketed_attention(q, keys, values, block_mask, block_size)

        torch.testing.assert_close(
            output, torch.zeros_like(q), atol=1e-9, rtol=1e-9,
        )
        assert meta['num_active'] == 0
