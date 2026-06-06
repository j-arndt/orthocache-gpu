"""Tests for orthocache_gpu.pipeline (orthocache_forward) — GPU Edition.

Validates:
1. _dense_attention produces correct standard attention output.
2. orthocache_forward 'dense' mode returns correct shape and metadata.
3. Invalid mode raises ValueError.
4. 'compact' mode returns valid output and metadata.
5. Auto-tau computation works.
6. Metadata contains expected timing and statistics fields.
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from orthocache_gpu.pipeline import orthocache_forward, _dense_attention

# ── Constants ────────────────────────────────────────────────────────────────

BLOCK_SIZE = 512
NUM_HEADS = 4
HEAD_DIM = 64


def _make_tensors(num_blocks: int, seq_len_q: int = 4) -> tuple:
    """Create q, keys, values for testing."""
    torch.manual_seed(42)
    seq_len_k = num_blocks * BLOCK_SIZE
    q = torch.randn(seq_len_q, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    keys = torch.randn(seq_len_k, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    values = torch.randn(seq_len_k, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    return q, keys, values


# ── _dense_attention ────────────────────────────────────────────────────────


class TestDenseAttention:
    """Tests for the internal _dense_attention reference function."""

    def test_matches_manual_einsum(self):
        """_dense_attention should match a manual einsum-based computation."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        result = _dense_attention(q, keys, values, HEAD_DIM)

        # Manual reference
        scale = torch.sqrt(torch.tensor(float(HEAD_DIM)))
        logits = torch.einsum('qhd,khd->qkh', q, keys) / scale
        weights = F.softmax(logits, dim=1)
        expected = torch.einsum('qkh,khd->qhd', weights, values)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        """Output should be (seq_len_q, num_heads, head_dim)."""
        num_blocks = 2
        q, keys, values = _make_tensors(num_blocks, seq_len_q=8)
        result = _dense_attention(q, keys, values, HEAD_DIM)
        assert result.shape == (8, NUM_HEADS, HEAD_DIM)

    def test_single_token_query(self):
        """Single-token query should work correctly."""
        num_blocks = 2
        q, keys, values = _make_tensors(num_blocks, seq_len_q=1)
        result = _dense_attention(q, keys, values, HEAD_DIM)
        assert result.shape == (1, NUM_HEADS, HEAD_DIM)
        assert not torch.any(torch.isnan(result))


# ── orthocache_forward: dense mode ──────────────────────────────────────────


class TestForwardDenseMode:
    """Tests for orthocache_forward with mode='dense'."""

    def test_output_shape(self):
        """Dense mode output should match query shape."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        output, meta = orthocache_forward(
            q, keys, values, block_size=BLOCK_SIZE, mode='dense',
        )

        assert output.shape == (q.shape[0], NUM_HEADS, HEAD_DIM)

    def test_metadata_fields(self):
        """Dense mode metadata should have expected keys."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        _, meta = orthocache_forward(
            q, keys, values, block_size=BLOCK_SIZE, mode='dense',
        )

        assert meta['mode'] == 'dense'
        assert meta['eviction_rate'] == 0.0
        assert 'latency_ms' in meta
        assert meta['num_blocks'] == num_blocks
        assert meta['num_heads'] == NUM_HEADS
        assert meta['head_dim'] == HEAD_DIM

    def test_matches_dense_attention(self):
        """Dense mode should produce identical output to _dense_attention."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        output, _ = orthocache_forward(
            q, keys, values, block_size=BLOCK_SIZE, mode='dense',
        )
        expected = _dense_attention(q, keys, values, HEAD_DIM)

        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)


# ── orthocache_forward: invalid mode ────────────────────────────────────────


class TestForwardInvalidMode:
    """Tests for error handling on bad mode parameter."""

    def test_invalid_mode_raises_valueerror(self):
        """An unrecognized mode string should raise ValueError."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        with pytest.raises(ValueError, match="Unknown mode"):
            orthocache_forward(
                q, keys, values, block_size=BLOCK_SIZE, mode='invalid',
            )


# ── orthocache_forward: compact mode ────────────────────────────────────────


class TestForwardCompactMode:
    """Tests for orthocache_forward with mode='compact'."""

    def test_output_shape(self):
        """Compact mode output should match query shape."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        output, meta = orthocache_forward(
            q, keys, values, block_size=BLOCK_SIZE,
            zeta_max=100.0,  # High threshold = low eviction
            mode='compact',
        )

        assert output.shape == (q.shape[0], NUM_HEADS, HEAD_DIM)
        assert meta['mode'] == 'compact'
        assert 'compact_num_active' in meta

    def test_spectral_metadata_present(self):
        """Compact mode should include spectral analysis metadata."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        _, meta = orthocache_forward(
            q, keys, values, block_size=BLOCK_SIZE,
            zeta_max=100.0,
            mode='compact',
        )

        assert 'zeta_mean' in meta
        assert 'zeta_std' in meta
        assert 'eviction_rate' in meta
        assert 'spectral_ms' in meta
        assert 'tau' in meta

    def test_timing_metadata(self):
        """Compact mode should include timing fields."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        _, meta = orthocache_forward(
            q, keys, values, block_size=BLOCK_SIZE,
            zeta_max=100.0,
            mode='compact',
        )

        assert 'total_ms' in meta
        assert 'attention_ms' in meta
        assert meta['total_ms'] >= 0


# ── Auto-tau computation ────────────────────────────────────────────────────


class TestAutoTau:
    """Tests for automatic tau computation."""

    def test_auto_tau_is_computed(self):
        """When tau=None, it should be computed automatically."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        _, meta = orthocache_forward(
            q, keys, values, block_size=BLOCK_SIZE,
            tau=None,  # Auto-compute
            zeta_max=100.0,
            mode='compact',
        )

        assert meta['tau_auto'] is True
        assert 'tau' in meta
        assert isinstance(meta['tau'], float)

    def test_explicit_tau_used(self):
        """When tau is provided explicitly, it should be used as-is."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        _, meta = orthocache_forward(
            q, keys, values, block_size=BLOCK_SIZE,
            tau=0.5,
            zeta_max=100.0,
            mode='compact',
        )

        assert meta['tau_auto'] is False
        assert meta['tau'] == 0.5


# ── Crossover Dispatcher ────────────────────────────────────────────────────


class TestCrossoverDispatcher:
    """Tests for the Adaptive Crossover Dispatcher fallback."""

    def test_crossover_fallback_triggered(self):
        """When seq_len_k < crossover_threshold, fallback to dense attention."""
        # 1 block = 512 tokens
        num_blocks = 1
        q, keys, values = _make_tensors(num_blocks)

        # Set threshold high so 512 is below it
        output, meta = orthocache_forward(
            q, keys, values, block_size=BLOCK_SIZE,
            mode='compact', crossover_threshold=1024,
        )

        assert meta['crossover_fallback'] is True
        assert meta['actual_mode'] == 'dense'
        assert meta['mode'] == 'compact'  # original mode preserved
        assert meta['eviction_rate'] == 0.0

    def test_crossover_fallback_not_triggered(self):
        """When seq_len_k >= crossover_threshold, do not fallback."""
        num_blocks = 2  # 1024 tokens
        q, keys, values = _make_tensors(num_blocks)

        # Set threshold low so 1024 is above/equal to it
        output, meta = orthocache_forward(
            q, keys, values, block_size=BLOCK_SIZE,
            mode='compact', crossover_threshold=512,
        )

        assert meta['crossover_fallback'] is False
        assert meta['actual_mode'] == 'compact'
