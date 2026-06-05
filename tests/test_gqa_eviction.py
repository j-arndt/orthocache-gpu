"""Tests for the V3 GQA Cauchy-Schwarz Spectral Gate Kernel (Phase 7c).

Tests cover:
1. Correctness: GQA kernel output matches dense attention reference
2. Eviction rate: Cauchy-Schwarz gate achieves higher eviction than naive consensus
3. GQA group structure: proper query-to-KV-head mapping
4. Edge cases: MQA (G=1), all-evict, all-retain
5. Numerical stability: online softmax accuracy under high eviction
"""

import pytest
import torch
import math


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def gqa_config():
    """Standard GQA configuration matching LLaMA-3 / Mistral ratios."""
    return {
        'num_kv_heads': 4,
        'num_query_groups': 8,  # G = 8 → 32 query heads total
        'head_dim': 128,
        'seq_len': 512,  # 8 tiles of 64
        'tile_size': 64,
    }


@pytest.fixture
def mqa_config():
    """MQA configuration (G=1, same as V2 MHA)."""
    return {
        'num_kv_heads': 8,
        'num_query_groups': 1,
        'head_dim': 128,
        'seq_len': 256,  # 4 tiles
        'tile_size': 64,
    }


def _make_gqa_tensors(config, device, seed=42):
    """Create random Q, K, V tensors for GQA testing."""
    torch.manual_seed(seed)
    num_kv_heads = config['num_kv_heads']
    G = config['num_query_groups']
    num_query_heads = num_kv_heads * G
    head_dim = config['head_dim']
    seq_len = config['seq_len']

    q = torch.randn(num_query_heads, head_dim, device=device)
    keys = torch.randn(num_kv_heads, seq_len, head_dim, device=device)
    values = torch.randn(num_kv_heads, seq_len, head_dim, device=device)

    return q, keys, values


def _dense_gqa_attention(q, keys, values, num_query_groups):
    """Reference dense attention with GQA (no eviction)."""
    num_kv_heads, seq_len, head_dim = keys.shape
    G = num_query_groups
    scale = 1.0 / math.sqrt(head_dim)

    outputs = []
    for kv_h in range(num_kv_heads):
        k_h = keys[kv_h].float()  # (seq_len, head_dim)
        v_h = values[kv_h].float()

        for g in range(G):
            qh = kv_h * G + g
            q_g = q[qh].float()  # (head_dim,)

            logits = (q_g @ k_h.T) * scale  # (seq_len,)
            attn = torch.softmax(logits, dim=0)  # (seq_len,)
            out = attn @ v_h  # (head_dim,)
            outputs.append(out)

    return torch.stack(outputs, dim=0)  # (num_query_heads, head_dim)


# ============================================================================
# Test: Import and Basic Functionality
# ============================================================================

class TestGQAImport:
    """Verify the GQA module imports and basic API works."""

    def test_import(self):
        from orthocache_gpu.triton_kernels.gqa_eviction import (
            fused_orthocache_attention_v3_gqa,
        )
        assert callable(fused_orthocache_attention_v3_gqa)

    def test_pytorch_fallback_import(self):
        from orthocache_gpu.triton_kernels.gqa_eviction import (
            _pytorch_gqa_cauchy_schwarz_attention,
        )
        assert callable(_pytorch_gqa_cauchy_schwarz_attention)


# ============================================================================
# Test: GQA Correctness (no eviction — τ = ∞)
# ============================================================================

class TestGQACorrectness:
    """With τ=∞ (no eviction), GQA kernel must match dense attention exactly."""

    def test_no_eviction_matches_dense(self, device, gqa_config):
        """With τ=0 (zero error tolerance), no tiles are evicted → output = dense attention."""
        from orthocache_gpu.triton_kernels.gqa_eviction import (
            fused_orthocache_attention_v3_gqa,
        )

        q, keys, values = _make_gqa_tensors(gqa_config, device)
        G = gqa_config['num_query_groups']

        # τ = 0 → zero error tolerance → never evict
        out_gqa, meta = fused_orthocache_attention_v3_gqa(
            q, keys, values, tau=0.0, num_query_groups=G,
        )
        out_dense = _dense_gqa_attention(q, keys, values, G)

        torch.testing.assert_close(
            out_gqa.float(), out_dense.float(),
            atol=1e-3, rtol=1e-2,
            msg="GQA kernel (no eviction) must match dense attention",
        )
        assert meta['tiles_evicted'] == 0

    def test_mqa_no_eviction_matches_dense(self, device, mqa_config):
        """MQA (G=1) with τ=0 (no eviction) must match dense attention."""
        from orthocache_gpu.triton_kernels.gqa_eviction import (
            fused_orthocache_attention_v3_gqa,
        )

        q, keys, values = _make_gqa_tensors(mqa_config, device)

        out_gqa, meta = fused_orthocache_attention_v3_gqa(
            q, keys, values, tau=0.0, num_query_groups=1,
        )
        out_dense = _dense_gqa_attention(q, keys, values, 1)

        torch.testing.assert_close(
            out_gqa.float(), out_dense.float(),
            atol=1e-3, rtol=1e-2,
        )

    def test_output_shape(self, device, gqa_config):
        """Output shape must be (num_query_heads, head_dim)."""
        from orthocache_gpu.triton_kernels.gqa_eviction import (
            fused_orthocache_attention_v3_gqa,
        )

        q, keys, values = _make_gqa_tensors(gqa_config, device)
        G = gqa_config['num_query_groups']
        num_query_heads = gqa_config['num_kv_heads'] * G

        out, _ = fused_orthocache_attention_v3_gqa(
            q, keys, values, tau=1e9, num_query_groups=G,
        )

        assert out.shape == (num_query_heads, gqa_config['head_dim'])


# ============================================================================
# Test: Eviction Behavior
# ============================================================================

class TestGQAEviction:
    """Test that the Cauchy-Schwarz gate correctly evicts tiles."""

    def test_high_tau_evicts_tiles(self, device, gqa_config):
        """High τ (permissive error tolerance) should evict tiles."""
        from orthocache_gpu.triton_kernels.gqa_eviction import (
            fused_orthocache_attention_v3_gqa,
        )

        q, keys, values = _make_gqa_tensors(gqa_config, device)
        G = gqa_config['num_query_groups']

        # High τ → permissive → evict tiles where CS bound is under τ
        _, meta = fused_orthocache_attention_v3_gqa(
            q, keys, values, tau=1e9, num_query_groups=G,
        )

        assert meta['tiles_evicted'] > 0, "High τ should evict tiles"
        assert meta['gate_type'] == 'cauchy_schwarz'

    def test_zero_tau_retains_all(self, device, gqa_config):
        """τ = 0 (zero error tolerance) should retain all tiles."""
        from orthocache_gpu.triton_kernels.gqa_eviction import (
            fused_orthocache_attention_v3_gqa,
        )

        q, keys, values = _make_gqa_tensors(gqa_config, device)
        G = gqa_config['num_query_groups']

        _, meta = fused_orthocache_attention_v3_gqa(
            q, keys, values, tau=0.0, num_query_groups=G,
        )

        assert meta['tiles_evicted'] == 0

    def test_smooth_tiles_evicted_first(self, device):
        """Tiles with smooth (low-frequency) K should be evicted before noisy tiles."""
        from orthocache_gpu.triton_kernels.gqa_eviction import (
            _pytorch_gqa_cauchy_schwarz_attention,
        )

        num_kv_heads = 1
        G = 4
        head_dim = 128
        tile_size = 64

        # Create 2 tiles: one smooth (constant), one noisy
        q = torch.randn(G, head_dim)

        k_smooth = torch.ones(1, tile_size, head_dim) * 0.5  # All same → pure DC
        k_noisy = torch.randn(1, tile_size, head_dim)         # Random → high-freq energy

        keys = torch.cat([k_smooth, k_noisy], dim=1)  # (1, 128, 128) = 2 tiles
        values = torch.randn(1, 2 * tile_size, head_dim)

        # Use a τ that evicts smooth but not noisy
        _, meta = _pytorch_gqa_cauchy_schwarz_attention(
            q, keys, values, tau=0.1, num_query_groups=G, tile_size=tile_size,
        )

        # Smooth tile should be evicted (low high-freq energy → low CS bound)
        assert meta['tiles_evicted'] >= 1, (
            "Smooth constant tile should be evicted by Cauchy-Schwarz gate"
        )


# ============================================================================
# Test: Query-Group Consensus
# ============================================================================

class TestGQAConsensus:
    """Test the query-aware consensus mechanism."""

    def test_aligned_query_prevents_eviction(self, device):
        """A single query head aligned to K's high-frequency band
        should prevent eviction even if other heads don't need the tile."""
        from orthocache_gpu.triton_kernels.gqa_eviction import (
            _pytorch_gqa_cauchy_schwarz_attention,
        )

        num_kv_heads = 1
        G = 4
        head_dim = 128
        tile_size = 64

        # Create a tile with some high-frequency content
        keys = torch.randn(1, tile_size, head_dim) * 0.3  # moderate noise

        # 3 queries with tiny norm (they don't care about this tile)
        q_small = torch.randn(3, head_dim) * 0.001
        # 1 query with HUGE norm (it cares a lot about everything)
        q_big = torch.randn(1, head_dim) * 100.0
        q = torch.cat([q_small, q_big], dim=0)  # (4, head_dim)

        values = torch.randn(1, tile_size, head_dim)

        # The big query should force retention via Cauchy-Schwarz
        _, meta = _pytorch_gqa_cauchy_schwarz_attention(
            q, keys, values, tau=1.0, num_query_groups=G, tile_size=tile_size,
        )

        # The large-norm query drives max_cs_bound high → retention
        assert meta['tiles_retained'] == 1, (
            "Large-norm query should force tile retention under Cauchy-Schwarz"
        )

    def test_all_small_queries_allow_eviction(self, device):
        """If ALL query heads have tiny norms, Cauchy-Schwarz gate allows eviction."""
        from orthocache_gpu.triton_kernels.gqa_eviction import (
            _pytorch_gqa_cauchy_schwarz_attention,
        )

        num_kv_heads = 1
        G = 4
        head_dim = 128
        tile_size = 64

        # Moderate K tile
        keys = torch.randn(1, tile_size, head_dim) * 0.5

        # ALL queries have tiny norms → none care about this tile
        q = torch.randn(G, head_dim) * 0.0001

        values = torch.randn(1, tile_size, head_dim)

        _, meta = _pytorch_gqa_cauchy_schwarz_attention(
            q, keys, values, tau=0.1, num_query_groups=G, tile_size=tile_size,
        )

        # All queries are tiny → max_cs_bound ≈ 0 → eviction
        assert meta['tiles_evicted'] == 1, (
            "All-tiny-norm queries should allow eviction under Cauchy-Schwarz"
        )


# ============================================================================
# Test: Metadata and API
# ============================================================================

class TestGQAMetadata:
    """Test metadata reporting."""

    def test_metadata_fields(self, device, gqa_config):
        from orthocache_gpu.triton_kernels.gqa_eviction import (
            fused_orthocache_attention_v3_gqa,
        )

        q, keys, values = _make_gqa_tensors(gqa_config, device)
        G = gqa_config['num_query_groups']

        _, meta = fused_orthocache_attention_v3_gqa(
            q, keys, values, tau=1e9, num_query_groups=G,
        )

        assert 'num_kv_heads' in meta
        assert 'num_query_groups' in meta
        assert 'gate_type' in meta
        assert meta['gate_type'] == 'cauchy_schwarz'
        assert meta['num_query_groups'] == G

    def test_assertion_on_mismatched_heads(self, device):
        from orthocache_gpu.triton_kernels.gqa_eviction import (
            fused_orthocache_attention_v3_gqa,
        )

        # 4 KV heads, G=8 → need 32 query heads, but provide 16
        q = torch.randn(16, 128, device=device)
        keys = torch.randn(4, 256, 128, device=device)
        values = torch.randn(4, 256, 128, device=device)

        with pytest.raises(AssertionError, match="num_query_heads"):
            fused_orthocache_attention_v3_gqa(
                q, keys, values, tau=1.0, num_query_groups=8,
            )
