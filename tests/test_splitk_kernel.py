"""Tests for Phase 7b: Split-K God Kernel with Interleaved Tile Assignment.

Tests the Split-K parallelization of the fused FWHT+ζ+attention kernel:
    - Split-K output matches V1 sequential kernel
    - Log-sum-exp reduction merge is exact
    - Multi-head single-launch correctness
    - Interleaved tile assignment balance
    - Edge cases: full eviction, zero eviction
    - Pipeline integration with mode='triton_fused'
"""

import pytest
import torch
import math

# Skip all tests if no CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    return torch.device('cuda')


@pytest.fixture
def head_dim():
    return 128


@pytest.fixture
def tile_size():
    return 64


def make_test_data(num_heads, seq_len, head_dim, device, dtype=torch.float32):
    """Create test Q, K, V tensors for multi-head Split-K tests."""
    torch.manual_seed(42)
    q = torch.randn(num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(num_heads, seq_len, head_dim, device=device, dtype=dtype)
    return q, k, v


def make_single_head_data(seq_len, head_dim, device, dtype=torch.float32):
    """Create test Q, K, V for single-head V1 tests."""
    torch.manual_seed(42)
    q = torch.randn(1, head_dim, device=device, dtype=dtype)
    k = torch.randn(seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(seq_len, head_dim, device=device, dtype=dtype)
    return q, k, v


# ============================================================================
# Test: Split-K matches V1 Sequential
# ============================================================================

class TestSplitKMatchesV1:
    """Split-K kernel should produce identical output to the V1 sequential kernel."""

    def test_splitk_matches_v1_short_seq(self, device, head_dim, tile_size):
        """At 1K tokens, Split-K and V1 should be numerically identical."""
        from orthocache_gpu.triton_kernels.fused_eviction import (
            fused_orthocache_attention,       # V1
            fused_orthocache_attention_v2,    # V2 Split-K
        )
        seq_len = 1024
        zeta_max = 5.0

        # V1: single-head
        q1, k1, v1 = make_single_head_data(seq_len, head_dim, device)
        out_v1, _ = fused_orthocache_attention(q1, k1, v1, zeta_max=zeta_max)

        # V2: wrap as multi-head (1 head), force num_splits=4
        q2 = q1.squeeze(0).unsqueeze(0)  # (1, head_dim)
        k2 = k1.unsqueeze(0)              # (1, seq_len, head_dim)
        v2 = v1.unsqueeze(0)              # (1, seq_len, head_dim)
        out_v2, meta_v2 = fused_orthocache_attention_v2(
            q2, k2, v2, zeta_max=zeta_max, num_splits=4
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            out_v1.flatten().float(), out_v2.flatten().float(), dim=0
        ).item()
        assert cos_sim > 0.999, f"Split-K vs V1 cos_sim = {cos_sim:.6f}"

    def test_splitk_matches_v1_medium_seq(self, device, head_dim, tile_size):
        """At 4K tokens with eviction."""
        from orthocache_gpu.triton_kernels.fused_eviction import (
            fused_orthocache_attention,
            fused_orthocache_attention_v2,
        )
        seq_len = 4096
        zeta_max = 5.0

        q1, k1, v1 = make_single_head_data(seq_len, head_dim, device)
        out_v1, _ = fused_orthocache_attention(q1, k1, v1, zeta_max=zeta_max)

        q2 = q1.squeeze(0).unsqueeze(0)
        k2 = k1.unsqueeze(0)
        v2 = v1.unsqueeze(0)
        out_v2, _ = fused_orthocache_attention_v2(
            q2, k2, v2, zeta_max=zeta_max, num_splits=8
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            out_v1.flatten().float(), out_v2.flatten().float(), dim=0
        ).item()
        assert cos_sim > 0.999, f"Split-K vs V1 cos_sim = {cos_sim:.6f}"


# ============================================================================
# Test: Log-Sum-Exp Reduction Correctness
# ============================================================================

class TestReduceCorrectness:
    """The online softmax merge must be mathematically exact."""

    def test_reduce_two_partials(self, device, head_dim):
        """Manual test: merge two partial states, compare to ground truth."""
        # Create two sets of "logits" that would produce known softmax states
        torch.manual_seed(123)

        # Simulate two tile groups with known attention patterns
        seq_len = 256  # 4 tiles
        q, k, v = make_single_head_data(seq_len, head_dim, device)

        from orthocache_gpu.triton_kernels.fused_eviction import (
            fused_orthocache_attention_v2,
        )

        # Run with 1 split (equivalent to V1) — ground truth
        q_mh = q.squeeze(0).unsqueeze(0)  # (1, head_dim)
        k_mh = k.unsqueeze(0)
        v_mh = v.unsqueeze(0)
        out_1split, _ = fused_orthocache_attention_v2(
            q_mh, k_mh, v_mh, zeta_max=100.0, num_splits=1  # no eviction
        )

        # Run with 2 splits — should match exactly
        out_2split, _ = fused_orthocache_attention_v2(
            q_mh, k_mh, v_mh, zeta_max=100.0, num_splits=2
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            out_1split.flatten().float(), out_2split.flatten().float(), dim=0
        ).item()
        assert cos_sim > 0.9999, f"1-split vs 2-split cos_sim = {cos_sim:.6f}"

    def test_reduce_many_splits(self, device, head_dim):
        """Run with max splits — should still match 1-split."""
        seq_len = 1024
        q_mh, k_mh, v_mh = make_test_data(1, seq_len, head_dim, device)

        from orthocache_gpu.triton_kernels.fused_eviction import (
            fused_orthocache_attention_v2,
        )

        out_1, _ = fused_orthocache_attention_v2(
            q_mh, k_mh, v_mh, zeta_max=100.0, num_splits=1
        )
        out_16, _ = fused_orthocache_attention_v2(
            q_mh, k_mh, v_mh, zeta_max=100.0, num_splits=16
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            out_1.flatten().float(), out_16.flatten().float(), dim=0
        ).item()
        assert cos_sim > 0.9999, f"1-split vs 16-split cos_sim = {cos_sim:.6f}"


# ============================================================================
# Test: Multi-Head Single Launch
# ============================================================================

class TestMultiHead:
    """All heads should be processed in a single kernel launch."""

    def test_multihead_4_heads(self, device, head_dim):
        """4 heads: output should have shape (4, head_dim)."""
        from orthocache_gpu.triton_kernels.fused_eviction import (
            fused_orthocache_attention_v2,
        )
        q, k, v = make_test_data(4, 1024, head_dim, device)
        out, meta = fused_orthocache_attention_v2(q, k, v, zeta_max=5.0)
        assert out.shape == (4, head_dim), f"Expected (4, {head_dim}), got {out.shape}"

    def test_multihead_per_head_correctness(self, device, head_dim):
        """Multi-head: each head's output should match V1 run with the same data."""
        from orthocache_gpu.triton_kernels.fused_eviction import (
            fused_orthocache_attention,
            fused_orthocache_attention_v2,
        )
        seq_len = 512
        num_heads = 4
        zeta_max = 5.0

        # Create multi-head data once
        torch.manual_seed(42)
        q_all = torch.randn(num_heads, head_dim, device=device)
        k_all = torch.randn(num_heads, seq_len, head_dim, device=device)
        v_all = torch.randn(num_heads, seq_len, head_dim, device=device)

        # V2: all heads in one launch
        out_v2, _ = fused_orthocache_attention_v2(
            q_all, k_all, v_all, zeta_max=zeta_max
        )

        # V1: per-head, using slices of the same data
        for h in range(num_heads):
            q_h = q_all[h].unsqueeze(0)     # (1, head_dim)
            k_h = k_all[h]                   # (seq_len, head_dim)
            v_h = v_all[h]                   # (seq_len, head_dim)
            out_v1, _ = fused_orthocache_attention(q_h, k_h, v_h, zeta_max=zeta_max)

            cos_sim = torch.nn.functional.cosine_similarity(
                out_v1.flatten().float(), out_v2[h].flatten().float(), dim=0
            ).item()
            assert cos_sim > 0.999, f"Head {h} cos_sim = {cos_sim:.6f}"

    def test_multihead_metadata(self, device, head_dim):
        """Metadata should include split info."""
        from orthocache_gpu.triton_kernels.fused_eviction import (
            fused_orthocache_attention_v2,
        )
        q, k, v = make_test_data(2, 512, head_dim, device)
        _, meta = fused_orthocache_attention_v2(q, k, v, zeta_max=5.0)
        assert 'num_splits' in meta
        assert 'tile_assignment' in meta
        assert meta['tile_assignment'] == 'interleaved'


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestSplitKEdgeCases:
    """Edge cases: full eviction, zero eviction, single tile."""

    def test_zero_eviction_matches_dense(self, device, head_dim):
        """With zeta_max=1e6, no tiles should be evicted → match dense."""
        from orthocache_gpu.triton_kernels.fused_eviction import (
            fused_orthocache_attention_v2,
        )
        seq_len = 512
        q, k, v = make_test_data(1, seq_len, head_dim, device)

        # Split-K with no eviction
        out_fused, _ = fused_orthocache_attention_v2(
            q, k, v, zeta_max=1e6, num_splits=4
        )

        # Dense attention reference
        scale = 1.0 / math.sqrt(head_dim)
        logits = (q.float() @ k[0].float().T) * scale  # (1, seq_len)
        weights = torch.softmax(logits, dim=-1)
        out_dense = weights @ v[0].float()  # (1, head_dim)

        cos_sim = torch.nn.functional.cosine_similarity(
            out_fused.flatten().float(), out_dense.flatten().float(), dim=0
        ).item()
        assert cos_sim > 0.999, f"Zero-eviction cos_sim = {cos_sim:.6f}"

    def test_full_eviction_no_crash(self, device, head_dim):
        """With zeta_max=0, all tiles evicted → should not crash."""
        from orthocache_gpu.triton_kernels.fused_eviction import (
            fused_orthocache_attention_v2,
        )
        q, k, v = make_test_data(1, 512, head_dim, device)
        out, meta = fused_orthocache_attention_v2(
            q, k, v, zeta_max=0.0, num_splits=4
        )
        assert out.shape == (1, head_dim)
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"

    def test_single_split_fallback(self, device, head_dim):
        """num_splits=1 should work (degrades to sequential)."""
        from orthocache_gpu.triton_kernels.fused_eviction import (
            fused_orthocache_attention_v2,
        )
        q, k, v = make_test_data(1, 512, head_dim, device)
        out, meta = fused_orthocache_attention_v2(
            q, k, v, zeta_max=5.0, num_splits=1
        )
        assert out.shape == (1, head_dim)
        assert meta['num_splits'] == 1


# ============================================================================
# Test: Interleaved Assignment Verification
# ============================================================================

class TestInterleavedAssignment:
    """Verify the tile-to-CTA mapping is interleaved, not contiguous."""

    def test_interleaved_pattern(self):
        """Verify the interleaved loop produces the right tile indices."""
        # Simulate: num_tiles=16, num_splits=4
        num_tiles = 16
        num_splits = 4

        for split_id in range(num_splits):
            tiles = list(range(split_id, num_tiles, num_splits))
            # Split 0 → [0, 4, 8, 12]
            # Split 1 → [1, 5, 9, 13]
            # Split 2 → [2, 6, 10, 14]
            # Split 3 → [3, 7, 11, 15]
            expected = [split_id + i * num_splits for i in range(num_tiles // num_splits)]
            assert tiles == expected, f"Split {split_id}: got {tiles}, expected {expected}"

    def test_all_tiles_covered(self):
        """Every tile should be assigned to exactly one CTA."""
        num_tiles = 512
        num_splits = 24

        all_tiles = set()
        for split_id in range(num_splits):
            tiles = set(range(split_id, num_tiles, num_splits))
            # No overlap with previously assigned tiles
            assert all_tiles.isdisjoint(tiles), f"Split {split_id} overlaps!"
            all_tiles.update(tiles)

        assert all_tiles == set(range(num_tiles)), "Not all tiles covered!"

    def test_load_balance(self):
        """Each CTA should get approximately the same number of tiles."""
        num_tiles = 512
        num_splits = 24

        counts = []
        for split_id in range(num_splits):
            tiles = list(range(split_id, num_tiles, num_splits))
            counts.append(len(tiles))

        min_count = min(counts)
        max_count = max(counts)
        # With interleaved, max difference should be ≤ 1
        assert max_count - min_count <= 1, (
            f"Load imbalance: min={min_count}, max={max_count}"
        )


# ============================================================================
# Test: Auto Num-Splits Selection
# ============================================================================

class TestAutoNumSplits:
    """Auto-selection should pick a reasonable split count."""

    def test_auto_selection_small_seq(self, device, head_dim):
        """Small sequence: should pick fewer splits."""
        from orthocache_gpu.triton_kernels.fused_eviction import (
            fused_orthocache_attention_v2,
        )
        q, k, v = make_test_data(1, 256, head_dim, device)  # 4 tiles
        _, meta = fused_orthocache_attention_v2(q, k, v, zeta_max=5.0)
        # 4 tiles → max 1 split (4//4 = 1)
        assert meta['num_splits'] >= 1
        assert meta['num_splits'] <= 4

    def test_auto_selection_large_seq(self, device, head_dim):
        """Large sequence: should use more splits (up to SM count)."""
        from orthocache_gpu.triton_kernels.fused_eviction import (
            fused_orthocache_attention_v2,
        )
        q, k, v = make_test_data(1, 8192, head_dim, device)  # 128 tiles
        _, meta = fused_orthocache_attention_v2(q, k, v, zeta_max=5.0)
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        assert meta['num_splits'] > 1
        assert meta['num_splits'] <= num_sms


# ============================================================================
# Test: Pipeline Integration
# ============================================================================

class TestPipelineSplitK:
    """Pipeline mode='triton_fused' should use Split-K v2."""

    def test_pipeline_triton_fused_output_shape(self, device, head_dim):
        """Pipeline should return correct shape."""
        from orthocache_gpu.pipeline import orthocache_forward

        seq_len = 512
        num_heads = 2
        q = torch.randn(1, num_heads, head_dim, device=device)
        k = torch.randn(seq_len, num_heads, head_dim, device=device)
        v = torch.randn(seq_len, num_heads, head_dim, device=device)

        out, meta = orthocache_forward(
            q, k, v, block_size=512, zeta_max=5.0, mode='triton_fused'
        )
        assert out.shape == (1, num_heads, head_dim)

    def test_pipeline_triton_fused_metadata(self, device, head_dim):
        """Pipeline metadata should include Split-K info."""
        from orthocache_gpu.pipeline import orthocache_forward

        seq_len = 512
        num_heads = 2
        q = torch.randn(1, num_heads, head_dim, device=device)
        k = torch.randn(seq_len, num_heads, head_dim, device=device)
        v = torch.randn(seq_len, num_heads, head_dim, device=device)

        _, meta = orthocache_forward(
            q, k, v, block_size=512, zeta_max=5.0, mode='triton_fused'
        )
        assert 'num_splits' in meta
        assert meta['tile_assignment'] == 'interleaved'
        assert 'latency_ms' in meta
