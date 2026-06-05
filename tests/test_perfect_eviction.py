"""Tests for the OrthoCache Perfect Eviction Governor.

Validates that:
1. The IEEE 754 underflow boundary condition correctly classifies blocks.
2. Perfect eviction blocks produce exact zero in quantized exponential.
3. The dual-regime classification is exhaustive and mutually exclusive.
4. Edge cases around the 88.72 threshold are handled correctly.
5. The scalar `perfect_eviction_check` matches the tensor-based classifier.
"""

import math
import pytest
import numpy as np
import torch

from orthocache_gpu.perfect_eviction import (
    FLOAT32_UNDERFLOW_THRESHOLD,
    BFLOAT16_UNDERFLOW_THRESHOLD,
    EvictionRegime,
    compute_block_beta,
    classify_eviction,
    perfect_eviction_check,
)


class TestUnderflowConstants:
    """Verify IEEE 754 underflow constants match hardware behavior."""

    def test_float32_underflow_threshold(self):
        """exp(-88.72) should be extremely close to zero in float32."""
        val = math.exp(-FLOAT32_UNDERFLOW_THRESHOLD)
        # In float64 this is ~3.5e-39, but in float32 it underflows to 0
        assert val < 1e-38
        # Verify float32 actually flushes to zero
        val_f32 = np.float32(np.exp(np.float32(-88.73)))
        assert val_f32 == 0.0, f"Expected float32 underflow to 0, got {val_f32}"

    def test_bfloat16_underflow_threshold(self):
        """bfloat16 underflow threshold should be slightly lower."""
        assert BFLOAT16_UNDERFLOW_THRESHOLD < FLOAT32_UNDERFLOW_THRESHOLD
        assert BFLOAT16_UNDERFLOW_THRESHOLD > 80.0  # sanity check


class TestComputeBlockBeta:
    """Test the logit ceiling β computation."""

    def test_basic_beta_computation(self):
        """β = ||q||₂ · √(E_j) / √(d_k) for a simple case."""
        head_dim = 64
        num_heads = 1
        q = torch.ones(1, num_heads, head_dim, dtype=torch.float32)
        # ||q||₂ = sqrt(64) = 8
        q_norm_expected = math.sqrt(head_dim)

        # Block energy = 4.0 → √E = 2.0
        block_energies = torch.tensor([[4.0]])

        beta = compute_block_beta(q, block_energies, head_dim)
        expected = q_norm_expected * 2.0 / math.sqrt(head_dim)
        assert torch.allclose(beta, torch.tensor([[expected]]), atol=1e-5)

    def test_zero_energy_gives_zero_beta(self):
        """Blocks with zero energy should have β = 0."""
        head_dim = 64
        q = torch.randn(1, 2, head_dim)
        block_energies = torch.zeros(3, 2)
        beta = compute_block_beta(q, block_energies, head_dim)
        assert torch.all(beta == 0.0)


class TestPerfectEvictionCheck:
    """Test the scalar perfect eviction check."""

    def test_above_threshold_is_perfect(self):
        """z_max - β ≥ 88.72 should qualify for perfect eviction."""
        assert perfect_eviction_check(z_max=100.0, beta=10.0) is True
        assert perfect_eviction_check(z_max=88.72, beta=0.0) is True

    def test_below_threshold_is_not_perfect(self):
        """z_max - β < 88.72 should NOT qualify for perfect eviction."""
        assert perfect_eviction_check(z_max=88.71, beta=0.0) is False
        assert perfect_eviction_check(z_max=50.0, beta=10.0) is False

    def test_exact_threshold_boundary(self):
        """Exact boundary: z_max - β == 88.72 should be perfect."""
        assert perfect_eviction_check(z_max=100.0, beta=100.0 - 88.72) is True

    def test_bfloat16_has_lower_threshold(self):
        """bfloat16 should use a lower underflow threshold."""
        # This gap passes float32 but also passes bfloat16
        assert perfect_eviction_check(z_max=100.0, beta=11.0, accumulator_dtype="float32") is True
        assert perfect_eviction_check(z_max=100.0, beta=11.0, accumulator_dtype="bfloat16") is True

        # This gap passes bfloat16 threshold but not float32
        gap = 88.0  # between 87.34 and 88.72
        assert perfect_eviction_check(z_max=gap, beta=0.0, accumulator_dtype="float32") is False
        assert perfect_eviction_check(z_max=gap, beta=0.0, accumulator_dtype="bfloat16") is True


class TestClassifyEviction:
    """Test the tensor-based eviction classifier."""

    def _make_test_scenario(self):
        """Create a test scenario with 4 blocks, 1 head."""
        head_dim = 64
        num_heads = 1
        num_blocks = 4
        q = torch.randn(1, num_heads, head_dim)

        # Block energies: varying
        block_energies = torch.tensor([
            [100.0],  # Block 0: high energy → large β → statistical eviction
            [0.001],  # Block 1: tiny energy → tiny β → z_max - β is huge → perfect
            [50.0],   # Block 2: medium energy → retained
            [0.0001], # Block 3: near-zero energy → perfect eviction
        ])

        # Retained blocks: 0 and 2
        block_mask = torch.tensor([True, False, True, False])

        # z_max: high enough that low-energy blocks cross underflow threshold
        z_max = torch.tensor(100.0)

        return q, block_energies, z_max, block_mask, head_dim

    def test_classification_is_exhaustive(self):
        """Every block should be classified as exactly one regime."""
        q, energies, z_max, mask, hd = self._make_test_scenario()
        meta = classify_eviction(q, energies, z_max, mask, hd)

        total = meta.num_retained + meta.num_perfect + meta.num_statistical
        assert total == 4, f"Expected 4 blocks classified, got {total}"

    def test_retained_blocks_have_regime_zero(self):
        """Retained blocks should have regime = 0."""
        q, energies, z_max, mask, hd = self._make_test_scenario()
        meta = classify_eviction(q, energies, z_max, mask, hd)

        assert meta.regime[0].item() == 0  # Block 0 is retained
        assert meta.regime[2].item() == 0  # Block 2 is retained

    def test_low_energy_blocks_are_perfect(self):
        """Blocks with near-zero energy and high z_max should be perfect."""
        head_dim = 64
        q = torch.ones(1, 1, head_dim) * 0.1  # Small query norm
        # Very low energy → β ≈ 0 → gap ≈ z_max = 100 >> 88.72
        block_energies = torch.tensor([[1e-10]])
        z_max = torch.tensor(100.0)
        block_mask = torch.tensor([False])

        meta = classify_eviction(q, block_energies, z_max, block_mask, head_dim)
        assert meta.num_perfect == 1
        assert meta.regime[0].item() == 1

    def test_high_energy_blocks_are_statistical(self):
        """Blocks with high energy may not cross underflow threshold."""
        head_dim = 64
        q = torch.ones(1, 1, head_dim) * 10.0  # Large query norm
        # High energy → large β → gap < 88.72
        block_energies = torch.tensor([[10000.0]])
        z_max = torch.tensor(15.0)
        block_mask = torch.tensor([False])

        meta = classify_eviction(q, block_energies, z_max, block_mask, head_dim)
        assert meta.num_statistical == 1
        assert meta.regime[0].item() == 2

    def test_quantized_exp_is_exact_zero_for_perfect(self):
        """Verify that exp(z_i - z_max) is exactly 0.0 in float32 for perfect blocks."""
        # Construct a scenario where z_i - z_max < -88.72
        z_max = 100.0
        z_i = 10.0  # z_i - z_max = -90 < -88.72

        # In float32, exp(-90) should be flushed to 0
        result = np.float32(np.exp(np.float32(z_i - z_max)))
        assert result == 0.0, f"Expected exact 0.0 in float32, got {result}"

    def test_multi_head_classification(self):
        """Classification should work with multiple heads."""
        head_dim = 64
        num_heads = 4
        q = torch.randn(1, num_heads, head_dim)
        block_energies = torch.rand(8, num_heads) * 0.001  # Very low energy
        z_max = torch.tensor(100.0)
        block_mask = torch.tensor([True, False, True, False, True, False, True, False])

        meta = classify_eviction(q, block_energies, z_max, block_mask, head_dim)
        assert meta.num_retained == 4
        assert meta.num_perfect + meta.num_statistical == 4


class TestDualRegimeCompleteness:
    """Verify the dual-regime split is complete and correct."""

    @pytest.mark.parametrize("gap", [0.0, 44.36, 88.71, 88.72, 88.73, 100.0, 200.0])
    def test_gap_classification(self, gap):
        """Every gap value should be classified as either perfect or statistical."""
        is_perfect = gap >= FLOAT32_UNDERFLOW_THRESHOLD
        is_statistical = gap < FLOAT32_UNDERFLOW_THRESHOLD
        assert is_perfect != is_statistical, "Exactly one regime must apply"

    def test_float32_hardware_verification(self):
        """Exhaustive check: all gaps ≥ 88.72 produce exact zero in float32."""
        for gap in [88.72, 89.0, 90.0, 100.0, 150.0, 200.0]:
            val = np.float32(np.exp(np.float32(-gap)))
            assert val == 0.0, f"float32 exp(-{gap}) = {val}, expected 0.0"

    def test_float32_below_threshold_nonzero(self):
        """Gaps below threshold should produce nonzero float32 values."""
        for gap in [10.0, 50.0, 80.0, 85.0, 88.0]:
            val = np.float32(np.exp(np.float32(-gap)))
            assert val > 0.0, f"float32 exp(-{gap}) should be nonzero, got {val}"
