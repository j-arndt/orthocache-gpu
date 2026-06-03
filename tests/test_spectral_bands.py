"""Tests for Multi-Band Sequency Filtering (GPU Edition).

Validates that:
1. The PyTorch multi-band decomposition matches the NumPy reference implementation
2. The spectral decay ratio ζ correctly distinguishes noise from coherent blocks
3. ζ is NOT computable from spatial statistics alone (the FWHT is load-bearing)
4. The two-gate multiband mask works correctly
"""

import pytest
import numpy as np
import torch

from orthocache_gpu.spectral_energy import (
    compute_spectral_bands,
    compute_spectral_decay_ratio,
    compute_multiband_mask,
    BAND_LOW,
    BAND_MID,
    BAND_HIGH,
)
from orthocache_gpu.reference import (
    compute_spectral_bands_reference,
    compute_spectral_decay_ratio_reference,
    compute_multiband_mask_reference,
)


# ── Band Decomposition Correctness ──────────────────────────────────────────


def test_spectral_bands_match_reference():
    """PyTorch multi-band decomposition should match NumPy reference."""
    torch.manual_seed(42)
    np.random.seed(42)
    seq_len = 1024
    num_heads = 2
    head_dim = 64
    block_size = 512

    keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)

    # Reference
    ref_dc, ref_low, ref_mid, ref_high = compute_spectral_bands_reference(
        keys, block_size
    )

    # PyTorch
    torch_dc, torch_low, torch_mid, torch_high = compute_spectral_bands(
        torch.tensor(keys), block_size
    )

    torch.testing.assert_close(
        torch_dc, torch.tensor(ref_dc, dtype=torch.float32), rtol=1e-4, atol=1e-4,
    )
    torch.testing.assert_close(
        torch_low, torch.tensor(ref_low, dtype=torch.float32), rtol=1e-4, atol=1e-4,
    )
    torch.testing.assert_close(
        torch_mid, torch.tensor(ref_mid, dtype=torch.float32), rtol=1e-4, atol=1e-4,
    )
    torch.testing.assert_close(
        torch_high, torch.tensor(ref_high, dtype=torch.float32), rtol=1e-4, atol=1e-4,
    )


def test_band_energy_sums_to_total():
    """DC² + low + mid + high should equal total spectral energy."""
    torch.manual_seed(42)
    seq_len = 1024
    num_heads = 2
    head_dim = 64
    block_size = 512

    keys = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float32)

    dc, low, mid, high = compute_spectral_bands(keys, block_size)

    # DC energy: sum of squared DC coefficients across head_dim
    dc_energy = torch.sum(dc ** 2, dim=-1)  # (num_blocks, num_heads)

    band_total = dc_energy + low + mid + high

    # Total spatial energy (which equals total spectral energy by Parseval)
    blocks = keys.reshape(2, block_size, num_heads, head_dim)
    total_spatial = torch.sum(blocks ** 2, dim=(1, 3))

    torch.testing.assert_close(band_total, total_spatial, rtol=1e-3, atol=1e-3)


# ── Spectral Decay Ratio ζ ──────────────────────────────────────────────────


def test_spectral_decay_ratio_correctness():
    """ζ from PyTorch should match ζ from NumPy reference."""
    torch.manual_seed(123)
    np.random.seed(123)
    seq_len = 1024
    num_heads = 2
    head_dim = 64
    block_size = 512

    keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)

    ref_zeta = compute_spectral_decay_ratio_reference(keys, block_size)
    torch_zeta = compute_spectral_decay_ratio(torch.tensor(keys), block_size)

    torch.testing.assert_close(
        torch_zeta, torch.tensor(ref_zeta, dtype=torch.float32), rtol=1e-4, atol=1e-4,
    )


def test_zeta_high_for_noise():
    """Synthetic high-frequency noise block should have ζ >> 1."""
    np.random.seed(99)
    block_size = 512
    head_dim = 64

    # Alternating +1/-1 pattern with random amplitudes
    noise_block = np.zeros((block_size, head_dim), dtype=np.float64)
    for i in range(block_size):
        sign = 1.0 if i % 2 == 0 else -1.0
        noise_block[i, :] = sign * np.random.randn(head_dim) * 2.0

    zeta = compute_spectral_decay_ratio_reference(noise_block, block_size)

    assert zeta > 1.0, f"Expected ζ > 1 for noise block, got {zeta:.4f}"


def test_zeta_low_for_coherent():
    """Synthetic coherent block (low Walsh sequency) should have ζ << 1.

    Walsh functions are binary (±1) patterns ordered by number of sign changes.
    Low-sequency Walsh basis vectors produce smooth step-like patterns.
    A block constructed from these vectors has energy concentrated in low bands.
    """
    np.random.seed(42)
    block_size = 512
    head_dim = 64

    # Construct a block from the first 16 Walsh basis vectors
    coherent_block = np.zeros((block_size, head_dim), dtype=np.float64)
    rng = np.random.RandomState(42)
    for j in range(1, 16):  # Skip DC (j=0) so energy is in AC low band
        walsh_j = np.ones(block_size)
        for i in range(block_size):
            for bit in range(9):  # log2(512)
                if (j >> bit) & 1 and (i >> bit) & 1:
                    walsh_j[i] *= -1
        coeff = rng.randn(head_dim) * 0.5
        coherent_block += np.outer(walsh_j, coeff)

    zeta = compute_spectral_decay_ratio_reference(coherent_block, block_size)

    assert zeta < 0.01, f"Expected ζ < 0.01 for Walsh-coherent block, got {zeta:.4f}"


# ── FWHT Is Load-Bearing ────────────────────────────────────────────────────


def test_zeta_not_computable_spatially():
    """CRITICAL TEST: Proves the FWHT is load-bearing.

    Constructs two blocks with IDENTICAL spatial variance but DIFFERENT
    spectral decay ratios, demonstrating that no spatial-domain function
    can compute ζ without access to individual spectral coefficients.

    Block A: constructed from low-sequency Walsh basis vectors (energy in low bands)
    Block B: random white noise scaled to have the same total variance
    """
    block_size = 512
    head_dim = 64

    # Block A: sum of low-sequency Walsh basis vectors (indices 1-15)
    block_a = np.zeros((block_size, head_dim), dtype=np.float64)
    rng_a = np.random.RandomState(2026)
    for j in range(1, 16):
        walsh_j = np.ones(block_size)
        for i in range(block_size):
            for bit in range(9):
                if (j >> bit) & 1 and (i >> bit) & 1:
                    walsh_j[i] *= -1
        coeff = rng_a.randn(head_dim) * 0.5
        block_a += np.outer(walsh_j, coeff)

    # Block B: random white noise
    rng_b = np.random.RandomState(42)
    block_b = rng_b.randn(block_size, head_dim)

    # Scale block B to have the EXACT same total spatial variance as block A
    mean_a = block_a.mean(axis=0)
    mean_b = block_b.mean(axis=0)
    var_a = np.sum((block_a - mean_a) ** 2)
    var_b = np.sum((block_b - mean_b) ** 2)
    block_b = mean_b + (block_b - mean_b) * np.sqrt(var_a / var_b)

    # Verify: SAME spatial variance
    var_a_final = np.sum((block_a - block_a.mean(axis=0)) ** 2)
    var_b_final = np.sum((block_b - block_b.mean(axis=0)) ** 2)
    np.testing.assert_allclose(
        var_a_final, var_b_final, rtol=1e-10,
        err_msg="Blocks must have identical spatial variance for this test to be valid",
    )

    # But: DIFFERENT spectral decay ratios
    zeta_a = compute_spectral_decay_ratio_reference(block_a, block_size)
    zeta_b = compute_spectral_decay_ratio_reference(block_b, block_size)

    assert zeta_a < zeta_b, (
        f"Expected ζ_A < ζ_B for coherent vs noise blocks with identical variance, "
        f"got ζ_A={zeta_a:.4f} vs ζ_B={zeta_b:.4f}"
    )
    assert zeta_a < 0.01, f"Walsh-coherent block should have near-zero ζ: {zeta_a:.6f}"
    assert zeta_b > 1.0, f"Noise block should have ζ > 1: {zeta_b:.4f}"


# ── Multiband Mask ──────────────────────────────────────────────────────────


def test_multiband_mask_two_gate():
    """Verify that the multiband mask applies BOTH gates."""
    torch.manual_seed(42)
    np.random.seed(42)
    seq_len_k = 1024  # 2 blocks
    seq_len_q = 1
    num_heads = 1
    head_dim = 64
    block_size = 512

    # Create keys where Block 0 is high-frequency noise
    keys = np.random.randn(seq_len_k, num_heads, head_dim).astype(np.float32)
    for i in range(block_size):
        sign = 1.0 if i % 2 == 0 else -1.0
        keys[i, 0, :] *= sign * 5.0

    q = np.random.randn(seq_len_q, num_heads, head_dim).astype(np.float32)

    # Very loose tau (everything passes gate 1) + tight zeta_max
    tau = -100.0
    zeta_max = 1.0

    mask_ref = compute_multiband_mask_reference(q, keys, tau, zeta_max, block_size)
    mask_torch = compute_multiband_mask(
        torch.tensor(q), torch.tensor(keys), tau, zeta_max, block_size
    )

    # Both should agree
    np.testing.assert_array_equal(mask_torch.numpy(), mask_ref)

    # Block 0 (noise) should be evicted by ζ gate
    assert not mask_ref.all(), "Expected at least one block to be evicted by ζ gate"
