"""Tests for orthocache_gpu.spectral_energy — block energy and query-aware bounds.

Validates:
1. compute_block_energy matches reference implementation
2. Energy is non-negative
3. Parseval: spectral energy == spatial energy
4. Query-aware bounds match reference
5. Query-aware mask generation
6. Threshold mask generation
"""

import pytest
import numpy as np
import torch

from orthocache_gpu.spectral_energy import (
    compute_block_energy,
    generate_threshold_mask,
    compute_query_aware_bounds,
    compute_query_aware_mask,
)
from orthocache_gpu.reference import (
    compute_block_energy_reference,
    compute_query_aware_bounds_reference,
)


# ── Block Energy Correctness ────────────────────────────────────────────────


def test_spectral_energy_correctness():
    """PyTorch compute_block_energy should match NumPy reference."""
    torch.manual_seed(42)
    np.random.seed(42)
    seq_len = 1024
    num_heads = 4
    head_dim = 64
    block_size = 512

    keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)

    # Reference spectral energies (NumPy)
    ref_energies = compute_block_energy_reference(keys, block_size)

    # PyTorch implementation
    torch_energies = compute_block_energy(
        torch.tensor(keys, dtype=torch.float32), block_size
    )

    torch.testing.assert_close(
        torch_energies,
        torch.tensor(ref_energies, dtype=torch.float32),
        rtol=1e-4,
        atol=1e-4,
    )


def test_energy_non_negative():
    """Block energy should always be non-negative (sum of squares)."""
    torch.manual_seed(123)
    keys = torch.randn(2048, 4, 64, dtype=torch.float32)
    energies = compute_block_energy(keys, 512)

    assert torch.all(energies >= 0), "Block energy should be non-negative"


def test_parseval_spectral_equals_spatial():
    """Parseval: spectral energy == spatial energy per block.

    Since compute_block_energy computes spatial energy (sum of squared key norms),
    and Parseval guarantees this equals spectral energy, the function output
    should equal the direct spatial computation.
    """
    torch.manual_seed(55)
    seq_len = 1024
    num_heads = 2
    head_dim = 64
    block_size = 512

    keys = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float32)
    blocks = keys.reshape(2, block_size, num_heads, head_dim)

    # Spatial energy: direct sum of squares
    spatial_energy = torch.sum(blocks ** 2, dim=(1, 3))

    # Via the function
    computed_energy = compute_block_energy(keys, block_size)

    torch.testing.assert_close(computed_energy, spatial_energy, rtol=1e-6, atol=1e-6)


# ── Query-Aware Bounds ──────────────────────────────────────────────────────


def test_query_aware_bounds_correctness():
    """PyTorch query-aware bounds should match NumPy reference."""
    torch.manual_seed(42)
    np.random.seed(42)
    seq_len_k = 1024
    seq_len_q = 8
    num_heads = 4
    head_dim = 64
    block_size = 512

    q = np.random.randn(seq_len_q, num_heads, head_dim).astype(np.float32)
    keys = np.random.randn(seq_len_k, num_heads, head_dim).astype(np.float32)

    # Reference
    ref_bounds = compute_query_aware_bounds_reference(q, keys, block_size)

    # PyTorch
    torch_bounds = compute_query_aware_bounds(
        torch.tensor(q), torch.tensor(keys), block_size
    )

    torch.testing.assert_close(
        torch_bounds,
        torch.tensor(ref_bounds, dtype=torch.float32),
        rtol=1e-4,
        atol=1e-4,
    )


# ── Query-Aware Mask ────────────────────────────────────────────────────────


def test_query_aware_mask():
    """Query-aware mask should have correct shape and dtype."""
    torch.manual_seed(42)
    np.random.seed(42)
    seq_len_k = 1024
    seq_len_q = 1
    num_heads = 4
    head_dim = 64
    block_size = 512

    q = torch.randn(seq_len_q, num_heads, head_dim, dtype=torch.float32)
    keys = torch.randn(seq_len_k, num_heads, head_dim, dtype=torch.float32)

    # Generate mask with threshold
    tau = 0.5
    mask = compute_query_aware_mask(q, keys, tau, block_size)

    # Check shape: (num_blocks, num_heads)
    assert mask.shape == (seq_len_k // block_size, num_heads)
    assert mask.dtype == torch.bool


# ── Threshold Mask ──────────────────────────────────────────────────────────


def test_threshold_mask():
    """generate_threshold_mask should retain blocks with energy >= epsilon."""
    energies = torch.tensor([[10.5, 2.3], [1.1, 15.6]], dtype=torch.float32)
    epsilon = 5.0
    mask = generate_threshold_mask(energies, epsilon)

    expected = torch.tensor([[True, False], [False, True]])
    assert torch.equal(mask, expected)
