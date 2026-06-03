"""Tests for orthocache_gpu.fwht — GPU (PyTorch) Walsh-Hadamard Transform.

Validates:
1. fwht_512 correctness against NumPy reference implementation
2. Parseval's identity (energy preservation)
3. Involution property (applying FWHT twice = scaled identity)
"""

import pytest
import numpy as np
import torch

from orthocache_gpu.fwht import fwht_512
from orthocache_gpu.reference import numpy_fwht


# ── Correctness ─────────────────────────────────────────────────────────────


def test_fwht_correctness():
    """PyTorch fwht_512 should match the NumPy reference on a 2D input."""
    torch.manual_seed(42)
    np.random.seed(42)
    a = np.random.randn(512, 128).astype(np.float32)

    # NumPy reference
    ref_out = numpy_fwht(a)

    # PyTorch implementation
    torch_out = fwht_512(torch.tensor(a, dtype=torch.float32))

    torch.testing.assert_close(
        torch_out,
        torch.tensor(ref_out, dtype=torch.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_fwht_1d():
    """fwht_512 should handle 1D input (512,) correctly."""
    torch.manual_seed(42)
    np.random.seed(42)
    a = np.random.randn(512).astype(np.float32)

    # NumPy reference treats 1D as (512, 1)
    ref_out = numpy_fwht(a[:, None]).squeeze(1)

    # PyTorch should handle 1D natively
    torch_out = fwht_512(torch.tensor(a, dtype=torch.float32))

    torch.testing.assert_close(
        torch_out,
        torch.tensor(ref_out, dtype=torch.float32),
        rtol=1e-5,
        atol=1e-5,
    )


# ── Parseval's Identity ────────────────────────────────────────────────────


def test_parseval_energy_preservation():
    """FWHT should preserve total energy: ||x||² == ||FWHT(x)||² (Parseval)."""
    torch.manual_seed(99)
    x = torch.randn(512, 64, dtype=torch.float32)

    spatial_energy = torch.sum(x ** 2).item()
    spectral = fwht_512(x)
    spectral_energy = torch.sum(spectral ** 2).item()

    # Parseval's identity: energies must match within float32 tolerance
    assert abs(spatial_energy - spectral_energy) / spatial_energy < 1e-4, (
        f"Parseval violated: spatial={spatial_energy:.6f} vs spectral={spectral_energy:.6f}"
    )


# ── Involution ──────────────────────────────────────────────────────────────


def test_fwht_involution():
    """Applying FWHT twice should recover the original input (involution).

    The unnormalized WHT is its own inverse. Since fwht_512 normalizes
    by 1/sqrt(N), applying it twice gives: FWHT(FWHT(x)) = x.
    """
    torch.manual_seed(7)
    x = torch.randn(512, 32, dtype=torch.float32)

    # Apply FWHT twice
    y = fwht_512(fwht_512(x))

    torch.testing.assert_close(
        y, x, rtol=1e-4, atol=1e-4,
        msg="FWHT is not involutory: FWHT(FWHT(x)) != x",
    )


def test_fwht_involution_1d():
    """Involution should also hold for 1D inputs."""
    torch.manual_seed(77)
    x = torch.randn(512, dtype=torch.float32)

    y = fwht_512(fwht_512(x))

    torch.testing.assert_close(y, x, rtol=1e-4, atol=1e-4)
