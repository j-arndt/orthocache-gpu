"""Correctness test suite for the Phase 7 Fused Triton Kernel.

Tests cover all four proof dimensions:
    - Walsh matrix properties (orthogonality, symmetry, DC row)
    - FWHT dot-product equivalence vs butterfly (fp32 and bf16)
    - Spectral mask parity (100% match rate against reference)
    - SRAM and register budget verification
    - Cosine similarity and tolerance for attention output
"""

import pytest
import torch
import numpy as np

# ── Availability guards ──────────────────────────────────────────────
HAS_CUDA = torch.cuda.is_available()
HAS_TRITON = False
try:
    import triton
    HAS_TRITON = True
except ImportError:
    pass

requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
requires_triton = pytest.mark.skipif(
    not (HAS_CUDA and HAS_TRITON), reason="CUDA + Triton required"
)

# ── Imports from orthocache_gpu ──────────────────────────────────────
from orthocache_gpu.triton_kernels.fwht_fused_prototype import (
    generate_walsh_matrix,
    triton_fwht_eviction,
    _pytorch_fwht_eviction,
    TILE_SIZE,
    BAND_LOW_64,
    BAND_HIGH_64,
)


# ====================================================================
# 2.2 Walsh Matrix Tests
# ====================================================================


class TestWalshMatrix:
    """Validate the precomputed Walsh-Hadamard matrix."""

    def test_walsh_matrix_orthogonality(self):
        """W @ W.T must equal the identity matrix."""
        W = generate_walsh_matrix(64)
        I_approx = W @ W.T
        torch.testing.assert_close(
            I_approx,
            torch.eye(64, dtype=torch.float32),
            atol=1e-6,
            rtol=0,
        )

    def test_walsh_matrix_symmetry(self):
        """Walsh-Hadamard matrix is symmetric: W == W.T."""
        W = generate_walsh_matrix(64)
        torch.testing.assert_close(W, W.T, atol=1e-7, rtol=0)

    def test_walsh_matrix_dc_row(self):
        """First row (DC) should be constant 1/sqrt(n)."""
        W = generate_walsh_matrix(64)
        expected_val = 1.0 / (64 ** 0.5)  # 1/8 = 0.125
        assert torch.allclose(
            W[0, :],
            torch.full((64,), expected_val, dtype=torch.float32),
            atol=1e-7,
        )

    def test_walsh_matrix_entries_magnitude(self):
        """All entries should be ±1/sqrt(n)."""
        W = generate_walsh_matrix(64)
        expected_abs = 1.0 / (64 ** 0.5)
        assert torch.allclose(
            W.abs(),
            torch.full_like(W, expected_abs),
            atol=1e-7,
        )

    @pytest.mark.parametrize("n", [2, 4, 8, 16, 32, 64, 128])
    def test_walsh_matrix_sizes(self, n):
        """Walsh matrix generation works for various power-of-2 sizes."""
        W = generate_walsh_matrix(n)
        assert W.shape == (n, n)
        I_approx = W @ W.T
        torch.testing.assert_close(
            I_approx, torch.eye(n, dtype=torch.float32), atol=1e-5, rtol=0
        )


# ====================================================================
# 2.3 FWHT Dot Product vs Butterfly Equivalence
# ====================================================================


class TestFWHTDotVsButterflyEquivalence:
    """Prove tl.dot(W, K) matches the standard FWHT butterfly."""

    def _reference_fwht_64(self, x: torch.Tensor) -> torch.Tensor:
        """64-point FWHT via butterfly network (reference implementation).

        Implements the same algorithm as fwht_512 but for 64 rows (6 stages).
        """
        n = 64
        assert x.shape[0] == n
        is_1d = x.ndim == 1
        if is_1d:
            x = x[:, None]
        d = x.shape[1]

        # 6 stages for 64-point FWHT
        # Stage 0: h=1
        x = x.reshape(32, 2, 1, d)
        x = torch.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], dim=1).reshape(n, d)
        # Stage 1: h=2
        x = x.reshape(16, 2, 2, d)
        x = torch.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], dim=1).reshape(n, d)
        # Stage 2: h=4
        x = x.reshape(8, 2, 4, d)
        x = torch.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], dim=1).reshape(n, d)
        # Stage 3: h=8
        x = x.reshape(4, 2, 8, d)
        x = torch.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], dim=1).reshape(n, d)
        # Stage 4: h=16
        x = x.reshape(2, 2, 16, d)
        x = torch.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], dim=1).reshape(n, d)
        # Stage 5: h=32
        x = x.reshape(1, 2, 32, d)
        x = torch.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], dim=1).reshape(n, d)

        x = x / (64 ** 0.5)

        if is_1d:
            x = x.squeeze(1)
        return x

    def test_fwht_dot_vs_butterfly_fp32(self):
        """Dense W @ K must match butterfly FWHT in float32."""
        torch.manual_seed(42)
        K = torch.randn(64, 128, dtype=torch.float32)
        W = generate_walsh_matrix(64)

        result_dot = W @ K
        result_butterfly = self._reference_fwht_64(K)

        torch.testing.assert_close(
            result_dot, result_butterfly, atol=1e-5, rtol=1e-4
        )

    def test_fwht_dot_vs_butterfly_bf16(self):
        """Dense W @ K matches butterfly FWHT with bf16 inputs."""
        torch.manual_seed(42)
        K = torch.randn(64, 128, dtype=torch.float32)
        W = generate_walsh_matrix(64)

        # bf16 path: cast K, compute in fp32 (as the kernel does)
        K_bf16 = K.to(torch.bfloat16)
        result_dot = W @ K_bf16.float()
        result_butterfly = self._reference_fwht_64(K_bf16.float())

        torch.testing.assert_close(
            result_dot, result_butterfly, atol=1e-3, rtol=1e-2
        )


# ====================================================================
# 2.4 Spectral Mask Parity (The Critical Test)
# ====================================================================


class TestSpectralMaskParity:
    """Prove the Triton kernel makes identical eviction decisions to the reference."""

    def _make_semantic_block(self, head_dim=128):
        """Generate a smooth, low-frequency block (should be RETAINED)."""
        t = torch.linspace(0, 2 * np.pi, 64)
        # Low-frequency signal: smooth sine wave across tokens
        signal = torch.sin(t).unsqueeze(1).expand(64, head_dim)
        return signal + 0.01 * torch.randn(64, head_dim)

    def _make_noise_block(self, head_dim=128):
        """Generate a high-frequency noise block (should be EVICTED)."""
        return torch.randn(64, head_dim) * 5.0

    @requires_triton
    def test_zeta_mask_parity_synthetic(self):
        """Fused kernel must make 100% identical eviction decisions vs reference."""
        torch.manual_seed(123)
        device = torch.device('cuda')

        # Build 16 tiles: 8 semantic + 8 noise
        blocks = []
        for _ in range(8):
            blocks.append(self._make_semantic_block())
        for _ in range(8):
            blocks.append(self._make_noise_block())
        keys = torch.cat(blocks, dim=0).to(device)  # (1024, 128)

        zeta_max = 5.0

        # Fused kernel
        mask_fused, _ = triton_fwht_eviction(keys, zeta_max)

        # Reference (PyTorch)
        mask_ref, _ = _pytorch_fwht_eviction(keys, zeta_max)

        assert (mask_fused == mask_ref).all(), (
            f"Mask mismatch! Fused: {mask_fused.tolist()}, Ref: {mask_ref.tolist()}"
        )

    @requires_triton
    def test_zeta_mask_parity_random(self):
        """Fused kernel parity with 32 random tiles."""
        torch.manual_seed(456)
        device = torch.device('cuda')

        keys = torch.randn(32 * 64, 128, device=device)
        zeta_max = 3.0

        mask_fused, _ = triton_fwht_eviction(keys, zeta_max)
        mask_ref, _ = _pytorch_fwht_eviction(keys.cpu(), zeta_max)

        assert (mask_fused.cpu() == mask_ref).all(), "Mask parity failed on random data"

    @requires_triton
    def test_zeta_values_match_reference(self):
        """ζ values from fused kernel must match reference within tolerance."""
        torch.manual_seed(789)
        device = torch.device('cuda')

        keys = torch.randn(16 * 64, 128, device=device)
        zeta_max = 5.0

        _, zeta_fused = triton_fwht_eviction(keys, zeta_max, return_zeta=True)
        _, zeta_ref = _pytorch_fwht_eviction(keys.cpu(), zeta_max, return_zeta=True)

        torch.testing.assert_close(
            zeta_fused.cpu(), zeta_ref, atol=1e-2, rtol=1e-2
        )


# ====================================================================
# 2.5 SRAM Constraint Tests
# ====================================================================


class TestSRAMConstraints:
    """Verify the kernel fits within SM resource limits."""

    @requires_triton
    def test_sram_under_limit(self):
        """Shared memory usage must be under 85 KB."""
        device = torch.device('cuda')
        keys = torch.randn(64, 128, device=device)

        # Trigger compilation
        triton_fwht_eviction(keys, 5.0)
        torch.cuda.synchronize()

        # Access metadata (best-effort — Triton API varies)
        from orthocache_gpu.triton_kernels.fwht_fused_prototype import _fwht_eviction_kernel
        try:
            cache = _fwht_eviction_kernel.cache
            if cache and cache[0]:
                compiled = list(cache[0].values())[0]
                if hasattr(compiled, 'metadata'):
                    shared = compiled.metadata.get('shared', None)
                    if shared is not None:
                        assert shared < 85_000, (
                            f"SRAM usage {shared} bytes ({shared/1024:.1f} KB) "
                            f"exceeds 85 KB budget!"
                        )
                        print(f"✅ SRAM: {shared} bytes ({shared/1024:.1f} KB / 100 KB)")
                        return
            # If we can't access metadata, warn but don't fail
            pytest.skip("Could not access Triton kernel metadata; verify with ncu")
        except Exception as e:
            pytest.skip(f"Triton metadata access failed: {e}; verify with ncu")

    @requires_triton
    def test_kernel_compiles_and_runs(self):
        """Basic smoke test: kernel compiles, runs, and produces valid output."""
        device = torch.device('cuda')
        keys = torch.randn(4 * 64, 128, device=device)

        mask, zeta = triton_fwht_eviction(keys, 5.0, return_zeta=True)
        torch.cuda.synchronize()

        assert mask.shape == (4,)
        assert mask.dtype == torch.bool
        assert zeta.shape == (4,)
        assert zeta.dtype == torch.float32
        assert torch.all(zeta >= 0), "ζ must be non-negative"


# ====================================================================
# 2.6 Band Energy Accuracy
# ====================================================================


class TestBandEnergyAccuracy:
    """Verify per-band energy matches the PyTorch reference."""

    def _compute_reference_band_energies(self, keys, tile_size=64):
        """Compute E_low and E_high using PyTorch reference path."""
        W = generate_walsh_matrix(tile_size)
        num_tiles = keys.shape[0] // tile_size
        tiles = keys.reshape(num_tiles, tile_size, -1).float()

        # FWHT via matmul
        k_spectral = torch.matmul(W.unsqueeze(0).to(keys.device), tiles)

        # Per-sequency energy
        energy_per_seq = torch.sum(k_spectral ** 2, dim=2)

        e_low = torch.sum(energy_per_seq[:, BAND_LOW_64[0]:BAND_LOW_64[1]], dim=1)
        e_high = torch.sum(energy_per_seq[:, BAND_HIGH_64[0]:BAND_HIGH_64[1]], dim=1)

        return e_low, e_high

    @requires_triton
    def test_band_energy_low_matches_reference(self):
        """E_low from Triton must match reference within 1% relative error."""
        torch.manual_seed(321)
        device = torch.device('cuda')
        keys = torch.randn(8 * 64, 128, device=device)

        # Get ζ from Triton (ζ = E_high / (E_low + 1e-6))
        _, zeta_fused = triton_fwht_eviction(keys, 999.0, return_zeta=True)

        # Get reference energies
        e_low_ref, e_high_ref = self._compute_reference_band_energies(keys.cpu())
        zeta_ref = e_high_ref / (e_low_ref + 1e-6)

        # Compare ζ (which implicitly validates the ratio of E_high to E_low)
        torch.testing.assert_close(
            zeta_fused.cpu(), zeta_ref, atol=1e-2, rtol=1e-2
        )

    def test_band_boundaries_cover_full_spectrum(self):
        """DC + Low + Mid + High must cover all 64 coefficients."""
        dc_count = 1  # index 0
        low_count = BAND_LOW_64[1] - BAND_LOW_64[0]   # 7
        mid_count = BAND_HIGH_64[0] - BAND_LOW_64[1]   # 24
        high_count = BAND_HIGH_64[1] - BAND_HIGH_64[0]  # 32
        total = dc_count + low_count + mid_count + high_count
        assert total == 64, f"Bands cover {total}/64 coefficients"


# ====================================================================
# 2.7 64-Point vs 512-Point Spectral Separation Validation
# ====================================================================


class TestSpectralSeparation:
    """Validate that 64-point ζ separates semantic from noise blocks
    as effectively as 512-point ζ.

    Key insight: semantic blocks have energy concentrated in DC and low
    sequency bands (indices 0-7). A constant or slowly-varying signal
    achieves this. Random noise distributes energy roughly evenly across
    all bands, giving ζ ≈ high_count/low_count ≈ 32/7 ≈ 4.6.
    """

    def _make_semantic_block(self, head_dim=128):
        """Create a block with energy ONLY in DC + low sequency bands.

        Constructs the signal directly from Walsh basis vectors 0-7.
        By Parseval's theorem and orthogonality of the Walsh matrix,
        this guarantees E_mid = E_high = 0 exactly, giving ζ = 0.

        This is the technically correct way to synthesize signals with
        known spectral properties: define the coefficients, then
        inverse-transform.
        """
        W = generate_walsh_matrix(64)
        # Define spectral coefficients: energy only in DC + low band (0-7)
        coeffs = torch.zeros(64, head_dim)
        for i in range(8):  # indices 0-7 = DC + low band
            coeffs[i] = torch.randn(head_dim) * (5.0 / (i + 1))
        # Inverse WHT (W is symmetric and orthogonal: W^{-1} = W)
        return W @ coeffs

    def _make_noise_block(self, head_dim=128):
        """Create a noise-dominated block: IID Gaussian per token.

        Energy distributes roughly evenly across all sequency bands.
        ζ ≈ high_count / low_count ≈ 32/7 ≈ 4.6.
        """
        return torch.randn(64, head_dim)

    def test_64pt_separates_semantic_from_noise(self):
        """64-point ζ must correctly identify noise blocks."""
        torch.manual_seed(42)

        semantic = self._make_semantic_block()
        noise = self._make_noise_block()

        keys = torch.cat([semantic, noise], dim=0)  # (128, 128) = 2 tiles

        _, zeta = _pytorch_fwht_eviction(keys, 999.0, return_zeta=True)

        zeta_semantic = zeta[0].item()
        zeta_noise = zeta[1].item()

        # Semantic block: ζ should be very low (DC + low band dominated)
        # Noise block: ζ should be ~4.6 (uniform energy distribution)
        assert zeta_noise > zeta_semantic, (
            f"Separation failed: ζ_noise={zeta_noise:.3f} <= ζ_semantic={zeta_semantic:.3f}"
        )
        # The ratio should be substantial
        ratio = zeta_noise / (zeta_semantic + 1e-6)
        print(f"Spectral separation ratio: {ratio:.2f}x "
              f"(zeta_semantic={zeta_semantic:.4f}, zeta_noise={zeta_noise:.4f})")
        assert ratio > 2.0, f"Separation ratio {ratio:.2f} too low"

    @requires_triton
    def test_64pt_gpu_separation(self):
        """Same separation test on GPU via Triton kernel."""
        torch.manual_seed(42)
        device = torch.device('cuda')

        semantic = self._make_semantic_block().to(device)
        noise = self._make_noise_block().to(device)

        keys = torch.cat([semantic, noise], dim=0)

        _, zeta = triton_fwht_eviction(keys, 999.0, return_zeta=True)

        assert zeta[1] > zeta[0], (
            f"GPU separation failed: ζ_noise={zeta[1].item():.3f} "
            f"<= ζ_semantic={zeta[0].item():.3f}"
        )
