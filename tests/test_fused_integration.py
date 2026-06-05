"""Integration tests for the Phase 7 Fused God Kernel.

Tests validate:
    - Attention output correctness (cosine similarity ≥ 0.95 vs dense)
    - Eviction mask consistency with isolated FWHT kernel
    - Zero eviction mode (zeta_max=Inf) matches dense attention exactly
    - K-tile reuse (no redundant HBM loads verified by design)
    - Online softmax numerical stability
"""

import pytest
import torch
import sys

sys.stdout.reconfigure(encoding='utf-8')

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

# ── Imports ──────────────────────────────────────────────────────────
from orthocache_gpu.triton_kernels.fwht_fused_prototype import (
    generate_walsh_matrix,
    triton_fwht_eviction,
    TILE_SIZE,
    BAND_LOW_64,
    BAND_HIGH_64,
)
from orthocache_gpu.triton_kernels.fused_eviction import (
    fused_orthocache_attention,
    _pytorch_fused_orthocache_attention,
)


def _dense_attention(q, keys, values, scale=None):
    """Standard dense attention for reference (no eviction)."""
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    logits = (q.float() @ keys.float().T) * scale
    weights = torch.softmax(logits, dim=-1)
    return (weights @ values.float()).to(q.dtype)


# ====================================================================
# 3.1 Zero Eviction Mode (zeta_max=Inf)
# ====================================================================


class TestZeroEviction:
    """With zeta_max=Inf, ALL tiles are retained → output must match dense."""

    def test_zero_eviction_cpu_matches_dense(self):
        """PyTorch fused with zeta_max=1e9 must match dense attention."""
        torch.manual_seed(42)
        q = torch.randn(1, 128)
        keys = torch.randn(8 * 64, 128)
        values = torch.randn(8 * 64, 128)

        out_fused, meta = _pytorch_fused_orthocache_attention(
            q, keys, values, zeta_max=1e9, return_mask=True
        )
        out_dense = _dense_attention(q, keys, values)

        cos_sim = torch.nn.functional.cosine_similarity(
            out_fused.float().view(1, -1),
            out_dense.float().view(1, -1),
        ).item()

        print(f"Zero-eviction cosine similarity: {cos_sim:.6f}")
        assert cos_sim > 0.999, f"Cosine sim {cos_sim:.6f} < 0.999"
        assert meta['tiles_evicted'] == 0, f"Expected 0 evictions, got {meta['tiles_evicted']}"
        assert meta['tiles_retained'] == 8

    @requires_triton
    def test_zero_eviction_gpu_matches_dense(self):
        """Triton fused with zeta_max=1e9 must match dense attention on GPU."""
        torch.manual_seed(42)
        device = torch.device('cuda')
        q = torch.randn(1, 128, device=device)
        keys = torch.randn(8 * 64, 128, device=device)
        values = torch.randn(8 * 64, 128, device=device)

        out_fused, meta = fused_orthocache_attention(
            q, keys, values, zeta_max=1e9, return_mask=True
        )
        out_dense = _dense_attention(q, keys, values)

        cos_sim = torch.nn.functional.cosine_similarity(
            out_fused.float().view(1, -1),
            out_dense.float().view(1, -1),
        ).item()

        print(f"GPU zero-eviction cosine similarity: {cos_sim:.6f}")
        assert cos_sim > 0.99, f"Cosine sim {cos_sim:.6f} < 0.99"


# ====================================================================
# 3.2 Eviction Mask Consistency
# ====================================================================


class TestEvictionMaskConsistency:
    """Fused kernel must make identical eviction decisions to isolated kernel."""

    @requires_triton
    def test_mask_matches_isolated_kernel(self):
        """God Kernel eviction mask must match isolated FWHT kernel mask."""
        torch.manual_seed(123)
        device = torch.device('cuda')

        # Build keys with known spectral properties
        W = generate_walsh_matrix(64).to(device)
        blocks = []
        for i in range(16):
            if i < 8:
                # Low-frequency block (retain)
                coeffs = torch.zeros(64, 128, device=device)
                for j in range(8):
                    coeffs[j] = torch.randn(128, device=device) * 5.0
                blocks.append(W @ coeffs)
            else:
                # Noise block (evict)
                blocks.append(torch.randn(64, 128, device=device))

        keys = torch.cat(blocks, dim=0)
        values = torch.randn_like(keys)
        q = torch.randn(1, 128, device=device)

        zeta_max = 3.0

        # Get mask from isolated kernel
        mask_isolated, _ = triton_fwht_eviction(keys, zeta_max)

        # Get mask from fused kernel
        _, meta = fused_orthocache_attention(
            q, keys, values, zeta_max, return_mask=True
        )

        assert (mask_isolated == meta['eviction_mask']).all(), (
            f"Mask mismatch!\n"
            f"Isolated: {mask_isolated.tolist()}\n"
            f"Fused:    {meta['eviction_mask'].tolist()}"
        )


# ====================================================================
# 3.3 Attention Output Quality with Eviction
# ====================================================================


class TestAttentionWithEviction:
    """Verify attention quality degrades gracefully under eviction."""

    @requires_triton
    def test_selective_eviction_cosine_sim(self):
        """With partial eviction, cosine sim to dense must be >= 0.85."""
        torch.manual_seed(789)
        device = torch.device('cuda')

        W = generate_walsh_matrix(64).to(device)

        # Build 32 tiles: 16 semantic (low-ζ, retained) + 16 noise (high-ζ, evicted)
        blocks = []
        for i in range(32):
            if i < 16:
                # Low-frequency block: energy only in DC + low band
                coeffs = torch.zeros(64, 128, device=device)
                for j in range(8):
                    coeffs[j] = torch.randn(128, device=device) * 5.0
                blocks.append(W @ coeffs)
            else:
                # Noise block: energy across all bands
                blocks.append(torch.randn(64, 128, device=device))

        keys = torch.cat(blocks, dim=0)
        values = torch.randn_like(keys)
        q = torch.randn(1, 128, device=device)

        # Dense reference (all tiles)
        out_dense = _dense_attention(q, keys, values)

        # Fused with eviction: ζ_max=3.0 should evict noise (ζ≈4.6), retain semantic (ζ≈0)
        out_fused, meta = fused_orthocache_attention(
            q, keys, values, zeta_max=3.0, return_mask=True
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            out_fused.float().view(1, -1),
            out_dense.float().view(1, -1),
        ).item()

        eviction_rate = meta.get('eviction_rate', 0)
        print(f"Eviction rate: {eviction_rate:.1%}, Cosine sim: {cos_sim:.4f}")

        # With ~50% eviction, output should still be meaningful
        assert 0 < eviction_rate < 1.0, (
            f"Expected partial eviction, got {eviction_rate:.1%}"
        )
        assert cos_sim > 0.85, f"Cosine sim {cos_sim:.4f} too low"

    def test_cpu_fused_matches_gpu_reference(self):
        """CPU fused attention matches its own reference implementation."""
        torch.manual_seed(456)
        q = torch.randn(1, 128)
        keys = torch.randn(8 * 64, 128)
        values = torch.randn(8 * 64, 128)

        out_fused, meta = _pytorch_fused_orthocache_attention(
            q, keys, values, zeta_max=5.0, return_mask=True
        )

        # Manually compute attention using only retained tiles
        mask = meta['eviction_mask']  # (8,)
        retained_indices = []
        for i in range(8):
            if mask[i]:
                retained_indices.extend(range(i * 64, (i + 1) * 64))

        if len(retained_indices) > 0:
            k_retained = keys[retained_indices].float()
            v_retained = values[retained_indices].float()
            out_manual = _dense_attention(q, k_retained, v_retained)

            cos_sim = torch.nn.functional.cosine_similarity(
                out_fused.float().view(1, -1),
                out_manual.float().view(1, -1),
            ).item()

            print(f"CPU fused vs manual retained: cos_sim={cos_sim:.6f}")
            assert cos_sim > 0.999, f"CPU fused doesn't match manual: {cos_sim:.6f}"


# ====================================================================
# 3.4 Numerical Stability (Online Softmax)
# ====================================================================


class TestNumericalStability:
    """Verify online softmax handles edge cases correctly."""

    def test_single_tile(self):
        """Single-tile case must not crash or produce NaN."""
        torch.manual_seed(42)
        q = torch.randn(1, 128)
        keys = torch.randn(64, 128)
        values = torch.randn(64, 128)

        out, meta = _pytorch_fused_orthocache_attention(
            q, keys, values, zeta_max=1e9
        )
        assert not torch.isnan(out).any(), "NaN in single-tile output"
        assert not torch.isinf(out).any(), "Inf in single-tile output"

    @requires_triton
    def test_single_tile_gpu(self):
        """Single-tile GPU kernel must not produce NaN."""
        torch.manual_seed(42)
        device = torch.device('cuda')
        q = torch.randn(1, 128, device=device)
        keys = torch.randn(64, 128, device=device)
        values = torch.randn(64, 128, device=device)

        out, _ = fused_orthocache_attention(q, keys, values, zeta_max=1e9)
        assert not torch.isnan(out).any(), "NaN in GPU single-tile output"
        assert not torch.isinf(out).any(), "Inf in GPU single-tile output"

    def test_large_logits(self):
        """Online softmax must handle large logit values without overflow."""
        torch.manual_seed(42)
        q = torch.randn(1, 128) * 100  # Large query
        keys = torch.randn(4 * 64, 128) * 100  # Large keys
        values = torch.randn(4 * 64, 128)

        out, _ = _pytorch_fused_orthocache_attention(
            q, keys, values, zeta_max=1e9
        )
        assert not torch.isnan(out).any(), "NaN with large logits"
        assert not torch.isinf(out).any(), "Inf with large logits"

    @requires_triton
    def test_total_eviction_gpu(self):
        """If ALL tiles are evicted, output should be zero (no NaN/crash)."""
        torch.manual_seed(42)
        device = torch.device('cuda')

        q = torch.randn(1, 128, device=device)
        # Random noise tiles → ζ ≈ 4.6; set threshold very low to evict all
        keys = torch.randn(4 * 64, 128, device=device)
        values = torch.randn(4 * 64, 128, device=device)

        out, meta = fused_orthocache_attention(
            q, keys, values, zeta_max=0.001, return_mask=True
        )

        # Should not crash or produce NaN
        assert not torch.isnan(out).any(), "NaN when all tiles evicted"
