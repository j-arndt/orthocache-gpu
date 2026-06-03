"""Tests for OrthoCache stream compaction (GPU Edition).

Validates:
1. Stream compaction correctness (active blocks gathered at front)
2. stream_decompact is the inverse of stream_compact
3. compact_and_attend end-to-end pipeline
4. Edge cases: 0% eviction, 100% eviction, single block
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from orthocache_gpu.compaction import stream_compact, stream_decompact, compact_and_attend


# ── Test Fixtures ────────────────────────────────────────────────────────────

BLOCK_SIZE = 512
NUM_HEADS = 4
HEAD_DIM = 64  # Use 64 for fast CPU tests


def _make_tensors(num_blocks: int, seq_len_q: int = 4) -> tuple:
    """Create test tensors with specified block count."""
    torch.manual_seed(42)
    seq_len_k = num_blocks * BLOCK_SIZE
    q = torch.randn(seq_len_q, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    keys = torch.randn(seq_len_k, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    values = torch.randn(seq_len_k, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    return q, keys, values


def _make_mask(num_blocks: int, eviction_rate: float = 0.5) -> torch.Tensor:
    """Create a block mask with the target eviction rate."""
    if eviction_rate == 0.5:
        mask = torch.tensor([i % 2 == 0 for i in range(num_blocks)], dtype=torch.bool)
    elif eviction_rate == 0.0:
        mask = torch.ones(num_blocks, dtype=torch.bool)
    elif eviction_rate == 1.0:
        mask = torch.zeros(num_blocks, dtype=torch.bool)
    else:
        torch.manual_seed(99)
        mask = torch.rand(num_blocks) > eviction_rate

    # Broadcast to (num_blocks, num_heads)
    return mask[:, None].expand(num_blocks, NUM_HEADS).contiguous()


# ── C.1: Stream Compaction Correctness ───────────────────────────────────────


class TestStreamCompact:
    """Tests for the stream_compact function."""

    def test_basic_compaction(self):
        """Active blocks should be gathered contiguously at the front."""
        num_blocks = 8
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=0.5)

        compact_k, compact_v, indices, num_active = stream_compact(
            keys, values, mask, BLOCK_SIZE
        )

        # 50% eviction → 4 active blocks
        assert int(num_active) == 4

        # Compact tensors should have correct shape
        assert compact_k.shape == (num_blocks, BLOCK_SIZE, NUM_HEADS, HEAD_DIM)
        assert compact_v.shape == (num_blocks, BLOCK_SIZE, NUM_HEADS, HEAD_DIM)
        assert indices.shape == (num_blocks,)

    def test_active_blocks_are_nonzero(self):
        """First num_active blocks in compact tensor should contain real data."""
        num_blocks = 8
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=0.5)

        compact_k, _, _, num_active = stream_compact(
            keys, values, mask, BLOCK_SIZE
        )

        na = int(num_active)

        # Active blocks should have nonzero energy
        for i in range(na):
            block_energy = torch.sum(compact_k[i] ** 2).item()
            assert block_energy > 0, f"Active block {i} is zero"

        # Inactive blocks should be zeroed
        for i in range(na, num_blocks):
            block_energy = torch.sum(compact_k[i] ** 2).item()
            assert block_energy == 0, f"Inactive block {i} is nonzero"

    def test_compaction_preserves_data(self):
        """Compacted blocks should contain the exact same data as originals."""
        num_blocks = 8
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=0.5)

        compact_k, _, indices, num_active = stream_compact(
            keys, values, mask, BLOCK_SIZE
        )

        keys_blocked = keys.reshape(num_blocks, BLOCK_SIZE, NUM_HEADS, HEAD_DIM)

        na = int(num_active)
        for i in range(na):
            orig_idx = int(indices[i])
            torch.testing.assert_close(
                compact_k[i], keys_blocked[orig_idx],
                rtol=1e-6, atol=1e-6,
                msg=f"Compact block {i} != original block {orig_idx}",
            )

    def test_zero_eviction(self):
        """With 0% eviction, all blocks should be retained."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=0.0)

        _, _, _, num_active = stream_compact(keys, values, mask, BLOCK_SIZE)
        assert int(num_active) == num_blocks

    def test_full_eviction(self):
        """With 100% eviction, no blocks should be retained."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=1.0)

        compact_k, _, _, num_active = stream_compact(keys, values, mask, BLOCK_SIZE)
        assert int(num_active) == 0

        # All blocks should be zero
        total_energy = torch.sum(compact_k ** 2).item()
        assert total_energy == 0.0

    def test_single_block(self):
        """Edge case: single block retained."""
        num_blocks = 1
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=0.0)

        _, _, _, num_active = stream_compact(keys, values, mask, BLOCK_SIZE)
        assert int(num_active) == 1


# ── C.2: Decompaction ────────────────────────────────────────────────────────


class TestStreamDecompact:
    """Tests that stream_decompact reverses stream_compact."""

    def test_roundtrip(self):
        """compact → decompact should recover original data for active blocks."""
        num_blocks = 8
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=0.5)

        compact_k, compact_v, indices, num_active = stream_compact(
            keys, values, mask, BLOCK_SIZE
        )

        # Decompact
        decompacted = stream_decompact(
            compact_k, indices, num_active, num_blocks, BLOCK_SIZE
        )

        keys_blocked = keys.reshape(num_blocks, BLOCK_SIZE, NUM_HEADS, HEAD_DIM)

        # Active blocks should be recovered
        block_active = torch.any(mask, dim=-1)
        for b in range(num_blocks):
            if block_active[b]:
                torch.testing.assert_close(
                    decompacted[b], keys_blocked[b], rtol=1e-5, atol=1e-5,
                    msg=f"Decompacted block {b} differs from original",
                )
            else:
                # Evicted blocks should be zero
                energy = torch.sum(decompacted[b] ** 2).item()
                assert energy == 0.0, f"Evicted block {b} is not zero"


# ── C.3: Compact and Attend ─────────────────────────────────────────────────


class TestCompactAttention:
    """Tests that compacted attention produces correct outputs."""

    def test_compact_at_zero_eviction_matches_dense(self):
        """At 0% eviction, compacted attention should match dense attention."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=0.0)

        # Dense reference
        scale = torch.sqrt(torch.tensor(float(HEAD_DIM)))
        logits = torch.einsum('qhd,khd->qkh', q, keys) / scale
        weights = F.softmax(logits, dim=1)
        dense_out = torch.einsum('qkh,khd->qhd', weights, values)

        # Compacted
        compact_out, meta = compact_and_attend(q, keys, values, mask, BLOCK_SIZE)

        assert float(meta['eviction_rate']) == 0.0

        torch.testing.assert_close(
            compact_out.to(torch.float32),
            dense_out,
            atol=1e-3,
            rtol=1e-3,
            msg="Compacted attention at 0% eviction differs from dense",
        )

    def test_metadata_correct(self):
        """compact_and_attend should return correct metadata."""
        num_blocks = 8
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=0.5)

        _, meta = compact_and_attend(q, keys, values, mask, BLOCK_SIZE)

        assert int(meta['num_active']) == 4
        assert int(meta['num_blocks']) == 8
        assert abs(float(meta['eviction_rate']) - 0.5) < 0.01
