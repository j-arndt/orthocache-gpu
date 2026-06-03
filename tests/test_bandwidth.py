"""Tests for orthocache_gpu.bandwidth_model (GPU Edition).

Validates the pure-analytical interconnect bandwidth model:
1. ici_bytes_per_step — formula correctness and edge cases.
2. ici_bandwidth_table — table generation with correct structure.
3. model_configs — pre-defined configs have required keys (including H100/B200).
4. _validate_reference_70b — internal sanity check.
5. Invalid inputs — out-of-range sparsity raises ValueError.
"""

import pytest
import numpy as np

from orthocache_gpu.bandwidth_model import (
    ici_bytes_per_step,
    ici_bandwidth_table,
    model_configs,
    _validate_reference_70b,
    _BYTES_PER_GB,
    _BYTES_PER_MB,
)


# ── ici_bytes_per_step ──────────────────────────────────────────────────────


class TestICIBytesPerStep:
    """Tests for the core interconnect transfer volume computation."""

    def test_zero_sparsity_no_savings(self):
        """At 0% sparsity, sparse_bytes == dense_bytes and savings == 0."""
        r = ici_bytes_per_step(
            num_layers=10, seq_len=1024, head_dim=64,
            num_kv_heads=4, num_devices=2, sparsity=0.0,
        )
        assert r['dense_bytes'] == r['sparse_bytes']
        assert r['savings_bytes'] == 0
        assert r['savings_pct'] == 0.0

    def test_full_sparsity_zero_transfer(self):
        """At 100% sparsity, sparse_bytes should be 0."""
        r = ici_bytes_per_step(
            num_layers=10, seq_len=1024, head_dim=64,
            num_kv_heads=4, num_devices=2, sparsity=1.0,
        )
        assert r['sparse_bytes'] == 0
        assert r['savings_pct'] == 100.0

    def test_50pct_sparsity_halves_transfer(self):
        """At 50% sparsity, sparse_bytes should be half of dense_bytes."""
        r = ici_bytes_per_step(
            num_layers=20, seq_len=2048, head_dim=128,
            num_kv_heads=8, num_devices=4, sparsity=0.5,
        )
        assert r['sparse_bytes'] == r['dense_bytes'] // 2
        assert r['savings_pct'] == pytest.approx(50.0)

    def test_formula_matches_manual(self):
        """Verify the formula: dense = L * S * d_k * H_kv * dtype / P²."""
        L, S, dk, Hkv, P = 4, 512, 32, 2, 2
        expected_dense = L * S * dk * Hkv * 2 // (P * P)
        r = ici_bytes_per_step(
            num_layers=L, seq_len=S, head_dim=dk,
            num_kv_heads=Hkv, num_devices=P, sparsity=0.0,
        )
        assert r['dense_bytes'] == expected_dense

    def test_savings_gb_per_1000(self):
        """Cumulative GB savings should be savings × 1000 / 1e9."""
        r = ici_bytes_per_step(
            num_layers=10, seq_len=4096, head_dim=128,
            num_kv_heads=8, num_devices=4, sparsity=0.5,
        )
        expected = (r['savings_bytes'] * 1000) / _BYTES_PER_GB
        assert r['savings_gb_per_1000_steps'] == pytest.approx(expected, rel=1e-3)


# ── Invalid inputs ──────────────────────────────────────────────────────────


class TestInvalidInputs:
    """Tests for error handling on bad parameters."""

    def test_negative_sparsity_raises(self):
        """Negative sparsity should raise ValueError."""
        with pytest.raises(ValueError, match="sparsity must be in"):
            ici_bytes_per_step(
                num_layers=1, seq_len=512, head_dim=64,
                num_kv_heads=1, num_devices=1, sparsity=-0.1,
            )

    def test_sparsity_above_one_raises(self):
        """Sparsity > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="sparsity must be in"):
            ici_bytes_per_step(
                num_layers=1, seq_len=512, head_dim=64,
                num_kv_heads=1, num_devices=1, sparsity=1.5,
            )


# ── ici_bandwidth_table ─────────────────────────────────────────────────────


class TestBandwidthTable:
    """Tests for the bandwidth table generator."""

    def test_default_sparsity_levels(self):
        """Default call should produce 7 rows (one per default sparsity)."""
        table = ici_bandwidth_table(
            num_layers=10, seq_len=1024, head_dim=64,
            num_kv_heads=4, num_devices=2,
        )
        assert len(table) == 7

    def test_custom_sparsity_levels(self):
        """Custom sparsity levels should produce correct number of rows."""
        table = ici_bandwidth_table(
            num_layers=10, seq_len=1024, head_dim=64,
            num_kv_heads=4, num_devices=2,
            sparsity_levels=[0.0, 0.5, 0.9],
        )
        assert len(table) == 3
        assert table[0]['sparsity'] == 0.0
        assert table[1]['sparsity'] == 0.5
        assert table[2]['sparsity'] == 0.9

    def test_row_keys(self):
        """Each row should have the expected keys."""
        table = ici_bandwidth_table(
            num_layers=10, seq_len=1024, head_dim=64,
            num_kv_heads=4, num_devices=2,
            sparsity_levels=[0.5],
        )
        row = table[0]
        assert 'sparsity' in row
        assert 'dense_mb' in row
        assert 'sparse_mb' in row
        assert 'savings_mb' in row
        assert 'savings_pct' in row

    def test_dense_mb_consistent_across_rows(self):
        """Dense MB should be the same in every row (independent of sparsity)."""
        table = ici_bandwidth_table(
            num_layers=10, seq_len=1024, head_dim=64,
            num_kv_heads=4, num_devices=2,
        )
        dense_values = [row['dense_mb'] for row in table]
        assert all(v == dense_values[0] for v in dense_values)


# ── model_configs ───────────────────────────────────────────────────────────


class TestModelConfigs:
    """Tests for the pre-defined model configuration registry."""

    def test_known_models_present(self):
        """Expected model names should exist (including GPU variants)."""
        configs = model_configs()
        assert 'llama3_70b' in configs
        assert 'gemma4_31b' in configs
        assert 'llama3_405b' in configs

    def test_gpu_models_present(self):
        """H100 and B200 GPU configurations should be present."""
        configs = model_configs()
        assert 'llama3_70b_h100' in configs
        assert 'llama3_405b_h100' in configs
        assert 'llama3_70b_b200' in configs
        assert 'llama3_405b_b200' in configs

    def test_required_keys(self):
        """Each config should have the required architecture keys."""
        required = {'num_layers', 'num_heads', 'num_kv_heads', 'head_dim',
                     'num_devices', 'label'}
        for name, cfg in model_configs().items():
            assert required.issubset(cfg.keys()), f"{name} missing keys"

    def test_gpu_configs_have_hw_specs(self):
        """GPU configs should include HBM capacity and interconnect bandwidth."""
        configs = model_configs()
        for name in ['llama3_70b_h100', 'llama3_70b_b200']:
            cfg = configs[name]
            assert 'hbm_capacity_gb' in cfg, f"{name} missing hbm_capacity_gb"
            assert 'interconnect_bw_gbs' in cfg, f"{name} missing interconnect_bw_gbs"
            assert cfg['interconnect'] == 'NVLink'


# ── _validate_reference_70b ─────────────────────────────────────────────────


class TestValidateReference:
    """Test the internal validation against the CBA reference value."""

    def test_reference_passes(self):
        """Internal 70B validation should not raise."""
        _validate_reference_70b()  # raises AssertionError if broken
