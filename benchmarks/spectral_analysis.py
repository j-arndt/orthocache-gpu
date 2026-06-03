"""OrthoCache GPU Spectral Analysis Benchmark.

Analyzes spectral properties of synthetic KV-cache blocks:
  - Generates structured (low-rank + noise) and random blocks
  - Computes ζ (spectral decay ratio) distributions
  - Measures separation between semantic and noise blocks
  - Shows multi-band energy decomposition

All computations use the real FWHT pipeline from orthocache_gpu.

Outputs
-------
* Spectral analysis results → benchmarks/results/spectral_analysis_results.json
* Summary statistics on stdout

Usage
-----
    python benchmarks/spectral_analysis.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orthocache_gpu.spectral_energy import (
    compute_spectral_bands,
    compute_spectral_decay_ratio,
    compute_block_energy,
)
from orthocache_gpu.fwht import fwht_512


# ---------------------------------------------------------------------------
# Block generators
# ---------------------------------------------------------------------------

BLOCK_SIZE = 512
HEAD_DIM = 128
NUM_HEADS = 8


def generate_semantic_block(
    rank: int = 8,
    noise_scale: float = 0.05,
    seed: int = 42,
) -> torch.Tensor:
    """Generate a structured (low-rank + small noise) KV-cache block.

    Semantic blocks have most energy concentrated in low-frequency
    Walsh-Hadamard coefficients, resulting in low ζ values.

    Args:
        rank: Rank of the low-rank component.
        noise_scale: Scale of additive Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        Tensor of shape (BLOCK_SIZE, NUM_HEADS, HEAD_DIM).
    """
    torch.manual_seed(seed)
    # Low-rank structure: smooth, slowly varying patterns
    U = torch.randn(BLOCK_SIZE, rank)
    V = torch.randn(rank, HEAD_DIM)
    base = U @ V  # (BLOCK_SIZE, HEAD_DIM)

    # Apply smooth windowing to concentrate energy in low sequency
    window = torch.linspace(0, 1, BLOCK_SIZE).unsqueeze(1)
    base = base * (0.5 + 0.5 * torch.cos(2 * np.pi * window))

    # Add small noise
    noise = torch.randn(BLOCK_SIZE, HEAD_DIM) * noise_scale

    # Expand to multi-head
    block = (base + noise).unsqueeze(1).expand(-1, NUM_HEADS, -1).contiguous()

    # Per-head perturbation to break symmetry
    for h in range(NUM_HEADS):
        block[:, h, :] += torch.randn(BLOCK_SIZE, HEAD_DIM) * noise_scale * 0.5

    return block


def generate_noise_block(
    noise_scale: float = 1.0,
    seed: int = 123,
) -> torch.Tensor:
    """Generate a pure noise KV-cache block.

    Noise blocks have energy spread across all frequency bands,
    resulting in high ζ values (high-frequency dominated).

    Args:
        noise_scale: Scale of the noise.
        seed: Random seed.

    Returns:
        Tensor of shape (BLOCK_SIZE, NUM_HEADS, HEAD_DIM).
    """
    torch.manual_seed(seed)
    return torch.randn(BLOCK_SIZE, NUM_HEADS, HEAD_DIM) * noise_scale


def generate_mixed_cache(
    num_semantic: int = 32,
    num_noise: int = 32,
) -> tuple[torch.Tensor, list[str]]:
    """Generate a mixed KV-cache with semantic and noise blocks.

    Args:
        num_semantic: Number of structured blocks.
        num_noise: Number of noise blocks.

    Returns:
        Tuple of (keys, labels) where keys is (total_seq_len, NUM_HEADS, HEAD_DIM)
        and labels is a list of 'semantic' or 'noise' per block.
    """
    blocks = []
    labels = []

    for i in range(num_semantic):
        blocks.append(generate_semantic_block(
            rank=max(2, 8 - i % 6),  # Vary rank for diversity
            noise_scale=0.02 + 0.01 * (i % 5),
            seed=42 + i,
        ))
        labels.append("semantic")

    for i in range(num_noise):
        blocks.append(generate_noise_block(
            noise_scale=0.8 + 0.4 * (i % 3),
            seed=123 + i,
        ))
        labels.append("noise")

    # Shuffle to interleave
    torch.manual_seed(999)
    perm = torch.randperm(len(blocks))
    blocks = [blocks[i] for i in perm]
    labels = [labels[i] for i in perm]

    keys = torch.cat(blocks, dim=0)  # (total_seq_len, NUM_HEADS, HEAD_DIM)
    return keys, labels


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def analyze_spectral_properties(
    keys: torch.Tensor,
    labels: list[str],
) -> dict:
    """Run comprehensive spectral analysis on the mixed KV-cache.

    Computes:
      - Per-block ζ (spectral decay ratio) distribution
      - Per-block band energy decomposition (DC, Low, Mid, High)
      - Separation metrics between semantic and noise blocks
      - ROC-like separability analysis

    Args:
        keys: (seq_len, NUM_HEADS, HEAD_DIM) tensor.
        labels: Per-block labels ('semantic' or 'noise').

    Returns:
        Dict with all analysis results.
    """
    num_blocks = len(labels)
    results: dict = {
        "num_blocks": num_blocks,
        "block_size": BLOCK_SIZE,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
    }

    # --- Spectral Decay Ratio ζ ---
    print("  Computing spectral decay ratio (ζ)…", flush=True)
    t0 = time.perf_counter()
    zeta = compute_spectral_decay_ratio(keys, BLOCK_SIZE)  # (num_blocks, num_heads)
    t_zeta = (time.perf_counter() - t0) * 1000
    print(f"    Done in {t_zeta:.1f} ms")

    # Average over heads for per-block score
    zeta_mean = torch.mean(zeta, dim=1).numpy()  # (num_blocks,)

    semantic_zeta = [float(zeta_mean[i]) for i, l in enumerate(labels) if l == "semantic"]
    noise_zeta = [float(zeta_mean[i]) for i, l in enumerate(labels) if l == "noise"]

    results["zeta"] = {
        "semantic": {
            "values": semantic_zeta,
            "mean": float(np.mean(semantic_zeta)),
            "std": float(np.std(semantic_zeta)),
            "min": float(np.min(semantic_zeta)),
            "max": float(np.max(semantic_zeta)),
            "median": float(np.median(semantic_zeta)),
        },
        "noise": {
            "values": noise_zeta,
            "mean": float(np.mean(noise_zeta)),
            "std": float(np.std(noise_zeta)),
            "min": float(np.min(noise_zeta)),
            "max": float(np.max(noise_zeta)),
            "median": float(np.median(noise_zeta)),
        },
        "computation_ms": t_zeta,
    }

    # Separability: find optimal ζ_max threshold
    all_zeta = np.concatenate([semantic_zeta, noise_zeta])
    all_labels_binary = np.array([0] * len(semantic_zeta) + [1] * len(noise_zeta))

    best_acc = 0.0
    best_threshold = 0.0
    for thresh in np.linspace(np.min(all_zeta), np.max(all_zeta), 200):
        predicted_noise = (all_zeta > thresh).astype(int)
        acc = np.mean(predicted_noise == all_labels_binary)
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(thresh)

    results["separability"] = {
        "optimal_zeta_max": best_threshold,
        "classification_accuracy": float(best_acc),
        "semantic_below_threshold": float(np.mean(np.array(semantic_zeta) <= best_threshold)),
        "noise_above_threshold": float(np.mean(np.array(noise_zeta) > best_threshold)),
    }

    # --- Multi-Band Energy Decomposition ---
    print("  Computing multi-band energy decomposition…", flush=True)
    t0 = time.perf_counter()
    dc, low_e, mid_e, high_e = compute_spectral_bands(keys, BLOCK_SIZE)
    t_bands = (time.perf_counter() - t0) * 1000
    print(f"    Done in {t_bands:.1f} ms")

    # Average over heads
    dc_energy = torch.sum(dc ** 2, dim=2).mean(dim=1).numpy()  # (num_blocks,)
    low_energy = low_e.mean(dim=1).numpy()
    mid_energy = mid_e.mean(dim=1).numpy()
    high_energy = high_e.mean(dim=1).numpy()

    total_energy = dc_energy + low_energy + mid_energy + high_energy
    total_energy = np.maximum(total_energy, 1e-10)

    results["band_energy"] = {
        "dc_fraction": (dc_energy / total_energy).tolist(),
        "low_fraction": (low_energy / total_energy).tolist(),
        "mid_fraction": (mid_energy / total_energy).tolist(),
        "high_fraction": (high_energy / total_energy).tolist(),
        "zeta_per_block": zeta_mean.tolist(),
        "labels": labels,
        "computation_ms": t_bands,
    }

    # --- Block Energy Distribution ---
    print("  Computing block energy distribution…", flush=True)
    energy = compute_block_energy(keys, BLOCK_SIZE)  # (num_blocks, num_heads)
    energy_mean = torch.mean(energy, dim=1).numpy()

    semantic_energy = [float(energy_mean[i]) for i, l in enumerate(labels) if l == "semantic"]
    noise_energy = [float(energy_mean[i]) for i, l in enumerate(labels) if l == "noise"]

    results["block_energy"] = {
        "semantic_mean": float(np.mean(semantic_energy)),
        "noise_mean": float(np.mean(noise_energy)),
        "semantic_std": float(np.std(semantic_energy)),
        "noise_std": float(np.std(noise_energy)),
    }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_spectral_analysis() -> dict:
    """Run the complete spectral analysis benchmark."""
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  OrthoCache GPU Spectral Analysis Benchmark")
    print("=" * 70)
    print()

    # Generate mixed KV-cache
    print("[1/2] Generating mixed KV-cache (32 semantic + 32 noise blocks)…")
    keys, labels = generate_mixed_cache(num_semantic=32, num_noise=32)
    total_tokens = keys.shape[0]
    print(f"  Total tokens: {total_tokens} ({len(labels)} blocks × {BLOCK_SIZE} tokens)")
    print()

    # Run analysis
    print("[2/2] Running spectral analysis…")
    results = analyze_spectral_properties(keys, labels)
    print()

    # Save results
    json_path = output_dir / "spectral_analysis_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("--- Summary ---")
    z = results["zeta"]
    print(f"  Semantic blocks: ζ = {z['semantic']['mean']:.4f} ± {z['semantic']['std']:.4f}")
    print(f"  Noise blocks:    ζ = {z['noise']['mean']:.4f} ± {z['noise']['std']:.4f}")
    sep = results["separability"]
    print(f"  Optimal ζ_max threshold: {sep['optimal_zeta_max']:.4f}")
    print(f"  Classification accuracy: {sep['classification_accuracy']:.1%}")
    print(f"  Semantic below threshold: {sep['semantic_below_threshold']:.1%}")
    print(f"  Noise above threshold:    {sep['noise_above_threshold']:.1%}")
    print()
    print(f"Results written to {json_path.resolve()}")

    return results


if __name__ == "__main__":
    run_spectral_analysis()
