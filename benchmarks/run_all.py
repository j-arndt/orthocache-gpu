#!/usr/bin/env python3
"""OrthoCache GPU -- One-Command Benchmark Runner.

Runs all benchmarks in sequence, then generates publication-quality figures.

Usage:
    python benchmarks/run_all.py

Output:
    benchmarks/results/*.json   -- Raw benchmark data
    benchmarks/plots/*.png      -- Publication figures (300 DPI)
    benchmarks/plots/*.svg      -- Publication figures (vector)
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

# Ensure src and project root are on the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def main():
    """Execute all benchmarks and generate figures."""
    print()
    print("=" * 70)
    print("  OrthoCache GPU -- Complete Benchmark Suite")
    print("=" * 70)
    print()

    # Create output directories
    results_dir = Path("benchmarks/results")
    plots_dir = Path("benchmarks/plots")
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    total_t0 = time.perf_counter()

    # Import benchmark modules (after path setup)
    from benchmarks.profiling import run_profiling
    from benchmarks.spectral_analysis import run_spectral_analysis
    from benchmarks.compaction_benchmark import run_compaction_benchmark
    from benchmarks.generate_figures import generate_all_figures

    # --- Phase 1: Profiling ---
    print()
    print("[Phase 1/6] Profiling Benchmark")
    print("-" * 50)
    t0 = time.perf_counter()
    try:
        run_profiling()
        print(f"[OK] Profiling complete ({time.perf_counter() - t0:.1f}s)")
    except Exception as e:
        print(f"[FAIL] Profiling FAILED: {e}")
        raise

    # --- Phase 2: Spectral Analysis ---
    print()
    print("[Phase 2/6] Spectral Analysis Benchmark")
    print("-" * 50)
    t0 = time.perf_counter()
    try:
        run_spectral_analysis()
        print(f"[OK] Spectral analysis complete ({time.perf_counter() - t0:.1f}s)")
    except Exception as e:
        print(f"[FAIL] Spectral analysis FAILED: {e}")
        raise

    # --- Phase 3: Compaction Benchmark ---
    print()
    print("[Phase 3/6] Compaction Benchmark")
    print("-" * 50)
    t0 = time.perf_counter()
    try:
        run_compaction_benchmark()
        print(f"[OK] Compaction benchmark complete ({time.perf_counter() - t0:.1f}s)")
    except Exception as e:
        print(f"[FAIL] Compaction benchmark FAILED: {e}")
        raise

    # --- Phase 4: Reconstruction Error Validation ---
    print()
    print("[Phase 4/6] Reconstruction Error Validation")
    print("-" * 50)
    t0 = time.perf_counter()
    try:
        from benchmarks.reconstruction_error import run_reconstruction_error
        run_reconstruction_error()
        print(f"[OK] Reconstruction error validation complete ({time.perf_counter() - t0:.1f}s)")
    except ImportError:
        print("[SKIP] reconstruction_error module not available")
    except Exception as e:
        print(f"[FAIL] Reconstruction error validation FAILED: {e}")
        raise

    # --- Phase 5: Target Validation ---
    print()
    print("[Phase 5/6] Target Validation")
    print("-" * 50)
    t0 = time.perf_counter()
    try:
        from benchmarks.target_validation import run_target_validation
        run_target_validation()
        print(f"[OK] Target validation complete ({time.perf_counter() - t0:.1f}s)")
    except ImportError:
        print("[SKIP] target_validation module not available")
    except Exception as e:
        print(f"[WARN] Target validation: {e}")

    # --- Phase 6: Figure Generation ---
    print()
    print("[Phase 6/6] Publication Figure Generation")
    print("-" * 50)
    t0 = time.perf_counter()
    try:
        generate_all_figures()
        print(f"[OK] Figure generation complete ({time.perf_counter() - t0:.1f}s)")
    except Exception as e:
        print(f"[FAIL] Figure generation FAILED: {e}")
        raise

    total_time = time.perf_counter() - total_t0

    # --- Summary ---
    print()
    print("=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    print()
    print(f"  Total time: {total_time:.1f}s")
    print()

    # List all generated files
    results_files = sorted(Path("benchmarks/results").glob("*.json"))
    plot_files = sorted(Path("benchmarks/plots").glob("*.*"))

    print(f"  Results ({len(results_files)} files):")
    for f in results_files:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:45s} {size_kb:>8.1f} KB")

    print(f"\n  Plots ({len(plot_files)} files):")
    for f in plot_files:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:45s} {size_kb:>8.1f} KB")

    print()


if __name__ == "__main__":
    main()
