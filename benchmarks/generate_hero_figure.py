#!/usr/bin/env python3
"""Generate the hero figure for OrthoCache GPU README.

Tells the honest story:
  - Split-K OrthoCache crosses over dense at ~4K tokens
  - At 32K, 1.28× faster while using 50% less KV-cache memory
  - Sub-linear scaling: eviction overhead becomes free at long contexts

Uses the multi-head benchmark data (32 heads, 50% eviction).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Fix Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#e6edf3',
    'text.color': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.6,
    'font.family': 'sans-serif',
    'font.size': 12,
})

COLOR_DENSE = '#f85149'    # GitHub red
COLOR_SPLITK = '#58a6ff'   # GitHub blue
COLOR_MEMORY = '#3fb950'   # GitHub green
COLOR_CROSSOVER = '#d2a8ff'  # GitHub purple


def load_data():
    data_path = Path(__file__).resolve().parent / "results" / "multihead_benchmark_results.json"
    with open(data_path) as f:
        return json.load(f)


def generate_hero_figure():
    """Single clean figure: latency crossover with memory savings callout."""
    data = load_data()

    seq_lens = [d["seq_len"] for d in data]
    dense_ms = [d["dense_mean_ms"] for d in data]
    dense_std = [d["dense_std_ms"] for d in data]
    splitk_ms = [d["splitk_mean_ms"] for d in data]
    splitk_std = [d["splitk_std_ms"] for d in data]
    speedups = [d["speedup"] for d in data]

    dense_arr = np.array(dense_ms)
    dense_std_arr = np.array(dense_std)
    splitk_arr = np.array(splitk_ms)
    splitk_std_arr = np.array(splitk_std)
    seq_arr = np.array(seq_lens)

    gpu_name = data[0].get("gpu_name", "RTX 4060")
    num_heads = data[0].get("num_heads", 32)
    eviction_rate = data[0].get("eviction_rate", 0.50)

    fig, ax = plt.subplots(figsize=(14, 8))

    # --- Dense attention line ---
    ax.plot(seq_lens, dense_ms, marker='o', linestyle='-',
            color=COLOR_DENSE, linewidth=2.8, markersize=10,
            label='Dense Attention (no eviction)',
            zorder=5, markeredgecolor='white', markeredgewidth=1.2)
    ax.fill_between(seq_lens,
                    dense_arr - dense_std_arr,
                    dense_arr + dense_std_arr,
                    color=COLOR_DENSE, alpha=0.12, zorder=2)

    # --- Split-K OrthoCache line ---
    ax.plot(seq_lens, splitk_ms, marker='^', linestyle='-',
            color=COLOR_SPLITK, linewidth=2.8, markersize=10,
            label=f'Split-K OrthoCache ({eviction_rate:.0%} eviction)',
            zorder=5, markeredgecolor='white', markeredgewidth=1.2)
    ax.fill_between(seq_lens,
                    splitk_arr - splitk_std_arr,
                    splitk_arr + splitk_std_arr,
                    color=COLOR_SPLITK, alpha=0.12, zorder=2)

    # --- Crossover marker ---
    # Find the first seq_len where Split-K beats Dense
    crossover_idx = None
    for i in range(len(speedups)):
        if speedups[i] >= 1.0:
            crossover_idx = i
            break

    if crossover_idx is not None:
        cx = seq_lens[crossover_idx]
        cy = splitk_ms[crossover_idx]
        ax.axvline(x=cx, color=COLOR_CROSSOVER, linestyle=':', linewidth=1.5, alpha=0.6, zorder=1)
        ax.annotate(
            f'Crossover at {cx:,} tokens',
            xy=(cx, cy), xytext=(cx * 0.35, cy * 1.8),
            fontsize=11, fontweight='bold', color=COLOR_CROSSOVER,
            arrowprops=dict(arrowstyle='->', color=COLOR_CROSSOVER, lw=1.5, alpha=0.7),
            zorder=10,
        )

    # --- Speedup annotation at 32K ---
    max_idx = len(seq_lens) - 1
    max_sl = seq_lens[max_idx]
    max_speedup = speedups[max_idx]
    ax.annotate(
        f'{max_speedup:.2f}× faster\n+ 50% less KV memory',
        xy=(max_sl, splitk_ms[max_idx]),
        xytext=(max_sl * 0.32, splitk_ms[max_idx] * 0.55),
        fontsize=13, fontweight='bold', color=COLOR_SPLITK,
        arrowprops=dict(arrowstyle='->', color=COLOR_SPLITK, lw=2, alpha=0.8),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#161b22',
                  edgecolor=COLOR_SPLITK, alpha=0.9, linewidth=1.5),
        zorder=10,
    )

    # --- Memory savings band annotation ---
    # Shade the gap between the lines where Split-K wins
    for i in range(len(seq_lens) - 1):
        if speedups[i] >= 1.0 and speedups[i + 1] >= 1.0:
            ax.fill_between(
                [seq_lens[i], seq_lens[i + 1]],
                [dense_ms[i], dense_ms[i + 1]],
                [splitk_ms[i], splitk_ms[i + 1]],
                color=COLOR_MEMORY, alpha=0.08, zorder=1,
            )

    # --- Axes ---
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.set_xlabel('Context Length (tokens)', fontsize=13, labelpad=10)
    ax.set_ylabel('Latency (ms) — lower is better', fontsize=13, labelpad=10)
    ax.grid(True, alpha=0.3)

    # --- Title ---
    ax.set_title(
        f'OrthoCache: Fused Spectral Eviction + Attention\n'
        f'{gpu_name} · {num_heads} heads · {eviction_rate:.0%} eviction · Lean 4 verified',
        fontsize=15, fontweight='bold', pad=18, linespacing=1.4,
    )

    # --- Legend ---
    ax.legend(loc='upper left', framealpha=0.8, facecolor='#161b22',
              edgecolor='#30363d', fontsize=11.5)

    # --- Watermark ---
    fig.text(0.98, 0.02, 'OrthoCache GPU v0.1.0',
             fontsize=8, color='#484f58', ha='right', alpha=0.6)

    # --- Hardware info ---
    fig.text(0.98, 0.96,
             f'CUDA Events timing · {data[0].get("num_heads", 32)}h × 128d · '
             f'25 iters · SM 8.9',
             fontsize=8, color='#484f58', ha='right', alpha=0.5)

    plt.tight_layout()

    # Save
    out_dir = Path(__file__).resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fmt in ['png', 'svg']:
        path = out_dir / f"hero_multihead.{fmt}"
        dpi = 300 if fmt == 'png' else None
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)

    print(f"  → hero_multihead.png + hero_multihead.svg")


def generate_speedup_bar():
    """Clean bar chart showing speedup at each sequence length."""
    data = load_data()

    seq_lens = [d["seq_len"] for d in data]
    speedups = [d["speedup"] for d in data]
    labels = [f'{s//1024}K' for s in seq_lens]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [COLOR_SPLITK if s >= 1.0 else '#f8514960' for s in speedups]
    edge_colors = [COLOR_SPLITK if s >= 1.0 else COLOR_DENSE for s in speedups]

    bars = ax.bar(labels, speedups, color=colors, edgecolor=edge_colors,
                  linewidth=1.5, width=0.6, zorder=3)

    # Baseline line at 1.0×
    ax.axhline(y=1.0, color=COLOR_DENSE, linestyle='--', linewidth=1.5,
               alpha=0.6, label='Break-even (1.0×)', zorder=2)

    # Value labels on bars
    for bar, spd in zip(bars, speedups):
        va = 'bottom' if spd >= 1.0 else 'top'
        offset = 0.03 if spd >= 1.0 else -0.03
        ax.text(bar.get_x() + bar.get_width() / 2, spd + offset,
                f'{spd:.2f}×', ha='center', va=va,
                fontsize=12, fontweight='bold',
                color='white')

    ax.set_xlabel('Context Length', fontsize=13, labelpad=10)
    ax.set_ylabel('Speedup vs Dense Attention', fontsize=13, labelpad=10)
    ax.set_title(
        'Split-K OrthoCache Speedup (32 heads, 50% eviction)',
        fontsize=14, fontweight='bold', pad=15,
    )
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.7, facecolor='#161b22',
              edgecolor='#30363d', fontsize=11)

    # Set y-axis to include some space
    ax.set_ylim(0, max(speedups) * 1.25)

    # Annotation for memory savings
    ax.text(0.97, 0.85,
            '+ 50% KV-cache\n   memory savings',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            color=COLOR_MEMORY, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#161b22',
                      edgecolor=COLOR_MEMORY, alpha=0.9, linewidth=1.5))

    plt.tight_layout()

    out_dir = Path(__file__).resolve().parent / "plots"
    for fmt in ['png', 'svg']:
        path = out_dir / f"hero_speedup_bars.{fmt}"
        dpi = 300 if fmt == 'png' else None
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)

    print(f"  → hero_speedup_bars.png + hero_speedup_bars.svg")


def main():
    print("=" * 60)
    print("  OrthoCache Hero Figure Generator")
    print("=" * 60)
    generate_hero_figure()
    generate_speedup_bar()
    print("  [DONE]")
    print()


if __name__ == "__main__":
    main()
