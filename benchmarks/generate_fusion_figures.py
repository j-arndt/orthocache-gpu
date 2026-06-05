#!/usr/bin/env python3
"""OrthoCache GPU — Fused God Kernel Publication-Quality Figure Generator.

Reads fusion benchmark JSON results and generates stunning dark-themed matplotlib
figures for the technical report / TechRxiv preprint.

Data sources:
  - benchmarks/results/fusion_profiling_results.json

Output:
  - benchmarks/plots/fusion_crossover.{png,svg}
  - benchmarks/plots/fusion_dram_traffic.{png,svg}
  - benchmarks/plots/fusion_sram_utilization.{png,svg}
  - benchmarks/plots/fusion_speedup_heatmap.{png,svg}

Usage:
    python benchmarks/generate_fusion_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Global Style Configuration ──────────────────────────────────────────────

plt.style.use('dark_background')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linestyle': '-',
    'grid.color': '#555555',
    'axes.facecolor': '#1a1a2e',
    'figure.facecolor': '#0f0f1a',
    'savefig.facecolor': '#0f0f1a',
    'axes.edgecolor': '#444444',
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#cccccc',
    'ytick.color': '#cccccc',
})

# ─── Color Palette ───────────────────────────────────────────────────────────

COLORS = {
    'dense': '#FF6B6B',      # Coral red
    'unfused': '#FFD93D',    # Gold
    'fused': '#4ECDC4',      # Teal (hero color)
    'evict_25': '#96CEB4',   # Mint
    'evict_50': '#4ECDC4',   # Teal
    'evict_75': '#45B7D1',   # Sky blue
}

WATERMARK = 'OrthoCache GPU v0.1.0'

# ─── Constants ────────────────────────────────────────────────────────────────

TILE_SIZE = 64
HEAD_DIM = 128


def add_watermark(fig):
    """Add subtle watermark to bottom-right corner."""
    fig.text(0.98, 0.02, WATERMARK, ha='right', va='bottom',
             fontsize=8, alpha=0.3, color='#888888',
             fontstyle='italic')


def save_figure(fig, name: str):
    """Save figure as both PNG and SVG."""
    png_path = OUTPUT_DIR / f"{name}.png"
    svg_path = OUTPUT_DIR / f"{name}.svg"
    fig.savefig(png_path, format='png')
    fig.savefig(svg_path, format='svg')
    print(f"    → {png_path.name} + {svg_path.name}")
    plt.close(fig)


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_fusion_data() -> list[dict]:
    """Load fusion profiling results."""
    path = RESULTS_DIR / "fusion_profiling_results.json"
    with open(path, 'r') as f:
        return json.load(f)


# ─── Figure 1: Crossover Point Plot ──────────────────────────────────────────

def generate_fig1_crossover():
    """Line plot: Dense vs Unfused vs Fused at 50% eviction rate.

    Log-log scale (x: base 2, y: standard log).
    Shaded std-dev error bands, annotated speedup at largest sequence length.
    """
    print("  [1/4] Fusion Crossover Point…")
    data = load_fusion_data()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Filter to 50% eviction rate (dense has no eviction, include all)
    config_map = {
        "Dense": {
            "color": COLORS['dense'], "marker": "o", "ls": "-",
            "filter": lambda d: d["mode"] == "dense",
        },
        "Unfused OrthoCache": {
            "color": COLORS['unfused'], "marker": "s", "ls": "-",
            "filter": lambda d: d["mode"] == "unfused" and abs(d["eviction_rate"] - 0.50) < 0.01,
        },
        "Fused OrthoCache": {
            "color": COLORS['fused'], "marker": "D", "ls": "-",
            "filter": lambda d: d["mode"] == "fused" and abs(d["eviction_rate"] - 0.50) < 0.01,
        },
    }

    max_sl = 0
    plotted_data = {}

    for label, style in config_map.items():
        entries = sorted(
            [d for d in data if style["filter"](d)],
            key=lambda d: d["seq_len"],
        )
        if not entries:
            continue

        seq_lens = [d["seq_len"] for d in entries]
        means = [d["mean_ms"] for d in entries]
        stds = [d["std_ms"] for d in entries]

        means_arr = np.array(means)
        stds_arr = np.array(stds)

        max_sl = max(max_sl, max(seq_lens))
        plotted_data[label] = {s: m for s, m in zip(seq_lens, means)}

        # Main line
        ax.plot(seq_lens, means, marker=style["marker"], linestyle=style["ls"],
                color=style["color"], linewidth=2.5, markersize=9,
                label=label, zorder=5, markeredgecolor='white', markeredgewidth=1)

        # Shaded error region
        ax.fill_between(seq_lens, means_arr - stds_arr, means_arr + stds_arr,
                        color=style["color"], alpha=0.15, zorder=2)

    # Log-log scale
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Annotate speedup ratio (fused/dense) at the largest sequence length
    if max_sl and "Dense" in plotted_data and "Fused OrthoCache" in plotted_data:
        dense_ms = plotted_data["Dense"].get(max_sl)
        fused_ms = plotted_data["Fused OrthoCache"].get(max_sl)
        if dense_ms and fused_ms:
            speedup = dense_ms / fused_ms
            ax.annotate(
                f'{speedup:.1f}× speedup',
                xy=(max_sl, fused_ms), xytext=(max_sl * 1.3, fused_ms * 0.6),
                fontsize=12, fontweight='bold',
                color=COLORS['fused'],
                arrowprops=dict(arrowstyle='->', color=COLORS['fused'],
                                lw=1.5, alpha=0.7),
            )

    ax.set_xlabel('Context Length (tokens)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('OrthoCache God Kernel: Crossover Point Analysis',
                 fontsize=16, fontweight='bold', pad=15)

    ax.legend(loc='upper left', framealpha=0.7, facecolor='#1a1a2e',
              edgecolor='#444444', fontsize=11)

    add_watermark(fig)
    save_figure(fig, 'fusion_crossover')


# ─── Figure 2: DRAM Traffic Comparison ────────────────────────────────────────

def generate_fig2_dram_traffic():
    """Grouped bar chart: DRAM traffic for Unfused vs Fused.

    3 sequence lengths (4096, 8192, 32768) at 50% eviction.
    DRAM estimates based on tile-based memory access patterns.
    """
    print("  [2/4] DRAM Traffic Comparison…")
    data = load_fusion_data()

    target_seq_lens = [4096, 8192, 32768]
    eviction_rate = 0.50

    fig, ax = plt.subplots(figsize=(12, 8))

    unfused_mb = []
    fused_mb = []
    labels = []

    for sl in target_seq_lens:
        # Find matching entries to extract num_tiles
        unfused_entry = [
            d for d in data
            if d["mode"] == "unfused"
            and d["seq_len"] == sl
            and abs(d["eviction_rate"] - eviction_rate) < 0.01
        ]
        fused_entry = [
            d for d in data
            if d["mode"] == "fused"
            and d["seq_len"] == sl
            and abs(d["eviction_rate"] - eviction_rate) < 0.01
        ]

        # Derive num_tiles from data or calculate
        if unfused_entry:
            num_tiles = unfused_entry[0].get("num_tiles", sl // TILE_SIZE)
        elif fused_entry:
            num_tiles = fused_entry[0].get("num_tiles", sl // TILE_SIZE)
        else:
            num_tiles = sl // TILE_SIZE

        # Unfused DRAM: reads K for FWHT + reads K again for attention + reads V
        # = num_tiles * TILE_SIZE * head_dim * 4 bytes * 3
        unfused_bytes = num_tiles * TILE_SIZE * HEAD_DIM * 4 * 3

        # Fused DRAM: reads K once (retained portion for QK^T), reads V only for
        # retained tiles, plus one full K read for FWHT scoring
        # = num_tiles * TILE_SIZE * head_dim * 4 * 2 * (1 - eviction_rate)
        #   + num_tiles * TILE_SIZE * head_dim * 4
        fused_bytes = (
            num_tiles * TILE_SIZE * HEAD_DIM * 4 * 2 * (1 - eviction_rate)
            + num_tiles * TILE_SIZE * HEAD_DIM * 4
        )

        unfused_mb.append(unfused_bytes / (1024 ** 2))
        fused_mb.append(fused_bytes / (1024 ** 2))
        labels.append(f'{sl:,}')

    x = np.arange(len(target_seq_lens))
    width = 0.35

    # Unfused bars
    bars_unfused = ax.bar(
        x - width / 2, unfused_mb, width,
        color=COLORS['unfused'], alpha=0.85, label='Unfused',
        edgecolor='white', linewidth=0.5,
    )

    # Fused bars
    bars_fused = ax.bar(
        x + width / 2, fused_mb, width,
        color=COLORS['fused'], alpha=0.85, label='Fused (God Kernel)',
        edgecolor='white', linewidth=0.5,
    )

    # Annotate % reduction on each fused bar
    for i in range(len(target_seq_lens)):
        reduction = (1 - fused_mb[i] / unfused_mb[i]) * 100
        ax.text(
            x[i] + width / 2, fused_mb[i] + max(unfused_mb) * 0.02,
            f'−{reduction:.0f}%',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#96CEB4',
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('DRAM Traffic (MB)')
    ax.set_title('DRAM Traffic: Fused vs Unfused (50% Eviction)',
                 fontsize=16, fontweight='bold', pad=15)

    ax.legend(loc='upper left', framealpha=0.7, facecolor='#1a1a2e',
              edgecolor='#444444', fontsize=11)

    add_watermark(fig)
    save_figure(fig, 'fusion_dram_traffic')


# ─── Figure 3: SRAM Utilization ──────────────────────────────────────────────

def generate_fig3_sram_utilization():
    """Stacked horizontal bar chart: SRAM budget breakdown.

    Shows Phase A (Eviction) vs Phase A+B (Peak) with SM SRAM limit.
    """
    print("  [3/4] SRAM Utilization…")

    # SRAM segments (KB)
    k_tile_kb = 32.0    # K_tile: TILE_SIZE × head_dim × 4 bytes / 1024
    w_64_kb = 16.0      # W_64: Hadamard matrix
    v_tile_kb = 32.0    # V_tile: TILE_SIZE × head_dim × 4 bytes / 1024
    q_acc_kb = 1.0      # Q + accumulators

    sm_limit_kb = 100.0  # SM SRAM limit for RTX 4060

    # Phase A (Eviction): K_tile + W_64
    phase_a_segments = {
        'K_tile': k_tile_kb,
        'W_64': w_64_kb,
    }

    # Phase A+B (Peak): K_tile + W_64 + V_tile + Q/acc
    phase_ab_segments = {
        'K_tile': k_tile_kb,
        'W_64': w_64_kb,
        'V_tile': v_tile_kb,
        'Q/acc': q_acc_kb,
    }

    segment_colors = {
        'K_tile': '#4ECDC4',
        'W_64': '#FFD93D',
        'V_tile': '#45B7D1',
        'Q/acc': '#96CEB4',
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    phases = ['Phase A\n(Eviction)', 'Phase A+B\n(Peak)']
    phase_data = [phase_a_segments, phase_ab_segments]
    all_segment_names = ['K_tile', 'W_64', 'V_tile', 'Q/acc']

    y_positions = np.arange(len(phases))

    for seg_name in all_segment_names:
        lefts = []
        widths = []
        for phase_segs in phase_data:
            # Calculate left position: sum of all previous segments
            left = sum(
                phase_segs.get(prev, 0.0)
                for prev in all_segment_names
                if all_segment_names.index(prev) < all_segment_names.index(seg_name)
            )
            w = phase_segs.get(seg_name, 0.0)
            lefts.append(left)
            widths.append(w)

        ax.barh(
            y_positions, widths, left=lefts, height=0.5,
            color=segment_colors[seg_name], alpha=0.85,
            edgecolor='white', linewidth=0.5,
            label=f'{seg_name} ({widths[-1]:.0f} KB)' if widths[-1] > 0 else None,
        )

        # Add text labels inside bars
        for i in range(len(phases)):
            if widths[i] > 2:  # Only label if segment is wide enough
                ax.text(
                    lefts[i] + widths[i] / 2, y_positions[i],
                    f'{widths[i]:.0f}',
                    ha='center', va='center',
                    fontsize=10, fontweight='bold', color='#0f0f1a',
                )

    # SM SRAM limit vertical line
    ax.axvline(sm_limit_kb, color='#FF6B6B', linewidth=2.5, linestyle='--',
               zorder=6, alpha=0.9)
    ax.text(
        sm_limit_kb + 1, len(phases) - 0.3,
        f'SM Limit ({sm_limit_kb:.0f} KB)',
        color='#FF6B6B', fontsize=11, fontweight='bold', va='center',
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(phases)
    ax.set_xlabel('SRAM Usage (KB)')
    ax.set_title('SRAM Budget: Fused God Kernel (RTX 4060, 100 KB/SM)',
                 fontsize=16, fontweight='bold', pad=15)

    # Extend x-axis a bit beyond the limit line
    ax.set_xlim(0, sm_limit_kb * 1.3)

    ax.legend(loc='lower right', framealpha=0.7, facecolor='#1a1a2e',
              edgecolor='#444444', fontsize=10)

    add_watermark(fig)
    save_figure(fig, 'fusion_sram_utilization')


# ─── Figure 4: Speedup Heatmap ───────────────────────────────────────────────

def generate_fig4_speedup_heatmap():
    """Heatmap: eviction_rate × sequence_length → fused/dense speedup ratio.

    Custom colormap from deep purple to bright gold.
    """
    print("  [4/4] Speedup Heatmap…")
    data = load_fusion_data()

    eviction_rates = [0.25, 0.50, 0.75]
    eviction_labels = ['25%', '50%', '75%']
    seq_lens = sorted(set(
        d["seq_len"] for d in data if d["mode"] in ("dense", "fused")
    ))

    # Build speedup matrix
    speedup_matrix = np.zeros((len(eviction_rates), len(seq_lens)))

    for i, er in enumerate(eviction_rates):
        for j, sl in enumerate(seq_lens):
            dense = [
                d for d in data
                if d["mode"] == "dense" and d["seq_len"] == sl
            ]
            fused = [
                d for d in data
                if d["mode"] == "fused"
                and d["seq_len"] == sl
                and abs(d["eviction_rate"] - er) < 0.01
            ]
            if dense and fused:
                speedup_matrix[i, j] = dense[0]["mean_ms"] / fused[0]["mean_ms"]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Custom colormap: deep purple → green → teal → mint → gold → orange
    cmap_colors = ['#2D1B69', '#1A6B4C', '#4ECDC4', '#96CEB4', '#FFD93D', '#FF8C00']
    custom_cmap = LinearSegmentedColormap.from_list('godkernel', cmap_colors, N=256)

    im = ax.imshow(
        speedup_matrix, cmap=custom_cmap, aspect='auto',
        vmin=max(0.8, speedup_matrix.min() * 0.9),
        vmax=speedup_matrix.max() * 1.1,
    )

    # Annotate cells
    for i in range(len(eviction_rates)):
        for j in range(len(seq_lens)):
            val = speedup_matrix[i, j]
            text_color = 'white' if val < 1.5 else 'black'
            ax.text(j, i, f'{val:.2f}×', ha='center', va='center',
                    fontsize=14, fontweight='bold', color=text_color)

    ax.set_xticks(range(len(seq_lens)))
    ax.set_xticklabels([f'{sl:,}' for sl in seq_lens])
    ax.set_yticks(range(len(eviction_labels)))
    ax.set_yticklabels(eviction_labels)
    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Eviction Rate')
    ax.set_title('God Kernel Speedup vs Dense Attention',
                 fontsize=16, fontweight='bold', pad=15)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Speedup Factor (×)', fontsize=11)

    add_watermark(fig)
    save_figure(fig, 'fusion_speedup_heatmap')


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_all_fusion_figures():
    """Generate all 4 fusion publication-quality figures."""
    print("=" * 70)
    print("  OrthoCache GPU — Fused God Kernel Figure Generator")
    print("=" * 70)
    print(f"  Data source:  {RESULTS_DIR}")
    print(f"  Output dir:   {OUTPUT_DIR}")
    print()

    generate_fig1_crossover()
    generate_fig2_dram_traffic()
    generate_fig3_sram_utilization()
    generate_fig4_speedup_heatmap()

    print()
    print(f"  [DONE] All 4 fusion figures saved to: {OUTPUT_DIR}")
    print("  Formats: PNG (300 DPI raster) + SVG (vector)")
    print()

    # List output files
    fusion_pngs = sorted(OUTPUT_DIR.glob("fusion_*.png"))
    fusion_svgs = sorted(OUTPUT_DIR.glob("fusion_*.svg"))
    print(f"  PNG files ({len(fusion_pngs)}):")
    for f in fusion_pngs:
        print(f"    {f.name}")
    print(f"  SVG files ({len(fusion_svgs)}):")
    for f in fusion_svgs:
        print(f"    {f.name}")


if __name__ == "__main__":
    generate_all_fusion_figures()
