#!/usr/bin/env python3
"""OrthoCache GPU — Publication-Quality Figure Generator.

Reads benchmark JSON results and generates stunning dark-themed matplotlib
figures for the technical report / TechRxiv preprint.

Data sources:
  - benchmarks/results/gpu_profiling_results.json
  - benchmarks/results/spectral_analysis_results.json
  - benchmarks/results/compaction_results.json

Output:
  - benchmarks/plots/latency_vs_seqlen.{png,svg}
  - benchmarks/plots/speedup_heatmap.{png,svg}
  - benchmarks/plots/spectral_separation.{png,svg}
  - benchmarks/plots/band_energy_stacked.{png,svg}
  - benchmarks/plots/compaction_speedup.{png,svg}
  - benchmarks/plots/memory_savings.{png,svg}

Usage:
    python benchmarks/generate_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    'dense': '#FF6B6B',       # Coral
    '50%': '#4ECDC4',         # Teal
    '62.5%': '#45B7D1',       # Sky blue
    '75%': '#96CEB4',         # Mint
    'dense_spectral': '#FFD93D',  # Gold
    'semantic': '#4ECDC4',    # Teal
    'noise': '#FF6B6B',       # Coral
    'dc': '#2C3E50',          # Dark blue-gray
    'low': '#3498DB',         # Blue
    'mid': '#E67E22',         # Orange
    'high': '#E74C3C',        # Red
    'no_evict': '#FF6B6B',    # Coral
    '50_evict': '#4ECDC4',    # Teal
    '75_evict': '#96CEB4',    # Mint
}

WATERMARK = 'OrthoCache GPU v0.1.0'


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

def load_profiling_data() -> list[dict]:
    """Load GPU profiling results."""
    path = RESULTS_DIR / "gpu_profiling_results.json"
    with open(path, 'r') as f:
        return json.load(f)


def load_spectral_data() -> dict:
    """Load spectral analysis results."""
    path = RESULTS_DIR / "spectral_analysis_results.json"
    with open(path, 'r') as f:
        return json.load(f)


def load_compaction_data() -> list[dict]:
    """Load compaction benchmark results."""
    path = RESULTS_DIR / "compaction_results.json"
    with open(path, 'r') as f:
        return json.load(f)


# ─── Figure 1: Latency vs Sequence Length ─────────────────────────────────────

def generate_fig1_latency_vs_seqlen():
    """Line plot: sequence length vs latency with error bands.

    Log-log scale, vibrant neon colors on dark background.
    Annotates speedup ratios at the 32K data point.
    """
    print("  [1/6] Latency vs Sequence Length…")
    data = load_profiling_data()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Group by configuration
    config_map = {
        "Dense (no eviction)": {"color": COLORS['dense'], "marker": "o", "ls": "-"},
        "OrthoCache 50% eviction": {"color": COLORS['50%'], "marker": "s", "ls": "-"},
        "OrthoCache 62.5% eviction": {"color": COLORS['62.5%'], "marker": "D", "ls": "-"},
        "OrthoCache 75% eviction": {"color": COLORS['75%'], "marker": "^", "ls": "-"},
    }

    for label, style in config_map.items():
        entries = [d for d in data if d["label"] == label]
        if not entries:
            continue
        seq_lens = [d["seq_len"] for d in entries]
        means = [d["mean_ms"] for d in entries]
        stds = [d["std_ms"] for d in entries]

        means_arr = np.array(means)
        stds_arr = np.array(stds)

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

    # Annotate speedup at 32K
    dense_32k = [d for d in data if d["label"] == "Dense (no eviction)" and d["seq_len"] == 32768]
    if dense_32k:
        dense_ms = dense_32k[0]["mean_ms"]
        for label in ["OrthoCache 50% eviction", "OrthoCache 62.5% eviction", "OrthoCache 75% eviction"]:
            entry = [d for d in data if d["label"] == label and d["seq_len"] == 32768]
            if entry:
                ortho_ms = entry[0]["mean_ms"]
                speedup = dense_ms / ortho_ms
                y_pos = ortho_ms
                ax.annotate(
                    f'{speedup:.1f}×',
                    xy=(32768, y_pos), xytext=(38000, y_pos * 0.7),
                    fontsize=11, fontweight='bold',
                    color=config_map[label]["color"],
                    arrowprops=dict(arrowstyle='->', color=config_map[label]["color"],
                                    lw=1.5, alpha=0.7),
                )

    ax.set_xlabel('Context Length (tokens)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('OrthoCache GPU: Attention Latency vs Context Length',
                 fontsize=16, fontweight='bold', pad=15)

    ax.legend(loc='upper left', framealpha=0.7, facecolor='#1a1a2e',
              edgecolor='#444444', fontsize=11)

    add_watermark(fig)
    save_figure(fig, 'latency_vs_seqlen')


# ─── Figure 2: Speedup Heatmap ───────────────────────────────────────────────

def generate_fig2_speedup_heatmap():
    """Heatmap: eviction_rate × sequence_length → speedup ratio.

    Custom colormap from deep purple to bright green to gold.
    """
    print("  [2/6] Speedup Heatmap…")
    data = load_profiling_data()

    eviction_labels = ["50%", "62.5%", "75%"]
    eviction_configs = [
        "OrthoCache 50% eviction",
        "OrthoCache 62.5% eviction",
        "OrthoCache 75% eviction",
    ]
    seq_lens = sorted(set(d["seq_len"] for d in data))

    # Build speedup matrix
    speedup_matrix = np.zeros((len(eviction_labels), len(seq_lens)))

    for i, config_label in enumerate(eviction_configs):
        for j, sl in enumerate(seq_lens):
            dense = [d for d in data if d["label"] == "Dense (no eviction)" and d["seq_len"] == sl]
            ortho = [d for d in data if d["label"] == config_label and d["seq_len"] == sl]
            if dense and ortho:
                speedup_matrix[i, j] = dense[0]["mean_ms"] / ortho[0]["mean_ms"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Custom colormap: deep purple → bright green → gold
    cmap_colors = ['#2D1B69', '#1A6B4C', '#4ECDC4', '#96CEB4', '#FFD93D', '#FF8C00']
    custom_cmap = LinearSegmentedColormap.from_list('orthocache', cmap_colors, N=256)

    im = ax.imshow(speedup_matrix, cmap=custom_cmap, aspect='auto',
                   vmin=max(0.8, speedup_matrix.min() * 0.9),
                   vmax=speedup_matrix.max() * 1.1)

    # Annotate cells
    for i in range(len(eviction_labels)):
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
    ax.set_title('OrthoCache GPU: Speedup vs Dense Attention',
                 fontsize=16, fontweight='bold', pad=15)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Speedup Factor (×)', fontsize=11)

    add_watermark(fig)
    save_figure(fig, 'speedup_heatmap')


# ─── Figure 3: Spectral Separation ───────────────────────────────────────────

def generate_fig3_spectral_separation():
    """Histogram/KDE: ζ distribution for structured vs noise blocks.

    Overlapping translucent histograms with vertical threshold line.
    """
    print("  [3/6] Spectral Separation…")
    spec_data = load_spectral_data()

    semantic_zeta = spec_data["zeta"]["semantic"]["values"]
    noise_zeta = spec_data["zeta"]["noise"]["values"]
    optimal_thresh = spec_data["separability"]["optimal_zeta_max"]
    accuracy = spec_data["separability"]["classification_accuracy"]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Histogram bins
    all_vals = semantic_zeta + noise_zeta
    bins = np.linspace(min(all_vals) * 0.9, max(all_vals) * 1.1, 50)

    # Semantic blocks
    ax.hist(semantic_zeta, bins=bins, alpha=0.55, color=COLORS['semantic'],
            edgecolor='white', linewidth=0.5, label=f'Semantic blocks (n={len(semantic_zeta)})',
            density=True, zorder=3)

    # Noise blocks
    ax.hist(noise_zeta, bins=bins, alpha=0.55, color=COLORS['noise'],
            edgecolor='white', linewidth=0.5, label=f'Noise blocks (n={len(noise_zeta)})',
            density=True, zorder=3)

    # Vertical threshold line
    ax.axvline(optimal_thresh, color='#FFD93D', linewidth=2.5, linestyle='--',
               label=f'ζ_max threshold = {optimal_thresh:.3f}', zorder=6)

    # Shade regions
    ax.axvspan(ax.get_xlim()[0] if ax.get_xlim()[0] < optimal_thresh else bins[0],
               optimal_thresh, alpha=0.08, color=COLORS['semantic'], zorder=1)
    ax.axvspan(optimal_thresh,
               ax.get_xlim()[1] if ax.get_xlim()[1] > optimal_thresh else bins[-1],
               alpha=0.08, color=COLORS['noise'], zorder=1)

    # Annotations
    ax.text(0.03, 0.95,
            f'Classification accuracy: {accuracy:.1%}\n'
            f'Semantic ζ: {np.mean(semantic_zeta):.3f} ± {np.std(semantic_zeta):.3f}\n'
            f'Noise ζ: {np.mean(noise_zeta):.3f} ± {np.std(noise_zeta):.3f}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                      edgecolor='#444444', alpha=0.9))

    ax.set_xlabel('Spectral Decay Ratio (ζ)')
    ax.set_ylabel('Density')
    ax.set_title('Spectral Decay Ratio (ζ) Separability',
                 fontsize=16, fontweight='bold', pad=15)

    ax.legend(loc='upper right', framealpha=0.8, facecolor='#1a1a2e',
              edgecolor='#444444')

    add_watermark(fig)
    save_figure(fig, 'spectral_separation')


# ─── Figure 4: Band Energy Stacked ───────────────────────────────────────────

def generate_fig4_band_energy():
    """Stacked area chart: energy fraction per band across blocks, sorted by ζ.

    Shows how high-sequency energy dominates noise blocks.
    """
    print("  [4/6] Band Energy Stacked…")
    spec_data = load_spectral_data()

    band = spec_data["band_energy"]
    dc_frac = np.array(band["dc_fraction"])
    low_frac = np.array(band["low_fraction"])
    mid_frac = np.array(band["mid_fraction"])
    high_frac = np.array(band["high_fraction"])
    zeta = np.array(band["zeta_per_block"])
    labels = band["labels"]

    # Sort by ζ
    sort_idx = np.argsort(zeta)
    dc_frac = dc_frac[sort_idx]
    low_frac = low_frac[sort_idx]
    mid_frac = mid_frac[sort_idx]
    high_frac = high_frac[sort_idx]
    zeta_sorted = zeta[sort_idx]
    labels_sorted = [labels[i] for i in sort_idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                     gridspec_kw={'height_ratios': [3, 1]},
                                     sharex=True)

    x = np.arange(len(zeta_sorted))

    # Stacked area chart
    ax1.fill_between(x, 0, dc_frac, alpha=0.85,
                     color=COLORS['dc'], label='DC (block mean)', zorder=4)
    ax1.fill_between(x, dc_frac, dc_frac + low_frac, alpha=0.85,
                     color=COLORS['low'], label='Low sequency (1–63)', zorder=3)
    ax1.fill_between(x, dc_frac + low_frac, dc_frac + low_frac + mid_frac,
                     alpha=0.85, color=COLORS['mid'], label='Mid sequency (64–255)', zorder=2)
    ax1.fill_between(x, dc_frac + low_frac + mid_frac, 1.0,
                     alpha=0.85, color=COLORS['high'], label='High sequency (256–511)', zorder=1)

    ax1.set_ylabel('Energy Fraction')
    ax1.set_title('Multi-Band Energy Decomposition (sorted by ζ)',
                  fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', framealpha=0.8, facecolor='#1a1a2e',
               edgecolor='#444444', ncol=2)
    ax1.set_ylim(0, 1.0)

    # Bottom panel: ζ values with color-coded bars
    bar_colors = [COLORS['semantic'] if l == 'semantic' else COLORS['noise']
                  for l in labels_sorted]
    ax2.bar(x, zeta_sorted, color=bar_colors, alpha=0.8, width=1.0, edgecolor='none')
    ax2.set_xlabel('Block Index (sorted by ζ)')
    ax2.set_ylabel('ζ')
    ax2.set_title('Spectral Decay Ratio per Block', fontsize=12)

    # Add semantic/noise legend to bottom panel
    sem_patch = mpatches.Patch(color=COLORS['semantic'], alpha=0.8, label='Semantic')
    noise_patch = mpatches.Patch(color=COLORS['noise'], alpha=0.8, label='Noise')
    ax2.legend(handles=[sem_patch, noise_patch], loc='upper left',
               framealpha=0.8, facecolor='#1a1a2e', edgecolor='#444444')

    add_watermark(fig)
    save_figure(fig, 'band_energy_stacked')


# ─── Figure 5: Compaction Speedup ─────────────────────────────────────────────

def generate_fig5_compaction_speedup():
    """Grouped bar chart: compaction overhead vs attention savings.

    For each eviction rate, shows compaction time + attention time vs dense.
    """
    print("  [5/6] Compaction Speedup…")
    data = load_compaction_data()

    # Pick the largest sequence length for the most dramatic comparison
    max_sl = max(d["seq_len"] for d in data)
    entries = [d for d in data if d["seq_len"] == max_sl]

    eviction_rates = [d["eviction_rate"] for d in entries]
    compact_times = [d["compact_mean_ms"] for d in entries]
    attn_times = [d["attention_mean_ms"] for d in entries]
    dense_time = entries[0]["dense_mean_ms"]
    speedups = [d["speedup"] for d in entries]

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(eviction_rates))
    width = 0.35

    # Dense baseline bar (single wide bar at each position)
    dense_bars = ax.bar(x - width/2, [dense_time] * len(x), width,
                        color=COLORS['dense'], alpha=0.7, label='Dense Attention',
                        edgecolor='white', linewidth=0.5)

    # Stacked OrthoCache bar: compaction + attention
    compact_bars = ax.bar(x + width/2, compact_times, width,
                          color='#45B7D1', alpha=0.85, label='Compaction Overhead',
                          edgecolor='white', linewidth=0.5)
    attn_bars = ax.bar(x + width/2, attn_times, width, bottom=compact_times,
                       color=COLORS['50%'], alpha=0.85, label='Attention (compacted)',
                       edgecolor='white', linewidth=0.5)

    # Speedup annotations above each group
    for i, (su, ct, at) in enumerate(zip(speedups, compact_times, attn_times)):
        total = ct + at
        y_pos = max(dense_time, total) + dense_time * 0.05
        color = '#96CEB4' if su > 1.0 else '#FF6B6B'
        ax.text(i, y_pos, f'{su:.2f}×', ha='center', va='bottom',
                fontsize=13, fontweight='bold', color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{r:.1%}' for r in eviction_rates])
    ax.set_xlabel('Eviction Rate')
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'OrthoCache GPU: Compaction Overhead vs Attention Savings (seq_len={max_sl:,})',
                 fontsize=15, fontweight='bold', pad=15)

    ax.legend(loc='upper right', framealpha=0.8, facecolor='#1a1a2e',
              edgecolor='#444444')

    # Add horizontal line at dense baseline
    ax.axhline(dense_time, color=COLORS['dense'], linewidth=1.0,
               linestyle=':', alpha=0.5)

    add_watermark(fig)
    save_figure(fig, 'compaction_speedup')


# ─── Figure 6: Memory Savings Projection ─────────────────────────────────────

def generate_fig6_memory_savings():
    """Projection plot: KV-cache memory usage vs context length.

    Model: Llama 3 70B (80 layers, 64 heads, head_dim=128)
    Shows no-eviction, 50%, 75% eviction projections.
    Horizontal line at 80GB H100 HBM3 capacity.
    """
    print("  [6/6] Memory Savings Projection…")

    # Llama 3 70B parameters
    num_layers = 80
    num_heads = 64  # KV heads (GQA: 8 groups × 8 heads)
    head_dim = 128
    bytes_per_element = 2  # bfloat16

    # Context lengths to project
    ctx_lengths = np.array([1024, 2048, 4096, 8192, 16384, 32768,
                            65536, 131072, 262144, 524288, 1048576])

    # KV-cache memory formula:
    # memory = num_layers × num_heads × head_dim × seq_len × 2 (K+V) × bytes_per_element
    def kv_memory_gb(seq_len: np.ndarray, eviction_rate: float = 0.0) -> np.ndarray:
        effective_len = seq_len * (1.0 - eviction_rate)
        bytes_total = num_layers * num_heads * head_dim * effective_len * 2 * bytes_per_element
        return bytes_total / (1024 ** 3)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Memory curves
    configs = [
        ("No eviction", 0.0, COLORS['no_evict'], '-', 'o'),
        ("50% eviction", 0.5, COLORS['50_evict'], '--', 's'),
        ("75% eviction", 0.75, COLORS['75_evict'], '-.', '^'),
    ]

    for label, evict, color, ls, marker in configs:
        mem = kv_memory_gb(ctx_lengths, evict)
        ax.plot(ctx_lengths, mem, color=color, linewidth=2.5, linestyle=ls,
                marker=marker, markersize=8, markeredgecolor='white',
                markeredgewidth=1, label=label, zorder=5)

    # H100 HBM3 capacity line
    ax.axhline(80, color='#FFD93D', linewidth=2, linestyle='--', alpha=0.8, zorder=4)
    ax.text(ctx_lengths[1], 82, 'H100 HBM3 (80 GB)', color='#FFD93D',
            fontsize=11, fontweight='bold', va='bottom')

    # OOM zone shading
    ax.fill_between(ctx_lengths, 80, max(200, kv_memory_gb(ctx_lengths[-1]) * 1.1),
                    color='#FF6B6B', alpha=0.1, zorder=1)
    ax.text(ctx_lengths[-3], 95, 'OOM ZONE', color='#FF6B6B', fontsize=14,
            fontweight='bold', alpha=0.4, ha='center')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{int(x/1024)}K' if x >= 1024 else str(int(x))
    ))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}'))

    ax.set_xlabel('Context Length (tokens)')
    ax.set_ylabel('KV-Cache Memory (GB)')
    ax.set_title('OrthoCache GPU: KV-Cache Memory Projection (Llama 3 70B)',
                 fontsize=16, fontweight='bold', pad=15)

    ax.legend(loc='upper left', framealpha=0.8, facecolor='#1a1a2e',
              edgecolor='#444444', fontsize=12)

    # Annotate key crossover points
    for label, evict, color, _, _ in configs:
        mem = kv_memory_gb(ctx_lengths, evict)
        cross_idx = np.searchsorted(mem, 80)
        if 0 < cross_idx < len(ctx_lengths):
            # Interpolate
            x_cross = ctx_lengths[cross_idx]
            ax.plot(x_cross, 80, 'X', color=color, markersize=12, zorder=10,
                    markeredgecolor='white', markeredgewidth=2)
            ax.annotate(f'{int(x_cross/1024)}K tokens',
                        xy=(x_cross, 80), xytext=(x_cross * 1.5, 60),
                        fontsize=10, color=color,
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    add_watermark(fig)
    save_figure(fig, 'memory_savings')


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_all_figures():
    """Generate all 6 publication-quality figures."""
    print("=" * 70)
    print("  OrthoCache GPU — Publication-Quality Figure Generator")
    print("=" * 70)
    print(f"  Data source:  {RESULTS_DIR}")
    print(f"  Output dir:   {OUTPUT_DIR}")
    print()

    generate_fig1_latency_vs_seqlen()
    generate_fig2_speedup_heatmap()
    generate_fig3_spectral_separation()
    generate_fig4_band_energy()
    generate_fig5_compaction_speedup()
    generate_fig6_memory_savings()

    print()
    print(f"  [DONE] All 6 figures saved to: {OUTPUT_DIR}")
    print("  Formats: PNG (300 DPI raster) + SVG (vector)")
    print()

    # List output files
    png_files = sorted(OUTPUT_DIR.glob("*.png"))
    svg_files = sorted(OUTPUT_DIR.glob("*.svg"))
    print(f"  PNG files ({len(png_files)}):")
    for f in png_files:
        print(f"    {f.name}")
    print(f"  SVG files ({len(svg_files)}):")
    for f in svg_files:
        print(f"    {f.name}")


if __name__ == "__main__":
    generate_all_figures()
