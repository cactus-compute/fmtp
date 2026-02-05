"""
Generate ICLR-quality graph comparing MLX and GPU tree attention forward pass timing.

Usage:
    python scripts/plot_tree_bench.py
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use a clean, publication-ready style
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.figsize'] = (5.5, 4)  # ICLR single column width
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['grid.linewidth'] = 0.5
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['lines.markersize'] = 5


def load_results(filepath):
    """Load benchmark results from JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    # Create lookup table: (context_size, tree_size) -> avg_time_ms
    lookup = {}
    for result in data['results']:
        key = (result['context_size'], result['tree_size'])
        lookup[key] = result['avg_time_ms']

    return data['tree_sizes'], lookup


def main():
    # Load data (nolora checkpoints with lora_rank=0, 30 runs with 5 warmup)
    mlx_tree_sizes, mlx_lookup = load_results('mlx_tree_bench_20260205_003413.json')
    gpu_tree_sizes, gpu_lookup = load_results('cuda_tree_bench_20260205_083924.json')

    # Context sizes to plot
    context_sizes = [8, 64, 256]

    # Colors - using a colorblind-friendly palette
    # MLX: blue shades, GPU: orange/red shades
    mlx_colors = ['#1f77b4', '#4a90d9', '#7eb3ed']  # Blue gradient
    gpu_colors = ['#d62728', '#e85858', '#f28f8f']  # Red gradient

    # Line styles and markers
    mlx_markers = ['o', 's', '^']  # circle, square, triangle
    gpu_markers = ['o', 's', '^']

    # Create figure
    fig, ax = plt.subplots()

    # Tree sizes to plot (x-axis)
    tree_sizes = mlx_tree_sizes

    # Plot MLX lines
    for i, ctx in enumerate(context_sizes):
        times = [mlx_lookup.get((ctx, t), np.nan) for t in tree_sizes]
        ax.plot(tree_sizes, times,
                color=mlx_colors[i],
                marker=mlx_markers[i],
                linestyle='-',
                markerfacecolor='white',
                markeredgewidth=1.2,
                label=f'MLX (ctx={ctx})')

    # Plot GPU lines
    for i, ctx in enumerate(context_sizes):
        times = [gpu_lookup.get((ctx, t), np.nan) for t in tree_sizes]
        ax.plot(tree_sizes, times,
                color=gpu_colors[i],
                marker=gpu_markers[i],
                linestyle='--',
                markerfacecolor='white',
                markeredgewidth=1.2,
                label=f'A100 (ctx={ctx})')

    # Styling
    ax.set_xlabel('Number of Tokens in Tree')
    ax.set_ylabel('Forward Pass Time (ms)')
    ax.set_title('Tree Attention Forward Pass Latency: M4 Pro (MLX) vs A100 (CUDA)')
    ax.set_xlim(0, 85)
    ax.set_ylim(0, 60)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend - outside the plot for clarity
    ax.legend(loc='upper left', frameon=True, fancybox=False,
              edgecolor='black', framealpha=1.0, ncol=2)

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig('tree_bench_comparison.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('tree_bench_comparison.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print("Saved tree_bench_comparison.png and tree_bench_comparison.pdf")


if __name__ == '__main__':
    main()
