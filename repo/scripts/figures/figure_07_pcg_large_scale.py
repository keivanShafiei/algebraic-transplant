"""Figure 7: Large-scale PCG scalability (N = 100,000).

Reproduces Table 17: timing and memory at industrial scale.
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.figures.utils import (
    setup_figure, save_figure, format_scientific, add_panel_label, COLORS
)

def main():
    np.random.seed(47)
    n_values = np.array([1000, 5000, 10000, 25000, 50000, 100000])
    time_s = np.array([0.05, 0.25, 0.55, 1.1, 1.8, 3.08])
    memory_gb = np.array([0.02, 0.08, 0.15, 0.25, 0.35, 0.45])

    fig, axes = setup_figure(width=3.5, height=2.5, nrows=1, ncols=2)

    ax = axes[0, 0]
    ax.plot(n_values, time_s, 'o-', color=COLORS['blue'], linewidth=1.5, markersize=6)
    ax.set_xlabel('Number of nodes $N$')
    ax.set_ylabel('Wall-clock time (s)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(800, 150000)
    ax.set_ylim(0.03, 5.0)
    ax.grid(True, alpha=0.3, which='both')
    add_panel_label(ax, 'a')

    ax = axes[0, 1]
    ax.plot(n_values, memory_gb, 's-', color=COLORS['orange'], linewidth=1.5, markersize=6)
    ax.set_xlabel('Number of nodes $N$')
    ax.set_ylabel('VRAM (GB)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(800, 150000)
    ax.set_ylim(0.01, 1.0)
    ax.grid(True, alpha=0.3, which='both')
    add_panel_label(ax, 'b')

    fig.suptitle('Large-Scale PCG Scalability', fontsize=12, fontweight='bold', y=1.02)
    save_figure(fig, 'figure_07_pcg_large_scale')
    plt.close()
    print("Figure 7 generated successfully.")

if __name__ == '__main__':
    main()
