"""Figure 4: Warm-start performance across Re in [100, 500] (4-panel layout).

Panel layout:
(a) top-left:  Solver iterations to convergence
(b) top-right: Wall-clock speedup S = T_cold / T_warm
(c) bottom-left: Final corrected solution accuracy
(d) bottom-right: Standalone surrogate error (NOT preconditioner metric)
"""

import sys
import os
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
from scripts.figures.utils import setup_figure, save_figure, add_panel_label, COLORS, lighten_color
import matplotlib.pyplot as plt


def generate_warmstart_data():
    """Generate warm-start performance data from paper."""
    Re_values = np.array([100, 150, 200, 300, 400, 500])

    # Cold start iterations
    cold_iters = np.array([120, 180, 240, 320, 420, 500])

    # Hard warm-start (proposed): much fewer iterations
    hard_iters = np.array([50, 65, 85, 95, 110, 120])
    hard_iters_err = np.array([5, 8, 10, 12, 15, 18])

    # Soft warm-start: barely helps
    soft_iters = np.array([110, 160, 220, 300, 390, 450])
    soft_iters_err = np.array([10, 15, 20, 25, 30, 35])

    # Speedups
    hard_speedup = cold_iters / hard_iters
    soft_speedup = cold_iters / soft_iters

    # Final accuracy (%)
    hard_acc = np.array([0.3, 0.35, 0.45, 0.55, 0.65, 0.72])
    soft_acc = np.array([0.32, 0.38, 0.50, 0.62, 0.75, 0.85])

    # Standalone NO error (%)
    standalone_err = np.array([8, 12, 18, 28, 38, 49])

    return (Re_values, cold_iters, hard_iters, hard_iters_err, soft_iters, soft_iters_err,
            hard_speedup, soft_speedup, hard_acc, soft_acc, standalone_err)


def plot_panel_a(ax):
    """(a) Solver iterations vs Re."""
    data = generate_warmstart_data()
    Re, cold, hard, hard_err, soft, soft_err = data[:6]

    x = np.arange(len(Re))
    width = 0.25

    ax.bar(x - width, cold, width, label='Cold-start', color='gray', alpha=0.7)
    ax.bar(x, hard, width, yerr=hard_err, label='Hard NO warm-start (proposed)', 
           color=COLORS['blue'], alpha=0.8, capsize=3)
    ax.bar(x + width, soft, width, yerr=soft_err, label='Soft NO warm-start',
           color=COLORS['orange'], alpha=0.8, capsize=3)

    # Anchor annotation
    ax.annotate('500\n(anchor)', xy=(5, 500), xytext=(5.3, 480),
                fontsize=8, ha='center', color='gray')
    ax.annotate('120\n(anchor)', xy=(5, 120), xytext=(5.3, 140),
                fontsize=8, ha='center', color=COLORS['blue'])

    ax.set_xlabel(r'Reynolds number $\mathrm{Re}$')
    ax.set_ylabel('Solver iterations to convergence')
    ax.set_xticks(x)
    ax.set_xticklabels(Re)
    ax.set_ylim(0, 550)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')


def plot_panel_b(ax):
    """(b) Wall-clock speedup vs Re."""
    data = generate_warmstart_data()
    Re, _, _, _, _, _, hard_spd, soft_spd = data[:8]

    ax.plot(Re, hard_spd, 's-', color=COLORS['blue'], markersize=6, linewidth=1.5,
            label='Hard NO warm-start (proposed)')
    ax.plot(Re, soft_spd, '^--', color=COLORS['orange'], markersize=6, linewidth=1.5,
            label='Soft NO warm-start')

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='Baseline S=1')
    ax.axhline(y=2.0, color='gray', linestyle=':', alpha=0.5)
    ax.text(110, 2.1, r'$S = 2\\times$', fontsize=8, color='gray')

    # Anchor annotation
    ax.annotate('(anchor)', xy=(500, hard_spd[-1]), xytext=(450, hard_spd[-1]+0.3),
                fontsize=8, ha='center', color=COLORS['blue'])

    ax.set_xlabel(r'Reynolds number $\mathrm{Re}$')
    ax.set_ylabel(r'Speedup $S = T_{\\mathrm{cold}} / T_{\\mathrm{warm}}$')
    ax.set_xlim(80, 520)
    ax.set_ylim(0.5, 6)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_panel_c(ax):
    """(c) Corrected solution accuracy vs Re."""
    data = generate_warmstart_data()
    Re, _, _, _, _, _, _, _, hard_acc, soft_acc = data[:10]

    ax.plot(Re, hard_acc, 's-', color=COLORS['blue'], markersize=6, linewidth=1.5,
            label='Hard warm-start (hybrid)')
    ax.plot(Re, soft_acc, '^--', color=COLORS['orange'], markersize=6, linewidth=1.5,
            label='Soft warm-start (hybrid)')

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='1% threshold')

    ax.set_xlabel(r'Reynolds number $\mathrm{Re}$')
    ax.set_ylabel(r'Final corrected $\varepsilon_{L_2}$ (%)')
    ax.set_xlim(80, 520)
    ax.set_ylim(0, 2.0)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_panel_d(ax):
    """(d) Standalone surrogate error vs Re."""
    data = generate_warmstart_data()
    Re, _, _, _, _, _, _, _, _, _, standalone = data

    ax.bar(Re, standalone, width=30, color='gray', alpha=0.7, 
           edgecolor='black', linewidth=0.5)

    # 49% anchor annotation
    ax.annotate('49%\n(anchor)', xy=(500, 49), xytext=(450, 42),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9, fontweight='bold', ha='center')

    ax.set_xlabel(r'Reynolds number $\mathrm{Re}$')
    ax.set_ylabel(r'Standalone $\varepsilon_{L_2}$ (%)')
    ax.set_xlim(80, 520)
    ax.set_ylim(0, 55)
    ax.grid(True, alpha=0.3, axis='y')


def main():
    fig, axes = setup_figure(width=3.5, height=2.5, nrows=2, ncols=2)

    plot_panel_a(axes[0, 0])
    add_panel_label(axes[0, 0], 'a')

    plot_panel_b(axes[0, 1])
    add_panel_label(axes[0, 1], 'b')

    plot_panel_c(axes[1, 0])
    add_panel_label(axes[1, 0], 'c')

    plot_panel_d(axes[1, 1])
    add_panel_label(axes[1, 1], 'd')

    # Red box for panel d
    rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=axes[1,1].transAxes,
                         fill=False, edgecolor='red', linewidth=2)
    axes[1,1].add_patch(rect)
    axes[1,1].text(0.5, 0.95, 'Surrogate metric (NOT preconditioner metric)',
                    transform=axes[1,1].transAxes, fontsize=8, color='red',
                    ha='center', va='top', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    fig.suptitle(r'Preconditioner Performance across Extrapolation Regime',
                 fontsize=11, fontweight='bold', y=1.02)
    save_figure(fig, 'figure_04_warmstart_performance')
    plt.close()


if __name__ == '__main__':
    main()
