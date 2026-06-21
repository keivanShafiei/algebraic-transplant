"""Figure 8: Drag force sensitivity with ±1σ shaded region.

Shows how relative drag error scales with upstream L2 velocity error.
x-axis in percent; ±1σ band from 5 seeds.
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
    np.random.seed(48)
    l2_vel_pct = np.array([1.0, 3.0, 5.0, 8.0, 10.0, 13.75])

    # Mean drag error from Table drag_sensitivity
    drag_mean = np.array([1.48, 4.44, 7.39, 11.83, 14.79, 20.33])
    drag_std = np.array([1.16, 3.47, 5.78, 9.24, 11.55, 15.88])

    # Linear fit A ≈ 1.47
    A = 1.47
    drag_fit = A * l2_vel_pct

    fig, axes = setup_figure(width=3.5, height=2.5, nrows=1, ncols=1)
    ax = axes[0, 0]

    ax.plot(l2_vel_pct, drag_mean, 'o-', color=COLORS['blue'], linewidth=1.5, markersize=6, label='Mean (5 seeds)')
    ax.fill_between(l2_vel_pct, drag_mean - drag_std, drag_mean + drag_std, color=COLORS['blue'], alpha=0.2, label=r'$\pm1\sigma$')
    ax.plot(l2_vel_pct, drag_fit, '--', color=COLORS['red'], linewidth=1.5, label=f'Fit: $A={A:.2f}$')

    ax.axhline(y=5.0, color='green', linestyle='--', linewidth=1.2, alpha=0.7, label='5% drag tolerance')
    ax.axvline(x=3.4, color='green', linestyle=':', linewidth=1.0, alpha=0.7, label=r'$\varepsilon_{\mathrm{vel}}^* \approx 3.4\%$')

    ax.set_xlabel(r'Upstream $L_2$ velocity error (%)')
    ax.set_ylabel('Relative drag error (%)')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 40)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    add_panel_label(ax, 'a')

    fig.suptitle('Drag Force Sensitivity to Velocity Error', fontsize=12, fontweight='bold', y=1.02)
    save_figure(fig, 'figure_08_drag_sensitivity')
    plt.close()
    print("Figure 8 generated successfully.")

if __name__ == '__main__':
    main()
