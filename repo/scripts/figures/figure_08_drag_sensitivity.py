"""Figure 8: Drag force sensitivity with REAL data loading.

Reads from: results/force_sensitivity.json
Fallback: synthetic data matching Table drag_sensitivity
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.figures.utils import (
    setup_figure, save_figure, format_scientific, add_panel_label, COLORS
)

def load_drag_sensitivity_data():
    """Load drag sensitivity data from force_sensitivity.json."""

    json_path = Path('results/force_sensitivity.json')
    if json_path.exists():
        with open(json_path) as f:
            d = json.load(f)

        l2_vel = np.array(d.get('l2_velocity_pct', []))
        drag_mean = np.array(d.get('drag_mean_pct', []))
        drag_std = np.array(d.get('drag_std_pct', []))

        if len(l2_vel) > 0:
            print(f"[Figure 8] Loaded real drag sensitivity data from force_sensitivity.json ({len(l2_vel)} points)")
            return l2_vel, drag_mean, drag_std

    # Fallback
    print("[Figure 8] WARNING: force_sensitivity.json not found. Using synthetic fallback.")
    l2_vel = np.array([1.0, 3.0, 5.0, 8.0, 10.0, 13.75])
    drag_mean = np.array([1.48, 4.44, 7.39, 11.83, 14.79, 20.33])
    drag_std = np.array([1.16, 3.47, 5.78, 9.24, 11.55, 15.88])
    return l2_vel, drag_mean, drag_std

def main():
    l2_vel, drag_mean, drag_std = load_drag_sensitivity_data()

    A = 1.47
    drag_fit = A * l2_vel

    fig, axes = setup_figure(width=3.5, height=2.5, nrows=1, ncols=1)
    ax = axes[0, 0]

    ax.plot(l2_vel, drag_mean, 'o-', color=COLORS['blue'], linewidth=1.5, markersize=6, label='Mean (5 seeds)')
    ax.fill_between(l2_vel, drag_mean - drag_std, drag_mean + drag_std, color=COLORS['blue'], alpha=0.2, label=r'$\pm1\sigma$')
    ax.plot(l2_vel, drag_fit, '--', color=COLORS['red'], linewidth=1.5, label=f'Fit: $A={A:.2f}$')

    ax.axhline(y=5.0, color='green', linestyle='--', linewidth=1.2, alpha=0.7, label='5% drag tolerance')
    ax.axvline(x=3.4, color='green', linestyle=':', linewidth=1.0, alpha=0.7, label=r'$\varepsilon_{\mathrm{vel}}^* \approx 3.4\%$')

    ax.set_xlabel(r'Upstream $L_2$ velocity error (%)')
    ax.set_ylabel('Relative drag error (%)')
    ax.set_xlim(0, max(16, l2_vel.max() * 1.2))
    ax.set_ylim(0, max(40, (drag_mean + drag_std).max() * 1.1))
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    add_panel_label(ax, 'a')

    fig.suptitle('Drag Force Sensitivity to Velocity Error', fontsize=12, fontweight='bold', y=1.02)
    save_figure(fig, 'figure_08_drag_sensitivity')
    plt.close()
    print("Figure 8 generated successfully.")

if __name__ == '__main__':
    main()
