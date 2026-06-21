"""Figure 8: Drag Force Sensitivity to Upstream Velocity Error."""

import sys
import os
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
from scripts.figures.utils import setup_figure, save_figure, COLORS, lighten_color
import matplotlib.pyplot as plt


def main():
    # Data from Table 17
    vel_errors = np.array([1.0, 3.0, 5.0, 8.0, 10.0, 13.75])
    drag_errors_mean = np.array([1.48, 4.44, 7.39, 11.83, 14.79, 20.33])
    drag_errors_std = np.array([1.16, 3.47, 5.78, 9.24, 11.55, 15.88])

    # Linear fit
    coeffs = np.polyfit(vel_errors, drag_errors_mean, 1)
    A = coeffs[0]  # ~1.47

    # 5% tolerance threshold
    vel_at_5pct = 5.0 / A  # ~3.4%

    fig, ax = setup_figure(width=3.5, height=2.5, nrows=1, ncols=1)
    ax = ax[0, 0]

    # +-1sigma band
    ax.fill_between(vel_errors, 
                    drag_errors_mean - drag_errors_std,
                    drag_errors_mean + drag_errors_std,
                    alpha=0.3, color=COLORS['blue'], label=r'$\pm 1\sigma$')

    # Mean line
    ax.plot(vel_errors, drag_errors_mean, 'o-', color=COLORS['blue'], 
            markersize=6, linewidth=1.5, label='Mean drag error (5 seeds)')

    # 5% engineering tolerance
    ax.axhline(y=5.0, color='red', linestyle='--', linewidth=1.5,
               label='5% engineering tolerance')

    # Intersection annotation
    ax.axvline(x=vel_at_5pct, color='gray', linestyle=':', alpha=0.7)
    ax.plot(vel_at_5pct, 5.0, 'ko', markersize=8)
    ax.annotate(rf'$\approx {vel_at_5pct:.1f}\%$ upstream error',
                xy=(vel_at_5pct, 5.0), xytext=(vel_at_5pct + 2, 8),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9)

    ax.set_xlabel(r'Upstream $L_2$ velocity error (%)')
    ax.set_ylabel('Integrated drag force error (%)')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 38)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    save_figure(fig, 'figure_08_drag_sensitivity')
    plt.close()


if __name__ == '__main__':
    main()
