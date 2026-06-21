"""Figure 3: Resolution behavior — RBF-FD solver vs Neural Operator.

Shows that the neural operator maintains ~10% relative L2 error across
tested resolutions, while the RBF-FD solver converges at sub-quadratic rate.
"""

import sys
import os
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
from scripts.figures.utils import setup_figure, save_figure, COLORS
import matplotlib.pyplot as plt


def main():
    # Resolution data from paper
    N_values = np.array([225, 961, 4096, 10000])
    h_values = 1.0 / np.sqrt(N_values)  # Approximate mesh spacing

    # RBF-FD solver error (sub-quadratic convergence)
    solver_error = np.array([2.4e-03, 1.1e-03, 5.5e-04, 3.1e-04])

    # Neural operator error (approximately flat, ~10%)
    no_error = np.array([0.1375, 0.125, 0.115, 0.102])

    fig, ax = setup_figure(width=3.5, height=2.5, nrows=1, ncols=1)
    ax = ax[0, 0]

    # RBF-FD solver
    ax.loglog(N_values, solver_error, 's-', color=COLORS['blue'], 
              markersize=6, linewidth=1.5, label='RBF-FD solver')

    # Neural operator
    ax.loglog(N_values, no_error, 'o-', color=COLORS['orange'],
              markersize=6, linewidth=1.5, 
              label='Neural operator ($\\approx 10\\%$ across tested resolutions)')

    # O(h^2) reference
    ax.loglog(N_values, solver_error[0] * (N_values[0] / N_values)**1.1, 
              'k--', linewidth=1.0, alpha=0.7, label=r'$\\mathcal{O}(h^2)$ reference')

    # Slope annotation
    ax.annotate('slope 0.54', xy=(1000, 1e-03), xytext=(2000, 2e-03),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9, ha='center')

    # Ratio annotation
    ratio = no_error[-1] / solver_error[-1]
    ax.text(0.7, 0.3, f'Ratio at N=10K:\\n{ratio:.0f}×', transform=ax.transAxes,
            fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(r'Nodal density $N$')
    ax.set_ylabel(r'Relative $L_2$ error')
    ax.set_xlim(150, 15000)
    ax.set_ylim(1e-4, 2e-1)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    save_figure(fig, 'figure_03_resolution_behavior')
    plt.close()


if __name__ == '__main__':
    main()
