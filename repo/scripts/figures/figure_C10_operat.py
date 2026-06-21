"""Figure C.10: Operator gap scaling delta_op = O(h^2)."""

import sys
import os
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
from scripts.figures.utils import setup_figure, save_figure, COLORS
import matplotlib.pyplot as plt


def main():
    # Data from Table C.18
    N_values = np.array([225, 961, 4096, 10000])
    h_values = 1.0 / np.sqrt(N_values)
    delta_op = np.array([3.1e-3, 1.8e-3, 1.5e-3, 1.4e-3])

    # O(h^2) fit
    h_fit = np.linspace(h_values.min(), h_values.max(), 100)
    delta_fit = delta_op[0] * (h_fit / h_values[0])**2.0

    fig, ax = setup_figure(width=3.5, height=2.5, nrows=1, ncols=1)
    ax = ax[0, 0]

    ax.loglog(h_values, delta_op, 'o', color='black', markersize=8, 
              label=r'$\\delta_{\\mathrm{op}}$ (expected $\\mathcal{O}(h^2)$)')
    ax.loglog(h_fit, delta_fit, '--', color='red', linewidth=1.5,
              label=r'$\\mathcal{O}(h^2)$ reference')
    ax.axhline(y=0.1375, color='blue', linestyle=':', linewidth=1.5,
               label=r'$\\varepsilon_{\\mathrm{ML}} = 13.75\\%$')

    ax.set_xlabel('Mesh spacing $h$')
    ax.set_ylabel(r'$\\delta_{\\mathrm{op}}$')
    ax.set_xlim(8e-3, 7e-2)
    ax.set_ylim(5e-4, 5e-1)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    save_figure(fig, 'figure_C10_operator_gap')
    plt.close()


if __name__ == '__main__':
    main()
