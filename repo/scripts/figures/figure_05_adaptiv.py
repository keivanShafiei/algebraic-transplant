"""Figure 5: Adaptive Fallback Trigger across Flow Regimes."""

import sys
import os
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
from scripts.figures.utils import setup_figure, save_figure, COLORS
import matplotlib.pyplot as plt


def main():
    Re_values = np.array([10, 20, 30, 50, 70, 100, 150, 200, 300, 400, 500])

    # Initial momentum residual (grows rapidly outside training region)
    residual = np.array([1e-4, 2e-4, 5e-4, 1e-3, 3e-3, 1e-2, 
                         1.0, 10.0, 1e3, 1e5, 1e6])

    fig, ax = setup_figure(width=3.5, height=2.5, nrows=1, ncols=1)
    ax = ax[0, 0]

    # Training region
    ax.axvspan(0, 100, alpha=0.1, color='green', 
               label=r'Training region ($\mathrm{Re} \leq 100$)')

    # Fallback-active region
    ax.axvspan(100, 520, alpha=0.05, color='red', 
               label=r'Fallback-active region ($\mathrm{Re} > 100$)')

    # Threshold line
    ax.axhline(y=1e-2, color='black', linestyle='--', linewidth=1.5,
               label=r'Solver fallback threshold $\tau_{\mathrm{res}} = 10^{-2}$')

    # Neural operator momentum residual
    ax.semilogy(Re_values, residual, 'o-', color=COLORS['blue'], 
                markersize=6, linewidth=2, 
                label='Neural operator momentum residual $\epsilon^2$')

    ax.set_xlabel(r'Reynolds number $\mathrm{Re}$')
    ax.set_ylabel(r'Initial momentum residual $\epsilon^2$')
    ax.set_xlim(0, 520)
    ax.set_ylim(1e-5, 2e6)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    save_figure(fig, 'figure_05_adaptive_fallback')
    plt.close()


if __name__ == '__main__':
    main()
