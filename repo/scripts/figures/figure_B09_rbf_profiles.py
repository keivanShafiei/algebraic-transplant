"""Figure B9: RBF kernel profiles (Appendix B).

Shows MQ RBF and its derivatives vs. radius r.
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

def mq_phi(r, c):
    return np.sqrt(1 + (r / c) ** 2)

def mq_dphi(r, c):
    return r / (c**2 * np.sqrt(1 + (r / c) ** 2))

def mq_laplacian(r, c, d=2):
    return d / (c**2 * (1 + (r / c) ** 2) ** 1.5)

def main():
    c = 1.2 * (1.0 / np.sqrt(225))  # c = 1.2 * h_avg for N=225
    r = np.linspace(0, 3 * c, 200)

    phi = mq_phi(r, c)
    dphi = mq_dphi(r, c)
    lap = mq_laplacian(r, c, d=2)

    fig, axes = setup_figure(width=3.5, height=2.5, nrows=1, ncols=1)
    ax = axes[0, 0]

    ax.plot(r, phi, '-', color=COLORS['blue'], linewidth=1.5, label=r'$\phi(r) = \sqrt{1+(r/c)^2}$')
    ax.plot(r, dphi, '--', color=COLORS['orange'], linewidth=1.5, label=r"$\phi'(r)$")
    ax.plot(r, lap, '-.', color=COLORS['green'], linewidth=1.5, label=r"$\nabla^2 \phi(r)$")

    ax.axvline(x=c, color='gray', linestyle=':', alpha=0.7, label=f'$c = {c:.4f}$')
    ax.set_xlabel(r'$r$')
    ax.set_ylabel('Function value')
    ax.set_xlim(0, 3 * c)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    add_panel_label(ax, 'a')

    fig.suptitle('Multiquadric RBF Kernel Profiles', fontsize=12, fontweight='bold', y=1.02)
    save_figure(fig, 'figure_B09_rbf_profiles')
    plt.close()
    print("Figure B9 generated successfully.")

if __name__ == '__main__':
    main()
