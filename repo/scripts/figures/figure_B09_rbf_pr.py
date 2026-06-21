"""Figure B.9: Multiquadric Kernel and Derivatives."""

import sys
import os
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
from scripts.figures.utils import setup_figure, save_figure, COLORS
import matplotlib.pyplot as plt


def mq(r, c):
    return np.sqrt(1 + (r/c)**2)


def dphi_dr(r, c):
    return r / (c**2 * np.sqrt(1 + (r/c)**2))


def laplacian_phi(r, c):
    return 2/(c**2 * np.sqrt(1 + (r/c)**2)) - r**2/(c**4 * (1 + (r/c)**2)**1.5)


def main():
    h = 0.05
    c = 0.06
    r = np.linspace(0, 0.3, 500)

    phi = mq(r, c)
    dphi = dphi_dr(r, c)
    lap = laplacian_phi(r, c)

    fig, axes = setup_figure(width=5.0, height=2.0, nrows=1, ncols=3)

    # Panel 1: phi(r)
    ax = axes[0, 0]
    ax.plot(r, phi/c, '-', color=COLORS['blue'], linewidth=1.5)
    ax.plot(0, 1.0, 'o', color=COLORS['blue'], markersize=6)
    ax.plot(h, mq(h,c)/c, 'o', color=COLORS['blue'], markersize=6)
    ax.text(h+0.01, mq(h,c)/c, f'$\\tilde{{\\phi}}(h)={mq(h,c)/c:.3f}$', fontsize=8)
    ax.set_xlabel('Distance $r$')
    ax.set_ylabel(r'$\\tilde{\\phi}(r) = \\phi(r)/c$')
    ax.set_title(r'Multiquadric Kernel ($C^\\infty$)')
    ax.grid(True, alpha=0.3)

    # Panel 2: dphi/dr
    ax = axes[0, 1]
    ax.plot(r, dphi * c, '-', color=COLORS['red'], linewidth=1.5)
    ax.plot(h, dphi_dr(h,c)*c, 'o', color=COLORS['red'], markersize=6)
    ax.text(h+0.01, dphi_dr(h,c)*c, f'$\\approx10.67$ at $r=h$', fontsize=8)
    ax.axhline(y=1/c, color='gray', linestyle='--', alpha=0.7, label=f'$1/c = {1/c:.2f}$')
    ax.set_xlabel('Distance $r$')
    ax.set_ylabel(r'$(1/c) \\partial\\phi/\\partial r$')
    ax.set_title(r'First Derivative $\\partial\\phi/\\partial r$')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Laplacian
    ax = axes[0, 2]
    ax.plot(r, lap * c**2, '-', color=COLORS['green'], linewidth=1.5)
    ax.plot(0, 2/c**2, 'o', color=COLORS['green'], markersize=6)
    ax.plot(h, laplacian_phi(h,c)*c**2, 'o', color=COLORS['green'], markersize=6)
    ax.text(h+0.01, laplacian_phi(h,c)*c**2, f'$\\approx339.3$ at $r=h$', fontsize=8)
    ax.set_xlabel('Distance $r$')
    ax.set_ylabel(r'$(1/c^2) \\nabla^2\\phi(r)$')
    ax.set_title(r'2D Laplacian $\\nabla^2\\phi(r)$')
    ax.grid(True, alpha=0.3)

    save_figure(fig, 'figure_B09_rbf_profiles')
    plt.close()


if __name__ == '__main__':
    main()
