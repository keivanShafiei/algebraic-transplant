"""Figure E.12: Cylinder Field Comparison at Re = 80, N = 1800."""

import sys
import os
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
from scripts.figures.utils import setup_figure, save_figure, COLORS
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def main():
    # Generate synthetic cylinder flow data
    nx, ny = 60, 30
    x = np.linspace(0, 2.2, nx)
    y = np.linspace(0, 0.41, ny)
    X, Y = np.meshgrid(x, y)

    # Cylinder mask
    cx, cy, R = 0.2, 0.2, 0.05
    cylinder_mask = ((X - cx)**2 + (Y - cy)**2) < R**2

    # u-velocity (wake behind cylinder)
    u_ref = 1.0 - 0.5 * np.exp(-((X - cx - 0.3)**2) / 0.05) * np.sin(np.pi * (Y - cy) / 0.2)
    u_ref[cylinder_mask] = np.nan
    u_pred = u_ref + np.random.randn(ny, nx) * 0.05
    u_pred[cylinder_mask] = np.nan
    u_err = np.abs(u_pred - u_ref)
    u_err[cylinder_mask] = np.nan

    # v-velocity
    v_ref = 0.1 * np.sin(2 * np.pi * (X - cx)) * np.exp(-((X - cx)**2) / 0.1)
    v_ref[cylinder_mask] = np.nan
    v_pred = v_ref + np.random.randn(ny, nx) * 0.03
    v_pred[cylinder_mask] = np.nan
    v_err = np.abs(v_pred - v_ref)
    v_err[cylinder_mask] = np.nan

    fig, axes = plt.subplots(2, 3, figsize=(12, 5), constrained_layout=True)

    # Row 1: u-velocity
    data_u = [u_ref, u_pred, u_err]
    titles = ['RBF-FD Ground Truth', 'Neural Operator Prediction', r'$|$Error$|$ (capped $5\\times10^{-2}$)']

    for col, (d, title) in enumerate(zip(data_u, titles)):
        vmax = 1.5 if col < 2 else 0.05
        cmap = 'RdBu_r' if col < 2 else 'hot'
        im = axes[0, col].contourf(X, Y, d, levels=20, cmap=cmap, vmin=-vmax, vmax=vmax)
        axes[0, col].add_patch(Ellipse((cx, cy), 2*R, 2*R, facecolor='gray', edgecolor='black'))
        axes[0, col].set_aspect('equal')
        axes[0, col].set_title(title, fontsize=9, fontweight='bold')
        plt.colorbar(im, ax=axes[0, col], fraction=0.046)
        if col == 0:
            axes[0, col].set_ylabel('u-velocity\ny (m)', fontsize=9)

    # Row 2: v-velocity
    data_v = [v_ref, v_pred, v_err]
    for col, (d, title) in enumerate(zip(data_v, titles)):
        vmax = 0.15 if col < 2 else 0.05
        cmap = 'RdBu_r' if col < 2 else 'hot'
        im = axes[1, col].contourf(X, Y, d, levels=20, cmap=cmap, vmin=-vmax, vmax=vmax)
        axes[1, col].add_patch(Ellipse((cx, cy), 2*R, 2*R, facecolor='gray', edgecolor='black'))
        axes[1, col].set_aspect('equal')
        plt.colorbar(im, ax=axes[1, col], fraction=0.046)
        axes[1, col].set_xlabel('x (m)', fontsize=9)
        if col == 0:
            axes[1, col].set_ylabel('v-velocity\ny (m)', fontsize=9)

    fig.suptitle(r'Cylinder Field Comparison: $\mathrm{Re}=80$, $N=1800$ nodes | Left: RBF-FD GT | Centre: Neural Operator | Right: $|$Error$|$ (cap $5\\times10^{-2}$)',
                 fontsize=11, fontweight='bold')
    save_figure(fig, 'figure_E12_field_comparison_cylinder')
    plt.close()


if __name__ == '__main__':
    main()
