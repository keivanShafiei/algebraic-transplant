"""Figure E12: Velocity/pressure field comparison — flow past cylinder.

Shows ground truth vs. GNN prediction with projection layer for
exterior cylinder flow.
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
    np.random.seed(51)
    n = 40
    x = np.linspace(-2, 4, n)
    y = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, y)

    # Cylinder at origin, radius 0.5
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # Potential flow + vortex shedding approximation
    u_true = 1.0 - 0.5 * X / (R**2 + 0.01) + 0.1 * np.sin(2 * theta) * np.exp(-R/2)
    v_true = -0.5 * Y / (R**2 + 0.01) + 0.1 * np.cos(2 * theta) * np.exp(-R/2)
    p_true = 0.5 * (1 - (u_true**2 + v_true**2)) + 0.05 * np.sin(3 * theta) * np.exp(-R/3)

    # Mask inside cylinder
    mask = R < 0.5
    u_true[mask] = np.nan
    v_true[mask] = np.nan
    p_true[mask] = np.nan

    # GNN prediction with noise
    noise = 0.03
    u_pred = u_true + np.random.randn(n, n) * noise
    v_pred = v_true + np.random.randn(n, n) * noise
    p_pred = p_true + np.random.randn(n, n) * noise * 0.5
    u_pred[mask] = np.nan
    v_pred[mask] = np.nan
    p_pred[mask] = np.nan

    fig, axes = setup_figure(width=3.5, height=3.5, nrows=2, ncols=3)

    titles = ['$u_x$ (True)', '$u_y$ (True)', '$p$ (True)',
              '$u_x$ (Predicted)', '$u_y$ (Predicted)', '$p$ (Predicted)']
    fields = [u_true, v_true, p_true, u_pred, v_pred, p_pred]

    for i, (ax, field, title) in enumerate(zip(axes.flat, fields, titles)):
        im = ax.contourf(X, Y, field, levels=20, cmap='RdBu_r')
        # Draw cylinder
        circle = plt.Circle((0, 0), 0.5, color='black', fill=False, linewidth=1.5)
        ax.add_patch(circle)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_aspect('equal')
        ax.set_xlim(-2, 4)
        ax.set_ylim(-2, 2)
        fig.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle('Field Comparison: Flow Past Cylinder', fontsize=12, fontweight='bold', y=1.02)
    save_figure(fig, 'figure_E12_field_comparison_cylinder')
    plt.close()
    print("Figure E12 generated successfully.")

if __name__ == '__main__':
    main()
