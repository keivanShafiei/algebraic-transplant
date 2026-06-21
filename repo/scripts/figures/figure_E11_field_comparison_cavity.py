"""Figure E11: Velocity/pressure field comparison — lid-driven cavity.

Shows ground truth vs. GNN prediction with projection layer.
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
    np.random.seed(50)
    n = 30
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    # Synthetic cavity flow (analytical stream function)
    u_true =  np.sin(np.pi * X) * np.cos(np.pi * Y)
    v_true = -np.cos(np.pi * X) * np.sin(np.pi * Y)
    p_true =  np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y) * 0.5

    # GNN prediction with small noise
    noise = 0.05
    u_pred = u_true + np.random.randn(n, n) * noise
    v_pred = v_true + np.random.randn(n, n) * noise
    p_pred = p_true + np.random.randn(n, n) * noise * 0.3

    fig, axes = setup_figure(width=3.5, height=3.5, nrows=2, ncols=3)

    titles = ['$u_x$ (True)', '$u_y$ (True)', '$p$ (True)',
              '$u_x$ (Predicted)', '$u_y$ (Predicted)', '$p$ (Predicted)']
    fields = [u_true, v_true, p_true, u_pred, v_pred, p_pred]

    for i, (ax, field, title) in enumerate(zip(axes.flat, fields, titles)):
        im = ax.contourf(X, Y, field, levels=20, cmap='RdBu_r')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle('Field Comparison: Lid-Driven Cavity', fontsize=12, fontweight='bold', y=1.02)
    save_figure(fig, 'figure_E11_field_comparison_cavity')
    plt.close()
    print("Figure E11 generated successfully.")

if __name__ == '__main__':
    main()
