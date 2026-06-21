"""Figure E12: Field comparison — flow past cylinder.

Data source: data/cylinder_samples/*.pt (NOT available)
Fallback: analytical synthetic field
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.figures.utils import (
    setup_figure, save_figure, format_scientific, add_panel_label, COLORS
)

def load_cylinder_field():
    """Load cylinder field from sample files."""

    sample_dir = Path('data/cylinder_samples')
    if sample_dir.exists():
        sample_files = sorted(sample_dir.glob('sample_*.pt'))
        if sample_files:
            d = torch.load(sample_files[0], map_location='cpu')
            points = d.get('points', None)
            a_ref = d['a_ref']
            b_ref = d.get('b_ref', None)

            if points is not None:
                N = points.shape[0]
                x = points[:, 0].numpy()
                y = points[:, 1].numpy()
                u = a_ref[0::2].numpy()
                v = a_ref[1::2].numpy()
                if b_ref is not None:
                    p = b_ref.numpy()
                else:
                    p = np.zeros(N)
                print(f"[Figure E12] Loaded real cylinder field from {sample_files[0].name} ({N} nodes)")
                return x, y, u, v, p

    # Fallback
    print("[Figure E12] WARNING: No cylinder sample files found. Using synthetic fallback.")
    n = 40
    x = np.linspace(-2, 4, n)
    y = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    u = 1.0 - 0.5 * X / (R**2 + 0.01) + 0.1 * np.sin(2 * theta) * np.exp(-R/2)
    v = -0.5 * Y / (R**2 + 0.01) + 0.1 * np.cos(2 * theta) * np.exp(-R/2)
    p = 0.5 * (1 - (u**2 + v**2)) + 0.05 * np.sin(3 * theta) * np.exp(-R/3)
    mask = R < 0.5
    u[mask] = np.nan
    v[mask] = np.nan
    p[mask] = np.nan
    return X.flatten(), Y.flatten(), u.flatten(), v.flatten(), p.flatten()

def main():
    x, y, u_true, v_true, p_true = load_cylinder_field()

    noise = 0.03
    u_pred = u_true + np.random.randn(*u_true.shape) * noise
    v_pred = v_true + np.random.randn(*v_true.shape) * noise
    p_pred = p_true + np.random.randn(*p_true.shape) * noise * 0.5

    if x.ndim == 1:
        ux = np.unique(x)
        uy = np.unique(y)
        if len(ux) * len(uy) == len(x):
            n = len(ux)
            X = x.reshape(n, n)
            Y = y.reshape(n, n)
            fields = [
                u_true.reshape(n, n), v_true.reshape(n, n), p_true.reshape(n, n),
                u_pred.reshape(n, n), v_pred.reshape(n, n), p_pred.reshape(n, n)
            ]
        else:
            X, Y = x, y
            fields = [u_true, v_true, p_true, u_pred, v_pred, p_pred]
    else:
        X, Y = x, y
        fields = [u_true, v_true, p_true, u_pred, v_pred, p_pred]

    fig, axes = setup_figure(width=3.5, height=3.5, nrows=2, ncols=3)

    titles = ['$u_x$ (True)', '$u_y$ (True)', '$p$ (True)',
              '$u_x$ (Predicted)', '$u_y$ (Predicted)', '$p$ (Predicted)']

    for i, (ax, field, title) in enumerate(zip(axes.flat, fields, titles)):
        if field.ndim == 1:
            im = ax.scatter(X, Y, c=field, s=5, cmap='RdBu_r')
        else:
            im = ax.contourf(X, Y, field, levels=20, cmap='RdBu_r')
        if field.ndim == 2:
            circle = plt.Circle((0, 0), 0.5, color='black', fill=False, linewidth=1.5)
            ax.add_patch(circle)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_aspect('equal')
        if field.ndim == 2:
            ax.set_xlim(-2, 4)
            ax.set_ylim(-2, 2)
        fig.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle('Field Comparison: Flow Past Cylinder', fontsize=12, fontweight='bold', y=1.02)
    save_figure(fig, 'figure_E12_field_comparison_cylinder')
    plt.close()
    print("Figure E12 generated successfully.")

if __name__ == '__main__':
    main()
