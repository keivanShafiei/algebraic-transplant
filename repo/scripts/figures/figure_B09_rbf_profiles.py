"""Figure B9: RBF kernel profiles with twin y-axes.

Reads c from: config.yaml (if available)
Fallback: c = 1.2 * h_avg for N=225
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

def load_c_parameter():
    """Load c from config.yaml if available."""
    config_path = Path('config.yaml')
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            if 'rbf_c_factor' in config and 'n_nodes_list' in config:
                N = config['n_nodes_list'][0]
                h = 1.0 / np.sqrt(N)
                c = config['rbf_c_factor'] * h
                print(f"[Figure B9] Loaded c={c:.4f} from config.yaml")
                return c
        except Exception:
            pass

    c = 1.2 * (1.0 / np.sqrt(225))
    print(f"[Figure B9] Using default c={c:.4f}")
    return c

def main():
    c = load_c_parameter()
    r = np.linspace(0, 3 * c, 200)
    phi = mq_phi(r, c)
    dphi = mq_dphi(r, c)
    lap = mq_laplacian(r, c, d=2)

    fig, axes = setup_figure(width=3.5, height=2.5, nrows=1, ncols=1)
    ax = axes[0, 0]

    ax.plot(r, phi, '-', color=COLORS['blue'], linewidth=1.5, label=r'$\phi(r)$')
    ax.plot(r, dphi, '--', color=COLORS['orange'], linewidth=1.5, label=r"$\phi'(r)$")
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$\phi(r)$, $\phi\'(r)$', color=COLORS['blue'])
    ax.tick_params(axis='y', labelcolor=COLORS['blue'])
    ax.set_ylim(0, 1.5)

    ax2 = ax.twinx()
    ax2.plot(r, lap, '-.', color=COLORS['green'], linewidth=1.5, label=r"$\nabla^2 \phi(r)$")
    ax2.set_ylabel(r"$\nabla^2 \phi(r)$", color=COLORS['green'])
    ax2.tick_params(axis='y', labelcolor=COLORS['green'])
    ax2.set_ylim(0, 350)

    ax.axvline(x=c, color='gray', linestyle=':', alpha=0.7)
    ax.text(c, 1.35, f'$c={c:.4f}$', fontsize=8, color='gray', ha='center')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    ax.grid(True, alpha=0.3)
    add_panel_label(ax, 'a')

    fig.suptitle('Multiquadric RBF Kernel Profiles', fontsize=12, fontweight='bold', y=1.02)
    save_figure(fig, 'figure_B09_rbf_profiles')
    plt.close()
    print("Figure B9 generated successfully.")

if __name__ == '__main__':
    main()
