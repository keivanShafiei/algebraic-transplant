"""Shared utilities for publication-ready figures."""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# Publication settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'legend.frameon': True,
    'legend.edgecolor': '0.8',
    'legend.fancybox': False,
    'mathtext.fontset': 'stix',  # STIX math fonts match Times
})

# Color palette (colorblind-friendly, B&W print compatible)
COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'olive': '#bcbd22',
    'cyan': '#17becf',
}

# Line styles for B&W compatibility
LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

# Marker styles
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']


def setup_figure(width=3.5, height=2.5, nrows=1, ncols=1):
    """Create a figure with publication dimensions.

    Parameters
    ----------
    width : float
        Single panel width in inches (3.5 = single column, 7.0 = double).
    height : float
        Single panel height in inches.
    nrows, ncols : int
        Number of subplots.

    Returns
    -------
    fig, axes
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), 
                             constrained_layout=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)
    return fig, axes


def save_figure(fig, name, output_dir='results/figures'):
    """Save figure as PDF with proper naming."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / f"{name}.pdf"
    fig.savefig(filepath, format='pdf', dpi=300)
    print(f"Saved: {filepath}")
    return filepath


def format_scientific(axis, axis_name='y'):
    """Format axis with scientific notation."""
    ax = getattr(axis, axis_name + 'axis')
    ax.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.get_offset_text().set_fontsize(8)


def add_panel_label(ax, label, x=-0.15, y=1.05):
    """Add (a), (b), (c), (d) labels to panels."""
    ax.text(x, y, f"({label})", transform=ax.transAxes, 
            fontsize=11, fontweight='bold', va='top', ha='right')


def lighten_color(color, amount=0.5):
    """Lighten a color for fill_between."""
    import matplotlib.colors as mcolors
    c = mcolors.to_rgb(color)
    return tuple(1 - (1 - x) * amount for x in c)
