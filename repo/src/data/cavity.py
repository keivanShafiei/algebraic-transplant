"""data/cavity.py — Lid-Driven Cavity Node Generation.

Generates scattered node sets for the lid-driven cavity benchmark
Omega = [0,1]^2 (Section 4, Experimental scope).

The lid-driven cavity is the primary benchmark used throughout the paper:
    - Training set: Re in [10, 100], Ns = 364 samples
    - Test set: 80 held-out samples at the same Re range
    - Resolution study: N in {225, 961, 1000, 5000, 10000}
    - Extrapolation test: Re = 500 (5x training maximum, Section 4.7)
    - Warm-start: Re = 500 (Table 13)
"""

import torch


def generate_cavity_points(n: int = 225) -> torch.Tensor:
    """Generate a uniform Cartesian grid on the unit square [0,1]^2.

    The number of nodes per side is int(sqrt(n)), so the actual node
    count is floor(sqrt(n))^2 (a perfect square). For n=225 this gives
    exactly 225 nodes on a 15x15 grid.

    Parameters
    ----------
    n : int, optional
        Target number of nodes. Must be a perfect square (default 225).
        Paper experiments use n in {225, 961, 1000, 5000, 10000}.

    Returns
    -------
    torch.Tensor
        Node coordinates, shape (n, 2), dtype float32, on CPU.
        Ordering: row-major (x varies fastest).

    Notes
    -----
    For Poisson-disk sampled irregular meshes (used in the large-scale
    PCG tests with N=100,000), see the separate mesh-generation utilities
    documented in test_scalability.py.
    """
    side = int(n ** 0.5)
    x = torch.linspace(0, 1, side, dtype=torch.float32)
    y = torch.linspace(0, 1, side, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    return points
