"""rbf_fd/kernel.py — Multiquadric RBF Kernel and Derivatives.

This module implements the **multiquadric (MQ) radial basis function** kernel
and its first- and second-order derivatives. These functions are the building
blocks for assembling the discrete differential operators (divergence G,
interpolation Phi, and Laplacian) used in both the RBF-FD solver and the
Algebraic Transplant projection layer.

Paper reference
---------------
Section 2.2 (Spatial discretisation), Appendix B (RBF Profiles and Stencil
Validation), Eqs. (4)-(6) and (B.1)-(B.2).

Mathematical background
-----------------------
The MQ kernel is defined as:

    phi(r) = sqrt(1 + (r/c)^2)    (C-inf, strictly positive definite)

where r = ||x - x_j||_2 is the Euclidean distance between two nodes and
c is the shape parameter, set to c = 1.2 * h_avg in all experiments.

The first derivative assembles the divergence operator G (Eq. 6):

    d_phi/dr = r / (c^2 * sqrt(1 + (r/c)^2))

The Laplacian assembles the diffusion operator for the momentum equation:

    Lap_phi(r) = d / (c^2 * sqrt(1+(r/c)^2)) - r^2 / (c^4 * (1+(r/c)^2)^{3/2})

where d is the spatial dimension (2 for all paper experiments).

Important numerical check (Appendix B):

    Lap_phi(0) = d / c^2 ≈ 272.22   (N=225, d=2, c ≈ 0.08571)

This peak value is verified by tests/test_consistency.py::test_laplacian_peak_value.
"""

import torch


def mq_phi(r: torch.Tensor, c: float) -> torch.Tensor:
    """Multiquadric kernel value phi(r) = sqrt(1 + (r/c)^2).

    Parameters
    ----------
    r : torch.Tensor
        Euclidean distances, shape arbitrary, dtype float32 or float64.
        Must be non-negative. Self-distances (r=0) give phi(0) = 1.
    c : float
        Shape parameter. Must be strictly positive.

    Returns
    -------
    torch.Tensor
        Kernel values, same shape and dtype as r.
    """
    return torch.sqrt(1.0 + (r / c) ** 2)


def mq_dphi_dr(r: torch.Tensor, c: float) -> torch.Tensor:
    """First derivative of the MQ kernel: d_phi/dr = r / (c^2 * phi(r)).

    This is the key ingredient in assembling the discrete divergence
    operator G (Eq. 6 of the paper). The gradient vector is obtained by
    multiplying by the unit direction (x_i - x_j) / r.

    Parameters
    ----------
    r : torch.Tensor
        Euclidean distances (non-negative). Shape arbitrary.
    c : float
        Shape parameter (strictly positive).

    Returns
    -------
    torch.Tensor
        Derivative values d_phi/dr, same shape as r.
        At r=0 the derivative is exactly 0 by L'Hopital.
    """
    return r / (c ** 2 * torch.sqrt(1.0 + (r / c) ** 2))


def mq_laplacian(r: torch.Tensor, c: float, d: int = 2) -> torch.Tensor:
    """Laplacian of the MQ kernel in d spatial dimensions.

    Lap_phi(r) = d / (c^2 * phi(r)) - r^2 / (c^4 * phi(r)^3)

    Used to assemble the discrete diffusion/Laplacian operator for the
    viscous term in the momentum equation.

    Parameters
    ----------
    r : torch.Tensor
        Euclidean distances. Must be clamped away from 0 by the caller
        (e.g., r.clamp(min=1e-12)) to avoid numerical issues.
    c : float
        Shape parameter (strictly positive).
    d : int, optional
        Spatial dimension. Default 2. The coefficient of the first term
        equals d (not a fixed constant). A prior bug used 3 instead of d,
        producing incorrect values for d=2.

    Returns
    -------
    torch.Tensor
        Laplacian values, same shape as r.

    Notes
    -----
    Dimensional check (Appendix B): at r ~ 0,
        Lap_phi(0) = d / c^2 ≈ 272.22   for N=225, d=2, c≈0.08571.
    """
    denom_sqrt = torch.sqrt(1.0 + (r / c) ** 2)      # phi(r)
    term1 = d / (c ** 2 * denom_sqrt)                 # d / (c^2 * phi)
    term2 = -r ** 2 / (c ** 4 * denom_sqrt ** 3)      # -r^2 / (c^4 * phi^3)
    return term1 + term2
