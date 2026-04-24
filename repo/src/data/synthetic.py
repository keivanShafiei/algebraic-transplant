"""Synthetic analytically divergence-free dataset (Appendix C).

Used ONLY for architectural sanity checks (Figure 4 / Figure 21).
Not used for main results in Section 4.
See Remark 5: these fields cannot validate projection efficacy because
they are already divergence-free before projection.
"""

import torch
from .cavity import generate_cavity_points   # S4 fix: missing import


def generate_synthetic_streamfunction(n: int = 225, K: int = 10,
                                      seed: int | None = None
                                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytically divergence-free velocity field from random streamfunction.

    Paper Appendix C, Eq. 42:
        ψ(x,y) = Σ_{k=1}^{K}  c_k * sin(m_k π x) * sin(n_k π y)
        u = ∂ψ/∂y = Σ_k  c_k * sin(m_k π x) * (n_k π) * cos(n_k π y)
        v = -∂ψ/∂x = Σ_k  -c_k * (m_k π) * cos(m_k π x) * sin(n_k π y)

    By construction ∇·(u,v) = ∂u/∂x + ∂v/∂y = 0 identically.

    Fixes applied:
        S1: c_k are K=10 scalar coefficients ~ U(-1,1), not randn(N) per-node.
        S2: u/v derived correctly as ∂ψ/∂y and -∂ψ/∂x respectively.
        S3: K=10 random Fourier modes with (m_k,n_k) ∈ {1,...,5}², not 1 mode.
        S4: added missing import of generate_cavity_points.

    Args:
        n    : number of nodes (must be a perfect square for uniform grid)
        K    : number of Fourier modes (paper: K=10)
        seed : optional RNG seed for reproducibility

    Returns:
        a      : (d*N,) flattened velocity coefficients [u_0,v_0,u_1,v_1,...]
        points : (N, 2) node coordinates
    """
    if seed is not None:
        torch.manual_seed(seed)

    points = generate_cavity_points(n)              # (N, 2)
    x, y   = points[:, 0], points[:, 1]            # each (N,)

    # S1 fix: K scalar coefficients ~ U(-1, 1)
    c_k = torch.empty(K).uniform_(-1.0, 1.0)       # (K,)

    # S3 fix: K random wavenumber pairs from {1,...,5}²
    m_k = torch.randint(1, 6, (K,)).float()        # (K,)
    n_k = torch.randint(1, 6, (K,)).float()        # (K,)

    # Evaluate each mode at all nodes: shapes broadcast (N,) x (K,) → (N, K)
    sin_mx = torch.sin(m_k * torch.pi * x.unsqueeze(1))   # (N, K)
    cos_mx = torch.cos(m_k * torch.pi * x.unsqueeze(1))   # (N, K)
    sin_ny = torch.sin(n_k * torch.pi * y.unsqueeze(1))   # (N, K)
    cos_ny = torch.cos(n_k * torch.pi * y.unsqueeze(1))   # (N, K)

    # S2 fix: correct partial derivatives of ψ
    # u = ∂ψ/∂y = Σ_k c_k * sin(m_k π x) * n_k π * cos(n_k π y)
    u = (c_k * n_k * torch.pi * sin_mx * cos_ny).sum(dim=1)   # (N,)

    # v = -∂ψ/∂x = Σ_k -c_k * m_k π * cos(m_k π x) * sin(n_k π y)
    v = -(c_k * m_k * torch.pi * cos_mx * sin_ny).sum(dim=1)  # (N,)

    # Pack as [u_0, v_0, u_1, v_1, ...] to match G coefficient ordering
    a = torch.stack([u, v], dim=1).reshape(-1)     # (2N,)
    return a, points
