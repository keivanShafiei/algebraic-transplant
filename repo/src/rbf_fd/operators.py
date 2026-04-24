"""rbf_fd/operators.py — Discrete Differential Operator Assembly.

Assembles the three key matrices used by the RBF-FD solver and the
Algebraic Transplant framework:

G   : R^{N x 2N}  — discrete divergence operator  (transplanted into GNN)
Phi : R^{N x N}   — RBF interpolation matrix
Lap : R^{N x N}   — discrete Laplacian (viscous term)

The most important matrix is G: it is assembled here and then used both
in the solver's fractional-step projection (Algorithm 1) and in the
neural operator's Helmholtz projection layer (Stage 4 of the architecture).
This shared, exact matrix is the essence of the Algebraic Transplant.

Paper reference
---------------
Section 2.2 (Spatial discretisation), Eq. (6):

    [G]_{ij} = nabla phi_j(x_i) . e_{comp(j)}

where e_{comp(j)} is the canonical basis vector for the spatial component
of velocity index j (x-component if j is even, y-component if j is odd).

Section 2.5 (Algebraic Transplant, Principle 2, item (ii)):
"The matrix G is taken verbatim from the solver assembly (Eq. 6) and
frozen as a parameter-free layer."
"""

import torch
from .kernel import mq_phi, mq_dphi_dr, mq_laplacian


def assemble_divergence_operator(
    points: torch.Tensor,
    stencils: torch.Tensor,
    c: float,
    sparse: bool = False,
) -> torch.Tensor:
    """Assemble the discrete RBF-FD divergence operator G.

    Constructs G in R^{N x 2N} where entry [G]_{ij} represents the
    contribution of velocity DOF j to the divergence constraint at node i.
    DOFs are interleaved: column 2j corresponds to u_x at node j,
    column 2j+1 corresponds to u_y at node j.

    The assembly iterates over each node's k-NN stencil, evaluates the
    gradient of the MQ basis function d_phi/dr * (Delta_x / r), and
    scatters the result into the appropriate (row, col) positions.

    Complexity: O(N * k * d) — linear in mesh size (Table 2, claim C1).

    Parameters
    ----------
    points : torch.Tensor
        Node coordinates, shape (N, d), float32.
    stencils : torch.Tensor
        k-NN indices, shape (N, k), int64. From build_stencils().
    c : float
        MQ shape parameter (c = 1.2 * h_avg, set by the solver).
    sparse : bool, optional
        If True, return a sparse COO tensor (for use with
        SparseHelmholtzProjection at large N). Default False (dense tensor
        for small/medium N and for training).

    Returns
    -------
    torch.Tensor
        Divergence operator G, shape (N, 2N), float32.
        If sparse=True, returns a torch.sparse_coo_tensor of the same shape.

    Notes
    -----
    The operator is assembled in float32. For the projection solve,
    the caller (HelmholtzProjection.__init__) promotes G to float64 before
    forming L = G G^T and computing the Cholesky factorisation, achieving
    the O(10^-13) divergence floor documented in Remark 2.
    """
    N, d = points.shape[0], points.shape[1]
    k = stencils.shape[1]

    row_idx = torch.arange(N, device=points.device)
    neigh_pts = points[stencils]                              # (N, k, d)
    src_pts = points.unsqueeze(1).expand_as(neigh_pts)        # (N, k, d)

    dx = src_pts - neigh_pts                                  # (N, k, d)
    r = torch.norm(dx, dim=-1, keepdim=True).clamp(min=1e-12) # (N, k, 1)

    dphi_dr = mq_dphi_dr(r, c)                               # (N, k, 1)
    grad_phi = (dx / r) * dphi_dr                            # (N, k, d)

    # Build COO indices: row i, col = neigh_j * d + dim
    rows = row_idx.repeat_interleave(k * d)
    neigh_idx = stencils.unsqueeze(-1).expand(N, k, d)
    dims = torch.arange(d, device=points.device).view(1, 1, d).expand(N, k, d)
    cols = (neigh_idx * d + dims).reshape(-1)
    vals = grad_phi.reshape(-1).to(torch.float32)

    if sparse:
        indices = torch.stack([rows, cols], dim=0)
        return torch.sparse_coo_tensor(
            indices, vals, size=(N, d * N),
            device=points.device, dtype=torch.float32,
        ).coalesce()

    G = torch.zeros((N, d * N), dtype=torch.float32, device=points.device)
    G.index_put_((rows, cols), vals, accumulate=True)
    return G


def assemble_phi_stencil(
    points: torch.Tensor,
    stencils: torch.Tensor,
    c: float,
) -> torch.Tensor:
    """Assemble the RBF interpolation matrix Phi.

    Phi[i, j] = phi(||x_i - x_j||) for j in stencil(i), else 0.
    Used to evaluate the interpolated velocity field u_h = Phi @ a at
    each collocation node (needed to form the nonlinear convection term).

    Parameters
    ----------
    points : torch.Tensor, shape (N, d), float32.
    stencils : torch.Tensor, shape (N, k), int64.
    c : float
        MQ shape parameter.

    Returns
    -------
    torch.Tensor
        Phi, shape (N, N), float32.
    """
    N = points.shape[0]
    k = stencils.shape[1]

    neigh_pts = points[stencils]
    src_pts = points.unsqueeze(1).expand_as(neigh_pts)
    r = torch.norm(src_pts - neigh_pts, dim=-1)

    phi_vals = mq_phi(r, c)

    rows = torch.arange(N, device=points.device).repeat_interleave(k)
    cols = stencils.reshape(-1)

    Phi = torch.zeros(N, N, dtype=torch.float32, device=points.device)
    Phi.index_put_((rows, cols), phi_vals.reshape(-1), accumulate=True)
    return Phi


def assemble_laplacian_stencil(
    points: torch.Tensor,
    stencils: torch.Tensor,
    c: float,
) -> torch.Tensor:
    """Assemble the discrete Laplacian operator matrix.

    Lap[i, j] = nabla^2 phi_j(x_i) for j in stencil(i), else 0.
    Used to form the diffusion term (1/Re) * nabla^2 u in the momentum
    equation.

    Note (Remark 1): strong-form collocation produces a non-symmetric
    Laplacian even in the Stokes limit. This is expected and does not
    compromise consistency.

    Parameters
    ----------
    points : torch.Tensor, shape (N, d), float32.
    stencils : torch.Tensor, shape (N, k), int64.
    c : float
        MQ shape parameter.

    Returns
    -------
    torch.Tensor
        Lap, shape (N, N), float32.
    """
    N, d = points.shape[0], points.shape[1]
    k = stencils.shape[1]

    neigh_pts = points[stencils]
    src_pts = points.unsqueeze(1).expand_as(neigh_pts)
    r = torch.norm(src_pts - neigh_pts, dim=-1).clamp(min=1e-12)

    lap_vals = mq_laplacian(r, c, d=d)

    rows = torch.arange(N, device=points.device).repeat_interleave(k)
    cols = stencils.reshape(-1)

    Lap = torch.zeros(N, N, dtype=torch.float32, device=points.device)
    Lap.index_put_((rows, cols), lap_vals.reshape(-1), accumulate=True)
    return Lap
