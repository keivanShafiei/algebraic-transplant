"""projection/layer.py — Differentiable Helmholtz Projection Layers.

This module implements the core Algebraic Transplant projection:

    q   = (G G^T + eps*I)^{-1} G a_hat          (Lagrange multiplier)
    a_NO = a_hat - G^T q                         (divergence-free velocity)
    p_corr = b_pred + q                          (physical pressure, Eq. 18)

Two implementations are provided:

HelmholtzProjection (dense Cholesky)
    Suitable for N up to ~10,000. Factorises L = G G^T + eps*I in float64
    at construction time, then each forward call is a back-substitution.
    Achieves eps_div ~ O(10^-13) in float64, ~ 4e-5 in float32 inference
    (Remark 2, Table 9).

SparseHelmholtzProjection (Jacobi-PCG)
    Suitable for N up to 100,000+ (validated at N=100,000: 3.08 s, 0.45 GB
    VRAM, eps_div < 1.89e-4, Section 3.4 and Table 17).
    Matrix-free: only sparse mat-vec products. O(N) working memory.

Both classes implement the same interface: forward(a_hat, return_q=False).

Paper reference
---------------
Theorem 2 (Discrete Helmholtz Projection), Section 2.3, Eq. (11):

    P_div(a_hat) := a_hat - G^T (L + eps*I)^{-1} G a_hat

Properties verified by tests/test_projection.py:
    (1) ||G P_div(a_hat)||_2 <= eps ||G a_hat||_2 / sigma_min(L)
    (2) P_div minimises ||v - a_hat|| subject to G v = 0  (Pythagorean identity)
    (3) P_div is differentiable (Jacobian eigenvalues in {0,1} to O(eps))
    (+) P_div is idempotent: P_div(P_div(a)) = P_div(a)

True Algebraic Transplant (Section 4.9, Proposition 4)
-------------------------------------------------------
For boundary-safe projection, G must be the **interior-restricted** operator
G_int (rows corresponding to interior nodes only). Using the full-domain
G_full corrupts Dirichlet boundary velocities and produces ~74% drag error.
The NavierStokesSolver automatically exposes solver.G_int for this purpose.
"""

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn


def _as_column(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Ensure x is a 2-D column tensor. Returns (x_2d, was_squeezed)."""
    if x.dim() == 1:
        return x.unsqueeze(-1), True
    return x, False


class HelmholtzProjection(nn.Module):
    """Dense Cholesky-based Helmholtz projection layer.

    Implements the Algebraic Transplant projection:

        q    = (G G^T + eps*I)^{-1} G a_hat    [float64 Cholesky solve]
        a_NO = a_hat - G^T q                    [divergence-free output]
        p_corr = b_pred + q                     [via Eq. 18, external call]

    The Cholesky factorisation of L = G G^T + eps*I is pre-computed in
    float64 at construction time and stored as a buffer. Each forward
    call performs only a back-substitution (O(N^2) for dense, versus
    O(N^3) for a fresh factorisation).

    Parameters
    ----------
    G : torch.Tensor
        Discrete divergence operator, shape (N_rows, 2N) for interior-
        restricted G_int, or (N, 2N) for full-domain G_full.
        Assembled in float32; promoted to float64 internally.
    eps : float, optional
        Tikhonov regularisation. Default 1e-8 (Table 4 of the paper).
        A small jitter of 1e-7 is added internally for Cholesky stability.

    Attributes
    ----------
    G : torch.Tensor
        Registered buffer holding the divergence operator (float32).
    chol : torch.Tensor
        Registered buffer holding the lower Cholesky factor of
        L + eps*I in float64.

    Notes
    -----
    Precision hierarchy (Remark 2):
    - Cholesky factorisation: float64 -> eps_div ~ O(10^-13)
    - forward() is called with float32 a_hat in training -> eps_div ~ 4e-5
    This is arithmetic, not a design flaw.
    """

    def __init__(self, G: torch.Tensor, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("G", G)

        N_nodes = G.shape[0]
        jitter = 1.0e-7
        G64 = G.to(torch.float64)
        L64 = (G64 @ G64.T) + (self.eps + jitter) * torch.eye(
            N_nodes, dtype=torch.float64, device=G.device
        )
        self.register_buffer("chol", torch.linalg.cholesky(L64))

    def forward(
        self,
        a_hat: torch.Tensor,
        return_q: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Project a_hat onto the discrete divergence-free manifold.

        Parameters
        ----------
        a_hat : torch.Tensor
            Raw velocity coefficients from the GNN decoder,
            shape (2N,) or (2N, B).
        return_q : bool, optional
            If True, also return the correction vector q (the discrete
            Lagrange multiplier / pressure correction). Required for
            pressure recovery: p_corr = b_pred + q (Eq. 18).

        Returns
        -------
        a_NO : torch.Tensor
            Divergence-free velocity, same shape as a_hat.
        q : torch.Tensor (only if return_q=True)
            Pressure correction vector, shape (N_rows,) or (N_rows, B).
        """
        a_col, squeezed = _as_column(a_hat)

        Ga = self.G @ a_col
        q64 = torch.cholesky_solve(Ga.to(torch.float64), self.chol, upper=False)
        q = q64.to(a_col.dtype)

        correction = self.G.T @ q
        a_NO = a_col - correction

        a_NO = a_NO.squeeze(-1) if squeezed else a_NO
        q = q.squeeze(-1) if squeezed else q
        return (a_NO, q) if return_q else a_NO

    def project_only(self, a_hat: torch.Tensor) -> torch.Tensor:
        """Convenience alias for forward(a_hat, return_q=False)."""
        return self.forward(a_hat, return_q=False)


class SparseHelmholtzProjection(nn.Module):
    """Jacobi-preconditioned CG projection for large-scale deployment.

    Replaces the dense Cholesky solve with a matrix-free, iterative
    Conjugate Gradient (CG) solve preconditioned by the Jacobi
    (diagonal) preconditioner M = diag(G G^T + eps*I).

    Complexity comparison:
        HelmholtzProjection  : O(N^2) memory (dense Cholesky factor)
        SparseHelmholtzProjection: O(N) memory (only sparse G + diagonal M)

    Empirical results (Section 3.4, Table 17):
        N = 100,000 nodes: 3.08 s, 0.45 GB peak VRAM, eps_div < 1.89e-4
        N =  49,207 (cylinder): 528 iterations, eps_div < 1e-4

    The PCG replacement is architecturally invariant (Remark 6): G is still
    transplanted exactly, the null space is unchanged, and l2-optimality
    is preserved up to the CG convergence tolerance.

    Parameters
    ----------
    G : torch.Tensor
        Sparse COO divergence operator, shape (N_rows, 2N).
        Must be a sparse tensor (G.is_sparse == True).
    eps : float, optional
        Tikhonov regularisation. Default 1e-8.
    tol : float, optional
        Relative CG residual tolerance. Default 1e-5.
    max_iter : int, optional
        Maximum CG iterations. Default 1500. For N=100,000, convergence
        to 1e-4 requires ~1800 iterations (Figure 9 of the paper).
    """

    def __init__(
        self,
        G: torch.Tensor,
        eps: float = 1e-8,
        tol: float = 1e-5,
        max_iter: int = 1500,
    ):
        super().__init__()
        if not G.is_sparse:
            raise ValueError("SparseHelmholtzProjection expects a sparse COO tensor.")
        self.eps = float(eps)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        G = G.coalesce()
        self.register_buffer("G", G)

        idx = G.indices()
        val = G.values()

        # Precompute G^T as a sparse tensor
        GT = torch.sparse_coo_tensor(
            idx[[1, 0], :], val, (G.shape[1], G.shape[0]),
            device=G.device, dtype=G.dtype,
        ).coalesce()
        self.register_buffer("GT", GT)

        # Jacobi preconditioner: M_inv = 1 / diag(G G^T + eps*I)
        # diag(G G^T)[i] = ||G[i,:]||^2 (sum of squares of row i)
        diag = torch.zeros(G.shape[0], dtype=val.dtype, device=val.device)
        diag.index_add_(0, idx[0], val.square())
        diag = diag + self.eps
        self.register_buffer("M_inv", diag.reciprocal())

    def _matvec_L(self, x: torch.Tensor) -> torch.Tensor:
        """Matrix-vector product with L = G G^T + eps*I."""
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        y = torch.sparse.mm(self.GT, x)
        y = torch.sparse.mm(self.G, y)
        if self.eps != 0.0:
            y = y + self.eps * x
        return y

    def _cg(self, b: torch.Tensor) -> torch.Tensor:
        """Jacobi-preconditioned CG solve for (L + eps*I) q = b."""
        b, squeezed = _as_column(b)
        x = torch.zeros_like(b)
        r = b - self._matvec_L(x)
        z = self.M_inv.unsqueeze(-1) * r
        p = z.clone()
        rz_old = torch.sum(r * z, dim=0, keepdim=True)
        b_norm = torch.linalg.norm(b).clamp_min(1e-12)

        for _ in range(self.max_iter):
            Ap = self._matvec_L(p)
            denom = torch.sum(p * Ap, dim=0, keepdim=True).clamp_min(1e-30)
            alpha = rz_old / denom
            x = x + p * alpha
            r = r - Ap * alpha

            if torch.linalg.norm(r).item() < self.tol * b_norm.item():
                break

            z = self.M_inv.unsqueeze(-1) * r
            rz_new = torch.sum(r * z, dim=0, keepdim=True)
            beta = rz_new / rz_old.clamp_min(1e-30)
            p = z + p * beta
            rz_old = rz_new

        return x.squeeze(-1) if squeezed else x

    def forward(
        self,
        a_hat: torch.Tensor,
        return_q: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Project a_hat onto the discrete divergence-free manifold (PCG).

        Parameters
        ----------
        a_hat : torch.Tensor
            Raw velocity coefficients, shape (2N,) or (2N, B).
        return_q : bool, optional
            If True, also return the pressure correction q.

        Returns
        -------
        a_NO : torch.Tensor
            Divergence-free velocity, same shape as a_hat.
        q : torch.Tensor (only if return_q=True)
            Pressure correction vector.
        """
        a_col, squeezed = _as_column(a_hat)

        rhs = torch.sparse.mm(self.G, a_col)
        q = self._cg(rhs)
        q_col, _ = _as_column(q)
        correction = torch.sparse.mm(self.GT, q_col)

        a_NO = a_col - correction

        a_NO = a_NO.squeeze(-1) if squeezed else a_NO
        q = q.squeeze(-1) if squeezed else q
        return (a_NO, q) if return_q else a_NO

    def project_only(self, a_hat: torch.Tensor) -> torch.Tensor:
        """Convenience alias for forward(a_hat, return_q=False)."""
        return self.forward(a_hat, return_q=False)
