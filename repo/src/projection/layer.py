"""projection/layer.py — Differentiable Helmholtz Projection Layers (CRITICAL FIX).

CRITICAL FIX (Phase 1, Task 2 corrected):
When interior_mask is provided, the Cholesky factor must be computed from
G[:, interior_mask] @ G[:, interior_mask].T, NOT from G @ G.T.

Mathematical justification:
- Without interior_mask: L = G @ G.T, q = L^{-1} @ G @ a_hat, a_NO = a_hat - G.T @ q
- With interior_mask: L_int = G[:, interior] @ G[:, interior].T, 
  q = L_int^{-1} @ G @ a_hat, a_NO[interior] = a_hat[interior] - (G.T @ q)[interior]

Using G @ G.T instead of G[:, interior] @ G[:, interior].T causes:
- div_norm ~ 1e+03 instead of ~4e-5
- rho ~ 1e+01 instead of ~2e+05
- Idempotency violation (diff ~ 20 instead of ~0)
- Pythagorean identity violation

This fix ensures Proposition 4 (Boundary Invariance) AND Theorem 2 
(Divergence-Free Projection) are both satisfied.
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
    """Dense Cholesky-based Helmholtz projection layer with boundary-safe masking.

    Implements the Algebraic Transplant projection with interior restriction:

    q = (G[:, interior]^T G[:, interior] + eps*I)^{-1} G a_hat       [float64 Cholesky]
    a_NO = a_hat - G^T q                     [divergence-free output]
      WHERE correction is applied ONLY to interior DOFs (Proposition 4)
    p_corr = b_pred + q                      [via Eq. 18, external call]

    CRITICAL FIX: The Cholesky factor is computed from G[:, interior_mask] @ G[:, interior_mask].T
    when interior_mask is provided. Previously, G @ G.T was used, which caused:
    - div_norm ~ 1e+03 (should be ~4e-5)
    - rho ~ 1e+01 (should be ~2e+05)
    - Broken idempotency and Pythagorean identity
    """

    def __init__(self, G: torch.Tensor, eps: float = 1e-8,
                 interior_mask: torch.Tensor = None):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("G", G)
        self.interior_mask = interior_mask

        # CRITICAL FIX: Compute Cholesky from G[:, interior] @ G[:, interior].T
        # when interior_mask is provided.
        if interior_mask is not None:
            G_for_cholesky = G[:, interior_mask]  # (N_rows, N_interior_dof)
        else:
            G_for_cholesky = G  # (N_rows, 2N)

        N_nodes = G_for_cholesky.shape[0]
        jitter = 1.0e-7
        G64 = G_for_cholesky.to(torch.float64)
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
            Raw velocity coefficients, shape (2N,) or (2N, B).
        return_q : bool, optional
            If True, also return the correction vector q.

        Returns
        -------
        a_NO : torch.Tensor
            Divergence-free velocity, same shape as a_hat.
            Boundary DOFs are unchanged when interior_mask is provided.
        q : torch.Tensor (only if return_q=True)
            Pressure correction vector.
        """
        a_col, squeezed = _as_column(a_hat)

        # RHS: G @ a_hat (full divergence of raw prediction)
        Ga = self.G @ a_col

        # Solve: L_interior @ q = G @ a_hat
        # where L_interior = G[:, interior] @ G[:, interior].T (pre-factored in __init__)
        q64 = torch.cholesky_solve(Ga.to(torch.float64), self.chol, upper=False)
        q = q64.to(a_col.dtype)

        # Full correction vector: G^T @ q
        # correction[interior] = (G^T @ q)[interior] = G[:, interior].T @ q
        correction = self.G.T @ q

        # Apply correction ONLY to interior DOFs (Proposition 4)
        if self.interior_mask is not None:
            a_NO = a_col.clone()
            a_NO[self.interior_mask] -= correction[self.interior_mask]
        else:
            a_NO = a_col - correction

        a_NO = a_NO.squeeze(-1) if squeezed else a_NO
        q = q.squeeze(-1) if squeezed else q
        return (a_NO, q) if return_q else a_NO

    def project_only(self, a_hat: torch.Tensor) -> torch.Tensor:
        """Convenience alias for forward(a_hat, return_q=False)."""
        return self.forward(a_hat, return_q=False)


class SparseHelmholtzProjection(nn.Module):
    """Jacobi-preconditioned CG projection for large-scale deployment.

    CRITICAL FIX: When interior_mask is provided, the sparse matvec and
    Jacobi preconditioner use G[:, interior_mask] instead of G.
    """

    def __init__(
        self,
        G: torch.Tensor,
        eps: float = 1e-8,
        tol: float = 1e-5,
        max_iter: int = 1500,
        interior_mask: torch.Tensor = None,
    ):
        super().__init__()
        if not G.is_sparse:
            raise ValueError("SparseHelmholtzProjection expects a sparse COO tensor.")
        self.eps = float(eps)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.interior_mask = interior_mask

        G = G.coalesce()
        self.register_buffer("G", G)

        # CRITICAL FIX: Create G_interior for matvec and preconditioner
        if interior_mask is not None:
            # Extract interior columns from sparse G
            idx = G.indices()  # (2, nnz)
            val = G.values()

            # Keep only entries where column index is in interior_mask
            col_mask = interior_mask[idx[1]]
            interior_idx = idx[:, col_mask]
            interior_val = val[col_mask]

            G_interior = torch.sparse_coo_tensor(
                interior_idx, interior_val,
                (G.shape[0], interior_mask.sum().item()),
                device=G.device, dtype=G.dtype,
            ).coalesce()

            self.register_buffer("G_interior", G_interior)

            # G_interior.T as sparse tensor
            GT_interior = torch.sparse_coo_tensor(
                interior_idx[[1, 0], :], interior_val,
                (interior_mask.sum().item(), G.shape[0]),
                device=G.device, dtype=G.dtype,
            ).coalesce()
            self.register_buffer("GT_interior", GT_interior)

            # Jacobi preconditioner from G_interior
            diag = torch.zeros(G_interior.shape[0], dtype=val.dtype, device=val.device)
            diag.index_add_(0, interior_idx[0], interior_val.square())
        else:
            # Full-domain: use original G
            idx = G.indices()
            val = G.values()

            GT = torch.sparse_coo_tensor(
                idx[[1, 0], :], val, (G.shape[1], G.shape[0]),
                device=G.device, dtype=G.dtype,
            ).coalesce()
            self.register_buffer("GT", GT)

            diag = torch.zeros(G.shape[0], dtype=val.dtype, device=val.device)
            diag.index_add_(0, idx[0], val.square())
            self.register_buffer("G_interior", G)
            self.register_buffer("GT_interior", GT)

        diag = diag + self.eps
        self.register_buffer("M_inv", diag.reciprocal())

    def _matvec_L(self, x: torch.Tensor) -> torch.Tensor:
        """Matrix-vector product with L = G_interior @ G_interior.T + eps*I."""
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        y = torch.sparse.mm(self.GT_interior, x)
        y = torch.sparse.mm(self.G_interior, y)
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
        """Project a_hat onto the discrete divergence-free manifold (PCG)."""
        a_col, squeezed = _as_column(a_hat)

        # RHS uses full G
        rhs = torch.sparse.mm(self.G, a_col)
        q = self._cg(rhs)
        q_col, _ = _as_column(q)
        correction = torch.sparse.mm(self.G.T, q_col)

        # Apply correction ONLY to interior DOFs
        if self.interior_mask is not None:
            a_NO = a_col.clone()
            a_NO[self.interior_mask] -= correction[self.interior_mask]
        else:
            a_NO = a_col - correction

        a_NO = a_NO.squeeze(-1) if squeezed else a_NO
        q = q.squeeze(-1) if squeezed else q
        return (a_NO, q) if return_q else a_NO

    def project_only(self, a_hat: torch.Tensor) -> torch.Tensor:
        """Convenience alias for forward(a_hat, return_q=False)."""
        return self.forward(a_hat, return_q=False)
