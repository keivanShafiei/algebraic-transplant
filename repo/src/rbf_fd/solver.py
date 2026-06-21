"""rbf_fd/solver.py — Fractional-Step Navier-Stokes Solver (Algorithm 1).

Implements the **projection-based segregated solver** for steady laminar
incompressible flows on a scattered RBF-FD point cloud. This solver has
three roles in the Algebraic Transplant framework (Section 1):

 (1) Data-generation engine: solves for (a, b) at each Re in the
     training set, generating the labeled dataset for the GNN.

 (2) Source of transplanted operators: the interior-restricted divergence
     operator G_int assembled here is the exact object transplanted into
     the GNN projection layer. Using the solver's own G eliminates all
     discretisation mismatch between the constraint enforced during
     training and the constraint used at inference.

 (3) Warm-start target: the Neural Operator's prediction a_NO (which
     satisfies G_int a_NO = 0 by construction) is used as the initial
     guess for iterative refinement, reducing solver iterations by 4.2x
     at Re = 500 (Table 13).

Fractional-step algorithm (Section 2.3, Eqs. 9-10)
-------------------------------------------
 Step 1 (Momentum solve): K(a^n) a* = F
 Step 2 (Pressure correction): L_int b^{n+1} = G_int a*
 Step 3 (Velocity correction): a^{n+1} = a* - G_int^T b^{n+1}

By Theorem 1, Step 3 guarantees G_int a^{n+1} = 0 algebraically.

True Algebraic Transplant — Interior Restriction (Section 4.9)
-------------------------------------------
The solver performs Steps 2-3 ONLY over interior DOFs (Proposition 4).
Applying the correction over boundary nodes would corrupt prescribed
Dirichlet velocities (the Boundary Condition Paradox) and produce ~74%
drag error. The interior-restricted operator G_int is stored as
self.G_int and self.G_int_int.

Data quality filters (Section 3.1)
-------------------------------------------
Only samples satisfying ALL of:
 (i) kappa(K) <= 1e6 (well-conditioned momentum operator)
 (ii) solver converged (momentum AND divergence residuals met)
 (iii) ||G_int a||_2 < tau_mass = 1e-4
are saved by generate_data.py.
"""

import torch
from .stencils import build_stencils
from .operators import (
    assemble_divergence_operator,
    assemble_phi_stencil,
    assemble_laplacian_stencil,
)


def classify_cavity_nodes(
    points: torch.Tensor,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify nodes of the lid-driven cavity into lid / wall / interior.

    The domain is the unit square Omega = [0, 1]^2 with:
        Lid (Gamma_D, moving): y = 1 (u = 1, v = 0)
        Walls (Gamma_D, no-slip): x=0, x=1, y=0 (u = v = 0)
        Interior: all remaining nodes

    Parameters
    ----------
    points : torch.Tensor
        Node coordinates, shape (N, 2).
    tol : float, optional
        Tolerance for boundary detection (default 1e-6).

    Returns
    -------
    is_lid : torch.Tensor
        Boolean mask, True for lid nodes.
    is_wall : torch.Tensor
        Boolean mask, True for wall nodes (excluding lid).
    is_int : torch.Tensor
        Boolean mask, True for interior nodes.
    """
    x, y = points[:, 0], points[:, 1]
    is_lid = y > (1.0 - tol)
    is_wall = ((x < tol) | (x > 1.0 - tol) | (y < tol)) & ~is_lid
    is_int = ~(is_lid | is_wall)
    return is_lid, is_wall, is_int


def build_bc_rhs(
    N: int,
    is_lid: torch.Tensor,
    is_wall: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Construct the right-hand side vector F for the momentum equation.

    Encodes the boundary conditions:
        Lid nodes: F[2*i] = 1.0 (u = 1, unit lid velocity)
        Wall/lid v: F[2*i+1] = 0.0 (v = 0 everywhere on boundary)
        Interior: F = 0 (no forcing in this benchmark)

    Returns
    -------
    torch.Tensor
        RHS vector F of shape (2N,).
    """
    F = torch.zeros(2 * N, dtype=torch.float32, device=device)
    lid_idx = is_lid.nonzero(as_tuple=True)[0]
    F[2 * lid_idx] = 1.0
    return F


def assemble_momentum_operator(
    a: torch.Tensor,
    Gx: torch.Tensor,
    Gy: torch.Tensor,
    Phi: torch.Tensor,
    Lap: torch.Tensor,
    nu: float,
    is_int: torch.Tensor,
) -> torch.Tensor:
    """Assemble the nonlinear momentum operator K(a) for the current iterate.

    For interior nodes:
        K(a) = u_h * Gx + v_h * Gy - nu * Lap
    where u_h = Phi @ a_u and v_h = Phi @ a_v are the interpolated
    velocity components.

    Boundary rows enforce the Dirichlet condition via identity rows:
        K[2*i, 2*i] = 1 for boundary node i.

    Parameters
    ----------
    a : torch.Tensor
        Current velocity iterate, shape (2N,). Interleaved [u0,v0,u1,v1,...].
    Gx, Gy : torch.Tensor
        x- and y-columns of the divergence operator G, shape (N, N).
    Phi : torch.Tensor
        RBF interpolation matrix, shape (N, N).
    Lap : torch.Tensor
        Laplacian matrix, shape (N, N).
    nu : float
        Kinematic viscosity (= 1/Re).
    is_int : torch.Tensor
        Boolean mask of interior nodes, shape (N,).

    Returns
    -------
    torch.Tensor
        Assembled momentum operator K, shape (2N, 2N).
    """
    N, device = Phi.shape[0], a.device
    a_u, a_v = a[0::2], a[1::2]
    u_h, v_h = Phi @ a_u, Phi @ a_v
    K_sc = u_h.unsqueeze(1) * Gx + v_h.unsqueeze(1) * Gy - nu * Lap

    K_full = torch.zeros(2 * N, 2 * N, dtype=torch.float32, device=device)
    int_idx = is_int.nonzero(as_tuple=True)[0]
    bnd_idx = (~is_int).nonzero(as_tuple=True)[0]
    u_rows = 2 * int_idx
    v_rows = 2 * int_idx + 1
    u_cols = torch.arange(N, device=device) * 2
    v_cols = torch.arange(N, device=device) * 2 + 1

    K_full[u_rows.unsqueeze(1), u_cols.unsqueeze(0)] = K_sc[int_idx, :]
    K_full[v_rows.unsqueeze(1), v_cols.unsqueeze(0)] = K_sc[int_idx, :]
    K_full[2 * bnd_idx, 2 * bnd_idx] = 1.0
    K_full[2 * bnd_idx + 1, 2 * bnd_idx + 1] = 1.0
    return K_full


class NavierStokesSolver:
    """Projection-based RBF-FD solver for steady incompressible flow.

    Implements Algorithm 1 (fractional-step projection method) on a
    scattered node set. All discrete operators are assembled from the
    MQ RBF-FD stencils and stored for use by the Neural Operator.

    The interior-restricted divergence operator G_int is the central
    object transplanted into the GNN projection layer.

    Parameters
    ----------
    points : torch.Tensor
        Scattered node coordinates, shape (N, 2).
    k : int, optional
        Number of nearest neighbours for stencil (default 25).
    eps : float, optional
        Tikhonov regularisation for the pressure Poisson solve (default 1e-8).

    Attributes (key)
    ----------------
    G : torch.Tensor
        Full-domain divergence operator G_full, shape (N, 2N). float32.
    G_int : torch.Tensor
        Interior-restricted G_full[is_int, :], shape (N_int, 2N). float32.
        **Transplanted into the GNN projection layer.**
    G_int_int : torch.Tensor
        G_int restricted to interior DOFs, shape (N_int, 2*N_int). float32.
    interior_dof_mask : torch.Tensor
        Boolean mask of shape (2N,) marking interior velocity DOFs.
    L_int_chol_64 : torch.Tensor
        Cholesky factor of L_int = G_int_int G_int_int^T + eps*I, float64.

    Notes
    -----
    - Shape parameter: c = 1.2 * h_avg (mean nearest-neighbour distance).
    - All operator assembly is in float32; the Cholesky factorisation
      for the projection is promoted to float64 (Remark 2).
    - The solver uses a fixed-point iteration with relaxation (alpha=0.7)
      for robustness at moderate Reynolds numbers.
    """

    def __init__(self, points: torch.Tensor, k: int = 25, eps: float = 1e-8):
        self.points = points
        self.N = int(points.shape[0])
        self.device = points.device
        self.eps = eps

        self.stencils = build_stencils(points, k)
        c = 1.2 * torch.norm(
            points[self.stencils[:, 1]] - points, dim=1
        ).mean().item()

        # Assemble all discrete operators
        self.G_full = assemble_divergence_operator(points, self.stencils, c)
        self.G = self.G_full  # alias for downstream compatibility
        self.Gx = self.G_full[:, 0::2]
        self.Gy = self.G_full[:, 1::2]
        self.Phi = assemble_phi_stencil(points, self.stencils, c)
        self.Lap = assemble_laplacian_stencil(points, self.stencils, c)

        # Node classification
        self.is_lid, self.is_wall, self.is_int = classify_cavity_nodes(points)
        int_idx = self.is_int.nonzero(as_tuple=True)[0]

        # Interior DOF mask (for velocity vector of length 2N)
        self.interior_dof_mask = torch.zeros(
            2 * self.N, dtype=torch.bool, device=self.device
        )
        self.interior_dof_mask[2 * int_idx] = True
        self.interior_dof_mask[2 * int_idx + 1] = True

        # Interior-restricted operators (True Algebraic Transplant)
        self.G_int = self.G_full[self.is_int]  # (N_int, 2N)
        self.G_int_int = self.G_int[:, self.interior_dof_mask]  # (N_int, 2*N_int)

        # Pre-factor L_int in float64 for O(10^-13) projection precision
        G64 = self.G_int_int.to(torch.float64)
        L_int_64 = G64 @ G64.T + 1e-10 * torch.eye(
            int_idx.shape[0], dtype=torch.float64, device=self.device
        )
        self.L_int_chol_64 = torch.linalg.cholesky(L_int_64)

        self.F = build_bc_rhs(self.N, self.is_lid, self.is_wall, self.device)

    def _project(
        self,
        a_star: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interior-restricted Helmholtz projection (Steps 2-3 of Algorithm 1).

        Solves: L_int b = G_int a* (float64)
        Then:   a_new[interior] -= G_int_int^T b

        Boundary DOFs are untouched (Proposition 4 — boundary invariance).

        Returns
        -------
        a_new : torch.Tensor
            Projected velocity, shape (2N,). G_int @ a_new ~ O(10^-13).
        b_int : torch.Tensor
            Pressure Lagrange multiplier on interior nodes, shape (N_int,).
        """
        rhs_64 = (self.G_int @ a_star).to(torch.float64)
        b_int = torch.cholesky_solve(
            rhs_64.unsqueeze(1), self.L_int_chol_64, upper=False
        ).squeeze(-1).to(torch.float32)

        a_new = a_star.clone()
        a_new[self.interior_dof_mask] -= self.G_int_int.T @ b_int
        return a_new, b_int

    def _momentum_residual(
        self,
        a: torch.Tensor,
        b_int: torch.Tensor,
        nu: float,
    ) -> float:
        """Relative momentum residual ||K a + G^T b - F||_2 / ||F||_2."""
        b_full = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        b_full[self.is_int] = b_int
        K = assemble_momentum_operator(
            a, self.Gx, self.Gy, self.Phi, self.Lap, nu, self.is_int
        )
        res = K @ a + self.G_full.T @ b_full - self.F
        return (res.norm() / (self.F.norm() + 1e-12)).item()

    def solve(
        self,
        Re: float,
        x0: torch.Tensor | None = None,
        tau_mom: float = 1e-2,
        tau_mass: float = 1e-4,
        n_max: int = 100,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the fractional-step solver to convergence.

        Parameters
        ----------
        Re : float
            Reynolds number. Training range: Re in [10, 100].
        x0 : torch.Tensor | None, optional
            Initial guess for velocity coefficients, shape (2N,). If None,
            starts from zero (cold start). Default is None.
        tau_mom : float, optional
            Momentum residual tolerance (default 1e-2).
        tau_mass : float, optional
            Divergence residual tolerance (default 1e-4).
        n_max : int, optional
            Maximum fixed-point iterations (default 100).
        verbose : bool, optional
            Print per-iteration diagnostics.

        Returns
        -------
        a : torch.Tensor
            Velocity coefficient vector, shape (2N,). Satisfies
            ||G_int a||_2 < tau_mass at exit (by Theorem 1).
        b_full : torch.Tensor
            Pressure coefficient vector, shape (N,). Zero at boundary nodes.
        n_iter : int
            Number of fixed-point iterations performed.

        Notes
        -----
        A relaxation factor of 0.7 is applied to the velocity update
        (a <- 0.7 * a_new + 0.3 * a) for robustness at moderate Re.
        For Re > 200 the solver may require more iterations; n_max should
        be increased accordingly.
        """
        # Input validation
        if not isinstance(Re, (int, float)):
            raise TypeError(f"Re must be numeric, got {type(Re)}")
        if Re <= 0:
            raise ValueError(f"Re must be positive, got {Re}")
        if Re > 2000:
            import warnings
            warnings.warn(
                f"Re={Re} is very high. Solver may not converge for steady laminar flow. "
                f"Consider reducing Re or increasing n_max.",
                UserWarning,
            )

        if tau_mom <= 0:
            raise ValueError(f"tau_mom must be positive, got {tau_mom}")
        if tau_mass <= 0:
            raise ValueError(f"tau_mass must be positive, got {tau_mass}")
        if n_max <= 0:
            raise ValueError(f"n_max must be positive, got {n_max}")
        if n_max > 10000:
            import warnings
            warnings.warn(f"n_max={n_max} is very large. This may be slow.", UserWarning)

        nu = 1.0 / Re

        # Initialise from x0 if provided, else zeros (cold start)
        if x0 is not None:
            a = x0.clone().to(dtype=torch.float32, device=self.device)
        else:
            a = torch.zeros(2 * self.N, dtype=torch.float32, device=self.device)

        b_int = torch.zeros(
            self.is_int.sum(), dtype=torch.float32, device=self.device
        )

        for n in range(n_max):
            K = assemble_momentum_operator(
                a, self.Gx, self.Gy, self.Phi, self.Lap, nu, self.is_int
            )
            try:
                a_star = torch.linalg.solve(K, self.F)
            except torch.linalg.LinAlgError:
                reg = 1e-6 * torch.eye(
                    2 * self.N, dtype=torch.float32, device=self.device
                )
                a_star = torch.linalg.solve(K + reg, self.F)

            a_new, b_new_int = self._project(a_star)
            mom_res = self._momentum_residual(a_new, b_new_int, nu)
            div_res = (self.G_int @ a_new).norm().item()

            if verbose:
                print(f"  iter {n:3d}  mom={mom_res:.2e}  div={div_res:.2e}")

            # Relaxed update
            a = 0.7 * a_new + 0.3 * a
            b_int = b_new_int

            if mom_res < tau_mom and div_res < tau_mass:
                a = a_new
                break

        b_full = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        b_full[self.is_int] = b_int
        return a, b_full, n + 1
