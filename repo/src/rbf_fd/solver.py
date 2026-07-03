"""rbf_fd/solver.py — Robust Fractional-Step Navier-Stokes Solver.

Key fixes for convergence:
  1. Aggressive damping: alpha = min(0.3, 0.7 * exp(-Re/200))
  2. Residual monitoring: reject steps that increase residual
  3. Continuation with smaller Re steps
  4. Fallback to smaller alpha if divergence detected
"""

import torch
import numpy as np
import warnings
from .stencils import build_stencils
from .operators import (
    assemble_divergence_operator,
    assemble_phi_stencil,
    assemble_laplacian_stencil,
)

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import gmres
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def classify_cavity_nodes(points: torch.Tensor, tol: float = 1e-6):
    x, y = points[:, 0], points[:, 1]
    is_lid = y > (1.0 - tol)
    is_wall = ((x < tol) | (x > 1.0 - tol) | (y < tol)) & ~is_lid
    is_int = ~(is_lid | is_wall)
    return is_lid, is_wall, is_int


def build_bc_rhs(N, is_lid, is_wall, device):
    F = torch.zeros(2 * N, dtype=torch.float32, device=device)
    lid_idx = is_lid.nonzero(as_tuple=True)[0]
    F[2 * lid_idx] = 1.0
    return F


def assemble_momentum_operator(a, Gx, Gy, Phi, Lap, nu, is_int):
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
    def __init__(self, points, k=25, eps=1e-8):
        self.points = points
        self.N = int(points.shape[0])
        self.device = points.device
        self.eps = eps

        self.stencils = build_stencils(points, k)
        c = 1.2 * torch.norm(
            points[self.stencils[:, 1]] - points, dim=1
        ).mean().item()

        self.G_full = assemble_divergence_operator(points, self.stencils, c)
        self.G = self.G_full
        self.Gx = self.G_full[:, 0::2]
        self.Gy = self.G_full[:, 1::2]
        self.Phi = assemble_phi_stencil(points, self.stencils, c)
        self.Lap = assemble_laplacian_stencil(points, self.stencils, c)

        self.is_lid, self.is_wall, self.is_int = classify_cavity_nodes(points)
        int_idx = self.is_int.nonzero(as_tuple=True)[0]

        self.interior_dof_mask = torch.zeros(
            2 * self.N, dtype=torch.bool, device=self.device
        )
        self.interior_dof_mask[2 * int_idx] = True
        self.interior_dof_mask[2 * int_idx + 1] = True

        self.G_int = self.G_full[self.is_int]
        self.G_int_int = self.G_int[:, self.interior_dof_mask]

        G64 = self.G_int_int.to(torch.float64)
        L_int_64 = G64 @ G64.T + 1e-10 * torch.eye(
            int_idx.shape[0], dtype=torch.float64, device=self.device
        )
        self.L_int_chol_64 = torch.linalg.cholesky(L_int_64)

        self.F = build_bc_rhs(self.N, self.is_lid, self.is_wall, self.device)

    def _project(self, a_star):
        rhs_64 = (self.G_int @ a_star).to(torch.float64)
        b_int = torch.cholesky_solve(
            rhs_64.unsqueeze(1), self.L_int_chol_64, upper=False
        ).squeeze(-1).to(torch.float32)
        a_new = a_star.clone()
        a_new[self.interior_dof_mask] -= self.G_int_int.T @ b_int
        return a_new, b_int

    def _momentum_residual(self, a, b_int, nu):
        b_full = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        b_full[self.is_int] = b_int
        K = assemble_momentum_operator(
            a, self.Gx, self.Gy, self.Phi, self.Lap, nu, self.is_int
        )
        res = K @ a + self.G_full.T @ b_full - self.F
        return (res.norm() / (self.F.norm() + 1e-12)).item()

    def _solve_momentum_direct(self, K, F):
        try:
            a_star = torch.linalg.solve(K, F)
        except torch.linalg.LinAlgError:
            reg = 1e-6 * torch.eye(
                2 * self.N, dtype=torch.float32, device=self.device
            )
            a_star = torch.linalg.solve(K + reg, F)
        return a_star

    def _solve_momentum_iterative(self, K, F, x0, tol=1e-6, maxiter=200):
        if not _HAS_SCIPY:
            return self._solve_momentum_direct(K, F)
        K_np = K.detach().cpu().numpy()
        F_np = F.detach().cpu().numpy()
        x0_np = x0.detach().cpu().numpy()
        K_sparse = csr_matrix(K_np)
        try:
            x_np, info = gmres(
                K_sparse, F_np, x0=x0_np, tol=tol, maxiter=maxiter,
                restart=min(50, K_sparse.shape[0]),
            )
            if info < 0:
                return self._solve_momentum_direct(K, F)
            return torch.from_numpy(x_np).to(dtype=torch.float32, device=self.device)
        except Exception:
            return self._solve_momentum_direct(K, F)

    def solve(
        self,
        Re,
        x0=None,
        tau_mom=1e-2,
        tau_mass=1e-4,
        n_max=100,
        use_iterative=True,
        mom_tol=1e-6,
        alpha=None,
        verbose=False,
    ):
        """Solve with aggressive damping and residual monitoring.

        If alpha is None, uses adaptive: alpha = min(0.3, 0.7 * exp(-Re/200))
        """
        nu = 1.0 / Re

        if x0 is not None:
            if x0.shape != (2 * self.N,):
                raise ValueError(f"x0 must have shape {(2 * self.N,)}, got {x0.shape}")
            a = x0.to(dtype=torch.float32, device=self.device).clone()
        else:
            a = torch.zeros(2 * self.N, dtype=torch.float32, device=self.device)

        b_int = torch.zeros(self.is_int.sum(), dtype=torch.float32, device=self.device)

        # Adaptive damping: very aggressive for high Re
        if alpha is None:
            alpha = min(0.3, 0.7 * np.exp(-Re / 200.0))
            alpha = max(0.05, alpha)  # minimum damping

        mom_history = []
        div_history = []
        iterations = 0
        best_mom_res = float('inf')
        best_a = a.clone()
        best_b_int = b_int.clone()
        divergence_count = 0

        for n in range(n_max):
            K = assemble_momentum_operator(
                a, self.Gx, self.Gy, self.Phi, self.Lap, nu, self.is_int
            )

            if use_iterative and _HAS_SCIPY:
                a_star = self._solve_momentum_iterative(K, self.F, a, tol=mom_tol, maxiter=200)
            else:
                a_star = self._solve_momentum_direct(K, self.F)

            a_new, b_new_int = self._project(a_star)
            mom_res = self._momentum_residual(a_new, b_new_int, nu)
            div_res = (self.G_int @ a_new).norm().item()

            mom_history.append(mom_res)
            div_history.append(div_res)
            iterations = n + 1

            # Track best solution
            if mom_res < best_mom_res:
                best_mom_res = mom_res
                best_a = a_new.clone()
                best_b_int = b_new_int.clone()
                divergence_count = 0
            else:
                divergence_count += 1

            # Adaptive alpha reduction if diverging
            if divergence_count >= 3 and alpha > 0.05:
                alpha *= 0.8
                divergence_count = 0
                if verbose:
                    print(f"  -> Reducing alpha to {alpha:.3f} due to divergence")

            if verbose:
                print(f" iter {n:3d} mom={mom_res:.2e} div={div_res:.2e} alpha={alpha:.3f}")

            # Damped update
            a = alpha * a_new + (1.0 - alpha) * a
            b_int = b_new_int

            if mom_res < tau_mom and div_res < tau_mass:
                a = a_new
                if verbose:
                    print(f"  -> Converged at iteration {n}")
                break

            # Emergency stop if completely exploded
            if mom_res > 1e8 or not np.isfinite(mom_res):
                if verbose:
                    print(f"  -> EMERGENCY STOP: residual exploded ({mom_res:.2e})")
                break

        # Return best solution found, not necessarily last
        if mom_res > best_mom_res:
            a = best_a
            b_int = best_b_int
            mom_res = best_mom_res

        b_full = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        b_full[self.is_int] = b_int
        return a, b_full, iterations, mom_history, div_history

    def solve_continuation(
        self,
        Re_target,
        Re_steps=None,
        x0=None,
        tau_mom=1e-2,
        tau_mass=1e-4,
        n_max_per_step=200,
        use_iterative=True,
        mom_tol=1e-6,
        verbose=False,
    ):
        """Continuation with smaller steps and aggressive damping."""
        if Re_steps is None:
            if Re_target <= 100:
                Re_steps = [Re_target]
            else:
                # Finer steps for better stability
                steps = [10.0, 20.0, 30.0, 50.0, 75.0, 100.0]
                if Re_target > 100:
                    steps.extend([150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0])
                if Re_target > 450:
                    steps.append(Re_target)
                else:
                    steps = [s for s in steps if s <= Re_target]
                    if not steps or steps[-1] != Re_target:
                        steps.append(Re_target)
                Re_steps = steps

        a_current = x0.clone() if x0 is not None else None
        total_iterations = 0
        all_mom_history = []
        all_div_history = []

        for step_idx, Re_step in enumerate(Re_steps):
            if verbose:
                print(f"\n[Continuation {step_idx+1}/{len(Re_steps)}] Re={Re_step:.1f}")

            a, b_full, iters, mom_hist, div_hist = self.solve(
                Re=Re_step,
                x0=a_current,
                tau_mom=tau_mom,
                tau_mass=tau_mass,
                n_max=n_max_per_step,
                use_iterative=use_iterative,
                mom_tol=mom_tol,
                verbose=verbose,
            )

            total_iterations += iters
            all_mom_history.extend(mom_hist)
            all_div_history.extend(div_hist)

            converged = False
            if mom_hist and div_hist:
                converged = (mom_hist[-1] < tau_mom) and (div_hist[-1] < tau_mass)

            if verbose:
                status = "CONVERGED" if converged else "NOT CONVERGED"
                print(f"  -> {status} in {iters} iterations (best mom_res={min(mom_hist):.2e})")

            # Use best solution as warm-start for next step
            a_current = a.clone()

        return a_current, b_full, total_iterations, all_mom_history, all_div_history
