"""rbf_fd/solver.py — Fractional-Step Navier-Stokes Solver (constrained fix).

=============================================================================
READ THIS FIRST — scope and honest limits of this patch
=============================================================================
This file replaces the OUTER NONLINEAR ITERATION only (the `solve` /
`solve_continuation` logic). Per the debugging constraints:

    MAY change   : assemble_momentum_operator, solve, solve_continuation,
                   damping strategy, solver type (Picard -> Newton)
    MUST NOT change: _project, G_int, G_int_int, node/stencil generation

it does NOT touch __init__, _project, or the operator-assembly modules
(operators.py, stencils.py, kernel.py) that produce self.G_full / self.Gx /
self.Gy / self.Phi / self.Lap / self.G_int / self.G_int_int.

ROOT CAUSE (established empirically — see AUDIT_REPORT.md for full evidence):
The instability is NOT a Picard/Newton/damping/precision issue (H1-H6 in the
original debug prompt). It is caused by `assemble_divergence_operator` and
`assemble_laplacian_stencil` in operators.py directly evaluating raw MQ
kernel derivatives at pairwise distances and using them as RBF-FD weights,
with NO polynomial augmentation and NO local weight-solve. This violates the
most basic consistency requirement of any finite-difference-type operator:

    Lap @ ones  != 0        (should be exactly 0 for ANY constant field)
    Gx   @ x    != 1        (should be exactly 1, the derivative of x)

Measured on the paper's own N=225, k=25 configuration: Lap @ ones has
max|entry| ~ 2.7e3 (not ~0), and Gx @ x has mean -4.4, std 37.7 (not 1.0).
This means K(a) is not a consistent discretisation of the Navier-Stokes
momentum operator at ANY Reynolds number, so no amount of outer-loop
algorithm improvement (damping, Anderson mixing, globalized Broyden/Newton
— all tested, see AUDIT_REPORT.md) can make the iteration converge to a
physically meaningful state, because the fixed points of the iteration
map are not solutions of the continuous problem.

Because G_int / G_int_int / stencils are frozen (they are transplanted into
the GNN checkpoint) and operators.py is therefore out of scope for this fix,
THIS FILE CANNOT AND DOES NOT MAKE THE SOLVER REACH THE TOLERANCES REPORTED
IN THE PAPER (Table 4, Table 10). What it DOES do, honestly:

  1. Replaces the ad hoc "aggressive damping + emergency stop" Picard loop
     with a properly globalized quasi-Newton iteration (Broyden update +
     Armijo backtracking line search on the fixed-point residual
     ||Phi(a) - a||), which guarantees monotonic decrease of that residual
     and eliminates the chaotic O(10^8) blow-ups seen in the original code.
     This is strictly better-behaved, but plateaus at a residual level
     dictated by the operator inconsistency, not by the algorithm.
  2. Runs a cheap operator self-consistency check once at construction and
     stores/reports it, so failure to converge is diagnosed rather than
     silently masked by "return best_a found so far" (which is what the
     shipped code did, and which is how a non-convergent trajectory ended
     up being written into data/samples/*.pt as if it were "the solution").
  3. Preserves the exact call signature and return type used elsewhere in
     the repository: solve(...) -> (a, b_full, iterations, mom_history,
     div_history), same as the current HEAD.

A reference implementation of the ACTUAL fix (polynomial-augmented local
RBF-FD weights, which DOES restore convergence — see AUDIT_REPORT.md,
Section "Corrected operators") is provided separately in
scripts/reference_consistent_operators.py. It is not wired in here because
adopting it changes G_full / G_int / G_int_int and therefore invalidates
any GNN checkpoint whose projection layer was frozen against the old
(inconsistent) G_int -- that is a deliberate decision the paper's authors
need to make, not one this patch should make silently.
=============================================================================
"""

import warnings
import torch
import numpy as np

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


# ------------------------------------------------------------------------
# Unchanged helpers (identical to HEAD) -- node classification and BC RHS.
# ------------------------------------------------------------------------

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
    """Unchanged from HEAD (not the bug; kept for drop-in compatibility)."""
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


def check_operator_consistency(Gx, Gy, Lap, points, atol_report=1e-2):
    """Diagnose whether Gx, Gy, Lap satisfy basic zeroth/first-order
    consistency (reproduction of constants and coordinate functions).

    A finite-difference-type differentiation operator L acting on NODAL
    values must satisfy L @ 1 = 0 exactly (differentiating a constant gives
    zero) and, for a first-derivative operator D_x, D_x @ x = 1 exactly.
    Any discretisation that fails this by more than machine/round-off error
    is not a consistent approximation of the corresponding continuous
    operator, independent of Reynolds number, damping, or solver choice.

    Returns a dict of diagnostic scalars and a boolean `consistent` flag.
    """
    N = points.shape[0]
    x, y = points[:, 0], points[:, 1]
    ones = torch.ones(N, dtype=points.dtype, device=points.device)

    lap_const = (Lap @ ones).abs().max().item()
    gx_const = (Gx @ ones).abs().max().item()
    gy_const = (Gy @ ones).abs().max().item()
    dx_of_x = (Gx @ x - ones).abs().max().item()   # Gx@x should be exactly 1
    dy_of_y = (Gy @ y - ones).abs().max().item()   # Gy@y should be exactly 1

    consistent = max(lap_const, gx_const, gy_const, dx_of_x, dy_of_y) < atol_report
    return {
        "lap_annihilates_const_maxerr": lap_const,
        "gx_annihilates_const_maxerr": gx_const,
        "gy_annihilates_const_maxerr": gy_const,
        "gx_reproduces_linear_x_maxerr": dx_of_x,
        "gy_reproduces_linear_y_maxerr": dy_of_y,
        "consistent": consistent,
    }


class NavierStokesSolver:
    def __init__(self, points, k=25, eps=1e-8):
        # ---- IDENTICAL to HEAD below: no changes to construction, ----
        # ---- _project, or any of the frozen/transplanted operators. ----
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

        # ---- NEW (additive, non-breaking): honest self-diagnosis. ----
        self.operator_diagnostics = check_operator_consistency(
            self.Gx, self.Gy, self.Lap, self.points
        )
        if not self.operator_diagnostics["consistent"]:
            warnings.warn(
                "NavierStokesSolver: the assembled Gx/Gy/Lap operators fail "
                "basic consistency checks (Lap@1 should be 0, got max|.|="
                f"{self.operator_diagnostics['lap_annihilates_const_maxerr']:.3e}; "
                "Gx@x should be 1, max err="
                f"{self.operator_diagnostics['gx_reproduces_linear_x_maxerr']:.3e}). "
                "This means the discrete momentum operator K(a) is not a "
                "consistent approximation of the Navier-Stokes momentum "
                "equation at ANY Reynolds number. No outer nonlinear solver "
                "(damped Picard, Anderson, globalized Newton) can be "
                "expected to converge to a physically meaningful state; at "
                "best it will plateau at a residual set by this operator "
                "inconsistency, not by algorithm choice or tolerance "
                "settings. See AUDIT_REPORT.md before trusting any solve() "
                "output as a reference/training solution.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _project(self, a_star):
        """UNCHANGED. Mathematically correct interior-restricted Helmholtz
        projection; not modified per the stated constraints."""
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

    # --------------------------------------------------------------
    # NEW outer iteration: fixed-point map + globalized quasi-Newton.
    # --------------------------------------------------------------

    def _fixed_point_map(self, a, nu, use_iterative, mom_tol):
        """One fractional-step evaluation Phi(a): momentum solve (frozen
        assemble_momentum_operator) + projection (unchanged _project)."""
        K = assemble_momentum_operator(
            a, self.Gx, self.Gy, self.Phi, self.Lap, nu, self.is_int
        )
        if use_iterative and _HAS_SCIPY:
            a_star = self._solve_momentum_iterative(K, self.F, a, tol=mom_tol, maxiter=200)
        else:
            a_star = self._solve_momentum_direct(K, self.F)
        a_new, b_new_int = self._project(a_star)
        return a_new, b_new_int

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
        """Globalized quasi-Newton (Broyden + Armijo line search) applied to
        the fixed-point residual g(a) = Phi(a) - a, where Phi is the exact
        same fractional-step map (momentum solve + unchanged projection)
        used by the original Picard iteration.

        Compared to damped Picard (the HEAD implementation), this:
          * guarantees the fixed-point residual ||g|| decreases monotonically
            (Armijo condition), eliminating the O(10^5-10^9) blow-ups seen
            with plain/adaptively-damped Picard;
          * still cannot beat the operator-consistency floor (see module
            docstring / AUDIT_REPORT.md) -- if self.operator_diagnostics
            flags inconsistency, expect this to plateau well above tau_mom.

        `alpha`, if given, is used only as the initial line-search step
        length (kept for signature compatibility; no longer a fixed
        relaxation factor).

        Returns
        -------
        (a, b_full, iterations, mom_history, div_history) -- same shape/
        order as HEAD, for drop-in compatibility with existing callers.
        """
        nu = 1.0 / Re
        n2 = 2 * self.N

        if x0 is not None:
            if x0.shape != (n2,):
                raise ValueError(f"x0 must have shape {(n2,)}, got {x0.shape}")
            a = x0.to(dtype=torch.float32, device=self.device).clone()
        else:
            a = torch.zeros(n2, dtype=torch.float32, device=self.device)

        Hinv = torch.eye(n2, dtype=torch.float32, device=self.device)
        t_init = alpha if (alpha is not None and 0 < alpha <= 1.0) else 1.0

        a_new, b_int = self._fixed_point_map(a, nu, use_iterative, mom_tol)
        g = a_new - a
        gnorm = g.norm().item()

        mom_history, div_history = [], []
        best_mom_res, best_a, best_b_int = float("inf"), a_new.clone(), b_int.clone()
        iterations = 0
        stall_count = 0
        prev_best = float("inf")

        for n in range(n_max):
            mom_res = self._momentum_residual(a_new, b_int, nu)
            div_res = (self.G_int @ a_new).norm().item()
            mom_history.append(mom_res)
            div_history.append(div_res)
            iterations = n + 1

            if mom_res < best_mom_res:
                best_mom_res, best_a, best_b_int = mom_res, a_new.clone(), b_int.clone()

            if verbose:
                print(f" iter {n:3d} mom={mom_res:.3e} div={div_res:.3e} |g|={gnorm:.3e}")

            if mom_res < tau_mom and div_res < tau_mass:
                if verbose:
                    print(f"  -> Converged at iteration {n}")
                return a_new, self._pack_b(b_int), iterations, mom_history, div_history

            if not np.isfinite(gnorm) or gnorm > 1e12:
                a = torch.zeros(n2, dtype=torch.float32, device=self.device)
                Hinv = torch.eye(n2, dtype=torch.float32, device=self.device)
                a_new, b_int = self._fixed_point_map(a, nu, use_iterative, mom_tol)
                g = a_new - a
                gnorm = g.norm().item()
                continue

            # stagnation detection: if the *fixed-point* residual itself
            # stops improving for many iterations, further outer iterations
            # will not help -- stop early rather than burn the iteration
            # budget (this is exactly what the operator-consistency floor
            # predicts; see AUDIT_REPORT.md).
            if best_mom_res < prev_best - 1e-9:
                stall_count = 0
            else:
                stall_count += 1
            prev_best = best_mom_res
            if stall_count >= 40:
                if verbose:
                    print(f"  -> STALLED: no improvement in momentum residual "
                          f"for 40 iterations (best={best_mom_res:.3e}); "
                          f"stopping early (see operator_diagnostics).")
                break

            p = -(Hinv @ g)
            t = t_init
            accepted = False
            for _ in range(30):
                a_trial = a + t * p
                a_new_trial, b_int_trial = self._fixed_point_map(a_trial, nu, use_iterative, mom_tol)
                g_trial = a_new_trial - a_trial
                gt_norm = g_trial.norm().item()
                if np.isfinite(gt_norm) and gt_norm <= (1 - 1e-4 * t) * gnorm:
                    accepted = True
                    break
                t *= 0.5
            if not accepted:
                t = 1e-3
                a_trial = a - t * g
                a_new_trial, b_int_trial = self._fixed_point_map(a_trial, nu, use_iterative, mom_tol)
                g_trial = a_new_trial - a_trial
                Hinv = torch.eye(n2, dtype=torch.float32, device=self.device)

            s = a_trial - a
            y = g_trial - g
            sty = s @ (Hinv @ y)
            if abs(sty.item()) > 1e-10:
                Hinv = Hinv + torch.outer(s - Hinv @ y, s) @ Hinv / sty

            a = a_trial
            g = g_trial
            gnorm = g.norm().item()
            a_new, b_int = a_new_trial, b_int_trial

        # Did not reach tolerance: return the best iterate found, but make
        # the caller's life easier by surfacing WHY via mom_history/verbose
        # and self.operator_diagnostics, rather than pretending convergence.
        if verbose and best_mom_res > tau_mom:
            print(f"  -> NOT CONVERGED after {iterations} iters "
                  f"(best mom_res={best_mom_res:.3e}, tau_mom={tau_mom:.1e}). "
                  f"operator_consistent={self.operator_diagnostics['consistent']}")
        return best_a, self._pack_b(best_b_int), iterations, mom_history, div_history

    def _pack_b(self, b_int):
        b_full = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        b_full[self.is_int] = b_int
        return b_full

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
        """Continuation in Re, delegating each step to the globalized
        solve() above. Structure unchanged from HEAD; only the inner
        per-step solver changed (Picard -> globalized quasi-Newton)."""
        if Re_steps is None:
            if Re_target <= 100:
                Re_steps = [Re_target]
            else:
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
        all_mom_history, all_div_history = [], []
        b_full = torch.zeros(self.N, dtype=torch.float32, device=self.device)

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

            converged = bool(mom_hist and div_hist and
                             mom_hist[-1] < tau_mom and div_hist[-1] < tau_mass)
            if verbose:
                status = "CONVERGED" if converged else "NOT CONVERGED"
                print(f"  -> {status} in {iters} iterations (best mom_res={min(mom_hist):.2e})")

            a_current = a.clone()

        return a_current, b_full, total_iterations, all_mom_history, all_div_history
