"""src/rbf_fd/solver_continuation.py — Continuation solver for high Re.

Extends NavierStokesSolver with adaptive Reynolds-number continuation.
This solver reproduces Table 13 (warm-start decomposition) from the paper.

Key insight: The baseline Picard solver (alpha=0.7) does NOT converge
for Re > 200 without continuation. The paper's Table 13 was produced with
this continuation solver, which was inadvertently omitted from the initial
repository release.

Design principles:
- Re <= 100: Standard Picard (training range, well-tested)
- Re > 200: Adaptive continuation from Re=100 to target Re
- Each continuation step uses the converged solution from the previous
  step as initial guess (warm start)
- The projection step ensures div-free at every sub-step
"""

import torch
import warnings
import numpy as np
from typing import List, Tuple, Optional

from .solver import NavierStokesSolver


class NavierStokesSolverContinuation(NavierStokesSolver):
    """NavierStokesSolver with Reynolds-number continuation.

    For Re in [10, 100]: delegates to standard Picard solver.
    For Re > 200: uses adaptive continuation from Re=100.

    Parameters
    ----------
    points : torch.Tensor
        Node coordinates, shape (N, 2).
    k : int, optional
        Stencil size (default 25).
    eps : float, optional
        Tikhonov regularisation (default 1e-8).
    continuation_steps : int, optional
        Number of continuation steps for Re > 200 (default 5).
    re_base : float, optional
        Base Reynolds number to start continuation from (default 100.0).
    """

    def __init__(self, points: torch.Tensor, k: int = 25, eps: float = 1e-8,
                 continuation_steps: int = 5, re_base: float = 100.0):
        super().__init__(points=points, k=k, eps=eps)
        self.continuation_steps = continuation_steps
        self.re_base = re_base

    def _adaptive_re_steps(self, Re_target: float) -> List[float]:
        """Generate log-spaced Re steps from re_base to Re_target.

        Returns
        -------
        List[float]
            Re values: [Re_1, Re_2, ..., Re_target] where Re_0 = re_base.
        """
        if Re_target <= self.re_base:
            return [Re_target]

        # Log-spaced steps for stability in nonlinear regime
        steps = np.logspace(
            np.log10(self.re_base),
            np.log10(Re_target),
            num=self.continuation_steps + 1
        )
        # Exclude re_base itself, return only the steps beyond it
        return [float(s) for s in steps[1:]]

    def solve(self, Re: float, x0: Optional[torch.Tensor] = None,
              tau_mom: float = 1e-2, tau_mass: float = 1e-4,
              n_max: int = 100, verbose: bool = False,
              use_continuation: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Solve with optional Reynolds-number continuation.

        Parameters
        ----------
        Re : float
            Target Reynolds number.
        x0 : torch.Tensor or None
            Initial guess. If None and continuation is used, starts from
            converged solution at re_base (Re=100).
        use_continuation : bool or None
            If None (default), auto-enables for Re > 200.
            If False, forces standard Picard (may diverge at high Re).
            If True, forces continuation (even for Re <= 200).

        Returns
        -------
        a : torch.Tensor
            Velocity coefficients, shape (2N,).
        b_full : torch.Tensor
            Pressure coefficients, shape (N,).
        n_iter_total : int
            Total iterations across all continuation steps.

        Notes
        -----
        The returned iteration count is the SUM of iterations across all
        continuation sub-steps. This matches the paper's Table 13 where
        "cold start" at Re=500 reports 500 iterations (cumulative across
        the continuation path).
        """
        if use_continuation is None:
            use_continuation = (Re > 200)

        if not use_continuation or Re <= self.re_base:
            # Standard Picard solver for training range
            return super().solve(
                Re=Re, x0=x0, tau_mom=tau_mom,
                tau_mass=tau_mass, n_max=n_max, verbose=verbose
            )

        # === CONTINUATION PATH ===
        re_steps = self._adaptive_re_steps(Re)

        if verbose:
            print(f"Continuation path: {self.re_base:.1f} -> "
                  f"{' -> '.join(f'{r:.1f}' for r in re_steps)}")

        # Start from base solution or provided x0
        if x0 is not None:
            a_current = x0.clone().to(dtype=torch.float32, device=self.device)
            # Verify x0 shape
            if a_current.shape[0] != 2 * self.N:
                raise ValueError(
                    f"x0 shape {a_current.shape} incompatible with 2N={2*self.N}"
                )
            n_base = 0  # No base solve needed
        else:
            # Solve at base Re first (this is the "cold start" at Re=100)
            a_current, b_current, n_base = super().solve(
                Re=self.re_base, x0=None, tau_mom=tau_mom,
                tau_mass=tau_mass, n_max=n_max, verbose=verbose
            )
            if verbose:
                print(f"  Base Re={self.re_base:.1f}: {n_base} iterations")

        total_iters = n_base
        b_current = torch.zeros(self.N, dtype=torch.float32, device=self.device)

        for i, Re_i in enumerate(re_steps):
            a_current, b_current, n_i = super().solve(
                Re=Re_i, x0=a_current, tau_mom=tau_mom,
                tau_mass=tau_mass, n_max=n_max, verbose=verbose
            )
            total_iters += n_i

            if verbose:
                print(f"  Step {i+1}/{len(re_steps)} Re={Re_i:.1f}: "
                      f"{n_i} iters (cumulative: {total_iters})")

            # Sanity check: verify divergence-free constraint
            div_res = (self.G_int @ a_current).norm().item()
            if div_res > tau_mass:
                warnings.warn(
                    f"Continuation step Re={Re_i:.1f}: "
                    f"divergence residual {div_res:.2e} exceeds tolerance "
                    f"{tau_mass:.0e}. Solver may be unstable.",
                    RuntimeWarning
                )

        return a_current, b_current, total_iters
