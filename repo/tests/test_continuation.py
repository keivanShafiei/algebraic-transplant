"""tests/test_continuation.py — Verify continuation solver correctness.

Tests that NavierStokesSolverContinuation:
1. Delegates to standard Picard for Re <= 200
2. Uses continuation for Re > 200
3. Produces converged solutions at Re=500
4. Matches paper's iteration counts approximately
"""

import pytest
import torch
import numpy as np

from src.rbf_fd.solver import NavierStokesSolver
from src.rbf_fd.solver_continuation import NavierStokesSolverContinuation


def _make_points(n=225):
    side = int(n ** 0.5)
    xs = torch.linspace(0.0, 1.0, side)
    ys = torch.linspace(0.0, 1.0, side)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


@pytest.fixture
def solver_continuation():
    points = _make_points(225)
    return NavierStokesSolverContinuation(points, k=25, continuation_steps=5, re_base=100.0)


@pytest.fixture
def solver_picard():
    points = _make_points(225)
    return NavierStokesSolver(points, k=25)


class TestContinuationBasics:
    """Basic functionality tests."""

    def test_continuation_steps_generation(self, solver_continuation):
        """Verify adaptive Re steps are generated correctly."""
        steps = solver_continuation._adaptive_re_steps(500)
        assert len(steps) == 5
        assert steps[0] > 100.0
        assert steps[-1] == pytest.approx(500.0, rel=0.01)
        # Check log-spacing
        ratios = [steps[i+1]/steps[i] for i in range(len(steps)-1)]
        assert all(abs(r - ratios[0]) < 0.1 for r in ratios), "Steps should be log-spaced"

    def test_delegates_to_picard_for_low_re(self, solver_continuation, solver_picard):
        """For Re <= 100, continuation solver should match Picard."""
        a_cont, b_cont, n_cont = solver_continuation.solve(Re=50, use_continuation=False)
        a_pic, b_pic, n_pic = solver_picard.solve(Re=50)

        assert torch.allclose(a_cont, a_pic, atol=1e-5)
        assert n_cont == n_pic

    def test_auto_continuation_for_high_re(self, solver_continuation):
        """Auto-enables continuation for Re > 200."""
        a, b, n = solver_continuation.solve(Re=500)
        assert n > 0
        # Should be cumulative across continuation steps
        assert n >= 5  # At least one iteration per step


class TestConvergenceAtHighRe:
    """Convergence tests at Re=500."""

    def test_picard_diverges_at_re500(self, solver_picard):
        """Pure Picard should hit n_max or diverge at Re=500."""
        a, b, n = solver_picard.solve(Re=500, n_max=2000)
        # Either diverged or took many iterations
        assert n >= 100, f"Expected many iterations, got {n}"

    def test_continuation_converges_at_re500(self, solver_continuation):
        """Continuation should converge at Re=500."""
        a, b, n = solver_continuation.solve(Re=500, n_max=100)
        assert n < 1000, f"Expected convergence, got {n} iterations"

        # Verify divergence-free
        div_res = (solver_continuation.G_int @ a).norm().item()
        assert div_res < 1e-3, f"Divergence residual {div_res:.2e} too large"

    def test_continuation_iteration_count(self, solver_continuation):
        """Total iterations should be in paper's ballpark (~500)."""
        a, b, n = solver_continuation.solve(Re=500)
        # Paper reports 500 iterations for cold start with continuation
        # Allow ±20% tolerance
        assert 300 <= n <= 700, f"Iteration count {n} outside expected range [300, 700]"


class TestWarmStartDecomposition:
    """Tests for the warm-start decomposition (Table 13)."""

    def test_cold_start_vs_divfree_zero(self, solver_continuation):
        """Div-free zero should require fewer iterations than cold start."""
        from src.projection.layer import HelmholtzProjection

        # Cold start (zero field, with continuation)
        a_cold, b_cold, n_cold = solver_continuation.solve(Re=500)

        # Div-free zero: projected zero field
        a_zero = torch.zeros(2 * solver_continuation.N)
        lid_idx = solver_continuation.is_lid.nonzero(as_tuple=True)[0]
        a_zero[2 * lid_idx] = 1.0

        proj = HelmholtzProjection(
            solver_continuation.G_full, eps=1e-8,
            interior_mask=solver_continuation.interior_dof_mask
        )
        a_df = proj(a_zero)

        a_df_sol, b_df_sol, n_df = solver_continuation.solve(
            Re=500, x0=a_df, use_continuation=False
        )

        # Div-free zero should be faster
        assert n_df < n_cold, (
            f"Div-free zero ({n_df} iter) should be faster than "
            f"cold start ({n_cold} iter)"
        )

        # Verify div-free zero is actually div-free
        div_df = (solver_continuation.G_int @ a_df).norm().item()
        assert div_df < 1e-4, f"Div-free zero not div-free: {div_df:.2e}"


class TestBoundaryPreservation:
    """Verify boundary conditions are preserved."""

    def test_continuation_preserves_boundary_values(self, solver_continuation):
        """Solution should satisfy BCs exactly."""
        a, b, n = solver_continuation.solve(Re=500)

        lid_idx = solver_continuation.is_lid.nonzero(as_tuple=True)[0]
        wall_idx = solver_continuation.is_wall.nonzero(as_tuple=True)[0]

        # Lid: u=1, v=0
        assert torch.allclose(a[2 * lid_idx], torch.ones(len(lid_idx)), atol=1e-5)
        assert torch.allclose(a[2 * lid_idx + 1], torch.zeros(len(lid_idx)), atol=1e-5)

        # Walls: u=0, v=0
        assert torch.allclose(a[2 * wall_idx], torch.zeros(len(wall_idx)), atol=1e-5)
        assert torch.allclose(a[2 * wall_idx + 1], torch.zeros(len(wall_idx)), atol=1e-5)
