"""scripts/debug_warmstart.py — Diagnostic script for warm-start convergence.

This script diagnoses why the div-free zero field doesn't reduce iterations
as expected (paper claims 3000→500, but user observes 3000→3000).

Tests:
1. Verify div-free zero field is actually div-free
2. Check solver convergence with different initial guesses
3. Compare momentum residuals for different conditions
4. Test with lower Re to verify solver works in training range
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from src.rbf_fd.solver import NavierStokesSolver
from src.data.cavity import generate_cavity_points


def test_solver_convergence(solver, init_field, label, Re, n_max=3000):
    """Test solver convergence and return detailed diagnostics."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {label} (Re={Re})")
    logger.info(f"{'='*60}")

    a, b, iters, mom_hist, div_hist = solver.solve(
        Re=Re, x0=init_field, n_max=n_max, verbose=False,
    )

    logger.info(f"  Iterations: {iters}")
    logger.info(f"  Final mom_res: {mom_hist[-1]:.2e}")
    logger.info(f"  Final div_res: {div_hist[-1]:.2e}")
    logger.info(f"  Converged: {mom_hist[-1] < 1e-2 and div_hist[-1] < 1e-4}")

    # Show first few and last few iterations
    logger.info(f"  Mom residual history (first 5): {[f'{x:.2e}' for x in mom_hist[:5]]}")
    logger.info(f"  Mom residual history (last 5):  {[f'{x:.2e}' for x in mom_hist[-5:]]}")
    logger.info(f"  Div residual history (first 5): {[f'{x:.2e}' for x in div_hist[:5]]}")
    logger.info(f"  Div residual history (last 5):  {[f'{x:.2e}' for x in div_hist[-5:]]}")

    return iters, mom_hist, div_hist


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Test 1: Re=100 (within training range)
    logger.info(f"\n{'#'*60}")
    logger.info("# TEST 1: Re=100 (within training range)")
    logger.info(f"{'#'*60}")

    points = generate_cavity_points(n=225).to(device)
    solver_100 = NavierStokesSolver(points=points, k=25, eps=1e-8)

    # Cold start
    test_solver_convergence(solver_100, None, "Cold start", Re=100, n_max=100)

    # Div-free zero
    df_zero = torch.zeros(2 * 225, device=device)
    test_solver_convergence(solver_100, df_zero, "Div-free zero", Re=100, n_max=100)

    # Test 2: Re=500 (extrapolation)
    logger.info(f"\n{'#'*60}")
    logger.info("# TEST 2: Re=500 (extrapolation)")
    logger.info(f"{'#'*60}")

    solver_500 = NavierStokesSolver(points=points, k=25, eps=1e-8)

    # Cold start
    test_solver_convergence(solver_500, None, "Cold start", Re=500, n_max=3000)

    # Div-free zero
    test_solver_convergence(solver_500, df_zero, "Div-free zero", Re=500, n_max=3000)

    # Test 3: Verify div-free property
    logger.info(f"\n{'#'*60}")
    logger.info("# TEST 3: Verify div-free property")
    logger.info(f"{'#'*60}")

    G_int = solver_500.G_int
    div_cold = (G_int @ torch.zeros(2*225, device=device)).norm().item()
    logger.info(f"  Cold start div residual: {div_cold:.2e}")
    logger.info(f"  Expected: ~0 (zero field is trivially div-free)")

    # Test 4: Check with small non-zero field
    logger.info(f"\n{'#'*60}")
    logger.info("# TEST 4: Small non-zero initial field")
    logger.info(f"{'#'*60}")

    small_field = torch.randn(2*225, device=device) * 0.01
    test_solver_convergence(solver_500, small_field, "Small random field", Re=500, n_max=100)
