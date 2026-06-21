"""
compute_force_sensitivity.py
Computes the sensitivity coefficients A, B, C from the force error
upper bound (Proposition~\ref{prop:force_error_bound}) using the
NavierStokesSolver API.

Usage:
    python scripts/compute_force_sensitivity.py

Outputs:
    results/force_sensitivity.json

Dependencies:
    src.rbf_fd.solver (NavierStokesSolver)
    numpy, scipy, json, torch
"""

import numpy as np
import scipy.sparse as sp
import json
import os
import sys
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, REPO_ROOT)

from src.rbf_fd.solver import NavierStokesSolver
from src.data.cavity import generate_cavity_points

# ---- Paper constants -------------------------------------------------------
EPS_VEL   = 0.1375    # 13.75% observed velocity error
DELTA_OP  = 3.1e-3    # upper bound from paper (eq. op_gap)
DRAG_OBS  = 30.03     # mean drag error (%) from Table tab:force_errors
DRAG_THRESHOLD = 5.0  # engineering tolerance (%)
N_REF     = 225
RE_REF    = 50


def build_solver(n_nodes: int = N_REF, k: int = 25, device: str = "cpu"):
    """Build NavierStokesSolver with cavity points."""
    points = generate_cavity_points(n_nodes).to(device)
    return NavierStokesSolver(points, k=k)


def compute_boundary_gradient_matrix(solver):
    """
    Construct boundary gradient operator D_Gamma from solver operators.

    For the lid-driven cavity, boundary nodes are lid + walls.
    We approximate the wall-normal gradient using the divergence operator
    components on boundary nodes.

    Returns:
        D_Gamma: np.ndarray, shape (N_boundary, 2N)
    """
    # Boundary node indices
    bnd_mask = ~(solver.is_int)
    bnd_idx = bnd_mask.nonzero(as_tuple=True)[0].cpu().numpy()

    # The divergence operator G = [Gx, Gy] gives us partial derivatives
    # Gx = d/dx of basis functions, Gy = d/dy of basis functions
    # For boundary gradient in normal direction, we use the components
    # For lid (y=1): normal = (0, 1) → use Gy rows
    # For walls: normal varies

    # Simplified: use G_full restricted to boundary nodes
    G_bnd = solver.G_full[bnd_idx].cpu().numpy()  # (N_bnd, 2N)

    return G_bnd, bnd_idx


def integration_weights_boundary(solver):
    """
    Approximate boundary quadrature weights.
    For uniform cavity grid, use equal weights scaled by arc length.
    """
    bnd_mask = ~(solver.is_int)
    bnd_idx = bnd_mask.nonzero(as_tuple=True)[0]
    n_bnd = len(bnd_idx)

    # Unit square perimeter = 4, but we exclude corners counted once
    # Approximate: each boundary segment has length ~1
    # For N=225 (15x15), boundary has ~4*15 - 4 = 56 nodes
    # Average spacing along boundary ~4/56 ≈ 0.07
    weights = torch.ones(n_bnd, dtype=torch.float32, device=solver.device)
    weights = weights * (4.0 / n_bnd)  # scale by perimeter/node_count

    return weights.cpu().numpy()


def reference_solution(solver, Re: float):
    """
    Compute reference solution by running the solver.

    Returns:
        a_ref: np.ndarray, shape (2N,)
        b_ref: np.ndarray, shape (N,)
    """
    a_ref, b_ref, _ = solver.solve(Re=Re, tau_mom=1e-2, tau_mass=1e-4, n_max=1000)
    return a_ref.cpu().numpy(), b_ref.cpu().numpy()


def compute_sensitivity_coefficients(N: int, Re: float) -> dict:
    """
    Compute force error sensitivity coefficients C_D, C_p.

    Parameters
    ----------
    N  : node count
    Re : Reynolds number

    Returns
    -------
    dict with C_D, C_p, A, B, and derived quantities
    """
    nu = 1.0 / Re
    solver = build_solver(n_nodes=N, k=25)

    # Boundary quadrature weights
    w = integration_weights_boundary(solver)

    # Boundary gradient matrix
    D_Gamma, bnd_idx = compute_boundary_gradient_matrix(solver)

    # Reference solution
    a_ref, b_ref = reference_solution(solver, Re=Re)

    # Extract boundary pressure
    b_ref_bnd = b_ref[bnd_idx]

    # Reference drag force: F_D = ∫ (ν ∂u/∂n - p n_y) ds
    # Simplified: F_ref = w · (ν * D_Gamma @ a_ref - b_ref_bnd)
    F_ref = float(w @ (nu * D_Gamma @ a_ref - b_ref_bnd))

    if abs(F_ref) < 1e-12:
        raise ValueError(
            f"Reference drag force |F_ref|={abs(F_ref):.2e} is near zero. "
            "Check boundary conditions or solver output."
        )

    # Coefficient C_D = ||ν * w^T D_Gamma||_2 * ||a_ref||_2 / |F_ref|
    wD = nu * w @ D_Gamma  # shape: (2N,)
    C_D = float(np.linalg.norm(wD, 2)) * float(np.linalg.norm(a_ref, 2)) / abs(F_ref)

    # Coefficient C_p = ||w||_1 * ||b_ref||_1 / |F_ref|
    C_p = float(np.linalg.norm(w, 1)) * float(np.linalg.norm(b_ref, 1)) / abs(F_ref)

    # Operator-gap contribution (upper bound)
    C_op_times_delta = (
        float(np.linalg.norm(nu * w @ D_Gamma, 2))
        * float(np.linalg.norm(a_ref, 2))
        / abs(F_ref)
        * DELTA_OP
    )

    # Predicted drag error at observed velocity error
    pred_drag_lower = C_D * EPS_VEL + C_op_times_delta
    pred_drag_upper = C_D * EPS_VEL + C_p * EPS_VEL + C_op_times_delta

    # Threshold: eps_vel such that bound ≤ DRAG_THRESHOLD
    if C_D > 0:
        threshold_vel = (DRAG_THRESHOLD / 100.0 - C_op_times_delta) / C_D
    else:
        threshold_vel = float('nan')

    # Verify bound not violated
    bound_violated = (pred_drag_lower * 100.0 > DRAG_OBS)

    result = {
        'N': N,
        'Re': Re,
        'nu': nu,
        'C_D': C_D,
        'C_p': C_p,
        'delta_op': DELTA_OP,
        'C_op_times_delta_pct': round(C_op_times_delta * 100.0, 4),
        'predicted_force_error_lower_pct': round(pred_drag_lower * 100.0, 2),
        'predicted_force_error_upper_pct': round(pred_drag_upper * 100.0, 2),
        'actual_force_error_mean_pct': DRAG_OBS,
        'predicted_threshold_vel_for_5pct_drag': round(threshold_vel * 100.0, 2),
        'bound_violated': bound_violated,
        'reference_drag_F_ref': F_ref,
    }
    return result


if __name__ == '__main__':
    os.makedirs(os.path.join(REPO_ROOT, 'results'), exist_ok=True)

    logger.info(f"Computing sensitivity coefficients (N={N_REF}, Re={RE_REF})")
    result = compute_sensitivity_coefficients(N=N_REF, Re=RE_REF)

    # Sanity checks
    if result['bound_violated']:
        logger.error(
            "CRITICAL: Force error bound VIOLATED — "
            f"predicted {result['predicted_force_error_lower_pct']:.2f}% "
            f"> observed {DRAG_OBS:.2f}%. "
            "Return to Mathematician for correction."
        )
    else:
        logger.info(
            f"  C_D = {result['C_D']:.4f}"
            f"  C_p = {result['C_p']:.4f}"
            f"  C*delta_op = {result['C_op_times_delta_pct']:.4f}%"
        )
        logger.info(
            f"  Predicted drag at eps_vel=13.75%: "
            f"[{result['predicted_force_error_lower_pct']:.2f}%, "
            f"{result['predicted_force_error_upper_pct']:.2f}%]"
            f"  Observed: {DRAG_OBS:.2f}%  → bound is {'TIGHT' if result['predicted_force_error_upper_pct']>DRAG_OBS*0.8 else 'CONSERVATIVE'}"
        )
        logger.info(
            f"  Threshold velocity error for 5% drag: "
            f"{result['predicted_threshold_vel_for_5pct_drag']:.2f}%"
        )

    out_path = os.path.join(REPO_ROOT, 'results', 'force_sensitivity.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved to {out_path}")
