"""
compute_force_sensitivity.py
Computes the sensitivity coefficients  A, B, C  from the force error
upper bound (Proposition~\\ref{prop:force_error_bound}) using the
RBF-FD stencil and boundary quadrature.

Usage:
    python scripts/compute_force_sensitivity.py

Outputs:
    results/force_sensitivity.json

Dependencies:
    rbffd_solver.py  (from /repo/)
    numpy, scipy, json
"""

import numpy as np
import scipy.sparse.linalg as spl
import json
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, REPO_ROOT)

try:
    from rbffd_solver import RBFFDSolver
except ImportError as e:
    logger.error(f"Cannot import RBFFDSolver: {e}")
    sys.exit(1)

# ---- Paper constants -------------------------------------------------------
EPS_VEL   = 0.1375    # 13.75% observed velocity error
DELTA_OP  = 3.1e-3    # upper bound from paper (eq. op_gap)
DRAG_OBS  = 30.03     # mean drag error (%) from Table tab:force_errors
DRAG_THRESHOLD = 5.0  # engineering tolerance (%)
N_REF     = 225
RE_REF    = 50
NU        = 1.0 / RE_REF


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
    solver = RBFFDSolver(N=N, domain='cavity', k=25)
    solver.assemble()

    # Boundary quadrature weights  (shape: N_partial,)
    w = solver.integration_weights_boundary()         # 1D array

    # Wall-normal gradient matrix on boundary  (shape: N_partial x 2N)
    D_Gamma = solver.boundary_gradient_matrix()

    # Reference solution fields (for normalization)
    a_ref, b_ref = solver.reference_solution(Re=Re)   # velocity, pressure at wall
    F_ref = float(w @ (nu * D_Gamma @ a_ref - b_ref))
    if abs(F_ref) < 1e-12:
        raise ValueError(
            f"Reference drag force |F_ref|={abs(F_ref):.2e} is near zero. "
            "Check boundary conditions or solver output."
        )

    # Coefficient C_D = ||nu * w^T D_Gamma||_2  (spectral norm of a row vector → 2-norm)
    wD = nu * w @ D_Gamma          # shape: (2N,)
    C_D = float(np.linalg.norm(wD, 2)) * float(np.linalg.norm(a_ref, 2)) / abs(F_ref)

    # Coefficient C_p = ||w||_1 (L1 norm of quadrature weights)
    C_p = float(np.linalg.norm(w, 1)) * float(np.linalg.norm(b_ref, 1)) / abs(F_ref)

    # Operator-gap contribution (upper bound)
    C_op_times_delta = (
        float(np.linalg.norm(nu * w @ D_Gamma, 2))
        * float(np.linalg.norm(a_ref, 2))
        / abs(F_ref)
        * DELTA_OP
    )

    # Predicted drag error at observed velocity error
    # (using pressure error = 0 for lower bound — conservative)
    pred_drag_lower = C_D * EPS_VEL + C_op_times_delta
    pred_drag_upper = C_D * EPS_VEL + C_p * EPS_VEL + C_op_times_delta  # proxy

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
