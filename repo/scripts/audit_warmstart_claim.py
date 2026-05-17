"""
audit_warmstart_claim.py
Disentangles the algebraic (divergence-free initialization) component
from the physics-learning component of the warm-start iteration speedup.

Experiment design:
  Condition A:  Cold start (zero field, non-div-free)       → iter_cold
  Condition B:  Div-free ZERO field (no physics, div-free)  → iter_zero_df
  Condition C:  Projected surrogate warm start (div-free)   → iter_surrogate

  Algebraic speedup   = iter_cold / iter_zero_df
  Physics  speedup    = iter_zero_df / iter_surrogate  (≥ 1 if field helps)
  Total speedup       = iter_cold / iter_surrogate  (reported: 4.2×)

If iter_zero_df ≈ iter_surrogate:
    → Speedup is PRIMARILY ALGEBRAIC (div-free initialization, no field info)
If iter_surrogate << iter_zero_df:
    → Field accuracy provides a genuine additional speedup component.

Usage:
    python scripts/audit_warmstart_claim.py

Outputs:
    results/warmstart_decomposition.json
"""

import numpy as np
import json
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, REPO_ROOT)

try:
    from rbffd_solver import RBFFDSolver
    from model import GraphNeuralSurrogate
    from projection import HelmholtzProjectionLayer
except ImportError as e:
    logger.error(f"Cannot import required module: {e}")
    sys.exit(1)

# ---- Experiment configuration ----------------------------------------------
RE          = 500
N           = 225
STENCIL_K   = 25
CHECKPOINT  = os.path.join(REPO_ROOT, 'checkpoints', 'best.pt')
TOL_PCG     = 1e-4          # PCG stopping tolerance (matches paper)
MAX_ITER    = 1000          # safety cap
PAPER_ITER_COLD       = 500   # from Table tab:re500_preconditioner
PAPER_ITER_SURROGATE  = 120   # from Table tab:re500_preconditioner
PAPER_SPEEDUP         = 4.2   # from paper


def run_solver(solver, init_field, label: str) -> dict:
    """Run PCG solver from a given initial field and record iterations + time."""
    logger.info(f"  Running: {label}")
    t0 = time.perf_counter()
    result = solver.solve_pcg(
        x0=init_field,
        tol=TOL_PCG,
        maxiter=MAX_ITER,
        return_iter=True
    )
    elapsed = time.perf_counter() - t0
    iters   = result['iterations']
    div_res = result.get('final_div_residual', float('nan'))
    logger.info(
        f"    {label}: {iters} iterations, {elapsed:.2f}s, "
        f"div_residual={div_res:.2e}"
    )
    return {'label': label, 'iterations': iters, 'time_s': elapsed,
            'final_div_residual': div_res}


def project_to_div_free(G, field, solver):
    """Project field to divergence-free subspace using Gram projection."""
    proj_layer = HelmholtzProjectionLayer(G=G, interior_only=True)
    return proj_layer(field)


def make_div_free_zero(solver) -> np.ndarray:
    """
    Construct a divergence-free zero field:
    v = 0 for all interior degrees of freedom.
    This trivially satisfies G v = 0 at machine precision.
    It carries no physics information beyond the boundary conditions.
    """
    N_dof = solver.n_interior_dof
    zero  = np.zeros(N_dof, dtype=np.float32)
    return zero


if __name__ == '__main__':
    import torch
    os.makedirs(os.path.join(REPO_ROOT, 'results'), exist_ok=True)

    # Assemble solver
    logger.info(f"Assembling solver (N={N}, Re={RE})")
    solver = RBFFDSolver(N=N, domain='cavity', k=STENCIL_K)
    solver.assemble()
    solver.set_reynolds(RE)

    G = solver.gradient_matrix

    # ---- Condition A: Cold start -------------------------------------------
    res_cold = run_solver(solver, init_field=None, label='Cold start (zero, non-div-free)')

    # ---- Condition B: Divergence-free zero field ----------------------------
    df_zero = make_div_free_zero(solver)
    div_res_before = float(np.linalg.norm(G @ df_zero))
    logger.info(f"  Div-free zero field: ||G v||={div_res_before:.2e} (should be ~0)")
    res_zero_df = run_solver(solver, init_field=df_zero,
                             label='Div-free zero field (no physics)')

    # ---- Condition C: Projected surrogate warm start -----------------------
    logger.info(f"  Loading surrogate checkpoint: {CHECKPOINT}")
    model = GraphNeuralSurrogate.load(CHECKPOINT)
    model.eval()
    with torch.no_grad():
        surrogate_field = model.predict(solver.graph, Re=RE)
    proj_field = project_to_div_free(G, surrogate_field, solver)
    div_res_proj = float(np.linalg.norm(G @ proj_field.numpy()))
    logger.info(f"  Projected surrogate div residual: {div_res_proj:.2e}")
    res_surrogate = run_solver(solver, init_field=proj_field.numpy(),
                               label='Projected surrogate warm start')

    # ---- Decomposition analysis -------------------------------------------
    iter_cold      = res_cold['iterations']
    iter_zero_df   = res_zero_df['iterations']
    iter_surrogate = res_surrogate['iterations']

    speedup_total      = iter_cold / max(iter_surrogate, 1)
    speedup_algebraic  = iter_cold / max(iter_zero_df,   1)
    speedup_physics    = iter_zero_df / max(iter_surrogate, 1)
    frac_algebraic     = (iter_cold - iter_zero_df) / max(iter_cold - iter_surrogate, 1)
    frac_physics       = (iter_zero_df - iter_surrogate) / max(iter_cold - iter_surrogate, 1)

    # Consistency check against paper
    iter_deviation_cold = abs(iter_cold - PAPER_ITER_COLD) / PAPER_ITER_COLD
    iter_deviation_surr = abs(iter_surrogate - PAPER_ITER_SURROGATE) / PAPER_ITER_SURROGATE
    if iter_deviation_cold > 0.05:
        logger.warning(
            f"FLAGGED: cold-start iteration count {iter_cold} deviates "
            f">{iter_deviation_cold*100:.1f}% from paper value {PAPER_ITER_COLD}."
        )
    if iter_deviation_surr > 0.10:
        logger.warning(
            f"FLAGGED: surrogate warm-start iteration count {iter_surrogate} deviates "
            f">{iter_deviation_surr*100:.1f}% from paper value {PAPER_ITER_SURROGATE}."
        )

    primary_component = 'ALGEBRAIC' if frac_algebraic > 0.6 else 'PHYSICS'
    logger.info(f"\n{'='*60}")
    logger.info(f"WARM-START DECOMPOSITION RESULTS  (Re={RE}, N={N})")
    logger.info(f"{'='*60}")
    logger.info(f"  Cold start iterations:          {iter_cold}")
    logger.info(f"  Div-free zero-field iterations: {iter_zero_df}")
    logger.info(f"  Surrogate warm-start iterations:{iter_surrogate}")
    logger.info(f"  Total speedup:     {speedup_total:.2f}×  (paper: {PAPER_SPEEDUP}×)")
    logger.info(f"  Algebraic speedup: {speedup_algebraic:.2f}× ({frac_algebraic*100:.1f}% of total)")
    logger.info(f"  Physics speedup:   {speedup_physics:.2f}× ({frac_physics*100:.1f}% of total)")
    logger.info(f"  Primary component: {primary_component}")
    logger.info(f"{'='*60}\n")

    output = {
        'Re': RE,
        'N': N,
        'iter_cold': iter_cold,
        'iter_div_free_zero': iter_zero_df,
        'iter_surrogate': iter_surrogate,
        'speedup_total': round(speedup_total, 3),
        'speedup_algebraic': round(speedup_algebraic, 3),
        'speedup_physics': round(speedup_physics, 3),
        'frac_algebraic_pct': round(frac_algebraic * 100.0, 1),
        'frac_physics_pct': round(frac_physics * 100.0, 1),
        'primary_component': primary_component,
        'paper_claimed_speedup': PAPER_SPEEDUP,
        'paper_iter_cold': PAPER_ITER_COLD,
        'paper_iter_surrogate': PAPER_ITER_SURROGATE,
        'time_cold_s': round(res_cold['time_s'], 3),
        'time_zero_df_s': round(res_zero_df['time_s'], 3),
        'time_surrogate_s': round(res_surrogate['time_s'], 3),
    }

    out_path = os.path.join(REPO_ROOT, 'results', 'warmstart_decomposition.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved decomposition results to {out_path}")
