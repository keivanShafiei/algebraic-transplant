"""scripts/audit_warmstart_claim.py — Warm-Start Decomposition Audit

Disentangles the algebraic (divergence-free initialization) component
from the physics-learning component of the warm-start iteration speedup.

Experiment design (Section 4.6, Table 10):
  Condition A: Cold start (zero field, non-div-free) → iter_cold
  Condition B: Div-free ZERO field (no physics, div-free) → iter_zero_df
  Condition C: Projected surrogate warm start (div-free) → iter_surrogate

Usage:
  python scripts/audit_warmstart_claim.py

Outputs:
  results/warmstart_decomposition.json
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from src.rbf_fd.solver import NavierStokesSolver
    from src.rbf_fd.stencils import build_stencils
    from src.gnn.neural_operator import NeuralOperator
    from src.projection.layer import HelmholtzProjection
    from src.data.cavity import generate_cavity_points
except ImportError as e:
    logger.error(f"Cannot import required module: {e}")
    sys.exit(1)


# ---- Configuration ----------------------------------------------------------
RE = 500.0
N_NODES = 225  # Parameter name for generate_cavity_points is 'n', not 'N'
STENCIL_K = 25
TOL_MOM = 1e-2
TOL_MASS = 1e-4
N_MAX = 1000
CHECKPOINT = REPO_ROOT / 'checkpoints' / 'best.pt'

PAPER_ITER_COLD = 500
PAPER_ITER_SURROGATE = 120
PAPER_SPEEDUP = 4.2


# ---- Helper functions -------------------------------------------------------

def build_edge_index_from_stencils(stencils: torch.Tensor) -> torch.Tensor:
    """Convert stencil matrix (N, k) to edge_index format (2, E)."""
    N, k = stencils.shape
    dst = torch.arange(N, device=stencils.device).repeat_interleave(k)
    src = stencils.flatten()
    return torch.stack([dst, src], dim=0)


def run_solver(solver, init_field, label):
    """Run solver from initial field and record iterations + time."""
    logger.info(f"Running: {label}")
    t0 = time.perf_counter()

    a, b_full, iterations, mom_history, div_history = solver.solve(
        Re=RE,
        x0=init_field,
        tau_mom=TOL_MOM,
        tau_mass=TOL_MASS,
        n_max=N_MAX,
        verbose=False,
    )

    elapsed = time.perf_counter() - t0
    final_mom = mom_history[-1] if mom_history else float('nan')
    final_div = div_history[-1] if div_history else float('nan')

    logger.info(
        f"  {label}: {iterations} iters, {elapsed:.2f}s, "
        f"mom={final_mom:.2e}, div={final_div:.2e}"
    )

    return {
        'label': label,
        'iterations': iterations,
        'time_s': elapsed,
        'final_mom_residual': final_mom,
        'final_div_residual': final_div,
    }


def project_to_div_free(G, field, interior_dof_mask):
    """Project field to divergence-free subspace."""
    proj_layer = HelmholtzProjection(
        G=G, eps=1e-8, interior_dof_mask=interior_dof_mask,
    )
    return proj_layer.project_only(field)


def make_div_free_zero(solver):
    """Construct divergence-free zero field."""
    return torch.zeros(2 * solver.N, dtype=torch.float32, device=solver.device)


def load_surrogate_model(checkpoint_path, solver):
    """Load trained neural surrogate."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = NeuralOperator(
        in_channels=2, hidden=64, layers=4, param_dim=1, eps=1e-8,
    ).to(solver.device)

    model.set_projection(
        G=solver.G_int,
        interior_dof_mask=solver.interior_dof_mask,
    )
    model.set_interior_mask(solver.is_int)

    state_dict = torch.load(checkpoint_path, map_location=solver.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    return model


def generate_surrogate_prediction(model, solver, Re):
    """Generate surrogate prediction."""
    stencils = build_stencils(solver.points, k=STENCIL_K)
    edge_index = build_edge_index_from_stencils(stencils)

    mu = torch.tensor([Re], dtype=torch.float32, device=solver.device)
    with torch.no_grad():
        a_NO, p_corr = model.predict(
            pos=solver.points,
            edge_index=edge_index,
            mu=mu,
            edge_scale=1.0,
        )
    return a_NO


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    results_dir = REPO_ROOT / 'results'
    results_dir.mkdir(exist_ok=True)

    logger.info(f"Assembling solver (n={N_NODES}, Re={RE})")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # CORRECTED: Use 'n=' not 'N=' for generate_cavity_points
    points = generate_cavity_points(n=N_NODES).to(device)
    solver = NavierStokesSolver(points=points, k=STENCIL_K, eps=1e-8)

    G_int = solver.G_int

    # Condition A: Cold start
    res_cold = run_solver(solver, None, 'Cold start (zero, non-div-free)')

    # Condition B: Div-free zero field
    df_zero = make_div_free_zero(solver)
    div_res_zero = float((G_int @ df_zero).norm().item())
    logger.info(f"  Div-free zero: ||G v||={div_res_zero:.2e}")

    res_zero_df = run_solver(solver, df_zero, 'Div-free zero field (no physics)')

    # Condition C: Surrogate warm start
    if CHECKPOINT.exists():
        model = load_surrogate_model(CHECKPOINT, solver)
        surrogate_field = generate_surrogate_prediction(model, solver, Re=RE)
        proj_field = project_to_div_free(G_int, surrogate_field, solver.interior_dof_mask)
        div_res_proj = float((G_int @ proj_field).norm().item())
        logger.info(f"  Projected surrogate div: {div_res_proj:.2e}")

        res_surrogate = run_solver(solver, proj_field, 'Projected surrogate warm start')
    else:
        logger.warning(f"Checkpoint not found: {CHECKPOINT}")
        logger.warning("Skipping surrogate condition. Train model first.")
        res_surrogate = {
            'label': 'Surrogate (SKIPPED)', 'iterations': 0,
            'time_s': 0.0, 'final_mom_residual': float('nan'),
            'final_div_residual': float('nan'),
        }

    # Analysis
    iter_cold = res_cold['iterations']
    iter_zero_df = res_zero_df['iterations']
    iter_surrogate = res_surrogate['iterations']

    if iter_surrogate > 0:
        speedup_total = iter_cold / max(iter_surrogate, 1)
        speedup_algebraic = iter_cold / max(iter_zero_df, 1)
        speedup_physics = iter_zero_df / max(iter_surrogate, 1)

        total_reduction = iter_cold - iter_surrogate
        frac_algebraic = (iter_cold - iter_zero_df) / total_reduction if total_reduction > 0 else 0.0
        frac_physics = (iter_zero_df - iter_surrogate) / total_reduction if total_reduction > 0 else 0.0

        primary_component = 'ALGEBRAIC' if frac_algebraic > 0.6 else 'PHYSICS'

        logger.info(f"\n{'='*60}")
        logger.info(f"WARM-START DECOMPOSITION (Re={RE}, n={N_NODES})")
        logger.info(f"{'='*60}")
        logger.info(f"  Cold start:      {iter_cold} iters")
        logger.info(f"  Div-free zero:   {iter_zero_df} iters")
        logger.info(f"  Surrogate:       {iter_surrogate} iters")
        logger.info(f"  Total speedup:   {speedup_total:.2f}× (paper: {PAPER_SPEEDUP}×)")
        logger.info(f"  Algebraic:       {speedup_algebraic:.2f}× ({frac_algebraic*100:.1f}%)")
        logger.info(f"  Physics:         {speedup_physics:.2f}× ({frac_physics*100:.1f}%)")
        logger.info(f"  Primary:         {primary_component}")
        logger.info(f"{'='*60}\n")
    else:
        logger.info(f"\n{'='*60}")
        logger.info(f"WARM-START DECOMPOSITION (Re={RE}, n={N_NODES})")
        logger.info(f"{'='*60}")
        logger.info(f"  Cold start:      {iter_cold} iters")
        logger.info(f"  Div-free zero:   {iter_zero_df} iters")
        logger.info(f"  Surrogate:       SKIPPED")
        logger.info(f"  Partial speedup: {iter_cold/max(iter_zero_df,1):.2f}× (algebraic)")
        logger.info(f"{'='*60}\n")
        speedup_total = 0.0
        speedup_algebraic = iter_cold / max(iter_zero_df, 1)
        speedup_physics = 0.0
        frac_algebraic = 1.0
        frac_physics = 0.0
        primary_component = 'ALGEBRAIC (partial)'

    output = {
        'Re': RE, 'N': N_NODES,
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
        'time_cold_s': round(res_cold['time_s'], 3),
        'time_zero_df_s': round(res_zero_df['time_s'], 3),
        'time_surrogate_s': round(res_surrogate.get('time_s', 0.0), 3),
        'checkpoint_exists': CHECKPOINT.exists(),
    }

    out_path = results_dir / 'warmstart_decomposition.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved to {out_path}")
