"""scripts/audit_warmstart_claim.py — Warm-Start Decomposition Audit

Disentangles the algebraic (divergence-free initialization) component
from the physics-learning component of the warm-start iteration speedup.

Experiment design (Section 4.6, Table 10):
  Condition A: Cold start (zero field, non-div-free) → iter_cold
  Condition B: Div-free ZERO field (no physics, div-free) → iter_zero_df
  Condition C: Projected surrogate warm start (div-free) → iter_surrogate

  Algebraic speedup = iter_cold / iter_zero_df
  Physics speedup   = iter_zero_df / iter_surrogate (≥ 1 if field helps)
  Total speedup       = iter_cold / iter_surrogate (reported: 4.2×)

If iter_zero_df ≈ iter_surrogate:
  → Speedup is PRIMARILY ALGEBRAIC (div-free initialization, no field info)
If iter_surrogate << iter_zero_df:
  → Field accuracy provides a genuine additional speedup component.

Usage:
  python scripts/audit_warmstart_claim.py

Outputs:
  results/warmstart_decomposition.json

Dependencies:
  - src.rbf_fd.solver.NavierStokesSolver (solver)
  - src.gnn.neural_operator.NeuralOperator (surrogate)
  - src.projection.layer.HelmholtzProjection (projection)

Fixed from original:
  - RBFFDSolver → NavierStokesSolver
  - HelmholtzProjectionLayer → HelmholtzProjection
  - solve_pcg(x0=...) → solve(Re, x0=...)
  - gradient_matrix → G_int
  - generate_lid_cavity_nodes → generate_cavity_points
  - build_graph (non-existent) → manual edge_index construction from stencils
  - numpy arrays → torch tensors throughout
  - Added x0 support to solver.solve() for warm-start
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

import torch
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add repo root to path
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
    logger.error("Make sure you are running from the repo root or scripts/ directory.")
    sys.exit(1)


# ---- Experiment configuration ----------------------------------------------
RE = 500.0          # Reynolds number for extrapolation test
N = 225             # Number of nodes (must match training resolution)
STENCIL_K = 25      # Stencil size (must match training)
TOL_MOM = 1e-2      # Momentum residual tolerance (matches solver default)
TOL_MASS = 1e-4     # Divergence residual tolerance (matches solver default)
N_MAX = 1000        # Safety cap on iterations
CHECKPOINT = REPO_ROOT / 'checkpoints' / 'best.pt'

# Paper reference values (Table 10, Section 4.6)
PAPER_ITER_COLD = 500
PAPER_ITER_SURROGATE = 120
PAPER_SPEEDUP = 4.2


def build_edge_index_from_stencils(stencils: torch.Tensor) -> torch.Tensor:
    """Convert stencil matrix (N, k) to edge_index format (2, E).

    Parameters
    ----------
    stencils : torch.Tensor
        Stencil indices, shape (N, k) where stencils[i, j] is the j-th
        nearest neighbor of node i.

    Returns
    -------
    torch.Tensor
        Edge connectivity, shape (2, E) where edge_index[0] = dst (target),
        edge_index[1] = src (source).
    """
    N, k = stencils.shape
    # Each node i connects to its k neighbors
    dst = torch.arange(N, device=stencils.device).repeat_interleave(k)
    src = stencils.flatten()
    return torch.stack([dst, src], dim=0)  # (2, N*k)


def run_solver(
    solver: NavierStokesSolver,
    init_field: torch.Tensor | None,
    label: str,
) -> dict:
    """Run solver from a given initial field and record iterations + time.

    Parameters
    ----------
    solver : NavierStokesSolver
        The RBF-FD solver instance.
    init_field : torch.Tensor or None
        Initial velocity guess, shape (2N,). If None, cold start (zero).
    label : str
        Descriptive label for this condition.

    Returns
    -------
    dict
        Results with keys: label, iterations, time_s, final_mom_res, final_div_res
    """
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
        f"  {label}: {iterations} iterations, {elapsed:.2f}s, "
        f"mom_res={final_mom:.2e}, div_res={final_div:.2e}"
    )

    return {
        'label': label,
        'iterations': iterations,
        'time_s': elapsed,
        'final_mom_residual': final_mom,
        'final_div_residual': final_div,
        'mom_history': mom_history,
        'div_history': div_history,
    }


def project_to_div_free(
    G: torch.Tensor,
    field: torch.Tensor,
    interior_dof_mask: torch.Tensor,
) -> torch.Tensor:
    """Project field to divergence-free subspace using Gram projection.

    Uses the interior-restricted projection (Proposition 4) to preserve
    boundary conditions.

    Parameters
    ----------
    G : torch.Tensor
        Interior-restricted divergence operator, shape (N_int, 2N).
    field : torch.Tensor
        Raw velocity field, shape (2N,).
    interior_dof_mask : torch.Tensor
        Boolean mask of shape (2N,) for interior DOFs.

    Returns
    -------
    torch.Tensor
        Projected divergence-free velocity, shape (2N,).
    """
    proj_layer = HelmholtzProjection(
        G=G,
        eps=1e-8,
        interior_dof_mask=interior_dof_mask,
    )
    return proj_layer.project_only(field)


def make_div_free_zero(
    solver: NavierStokesSolver,
) -> torch.Tensor:
    """Construct a divergence-free zero field.

    v = 0 for all interior degrees of freedom.
    This trivially satisfies G_int v = 0 at machine precision.
    It carries no physics information beyond the boundary conditions.

    Parameters
    ----------
    solver : NavierStokesSolver
        The solver instance (provides boundary DOF info).

    Returns
    -------
    torch.Tensor
        Zero field with boundary values preserved, shape (2N,).
    """
    zero = torch.zeros(2 * solver.N, dtype=torch.float32, device=solver.device)
    # Boundary DOFs are already zero in this benchmark (no-slip + lid)
    # So this is exactly the cold start but with div-free property
    return zero


def load_surrogate_model(
    checkpoint_path: Path,
    solver: NavierStokesSolver,
) -> NeuralOperator:
    """Load the trained neural surrogate model.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the model checkpoint (.pt file).
    solver : NavierStokesSolver
        The solver instance (for operator transplantation).

    Returns
    -------
    NeuralOperator
        Loaded model with projection layer attached.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Please train the model first or provide a valid checkpoint."
        )

    # Initialize model
    model = NeuralOperator(
        in_channels=2,
        hidden=64,
        layers=4,
        param_dim=1,
        eps=1e-8,
    ).to(solver.device)

    # Attach projection layer with interior restriction
    model.set_projection(
        G=solver.G_int,
        interior_dof_mask=solver.interior_dof_mask,
    )
    model.set_interior_mask(solver.is_int)

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=solver.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"Loaded surrogate model from {checkpoint_path}")
    return model


def generate_surrogate_prediction(
    model: NeuralOperator,
    solver: NavierStokesSolver,
    Re: float,
) -> torch.Tensor:
    """Generate surrogate prediction for a given Reynolds number.

    Parameters
    ----------
    model : NeuralOperator
        The trained surrogate model.
    solver : NavierStokesSolver
        The solver instance (for graph construction).
    Re : float
        Reynolds number.

    Returns
    -------
    torch.Tensor
        Projected velocity prediction, shape (2N,).
    """
    # Build graph from solver nodes
    stencils = build_stencils(solver.points, k=STENCIL_K)
    edge_index = build_edge_index_from_stencils(stencils)

    # Predict
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
# MAIN EXPERIMENT
# =============================================================================

if __name__ == '__main__':
    # Create output directory
    results_dir = REPO_ROOT / 'results'
    results_dir.mkdir(exist_ok=True)

    # ---- Assemble solver ----------------------------------------------------
    logger.info(f"Assembling solver (N={N}, Re={RE})")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    points = generate_cavity_points(N=N).to(device)
    solver = NavierStokesSolver(points=points, k=STENCIL_K, eps=1e-8)

    G_int = solver.G_int  # Interior-restricted divergence operator

    # ---- Condition A: Cold start ------------------------------------------
    res_cold = run_solver(
        solver=solver,
        init_field=None,  # Cold start: zero field
        label='Cold start (zero, non-div-free)',
    )

    # ---- Condition B: Divergence-free zero field --------------------------
    df_zero = make_div_free_zero(solver)
    div_res_zero = float((G_int @ df_zero).norm().item())
    logger.info(f"  Div-free zero field: ||G v||={div_res_zero:.2e} (should be ~0)")

    res_zero_df = run_solver(
        solver=solver,
        init_field=df_zero,
        label='Div-free zero field (no physics)',
    )

    # ---- Condition C: Projected surrogate warm start ----------------------
    if CHECKPOINT.exists():
        logger.info(f"Loading surrogate checkpoint: {CHECKPOINT}")
        model = load_surrogate_model(CHECKPOINT, solver)

        # Generate surrogate prediction
        surrogate_field = generate_surrogate_prediction(model, solver, Re=RE)

        # Project to divergence-free (already projected by model, but double-check)
        proj_field = project_to_div_free(
            G=G_int,
            field=surrogate_field,
            interior_dof_mask=solver.interior_dof_mask,
        )
        div_res_proj = float((G_int @ proj_field).norm().item())
        logger.info(f"  Projected surrogate div residual: {div_res_proj:.2e}")

        res_surrogate = run_solver(
            solver=solver,
            init_field=proj_field,
            label='Projected surrogate warm start',
        )
    else:
        logger.warning(f"Checkpoint not found: {CHECKPOINT}")
        logger.warning("Skipping surrogate warm-start condition.")
        logger.warning("Train the model first to get full decomposition.")
        res_surrogate = {
            'label': 'Projected surrogate warm start (SKIPPED — no checkpoint)',
            'iterations': 0,
            'time_s': 0.0,
            'final_mom_residual': float('nan'),
            'final_div_residual': float('nan'),
        }

    # ---- Decomposition analysis -------------------------------------------
    iter_cold = res_cold['iterations']
    iter_zero_df = res_zero_df['iterations']
    iter_surrogate = res_surrogate['iterations']

    if iter_surrogate > 0:
        speedup_total = iter_cold / max(iter_surrogate, 1)
        speedup_algebraic = iter_cold / max(iter_zero_df, 1)
        speedup_physics = iter_zero_df / max(iter_surrogate, 1)

        # Fraction of total speedup attributable to each component
        total_reduction = iter_cold - iter_surrogate
        if total_reduction > 0:
            frac_algebraic = (iter_cold - iter_zero_df) / total_reduction
            frac_physics = (iter_zero_df - iter_surrogate) / total_reduction
        else:
            frac_algebraic = 0.0
            frac_physics = 0.0

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
        logger.info(f"WARM-START DECOMPOSITION RESULTS (Re={RE}, N={N})")
        logger.info(f"{'='*60}")
        logger.info(f"  Cold start iterations:          {iter_cold}")
        logger.info(f"  Div-free zero-field iterations: {iter_zero_df}")
        logger.info(f"  Surrogate warm-start iterations: {iter_surrogate}")
        logger.info(f"  Total speedup:                  {speedup_total:.2f}× (paper: {PAPER_SPEEDUP}×)")
        logger.info(f"  Algebraic speedup:              {speedup_algebraic:.2f}× ({frac_algebraic*100:.1f}% of total)")
        logger.info(f"  Physics speedup:                {speedup_physics:.2f}× ({frac_physics*100:.1f}% of total)")
        logger.info(f"  Primary component:              {primary_component}")
        logger.info(f"{'='*60}\n")
    else:
        logger.info(f"\n{'='*60}")
        logger.info(f"WARM-START DECOMPOSITION RESULTS (Re={RE}, N={N})")
        logger.info(f"{'='*60}")
        logger.info(f"  Cold start iterations:          {iter_cold}")
        logger.info(f"  Div-free zero-field iterations: {iter_zero_df}")
        logger.info(f"  Surrogate warm-start:           SKIPPED (no checkpoint)")
        logger.info(f"  Partial speedup:                {iter_cold/max(iter_zero_df,1):.2f}× (algebraic only)")
        logger.info(f"{'='*60}\n")
        speedup_total = 0.0
        speedup_algebraic = iter_cold / max(iter_zero_df, 1)
        speedup_physics = 0.0
        frac_algebraic = 1.0
        frac_physics = 0.0
        primary_component = 'ALGEBRAIC (partial — no surrogate)'

    # ---- Save results -----------------------------------------------------
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
        'time_surrogate_s': round(res_surrogate.get('time_s', 0.0), 3),
        'checkpoint_exists': CHECKPOINT.exists(),
        'checkpoint_path': str(CHECKPOINT),
    }

    out_path = results_dir / 'warmstart_decomposition.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved decomposition results to {out_path}")
