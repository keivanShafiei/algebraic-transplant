#!/usr/bin/env python3
"""audit_warmstart_claim_v2.py — Corrected warm-start audit (Table 13).

CORRECTIONS from v1:
====================
1. "Cold start" in the paper refers to zero field WITH continuation 
   (adaptive Re stepping from 100 to 500), NOT pure Picard.
   The baseline Picard solver diverges at Re=500 (n_max=2000 hit).

2. "Div-free zero" is a PROJECTED zero field: v=0 on interior, BCs on 
   boundary, then projected to satisfy G_int @ v = 0 exactly.
   This eliminates mass-conservation iterations.

3. The iteration reduction from 500→145 is decomposed as:
   - 500→145: algebraic (div-free initialization)
   - 145→120: physics (learned flow structure)

4. The continuation solver (solver_continuation.py) is REQUIRED for
   reproducing Table 13. It was omitted from the initial repo release.

DEFINITIONS (aligned with paper):
=================================
Condition A — Cold start (with continuation):
    v=0 everywhere, solver uses adaptive Re continuation from Re=100.
    Total iterations = sum across continuation sub-steps.
    This is the paper's "cold start" baseline.

Condition B — Div-free zero field:
    v=0 on interior DOFs, BCs enforced on boundary, THEN projected
    via HelmholtzProjection to satisfy G_int @ v = 0.
    This field is structure-less (no vortices) but divergence-free.

Condition C — NO warm-start:
    Projected surrogate prediction (a_NO) used as initial guess.
    Combines algebraic (div-free) + physics (learned structure) benefits.

Condition D — Cold start (NO continuation, PURE PICARD):
    v=0 everywhere, standard Picard with alpha=0.7.
    EXPECTED TO DIVERGE at Re=500 (n_max=2000 hit).
    Included only to demonstrate the NEED for continuation.
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

try:
    from src.rbf_fd.solver import NavierStokesSolver
    from src.rbf_fd.solver_continuation import NavierStokesSolverContinuation
    from src.gnn.neural_operator import NeuralOperator
    from src.projection.layer import HelmholtzProjection
except ImportError as e:
    print(f"FATAL: Could not import from src/: {e}")
    sys.exit(1)

# ── Configuration ────────────────────────────────────────────────────────────
RE = 500
N_NODES = 225
K_NEIGHBORS = 25
TOL_MASS = 1e-4
TOL_MOM = 1e-2
N_MAX = 2000  # Generous for Picard; continuation uses 100 per step

RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINT_PATH = RESULTS_DIR / "model_best.pt"
OUTPUT_JSON = RESULTS_DIR / "warmstart_decomposition_v2.json"

# Paper reference values (Table 13)
PAPER_ITER_COLD = 500
PAPER_ITER_ZERO_DF = 145
PAPER_ITER_SURROGATE = 120
PAPER_SPEEDUP = 4.2

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_grid_points(n: int = N_NODES, device: str = "cpu") -> torch.Tensor:
    side = int(round(n ** 0.5))
    assert side * side == n, f"N_NODES={n} must be perfect square"
    xs = torch.linspace(0.0, 1.0, side)
    ys = torch.linspace(0.0, 1.0, side)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    return pts.to(torch.float32).to(device)

def build_solver(re: float, n_nodes: int = N_NODES,
                 device: str = "cpu", use_continuation: bool = False):
    points = make_grid_points(n_nodes, device=device)
    if use_continuation:
        solver = NavierStokesSolverContinuation(
            points=points, k=K_NEIGHBORS, continuation_steps=5, re_base=100.0
        )
    else:
        solver = NavierStokesSolver(points=points, k=K_NEIGHBORS)
    return solver

def solve_with_init(
    solver, re: float, x0, tol_mass: float = TOL_MASS,
    tol_mom: float = TOL_MOM, n_max: int = N_MAX,
    use_continuation: bool = False
) -> tuple:
    if isinstance(x0, np.ndarray):
        x0 = torch.from_numpy(x0.astype(np.float32)).to(solver.device)
    elif isinstance(x0, torch.Tensor):
        x0 = x0.to(torch.float32).to(solver.device)

    t0 = time.perf_counter()

    if isinstance(solver, NavierStokesSolverContinuation):
        a, b_full, n_iter = solver.solve(
            Re=re, x0=x0 if x0.abs().sum() > 0 else None,
            tau_mom=tol_mom, tau_mass=tol_mass, n_max=n_max,
            use_continuation=use_continuation
        )
    else:
        a, b_full, n_iter = solver.solve(
            Re=re, x0=x0, tau_mom=tol_mom,
            tau_mass=tol_mass, n_max=n_max
        )

    elapsed = time.perf_counter() - t0

    if n_iter >= n_max:
        warnings.warn(
            f"AUDIT FLAG: solver hit n_max={n_max} at Re={re}. "
            f"This is EXPECTED for pure Picard at Re=500.",
            stacklevel=2
        )

    return a, n_iter, elapsed

def build_divfree_zero(solver: NavierStokesSolver) -> np.ndarray:
    """Build a divergence-free zero field: v=0 interior, BCs boundary, projected.

    This matches the paper's "div-free zero" definition:
    - Interior DOFs: v=0 (will be corrected by projection)
    - Boundary DOFs: correct BCs (preserved by Proposition 4)
    - After projection: G_int @ v = 0 exactly
    """
    a = np.zeros(2 * solver.N, dtype=np.float32)

    # Apply BCs on boundary
    lid_idx = solver.is_lid.nonzero(as_tuple=True)[0].cpu().numpy()
    a[2 * lid_idx] = 1.0  # lid velocity u=1

    # Project to make div-free (interior-restricted, boundary preserved)
    a_t = torch.from_numpy(a).to(solver.device)
    proj = HelmholtzProjection(
        G=solver.G_full, eps=1e-8,
        interior_mask=solver.interior_dof_mask
    ).to(solver.device)

    with torch.no_grad():
        a_proj = proj(a_t)

    return a_proj.cpu().numpy()

def verify_divfree(a: np.ndarray, solver: NavierStokesSolver) -> float:
    a_t = torch.from_numpy(a.astype(np.float32)).to(solver.device)
    G_int = solver.G_full[solver.is_int]
    return float((G_int @ a_t).norm().item())

# ── Surrogate helpers (same as v1) ─────────────────────────────────────────────

def load_config() -> dict:
    import yaml
    config_path = REPO_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {
        "model": {"hidden": 64, "layers": 4, "k": K_NEIGHBORS},
        "data": {"n_nodes": N_NODES, "re_max": 100.0},
        "training": {"projection_eps": 1e-8},
    }

def _build_edge_index(stencils: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    N, k = stencils.shape
    src = stencils.reshape(-1)
    dst = torch.arange(N, device=device).repeat_interleave(k)
    return torch.stack([src, dst], dim=0)

def load_surrogate(cfg: dict, solver, device: str = "cpu"):
    if not CHECKPOINT_PATH.exists():
        return None
    mc = cfg.get("model", {})
    model = NeuralOperator(
        n_nodes=solver.N, d=2, k=mc.get("k", K_NEIGHBORS),
        hidden=mc.get("hidden", 64), layers=mc.get("layers", 4),
        eps=cfg.get("training", {}).get("projection_eps", 1e-8),
    )
    model.set_points(solver.points, solver.stencils)
    model.set_projection(solver.G_full, solver.interior_dof_mask,
                         interior_node_mask=solver.is_int)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    return model

def predict_surrogate(model, re, solver, re_max=100.0, device="cpu"):
    edge_index = _build_edge_index(solver.stencils, device=device)
    mu = torch.tensor(re / re_max, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, a_NO, b_pred = model(mu, edge_index, inference=True)
    return a_NO.cpu().numpy(), b_pred.cpu().numpy()

def apply_projection(a_raw, solver, eps=1e-8):
    a_t = torch.from_numpy(a_raw.astype(np.float32)).to(solver.device)
    proj = HelmholtzProjection(
        G=solver.G_full, eps=eps, interior_mask=solver.interior_dof_mask
    ).to(solver.device)
    with torch.no_grad():
        a_proj = proj(a_t)
    return a_proj.cpu().numpy()

# ── Decomposition analytics ───────────────────────────────────────────────────

def compute_decomposition(iter_cold, iter_zero_df, iter_surrogate):
    denom = iter_cold - iter_surrogate
    if denom <= 0:
        return {}
    f_alg = (iter_cold - iter_zero_df) / denom
    f_phys = (iter_zero_df - iter_surrogate) / denom
    return {
        "speedup_total": round(iter_cold / iter_surrogate, 3),
        "speedup_algebraic": round(iter_cold / iter_zero_df, 3),
        "speedup_physics": round(iter_zero_df / iter_surrogate, 3),
        "frac_algebraic": f_alg,
        "frac_physics": f_phys,
        "frac_algebraic_pct": round(f_alg * 100, 1),
        "frac_physics_pct": round(f_phys * 100, 1),
    }

def flag_deviations(iter_cold, iter_zero_df, iter_surrogate):
    flags = []
    if abs(iter_cold - PAPER_ITER_COLD) / PAPER_ITER_COLD > 0.10:
        flags.append(
            f"iter_cold={iter_cold} deviates >10% from paper {PAPER_ITER_COLD}"
        )
    if abs(iter_zero_df - PAPER_ITER_ZERO_DF) / PAPER_ITER_ZERO_DF > 0.10:
        flags.append(
            f"iter_zero_df={iter_zero_df} deviates >10% from paper {PAPER_ITER_ZERO_DF}"
        )
    if iter_surrogate is not None:
        if abs(iter_surrogate - PAPER_ITER_SURROGATE) / PAPER_ITER_SURROGATE > 0.15:
            flags.append(
                f"iter_surrogate={iter_surrogate} deviates >15% from paper {PAPER_ITER_SURROGATE}"
            )
    return flags

# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> dict:
    print("=" * 70)
    print("Warm-Start Audit v2: Corrected Table 13 Reproduction")
    print("=" * 70)

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    re_max = cfg.get("data", {}).get("re_max", 100.0)
    eps = cfg.get("training", {}).get("projection_eps", 1e-8)
    print(f"Device: {device}")

    # ── Build solvers ─────────────────────────────────────────────
    print(f"[1/6] Assembling solvers at Re={RE}, N={N_NODES}...")
    solver_picard = build_solver(RE, use_continuation=False)  # Pure Picard
    solver_cont = build_solver(RE, use_continuation=True)      # With continuation
    print("  Picard solver assembled.")
    print("  Continuation solver assembled (5 steps from Re=100).")

    # ── Condition D: Pure Picard (EXPECTED TO FAIL) ─────────────
    print("[2/6] Condition D: Cold start — PURE PICARD (NO continuation)")
    print("  EXPECTED: Divergence or n_max=2000 hit (Re=500 out of range)")
    x0_zero = np.zeros(2 * N_NODES, dtype=np.float32)
    sol_d, iter_d, t_d = solve_with_init(
        solver_picard, RE, x0_zero, n_max=N_MAX, use_continuation=False
    )
    print(f"  Iterations: {iter_d} {'(DIVERGED — expected)' if iter_d >= N_MAX else '(converged — unexpected)'}")
    print(f"  Time (s): {t_d:.3f}")

    # ── Condition A: Cold start WITH continuation ────────────────
    print("[3/6] Condition A: Cold start — WITH continuation (paper baseline)")
    print("  v=0 everywhere, adaptive Re stepping from 100→500")
    sol_a, iter_a, t_a = solve_with_init(
        solver_cont, RE, x0_zero, use_continuation=True
    )
    print(f"  Iterations: {iter_a}")
    print(f"  Time (s): {t_a:.3f}")

    # ── Condition B: Div-free zero field ──────────────────────────
    print("[4/6] Condition B: Div-free zero field")
    print("  v=0 interior + BCs boundary, then projected to G_int @ v = 0")
    x0_divfree = build_divfree_zero(solver_cont)
    eps_div_df = verify_divfree(x0_divfree, solver_cont)
    print(f"  Pre-solve ε_div: {eps_div_df:.3e}")
    sol_b, iter_b, t_b = solve_with_init(
        solver_cont, RE, x0_divfree, use_continuation=False  # No need, already at Re=500
    )
    print(f"  Iterations: {iter_b}")
    print(f"  Time (s): {t_b:.3f}")

    # ── Condition C: NO warm-start ────────────────────────────────
    print("[5/6] Condition C: Neural Operator warm-start")
    model = load_surrogate(cfg, solver_cont, device=device)

    if model is None:
        warnings.warn(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Skipping Condition C; reporting A, B, D only."
        )
        iter_c = t_c = eps_div_c = None
    else:
        a_raw, b_pred = predict_surrogate(model, RE, solver_cont, re_max, device)
        a_proj = apply_projection(a_raw, solver_cont, eps)
        eps_div_c = verify_divfree(a_proj, solver_cont)
        print(f"  Post-projection ε_div: {eps_div_c:.3e}")

        sol_c, iter_c, t_c = solve_with_init(
            solver_cont, RE, a_proj, use_continuation=False
        )
        print(f"  Iterations: {iter_c}")
        print(f"  Time (s): {t_c:.3f}")

    # ── Decomposition ───────────────────────────────────────────
    print("[6/6] Decomposition analysis")
    if iter_c is not None:
        decomp = compute_decomposition(iter_a, iter_b, iter_c)
        print(f"  Total speedup (cold/NO): {decomp.get('speedup_total', 'N/A')}")
        print(f"  Algebraic fraction: {decomp.get('frac_algebraic_pct', 'N/A')}%")
        print(f"  Physics fraction: {decomp.get('frac_physics_pct', 'N/A')}%")
    else:
        decomp = {}
        print("  (NO surrogate available — decomposition incomplete)")

    flags = flag_deviations(iter_a, iter_b, iter_c)

    # ── Results ─────────────────────────────────────────────────
    result = {
        "Re": RE, "N": N_NODES,
        "condition_D_picard_only": {
            "description": "Pure Picard, v=0, NO continuation",
            "iterations": iter_d,
            "converged": iter_d < N_MAX,
            "time_s": round(t_d, 3),
            "note": "EXPECTED TO DIVERGE at Re=500"
        },
        "condition_A_cold_start": {
            "description": "v=0 with continuation from Re=100 (paper baseline)",
            "iterations": iter_a,
            "time_s": round(t_a, 3),
            "paper_reference": PAPER_ITER_COLD
        },
        "condition_B_divfree_zero": {
            "description": "Projected zero field (div-free, BCs preserved)",
            "pre_solve_eps_div": eps_div_df,
            "iterations": iter_b,
            "time_s": round(t_b, 3),
            "paper_reference": PAPER_ITER_ZERO_DF
        },
        "condition_C_NO_warmstart": {
            "description": "Projected surrogate prediction",
            "pre_solve_eps_div": eps_div_c,
            "iterations": iter_c,
            "time_s": round(t_c, 3) if t_c else None,
            "paper_reference": PAPER_ITER_SURROGATE
        },
        "decomposition": decomp,
        "flagged_deviations": flags,
        "notes": [
            "Continuation solver (solver_continuation.py) is REQUIRED for Table 13.",
            "Pure Picard diverges at Re=500; this is a known limitation documented in solver docstring.",
            "Div-free zero eliminates mass-conservation iterations (algebraic benefit).",
            "NO warm-start adds learned flow structure (physics benefit)."
        ]
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print("" + "=" * 70)
    print("Results written to:", OUTPUT_JSON)
    print(json.dumps(result, indent=2))
    print("=" * 70)

    if flags:
        print("⚠️  WARNINGS:")
        for fl in flags:
            print(f"  - {fl}")

    return result

if __name__ == "__main__":
    main()
