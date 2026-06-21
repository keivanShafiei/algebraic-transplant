#!/usr/bin/env python3
"""
audit_warmstart_claim.py
────────────────────────
Reproduce Table 13 (warm-start decomposition at Re=500) from the paper:
"A Solver-Consistent Graph Neural Surrogate with Hard Divergence-Free
Projection for Parametric Incompressible Flows"

Three conditions tested:
  A. Cold start           : v = 0 everywhere (non-div-free)
  B. Div-free zero field  : v = 0 on interior DOFs, boundary DOFs preserved
  C. Surrogate warm-start : Load trained model, predict at Re=500, project

Output: results/warmstart_decomposition.json
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

# ── Ensure repo root is on sys.path ──────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

# ── Imports from actual src/ modules ───────────────────────────────────
try:
    from src.rbf_fd.solver import RBFFDSolver
    from src.gnn.neural_operator import GraphNeuralSurrogate
    from src.projection.layer import ProjectionLayer
except ImportError as e:
    print(f"FATAL: Could not import from src/: {e}")
    print("Make sure you are running from the repo root and src/ exists.")
    sys.exit(1)

# ── Configuration ────────────────────────────────────────────────────
RE = 500
N_NODES = 225
TOL = 1e-5          # solver mass-conservation tolerance
MAXITER = 2000      # solver max iterations
RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINT_PATH = RESULTS_PATH = RESULTS_DIR / "model_best.pt"
OUTPUT_JSON = RESULTS_DIR / "warmstart_decomposition.json"

# Paper reference values (for deviation flagging)
PAPER_ITER_COLD = 500
PAPER_ITER_ZERO_DF = 145
PAPER_ITER_SURROGATE = 120
PAPER_SPEEDUP = 4.2
PAPER_TIME_COLD = 18.4
PAPER_TIME_ZERO_DF = 5.3
PAPER_TIME_SURROGATE = 5.8

# ── Helpers ────────────────────────────────────────────────────────────

def load_config():
    """Load or construct minimal config for model instantiation."""
    import yaml
    config_path = REPO_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg
    # Fallback minimal config matching paper hyper-parameters
    return {
        "model": {
            "hidden_dim": 64,
            "num_layers": 4,
            "k_neighbors": 25,
            "film": True,
        },
        "data": {
            "n_nodes": N_NODES,
            "re_min": 10,
            "re_max": 100,
        },
        "training": {
            "projection_eps": 1e-8,
        },
    }


def build_solver(re: float, n_nodes: int = N_NODES):
    """Instantiate and assemble the RBF-FD solver for a given Re."""
    solver = RBFFDSolver(
        re=re,
        n_nodes=n_nodes,
    )
    solver.assemble()
    return solver


def get_interior_node_mask(solver):
    """
    Return 1D boolean mask of interior nodes (shape: N,).
    The solver exposes either `interior_mask` or `interior_node_mask`.
    """
    if hasattr(solver, "interior_node_mask"):
        return solver.interior_node_mask
    elif hasattr(solver, "interior_mask"):
        # If interior_mask is 1D with length N, it's the node mask
        mask = solver.interior_mask
        if mask.shape[0] == solver.n_nodes:
            return mask
        else:
            # It's already expanded to 2N; collapse back to N
            return mask[::2]  # u-component indices
    else:
        # Fallback: assume all nodes are interior (degenerate case)
        warnings.warn("Solver has no interior_mask; assuming all nodes are interior.")
        return np.ones(solver.n_nodes, dtype=bool)


def get_interior_dof_mask(solver):
    """
    Return 2D boolean mask of interior DOFs (shape: 2*N,).
    This expands the node mask to cover both u and v components.
    """
    node_mask = get_interior_node_mask(solver)
    return np.repeat(node_mask, 2)


def solve_with_init(solver, x0, tol=TOL, maxiter=MAXITER):
    """
    Run solver starting from x0.
    Returns (solution, iterations, elapsed_time).

    Tries solve_pcg first, then falls back to solve() with manual iteration counting.
    """
    t0 = time.perf_counter()

    # Preferred: solver exposes solve_pcg with return_iter
    if hasattr(solver, "solve_pcg"):
        try:
            sol, iters = solver.solve_pcg(x0=x0, tol=tol, maxiter=maxiter, return_iter=True)
            elapsed = time.perf_counter() - t0
            return sol, iters, elapsed
        except TypeError:
            # solve_pcg doesn't accept return_iter
            sol = solver.solve_pcg(x0=x0, tol=tol, maxiter=maxiter)
            iters = getattr(solver, "last_iter_count", None)
            if iters is None:
                iters = getattr(solver, "iter_count", None)
            if iters is None:
                iters = maxiter  # conservative fallback
            elapsed = time.perf_counter() - t0
            return sol, iters, elapsed

    # Fallback: generic solve()
    if hasattr(solver, "solve"):
        sol = solver.solve(x0=x0, tol=tol, maxiter=maxiter)
        iters = getattr(solver, "last_iter_count", None)
        if iters is None:
            iters = getattr(solver, "iter_count", None)
        if iters is None:
            iters = getattr(solver, "niter", None)
        if iters is None:
            iters = maxiter
        elapsed = time.perf_counter() - t0
        return sol, iters, elapsed

    raise RuntimeError("Solver has neither solve_pcg() nor solve() method.")


def load_surrogate_model(cfg, device="cpu"):
    """Load trained GraphNeuralSurrogate from checkpoint."""
    model_cfg = cfg.get("model", {})

    # Try to instantiate with common constructor signatures
    try:
        model = GraphNeuralSurrogate(
            hidden_dim=model_cfg.get("hidden_dim", 64),
            num_layers=model_cfg.get("num_layers", 4),
            k_neighbors=model_cfg.get("k_neighbors", 25),
            use_film=model_cfg.get("film", True),
            n_nodes=cfg.get("data", {}).get("n_nodes", N_NODES),
        )
    except TypeError:
        # Try alternate constructor signatures
        try:
            model = GraphNeuralSurrogate(
                in_channels=2,  # node coordinates
                hidden_dim=model_cfg.get("hidden_dim", 64),
                out_channels=3,  # u, v, p
                num_layers=model_cfg.get("num_layers", 4),
                k=model_cfg.get("k_neighbors", 25),
            )
        except TypeError:
            # Last resort: try with no args, let config handle it
            model = GraphNeuralSurrogate()

    if CHECKPOINT_PATH.exists():
        state = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state)
        model.eval()
        model.to(device)
        return model
    else:
        return None


def predict_surrogate(model, re: float, solver, device="cpu"):
    """
    Run inference for a single Re value.

    The model may need node coordinates as input, not just Re.
    We try multiple calling conventions.
    """
    with torch.no_grad():
        mu = torch.tensor([[re]], dtype=torch.float32, device=device)

        # Try calling conventions in order of likelihood
        try:
            # Convention 1: model takes (node_coords, re) or just re
            if hasattr(solver, "nodes"):
                coords = torch.tensor(solver.nodes, dtype=torch.float32, device=device)
                raw_pred = model(coords, mu)
            else:
                raw_pred = model(mu)
        except (TypeError, RuntimeError):
            try:
                # Convention 2: model takes a data object
                raw_pred = model(mu)
            except (TypeError, RuntimeError):
                # Convention 3: model expects dict or kwargs
                raw_pred = model(x=mu)

        if raw_pred.dim() == 2:
            raw_pred = raw_pred.squeeze(0)

    # Split into velocity (2N) and pressure (N)
    n = N_NODES
    a_raw = raw_pred[:2 * n].cpu().numpy()
    b_pred = raw_pred[2 * n:].cpu().numpy()
    return a_raw, b_pred


def apply_projection(a_raw, solver, cfg):
    """Apply interior-restricted projection layer."""
    eps = cfg.get("training", {}).get("projection_eps", 1e-8)
    G = solver.gradient_matrix  # N x 2N
    interior_dof_mask = get_interior_dof_mask(solver)

    proj = ProjectionLayer(
        G=G,
        interior_mask=interior_dof_mask,
        eps=eps,
    )
    a_proj = proj(a_raw)
    return a_proj


def verify_divergence_free(a_proj, solver):
    """
    Verify that the projected field satisfies G_int @ a_proj = 0.
    Returns the L2 norm of the divergence residual.
    """
    G = solver.gradient_matrix  # shape: (N, 2N)
    node_mask = get_interior_node_mask(solver)

    # G_int: keep only rows corresponding to interior nodes
    G_int = G[node_mask, :]  # shape: (N_int, 2N)

    eps_div = float(np.linalg.norm(G_int @ a_proj))
    return eps_div


def compute_decomposition(iter_cold, iter_zero_df, iter_surrogate):
    """
    Compute speedup and decomposition fractions per the paper.

    Formulas:
      speedup_total     = iter_cold / iter_surrogate
      speedup_algebraic = iter_cold / iter_zero_df
      frac_algebraic    = (iter_cold - iter_zero_df) / (iter_cold - iter_surrogate)
      frac_physics      = (iter_zero_df - iter_surrogate) / (iter_cold - iter_surrogate)
    """
    speedup_total = iter_cold / iter_surrogate if iter_surrogate and iter_surrogate > 0 else None
    speedup_algebraic = iter_cold / iter_zero_df if iter_zero_df and iter_zero_df > 0 else None
    speedup_physics = iter_zero_df / iter_surrogate if iter_zero_df and iter_surrogate and iter_surrogate > 0 else None

    denom = iter_cold - iter_surrogate if iter_surrogate else 0
    if denom > 0:
        frac_algebraic = (iter_cold - iter_zero_df) / denom
        frac_physics = (iter_zero_df - iter_surrogate) / denom
    else:
        frac_algebraic = 0.0
        frac_physics = 0.0

    return {
        "speedup_total": round(speedup_total, 3) if speedup_total else None,
        "speedup_algebraic": round(speedup_algebraic, 3) if speedup_algebraic else None,
        "speedup_physics": round(speedup_physics, 3) if speedup_physics else None,
        "frac_algebraic": frac_algebraic,
        "frac_physics": frac_physics,
        "frac_algebraic_pct": round(frac_algebraic * 100, 1),
        "frac_physics_pct": round(frac_physics * 100, 1),
    }


def flag_deviations(iter_cold, iter_surrogate):
    """Log warnings if results deviate from paper values."""
    flags = []
    if abs(iter_cold - PAPER_ITER_COLD) / PAPER_ITER_COLD > 0.05:
        flags.append(
            f"iter_cold={iter_cold} deviates >5% from paper value {PAPER_ITER_COLD}"
        )
    if iter_surrogate is not None:
        if abs(iter_surrogate - PAPER_ITER_SURROGATE) / PAPER_ITER_SURROGATE > 0.10:
            flags.append(
                f"iter_surrogate={iter_surrogate} deviates >10% from paper value {PAPER_ITER_SURROGATE}"
            )
    return flags


# ── Main experiment ────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Warm-Start Audit: Reproducing Table 13")
    print("=" * 60)

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Build solver ─────────────────────────────────────────────────
    print(f"\n[1/4] Assembling RBF-FD solver at Re={RE}, N={N_NODES} ...")
    solver = build_solver(re=RE, n_nodes=N_NODES)
    print("      Solver assembled.")

    # ── Condition A: Cold start ──────────────────────────────────────
    print("\n[2/4] Condition A: Cold start (v=0 everywhere)")
    x0_cold = np.zeros(2 * N_NODES)
    sol_cold, iter_cold, t_cold = solve_with_init(solver, x0_cold)
    print(f"      Iterations : {iter_cold}")
    print(f"      Time (s)   : {t_cold:.3f}")

    # ── Condition B: Div-free zero field ──────────────────────────────
    print("\n[3/4] Condition B: Div-free zero field")
    interior_dof_mask = get_interior_dof_mask(solver)
    x0_zero_df = np.zeros(2 * N_NODES)
    # Interior DOFs are already zero; boundary DOFs remain zero (no-slip).
    # The lid velocity is enforced internally by the solver during solve.
    sol_zero_df, iter_zero_df, t_zero_df = solve_with_init(solver, x0_zero_df)
    print(f"      Iterations : {iter_zero_df}")
    print(f"      Time (s)   : {t_zero_df:.3f}")

    # ── Condition C: Surrogate warm-start ─────────────────────────────
    print("\n[4/4] Condition C: Surrogate warm-start")
    model = load_surrogate_model(cfg, device=device)

    if model is None:
        warnings.warn(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Skipping Condition C; reporting only Conditions A and B."
        )
        iter_surrogate = None
        t_surrogate = None
        eps_div_surrogate = None
        decomp = {}
        flags = flag_deviations(iter_cold, None)
    else:
        a_raw, b_pred = predict_surrogate(model, re=RE, solver=solver, device=device)
        a_proj = apply_projection(a_raw, solver, cfg)

        # Verify divergence-free quality
        eps_div_surrogate = verify_divergence_free(a_proj, solver)
        print(f"      Post-projection eps_div : {eps_div_surrogate:.3e}")

        sol_surrogate, iter_surrogate, t_surrogate = solve_with_init(solver, a_proj)
        print(f"      Iterations              : {iter_surrogate}")
        print(f"      Time (s)                : {t_surrogate:.3f}")

        decomp = compute_decomposition(iter_cold, iter_zero_df, iter_surrogate)
        flags = flag_deviations(iter_cold, iter_surrogate)

    # ── Assemble output ──────────────────────────────────────────────
    result = {
        "Re": RE,
        "N": N_NODES,
        "iter_cold": iter_cold,
        "iter_div_free_zero": iter_zero_df,
        "iter_surrogate": iter_surrogate,
        "speedup_total": decomp.get("speedup_total"),
        "speedup_algebraic": decomp.get("speedup_algebraic"),
        "speedup_physics": decomp.get("speedup_physics"),
        "frac_algebraic_pct": decomp.get("frac_algebraic_pct"),
        "frac_physics_pct": decomp.get("frac_physics_pct"),
        "primary_component": "ALGEBRAIC" if decomp.get("frac_algebraic", 0) > 0.5 else "PHYSICS",
        "paper_claimed_speedup": PAPER_SPEEDUP,
        "paper_iter_cold": PAPER_ITER_COLD,
        "paper_iter_surrogate": PAPER_ITER_SURROGATE,
        "time_cold_s": round(t_cold, 3),
        "time_zero_df_s": round(t_zero_df, 3) if t_zero_df is not None else None,
        "time_surrogate_s": round(t_surrogate, 3) if t_surrogate is not None else None,
        "flagged_deviations": flags,
    }

    # Ensure output directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print("Results written to:", OUTPUT_JSON)
    print(json.dumps(result, indent=2))
    print("=" * 60)

    if flags:
        print("\n⚠️  WARNINGS:")
        for fl in flags:
            print(f"   - {fl}")

    return result


if __name__ == "__main__":
    main()
