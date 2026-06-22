#!/usr/bin/env python3
"""
audit_warmstart_claim.py  (FIXED — uses actual repo API)
────────────────────────────────────────────────────────
Reproduce Table 13 (warm-start decomposition at Re=500).

API corrections vs original script
───────────────────────────────────
  RBFFDSolver       → NavierStokesSolver (points: Tensor, k, eps)
  GraphNeuralSurrogate → NeuralOperator  (n_nodes, hidden, layers, eps)
  ProjectionLayer   → HelmholtzProjection (G: Tensor, eps, interior_mask)

  solver.gradient_matrix  → solver.G_full
  solver.nodes            → solver.points
  solver.interior_node_mask → solver.is_int  (bool tensor, shape N)
  solver.n_nodes          → solver.N
  solver.solve_pcg(...)   → solver.solve(Re, x0, tau_mom, tau_mass, n_max)
                            returns (a, b_full, n_iter) — all three values
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

# ── Imports using real class names ──────────────────────────────────────────
try:
    from src.rbf_fd.solver import NavierStokesSolver
    from src.gnn.neural_operator import NeuralOperator
    from src.projection.layer import HelmholtzProjection
except ImportError as e:
    print(f"FATAL: Could not import from src/: {e}")
    print("Make sure you are running from the repo root and src/ exists.")
    sys.exit(1)

# ── Configuration ────────────────────────────────────────────────────────────
RE          = 500
N_NODES     = 225          # 15 × 15 grid
K_NEIGHBORS = 25
TOL_MASS    = 1e-4         # tau_mass in solver.solve()
TOL_MOM     = 1e-2         # tau_mom  in solver.solve()
N_MAX       = 2000

RESULTS_DIR      = REPO_ROOT / "results"
CHECKPOINT_PATH  = RESULTS_DIR / "model_best.pt"
OUTPUT_JSON      = RESULTS_DIR / "warmstart_decomposition.json"

# Paper reference values (Table 13)
PAPER_ITER_COLD      = 500
PAPER_ITER_ZERO_DF   = 145
PAPER_ITER_SURROGATE = 120
PAPER_SPEEDUP        = 4.2

# ── Node generation ──────────────────────────────────────────────────────────

def make_grid_points(n: int = N_NODES, device: str = "cpu") -> torch.Tensor:
    """
    Return a uniform Cartesian grid on [0,1]^2.
    n must be a perfect square (225 = 15×15).
    """
    side = int(round(n ** 0.5))
    assert side * side == n, f"N_NODES={n} must be a perfect square"
    xs = torch.linspace(0.0, 1.0, side)
    ys = torch.linspace(0.0, 1.0, side)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    return pts.to(torch.float32).to(device)


# ── Solver helpers ────────────────────────────────────────────────────────────

def build_solver(re: float, n_nodes: int = N_NODES,
                 device: str = "cpu") -> NavierStokesSolver:
    """
    NavierStokesSolver.__init__ assembles all operators immediately.
    No separate .assemble() call needed.
    Constructor signature: (points: Tensor, k: int = 25, eps: float = 1e-8)
    """
    points = make_grid_points(n_nodes, device=device)
    solver = NavierStokesSolver(points=points, k=K_NEIGHBORS)
    return solver


def solve_with_init(
    solver: NavierStokesSolver,
    re: float,
    x0: np.ndarray | torch.Tensor,
    tol_mass: float = TOL_MASS,
    tol_mom:  float = TOL_MOM,
    n_max:    int   = N_MAX,
) -> tuple[torch.Tensor, int, float]:
    """
    Wrapper around NavierStokesSolver.solve().

    API: solve(Re, x0=None, tau_mom, tau_mass, n_max)
    Returns: (a: Tensor, b_full: Tensor, n_iter: int)

    NOTE: x0 may be numpy (from prior audit code); convert to Tensor here.
    """
    if isinstance(x0, np.ndarray):
        x0 = torch.from_numpy(x0.astype(np.float32)).to(solver.device)
    elif isinstance(x0, torch.Tensor):
        x0 = x0.to(torch.float32).to(solver.device)

    t0 = time.perf_counter()
    a, b_full, n_iter = solver.solve(
        Re=re,
        x0=x0,
        tau_mom=tol_mom,
        tau_mass=tol_mass,
        n_max=n_max,
    )
    elapsed = time.perf_counter() - t0
    return a, n_iter, elapsed


# ── Surrogate helpers ─────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load config.yaml or fall back to paper hyper-parameters."""
    import yaml
    config_path = REPO_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {
        "model": {"hidden": 64, "layers": 4, "k": K_NEIGHBORS},
        "data":  {"n_nodes": N_NODES, "re_max": 100},
        "training": {"projection_eps": 1e-8},
    }


def _build_edge_index(stencils: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """
    Convert (N, k) stencil index array to COO edge_index (2, N*k).
    Row 0: source nodes (neighbours), Row 1: target/center nodes.
    Stencil col 0 is the node itself; include or exclude as desired —
    the GNN and solver use the same stencil, so we keep all k entries.
    """
    N, k = stencils.shape
    src = stencils.reshape(-1)                    # neighbours
    dst = torch.arange(N, device=device).repeat_interleave(k)  # centers
    return torch.stack([src, dst], dim=0)          # (2, N*k)


def load_surrogate(cfg: dict, solver: NavierStokesSolver,
                   device: str = "cpu") -> NeuralOperator | None:
    """
    Instantiate NeuralOperator and load checkpoint.

    NeuralOperator constructor:
        (n_nodes, d=2, param_dim=None, k=25, hidden=64, layers=4, eps=1e-8)
    After construction: set_points(points, stencils) + set_projection(G, mask)
    """
    if not CHECKPOINT_PATH.exists():
        return None

    mc = cfg.get("model", {})
    model = NeuralOperator(
        n_nodes=solver.N,
        d=2,
        k=mc.get("k", K_NEIGHBORS),
        hidden=mc.get("hidden", 64),
        layers=mc.get("layers", 4),
        eps=cfg.get("training", {}).get("projection_eps", 1e-8),
    )
    # Register node coordinates + stencil topology (Stencil Isomorphism)
    model.set_points(solver.points, solver.stencils)
    # Transplant the solver's interior-restricted G into the projection layer
    model.set_projection(solver.G_full, solver.is_int)

    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def predict_surrogate(
    model: NeuralOperator,
    re: float,
    solver: NavierStokesSolver,
    re_max: float = 100.0,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference for a single Re.

    NeuralOperator.forward(mu, edge_index) → (a_hat_raw, a_NO, b_pred)
    mu = Re / re_max  (normalised, as in training)
    """
    edge_index = _build_edge_index(solver.stencils, device=device)
    mu = torch.tensor(re / re_max, dtype=torch.float32, device=device)

    with torch.no_grad():
        # forward returns (a_hat_raw, a_NO_projected, b_pred)
        _, a_NO, b_pred = model(mu, edge_index, inference=True)

    return a_NO.cpu().numpy(), b_pred.cpu().numpy()


def apply_projection(
    a_raw: np.ndarray,
    solver: NavierStokesSolver,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Apply HelmholtzProjection to a numpy velocity vector.

    HelmholtzProjection(G: Tensor, eps, interior_mask: BoolTensor)
    forward(a_hat: Tensor) → a_NO: Tensor
    """
    a_t = torch.from_numpy(a_raw.astype(np.float32)).to(solver.device)
    proj = HelmholtzProjection(
        G=solver.G_full,
        eps=eps,
        interior_mask=solver.is_int,   # real attr name (bool tensor, shape N)
    ).to(solver.device)

    with torch.no_grad():
        a_proj_t = proj(a_t)
    return a_proj_t.cpu().numpy()


def verify_divergence_free(
    a_proj: np.ndarray,
    solver: NavierStokesSolver,
) -> float:
    """||G_int @ a_proj||_2  (should be O(1e-13) after Cholesky projection)."""
    a_t    = torch.from_numpy(a_proj.astype(np.float32)).to(solver.device)
    G_int  = solver.G_full[solver.is_int]      # (N_int, 2N) — real attr name
    return float((G_int @ a_t).norm().item())


# ── Decomposition analytics ───────────────────────────────────────────────────

def compute_decomposition(
    iter_cold: int, iter_zero_df: int, iter_surrogate: int
) -> dict:
    denom = iter_cold - iter_surrogate
    if denom <= 0:
        return {}
    f_alg  = (iter_cold   - iter_zero_df)  / denom
    f_phys = (iter_zero_df - iter_surrogate) / denom
    return {
        "speedup_total":     round(iter_cold / iter_surrogate, 3),
        "speedup_algebraic": round(iter_cold / iter_zero_df,   3),
        "speedup_physics":   round(iter_zero_df / iter_surrogate, 3),
        "frac_algebraic":    f_alg,
        "frac_physics":      f_phys,
        "frac_algebraic_pct": round(f_alg  * 100, 1),
        "frac_physics_pct":   round(f_phys * 100, 1),
    }


def flag_deviations(iter_cold: int, iter_surrogate: int | None) -> list[str]:
    flags = []
    if abs(iter_cold - PAPER_ITER_COLD) / PAPER_ITER_COLD > 0.05:
        flags.append(
            f"iter_cold={iter_cold} deviates >5% from paper value {PAPER_ITER_COLD}"
        )
    if iter_surrogate is not None:
        if abs(iter_surrogate - PAPER_ITER_SURROGATE) / PAPER_ITER_SURROGATE > 0.10:
            flags.append(
                f"iter_surrogate={iter_surrogate} deviates >10% from paper value "
                f"{PAPER_ITER_SURROGATE}"
            )
    return flags


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> dict:
    print("=" * 60)
    print("Warm-Start Audit: Reproducing Table 13")
    print("=" * 60)

    cfg    = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    re_max = cfg.get("data", {}).get("re_max", 100.0)
    eps    = cfg.get("training", {}).get("projection_eps", 1e-8)
    print(f"Device : {device}")

    # ── Build solver ────────────────────────────────────────────────
    print(f"\n[1/4] Assembling NavierStokesSolver at Re={RE}, N={N_NODES} ...")
    solver = build_solver(re=RE, n_nodes=N_NODES, device=device)
    print("      Solver assembled.")

    # ── Condition A: Cold start ──────────────────────────────────────
    print("\n[2/4] Condition A: Cold start (v=0 everywhere)")
    x0_cold = np.zeros(2 * N_NODES, dtype=np.float32)
    sol_cold, iter_cold, t_cold = solve_with_init(solver, RE, x0_cold)
    print(f"      Iterations : {iter_cold}")
    print(f"      Time (s)   : {t_cold:.3f}")

    # ── Condition B: Div-free zero field ────────────────────────────
    print("\n[3/4] Condition B: Div-free zero field")
    # Zero velocity is trivially div-free; lid BCs enforced internally
    x0_zero_df = np.zeros(2 * N_NODES, dtype=np.float32)
    sol_zero_df, iter_zero_df, t_zero_df = solve_with_init(solver, RE, x0_zero_df)
    print(f"      Iterations : {iter_zero_df}")
    print(f"      Time (s)   : {t_zero_df:.3f}")

    # ── Condition C: Surrogate warm-start ────────────────────────────
    print("\n[4/4] Condition C: Surrogate warm-start")
    model = load_surrogate(cfg, solver, device=device)

    if model is None:
        warnings.warn(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Skipping Condition C; reporting only Conditions A and B."
        )
        iter_surrogate = t_surrogate = eps_div_surrogate = None
        decomp = {}
        flags  = flag_deviations(iter_cold, None)
    else:
        a_raw, _b_pred = predict_surrogate(model, re=RE, solver=solver,
                                           re_max=re_max, device=device)
        a_proj        = apply_projection(a_raw, solver, eps=eps)
        eps_div       = verify_divergence_free(a_proj, solver)
        print(f"      Post-projection eps_div : {eps_div:.3e}")

        _, iter_surrogate, t_surrogate = solve_with_init(solver, RE, a_proj)
        print(f"      Iterations              : {iter_surrogate}")
        print(f"      Time (s)                : {t_surrogate:.3f}")

        decomp = compute_decomposition(iter_cold, iter_zero_df, iter_surrogate)
        flags  = flag_deviations(iter_cold, iter_surrogate)
        eps_div_surrogate = eps_div

    # ── Assemble and write result ────────────────────────────────────
    result = {
        "Re": RE, "N": N_NODES,
        "iter_cold":          iter_cold,
        "iter_div_free_zero": iter_zero_df,
        "iter_surrogate":     iter_surrogate,
        "speedup_total":         decomp.get("speedup_total"),
        "speedup_algebraic":     decomp.get("speedup_algebraic"),
        "speedup_physics":       decomp.get("speedup_physics"),
        "frac_algebraic_pct":    decomp.get("frac_algebraic_pct"),
        "frac_physics_pct":      decomp.get("frac_physics_pct"),
        "primary_component": (
            "ALGEBRAIC"
            if decomp.get("frac_algebraic", 0) > 0.5 else "PHYSICS"
        ),
        "paper_claimed_speedup":  PAPER_SPEEDUP,
        "paper_iter_cold":        PAPER_ITER_COLD,
        "paper_iter_surrogate":   PAPER_ITER_SURROGATE,
        "time_cold_s":       round(t_cold,       3),
        "time_zero_df_s":    round(t_zero_df,    3) if t_zero_df    else None,
        "time_surrogate_s":  round(t_surrogate,  3) if t_surrogate  else None,
        "eps_div_surrogate": eps_div_surrogate,
        "flagged_deviations": flags,
    }

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
