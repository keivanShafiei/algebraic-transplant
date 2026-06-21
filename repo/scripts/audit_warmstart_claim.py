#!/usr/bin/env python3
"""
audit_warmstart_claim.py
────────────────────────
Reproduce Table 13 (warm-start decomposition at Re=500)
"""

import os, sys, json, time, warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.rbf_fd.solver import NavierStokesSolver, assemble_momentum_operator
from src.rbf_fd.stencils import build_stencils
from src.data.cavity import generate_cavity_points
from src.gnn.neural_operator import NeuralOperator

RE = 500
N_NODES = 225
TOL_MOM = 1e-2
TOL_MASS = 1e-4
N_MAX = 500
RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINT_PATH = RESULTS_DIR / "model_best.pt"
OUTPUT_JSON = RESULTS_DIR / "warmstart_decomposition.json"

PAPER_ITER_COLD = 500
PAPER_ITER_SURROGATE = 120
PAPER_SPEEDUP = 4.2

def load_config():
    import yaml
    with open(REPO_ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def build_solver(n_nodes=N_NODES, k=25, eps=1e-8, device="cpu"):
    points = generate_cavity_points(n_nodes).to(device)
    return NavierStokesSolver(points, k=k, eps=eps)

def solve_with_init(solver, x0, Re, tau_mom=TOL_MOM, tau_mass=TOL_MASS, n_max=N_MAX):
    """Run solver with custom initial guess."""
    t0 = time.perf_counter()

    if x0 is not None:
        x0_t = torch.from_numpy(x0).float().to(solver.device) if isinstance(x0, np.ndarray) else x0.float().to(solver.device)
        nu = 1.0 / Re
        a = x0_t.clone()
        b_int = torch.zeros(solver.is_int.sum(), dtype=torch.float32, device=solver.device)

        for n in range(n_max):
            K = assemble_momentum_operator(a, solver.Gx, solver.Gy, solver.Phi, solver.Lap, nu, solver.is_int)
            try:
                a_star = torch.linalg.solve(K, solver.F)
            except torch.linalg.LinAlgError:
                reg = 1e-6 * torch.eye(2 * solver.N, dtype=torch.float32, device=solver.device)
                a_star = torch.linalg.solve(K + reg, solver.F)

            a_new, b_new_int = solver._project(a_star)
            mom_res = solver._momentum_residual(a_new, b_new_int, nu)
            div_res = (solver.G_int @ a_new).norm().item()

            a = 0.7 * a_new + 0.3 * a
            b_int = b_new_int

            if mom_res < tau_mom and div_res < tau_mass:
                a = a_new
                break

        b_full = torch.zeros(solver.N, dtype=torch.float32, device=solver.device)
        b_full[solver.is_int] = b_int
        elapsed = time.perf_counter() - t0
        return a, b_full, n + 1, elapsed
    else:
        a, b = solver.solve(Re=Re, tau_mom=tau_mom, tau_mass=tau_mass, n_max=n_max)
        elapsed = time.perf_counter() - t0
        return a, b, n_max, elapsed

def load_surrogate_model(cfg, device="cpu"):
    model = NeuralOperator(
        n_nodes=N_NODES,
        hidden=cfg.get("hidden_dim", 64),
        layers=cfg.get("gnn_layers", 4),
        k=cfg.get("stencil_k", 25),
        eps=float(cfg.get("projection_eps", 1e-8)),
    )
    if CHECKPOINT_PATH.exists():
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        model.eval().to(device)
        return model
    return None

def predict_surrogate(model, Re, solver, cfg, device="cpu"):
    points = solver.points
    stencils = build_stencils(points, cfg.get("stencil_k", 25)).to(device)
    edge_dst = stencils.reshape(-1)
    edge_src = torch.arange(N_NODES, device=device).repeat_interleave(cfg.get("stencil_k", 25))
    edge_index = torch.stack([edge_dst, edge_src])

    model.set_points(points, stencils)
    G = solver.G_int.to(device)
    interior_mask = solver.interior_dof_mask.to(device)
    interior_node_mask = solver.is_int.to(device)
    model.set_projection(G, interior_mask=interior_mask, interior_node_mask=interior_node_mask)

    re_max = cfg.get("re_max", 100.0)
    mu = torch.tensor([Re / re_max], dtype=torch.float32, device=device)

    with torch.no_grad():
        a_NO, _ = model.predict(mu, edge_index)
    return a_NO.squeeze(0).cpu().numpy()

def verify_divergence_free(a, solver):
    a_t = torch.from_numpy(a).float() if isinstance(a, np.ndarray) else a.float()
    return (solver.G_int @ a_t).norm().item()

def compute_decomposition(iter_cold, iter_zero_df, iter_surrogate):
    speedup_total = iter_cold / iter_surrogate if iter_surrogate and iter_surrogate > 0 else None
    speedup_algebraic = iter_cold / iter_zero_df if iter_zero_df and iter_zero_df > 0 else None
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
        "frac_algebraic_pct": round(frac_algebraic * 100, 1),
        "frac_physics_pct": round(frac_physics * 100, 1),
    }

def main():
    print("=" * 60)
    print("Warm-Start Audit: Reproducing Table 13")
    print("=" * 60)

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print("[1/4] Assembling solver at Re=" + str(RE) + ", N=" + str(N_NODES) + "...")
    solver = build_solver(n_nodes=N_NODES, k=cfg.get("stencil_k", 25), eps=float(cfg.get("projection_eps", 1e-8)), device=device)
    print("      Interior nodes:", solver.is_int.sum().item(), "/", solver.N)

    print("[2/4] Condition A: Cold start")
    _, _, iter_cold, t_cold = solve_with_init(solver, np.zeros(2*N_NODES), Re=RE)
    print("      Iterations:", iter_cold, ", Time:", round(t_cold, 3), "s")

    print("[3/4] Condition B: Div-free zero field")
    _, _, iter_zero_df, t_zero_df = solve_with_init(solver, np.zeros(2*N_NODES), Re=RE)
    print("      Iterations:", iter_zero_df, ", Time:", round(t_zero_df, 3), "s")

    print("[4/4] Condition C: Surrogate warm-start")
    model = load_surrogate_model(cfg, device=device)

    if model is None:
        warnings.warn("Checkpoint not found at " + str(CHECKPOINT_PATH) + ". Skipping Condition C.")
        iter_surrogate = None
        t_surrogate = None
        decomp = {}
    else:
        a_pred = predict_surrogate(model, Re=RE, solver=solver, cfg=cfg, device=device)
        eps_div = verify_divergence_free(a_pred, solver)
        print("      Post-projection eps_div:", "{:.3e}".format(eps_div))
        _, _, iter_surrogate, t_surrogate = solve_with_init(solver, a_pred, Re=RE)
        print("      Iterations:", iter_surrogate, ", Time:", round(t_surrogate, 3), "s")
        decomp = compute_decomposition(iter_cold, iter_zero_df, iter_surrogate)

    result = {
        "Re": RE, "N": N_NODES,
        "iter_cold": iter_cold,
        "iter_div_free_zero": iter_zero_df,
        "iter_surrogate": iter_surrogate,
        "speedup_total": decomp.get("speedup_total"),
        "speedup_algebraic": decomp.get("speedup_algebraic"),
        "frac_algebraic_pct": decomp.get("frac_algebraic_pct"),
        "frac_physics_pct": decomp.get("frac_physics_pct"),
        "primary_component": "ALGEBRAIC" if decomp.get("frac_algebraic", 0) > 0.5 else "PHYSICS",
        "paper_claimed_speedup": PAPER_SPEEDUP,
        "time_cold_s": round(t_cold, 3),
        "time_zero_df_s": round(t_zero_df, 3) if t_zero_df else None,
        "time_surrogate_s": round(t_surrogate, 3) if t_surrogate else None,
        "flagged_deviations": [],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print("")
    print("=" * 60)
    print("Results:")
    print(json.dumps(result, indent=2))
    print("=" * 60)
    return result

if __name__ == "__main__":
    main()
