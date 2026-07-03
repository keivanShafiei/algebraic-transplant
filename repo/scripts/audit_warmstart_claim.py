#!/usr/bin/env python3
"""audit_warmstart_claim.py — Reproduce Table 10 & Table 13 warm-start claims.

This script audits the paper's warm-start claims by:
  1. Testing convergence at Re=100 (training range) with direct solver
  2. Testing Re=500 with BOTH pure Picard and continuation solver
  3. Testing warm-start with div-free zero field and GNN surrogate
  4. Comparing against paper's reported values

Usage:
    python scripts/audit_warmstart_claim.py
"""

import sys
import os
import json
import time
import torch
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rbf_fd.solver import NavierStokesSolver
from src.data.cavity import generate_cavity_nodes
from src.gnn.neural_operator import NeuralOperator
from src.projection.layer import HelmholtzProjection


def generate_div_free_zero(N, device):
    """Generate a divergence-free zero field (all zeros)."""
    return torch.zeros(2 * N, dtype=torch.float32, device=device)


def load_gnn_surrogate(checkpoint_path, solver, device):
    """Load trained GNN and return a callable surrogate function."""
    if not os.path.exists(checkpoint_path):
        return None
    try:
        from src.gnn.neural_operator import NeuralOperator
        model = NeuralOperator(
            in_dim=2,
            hidden_dim=128,
            out_dim=2,
            num_layers=6,
            num_nodes=solver.N,
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Transplant G_int into projection layer
        model.set_projection(solver.G_int)

        def surrogate_fn(Re):
            mu = torch.tensor([[Re / 100.0]], dtype=torch.float32, device=device)
            edge_index = torch.load(
                os.path.join(os.path.dirname(checkpoint_path), '..', 'data', 'edge_index.pt')
            ).to(device)
            with torch.no_grad():
                a_hat, a_no, b_pred = model(mu, edge_index)
            return a_no.squeeze(0)  # shape (2N,)

        return surrogate_fn
    except Exception as e:
        print(f"WARNING: Failed to load surrogate: {e}")
        return None


def run_solver_direct(solver, Re, x0, label, n_max=100, verbose=False):
    """Run direct Picard solver (for Re <= 100)."""
    start = time.time()
    a, b_full, iters, mom_hist, div_hist = solver.solve(
        Re=Re,
        x0=x0,
        tau_mom=1e-2,
        tau_mass=1e-4,
        n_max=n_max,
        use_iterative=True,
        adaptive_relax=True,
        verbose=verbose,
    )
    elapsed = time.time() - start
    converged = (mom_hist[-1] < 1e-2) and (div_hist[-1] < 1e-4) if mom_hist else False
    status = "CONVERGED" if converged else "MAX_ITER" if iters >= n_max else "NOT_CONVERGED"

    print(f"INFO:   {label}: {iters} iters, {elapsed:.2f}s, {status} (mom={mom_hist[-1]:.2e})")

    return {
        'a': a,
        'b': b_full,
        'iters': iters,
        'mom_hist': mom_hist,
        'div_hist': div_hist,
        'converged': converged,
        'time_s': elapsed,
    }


def run_solver_continuation(solver, Re_target, x0, label, n_max_per_step=100, verbose=False):
    """Run continuation solver (for Re > 100)."""
    start = time.time()
    a, b_full, total_iters, mom_hist, div_hist = solver.solve_continuation(
        Re_target=Re_target,
        x0=x0,
        tau_mom=1e-2,
        tau_mass=1e-4,
        n_max_per_step=n_max_per_step,
        use_iterative=True,
        adaptive_relax=True,
        verbose=verbose,
    )
    elapsed = time.time() - start
    converged = (mom_hist[-1] < 1e-2) and (div_hist[-1] < 1e-4) if mom_hist else False
    status = "CONVERGED" if converged else "NOT_CONVERGED"

    print(f"INFO:   {label}: {total_iters} total iters, {elapsed:.2f}s, {status} (mom={mom_hist[-1]:.2e})")

    return {
        'a': a,
        'b': b_full,
        'iters': total_iters,
        'mom_hist': mom_hist,
        'div_hist': div_hist,
        'converged': converged,
        'time_s': elapsed,
    }


def print_history(label, mom_hist, div_hist, n_show=5):
    """Pretty-print residual history."""
    print(f"INFO:   Mom residual history (first {n_show}): {[f'{v:.2e}' for v in mom_hist[:n_show]]}")
    print(f"INFO:   Mom residual history (last {n_show}):  {[f'{v:.2e}' for v in mom_hist[-n_show:]]}")
    print(f"INFO:   Div residual history (first {n_show}): {[f'{v:.2e}' for v in div_hist[:n_show]]}")
    print(f"INFO:   Div residual history (last {n_show}):  {[f'{v:.2e}' for v in div_hist[-n_show:]]}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"INFO: Device: {device}")

    # Configuration
    N = 225
    Re_test = 500.0
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best.pt')

    # Generate node set
    points = generate_cavity_points(N).to(device)
    print(f"INFO: Assembling solver (n={N}, Re={Re_test})")

    solver = NavierStokesSolver(points, k=25, eps=1e-8)

    # ============================================================
    # TEST 1: Re=100 (within training range) — direct solver
    # ============================================================
    print("\n############################################################")
    print("INFO: # TEST 1: Re=100 (within training range)")
    print("INFO: ############################################################")

    res_cold_100 = run_solver_direct(solver, 100.0, None, "Cold start (Re=100)", n_max=100)
    print_history("Cold start (Re=100)", res_cold_100['mom_hist'], res_cold_100['div_hist'])

    div_free_100 = generate_div_free_zero(N, device)
    res_df_100 = run_solver_direct(solver, 100.0, div_free_100, "Div-free zero (Re=100)", n_max=100)
    print_history("Div-free zero (Re=100)", res_df_100['mom_hist'], res_df_100['div_hist'])

    # ============================================================
    # TEST 2: Re=500 (extrapolation) — continuation solver
    # ============================================================
    print("\n############################################################")
    print("INFO: # TEST 2: Re=500 (extrapolation)")
    print("INFO: ############################################################")
    print("INFO: Using continuation solver (Re=10→20→50→100→200→300→400→500)")
    print("INFO: Paper's Table 10 reports iteration counts with continuation.")

    # Cold start with continuation
    res_cold_500 = run_solver_continuation(
        solver, 500.0, None, "Cold start continuation (Re=500)", n_max_per_step=100
    )
    print_history("Cold start cont.", res_cold_500['mom_hist'], res_cold_500['div_hist'])

    # Div-free zero with continuation
    div_free_500 = generate_div_free_zero(N, device)
    res_df_500 = run_solver_continuation(
        solver, 500.0, div_free_500, "Div-free zero continuation (Re=500)", n_max_per_step=100
    )
    print_history("Div-free zero cont.", res_df_500['mom_hist'], res_df_500['div_hist'])

    # GNN surrogate with continuation
    surrogate_fn = load_gnn_surrogate(checkpoint_path, solver, device)
    if surrogate_fn is not None:
        a_no = surrogate_fn(Re_test)
        # Verify div-free property
        div_no = (solver.G_int @ a_no).norm().item()
        print(f"INFO:   GNN surrogate ||G_int a_NO|| = {div_no:.2e}")

        res_surrogate_500 = run_solver_continuation(
            solver, 500.0, a_no, "Surrogate continuation (Re=500)", n_max_per_step=100
        )
        print_history("Surrogate cont.", res_surrogate_500['mom_hist'], res_surrogate_500['div_hist'])
    else:
        print("INFO: WARNING: No GNN checkpoint found. Skipping surrogate test.")
        res_surrogate_500 = None

    # ============================================================
    # TEST 3: Pure Picard comparison (for reference)
    # ============================================================
    print("\n############################################################")
    print("INFO: # TEST 3: Pure Picard at Re=500 (paper footnote)")
    print("INFO: ############################################################")
    print("INFO: Paper footnote: pure Picard cold=3000, zero-df=500, surrogate=500")

    res_picard_cold = run_solver_direct(
        solver, 500.0, None, "Pure Picard cold (Re=500)", n_max=3000
    )
    print_history("Pure Picard cold", res_picard_cold['mom_hist'], res_picard_cold['div_hist'])

    # ============================================================
    # TEST 4: Div-free property verification
    # ============================================================
    print("\n############################################################")
    print("INFO: # TEST 4: Verify div-free property")
    print("INFO: ############################################################")
    zero_field = torch.zeros(2 * N, device=device)
    div_zero = (solver.G_int @ zero_field).norm().item()
    print(f"INFO:   Zero field div residual: {div_zero:.2e}")
    print(f"INFO:   Expected: ~0 (zero field is trivially div-free)")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n======================================================================")
    print("INFO: WARM-START AUDIT SUMMARY")
    print("INFO: ==================================================================")

    # Re=100 results
    print(f"INFO: [Re=100] Cold start:      {res_cold_100['iters']} iters, converged={res_cold_100['converged']}")
    print(f"INFO: [Re=100] Div-free zero:   {res_df_100['iters']} iters, converged={res_df_100['converged']}")

    # Re=500 continuation results
    print(f"INFO: [Re=500] Cont. cold:      {res_cold_500['iters']} iters, converged={res_cold_500['converged']}")
    print(f"INFO: [Re=500] Cont. zero-df:   {res_df_500['iters']} iters, converged={res_df_500['converged']}")
    if res_surrogate_500:
        print(f"INFO: [Re=500] Cont. surrogate: {res_surrogate_500['iters']} iters, converged={res_surrogate_500['converged']}")

        # Speedup calculation
        speedup = res_cold_500['iters'] / max(1, res_surrogate_500['iters'])
        print(f"INFO: [Re=500] Speedup (cold/surrogate): {speedup:.2f}×")
        print(f"INFO: Paper claim: 4.2×")

    # Pure Picard comparison
    print(f"INFO: [Re=500] Pure Picard cold: {res_picard_cold['iters']} iters, converged={res_picard_cold['converged']}")

    # Save results
    results = {
        "Re": Re_test,
        "N": N,
        "device": str(device),
        "Re_100": {
            "cold_iters": res_cold_100['iters'],
            "cold_converged": res_cold_100['converged'],
            "zero_df_iters": res_df_100['iters'],
            "zero_df_converged": res_df_100['converged'],
        },
        "Re_500_continuation": {
            "cold_iters": res_cold_500['iters'],
            "cold_converged": res_cold_500['converged'],
            "zero_df_iters": res_df_500['iters'],
            "zero_df_converged": res_df_500['converged'],
            "surrogate_iters": res_surrogate_500['iters'] if res_surrogate_500 else None,
            "surrogate_converged": res_surrogate_500['converged'] if res_surrogate_500 else None,
            "speedup": res_cold_500['iters'] / max(1, res_surrogate_500['iters']) if res_surrogate_500 else None,
        },
        "Re_500_pure_picard": {
            "cold_iters": res_picard_cold['iters'],
            "cold_converged": res_picard_cold['converged'],
        },
        "paper_claims": {
            "table10_continuation_cold": 500,
            "table10_continuation_surrogate": 120,
            "table10_speedup": 4.2,
            "footnote_picard_cold": 3000,
            "footnote_picard_zero_df": 500,
            "footnote_picard_surrogate": 500,
        }
    }

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'warmstart_audit.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"INFO: Saved to {out_path}")

    print("INFO: ==================================================================")


if __name__ == '__main__':
    main()
