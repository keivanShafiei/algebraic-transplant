#!/usr/bin/env python3
"""quick_forensic_sweep.py — Quick parameter sweep (reduced space).

Tests a smaller parameter space to quickly assess reproducibility.
If this fails, the full sweep is unlikely to succeed.
"""

import os
import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.rbf_fd.solver import NavierStokesSolver
from src.projection.layer import HelmholtzProjection

RE = 500
N_NODES = 225
K_NEIGHBORS = 25

# Reduced sweep space
ALPHA_VALUES = [0.1, 0.2, 0.3, 0.5, 0.7]
TAU_MOM_VALUES = [1e-1, 5e-2, 1e-2]
TAU_MASS_VALUES = [1e-2, 5e-3, 1e-3]
N_MAX = 500

PAPER_COLD = 500
PAPER_ZERO = 145

def make_grid_points(n=225, device="cpu"):
    side = int(round(n ** 0.5))
    xs = torch.linspace(0.0, 1.0, side)
    ys = torch.linspace(0.0, 1.0, side)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    return pts.to(torch.float32).to(device)

def build_solver(device="cpu"):
    points = make_grid_points(N_NODES, device=device)
    return NavierStokesSolver(points=points, k=K_NEIGHBORS)

def solve_with_params(solver, re, x0, alpha, tau_mom, tau_mass, n_max):
    if isinstance(x0, np.ndarray):
        x0 = torch.from_numpy(x0.astype(np.float32)).to(solver.device)

    original_alpha = solver.alpha
    solver.alpha = alpha

    try:
        a, b, n_iter = solver.solve(
            Re=re, x0=x0, tau_mom=tau_mom,
            tau_mass=tau_mass, n_max=n_max
        )
        converged = n_iter < n_max
    except Exception as e:
        n_iter = n_max
        converged = False
    finally:
        solver.alpha = original_alpha

    return n_iter, converged

def build_divfree_zero(solver):
    a = np.zeros(2 * solver.N, dtype=np.float32)
    lid_idx = solver.is_lid.nonzero(as_tuple=True)[0].cpu().numpy()
    a[2 * lid_idx] = 1.0

    a_t = torch.from_numpy(a).to(solver.device)
    proj = HelmholtzProjection(
        G=solver.G_int, eps=1e-8,
        interior_mask=solver.interior_dof_mask
    ).to(solver.device)

    with torch.no_grad():
        a_proj = proj(a_t)

    return a_proj.cpu().numpy()

def main():
    print("=" * 70)
    print("QUICK FORENSIC SWEEP (reduced space)")
    print("=" * 70)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    solver = build_solver(device=device)

    x0_zero = np.zeros(2 * N_NODES, dtype=np.float32)
    x0_divfree = build_divfree_zero(solver)

    results = []
    matches = []

    from itertools import product
    total = len(ALPHA_VALUES) * len(TAU_MOM_VALUES) * len(TAU_MASS_VALUES)
    count = 0

    for alpha, tau_mom, tau_mass in product(ALPHA_VALUES, TAU_MOM_VALUES, TAU_MASS_VALUES):
        count += 1

        iter_cold, conv_cold = solve_with_params(
            solver, RE, x0_zero, alpha, tau_mom, tau_mass, N_MAX
        )
        iter_zero, conv_zero = solve_with_params(
            solver, RE, x0_divfree, alpha, tau_mom, tau_mass, N_MAX
        )

        results.append({
            "alpha": alpha,
            "tau_mom": tau_mom,
            "tau_mass": tau_mass,
            "cold_start": {"iterations": iter_cold, "converged": conv_cold},
            "divfree_zero": {"iterations": iter_zero, "converged": conv_zero},
        })

        if conv_cold and conv_zero:
            cold_diff = abs(iter_cold - PAPER_COLD) / PAPER_COLD
            zero_diff = abs(iter_zero - PAPER_ZERO) / PAPER_ZERO
            if cold_diff <= 0.20 and zero_diff <= 0.20:
                matches.append({
                    "alpha": alpha, "tau_mom": tau_mom, "tau_mass": tau_mass,
                    "cold_start": iter_cold, "divfree_zero": iter_zero,
                    "cold_diff": cold_diff * 100, "zero_diff": zero_diff * 100
                })

    print(f"Total combinations: {len(results)}")
    print(f"Cold start converged: {sum(1 for r in results if r['cold_start']['converged'])}/{len(results)}")
    print(f"Div-free zero converged: {sum(1 for r in results if r['divfree_zero']['converged'])}/{len(results)}")
    print(f"Matches found: {len(matches)}")
    print()

    if matches:
        print("✅ MATCHES FOUND:")
        for m in matches:
            print(f"  alpha={m['alpha']}, tau_mom={m['tau_mom']:.0e}, tau_mass={m['tau_mass']:.0e}")
            print(f"  Cold: {m['cold_start']} iter (diff={m['cold_diff']:.1f}%), Zero: {m['divfree_zero']} iter (diff={m['zero_diff']:.1f}%)")
    else:
        print("❌ NO MATCHES. Table 13 not reproducible with Picard solver.")

        # Show closest
        closest = []
        for r in results:
            if r['cold_start']['converged'] and r['divfree_zero']['converged']:
                cold_diff = abs(r['cold_start']['iterations'] - PAPER_COLD) / PAPER_COLD
                zero_diff = abs(r['divfree_zero']['iterations'] - PAPER_ZERO) / PAPER_ZERO
                closest.append({
                    "params": (r['alpha'], r['tau_mom'], r['tau_mass']),
                    "cold": r['cold_start']['iterations'],
                    "zero": r['divfree_zero']['iterations'],
                    "avg_diff": (cold_diff + zero_diff) / 2 * 100
                })

        if closest:
            closest.sort(key=lambda x: x['avg_diff'])
            print("\nClosest attempts:")
            for c in closest[:5]:
                print(f"  alpha={c['params'][0]}, tau_mom={c['params'][1]:.0e}, tau_mass={c['params'][2]:.0e}")
                print(f"  Cold: {c['cold']} iter, Zero: {c['zero']} iter (avg diff={c['avg_diff']:.1f}%)")

    print()
    print("=" * 70)

if __name__ == "__main__":
    main()
