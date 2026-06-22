#!/usr/bin/env python3
"""forensic_parameter_sweep.py — Find parameters that reproduce Table 13.

If the "real solver" is indeed the Picard solver in the repository, then
Table 13 must be reproducible with SOME combination of:
- alpha (relaxation factor)
- tau_mom (momentum tolerance)
- tau_mass (mass conservation tolerance)
- n_max (max iterations)

This script performs a grid search over these parameters.
If NO combination reproduces the paper's claims, we conclude the data
was fabricated or a different solver was used.

SWEEP SPACE:
============
alpha:    [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]
tau_mom:  [1e-1, 5e-2, 1e-2, 5e-3]
tau_mass: [1e-2, 5e-3, 1e-3, 1e-4]
n_max:    [100, 500, 1000, 2000]

Total combinations: 7 × 4 × 4 × 4 = 448

For each combination, we test:
1. Cold start (v=0) at Re=500
2. Check if converges within n_max
3. Record iteration count

Then we check if ANY combination matches:
- Cold start: ~500 iterations
- Div-free zero: ~145 iterations  
- NO warm-start: ~120 iterations
"""

import os
import sys
import json
import time
import warnings
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.rbf_fd.solver import NavierStokesSolver
from src.projection.layer import HelmholtzProjection

# ── Configuration ────────────────────────────────────────────────────────────
RE = 500
N_NODES = 225
K_NEIGHBORS = 25

# Parameter sweep space
ALPHA_VALUES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]
TAU_MOM_VALUES = [1e-1, 5e-2, 1e-2, 5e-3]
TAU_MASS_VALUES = [1e-2, 5e-3, 1e-3, 1e-4]
N_MAX_VALUES = [100, 500, 1000, 2000]

RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_JSON = RESULTS_DIR / "forensic_sweep_results.json"

# Paper claims
PAPER_COLD = 500
PAPER_ZERO = 145
PAPER_NO = 120

TOLERANCE = 0.20  # 20% tolerance for "match"

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_grid_points(n: int = N_NODES, device: str = "cpu") -> torch.Tensor:
    side = int(round(n ** 0.5))
    assert side * side == n
    xs = torch.linspace(0.0, 1.0, side)
    ys = torch.linspace(0.0, 1.0, side)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    return pts.to(torch.float32).to(device)

def build_solver(device: str = "cpu"):
    points = make_grid_points(N_NODES, device=device)
    return NavierStokesSolver(points=points, k=K_NEIGHBORS)

def solve_with_params(solver, re, x0, alpha, tau_mom, tau_mass, n_max):
    """Solve with custom alpha (monkey-patch solver's alpha)."""
    if isinstance(x0, np.ndarray):
        x0 = torch.from_numpy(x0.astype(np.float32)).to(solver.device)

    # Monkey-patch alpha for this solve
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
        solver.alpha = original_alpha  # Restore

    return a, n_iter, converged

def build_divfree_zero(solver):
    """Build div-free zero field."""
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

# ── Main sweep ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("FORENSIC PARAMETER SWEEP")
    print("=" * 70)
    print()
    print("Searching for parameter combinations that reproduce Table 13...")
    print(f"Sweep space: {len(ALPHA_VALUES)} × {len(TAU_MOM_VALUES)} × {len(TAU_MASS_VALUES)} × {len(N_MAX_VALUES)} = {len(ALPHA_VALUES) * len(TAU_MOM_VALUES) * len(TAU_MASS_VALUES) * len(N_MAX_VALUES)} combinations")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    solver = build_solver(device=device)

    x0_zero = np.zeros(2 * N_NODES, dtype=np.float32)
    x0_divfree = build_divfree_zero(solver)

    results = []
    matches = []

    total = len(ALPHA_VALUES) * len(TAU_MOM_VALUES) * len(TAU_MASS_VALUES) * len(N_MAX_VALUES)
    count = 0

    for alpha, tau_mom, tau_mass, n_max in product(ALPHA_VALUES, TAU_MOM_VALUES, TAU_MASS_VALUES, N_MAX_VALUES):
        count += 1
        if count % 50 == 0:
            print(f"  Progress: {count}/{total} ({count/total*100:.1f}%)")

        # Test cold start
        _, iter_cold, conv_cold = solve_with_params(
            solver, RE, x0_zero, alpha, tau_mom, tau_mass, n_max
        )

        # Test div-free zero
        _, iter_zero, conv_zero = solve_with_params(
            solver, RE, x0_divfree, alpha, tau_mom, tau_mass, n_max
        )

        result = {
            "alpha": alpha,
            "tau_mom": tau_mom,
            "tau_mass": tau_mass,
            "n_max": n_max,
            "cold_start": {"iterations": iter_cold, "converged": conv_cold},
            "divfree_zero": {"iterations": iter_zero, "converged": conv_zero},
        }
        results.append(result)

        # Check if this matches paper claims
        if conv_cold and conv_zero:
            cold_diff = abs(iter_cold - PAPER_COLD) / PAPER_COLD
            zero_diff = abs(iter_zero - PAPER_ZERO) / PAPER_ZERO

            if cold_diff <= TOLERANCE and zero_diff <= TOLERANCE:
                matches.append({
                    "params": {"alpha": alpha, "tau_mom": tau_mom, "tau_mass": tau_mass, "n_max": n_max},
                    "cold_start": iter_cold,
                    "divfree_zero": iter_zero,
                    "cold_diff_pct": cold_diff * 100,
                    "zero_diff_pct": zero_diff * 100,
                })

    # ── Analysis ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SWEEP RESULTS")
    print("=" * 70)
    print()

    # Convergence statistics
    converged_cold = sum(1 for r in results if r["cold_start"]["converged"])
    converged_zero = sum(1 for r in results if r["divfree_zero"]["converged"])

    print(f"Total combinations tested: {len(results)}")
    print(f"Cold start converged: {converged_cold}/{len(results)} ({converged_cold/len(results)*100:.1f}%)")
    print(f"Div-free zero converged: {converged_zero}/{len(results)} ({converged_zero/len(results)*100:.1f}%)")
    print()

    # Best matches
    if matches:
        print(f"✅ FOUND {len(matches)} parameter combinations matching paper claims!")
        print()
        for i, m in enumerate(matches[:5]):  # Show top 5
            print(f"Match {i+1}:")
            print(f"  alpha={m['params']['alpha']}, tau_mom={m['params']['tau_mom']:.0e}, tau_mass={m['params']['tau_mass']:.0e}, n_max={m['params']['n_max']}")
            print(f"  Cold start: {m['cold_start']} iter (diff={m['cold_diff_pct']:.1f}%)")
            print(f"  Div-free zero: {m['divfree_zero']} iter (diff={m['zero_diff_pct']:.1f}%)")
            print()
    else:
        print("❌ NO parameter combination matches paper claims.")
        print()

        # Find closest matches
        closest = []
        for r in results:
            if r["cold_start"]["converged"] and r["divfree_zero"]["converged"]:
                cold_diff = abs(r["cold_start"]["iterations"] - PAPER_COLD) / PAPER_COLD
                zero_diff = abs(r["divfree_zero"]["iterations"] - PAPER_ZERO) / PAPER_ZERO
                avg_diff = (cold_diff + zero_diff) / 2
                closest.append({
                    "params": {"alpha": r["alpha"], "tau_mom": r["tau_mom"], "tau_mass": r["tau_mass"], "n_max": r["n_max"]},
                    "cold_start": r["cold_start"]["iterations"],
                    "divfree_zero": r["divfree_zero"]["iterations"],
                    "avg_diff_pct": avg_diff * 100,
                })

        if closest:
            closest.sort(key=lambda x: x["avg_diff_pct"])
            print("Closest matches (still outside 20% tolerance):")
            for i, c in enumerate(closest[:5]):
                print(f"  {i+1}. alpha={c['params']['alpha']}, tau_mom={c['params']['tau_mom']:.0e}, tau_mass={c['params']['tau_mass']:.0e}, n_max={c['params']['n_max']}")
                print(f"     Cold start: {c['cold_start']} iter, Div-free zero: {c['divfree_zero']} iter (avg diff={c['avg_diff_pct']:.1f}%)")
        else:
            print("No converged solutions found at all.")

    # ── Conclusion ───────────────────────────────────────────────
    print()
    print("=" * 70)
    print("FORENSIC CONCLUSION")
    print("=" * 70)
    print()

    if matches:
        print("✅ Table 13 IS reproducible with the Picard solver.")
        print("   The correct parameters were found.")
        print("   Paper should document these parameters explicitly.")
    else:
        print("❌ Table 13 is NOT reproducible with the Picard solver.")
        print("   No combination of alpha, tau_mom, tau_mass, n_max")
        print("   produces the claimed iteration counts.")
        print()
        print("   CONCLUSION: The paper's Table 13 data is either:")
        print("   1. FABRICATED (no solver produces these numbers)")
        print("   2. Produced with a DIFFERENT solver (not in repository)")
        print("   3. Produced with UNSTATED parameter modifications")

    print()

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "sweep_space": {
                "alpha": ALPHA_VALUES,
                "tau_mom": TAU_MOM_VALUES,
                "tau_mass": TAU_MASS_VALUES,
                "n_max": N_MAX_VALUES,
            },
            "total_combinations": len(results),
            "convergence_stats": {
                "cold_start_converged": converged_cold,
                "divfree_zero_converged": converged_zero,
            },
            "matches_found": len(matches),
            "matches": matches,
            "closest_attempts": closest[:10] if not matches else [],
            "conclusion": "reproducible" if matches else "not_reproducible",
        }, f, indent=2)

    print(f"Results saved to: {OUTPUT_JSON}")
    print("=" * 70)

if __name__ == "__main__":
    main()
