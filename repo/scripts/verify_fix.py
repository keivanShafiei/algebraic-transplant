#!/usr/bin/env python3
"""scripts/verify_fix.py — Runs the exact Test 1/2/3 success criteria from
debug_prompt.md against the constrained-fix solver (src/rbf_fd/solver.py).

Honest expectation, established in AUDIT_REPORT.md: Tests 1 and 2 are
expected to FAIL, not because the outer nonlinear solver is broken, but
because the frozen operators (Gx, Gy, Lap -- which this fix is constrained
not to touch, since they determine G_int/G_int_int that are transplanted
into the GNN) fail basic consistency and cap the achievable momentum
residual at O(1e2-1e3), five orders of magnitude above tau_mom=1e-2.

This script does not soften or hide that outcome. It reports pass/fail
exactly as specified, plus the achieved metric, so the gap is visible.
"""
import sys
import os
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.cavity import generate_cavity_points
from src.rbf_fd.solver import NavierStokesSolver


def report(name, passed, detail):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return passed


def main():
    torch.manual_seed(0)
    points = generate_cavity_points(225)
    N = points.shape[0]
    solver = NavierStokesSolver(points, k=25, eps=1e-8)

    print("Operator self-diagnosis (see solver.operator_diagnostics):")
    for k, v in solver.operator_diagnostics.items():
        print(f"  {k}: {v}")
    print()

    results = []

    # ---------------- Test 1: Re=100 convergence ----------------
    print("=" * 70)
    print("TEST 1: Re=100 convergence, n_max=100")
    print("=" * 70)
    t0 = time.time()
    a, b, iters, mom_hist, div_hist = solver.solve(Re=100, n_max=100)
    dt = time.time() - t0
    div_final = (solver.G_int @ a).norm().item()
    mom_final = mom_hist[-1] if mom_hist else float("nan")
    ok_iters = iters < 100
    ok_div = div_final < 1e-4
    ok_mom = mom_final < 1e-2
    passed = ok_iters and ok_div and ok_mom
    results.append(report(
        "Test 1 (Re=100)", passed,
        f"iters={iters} (<100: {ok_iters}), div={div_final:.3e} (<1e-4: {ok_div}), "
        f"mom={mom_final:.3e} (<1e-2: {ok_mom}), time={dt:.1f}s",
    ))

    # ---------------- Test 2: Re=500 continuation ----------------
    print("\n" + "=" * 70)
    print("TEST 2: Re=500 continuation convergence, n_max_per_step=100")
    print("=" * 70)
    t0 = time.time()
    a2, b2, iters2, mom_hist2, div_hist2 = solver.solve_continuation(
        Re_target=500, n_max_per_step=100
    )
    dt2 = time.time() - t0
    ok_iters2 = iters2 < 500
    mom_final2 = mom_hist2[-1] if mom_hist2 else float("nan")
    results.append(report(
        "Test 2 (Re=500 continuation)", ok_iters2,
        f"iters={iters2} (<500: {ok_iters2}), final mom={mom_final2:.3e}, time={dt2:.1f}s",
    ))

    # ---------------- Test 3: warm-start effectiveness ----------------
    print("\n" + "=" * 70)
    print("TEST 3: Warm-start effectiveness")
    print("=" * 70)
    a_no = torch.randn(2 * N) * 0.1  # stand-in for a GNN output
    t0 = time.time()
    a_ws, b_ws, iters_ws, mh_ws, dh_ws = solver.solve_continuation(
        Re_target=500, x0=a_no, n_max_per_step=100
    )
    dt_ws = time.time() - t0
    t0 = time.time()
    a_cs, b_cs, iters_cs, mh_cs, dh_cs = solver.solve_continuation(
        Re_target=500, x0=None, n_max_per_step=100
    )
    dt_cs = time.time() - t0
    ok3 = iters_ws < iters_cs
    speedup = iters_cs / max(iters_ws, 1)
    results.append(report(
        "Test 3 (warm-start)", ok3,
        f"warm={iters_ws} iters ({dt_ws:.1f}s), cold={iters_cs} iters ({dt_cs:.1f}s), "
        f"speedup={speedup:.2f}x" if ok3 else
        f"warm={iters_ws} iters >= cold={iters_cs} iters -- NO speedup observed",
    ))

    print("\n" + "=" * 70)
    print("OVERALL")
    print("=" * 70)
    n_pass = sum(results)
    print(f"{n_pass}/3 tests passed.")
    if n_pass < 3:
        print(
            "\nExpected, given the constraints of this task. The frozen Gx/Gy/Lap "
            "operators (solver.operator_diagnostics['consistent'] == "
            f"{solver.operator_diagnostics['consistent']}) fail basic consistency "
            "(Lap@1 should be 0; it is not). No outer nonlinear solver strategy "
            "can be expected to pass tau_mom=1e-2 against operators that are not "
            "a consistent discretisation of the Navier-Stokes momentum operator. "
            "See AUDIT_REPORT.md, Section 'Corrected operators', for a reference "
            "fix that DOES pass all three tests -- at the cost of changing G_int / "
            "G_int_int, which requires regenerating the training data and "
            "retraining the GNN projection layer."
        )
    return 0 if n_pass == 3 else 1


if __name__ == "__main__":
    sys.exit(main())
