#!/usr/bin/env python3
"""scripts/reference_consistent_operators.py — Reference fix for the actual
root cause (see AUDIT_REPORT.md).

This module is intentionally NOT imported by src/rbf_fd/solver.py. Doing so
would change self.G_full -> self.G_int -> self.G_int_int, which the task
explicitly freezes ("transplanted into the GNN checkpoint"). Adopting this
fix for real is a decision for the paper's authors: it requires regenerating
data/samples/*.pt (the existing samples are not momentum-converged, see
AUDIT_REPORT.md) and retraining the GNN's projection layer against the new
G_int.

Method: for each node i with local stencil {x_j}_{j in S_i} (k points incl.
self), local coordinates xi_j = x_j - x_i, the augmented (k+P) x (k+P)
system

    [ Phi_loc   P_loc ] [ w   ]   [ L[phi_a](0), a in stencil        ]
    [ P_loc^T    0    ] [ lam ] = [ L[p_l](0),   l = 1..P (poly basis) ]

is solved for stencil weights w, where Phi_loc[a,b] = phi(||xi_a-xi_b||),
P_loc[a,:] = monomial basis evaluated at xi_a, and P = 6 for quadratic
augmentation in 2D ([1, x, y, x^2, xy, y^2] -- matching the paper's own
Eq. (3) / Assumption 2, m=2). This is standard RBF-FD (Fornberg & Flyer,
2015, Acta Numerica; Bayona, Flyer, Fornberg & Barnett, 2017, JCP) and is
what the paper claims to implement but does not.

Run as a script to reproduce the consistency-check comparison and the
Re=100 / Re=500 convergence demonstration reported in AUDIT_REPORT.md.
"""
import sys
import os
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.cavity import generate_cavity_points
from src.rbf_fd.stencils import build_stencils
from src.rbf_fd.solver import classify_cavity_nodes, build_bc_rhs, assemble_momentum_operator


def _poly_basis_2d(xi, degree=2):
    """Monomial basis at local coords xi (...,2). degree=2 -> [1,x,y,x^2,xy,y^2]."""
    x, y = xi[..., 0], xi[..., 1]
    terms = [torch.ones_like(x), x, y]
    if degree >= 2:
        terms += [x * x, x * y, y * y]
    return torch.stack(terms, dim=-1)


def _poly_L_at_origin_2d(degree, op):
    if op == "id":
        v = [1, 0, 0, 0, 0, 0]
    elif op == "dx":
        v = [0, 1, 0, 0, 0, 0]
    elif op == "dy":
        v = [0, 0, 1, 0, 0, 0]
    elif op == "lap":
        v = [0, 0, 0, 2, 0, 2]
    else:
        raise ValueError(op)
    if degree < 2:
        v = v[:3]
    return torch.tensor(v, dtype=torch.float64)


def assemble_consistent_operator(points, stencils, c, op, degree=2, ridge=1e-10):
    """Vectorized, polynomial-augmented local RBF-FD assembly for
    op in {'id','dx','dy','lap'}. Returns a dense (N,N) matrix (the
    underlying stencil structure is still k-sparse; densify only for
    convenience of drop-in comparison with the existing code)."""
    N, k = stencils.shape[0], stencils.shape[1]
    P = 6 if degree >= 2 else 3
    pts64 = points.to(torch.float64)

    neigh = pts64[stencils]
    center = pts64.unsqueeze(1)
    xi = neigh - center                                    # (N,k,2) local coords

    diff = xi.unsqueeze(2) - xi.unsqueeze(1)
    r_ab = torch.norm(diff, dim=-1)
    Phi_loc = torch.sqrt(1.0 + (r_ab / c) ** 2)             # (N,k,k)

    P_loc = _poly_basis_2d(xi, degree=degree)               # (N,k,P)

    Z = torch.zeros(N, P, P, dtype=torch.float64)
    top = torch.cat([Phi_loc, P_loc], dim=-1)
    bot = torch.cat([P_loc.transpose(1, 2), Z], dim=-1)
    A = torch.cat([top, bot], dim=1)
    A = A + ridge * torch.eye(k + P, dtype=torch.float64)

    r0 = torch.norm(xi, dim=-1).clamp(min=1e-12)
    if op == "id":
        rhs_phi = torch.sqrt(1.0 + (r0 / c) ** 2)
    elif op in ("dx", "dy"):
        dphi_dr = r0 / (c ** 2 * torch.sqrt(1.0 + (r0 / c) ** 2))
        comp = xi[..., 0] if op == "dx" else xi[..., 1]
        rhs_phi = -dphi_dr * (comp / r0)
    elif op == "lap":
        denom_sqrt = torch.sqrt(1.0 + (r0 / c) ** 2)
        rhs_phi = 2.0 / (c ** 2 * denom_sqrt) - (r0 ** 2) / (c ** 4 * denom_sqrt ** 3)
    else:
        raise ValueError(op)

    rhs_poly = _poly_L_at_origin_2d(degree, op).unsqueeze(0).expand(N, -1).to(torch.float64)
    rhs = torch.cat([rhs_phi, rhs_poly], dim=-1).unsqueeze(-1)

    sol = torch.linalg.solve(A, rhs).squeeze(-1)
    w = sol[:, :k]

    rows = torch.arange(N).repeat_interleave(k)
    cols = stencils.reshape(-1)
    M = torch.zeros(N, N, dtype=torch.float64)
    M.index_put_((rows, cols), w.reshape(-1), accumulate=True)
    return M.to(torch.float32)


def _demo():
    points = generate_cavity_points(225)
    N = points.shape[0]
    stencils = build_stencils(points, 25)
    c = 1.2 * torch.norm(points[stencils[:, 1]] - points, dim=1).mean().item()

    Dx = assemble_consistent_operator(points, stencils, c, op="dx")
    Dy = assemble_consistent_operator(points, stencils, c, op="dy")
    Lap = assemble_consistent_operator(points, stencils, c, op="lap")

    x, y = points[:, 0], points[:, 1]
    ones = torch.ones(N)
    print("=== Consistency checks (corrected operators) ===")
    print(f"Lap @ 1        (want 0): max|.| = {(Lap @ ones).abs().max().item():.3e}")
    print(f"Lap @ (x^2+y^2) (want 4): mean = {(Lap @ (x**2+y**2)).mean().item():.6f}, "
          f"std = {(Lap @ (x**2+y**2)).std().item():.3e}")
    print(f"Dx @ x         (want 1): mean = {(Dx @ x).mean().item():.6f}, "
          f"std = {(Dx @ x).std().item():.3e}")
    print(f"Dx @ 1         (want 0): max|.| = {(Dx @ ones).abs().max().item():.3e}")

    is_lid, is_wall, is_int = classify_cavity_nodes(points)
    F = build_bc_rhs(N, is_lid, is_wall, points.device)
    Phi_id = torch.eye(N)   # with proper RBF-FD, `a` are literal nodal values

    def solve_corrected(Re, n_max=100, alpha=0.7, tau_mom=1e-2):
        nu = 1.0 / Re
        a = torch.zeros(2 * N)
        for n in range(n_max):
            K = assemble_momentum_operator(a, Dx, Dy, Phi_id, Lap, nu, is_int)
            a_star = torch.linalg.solve(K, F)
            K_at_new = assemble_momentum_operator(a_star, Dx, Dy, Phi_id, Lap, nu, is_int)
            res = (K_at_new @ a_star - F).norm().item() / (F.norm().item() + 1e-12)
            a = alpha * a_star + (1 - alpha) * a
            if res < tau_mom:
                return a, n + 1, True
        return a, n_max, False

    print("\n=== Re=100, plain damped Picard (alpha=0.7), corrected operators ===")
    t0 = time.time()
    a, iters, conv = solve_corrected(100.0)
    print(f"iters={iters} converged={conv} time={time.time()-t0:.2f}s "
          f"u_range=[{a[0::2].min().item():.3f},{a[0::2].max().item():.3f}]")

    print("\n=== Re=500 via naive continuation, corrected operators ===")
    t0 = time.time()
    total = 0
    a = torch.zeros(2 * N)
    for Re_step in [10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
        nu = 1.0 / Re_step
        for n in range(100):
            K = assemble_momentum_operator(a, Dx, Dy, Phi_id, Lap, nu, is_int)
            a_star = torch.linalg.solve(K, F)
            K_new = assemble_momentum_operator(a_star, Dx, Dy, Phi_id, Lap, nu, is_int)
            res = (K_new @ a_star - F).norm().item() / (F.norm().item() + 1e-12)
            a = 0.7 * a_star + 0.3 * a
            total += 1
            if res < 1e-2:
                break
    print(f"total continuation iters to Re=500: {total}  time={time.time()-t0:.2f}s  "
          f"u_range=[{a[0::2].min().item():.4f},{a[0::2].max().item():.4f}]")
    print("\nNote: cond(Dx)/cond(Lap) with this MQ+quadratic-augmentation choice")
    print("are large (~1e9-1e10, see AUDIT_REPORT.md) even though the weights")
    print("themselves are accurate in float64. A production adoption should")
    print("re-tune the shape parameter c or switch to polyharmonic-spline+")
    print("polynomial RBF-FD (Bayona et al. 2017), which is well-conditioned")
    print("without a shape parameter to mistune.")


if __name__ == "__main__":
    _demo()
