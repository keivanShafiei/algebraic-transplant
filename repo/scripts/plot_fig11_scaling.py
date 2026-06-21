"""
plot_fig11_scaling_fixed.py — نسخه تصحیح‌شده
============================================

باگ اصلی نسخه قبلی:
  مدل با N=225 ساخته شده بود اما برای N=961 صدا زده می‌شد.
  G matrix ابعاد (225, 450) دارد اما a_hat برای N=961 ابعاد (1922,) دارد.
  → RuntimeError: size mismatch

راه‌حل: برای هر N جدید، مدل، G، و edge_index از نو ساخته می‌شوند.
"""

import time
import sys
import os
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cavity import generate_cavity_points
from src.rbf_fd.stencils import build_stencils
from src.rbf_fd.operators import assemble_divergence_operator
from src.rbf_fd.solver import NavierStokesSolver
from src.gnn.neural_operator import NeuralOperator


def build_model_for_N(N: int, config: dict, device: torch.device):
    """مدل + G + edge_index را برای یک N مشخص بساز."""
    points   = generate_cavity_points(N).to(device)
    stencils = build_stencils(points, config['stencil_k']).to(device)
    diffs    = points[stencils[:, 1]] - points
    h_avg    = torch.norm(diffs, dim=1).mean().item()
    c        = config['rbf_c_factor'] * h_avg
    G        = assemble_divergence_operator(points, stencils, c).to(device)

    k = config['stencil_k']
    edge_dst   = stencils.reshape(-1)
    edge_src   = torch.arange(N, device=device).repeat_interleave(k)
    edge_index = torch.stack([edge_dst, edge_src])

    model = NeuralOperator(
        n_nodes=N, d=2, param_dim=1, k=k,
        hidden=config['hidden_dim'], layers=config['gnn_layers'],
        eps=float(config['projection_eps']),
    ).to(device)
    model.set_projection(G)
    model.set_points(points)
    model.eval()

    return model, points, G, edge_index


def time_solver(N: int, Re: float, config: dict, device: torch.device,
                n_repeat: int = 3):
    """زمان‌بندی solver برای N مشخص."""
    points = generate_cavity_points(N).to(device)
    solver = NavierStokesSolver(points, k=config['stencil_k'],
                                eps=float(config['projection_eps']))
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        _ = solver.solve(Re=Re, tau_mom=1e-2, n_max=100, verbose=False)
        times.append(time.perf_counter() - t0)
    return min(times)  # best-of-n برای حذف jitter


def time_model(model, mu: torch.Tensor, edge_index: torch.Tensor,
               n_repeat: int = 10):
    """زمان‌بندی GNN inference."""
    with torch.no_grad():
        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            model(mu, edge_index, inference=True)
            times.append(time.perf_counter() - t0)
    return min(times)


def main():
    config = yaml.safe_load(open('config.yaml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # N values from paper: 225, 1000, 5000, 10000
    # actual N = (int(sqrt(N)))² for uniform Cartesian grid
    N_targets = [225, 1000, 5000, 10000]
    Re_test   = 50.0

    print(f"Measuring timing for Figure 11 (device={device})")
    print(f"{'N_target':>10} {'N_actual':>10} {'Solver(s)':>12} {'Model(ms)':>12} {'Speedup':>10}")
    print("-" * 58)

    results = []
    for N_target in N_targets:
        # grid size: sqrt(N) × sqrt(N)
        side    = int(N_target ** 0.5)
        N_actual = side * side
        print(f"{N_target:>10} → {N_actual:>8} ...", end='', flush=True)

        # FIX: ساخت مدل جدید برای هر N
        model, points, G, edge_index = build_model_for_N(N_actual, config, device)
        mu = torch.tensor([Re_test / 100.0], dtype=torch.float32, device=device)

        t_solver = time_solver(N_actual, Re_test, config, device)
        t_model  = time_model(model, mu, edge_index)
        speedup  = t_solver / t_model

        print(f" | {t_solver:10.3f}s | {t_model*1000:10.2f}ms | {speedup:8.1f}x")
        results.append((N_actual, t_solver, t_model, speedup))

    # ذخیره نتایج برای رسم نمودار
    os.makedirs('results/figures', exist_ok=True)
    torch.save({'results': results}, 'results/fig11_data.pt')
    print("\nResults saved to results/fig11_data.pt")

    # رسم نمودار (اگر matplotlib موجود باشد)
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        Ns      = [r[0] for r in results]
        t_solv  = [r[1] * 1000 for r in results]   # ms
        t_mod   = [r[2] * 1000 for r in results]   # ms
        speedups = [r[3] for r in results]

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax2 = ax1.twinx()

        ax1.loglog(Ns, t_solv, 'b--o', label='RBF-FD Solver', linewidth=1.5)
        ax1.loglog(Ns, t_mod,  'g-o',  label='Neural Operator', linewidth=1.5)
        ax2.semilogx(Ns, speedups, 'r-^', label='Speedup', linewidth=1.5, alpha=0.7)

        ax1.set_xlabel('N (number of nodes)')
        ax1.set_ylabel('Time (ms)')
        ax2.set_ylabel('Speedup (T_solver / T_model)', color='r')
        ax1.legend(loc='upper left')
        ax2.legend(loc='lower right')
        ax1.set_title('Fig 11: Computational Scaling Analysis')
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/figures/fig11_scaling_fixed.png', dpi=150)
        print("Figure saved: results/figures/fig11_scaling_fixed.png")
        plt.close()
    except ImportError:
        print("matplotlib not available — skipping plot, data saved to fig11_data.pt")


if __name__ == '__main__':
    main()
