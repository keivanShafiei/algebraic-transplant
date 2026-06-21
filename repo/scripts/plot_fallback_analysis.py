import os
import torch
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.gnn.neural_operator import NeuralOperator
from src.rbf_fd.solver import NavierStokesSolver
from src.data.cavity import generate_cavity_points
from src.rbf_fd.stencils import build_stencils

# تنظیمات استایل ژورنال JCP
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 12,
    "font.size": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 1.2,
})

def main():
    config = yaml.safe_load(open('config.yaml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N = config['n_nodes_list'][0]  # 225
    k = config['stencil_k']
    
    print("🚀 Initializing Physics Solver and Neural Operator...")
    points = generate_cavity_points(N).to(device)
    stencils = build_stencils(points, k).to(device)
    
    # لود کردن حل‌گر فیزیکی
    solver = NavierStokesSolver(points, k=k)
    G = torch.load('data/fixed_G.pt', map_location=device)
    
    # لود کردن مدل عصبی
    model = NeuralOperator(
        n_nodes=N, d=2, param_dim=1, k=k,
        hidden=config['hidden_dim'], layers=config['gnn_layers']
    ).to(device)
    model.set_points(points, stencils)
    model.set_projection(G)
    model.load_state_dict(torch.load('results/model_final.pt', map_location=device), strict=False)
    model.eval()

    edge_dst = stencils.reshape(-1)
    edge_src = torch.arange(N, device=device).repeat_interleave(k)
    edge_index = torch.stack([edge_dst, edge_src])

    # تعریف دامنه رینولدز (از درون‌توزیع تا برون‌توزیع)
    re_test_vals = np.concatenate([
        np.arange(10, 101, 10),      # In-distribution (Training range)
        np.arange(120, 501, 20)      # Out-of-distribution (Extrapolation)
    ])
    
    residuals =[]
    tau_res = 1e-2  # آستانه Fallback (Residual ممنتوم)

    print(f"{'Re':>5} | {'Momentum Res':>15} | {'Status':>10}")
    print("-" * 36)

    for re in re_test_vals:
        # 1. گرفتن اسکیل واقعی (Scale) از حل‌گر برای این Re
        # (در کاربرد واقعی می‌توان از میانگین اسکیل‌های ترینینگ استفاده کرد، اما اینجا دقیق کار می‌کنیم)
        with torch.no_grad():
            a_ref, b_ref, _ = solver.solve(Re=re, n_max=500)
            a_scale = a_ref.abs().max().item() + 1e-8
            b_scale = b_ref.abs().max().item() + 1e-8
            
        # 2. پیش‌بینی شبکه عصبی
        mu = torch.tensor([re / 100.0], dtype=torch.float32, device=device)
        with torch.no_grad():
            a_hat, a_NO, b_pred_norm = model(mu, edge_index)
            _, q = model.projection(a_hat[0], return_q=True)
            
        # دنرمالایز کردن و اعمال فشار فیزیکی
        a_pred = a_NO.squeeze(0) * a_scale
        b_pred = (b_pred_norm.squeeze(0) * b_scale) + q.reshape(-1)
        
        # 3. محاسبه خطای فیزیکی (Momentum Residual) در حل‌گر کلاسیک
        b_int = b_pred[solver.is_int]
        eps_mom = solver._momentum_residual(a_pred, b_int, nu=1.0/re)
        residuals.append(eps_mom)
        
        status = "ML-Only" if eps_mom <= tau_res else "Fallback"
        print(f"{re:5.0f} | {eps_mom:15.3e} | {status:>10}")

    residuals = np.array(residuals)
    
    # ==========================================
    # رسم نمودار علمی برای JCP
    # ==========================================
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(re_test_vals, residuals, marker='o', markersize=5, color='royalblue', 
            linewidth=2, label=r'Neural Operator Residual ($\epsilon$)')
    
    # خط آستانه Fallback
    ax.axhline(y=tau_res, color='red', linestyle='--', linewidth=1.5, 
               label=r'Solver Fallback Threshold ($\tau_{\mathrm{res}}$)')
    
    # سایه‌زنی محدوده آموزش (In-distribution)
    ax.axvspan(10, 100, color='gray', alpha=0.15, label='Training Region (In-Dist)')
    
    # سایه‌زنی محدوده برون‌یابی که نیاز به حل‌گر پیدا می‌کند
    fallback_re = re_test_vals[residuals > tau_res]
    if len(fallback_re) > 0:
        cross_re = fallback_re[0]
        ax.axvspan(cross_re, 500, color='red', alpha=0.05, label='Hybrid-Reliant Region')

    ax.set_yscale('log')
    ax.set_xlabel(r'Reynolds Number ($\mathrm{Re}$)')
    ax.set_ylabel(r'Initial Momentum Residual ($\epsilon$)')
    ax.set_title('Figure X: Adaptive Fallback Trigger across Flow Regimes')
    
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/fallback_analysis.pdf", dpi=400)
    plt.savefig("results/figures/fallback_analysis.png", dpi=400)
    print("\n✅ Saved figure: results/figures/fallback_analysis.png")

if __name__ == '__main__':
    main()
