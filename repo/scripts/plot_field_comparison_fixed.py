"""Figure 6 — Field comparison at Re≈92 (fixed version)."""
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.gnn.neural_operator import NeuralOperator
from src.projection.layer import HelmholtzProjection
from src.data.cavity import generate_cavity_points
from src.rbf_fd.stencils import build_stencils
from src.utils.metrics import relative_l2_error

def main():
    config = yaml.safe_load(open('config.yaml'))
    device = torch.device('cpu')
    N = config['n_nodes_list'][0]
    k = config['stencil_k']

    model = NeuralOperator(n_nodes=N, d=2, param_dim=1, k=k,
                           hidden=config['hidden_dim'], layers=config['gnn_layers'])
    model.load_state_dict(torch.load('results/model_final.pt', map_location=device), strict=False)
    model.set_projection(G)
    model.eval()
    G = torch.load('data/fixed_G.pt', map_location=device)
    proj = HelmholtzProjection(G)

    points = generate_cavity_points(N).to(device)
    stencils = build_stencils(points, k)
    model.set_points(points, stencils)
    edge_dst = stencils.reshape(-1)
    edge_src = torch.arange(N, device=device).repeat_interleave(k)
    edge_index = torch.stack([edge_dst, edge_src])

    sample_dir = 'data/samples'
    files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.pt')])
    best_idx, best_re = 0, 0
    for i, f in enumerate(files):
        d = torch.load(os.path.join(sample_dir, f), map_location='cpu')
        re = d['re'].item()
        if abs(re - 92) < abs(best_re - 92):
            best_re, best_idx = re, i

    sample = torch.load(os.path.join(sample_dir, f"sample_{best_idx:04d}.pt"), map_location=device)
    mu = sample['mu'].reshape(-1).to(device)
    a_mean = sample['a_mean'].item()
    a_std  = sample['a_std'].item()
    b_mean = sample['b_mean'].item()
    b_std  = sample['b_std'].item()

    with torch.no_grad():
        a_hat, a_NO, b = model(mu, edge_index)
        a_pred = a_NO.squeeze(0) * a_std + a_mean
        b_pred = b.squeeze(0) * b_std + b_mean

    a_ref = sample['a_ref']
    b_ref = sample['b_ref']

    gs = int(N**0.5)
    u_ref = a_ref[0::2].reshape(gs, gs).cpu().numpy()
    v_ref = a_ref[1::2].reshape(gs, gs).cpu().numpy()
    p_ref = b_ref.reshape(gs, gs).cpu().numpy()
    u_pred = a_pred[0::2].reshape(gs, gs).cpu().numpy()
    v_pred = a_pred[1::2].reshape(gs, gs).cpu().numpy()
    p_pred = b_pred.reshape(gs, gs).cpu().numpy()

    err_u = relative_l2_error(torch.from_numpy(u_pred).flatten(), torch.from_numpy(u_ref).flatten())
    err_v = relative_l2_error(torch.from_numpy(v_pred).flatten(), torch.from_numpy(v_ref).flatten())
    err_p = relative_l2_error(torch.from_numpy(p_pred).flatten(), torch.from_numpy(p_ref).flatten())

    print(f"Re = {best_re:.1f} | L2 errors → u: {err_u:.1%} | v: {err_v:.1%} | p: {err_p:.1%}")

    fig, axs = plt.subplots(3, 3, figsize=(16, 13))
    cmap = 'RdBu_r'
    titles = ['Reference (RBF-FD)', 'Neural Operator (Hard)', 'Absolute Error']

    for i, (u, v, p) in enumerate([(u_ref, v_ref, p_ref), (u_pred, v_pred, p_pred)]):
        axs[0, i].contourf(u, levels=60, cmap=cmap)
        axs[0, i].set_title(f'u-velocity — {titles[i]}')
        axs[1, i].contourf(v, levels=60, cmap=cmap)
        axs[1, i].set_title(f'v-velocity — {titles[i]}')
        axs[2, i].contourf(p, levels=60, cmap='viridis')
        axs[2, i].set_title(f'pressure — {titles[i]}')

    axs[0, 2].contourf(np.abs(u_pred - u_ref), levels=60, cmap='hot')
    axs[0, 2].set_title(f'|Δu|  (L₂ rel = {err_u:.1%})')
    axs[1, 2].contourf(np.abs(v_pred - v_ref), levels=60, cmap='hot')
    axs[1, 2].set_title(f'|Δv|  (L₂ rel = {err_v:.1%})')
    axs[2, 2].contourf(np.abs(p_pred - p_ref), levels=60, cmap='hot')
    axs[2, 2].set_title(f'|Δp|  (L₂ rel = {err_p:.1%})')

    for ax in axs.flat:
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f'Figure 6: Field Comparison at Re={best_re:.1f}  (N=225)', fontsize=16)
    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/fig6_field_comparison_Re92_fixed.png', dpi=500, bbox_inches='tight')
    print("✅ Figure 6 (fixed) saved!")
    plt.show()

if __name__ == '__main__':
    main()
