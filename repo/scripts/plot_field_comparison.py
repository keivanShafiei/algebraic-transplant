"""Generate Figure 6: Field comparison at Re=92 (solver vs Neural Operator)."""
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.gnn.neural_operator import NeuralOperator
from src.projection.layer import HelmholtzProjection
from src.data.cavity import generate_cavity_points
from src.rbf_fd.stencils import build_stencils

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

    points = generate_cavity_points(N).to(device)
    stencils = build_stencils(points, k)
    model.set_points(points, stencils)
    edge_dst = stencils.reshape(-1)
    edge_src = torch.arange(N, device=device).repeat_interleave(k)
    edge_index = torch.stack([edge_dst, edge_src])

    G = torch.load('data/fixed_G.pt', map_location=device)
    proj = HelmholtzProjection(G)

    sample_dir = 'data/samples'
    files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.pt')])
    best_idx, best_re = 0, 0
    target = 92.0
    for i, f in enumerate(files):
        d = torch.load(os.path.join(sample_dir, f), map_location='cpu')
        re = d['re'].item()
        if abs(re - target) < abs(best_re - target):
            best_re = re
            best_idx = i
    print(f"Using sample_{best_idx:04d}.pt  (Re = {best_re:.2f})")

    sample = torch.load(os.path.join(sample_dir, f"sample_{best_idx:04d}.pt"), map_location=device)
    mu = sample['mu'].reshape(-1)
    a_ref_norm = sample['a_ref']
    b_ref_norm = sample['b_ref']
    a_mean, a_std = sample['a_mean'].item(), sample['a_std'].item()
    b_mean, b_std = sample['b_mean'].item(), sample['b_std'].item()

    with torch.no_grad():
        a_hat, a_NO, b = model(mu, edge_index)
        a_pred = a_NO * a_std + a_mean
        b_pred = b * b_std + b_mean

    a_ref  = a_ref_norm * a_std + a_mean
    b_ref  = b_ref_norm * b_std + b_mean

    grid_size = int(N**0.5)
    u_ref = a_ref[0::2].reshape(grid_size, grid_size).numpy()
    v_ref = a_ref[1::2].reshape(grid_size, grid_size).numpy()
    p_ref = b_ref.reshape(grid_size, grid_size).numpy()
    u_pred = a_pred[0::2].reshape(grid_size, grid_size).numpy()
    v_pred = a_pred[1::2].reshape(grid_size, grid_size).numpy()
    p_pred = b_pred.reshape(grid_size, grid_size).numpy()

    err_u = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-8)
    err_v = np.linalg.norm(v_pred - v_ref) / (np.linalg.norm(v_ref) + 1e-8)
    err_p = np.linalg.norm(p_pred - p_ref) / (np.linalg.norm(p_ref) + 1e-8)

    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    titles = ['Reference (RBF-FD)', 'Neural Operator', 'Absolute Error']
    cmap = 'RdBu_r'

    for i, (u, v, p) in enumerate([(u_ref, v_ref, p_ref), (u_pred, v_pred, p_pred)]):
        im = axs[0, i].contourf(u, levels=50, cmap=cmap)
        axs[0, i].set_title(f'u-velocity — {titles[i]}')
        plt.colorbar(im, ax=axs[0, i], shrink=0.8)

        im = axs[1, i].contourf(v, levels=50, cmap=cmap)
        axs[1, i].set_title(f'v-velocity — {titles[i]}')
        plt.colorbar(im, ax=axs[1, i], shrink=0.8)

        im = axs[2, i].contourf(p, levels=50, cmap='viridis')
        axs[2, i].set_title(f'pressure — {titles[i]}')
        plt.colorbar(im, ax=axs[2, i], shrink=0.8)

    axs[0, 2].contourf(np.abs(u_pred - u_ref), levels=50, cmap='hot')
    axs[0, 2].set_title(f'|u err|  (L₂ rel = {err_u:.1%})')
    axs[1, 2].contourf(np.abs(v_pred - v_ref), levels=50, cmap='hot')
    axs[1, 2].set_title(f'|v err|  (L₂ rel = {err_v:.1%})')
    axs[2, 2].contourf(np.abs(p_pred - p_ref), levels=50, cmap='hot')
    axs[2, 2].set_title(f'|p err|  (L₂ rel = {err_p:.1%})')

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'Figure 6: Field Comparison at Re={best_re:.1f}  (N=225)', fontsize=16)
    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/fig6_field_comparison_Re92.png', dpi=400, bbox_inches='tight')
    print("✅ Figure 6 saved: results/figures/fig6_field_comparison_Re92.png")
    plt.show()

if __name__ == '__main__':
    main()
