"""eval_projection_fixed.py — کاملاً سازگار با مدل جدید"""
import os
import torch
import yaml
from src.gnn.neural_operator import NeuralOperator
from src.data.cavity import generate_cavity_points
from src.rbf_fd.stencils import build_stencils
from src.utils.metrics import divergence_residual

def main(sample_dir: str = 'data/samples',
         model_path: str = 'results/model_final.pt',
         n_test: int = 80):
    config = yaml.safe_load(open('config.yaml'))
    device = torch.device('cpu')

    N = config['n_nodes_list'][0]
    k = config['stencil_k']

    points = generate_cavity_points(N).to(device)
    stencils = build_stencils(points, k).to(device)
    G = torch.load('data/fixed_G.pt', map_location=device)
    interior_mask = torch.load('data/interior_mask.pt', map_location=device)

    edge_dst = stencils.reshape(-1)
    edge_src = torch.arange(N, device=device).repeat_interleave(k)
    edge_index = torch.stack([edge_dst, edge_src])

    model = NeuralOperator(
        n_nodes=N, d=2, param_dim=1, k=k,
        hidden=config['hidden_dim'], layers=config['gnn_layers'],
        eps=float(config['projection_eps']),
    ).to(device)

    model.set_points(points, stencils)

    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.set_projection(G)
    
    model.eval()

    rb, ra, rhos = [], [], []
    files = sorted([os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith('.pt')])[:n_test]

    with torch.no_grad():
        for f in files:
            d = torch.load(f, map_location=device)
            mu = d['mu'].reshape(-1).to(device)

            a_hat, a_NO, _ = model(mu, edge_index)

            r_before = torch.norm(G @ a_hat.reshape(-1)).item()
            r_after = torch.norm(G @ a_NO.reshape(-1)).item()
            rho = r_before / (r_after + 1e-12)

            rb.append(r_before)
            ra.append(r_after)
            rhos.append(rho)

    print("=== Table 8: Projection Layer Efficacy (بعد از فیکس) ===")
    print(f"N_test={len(rb)}, N={N}")
    print(f"r_before : {torch.tensor(rb).mean():.2e} ± {torch.tensor(rb).std():.2e}")
    print(f"r_after  : {torch.tensor(ra).mean():.2e} ± {torch.tensor(ra).std():.2e}")
    print(f"ρ        : {torch.tensor(rhos).mean():.2e} ± {torch.tensor(rhos).std():.2e}")
    print(f"log10(ρ) : {torch.log10(torch.tensor(rhos)).mean():.2f} ± {torch.log10(torch.tensor(rhos)).std():.2f}")

if __name__ == '__main__':
    main()
