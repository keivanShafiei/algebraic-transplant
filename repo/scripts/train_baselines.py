"""Train all baselines for Figure 9 (Constraint Enforcement Bar Chart)."""
import os
import torch
import yaml
import json
import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import PrecomputedDataset          # ← حالا کار می‌کند
from src.rbf_fd.operators import assemble_divergence_operator
from src.data.cavity import generate_cavity_points
from src.rbf_fd.stencils import build_stencils
from src.utils.metrics import divergence_residual

# baselines
from src.baselines.fno import FourierNeuralOperator
from src.baselines.deeponet import DeepONet
from src.baselines.pod_rbf import PODRBF

config = yaml.safe_load(open('config.yaml'))
device = torch.device('cpu')
N = config['n_nodes_list'][0]

# ── Build edge_index once (for GNN) ─────────────────────────────────────
points   = generate_cavity_points(N).to(device)
stencils = build_stencils(points, config['stencil_k'])
edge_dst = stencils.reshape(-1)
edge_src = torch.arange(N, device=device).repeat_interleave(config['stencil_k'])
edge_index = torch.stack([edge_dst, edge_src]).to(device)

# ── G for divergence check ───────────────────────────────────────────────
G = torch.load('data/fixed_G.pt', map_location=device)

# ── Dataset ──────────────────────────────────────────────────────────────
dataset = PrecomputedDataset()
loader  = DataLoader(dataset, batch_size=1, shuffle=False)

print(f"Training baselines on {len(dataset)} samples...")

baselines = {
    "Hard GNN (proposed)": None,           # will load the real model with projection
    "Soft GNN (no proj.)": None,
    "FNO": FourierNeuralOperator(),
    "DeepONet": DeepONet(),
    "POD-RBF": PODRBF(n_components=30),
}

stats = {}

for name, model in baselines.items():
    print(f"\n→ Training {name} ...")
    divs = []

    if name == "POD-RBF":
        mus, a_refs = [], []
        for mu, a_ref, *_ in loader:
            mus.append(mu)
            a_refs.append(a_ref)
        model.fit(torch.cat(mus), torch.cat(a_refs))
        # evaluate
        for mu, a_ref, *_ in loader:
            a_pred = model.predict(mu)
            divs.append(divergence_residual(G, a_pred.reshape(-1)))

    elif name in ["Hard GNN (proposed)", "Soft GNN (no proj.)"]:
        # Load the trained Algebraic Transplant model
        from src.gnn.neural_operator import NeuralOperator
        model = NeuralOperator(
            n_nodes=N, d=2, param_dim=1, k=config['stencil_k'],
            hidden=config['hidden_dim'], layers=config['gnn_layers']
        )
        model.to(device)
        model.set_points(points, stencils)
        model.load_state_dict(torch.load('results/model_final.pt', map_location=device), strict=False)
        model.eval()
        if name == "Hard GNN (proposed)":
            model.set_projection(G)          # ← hard projection

        with torch.no_grad():
            for mu, *_ in loader:
                mu = mu.to(device).reshape(-1)
                a_hat, a_NO, _ = model(mu, edge_index)
                a_pred = a_NO if name == "Hard GNN (proposed)" else a_hat
                divs.append(divergence_residual(G, a_pred.reshape(-1)))

    else:
        # FNO / DeepONet
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(80):                     # کافی برای baseline
            for mu, a_ref, *_ in loader:
                mu = mu.to(device).reshape(-1)
                a_ref = a_ref.to(device)
                a_hat, _ = model(mu, points=points)
                loss = torch.nn.functional.mse_loss(a_hat, a_ref.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # evaluate divergence
        with torch.no_grad():
            for mu, *_ in loader:
                mu = mu.to(device).reshape(-1)
                a_hat, _ = model(mu, points=points)
                divs.append(divergence_residual(G, a_hat.reshape(-1)))

    stats[name] = {
        "mean": float(np.mean(divs)),
        "std":  float(np.std(divs)),
        "values": [float(x) for x in divs]
    }

# ── Save results ─────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)
with open('results/baseline_div_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("\n✅ All baselines finished!")
print("   Stats saved → results/baseline_div_stats.json")
for name, s in stats.items():
    print(f"   {name:20} → ε_div = {s['mean']:.2e} ± {s['std']:.2e}")
