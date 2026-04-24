import os
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from src.data.cavity import generate_cavity_points
from src.gnn.neural_operator import NeuralOperator
from src.rbf_fd.stencils import build_stencils

def _closest_re_sample(sample_dir: Path, target: float = 92.0) -> Path:
    best_path, best_gap = None, float("inf")
    for p in sorted(sample_dir.glob("*.pt")):
        d = torch.load(p, map_location="cpu")
        re = float(d.get("re", float("nan")))
        if abs(re - target) < best_gap:
            best_gap = abs(re - target)
            best_path = p
    return best_path

def main():
    config = yaml.safe_load(open("config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_path = _closest_re_sample(Path("data/samples"), target=92.0)
    sample = torch.load(sample_path, map_location=device)

    N = int(sample["a_ref"].numel() // 2)
    k = int(config["stencil_k"])

    model = NeuralOperator(
        n_nodes=N, d=2, k=k,
        hidden=config["hidden_dim"], layers=config["gnn_layers"], eps=float(config["projection_eps"])
    ).to(device)
    
    # Load correct v8 model
    model.load_state_dict(torch.load("results/model_final.pt", map_location=device), strict=False)
    model.eval()

    points = sample.get("points", generate_cavity_points(N)).to(device)
    stencils = sample.get("stencils", build_stencils(points, k)).to(device)
    model.set_points(points, stencils)
    
    G = torch.load('data/fixed_G.pt', map_location=device)
    model.set_projection(G)

    edge_dst = stencils.reshape(-1)
    edge_src = torch.arange(N, device=device).repeat_interleave(k)
    edge_index = torch.stack([edge_dst, edge_src])

    mu = sample["mu"].reshape(-1).to(device)
    b_ref = sample["b_ref"].to(device)
    b_scale = float(sample.get("b_scale", 1.0))

    with torch.no_grad():
        a_hat, a_NO, b_pred_norm = model(mu, edge_index)
        _, q = model.projection(a_hat[0], return_q=True)

    b_pred = b_pred_norm.squeeze(0) * b_scale
    q_full = q.reshape(-1)
    b_corrected = b_pred + q_full

    rel_l2 = torch.linalg.norm(b_corrected - b_ref).item() / (torch.linalg.norm(b_ref).item() + 1e-12)
    
    print(f"Sample: {sample_path.name} (Re={float(sample['re']):.2f})")
    print(f"||q||_2: {torch.linalg.norm(q).item():.3e}")
    print(f"Pressure relative L2 error: {rel_l2:.4f}")

    gs = int(round(N ** 0.5))
    b_ref_2d = b_ref.reshape(gs, gs).detach().cpu().numpy()
    b_pred_2d = b_corrected.reshape(gs, gs).detach().cpu().numpy()
    err_2d = np.abs(b_pred_2d - b_ref_2d)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax in axs: ax.set_xticks([]); ax.set_yticks([])

    c0 = axs[0].contourf(b_ref_2d, levels=60, cmap="viridis")
    axs[0].set_title("Ground Truth Pressure Field")
    plt.colorbar(c0, ax=axs[0], shrink=0.85)

    c1 = axs[1].contourf(b_pred_2d, levels=60, cmap="viridis")
    axs[1].set_title("GNN Predicted Pressure Field")
    plt.colorbar(c1, ax=axs[1], shrink=0.85)

    c2 = axs[2].contourf(err_2d, levels=60, cmap="hot")
    axs[2].set_title(f"Absolute Error | Rel L2 = {rel_l2:.3f}")
    plt.colorbar(c2, ax=axs[2], shrink=0.85)

    import os
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/pressure_recovery.png", dpi=400, bbox_inches="tight")
    print("✅ Saved figure: results/figures/pressure_recovery.png")

if __name__ == "__main__":
    main()
