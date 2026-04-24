import argparse
import math
from pathlib import Path
import torch
import yaml
from src.data.cavity import generate_cavity_points
from src.gnn.neural_operator import NeuralOperator
from src.rbf_fd.stencils import build_stencils
from src.rbf_fd.operators import assemble_divergence_operator

def compute_h_avg(points: torch.Tensor, stencils: torch.Tensor) -> torch.Tensor:
    neigh = points[stencils.long()]
    center = points.unsqueeze(1).expand_as(neigh)
    return torch.norm(neigh - center, dim=-1).mean()

def l2_rel(pred: torch.Tensor, ref: torch.Tensor) -> float:
    return (torch.norm(pred - ref) / (torch.norm(ref) + 1e-12)).item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="results/model_final.pt") # استفاده از وزن‌های درست
    parser.add_argument("--coarse_sample", type=str, required=True)
    parser.add_argument("--fine_sample", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(open(args.config))

    coarse = torch.load(Path(args.coarse_sample), map_location=device)
    fine = torch.load(Path(args.fine_sample), map_location=device)

    N_train = int(coarse.get("N", coarse["a_ref"].numel() // 2))
    N_test = int(fine.get("N", fine["a_ref"].numel() // 2))
    k = int(config["stencil_k"])

    model = NeuralOperator(
        n_nodes=N_test, d=2, k=k,
        hidden=config["hidden_dim"], layers=config["gnn_layers"], eps=float(config["projection_eps"])
    ).to(device)
    
    model.load_state_dict(torch.load(args.model, map_location=device), strict=False)
    model.eval()

    points_train = coarse.get("points", generate_cavity_points(N_train)).to(device)
    stencils_train = coarse.get("stencils", build_stencils(points_train, k)).to(device)
    points_test = fine.get("points", generate_cavity_points(N_test)).to(device)
    stencils_test = fine.get("stencils", build_stencils(points_test, k)).to(device)

    h_train = compute_h_avg(points_train, stencils_train)
    h_infer = compute_h_avg(points_test, stencils_test)

    c_test = 1.2 * float(h_infer.item())
    G_test = assemble_divergence_operator(points_test, stencils_test, c_test, sparse=True)

    model.set_points(points_test, stencils_test)
    model.set_projection(G_test)

    edge_dst = stencils_test.reshape(-1)
    edge_src = torch.arange(N_test, device=device).repeat_interleave(k)
    edge_index = torch.stack([edge_dst, edge_src])

    mu = fine["mu"].reshape(-1).to(device)
    a_ref = fine["a_ref"].to(device)
    b_ref = fine["b_ref"].to(device)
    a_scale = float(fine.get("a_scale", 1.0))
    b_scale = float(fine.get("b_scale", 1.0))

    # ارزیابی بدون اعمال Scale
    model.set_scales(h_train=h_train, h_infer=h_train)
    with torch.no_grad():
        _, a_NO0, b0 = model(mu, edge_index)
    err_u0 = l2_rel(a_NO0.squeeze(0) * a_scale, a_ref)
    
    # ارزیابی با اعمال Scale
    model.set_scales(h_train=h_train, h_infer=h_infer)
    with torch.no_grad():
        _, a_NO1, b1 = model(mu, edge_index)
    err_u1 = l2_rel(a_NO1.squeeze(0) * a_scale, a_ref)

    print("=== Zero-shot resolution transfer ===")
    print(f"h_train = {h_train.item():.6e}")
    print(f"h_infer = {h_infer.item():.6e}")
    print(f"edge_scale = {(h_train / h_infer).item():.6f}")
    print(f"velocity L2 (no scale)  = {err_u0:.4f}")
    print(f"velocity L2 (scaled)    = {err_u1:.4f}")

if __name__ == "__main__":
    main()
