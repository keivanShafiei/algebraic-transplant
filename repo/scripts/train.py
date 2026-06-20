"""train.py — Hybrid Manifold Guidance (Dual-Path Loss) — Kaggle Compatible.

Critical Fixes for Kaggle execution:
1. Added sys.path for running without pip install -e .
2. Loss weights read from config using exact keys w_prs and w_div.
3. interior_mask passed to model.set_projection() for boundary-safe projection (Proposition 4).
4. Added interior_node_mask for pressure recovery (Phase 2, Task 1).
5. Replaced print() with structured logging (Phase 2, Task 3).
"""

import os
import sys
from datetime import datetime

# KAGGLE FIX: Add repo root to Python path when running standalone
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

# NEW: Import logging
from src.utils.logging_config import setup_logging, get_logger

from src.gnn.neural_operator import NeuralOperator
from src.rbf_fd.stencils import build_stencils
from src.data.cavity import generate_cavity_points


class PhysicalDataset(Dataset):
    def __init__(self, sample_dir='data/samples'):
        self.files = sorted([
            os.path.join(sample_dir, f)
            for f in os.listdir(sample_dir) if f.endswith('.pt')
        ])
        if not self.files:
            raise FileNotFoundError("No .pt files.")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location='cpu')
        return d['mu'], d['a_ref'], d['b_ref'], d['a_scale'], d['b_scale']


def compute_variance_weights(dataset: PhysicalDataset,
                             device: torch.device,
                             logger: logging.Logger) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute variance weights from dataset."""
    logger.info("Computing variance weights from dataset...")

    all_a_norm = []
    all_b_norm = []
    for i in range(len(dataset)):
        mu, a_ref, b_ref, a_sc, b_sc = dataset[i]
        all_a_norm.append((a_ref / a_sc.item()).unsqueeze(0))
        all_b_norm.append((b_ref / b_sc.item()).unsqueeze(0))

    a_stack = torch.cat(all_a_norm, dim=0)  # (M, 2N)
    b_stack = torch.cat(all_b_norm, dim=0)  # (M, N)

    vel_var = a_stack.var(dim=0)  # (2N,)
    prs_var = b_stack.var(dim=0)  # (N,)

    vel_w = vel_var / (vel_var.mean() + 1e-8)
    prs_w = prs_var / (prs_var.mean() + 1e-8)

    logger.info(
        f"Velocity weights — min: {vel_w.min():.3f}, "
        f"mean: {vel_w.mean():.3f}, max: {vel_w.max():.3f}"
    )
    logger.info(
        f"Important nodes (weight > 1.0): {(vel_w > 1.0).float().mean().item()*100:.1f}%"
    )
    logger.info(
        f"Unimportant nodes (weight < 0.1): {(vel_w < 0.1).float().mean().item()*100:.1f}%"
    )

    return vel_w.to(device), prs_w.to(device)


def variance_weighted_loss(pred: torch.Tensor, target: torch.Tensor,
                           weights: torch.Tensor) -> torch.Tensor:
    """Variance-weighted MSE."""
    sq_err = (pred - target).pow(2)  # (B, F)
    return (sq_err * weights.unsqueeze(0)).mean()


def train():
    # NEW: Setup logging at the start
    logger = setup_logging(
        log_dir="results/logs",
        log_level=logging.INFO,
        experiment_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    logger.info("=" * 60)
    logger.info("RBF-FD GNN Projection — Training Started")
    logger.info("=" * 60)

    config = yaml.safe_load(open('config.yaml'))
    logger.info(f"Loaded config: {config}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    os.makedirs('results', exist_ok=True)

    N = config['n_nodes_list'][0]
    points = generate_cavity_points(N).to(device)
    stencils = build_stencils(points, config['stencil_k']).to(device)
    edge_dst = stencils.reshape(-1)
    edge_src = torch.arange(N, device=device).repeat_interleave(config['stencil_k'])
    edge_index = torch.stack([edge_dst, edge_src])

    model = NeuralOperator(
        n_nodes=N, hidden=config['hidden_dim'], layers=config['gnn_layers']
    ).to(device)
    model.set_points(points, stencils)

    # CRITICAL FIX: Load and pass interior_mask and interior_node_mask to projection
    G = torch.load('data/fixed_G.pt', map_location=device)
    interior_mask = torch.load('data/interior_mask.pt', map_location=device)
    # interior_node_mask is the node-level version (N,)
    # It can be derived from interior_dof_mask by taking every other element
    interior_node_mask = interior_mask[::2]  # u-DOFs correspond to nodes
    model.set_projection(G, interior_mask=interior_mask, interior_node_mask=interior_node_mask)
    logger.info("Projection layer set with interior_mask and interior_node_mask")

    lambda_guidance = 0.1
    lambda_milestone = 150

    dataset = PhysicalDataset()
    loader = DataLoader(dataset, batch_size=config['batch_size'],
                        shuffle=True, drop_last=True)

    vel_w, prs_w = compute_variance_weights(dataset, device, logger)

    lr = config.get('lr', 1e-3)
    film_params = list(model.film_conditioners.parameters())
    film_ids = {id(p) for p in film_params}
    base_params = [p for p in model.parameters() if id(p) not in film_ids]

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': lr},
        {'params': film_params, 'lr': lr * 0.1},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[lr, lr * 0.1],
        steps_per_epoch=len(loader), epochs=config['epochs'],
        pct_start=0.1, div_factor=10.0, final_div_factor=100,
    )

    # CRITICAL FIX: Use exact config keys w_prs and w_div
    w_vel = 1.0
    w_prs = config['w_prs']  # Must exist in config.yaml
    w_div = config['w_div']  # Must exist in config.yaml

    logger.info("Training configuration:")
    logger.info(f"  Architecture: Hybrid Manifold Guidance (Dual-Path)")
    logger.info(f"  Lambda: {lambda_guidance} -> 0.01 at epoch >= {lambda_milestone}")
    logger.info(f"  Dataset: {len(dataset)} | Batch: {config['batch_size']} | Epochs: {config['epochs']}")
    logger.info(f"  Loss weights: w_vel={w_vel} | w_prs={w_prs} | w_div={w_div}")

    best_loss = float('inf')

    for epoch in range(config['epochs']):
        model.train()
        tot = tot_physics = tot_guidance = tot_p = tot_d = 0.0

        current_lambda = lambda_guidance if (epoch + 1) < lambda_milestone else 0.01

        for mu, a_ref, b_ref, a_sc, b_sc in loader:
            mu = mu.to(device).float()
            a_ref = a_ref.to(device).float()
            b_ref = b_ref.to(device).float()
            a_sc = a_sc.to(device).float()
            b_sc = b_sc.to(device).float()

            a_norm = a_ref / a_sc.view(-1, 1)
            b_norm = b_ref / b_sc.view(-1, 1)

            optimizer.zero_grad()
            a_hat, a_NO, b = model(mu, edge_index)

            loss_physics = variance_weighted_loss(a_NO, a_norm, vel_w)
            loss_guidance = variance_weighted_loss(a_hat, a_norm, vel_w)
            loss_p = variance_weighted_loss(b, b_norm, prs_w)

            # NOTE: loss_d is redundant because a_NO is already projected.
            # G @ a_NO ~ 4e-5 (float32 floor). Kept for numerical stability.
            div = torch.einsum('md,bd->bm', G, a_NO)
            loss_d = div.pow(2).mean()

            loss = (loss_physics +
                    current_lambda * loss_guidance +
                    w_prs * loss_p +
                    w_div * loss_d)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            tot += loss.item()
            tot_physics += loss_physics.item()
            tot_guidance += loss_guidance.item()
            tot_p += loss_p.item()
            tot_d += loss_d.item()

        n = len(loader)
        avg = tot / n
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), 'results/model_best.pt')

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch [{epoch+1:4d}/{config['epochs']}] | "
                f"Total: {avg:.4e} | Physics: {tot_physics/n:.4e} | "
                f"Guidance: {tot_guidance/n:.4e} (lambda={current_lambda:.3f}) | "
                f"Prs: {tot_p/n:.4e} | Div: {tot_d/n:.4e} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    torch.save(model.state_dict(), 'results/model_final.pt')
    logger.info(f"Best weighted loss: {best_loss:.4e}")
    logger.info("Training completed successfully")
    logger.info("=" * 60)


if __name__ == '__main__':
    train()
