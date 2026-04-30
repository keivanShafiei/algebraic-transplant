"""train_v8.py — Hybrid Manifold Guidance (Dual-Path Loss) with Multi-GPU & AMP Optimization (FIXED)."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler

from src.gnn.neural_operator import NeuralOperator
from src.rbf_fd.stencils import build_stencils
from src.data.cavity import generate_cavity_points
from src.utils.checkpoint import save_checkpoint


# =========================
# Dataset
# =========================
class PhysicalDataset(Dataset):
    def __init__(self, sample_dir='data/samples'):
        self.files = sorted([
            os.path.join(sample_dir, f)
            for f in os.listdir(sample_dir) if f.endswith('.pt')
        ])
        if not self.files:
            raise FileNotFoundError("No .pt files found.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location='cpu')
        return d['mu'], d['a_ref'], d['b_ref'], d['a_scale'], d['b_scale']


# =========================
# Variance Weights
# =========================
def compute_variance_weights(dataset: PhysicalDataset, device):
    print("  Computing variance weights...")

    a_list, b_list = [], []

    for i in range(len(dataset)):
        mu, a_ref, b_ref, a_sc, b_sc = dataset[i]
        a_list.append(a_ref / a_sc.item())
        b_list.append(b_ref / b_sc.item())

    a_stack = torch.stack(a_list).to(device)  # (M, 2N)
    b_stack = torch.stack(b_list).to(device)  # (M, N)

    vel_var = a_stack.var(dim=0)
    prs_var = b_stack.var(dim=0)

    vel_w = vel_var / (vel_var.mean() + 1e-8)
    prs_w = prs_var / (prs_var.mean() + 1e-8)

    print(f"  Vel weights mean={vel_w.mean():.4f}, max={vel_w.max():.4f}")
    return vel_w, prs_w


# =========================
# Loss
# =========================
def variance_weighted_loss(pred, target, weights):
    err = (pred - target) ** 2
    return (err * weights.unsqueeze(0)).mean()


# =========================
# Train
# =========================
def train():
    config = yaml.safe_load(open('config.yaml'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('results', exist_ok=True)

    use_multi_gpu = torch.cuda.device_count() > 1

    # =========================
    # Geometry
    # =========================
    N = config['n_nodes_list'][0]
    points = generate_cavity_points(N).to(device)
    stencils = build_stencils(points, config['stencil_k']).to(device)

    edge_dst = stencils.reshape(-1)
    edge_src = torch.arange(N, device=device).repeat_interleave(config['stencil_k'])
    edge_index = torch.stack([edge_dst, edge_src])
    n_nodes = N
    in_ch = config.get('in_channels', 3)
    hidden_ch = config.get('hidden_channels', 64)
    out_ch = config.get('out_channels', 3)
    num_layers = config.get('num_layers', 4)
    stencil_k = config.get('stencil_k', 25)
    
    print(f"🔧 Model Config: in={in_ch}, hidden={hidden_ch}, out={out_ch}, layers={num_layers}")

    # ساخت مدل با نام‌گذاری صحیح پارامترها (بر اساس خطای دریافتی)
    # فرض بر این است که کلاس از input_dim, hidden_dim, output_dim استفاده می‌کند
    try:
        base_model = NeuralOperator(
            n_nodes=N,
            d=config.get('in_channels', 3),          # تطبیق با in_channels
            param_dim=1,                             # فرض بر تک‌پارامتری بودن (Re)
            k=config.get('stencil_k', 25),           # تطبیق با stencil_k
            hidden=config.get('hidden_channels', 64),# تطبیق با hidden_channels
            layers=config.get('num_layers', 4),      # تطبیق با num_layers
            eps=1e-06
        ).to(device)
    except Exception as e:
        print(f"⚠️ Error with keyword args: {e}")
        print("💡 Falling back to positional arguments...")
        # فراخوانی موقعیتی در صورت شکست نام‌گذاری
        base_model = NeuralOperator(
            N,                                     # n_nodes
            config.get('in_channels', 3),          # d
            1,                                     # param_dim
            config.get('stencil_k', 25),           # k
            config.get('hidden_channels', 64),     # hidden
            config.get('num_layers', 4),           # layers
            1e-06                                  # eps
        ).to(device)

    # مدیریت Multi-GPU
    if use_multi_gpu:
        model = nn.DataParallel(base_model)
        print(f"✅ Wrapped with DataParallel ({torch.cuda.device_count()} GPUs)")
    else:
        model = base_model

    # تنظیم نقاط و استنسیل‌ها
    if use_multi_gpu:
        model.module.set_points(points, stencils)
    else:
        model.set_points(points, stencils)

    # تنظیم پروژکشن
    G = torch.load('data/fixed_G.pt', map_location=device)
    if use_multi_gpu:
        model.module.set_projection(G)
    else:
        model.set_projection(G)

    (model.module if use_multi_gpu else model).set_projection(G)

    # =========================
    # Data
    # =========================
    dataset = PhysicalDataset()
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )

    vel_w, prs_w = compute_variance_weights(dataset, device)

    # =========================
    # Optimizer
    # =========================
    lr = config.get('lr', 1e-3)

    model_params = model.module.parameters() if use_multi_gpu else model.parameters()

    optimizer = optim.AdamW(model_params, lr=lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(loader),
        epochs=config['epochs']
    )

    # =========================
    # AMP Setup (Mixed Precision)
    # =========================
    # استفاده از API جدید torch.amp برای سازگاری با نسخه‌های جدید
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())
    
    lambda_guidance = 0.1
    lambda_milestone = 150

    print("🚀 Training started (Hybrid Dual-Path)")
    print(f"   Devices: {next(model.parameters()).device}")
    print(f"   Mixed Precision: {'Enabled' if torch.cuda.is_available() else 'Disabled'}")

    best_loss = float('inf')

    # =========================
    # Training Loop
    # =========================
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        current_lambda = 0.01 if epoch >= lambda_milestone else lambda_guidance

        for mu, a_ref, b_ref, a_sc, b_sc in loader:
            # انتقال داده‌ها به GPU
            mu = mu.to(device).float()
            a_ref = a_ref.to(device).float()
            b_ref = b_ref.to(device).float()
            a_sc = a_sc.to(device).float()
            b_sc = b_sc.to(device).float()

            # نرمال‌سازی targets
            a_norm = a_ref / a_sc.view(-1, 1)
            b_norm = b_ref / b_sc.view(-1, 1)

            optimizer.zero_grad()

            # Forward pass با Mixed Precision
            with autocast(device_type='cuda', dtype=torch.float16):
                a_hat, a_no, b = model(mu, edge_index)

                loss_physics = variance_weighted_loss(a_no, a_norm, vel_w)
                loss_guidance = variance_weighted_loss(a_hat, a_norm, vel_w)
                loss_p = variance_weighted_loss(b, b_norm, prs_w)

                loss = loss_physics + current_lambda * loss_guidance + 0.1 * loss_p

            # Backward pass با Scaler
            scaler.scale(loss).backward()
            
            # Unscale و Clip گرادیان‌ها
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # به‌روزرسانی وزن‌ها
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        # =========================
        # Save Best Model
        # =========================
        unwrapped = model.module if use_multi_gpu else model

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model=unwrapped,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=best_loss,
                config=config,
                path='results/model_best_v8.pt',
                metadata={
                    'training_n_nodes': N,
                    'stencil_k': config.get('stencil_k', 25),
                    'lambda_guidance': current_lambda
                }
            )

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4e} | λ: {current_lambda:.3f}")

    print(f"\n✅ Training finished. Best loss: {best_loss:.4e}")


# =========================
# Entry
# =========================
if __name__ == "__main__":
    train()
