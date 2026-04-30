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

        # =========================
    # Model Initialization (FIXED: Force input dim match)
    # =========================
    
    # خواندن یک بچ نمونه برای تشخیص دقیق بعد ورودی
    print("📦 Loading Dataset...")
    dataset = PhysicalDataset()
    
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )
    print(f"   Dataset size: {len(dataset)}, Batch size: {config['batch_size']}")

    sample_mu, _, _, _, _ = next(iter(loader))
    actual_input_dim = sample_mu.shape[1]  # باید 1 باشد
    
    print(f"🔧 Detecting input dimension from data: {actual_input_dim}")
    print(f"   Config in_channels: {config.get('in_channels', 'N/A')}")
    print(f"   -> Forcing model input dim (d) to match data: {actual_input_dim}")

    try:
        base_model = NeuralOperator(
            n_nodes=N,
            d=actual_input_dim,              # ✅ اصلاح قطعی: استفاده مستقیم از بعد داده
            param_dim=1,                     # فرض بر تک‌پارامتری بودن (Re)
            k=config.get('stencil_k', 25),
            hidden=config.get('hidden_channels', 64),
            layers=config.get('num_layers', 4),
            eps=config.get('projection_eps', 1e-06)
        ).to(device)
    except Exception as e:
        print(f"⚠️ Error with keyword args: {e}")
        # فراخوانی موقعیتی به عنوان پشتیبان
        base_model = NeuralOperator(
            N,                   # n_nodes
            actual_input_dim,    # d (ورودی)
            1,                   # param_dim
            config.get('stencil_k', 25), # k
            config.get('hidden_channels', 64), # hidden
            config.get('num_layers', 4), # layers
            config.get('projection_eps', 1e-06) # eps
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
    if os.path.exists('data/fixed_G.pt'):
        G = torch.load('data/fixed_G.pt', map_location=device)
        if use_multi_gpu:
            model.module.set_projection(G)
        else:
            model.set_projection(G)
        print("✅ Projection operator loaded.")
    else:
        print("⚠️ Warning: data/fixed_G.pt not found. Skipping projection setup.")

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
    # =========================
    # Training Loop
    # =========================
    print(f"   Starting training loop for {config['epochs']} epochs...")
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        current_lambda = 0.01 if epoch >= lambda_milestone else lambda_guidance

        # اضافه کردن شمارنده بچ برای دیباگ
        for batch_idx, (mu, a_ref, b_ref, a_sc, b_sc) in enumerate(loader):
            # انتقال داده‌ها به دستگاه اصلی
            mu = mu.to(device).float()       # Shape مورد انتظار: (B, param_dim) یا (B, 1)
            a_ref = a_ref.to(device).float() # Shape: (B, 2N)
            b_ref = b_ref.to(device).float() # Shape: (B, N)
            a_sc = a_sc.to(device).float()   # Shape: (B,)
            b_sc = b_sc.to(device).float()   # Shape: (B,)

            # نرمال‌سازی هدف
            a_norm = a_ref / a_sc.view(-1, 1)
            b_norm = b_ref / b_sc.view(-1, 1)

            optimizer.zero_grad()

            # بررسی ابعاد برای دیباگ (فقط در اولین بچ از اولین اپوک)
            if epoch == 0 and batch_idx == 0:
                print(f"\n🔍 Debug Info (Epoch 0, Batch 0):")
                print(f"   Input mu shape: {mu.shape}")
                print(f"   Target a_norm shape: {a_norm.shape}")
                print(f"   Target b_norm shape: {b_norm.shape}")
                print(f"   edge_index shape: {edge_index.shape}")
                
                # بررسی سازگاری بعد ورودی مدل با داده‌ها
                expected_input_dim = base_model.d
                actual_input_dim = mu.shape[1] if len(mu.shape) > 1 else 1
                
                if actual_input_dim != expected_input_dim:
                    print(f"   ⚠️ WARNING: Input dimension mismatch!")
                    print(f"      Model expects 'd' (input_dim) = {expected_input_dim}")
                    print(f"      Data provides dimension = {actual_input_dim}")
                    print(f"      Fix: Adjust config['in_channels'] or check data generation.")
                    # اگر ناسازگاری جدی است، می‌توانیم اینجا خطا بگیریم یا ادامه دهیم تا خطای اصلی پایتورچ بیاید
                    # raise ValueError(f"Input dim mismatch: Model={expected_input_dim}, Data={actual_input_dim}")

            with autocast('cuda', dtype=torch.float16):
                try:
                    # Forward pass
                    # نکته: در DataParallel، mu بین GPUها تقسیم می‌شود. edge_index ثابت است.
                    a_hat, a_no, b = model(mu, edge_index)
                    
                except RuntimeError as e:
                    error_msg = str(e)
                    if "mat1 and mat2" in error_msg or "shape" in error_msg.lower():
                        print(f"\n❌ Dimension Mismatch Detected during Forward Pass!")
                        print(f"   Error: {e}")
                        print(f"   Mu shape: {mu.shape}, Edge Index: {edge_index.shape}")
                        print(f"   Please check if 'd' in NeuralOperator matches mu.shape[1]")
                        raise e
                    else:
                        raise e

                # محاسبه ضرر
                loss_physics = variance_weighted_loss(a_no, a_norm, vel_w)
                loss_guidance = variance_weighted_loss(a_hat, a_norm, vel_w)
                loss_p = variance_weighted_loss(b, b_norm, prs_w)

                loss = loss_physics + current_lambda * loss_guidance + 0.1 * loss_p

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
