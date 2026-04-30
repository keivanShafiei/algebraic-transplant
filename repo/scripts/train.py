"""train_v8.py — Hybrid Manifold Guidance (Dual-Path Loss)"""

import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

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
                              device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    یک بار قبل از آموزش، واریانس هر نود را در کل dataset محاسبه می‌کند.

    Returns:
        vel_w : (2N,) — وزن هر velocity component، نرمال‌شده به mean=1
        prs_w : (N,)  — وزن هر pressure component
    """
    print("  محاسبه‌ی Variance Weights از dataset...")

    all_a_norm = []
    all_b_norm = []
    for i in range(len(dataset)):
        mu, a_ref, b_ref, a_sc, b_sc = dataset[i]
        all_a_norm.append((a_ref / a_sc.item()).unsqueeze(0))
        all_b_norm.append((b_ref / b_sc.item()).unsqueeze(0))

    a_stack = torch.cat(all_a_norm, dim=0)   # (M, 2N)
    b_stack = torch.cat(all_b_norm, dim=0)   # (M, N)

    # واریانس هر نود در طول dataset
    vel_var = a_stack.var(dim=0)   # (2N,)
    prs_var = b_stack.var(dim=0)   # (N,)

    # نرمال‌سازی: mean weight = 1 (برای حفظ مقیاس loss)
    vel_w = vel_var / (vel_var.mean() + 1e-8)
    prs_w = prs_var / (prs_var.mean() + 1e-8)

    # آماره‌های تشخیصی
    print(f"  Vel weights: min={vel_w.min():.3f} | "
          f"mean={vel_w.mean():.3f} | max={vel_w.max():.3f}")
    print(f"  % نودها با weight > 1.0 (مهم): "
          f"{(vel_w > 1.0).float().mean().item()*100:.1f}%")
    print(f"  % نودها با weight < 0.1 (بی‌اهمیت): "
          f"{(vel_w < 0.1).float().mean().item()*100:.1f}%\n")

    return vel_w.to(device), prs_w.to(device)


def variance_weighted_loss(pred: torch.Tensor, target: torch.Tensor,
                            weights: torch.Tensor) -> torch.Tensor:
    """
    MSE وزن‌دار بر اساس واریانس هر نود.

    pred, target : (B, F)
    weights      : (F,)    — وزن هر feature/node
    """
    sq_err = (pred - target).pow(2)          # (B, F)
    return (sq_err * weights.unsqueeze(0)).mean()


def train():
    config = yaml.safe_load(open('config.yaml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('results', exist_ok=True)

    N        = config['n_nodes_list'][0]
    points   = generate_cavity_points(N).to(device)
    stencils = build_stencils(points, config['stencil_k']).to(device)
    edge_dst   = stencils.reshape(-1)
    edge_src   = torch.arange(N, device=device).repeat_interleave(config['stencil_k'])
    edge_index = torch.stack([edge_dst, edge_src])

    model = NeuralOperator(
        n_nodes=N, hidden=config['hidden_dim'], layers=config['gnn_layers']
    ).to(device)
    model.set_points(points, stencils)

    G = torch.load('data/fixed_G.pt', map_location=device)
    model.set_projection(G)

    # === Hybrid Manifold Guidance hyperparameters (قبل از هر پرینت) ===
    lambda_guidance = 0.1          # مقدار اولیه (قوی)
    lambda_milestone = 150         # از اپوک ۱۵۰ به بعد → ۰٫۰۱ (پروجکشن سخت غالب شود)

    dataset = PhysicalDataset()
    loader  = DataLoader(dataset, batch_size=config['batch_size'],
                         shuffle=True, drop_last=True)

    vel_w, prs_w = compute_variance_weights(dataset, device)

    lr = config.get('lr', 1e-3)
    film_params = list(model.film_conditioners.parameters())
    film_ids    = {id(p) for p in film_params}
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

    w_vel, w_prs, w_div = 1.0, config.get('w_pressure', 0.1), config.get('w_divergence', 0.01)

    print(f"train_v8 | Hybrid Manifold Guidance (Dual-Path)")
    print(f"lambda_guidance start={lambda_guidance} → final=0.01 at epoch >= {lambda_milestone}")
    print(f"Dataset={len(dataset)} | Batch={config['batch_size']} | Epochs={config['epochs']}")
    print(f"w_vel={w_vel} | w_prs={w_prs} | w_div={w_div}\n")

    best_loss = float('inf')

    for epoch in range(config['epochs']):
        model.train()
        tot = tot_physics = tot_guidance = tot_p = tot_d = 0.0

        # کاهش lambda_guidance در اپوک‌های آخر
        current_lambda = lambda_guidance if (epoch + 1) < lambda_milestone else 0.01

        for mu, a_ref, b_ref, a_sc, b_sc in loader:
            mu     = mu.to(device).float()
            a_ref  = a_ref.to(device).float()
            b_ref  = b_ref.to(device).float()
            a_sc   = a_sc.to(device).float()
            b_sc   = b_sc.to(device).float()

            a_norm = a_ref / a_sc.view(-1, 1)
            b_norm = b_ref / b_sc.view(-1, 1)

            optimizer.zero_grad()
            # Hybrid forward: همیشه a_hat_raw و a_NO_projected
            a_hat, a_NO, b = model(mu, edge_index)

            loss_physics  = variance_weighted_loss(a_NO,  a_norm, vel_w)
            loss_guidance = variance_weighted_loss(a_hat, a_norm, vel_w)
            loss_p        = variance_weighted_loss(b,     b_norm, prs_w)
            div           = torch.einsum('md,bd->bm', G, a_NO)
            loss_d        = div.pow(2).mean()

            loss = (loss_physics +
                    current_lambda * loss_guidance +
                    w_prs * loss_p +
                    w_div * loss_d)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            tot += loss.item()
            tot_physics  += loss_physics.item()
            tot_guidance += loss_guidance.item()
            tot_p += loss_p.item()
            tot_d += loss_d.item()

        n   = len(loader)
        avg = tot / n
        if avg < best_loss:
            best_loss = avg
            # Save with new resolution-aware checkpoint format
            from src.utils.checkpoint import save_checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=best_loss,
                config=config,
                path='results/model_best_v8.pt',
                metadata={
                    'training_n_nodes': N,
                    'training_h_avg': model.h_infer.item(),
                    'stencil_k': config['stencil_k'],
                    'projection_eps': config['projection_eps'],
                }
            )
            # Also save legacy format for backward compatibility
            torch.save(model.state_dict(), 'results/model_best.pt')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:4d}/{config['epochs']}] "
                  f"| Total: {avg:.4e} | Physics: {tot_physics/n:.4e} "
                  f"| Guidance: {tot_guidance/n:.4e} (λ={current_lambda:.3f}) "
                  f"| Prs: {tot_p/n:.4e} | Div: {tot_d/n:.4e} "
                  f"| LR: {optimizer.param_groups[0]['lr']:.2e}")

        if (epoch + 1) % 50 == 0:
            _film_check(model, edge_index, device)
            _baseline_check(dataset, vel_w, device)

    # Save final checkpoint with resolution metadata
    from src.utils.checkpoint import save_checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=config['epochs'],
        loss=best_loss,
        config=config,
        path='results/model_final_v8.pt',
        metadata={
            'training_n_nodes': N,
            'training_h_avg': model.h_infer.item(),
            'stencil_k': config['stencil_k'],
            'projection_eps': config['projection_eps'],
        }
    )
    torch.save(model.state_dict(), 'results/model_final.pt')   # نسخه‌ی مخصوص این آزمایش
    print(f"\nBest weighted loss: {best_loss:.4e} (با Hybrid Guidance)")
    print("✅ آموزش با Dual-Path Loss تمام شد")
    print("سپس: python scripts/diagnostic.py")


def _baseline_check(dataset, vel_w, device):
    """مقایسه با zero و mean predictor با همان وزن‌ها"""
    all_a_norm = []
    for i in range(len(dataset)):
        _, a_ref, _, a_sc, _ = dataset[i]
        all_a_norm.append((a_ref / a_sc.item()))
    stack = torch.stack(all_a_norm, dim=0).to(device)   # (M, 2N)
    mean_f = stack.mean(dim=0)   # (2N,)

    zero_wloss = ((stack) * vel_w.unsqueeze(0)).pow(2).mean().item()
    mean_wloss = ((stack - mean_f) * vel_w.unsqueeze(0)).pow(2).mean().item()
    print(f"  [Baseline] Zero={zero_wloss:.4e} | Mean={mean_wloss:.4e} "
          f"(با variance weights)")


def _film_check(model, edge_index, device):
    model.eval()
    with torch.no_grad():
        lo = torch.tensor([0.10], device=device)
        hi = torch.tensor([1.00], device=device)
        _, a_lo, _ = model(lo, edge_index, inference=True)
        _, a_hi, _ = model(hi, edge_index, inference=True)
        cos = F.cosine_similarity(a_lo, a_hi).item()
        rel = (a_lo - a_hi).norm().item() / (a_lo.norm() + 1e-8).item()
        print(f"\n  FiLM: cos={cos:.4f} | rel_diff={rel:.4e} "
              f"{'✅' if cos < 0.99 else '❌ COLLAPSED'}")
    model.train()


if __name__ == '__main__':
    train()
