"""
diagnostic_v3.py — ارزیابی کامل با Baseline Comparison

سوال اصلی: آیا model بهتر از "predict mean" است؟
اگر نه، مشکل از معماری است. اگر بله، مشکل از metric است.
"""

import torch
import os
import math
import numpy as np
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.gnn.neural_operator import NeuralOperator
from src.rbf_fd.stencils import build_stencils
from src.data.cavity import generate_cavity_points


def run_diagnostics():
    config = yaml.safe_load(open('config.yaml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N        = config['n_nodes_list'][0]
    points   = generate_cavity_points(N).to(device)
    stencils = build_stencils(points, config['stencil_k']).to(device)
    edge_dst   = stencils.reshape(-1)
    edge_src   = torch.arange(N, device=device).repeat_interleave(config['stencil_k'])
    edge_index = torch.stack([edge_dst, edge_src])

    sample_dir = 'data/samples'
    files = sorted([os.path.join(sample_dir, f)
                    for f in os.listdir(sample_dir) if f.endswith('.pt')])

    # ---------------------------------------------------------------
    # TEST 0: Data Consistency
    # ---------------------------------------------------------------
    print("=" * 60)
    print("TEST 0: سازگاری داده با G")
    print("=" * 60)
    G = None
    if os.path.exists('data/G_int.pt'):
        G = torch.load('data/G_int.pt', map_location='cpu')
        divs = []
        for f in files[:10]:
            d    = torch.load(f, map_location='cpu')
            div  = (G @ d['a_ref'].float()).norm().item()
            divs.append(div / (d['a_ref'].norm().item() + 1e-8))
        mean_div = np.mean(divs)
        print(f"  Mean ||G a_ref||/||a_ref|| = {mean_div:.4e}")
        print(f"  {'✅ Physical & consistent' if mean_div < 1e-3 else '❌ INCONSISTENT'}")
    else:
        print("  ❌ data/G_int.pt not found")

    # ---------------------------------------------------------------
    # BASELINE: Zero Predictor و Mean Predictor
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE: مقایسه با predictor های ساده")
    print("=" * 60)

    all_a_norm = []
    re_list    = []
    for f in files:
        d = torch.load(f, map_location='cpu')
        a_sc = d['a_scale'].item()
        all_a_norm.append(d['a_ref'].float() / a_sc)
        re_list.append(d['mu'].item() * 100.0)

    all_a_stack = torch.stack(all_a_norm, dim=0)   # (M, 2N)
    mean_field  = all_a_stack.mean(dim=0)          # (2N,) — mean predictor

    zero_loss = all_a_stack.pow(2).mean().item()
    mean_loss = (all_a_stack - mean_field.unsqueeze(0)).pow(2).mean().item()

    print(f"  Zero Predictor MSE : {zero_loss:.4e}  ← اگر مدل از این بدتر باشد، فاجعه است")
    print(f"  Mean Predictor MSE : {mean_loss:.4e}  ← اگر مدل از این بهتر باشد، چیزی یاد گرفته")
    print(f"  Signal Variance    : {all_a_stack.var().item():.4e}")
    print(f"  Std across Re      : {all_a_stack.std(dim=0).mean().item():.4e}  ← تنوع across Re")

    # ---------------------------------------------------------------
    # Load Model
    # ---------------------------------------------------------------
    ckpt_candidates = [
        'results/model_best_v5.pt',
        'results/model_final_v5.pt',
        'results/model_best_v4.pt',
        'results/model_best_v3.pt',
        'results/model_best.pt',
    ]
    ckpt = next((c for c in ckpt_candidates if os.path.exists(c)), None)

    if ckpt is None:
        print("\n❌ No checkpoint found. Checked:")
        for c in ckpt_candidates:
            print(f"   {c}")
        print("\nاما Baseline ها بالا نشان می‌دهند آیا مشکل از data است.")
        return

    print(f"\nCheckpoint: {ckpt}")
    model = NeuralOperator(
        n_nodes=N, hidden=config['hidden_dim'], layers=config['gnn_layers']
    ).to(device)
    model.set_points(points)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # ---------------------------------------------------------------
    # MODEL EVALUATION vs BASELINES
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL vs BASELINES")
    print("=" * 60)

    model_losses = []
    re_errors    = {}   # Re → list of errors

    with torch.no_grad():
        for f in files:
            d      = torch.load(f, map_location='cpu')
            re     = d['mu'].item() * 100.0
            a_ref  = d['a_ref'].float()
            a_sc   = d['a_scale'].item()
            a_norm = a_ref / a_sc

            mu = d['mu'].unsqueeze(0).to(device)
            _, a_NO, _ = model(mu, edge_index, inference=True)
            a_NO_cpu   = a_NO[0].cpu()

            err = (a_NO_cpu - a_norm).pow(2).mean().item()
            model_losses.append(err)

            bin_re = round(re / 10) * 10   # nearest 10
            re_errors.setdefault(bin_re, []).append(err)

    model_mse = np.mean(model_losses)
    print(f"\n  Model MSE       : {model_mse:.4e}")
    print(f"  Zero Pred MSE   : {zero_loss:.4e}")
    print(f"  Mean Pred MSE   : {mean_loss:.4e}")

    improvement_over_zero = (zero_loss - model_mse) / zero_loss * 100
    improvement_over_mean = (mean_loss - model_mse) / mean_loss * 100
    print(f"\n  بهبود نسبت به Zero : {improvement_over_zero:+.1f}%")
    print(f"  بهبود نسبت به Mean : {improvement_over_mean:+.1f}%")

    if model_mse > zero_loss * 0.99:
        print("\n  ❌ مدل از Zero Predictor بهتر نیست — Mean Predictor Collapse")
    elif model_mse > mean_loss * 0.99:
        print("\n  ⚠️  مدل فقط Mean Predictor است — Re را یاد نگرفته")
    else:
        print("\n  ✅ مدل از Mean Predictor بهتر است — Re sensitivity واقعی")

    # ---------------------------------------------------------------
    # PER-Re ERROR
    # ---------------------------------------------------------------
    print("\n  خطا به ازای هر Re:")
    for re_bin in sorted(re_errors.keys()):
        errs = re_errors[re_bin]
        print(f"    Re≈{re_bin:3d}: MSE={np.mean(errs):.4e} (n={len(errs)})")

    # ---------------------------------------------------------------
    # DATA VARIANCE CHECK
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DATA VARIANCE: آیا داده‌ها به Re وابسته‌اند؟")
    print("=" * 60)
    re_arr  = np.array(re_list)
    # STD of a_norm across Re for each node — if small, fields are nearly identical
    per_node_std = all_a_stack.std(dim=0)   # (2N,)
    print(f"  Mean per-node std across Re : {per_node_std.mean():.4e}")
    print(f"  Max  per-node std across Re : {per_node_std.max():.4e}")
    print(f"  % nodes with std > 0.05     : "
          f"{(per_node_std > 0.05).float().mean().item()*100:.1f}%")
    print(f"\n  اگر mean std < 0.05 باشد، داده‌ها تقریباً Re-independent هستند")
    print(f"  و مدل نمی‌تواند از mean predictor بهتر شود.")


if __name__ == '__main__':
    run_diagnostics()
