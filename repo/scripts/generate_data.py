"""
generate_data_fixed.py — نسخه تصحیح‌شده
=========================================

تغییر اصلی نسبت به نسخه اصلی:
  - داده‌ها در فضای فیزیکی (بدون نرمال‌سازی) ذخیره می‌شوند
  - مدل باید با مقادیر فیزیکی کار کند تا G-consistency حفظ شود
  - نرمال‌سازی فقط برای پایداری عددی در ورودی mu (Re/Re_max) نگه داشته شده

ریشه مشکل:
  نسخه قبلی a_ref را نرمال می‌کرد اما G را بدون تغییر نگه می‌داشت.
  از آنجا که G عملگر خطی است:
    G @ a_ref_norm = G @ (a_ref - μ)/σ = (G @ a_ref)/σ - μ/σ · (G @ 1_vec)
  این مقدار صفر نیست حتی اگر G @ a_ref = 0 باشد.
  لایه projection در training تلاش می‌کند â را به ker(G_physical) ببرد،
  در حالی که loss gradient آن را به سمت a_ref_norm می‌کشد → تضاد → loss گیر می‌کند.
"""

import os
import sys
import argparse
import time
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cavity import generate_cavity_points
from src.rbf_fd.stencils import build_stencils
from src.rbf_fd.operators import assemble_divergence_operator
from src.rbf_fd.solver import NavierStokesSolver


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n',       type=int,   default=225)
    p.add_argument('--ns',      type=int,   default=400)
    p.add_argument('--re-min',  type=float, default=10.0)
    p.add_argument('--re-max',  type=float, default=100.0)
    p.add_argument('--tau-mom', type=float, default=1e-2)
    p.add_argument('--n-max',   type=int,   default=300)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--idx-start', type=int, default=0)
    p.add_argument('--device', type=str, default='cpu')
    return p.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(open('config.yaml'))
    device = torch.device(args.device)

    os.makedirs('data/samples', exist_ok=True)

    N = args.n
    points = generate_cavity_points(N).to(device)
    print(f"Nodes N={N}, generating Ns={args.ns} samples (physical space, no normalization)")
    print(f"Re ∈ [{args.re_min}, {args.re_max}]")

    t0 = time.time()
    solver = NavierStokesSolver(points, k=config['stencil_k'],
                                eps=float(config['projection_eps']))
    print(f"Solver precomputed in {time.time()-t0:.1f}s")

    # Save G (interior rows, all cols) — consistent with eval metric
    torch.save(solver.G, 'data/G.pt')
    # Save G_full for projection layer
    torch.save(solver.G, 'data/fixed_G.pt')
    # Save interior mask for eval
    torch.save(solver.interior_dof_mask, 'data/interior_mask.pt')
    print("Saved data/fixed_G.pt, data/G.pt, data/interior_mask.pt")

    # Compute global scale stats from a few samples for stable training
    # (scale input/output but keep zero-mean = physical mean, not zero)
    re_values = torch.linspace(args.re_min, args.re_max, args.ns, dtype=torch.float32)

    accepted = 0
    rejected = 0
    t_start = time.time()

    for idx, re in enumerate(re_values):
        re_val = re.item()

        try:
            a_ref, b_ref = solver.solve(
                Re=re_val,
                tau_mom=args.tau_mom,
                n_max=args.n_max,
                verbose=args.verbose,
            )
        except Exception as exc:
            print(f"  [SKIP] Re={re_val:.1f}: solver failed ({exc})")
            rejected += 1
            continue

        # FIX: use G (interior rows) for filtering, NOT G_full
        div_res = (solver.G_int @ a_ref).norm().item()
        if div_res > 5e-3:
            print(f"  [SKIP] Re={re_val:.1f}: div_res={div_res:.2e} > 5e-3")
            rejected += 1
            continue

        # FIX: store PHYSICAL (un-normalized) coefficients
        # The projection layer G @ a = 0 is defined in physical space
        # Training loss will be computed in physical space
        # We also store scale stats for the training script to use if desired
        a_scale = a_ref.abs().max().item() + 1e-8  # for bounded loss
        b_scale = b_ref.abs().max().item() + 1e-8

        sample = {
            'mu':     torch.tensor([re_val / args.re_max], dtype=torch.float32),
            're':     re,
            # Physical fields (un-normalized) — used for loss and projection
            'a_ref':  a_ref.cpu().float(),
            'b_ref':  b_ref.cpu().float(),
            # Scale info (for optional normalization in training, apply AFTER projection)
            'a_scale': torch.tensor([a_scale], dtype=torch.float32),
            'b_scale': torch.tensor([b_scale], dtype=torch.float32),
        }
        torch.save(sample, f'data/samples/sample_{idx + args.idx_start:04d}.pt')
        accepted += 1

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            eta = (args.ns - idx - 1) / rate
            print(f"  [{idx+1:4d}/{args.ns}] Re={re_val:6.1f} "
                  f"div={div_res:.2e}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    total = time.time() - t_start
    print(f"\nDataset complete. Accepted: {accepted}/{args.ns}  ({100*accepted/args.ns:.1f}%)")
    print(f"Total time: {total:.1f}s  ({total/max(accepted,1):.2f}s/sample)")


if __name__ == '__main__':
    main()
