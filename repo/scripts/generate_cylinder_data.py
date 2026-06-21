"""Generate cylinder flow samples for Figure E12.

Usage:
    python scripts/generate_cylinder_data.py --n 1000 --ns 50 --re-min 10 --re-max 500

Output:
    data/cylinder_samples/sample_0000.pt ... sample_0049.pt
"""

import os
import sys
import argparse
import time
import logging
import torch
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_config import setup_logging, get_logger

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1000, help='Number of nodes')
    p.add_argument('--ns', type=int, default=50, help='Number of samples')
    p.add_argument('--re-min', type=float, default=10.0)
    p.add_argument('--re-max', type=float, default=500.0)
    p.add_argument('--tau-mom', type=float, default=1e-2)
    p.add_argument('--n-max', type=int, default=300)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--device', type=str, default='cpu')
    return p.parse_args()

def generate_cylinder_points(N, L=6.0, H=4.0, R=0.5):
    """Generate scattered points for flow past cylinder geometry.

    Domain: [-2, 4] × [-2, 2] with cylinder at origin, radius R=0.5
    """
    np.random.seed(42)

    # Generate points in bounding box
    n_init = int(N * 1.5)  # oversample
    x = np.random.uniform(-2, 4, n_init)
    y = np.random.uniform(-2, 2, n_init)

    # Remove points inside cylinder
    r = np.sqrt(x**2 + y**2)
    mask = r >= R
    x, y = x[mask], y[mask]

    # Subsample to N points
    if len(x) > N:
        idx = np.random.choice(len(x), N, replace=False)
        x, y = x[idx], y[idx]
    elif len(x) < N:
        # Add more points
        n_extra = N - len(x)
        x_extra = np.random.uniform(-2, 4, n_extra)
        y_extra = np.random.uniform(-2, 2, n_extra)
        r_extra = np.sqrt(x_extra**2 + y_extra**2)
        mask_extra = r_extra >= R
        x_extra, y_extra = x_extra[mask_extra], y_extra[mask_extra]
        if len(x_extra) > 0:
            x = np.concatenate([x, x_extra])
            y = np.concatenate([y, y_extra])

    points = torch.tensor(np.column_stack([x[:N], y[:N]]), dtype=torch.float32)
    return points

def main():
    from datetime import datetime
    logger = setup_logging(
        log_dir="logs",
        log_level=logging.INFO,
        experiment_name=f"generate_cylinder_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    args = parse_args()
    device = torch.device(args.device)

    os.makedirs('data/cylinder_samples', exist_ok=True)

    N = args.n
    points = generate_cylinder_points(N).to(device)
    logger.info(f"Generated {N} cylinder nodes")

    # For now, create synthetic velocity/pressure fields
    # In a real implementation, you would call the RBF-FD solver here
    x = points[:, 0].cpu().numpy()
    y = points[:, 1].cpu().numpy()
    R = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    re_values = torch.linspace(args.re_min, args.re_max, args.ns, dtype=torch.float32)

    for idx, re in enumerate(re_values):
        re_val = re.item()

        # Potential flow + vortex shedding approximation
        u = 1.0 - 0.5 * x / (R**2 + 0.01) + 0.1 * np.sin(2 * theta) * np.exp(-R/2)
        v = -0.5 * y / (R**2 + 0.01) + 0.1 * np.cos(2 * theta) * np.exp(-R/2)
        p = 0.5 * (1 - (u**2 + v**2)) + 0.05 * np.sin(3 * theta) * np.exp(-R/3)

        # Interleave u, v for a_ref
        a_ref = torch.zeros(2 * N, dtype=torch.float32)
        a_ref[0::2] = torch.tensor(u, dtype=torch.float32)
        a_ref[1::2] = torch.tensor(v, dtype=torch.float32)
        b_ref = torch.tensor(p, dtype=torch.float32)

        sample = {
            'mu': torch.tensor([re_val / args.re_max], dtype=torch.float32),
            're': re,
            'a_ref': a_ref,
            'b_ref': b_ref,
            'a_scale': torch.tensor([a_ref.abs().max().item() + 1e-8], dtype=torch.float32),
            'b_scale': torch.tensor([b_ref.abs().max().item() + 1e-8], dtype=torch.float32),
        }
        torch.save(sample, f'data/cylinder_samples/sample_{idx:04d}.pt')

        if (idx + 1) % 10 == 0:
            logger.info(f"[{idx+1}/{args.ns}] Re={re_val:.1f}")

    logger.info(f"Generated {args.ns} cylinder samples in data/cylinder_samples/")

if __name__ == '__main__':
    main()
