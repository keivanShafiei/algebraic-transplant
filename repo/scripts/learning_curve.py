#!/usr/bin/env python3
"""scripts/learning_curve.py — Dataset-size sensitivity for Appendix F.

Reproduces the learning-curve analysis referenced in Section 4.6
(Limitations) and documented in Appendix F.  The curve shows that
MSE_w saturates beyond ~200 samples, confirming that the model has
learned the parametric structure rather than memorising individual
training instances.  The constraint residual ε_div is independent of
dataset size (governed by linear algebra), so only MSE_w is reported
here.

Usage
-----
    python scripts/learning_curve.py \
        --dataset_dir data/lid_driven_cavity \
        --test_dir    data/lid_driven_cavity/test \
        --output_dir  results/learning_curve \
        --seeds       42 123 456 \
        --device      cuda

The script writes:
    results/learning_curve/learning_curve.json          — raw numbers
    results/learning_curve/learning_curve.pdf           — matplotlib plot
    results/learning_curve/learning_curve_table.tex     — LaTeX table

Dependencies
------------
    torch, numpy, matplotlib, tqdm
    src.gnn.train  (train_model)
    src.gnn.evaluate (evaluate_model)

Author
------
    Amirkeivan Shafiei  <k.shafiei@birjand.ac.ir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# ------------------------------------------------------------------
# Project imports — adjust PYTHONPATH if running outside repo root
# ------------------------------------------------------------------
try:
    from src.gnn.evaluate import evaluate_model
    from src.gnn.train import train_model
except ImportError as exc:
    raise ImportError(
        "Could not import src.gnn.{train,evaluate}. "
        "Run this script from the repository root or set PYTHONPATH."
    ) from exc

# ------------------------------------------------------------------
# Defaults (must match Table 1 / hyperparams in the paper)
# ------------------------------------------------------------------
DEFAULT_DATASET_DIR = Path("data/lid_driven_cavity")
DEFAULT_TEST_DIR = Path("data/lid_driven_cavity/test")
DEFAULT_OUTPUT_DIR = Path("results/learning_curve")
DEFAULT_SEEDS = (42, 123, 456)
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset sizes to sweep.  The last entry (353) is the full filtered
# training set used in the paper (364 initial minus 11 filtered).
DATASET_SIZES = (50, 100, 150, 200, 250, 300, 353)

# Training hyper-parameters — frozen to match Table 1
TRAIN_KWARGS: dict[str, Any] = {
    "epochs": 200,
    "batch_size": 16,
    "lr_init": 1e-3,
    "lr_final": 1e-6,
    "hidden_dim": 64,
    "num_layers": 4,
    "lambda_early": 0.1,
    "lambda_late": 0.01,
    "lambda_transition": 150,
    "loss_w_prs": 0.1,
    "loss_w_div": 0.01,
    "verbose": False,
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _print_banner(msg: str) -> None:
    """Pretty-print a section banner."""
    line = "=" * 70
    print(f"\n{line}\n{msg}\n{line}")


def _save_json(path: Path, obj: Any) -> None:
    """Atomically write JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _save_latex_table(path: Path, records: list[dict]) -> None:
    """Write a LaTeX table matching the paper's booktabs style."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Learning curve: variance-weighted MSE $\mathrm{MSE}_w$ "
        r"and constraint residual $\varepsilon_{\mathrm{div}}$ versus "
        r"training-set size $N_s$.  Mean $\pm$ std.\ dev.\ over three seeds.}",
        r"\label{tab:learning_curve}",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"$N_s$ & $\mathrm{MSE}_w$ & $\varepsilon_{\mathrm{div}}$ (float64) "
        r"& Saturation? \\",
        r"\midrule",
    ]

    for rec in records:
        ns = rec["n_samples"]
        mse_mean = rec["mse_w_mean"]
        mse_std = rec["mse_w_std"]
        eps_div = rec["eps_div_mean"]  # nearly constant; std negligible
        # Mark saturation visually when MSE_w changes < 5 % from previous
        sat = r"\checkmark" if rec.get("saturated", False) else ""
        lines.append(
            f"{ns} & ${mse_mean:.4f} \\pm {mse_std:.4f}$ & "
            f"${eps_div:.2e}$ & {sat} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_learning_curve(
    records: list[dict],
    output_path: Path,
    dpi: int = 300,
) -> None:
    """Generate the MSE_w vs N_s figure (Appendix F)."""
    ns_vals = np.array([r["n_samples"] for r in records])
    mse_mean = np.array([r["mse_w_mean"] for r in records])
    mse_std = np.array([r["mse_w_std"] for r in records])

    fig, ax = plt.subplots(figsize=(6, 4))

    # Main curve with error bars
    ax.errorbar(
        ns_vals,
        mse_mean,
        yerr=mse_std,
        fmt="o-",
        color="#1f77b4",
        ecolor="#ff7f0e",
        capsize=4,
        linewidth=1.5,
        markersize=6,
        label=r"$\mathrm{MSE}_w$ (mean $\pm$ std)",
    )

    # Signal-variance reference line (from Table 3)
    signal_var = 3.765e-2
    ax.axhline(
        signal_var,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=r"Signal variance $\bar{\sigma}^2 = 3.765\times10^{-2}$",
    )

    # Saturation band
    ax.axvspan(200, 360, alpha=0.1, color="green", label="Saturation regime")

    ax.set_xlabel(r"Training-set size $N_s$", fontsize=11)
    ax.set_ylabel(r"Variance-weighted MSE $\mathrm{MSE}_w$", fontsize=11)
    ax.set_title("Learning Curve: Lid-Driven Cavity, Re$\\in[10,100]$, $N=225$", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(40, 370)
    ax.set_ylim(0.0, 0.12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


# ------------------------------------------------------------------
# Core experiment
# ------------------------------------------------------------------
def run_learning_curve(
    dataset_dir: Path,
    test_dir: Path,
    output_dir: Path,
    seeds: tuple[int, ...],
    device: torch.device,
) -> list[dict]:
    """Train and evaluate at multiple dataset sizes; return records."""
    records: list[dict] = []

    for n_samples in DATASET_SIZES:
        _print_banner(f"Training with N_s = {n_samples}")

        seed_results: list[dict] = []
        for seed in seeds:
            print(f"  Seed {seed} … ", end="", flush=True)
            t0 = time.perf_counter()

            # Train
            model, history = train_model(
                data_dir=dataset_dir,
                n_samples=n_samples,
                seed=seed,
                device=device,
                **TRAIN_KWARGS,
            )

            # Evaluate on held-out test set (identical for all n_samples)
            metrics = evaluate_model(
                model=model,
                test_dir=test_dir,
                device=device,
            )

            elapsed = time.perf_counter() - t0
            mse_w = metrics["mse_w"]
            eps_div = metrics.get("eps_div_float64", float("nan"))

            print(f"MSE_w={mse_w:.4f}, ε_div={eps_div:.2e}, {elapsed:.1f}s")

            seed_results.append(
                {
                    "seed": seed,
                    "mse_w": mse_w,
                    "eps_div": eps_div,
                    "elapsed_s": elapsed,
                }
            )

            # Free GPU memory between seeds
            del model
            torch.cuda.empty_cache()

        # Aggregate across seeds
        mse_vals = [s["mse_w"] for s in seed_results]
        eps_vals = [s["eps_div"] for s in seed_results]

        record = {
            "n_samples": n_samples,
            "mse_w_mean": float(np.mean(mse_vals)),
            "mse_w_std": float(np.std(mse_vals, ddof=1)),
            "eps_div_mean": float(np.mean(eps_vals)),
            "eps_div_std": float(np.std(eps_vals, ddof=1)),
            "seed_runs": seed_results,
        }
        records.append(record)

    # Mark saturation (change < 5 % from previous point)
    for i, rec in enumerate(records):
        if i == 0:
            rec["saturated"] = False
            continue
        prev = records[i - 1]["mse_w_mean"]
        curr = rec["mse_w_mean"]
        rec["saturated"] = abs(curr - prev) / (prev + 1e-12) < 0.05

    return records


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learning-curve analysis for the RBF-FD GNN projection paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing the training dataset .pt files.",
    )
    parser.add_argument(
        "--test_dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help="Directory containing the held-out test set.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write JSON, PDF, PNG, and LaTeX outputs.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Random seeds for ensemble training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="torch device string.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
    if not args.test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Dataset sizes: {DATASET_SIZES}")
    print(f"Seeds: {args.seeds}")

    # Run experiment
    records = run_learning_curve(
        dataset_dir=args.dataset_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        seeds=tuple(args.seeds),
        device=device,
    )

    # Save JSON
    json_path = args.output_dir / "learning_curve.json"
    _save_json(json_path, records)
    print(f"\nJSON saved: {json_path}")

    # Save LaTeX table
    tex_path = args.output_dir / "learning_curve_table.tex"
    _save_latex_table(tex_path, records)
    print(f"LaTeX table saved: {tex_path}")

    # Save plot
    plot_path = args.output_dir / "learning_curve.pdf"
    _plot_learning_curve(records, plot_path)
    print(f"Plot saved: {plot_path} (+ .png)")

    # Console summary
    _print_banner("SUMMARY")
    print(f"{'N_s':>6} | {'MSE_w':>12} | {'ε_div (64)':>14} | Sat?")
    print("-" * 55)
    for rec in records:
        sat = "YES" if rec["saturated"] else "NO"
        print(
            f"{rec['n_samples']:>6} | "
            f"{rec['mse_w_mean']:.4f} ± {rec['mse_w_std']:.4f} | "
            f"{rec['eps_div_mean']:.2e} | {sat}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
