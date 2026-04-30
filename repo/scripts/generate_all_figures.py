"""Generate reproducible figures from the Algebraic Transplant paper.

GF1 fix: all execution moved inside main() with if __name__ == '__main__'
    guard — prevents side-effects on import.

GF2 fix: honest inventory of which figures are produced vs. which require
    data that is not yet generated (NS solver output, trained model, etc.).

GF3 fix: figure filenames corrected to match paper numbering.

Usage:
    python scripts/generate_all_figures.py

Prerequisites:
    data/fixed_G.pt          — from scripts/generate_data.py
    results/model_final.pt   — from scripts/train.py
    data/samples/sample_*.pt — from RBF-FD solver (Algorithm 1, not yet impl.)
"""

import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure project root is on path when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.viz import plot_projection_efficacy, plot_resolution_invariance


def main() -> None:
    os.makedirs('results/figures', exist_ok=True)

    print("Generating figures from the Algebraic Transplant manuscript...")
    print()

    # ── Figures that CAN be generated from current code ──────────────────────

    # Figure 8: Resolution invariance (hardcoded reference values, Section 4.1)
    print("[Fig 8] Resolution invariance (reference values from paper Section 4.1)")
    plot_resolution_invariance()

    # Figure 10: Projection efficacy — requires trained model + RBF-FD samples
    model_path = 'results/model_final.pt'
    sample_dir = 'data/samples'
    
    if os.path.exists(model_path) and os.path.isdir(sample_dir) and \
            any(f.endswith('.pt') for f in os.listdir(sample_dir)):
        print("[Fig 10] Projection efficacy (Algorithm 3 — using saved model and samples)")
        from scripts.eval_projection_fixed import run_projection_eval
    
        rb, ra, rhos = run_projection_eval(
            sample_dir=sample_dir,
            model_path=model_path,
        )
        plot_projection_efficacy(rb, ra, rhos)
    else:
        print("[Fig 10] SKIPPED — requires:")
        if not os.path.exists(model_path):
            print(f"          ✗ {model_path}  (run scripts/train.py)")
        if not os.path.isdir(sample_dir) or not any(f.endswith('.pt') for f in os.listdir(sample_dir)):
            print(f"          ✗ {sample_dir}/sample_*.pt  (run RBF-FD solver, Algorithm 1)")

    # ── GF2: Figures NOT yet generatable — honest inventory ──────────────────
    missing = [
        ("Fig.  1", "Pipeline timing decomposition",
         "requires timing runs across all 4 discretization methods"),
        ("Fig.  2", "Per-iteration mass conservation",
         "requires full Algorithm 1 NS solver (not yet implemented)"),
        ("Fig.  3/7", "Kinetic energy spectrum",
         "requires high-res solver run at N=10000"),
        ("Fig.  4/21", "Training dynamics (synthetic)",
         "run after fixing synthetic.py — sanity check only"),
        ("Fig.  5", "Performance analysis",
         "requires trained model evaluation on test set"),
        ("Fig.  6", "Field comparison at Re=92",
         "requires trained model + solver reference"),
        ("Fig.  9", "Constraint enforcement bar chart",
         "requires baseline models (FNO, DeepONet, POD-RBF)"),
        ("Fig. 11", "Computational scaling analysis",
         "requires timing measurements at N∈{225,1000,5000,10000}"),
        ("Fig. 12", "Statistical robustness (5 seeds)",
         "requires 5 full training runs"),
        ("Fig. 13", "Ablation study",
         "requires soft-constrained baseline model"),
        ("Fig. 14", "Preconditioner Re sweep",
         "requires solver runs at Re∈{100,150,200,300,400,500}"),
        ("Fig. 15/18", "Hybrid warm-start convergence history",
         "requires solver + model at Re=200, Re=500"),
    ]
    print("Figures NOT yet generatable (prerequisites missing):")
    for fig, title, reason in missing:
        print(f"  {fig:<10} {title:<40} ← {reason}")

    print()
    print("Done. Generated figures saved to results/figures/")


if __name__ == '__main__':
    main()
