"""
compute_operator_gap.py
Computes the operator gap  delta_op = ||D - G^T||_2 / ||G^T||_2
for each tested node count, verifying the O(h^2) scaling claim.

Usage:
    python scripts/compute_operator_gap.py

Outputs:
    results/operator_gap_table.csv   — (N, h, delta_op, delta_op/eps_ML, slope)
    results/operator_gap_loglog.pdf  — log-log plot with regression line

Dependencies:
    rbffd_solver.py  (from /repo/)
    numpy, scipy, pandas, matplotlib
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---- Ensure repo is on path ------------------------------------------------
REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, REPO_ROOT)

try:
    from rbffd_solver import RBFFDSolver
except ImportError as e:
    logger.error(
        "Cannot import RBFFDSolver. Ensure rbffd_solver.py is in /repo/. "
        f"Original error: {e}"
    )
    sys.exit(1)

# ---- Constants from paper --------------------------------------------------
EPS_ML = 0.1375          # observed learning error (13.75%)
NODE_COUNTS = [225, 961, 4096, 10_000]
DOMAIN = 'cavity'        # unit square lid-driven cavity
STENCIL_SIZE = 25        # k=25, as stated in paper
EPS_MACHINE = np.finfo(float).eps   # ~2.2e-16


def compute_h(N: int, domain: str = 'cavity') -> float:
    """Approximate mesh spacing for a quasi-uniform 2D node set."""
    area = 1.0   # unit square
    return np.sqrt(area / N)


def compute_operator_gap(N: int, domain: str, k: int) -> dict:
    """
    Assemble the RBF-FD system and compute the operator gap.

    Parameters
    ----------
    N : int   total node count
    domain : str   geometry identifier
    k  : int  stencil size

    Returns
    -------
    dict with keys: N, h, delta_op, delta_op_vs_eps_ML
    """
    logger.info(f"Assembling solver for N={N}, domain={domain}, k={k}")
    solver = RBFFDSolver(N=N, domain=domain, k=k)
    solver.assemble()

    # Retrieve sparse matrices
    D: sp.spmatrix = solver.divergence_matrix    # shape (N_int, 2N)
    G: sp.spmatrix = solver.gradient_matrix      # shape (N_int*2, N) OR (2*N_int, N)

    # G^T should have shape (N, 2*N_int) → restrict to rows matching D's row count
    # D has shape (N_int, 2N).  G^T has shape (N, 2*N_int).
    # We need both in (N_int, 2N) space. Validate and reshape.
    GT = G.T    # shape: (N, something) or (2N, N_int)

    # Normalise GT to match D's shape if possible
    # Expected: D.shape == GT.shape for the operator gap comparison
    if D.shape != GT.shape:
        # Attempt to sub-select the interior block
        r, c = D.shape
        logger.warning(
            f"Shape mismatch: D.shape={D.shape}, G^T.shape={GT.shape}. "
            f"Attempting to align to ({r},{c})."
        )
        if GT.shape[0] >= r and GT.shape[1] == c:
            GT = GT[:r, :]
        elif GT.shape[0] == r and GT.shape[1] >= c:
            GT = GT[:, :c]
        else:
            raise ValueError(
                f"Cannot reconcile D.shape={D.shape} and GT.shape={GT.shape}. "
                "Check RBFFDSolver matrix dimensions."
            )

    assert D.shape == GT.shape, (
        f"After alignment: D.shape={D.shape} != GT.shape={GT.shape}"
    )

    # Convert difference to dense if small enough, else use sparse
    diff = D - GT
    norm_GT = spl.norm(GT, ord='fro')   # Frobenius norm for conditioning check

    if norm_GT < EPS_MACHINE * 1e6:
        raise ZeroDivisionError(
            f"||G^T||_F = {norm_GT:.2e} is too small; "
            "possible assembly error."
        )

    # Compute spectral norm via largest singular value (power iteration for large N)
    # For N <= 4096 use full svds; for N=10000 use randomised
    if N <= 4096:
        sigma_diff = spl.svds(diff, k=1, return_singular_vectors=False)[0]
        sigma_GT   = spl.svds(GT,   k=1, return_singular_vectors=False)[0]
    else:
        # Randomised SVD fallback
        rng = np.random.default_rng(42)
        v = rng.standard_normal(diff.shape[1])
        for _ in range(50):
            v = diff.T @ (diff @ v)
            v /= np.linalg.norm(v)
        sigma_diff = np.sqrt(v @ (diff.T @ (diff @ v)))
        v = rng.standard_normal(GT.shape[1])
        for _ in range(50):
            v = GT.T @ (GT @ v)
            v /= np.linalg.norm(v)
        sigma_GT = np.sqrt(v @ (GT.T @ (GT @ v)))

    delta_op = sigma_diff / sigma_GT
    h = compute_h(N)

    result = {
        'N': N,
        'h': h,
        'delta_op': delta_op,
        'delta_op/eps_ML': delta_op / EPS_ML,
    }
    logger.info(
        f"  N={N:6d}  h={h:.4f}  delta_op={delta_op:.4e}  "
        f"delta_op/eps_ML={delta_op/EPS_ML:.4e}"
    )
    return result


def verify_h2_scaling(records: list) -> float:
    """
    Fit log(delta_op) = slope * log(h) + const.
    Returns the estimated slope.  Expected: slope ≈ 2.0 (±0.3).
    """
    hs       = np.array([r['h']        for r in records])
    deltas   = np.array([r['delta_op'] for r in records])
    log_h    = np.log(hs)
    log_d    = np.log(deltas)
    slope, intercept = np.polyfit(log_h, log_d, 1)
    return slope


def make_loglog_plot(records: list, slope: float, out_path: str):
    hs     = [r['h']        for r in records]
    deltas = [r['delta_op'] for r in records]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(hs, deltas, 'ko-', markersize=6, label=r'$\delta_\mathrm{op}$')

    # Reference O(h^2) line
    h_ref = np.array([min(hs)*0.8, max(hs)*1.2])
    d_ref = deltas[0] * (h_ref / hs[0])**2
    ax.loglog(h_ref, d_ref, 'r--', alpha=0.7, label=r'$\mathcal{O}(h^2)$')

    ax.axhline(EPS_ML, color='blue', linestyle=':', alpha=0.7,
               label=r'$\varepsilon_\mathrm{ML}=13.75\%$')

    ax.set_xlabel(r'Mesh spacing $h$')
    ax.set_ylabel(r'$\delta_\mathrm{op} = \|\mathbf{D}-\mathbf{G}^T\|_2 / \|\mathbf{G}^T\|_2$')
    ax.set_title(rf'Operator gap scaling  (fitted slope = {slope:.2f})')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # Annotate slope
    mid_h = np.sqrt(hs[0] * hs[-1])
    mid_d = np.exp(np.interp(np.log(mid_h),
                              np.log(hs), np.log(deltas)))
    ax.annotate(rf'slope $\approx {slope:.2f}$',
                xy=(mid_h, mid_d), fontsize=9,
                xytext=(mid_h*1.5, mid_d*2),
                arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Log-log plot saved to {out_path}")


if __name__ == '__main__':
    os.makedirs(os.path.join(REPO_ROOT, 'results'), exist_ok=True)

    records = []
    for N in NODE_COUNTS:
        rec = compute_operator_gap(N, domain=DOMAIN, k=STENCIL_SIZE)
        records.append(rec)

    slope = verify_h2_scaling(records)
    logger.info(f"\nFitted slope: {slope:.3f}  (expected: 2.0 ± 0.3)")

    if not (1.7 <= slope <= 2.3):
        logger.warning(
            f"FLAGGED: Slope {slope:.3f} outside expected range [1.7, 2.3]. "
            "Check stencil construction or return to Mathematician agent."
        )

    # Check delta_op < eps_ML at all N
    for rec in records:
        if rec['delta_op'] >= EPS_ML:
            logger.error(
                f"CRITICAL: delta_op={rec['delta_op']:.4e} >= eps_ML={EPS_ML:.4e} "
                f"at N={rec['N']}. Gram substitution validity is COMPROMISED at this N."
            )

    # Save CSV
    df = pd.DataFrame(records)
    df['fitted_slope'] = slope
    csv_path = os.path.join(REPO_ROOT, 'results', 'operator_gap_table.csv')
    df.to_csv(csv_path, index=False, float_format='%.6e')
    logger.info(f"Results saved to {csv_path}")
    print(df.to_string(index=False))

    # Save plot
    plot_path = os.path.join(REPO_ROOT, 'results', 'operator_gap_loglog.pdf')
    make_loglog_plot(records, slope, plot_path)
