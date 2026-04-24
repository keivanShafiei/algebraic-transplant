"""Visualisation utilities for the Algebraic Transplant paper.

Each function corresponds to a specific figure in the manuscript.
Figure numbers match the paper exactly.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    """V4 fix: create output directory if it does not exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_projection_efficacy(
        rb: list, ra: list, rhos: list,
        epoch_hard_divs: list | None = None,
        epoch_soft_divs: list | None = None,
        save_path: str = "results/figures/fig10_projection_efficacy.png"
) -> None:
    """Figure 10 — Projection layer efficacy on RBF-FD test set.

    V1 fix: panel (d) no longer plots rb (80 test residuals) against
        np.arange(200) (200 epoch indices) — that caused a shape-mismatch
        crash (ValueError: x and y must have same first dimension).
        Panel (d) now requires actual per-epoch divergence histories.

    V2 fix: panel (d) labels corrected to match Figure 10(d) semantics
        (εdiv vs. epoch, not test residuals vs. arbitrary x-axis).

    V3 note: panels (a)-(c) use real rb/ra/rhos from Algorithm 3.

    V4 fix: output directory created before save.

    GF3 fix: file saved as fig10_* (paper Fig. 10), not fig8_*.

    Args:
        rb               : list of 80 pre-projection residuals r_before^(s)
        ra               : list of 80 post-projection residuals r_after^(s)
        rhos             : list of 80 reduction ratios ρ^(s)
        epoch_hard_divs  : per-epoch mean εdiv for hard-constrained model (len=200)
                           Pass None to show placeholder line at 4e-5.
        epoch_soft_divs  : per-epoch mean εdiv for soft-constrained baseline
                           Pass None to show illustrative exponential decay.
        save_path        : output path
    """
    _ensure_dir(save_path)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # (a) pre-projection residual distribution
    axs[0, 0].hist(rb, bins=30, color='steelblue', alpha=0.8, edgecolor='k', lw=0.5)
    axs[0, 0].axvline(np.mean(rb), color='red', ls='--', label=f'mean={np.mean(rb):.2e}')
    axs[0, 0].set_title('(a) Pre-projection residuals')
    axs[0, 0].set_xlabel(r'$r_{\mathrm{before}} = \|\mathbf{G}\hat{\mathbf{a}}\|_2$')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].legend(fontsize=8)

    # (b) post-projection residual distribution
    axs[0, 1].hist(ra, bins=30, color='seagreen', alpha=0.8, edgecolor='k', lw=0.5)
    axs[0, 1].axvline(np.mean(ra), color='red', ls='--', label=f'mean={np.mean(ra):.2e}')
    axs[0, 1].set_title('(b) Post-projection residuals')
    axs[0, 1].set_xlabel(r'$r_{\mathrm{after}} = \|\mathbf{G}\mathbf{a}_{\mathrm{NO}}\|_2\ (\times10^{-5})$')
    axs[0, 1].set_ylabel('Count')
    axs[0, 1].legend(fontsize=8)

    # (c) log-log scatter: rafter vs rbefore (Figure 10(c))
    axs[1, 0].scatter(rb, ra, alpha=0.6, s=20, color='steelblue')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')
    axs[1, 0].axhline(1.8e-6, color='gray', ls=':', label='float32 floor ≈1.8e-6')
    axs[1, 0].set_title('(c) Log-log: $r_{\\mathrm{after}}$ vs $r_{\\mathrm{before}}$')
    axs[1, 0].set_xlabel(r'$r_{\mathrm{before}}$')
    axs[1, 0].set_ylabel(r'$r_{\mathrm{after}}$')
    axs[1, 0].legend(fontsize=8)

    # (d) V1+V2 fix: εdiv vs. training epoch (requires per-epoch logging)
    epochs = np.arange(1, 201)
    if epoch_hard_divs is not None and len(epoch_hard_divs) == 200:
        axs[1, 1].semilogy(epochs, epoch_hard_divs, color='blue',
                           label='Hard (proposed)')
    else:
        # Placeholder: flat line at paper-reported value
        axs[1, 1].semilogy(epochs, np.full(200, 4e-5), color='blue', ls='--',
                           label='Hard (proposed) — reference value')

    if epoch_soft_divs is not None and len(epoch_soft_divs) == 200:
        axs[1, 1].semilogy(epochs, epoch_soft_divs, color='orange',
                           label='Soft baseline')
    else:
        # Placeholder: illustrative exponential decay from 1e-1
        soft_illustrative = 1e-1 * np.exp(-0.02 * epochs) + 1e-2
        axs[1, 1].semilogy(epochs, soft_illustrative, color='orange', ls='--',
                           label='Soft baseline — illustrative')

    axs[1, 1].axhline(4e-5, color='gray', ls=':', alpha=0.5, label='4×10⁻⁵ floor')
    axs[1, 1].set_title(r'(d) $\varepsilon_{\mathrm{div}}$ vs. epoch')
    axs[1, 1].set_xlabel('Training epoch')
    axs[1, 1].set_ylabel(r'$\|\mathbf{G}\mathbf{a}\|_2$')
    axs[1, 1].legend(fontsize=8)

    plt.suptitle('Figure 10: Projection Layer Efficacy — RBF-FD Test Set\n'
                 f'(Ntest={len(rb)}, N=225, Re∈[10,100], float32)',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 10 saved: {save_path}")


def plot_resolution_invariance(
        solver_errors: dict | None = None,
        no_errors: dict | None = None,
        save_path: str = "results/figures/fig8_resolution_invariance.png"
) -> None:
    """Figure 8 — Resolution invariance and numerical consistency.

    V3 note: when solver_errors/no_errors are None, uses reference values
        from the paper (Section 4.1). Documented explicitly as hardcoded
        reference data, not computed from a running solver/model.

    GF3 fix: saved as fig8_* (paper Fig. 8), not fig6.png.
    V4 fix: output directory created before save.

    Args:
        solver_errors : {N: L2_error} dict from actual RBF-FD runs, or None
        no_errors     : {N: L2_error} dict from actual model evaluation, or None
        save_path     : output path
    """
    _ensure_dir(save_path)

    N_vals = np.array([225, 1000, 5000, 10000])

    if solver_errors is not None:
        s_errs = np.array([solver_errors[n] for n in N_vals])
    else:
        # V3: paper Section 4.1 reference values (hardcoded — not computed)
        s_errs = 2.4e-3 * (225 / N_vals) ** 0.89
        print("  Note: solver errors are paper reference values (Section 4.1), not computed.")

    if no_errors is not None:
        n_errs = np.array([no_errors[n] for n in N_vals])
    else:
        # V3: paper Section 4.1 (resolution-invariant ≈ 10%)
        n_errs = np.full_like(N_vals, 0.108, dtype=float)
        print("  Note: neural operator errors are paper reference values, not computed.")

    plt.figure(figsize=(8, 6))
    plt.loglog(N_vals, s_errs, 'b-o', label='RBF-FD Solver  O(h²)', lw=2, ms=8)
    plt.loglog(N_vals, n_errs, 'r-s', label='Neural Operator  ~10% (invariant)', lw=2, ms=8)

    # Reference slope O(h²) ∝ N^{-1}
    h2_ref = s_errs[0] * (N_vals[0] / N_vals)
    plt.loglog(N_vals, h2_ref, 'k--', alpha=0.4, label='O(h²) reference')

    plt.xlabel('Nodal Density $N$', fontsize=12)
    plt.ylabel('Relative $L_2$ Error', fontsize=12)
    plt.title('Figure 8: Resolution Invariance and Numerical Consistency', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 8 saved: {save_path}")
