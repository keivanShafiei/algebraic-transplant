"""utils/metrics.py — Evaluation Metrics for the Algebraic Transplant Framework.

Implements the three core evaluation metrics used throughout Section 4 of
the paper (Eq. 22).

Paper reference
---------------
Section 4 (Numerical Results), Eq. (22):

    eps_L2(u) = ||u_pred - u_ref||_2 / ||u_ref||_2

    eps_div = ||G a_pred||_2

    MSE_w = (1/N_t) * sum_s sum_i  w_i * ||u_hat_i^s - u_i^s||^2
"""

import torch


def divergence_residual(G: torch.Tensor, a: torch.Tensor) -> float:
    """Compute the discrete divergence residual eps_div = ||G a||_2.

    This is the primary constraint metric. After the Algebraic Transplant
    projection, eps_div should be ~ 4e-5 in float32 and ~ O(10^-13) in
    float64 (Remark 2, Table 9).

    Parameters
    ----------
    G : torch.Tensor
        Discrete divergence operator, shape (N, 2N) or (N_int, 2N).
    a : torch.Tensor
        Velocity coefficient vector, shape (2N,).

    Returns
    -------
    float
        The l2 norm of the divergence residual.
    """
    return torch.norm(G @ a).item()


def relative_l2_error(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """Compute the relative L2 field error.

    eps_L2 = ||pred - ref||_2 / (||ref||_2 + eps)

    Parameters
    ----------
    pred : torch.Tensor
        Predicted field (velocity or pressure coefficients).
    ref : torch.Tensor
        Reference (ground truth) field.

    Returns
    -------
    float
        Relative L2 error in [0, inf).

    Notes
    -----
    The current model achieves eps_L2(u) = 13.75 +/- 1.25% on the
    in-distribution test set (Table 11). The engineering-grade force
    accuracy threshold requires eps_L2(u) < 3-4% (Section 5.3, Table 19).
    """
    return (torch.norm(pred - ref) / (torch.norm(ref) + 1e-8)).item()


def variance_weighted_mse(
    pred: torch.Tensor,
    ref: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    """Compute the variance-weighted MSE (Eq. 22, main training loss).

    MSE_w = mean_{i} [ w_i * (pred_i - ref_i)^2 ]

    where w_i = sigma_i^2 / sigma_bar^2 is the empirical variance of
    node i across training samples, normalised to unit mean.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted field, shape (N_fields,) or (B, N_fields).
    ref : torch.Tensor
        Reference field, same shape as pred.
    weights : torch.Tensor
        Per-DOF variance weights, shape (N_fields,).

    Returns
    -------
    float
        Scalar weighted MSE value.

    Notes
    -----
    The paper reports MSE_w = 3.728e-2 for the HMG model, versus signal
    variance 3.765e-2 (Table 6), indicating that remaining error is of the
    same order as the intrinsic parametric variability.
    """
    sq_err = (pred - ref).pow(2)
    if sq_err.dim() > 1:
        return (sq_err * weights.unsqueeze(0)).mean().item()
    return (sq_err * weights).mean().item()
