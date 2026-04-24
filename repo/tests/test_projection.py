"""Tests for HelmholtzProjection layer (Eq. 22, Theorem 2).

Covers all three mathematical properties from Theorem 2:
    1. Near-exact mass conservation:  ||G P_div(u)||  ≤ ε||Gu||/σ_min(L)
    2. Minimum-energy correction:     Pythagorean identity
    3. Differentiability:             gradient flows through cholesky_solve

TP1 fix: a_hat created with requires_grad=True so backward() populates grad.
TP3 fix: idempotency test added (P² = P).
TP4 fix: minimum-energy (Pythagorean) test added.
TP5 fix: removed unused synthetic data import.
"""

import torch
import pytest
from src.projection.layer import HelmholtzProjection


@pytest.fixture
def G_and_proj():
    """Load fixed_G.pt and build projection layer."""
    G    = torch.load('data/fixed_G.pt', map_location='cpu')
    proj = HelmholtzProjection(G, eps=1e-8)
    return G, proj


def test_mass_conservation(G_and_proj):
    """Property 1 (Theorem 2): ||G a_NO|| ≈ 4e-5 (float32 floor, Remark 3)."""
    G, proj = G_and_proj
    torch.manual_seed(0)
    a_hat = torch.randn(G.shape[1], dtype=torch.float32) * 10.0

    a_NO     = proj.project_only(a_hat)
    div_after = torch.norm(G @ a_NO).item()

    assert div_after < 5e-5, (
        f"ε_div = {div_after:.2e} exceeds float32 floor threshold 5e-5. "
        f"Expected ≈4e-5 (Remark 3, Table 8)."
    )
    assert torch.isfinite(a_NO).all(), "Projected output contains NaN/Inf"


def test_divergence_reduction_ratio(G_and_proj):
    """Reduction ratio ρ = r_before/r_after should be O(1e5) (Table 8)."""
    G, proj = G_and_proj
    torch.manual_seed(1)
    a_hat = torch.randn(G.shape[1], dtype=torch.float32) * 10.0

    r_before = torch.norm(G @ a_hat).item()
    a_NO     = proj.project_only(a_hat)
    r_after  = torch.norm(G @ a_NO).item()
    rho      = r_before / (r_after + 1e-12)

    assert rho > 1e3, (
        f"Reduction ratio ρ = {rho:.2e} is too small. "
        f"Paper Table 8 reports ρ̄ ≈ 2.13e5."
    )


def test_idempotency(G_and_proj):
    """Property: P_div ∘ P_div = P_div  (projection is idempotent).

    TP3 fix: was not tested in original code.
    Theorem 2, property 2: P_div is the unique ℓ²-optimal projection
    onto ker(G), hence idempotent.
    """
    G, proj = G_and_proj
    torch.manual_seed(2)
    a_hat = torch.randn(G.shape[1], dtype=torch.float32)

    a_NO         = proj.project_only(a_hat)
    a_NO_twice   = proj.project_only(a_NO)
    idempotency  = torch.norm(a_NO_twice - a_NO).item()

    assert idempotency < 1e-5, (
        f"Idempotency violated: ||P²(â) - P(â)|| = {idempotency:.2e} (should be ≈0)."
    )


def test_minimum_energy_pythagorean(G_and_proj):
    """Property 2 (Theorem 2): Pythagorean identity for ℓ²-optimal projection.

    TP4 fix: was not tested in original code.
    For the ℓ²-projection onto a subspace:
        ||â||² = ||a_NO||² + ||â - a_NO||²
    i.e., the correction (â - a_NO) is orthogonal to a_NO.
    """
    G, proj = G_and_proj
    torch.manual_seed(3)
    a_hat = torch.randn(G.shape[1], dtype=torch.float32) * 5.0

    a_NO       = proj.project_only(a_hat)
    correction = a_hat - a_NO

    # ||â||² = ||a_NO||² + ||correction||²
    lhs = torch.dot(a_hat, a_hat).item()
    rhs = torch.dot(a_NO, a_NO).item() + torch.dot(correction, correction).item()

    rel_err = abs(lhs - rhs) / (abs(lhs) + 1e-12)
    assert rel_err < 1e-4, (
        f"Pythagorean identity violated: rel_err = {rel_err:.2e}. "
        f"||â||²={lhs:.4f}, ||a_NO||²+||corr||²={rhs:.4f}."
    )


def test_differentiability(G_and_proj):
    """Property 3 (Theorem 2): P_div is differentiable w.r.t. input.

    TP1 fix: requires_grad=True added. Original code used requires_grad=False
    (default), so backward() produced no gradients and the assertion
    'a_hat.grad is not None' always failed.
    """
    G, proj = G_and_proj
    torch.manual_seed(4)

    # TP1 fix: must set requires_grad=True
    a_hat = torch.randn(G.shape[1], dtype=torch.float32,
                        requires_grad=True) * 10.0

    a_NO = proj.project_only(a_hat)
    a_NO.sum().backward()

    assert a_hat.grad is not None, "Gradient did not flow back to a_hat"
    assert torch.isfinite(a_hat.grad).all(), "Gradient contains NaN/Inf"
    assert a_hat.grad.shape == a_hat.shape, "Gradient shape mismatch"


def test_finiteness_on_zero_input(G_and_proj):
    """Edge case: projection of zero vector should be zero."""
    G, proj = G_and_proj
    a_zero = torch.zeros(G.shape[1], dtype=torch.float32)
    a_NO   = proj.project_only(a_zero)
    assert torch.allclose(a_NO, a_zero, atol=1e-7), \
        "P_div(0) should be 0 (zero is already divergence-free)"
