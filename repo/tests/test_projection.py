"""tests/test_projection.py - Uses REAL G_int from solver.

Mathematical note on Pythagorean identity (Property 4):
- Full-domain projection (no interior_mask): TRUE orthogonal projection.
  Pythagorean identity holds exactly: ||a||^2 = ||P(a)||^2 + ||a - P(a)||^2

- Interior-restricted projection (with interior_mask): PARTIAL projection.
  Boundary DOFs are fixed, interior DOFs are optimized.
  The Pythagorean identity does NOT hold exactly because the boundary
  values "pollute" the Lagrange multiplier solve:
  (G_int[:, int] @ G_int[:, int].T + eps*I) q = G_int @ a_hat
  where RHS includes boundary contributions: G_int[:, bnd] @ a_hat[bnd]

  This is expected and correct behavior per Proposition 4.
  The trade-off is: boundary invariance ↔ exact Pythagorean identity.

Therefore, we test:
- Properties 1-3, 5-6 on interior-restricted projection (real usage)
- Property 4 on full-domain projection (mathematical purity)
"""

import pytest
import torch

from src.projection.layer import HelmholtzProjection


@pytest.fixture
def G_int_and_proj():
    """Fixture with REAL G_int and interior-restricted projection."""
    G_int = torch.load('data/fixed_G.pt', map_location='cpu')
    interior_mask = torch.load('data/interior_mask.pt', map_location='cpu')
    proj = HelmholtzProjection(G_int, eps=1e-8, interior_mask=interior_mask)
    return G_int, proj


@pytest.fixture
def G_full_and_proj():
    """Fixture with G_full and full-domain projection (for Pythagorean test)."""
    G_full = torch.load('data/G_full.pt', map_location='cpu')
    # Full-domain projection: no interior_mask
    proj = HelmholtzProjection(G_full, eps=1e-8, interior_mask=None)
    return G_full, proj


def test_mass_conservation(G_int_and_proj):
    """Property 1: ||G_int a_NO|| < 1e-2 (interior-restricted projection)."""
    G_int, proj = G_int_and_proj
    torch.manual_seed(0)
    a_hat = torch.randn(G_int.shape[1], dtype=torch.float32) * 10.0
    a_NO = proj.project_only(a_hat)
    div_after = torch.norm(G_int @ a_NO).item()
    assert div_after < 1e-2, f"ε_div = {div_after:.2e} too large. Expected < 1e-2."


def test_divergence_reduction_ratio(G_int_and_proj):
    """Property 2: ρ = ||G_int a_hat|| / ||G_int a_NO|| > 100."""
    G_int, proj = G_int_and_proj
    torch.manual_seed(1)
    a_hat = torch.randn(G_int.shape[1], dtype=torch.float32) * 10.0
    r_before = torch.norm(G_int @ a_hat).item()
    a_NO = proj.project_only(a_hat)
    r_after = torch.norm(G_int @ a_NO).item()
    rho = r_before / (r_after + 1e-12)
    assert rho > 100, f"ρ = {rho:.2e} too small. Expected > 100."


def test_idempotency(G_int_and_proj):
    """Property 3: P(P(a)) = P(a) (interior-restricted projection)."""
    G_int, proj = G_int_and_proj
    torch.manual_seed(2)
    a_hat = torch.randn(G_int.shape[1], dtype=torch.float32) * 10.0
    a1 = proj.project_only(a_hat)
    a2 = proj.project_only(a1)
    diff = torch.norm(a1 - a2).item()
    assert diff < 1e-2, f"Idempotency violated: diff = {diff:.2e}. Expected < 1e-2."


def test_minimum_energy_pythagorean(G_full_and_proj):
    """Property 4: Pythagorean identity holds for FULL-DOMAIN projection.

    CRITICAL FIX: This test uses G_full (not G_int) with interior_mask=None
    to verify the Pythagorean identity on a TRUE orthogonal projection.

    With interior-restricted projection, boundary values are fixed and the
    identity does not hold exactly. This is mathematically expected.
    """
    G_full, proj = G_full_and_proj
    torch.manual_seed(3)
    a_hat = torch.randn(G_full.shape[1], dtype=torch.float32) * 10.0

    a_NO = proj.project_only(a_hat)

    lhs = torch.norm(a_hat).item() ** 2
    rhs = torch.norm(a_NO).item() ** 2 + torch.norm(a_hat - a_NO).item() ** 2

    rel_error = abs(lhs - rhs) / lhs
    assert rel_error < 1e-3, (
        f"Pythagorean identity violated: rel_error = {rel_error:.2e}. "
        f"Expected < 1e-3 for full-domain orthogonal projection."
    )


def test_differentiability(G_int_and_proj):
    """Property 5: P_div is differentiable w.r.t. input."""
    G_int, proj = G_int_and_proj
    torch.manual_seed(4)
    a_hat = torch.randn(G_int.shape[1], dtype=torch.float32) * 10.0
    a_hat.requires_grad_(True)
    a_NO = proj.project_only(a_hat)
    a_NO.sum().backward()
    assert a_hat.grad is not None, "Gradient did not flow back"
    assert a_hat.grad.norm().item() > 0, "Gradient norm is zero"


def test_finiteness_on_zero_input(G_int_and_proj):
    """Property 6: P(0) = 0."""
    G_int, proj = G_int_and_proj
    a_zero = torch.zeros(G_int.shape[1], dtype=torch.float32)
    a_NO = proj.project_only(a_zero)
    assert torch.allclose(a_NO, a_zero, atol=1e-6), "P(0) != 0"
    assert not torch.isnan(a_NO).any(), "NaN detected"
    assert not torch.isinf(a_NO).any(), "Inf detected"
