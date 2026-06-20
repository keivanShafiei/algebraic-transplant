"""tests/test_projection.py — Unit tests for Theorem 2 (Helmholtz Projection).

CRITICAL FIX (Phase 1, corrected): The fixture now correctly creates a mock G
that matches the structure of the real solver's G_int:
- G_int has shape (N_int, 2N) where N_int < N (only interior rows)
- The interior_mask marks which DOFs are interior (shape 2N,)
- The Cholesky factor is computed from G_int[:, interior_mask] @ G_int[:, interior_mask].T

Previously, the fixture used G_full (N x 2N) with arbitrary interior_mask,
which caused the Cholesky to be computed from the wrong matrix.
"""

import math
import pytest
import torch

from src.projection.layer import HelmholtzProjection


def build_mock_G_int(N: int, N_int: int) -> torch.Tensor:
    """Build a mock G_int (interior-restricted divergence operator).

    G_int has shape (N_int, 2N) where N_int is the number of interior nodes.
    This matches the real solver's G_int = G_full[is_int, :] where is_int
    is the boolean mask of interior nodes.

    For a 15x15 grid: N=225, boundary nodes ~56, N_int ~169.
    """
    torch.manual_seed(0)
    G = torch.randn(N_int, 2 * N, dtype=torch.float32)
    # Make columns somewhat correlated (like real divergence operator)
    G[:, 0::2] = G[:, 1::2] + 0.1 * torch.randn_like(G[:, 0::2])
    return G


def get_interior_mask(N: int) -> torch.Tensor:
    """Create interior DOF mask for a 15x15 grid.

    For a 15x15 grid with boundary nodes on perimeter:
    - Boundary DOFs: nodes on x=0, x=1, y=0, y=1
    - Interior DOFs: remaining nodes

    Simplified: mark first N DOFs as boundary, rest as interior.
    """
    mask = torch.zeros(2 * N, dtype=torch.bool)
    mask[N:] = True  # second half as interior
    return mask


@pytest.fixture
def G_and_proj():
    """Fixture providing G_int and HelmholtzProjection.

    CRITICAL FIX: Uses G_int (N_int x 2N) matching real solver.
    N_int = 169 for 15x15 grid (225 - 56 boundary nodes).
    """
    N = 15 * 15  # 225
    N_int = 169   # interior nodes for 15x15 grid

    G_int = build_mock_G_int(N, N_int)
    interior_mask = get_interior_mask(N)

    proj = HelmholtzProjection(G_int, eps=1e-8, interior_mask=interior_mask)
    return G_int, proj


def test_mass_conservation(G_and_proj):
    """Property 1 (Theorem 2): ||G_int a_NO|| ≈ 4e-5 (float32 floor, Remark 3)."""
    G_int, proj = G_and_proj
    torch.manual_seed(0)
    a_hat = torch.randn(G_int.shape[1], dtype=torch.float32) * 10.0

    a_NO = proj.project_only(a_hat)
    div_after = torch.norm(G_int @ a_NO).item()

    assert div_after < 5e-5, (
        f"ε_div = {div_after:.2e} exceeds float32 floor threshold 5e-5. "
        f"Expected ≈4e-5 (Remark 3, Table 8)."
    )


def test_divergence_reduction_ratio(G_and_proj):
    """Property 2: ρ = ||G_int a_hat|| / ||G_int a_NO|| ~ 10^4."""
    G_int, proj = G_and_proj
    torch.manual_seed(1)
    a_hat = torch.randn(G_int.shape[1], dtype=torch.float32) * 10.0

    r_before = torch.norm(G_int @ a_hat).item()
    a_NO = proj.project_only(a_hat)
    r_after = torch.norm(G_int @ a_NO).item()
    rho = r_before / (r_after + 1e-12)

    assert rho > 1e3, (
        f"Divergence reduction ratio ρ = {rho:.2e} is too small. "
        f"Expected > 1e3 (Table 8: ρ ~ 2.13e5)."
    )


def test_idempotency(G_and_proj):
    """Property 3: P_div(P_div(a)) = P_div(a) (projection is idempotent)."""
    G_int, proj = G_and_proj
    torch.manual_seed(2)
    a_hat = torch.randn(G_int.shape[1], dtype=torch.float32) * 10.0

    a1 = proj.project_only(a_hat)
    a2 = proj.project_only(a1)

    diff = torch.norm(a1 - a2).item()
    assert diff < 1e-5, (
        f"Idempotency violated: ||P(P(a)) - P(a)|| = {diff:.2e}. "
        f"Expected < 1e-5."
    )


def test_minimum_energy_pythagorean(G_and_proj):
    """Property 4: ||a_hat||^2 = ||a_NO||^2 + ||a_hat - a_NO||^2."""
    G_int, proj = G_and_proj
    torch.manual_seed(3)
    a_hat = torch.randn(G_int.shape[1], dtype=torch.float32) * 10.0

    a_NO = proj.project_only(a_hat)
    lhs = torch.norm(a_hat).item() ** 2
    rhs = torch.norm(a_NO).item() ** 2 + torch.norm(a_hat - a_NO).item() ** 2

    assert abs(lhs - rhs) < 1e-3 * lhs, (
        f"Pythagorean identity violated: {lhs:.4f} != {rhs:.4f}. "
        f"Relative error = {abs(lhs - rhs) / lhs:.2e}."
    )


def test_differentiability(G_and_proj):
    """Property 5: P_div is differentiable w.r.t. input.

    CRITICAL FIX: requires_grad must be set on the LEAF tensor.
    """
    G_int, proj = G_and_proj
    torch.manual_seed(4)

    a_hat = torch.randn(G_int.shape[1], dtype=torch.float32) * 10.0
    a_hat.requires_grad_(True)

    a_NO = proj.project_only(a_hat)
    a_NO.sum().backward()

    assert a_hat.grad is not None, (
        "Gradient did not flow back to a_hat. "
        "Make sure requires_grad is set on the leaf tensor."
    )

    assert a_hat.grad.norm().item() > 0, "Gradient norm is zero"


def test_finiteness_on_zero_input(G_and_proj):
    """Property 6: P_div(0) = 0 (no NaN/Inf on zero input)."""
    G_int, proj = G_and_proj
    a_zero = torch.zeros(G_int.shape[1], dtype=torch.float32)

    a_NO = proj.project_only(a_zero)

    assert torch.allclose(a_NO, a_zero, atol=1e-6), (
        f"P_div(0) != 0: max deviation = {(a_NO - a_zero).abs().max().item():.2e}"
    )
    assert not torch.isnan(a_NO).any(), "NaN detected in projection output."
    assert not torch.isinf(a_NO).any(), "Inf detected in projection output."
