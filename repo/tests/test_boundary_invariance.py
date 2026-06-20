"""Tests for Proposition 4: Boundary Invariance under Interior-Restricted Projection.

Proposition 4 states that for all boundary DOFs i ∈ I_∂:
    [P_div^int(â)]_i = â_i    (to machine precision)

This test verifies that the interior-restricted projection does NOT corrupt
prescribed Dirichlet boundary velocities.

Phase 1, Task 5: New test file for automated verification of Proposition 4.
"""

import torch
import pytest
from src.projection.layer import HelmholtzProjection


@pytest.fixture
def setup_boundary_invariance():
    """Load G_int and interior_mask for boundary invariance test."""
    G_int = torch.load('data/fixed_G.pt', map_location='cpu')
    interior_mask = torch.load('data/interior_mask.pt', map_location='cpu')

    # Create projection layer with interior mask
    proj = HelmholtzProjection(G_int, eps=1e-8, interior_mask=interior_mask)

    # Boundary mask is complement of interior_mask
    boundary_mask = ~interior_mask

    return proj, interior_mask, boundary_mask


def test_boundary_values_unchanged(setup_boundary_invariance):
    """Proposition 4: Boundary DOFs must be invariant under projection."""
    proj, interior_mask, boundary_mask = setup_boundary_invariance

    torch.manual_seed(42)
    a_hat = torch.randn(proj.G.shape[1], dtype=torch.float32) * 10.0

    # Apply projection
    a_NO = proj.project_only(a_hat)

    # Extract boundary values before and after
    a_hat_boundary = a_hat[boundary_mask]
    a_NO_boundary = a_NO[boundary_mask]

    # Boundary values should be unchanged (to float32 precision)
    diff = torch.abs(a_NO_boundary - a_hat_boundary).max().item()

    assert diff < 1e-6, (
        f"Boundary invariance violated! Max boundary change: {diff:.2e}. "
        f"Expected < 1e-6 (machine precision for float32). "
        f"This indicates projection is corrupting Dirichlet BCs."
    )


def test_interior_values_changed(setup_boundary_invariance):
    """Interior DOFs SHOULD change (projection is non-trivial)."""
    proj, interior_mask, boundary_mask = setup_boundary_invariance

    torch.manual_seed(43)
    a_hat = torch.randn(proj.G.shape[1], dtype=torch.float32) * 10.0

    a_NO = proj.project_only(a_hat)

    a_hat_interior = a_hat[interior_mask]
    a_NO_interior = a_NO[interior_mask]

    # Interior values should change (otherwise projection is identity)
    diff = torch.abs(a_NO_interior - a_hat_interior).max().item()

    assert diff > 1e-3, (
        f"Projection appears to be identity on interior! "
        f"Max interior change: {diff:.2e}. Expected > 1e-3."
    )


def test_divergence_zero_at_interior(setup_boundary_invariance):
    """G_int @ a_NO should be ~0 at interior nodes."""
    proj, interior_mask, boundary_mask = setup_boundary_invariance

    torch.manual_seed(44)
    a_hat = torch.randn(proj.G.shape[1], dtype=torch.float32) * 10.0

    a_NO = proj.project_only(a_hat)

    # Compute divergence at interior nodes
    div = proj.G @ a_NO
    div_norm = torch.norm(div).item()

    assert div_norm < 5e-5, (
        f"Divergence residual too large: {div_norm:.2e}. "
        f"Expected < 5e-5 (float32 precision floor)."
    )


def test_full_domain_vs_interior_projection():
    """Compare full-domain (G_full) vs interior-restricted (G_int) projection.

    This test demonstrates the 'Boundary Condition Paradox':
    - Full-domain projection corrupts boundary values (~74% drag error)
    - Interior-restricted projection preserves them (3.363e-5% drag error)
    """
    # This test requires G_full to be available
    try:
        G_full = torch.load('data/G_full.pt', map_location='cpu')
    except FileNotFoundError:
        pytest.skip("G_full.pt not found. Run generate_data.py first.")

    G_int = torch.load('data/fixed_G.pt', map_location='cpu')
    interior_mask = torch.load('data/interior_mask.pt', map_location='cpu')

    torch.manual_seed(45)
    a_hat = torch.randn(G_full.shape[1], dtype=torch.float32) * 10.0

    # Full-domain projection (WRONG - corrupts boundaries)
    proj_full = HelmholtzProjection(G_full, eps=1e-8, interior_mask=None)
    a_NO_full = proj_full.project_only(a_hat)

    # Interior-restricted projection (CORRECT - preserves boundaries)
    proj_int = HelmholtzProjection(G_int, eps=1e-8, interior_mask=interior_mask)
    a_NO_int = proj_int.project_only(a_hat)

    # Boundary change: full-domain should be large, interior should be tiny
    boundary_mask = ~interior_mask
    change_full = torch.abs(a_NO_full[boundary_mask] - a_hat[boundary_mask]).max().item()
    change_int = torch.abs(a_NO_int[boundary_mask] - a_hat[boundary_mask]).max().item()

    assert change_full > 1e-3, "Full-domain projection should change boundaries"
    assert change_int < 1e-6, "Interior-restricted projection should preserve boundaries"

    print(f"\nBoundary change — Full-domain: {change_full:.2e}, "
          f"Interior-restricted: {change_int:.2e}")
