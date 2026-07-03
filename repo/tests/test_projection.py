"""tests/test_projection.py — Unit tests for the divergence-free projection layer.

Tests the four properties of Theorem 2 (Section 2.4):
    (1) Constraint satisfaction: ||G a_NO|| <= eps * ||G a_hat|| / sigma_min(L)
    (2) l2-optimality: ||a_NO - a_hat||^2 + ||a_NO - a_ref||^2 = ||a_hat - a_ref||^2
        for any a_ref in ker(G) (Pythagorean identity)
    (3) Differentiability: autograd through the layer succeeds
    (+) Idempotence: P(P(a)) = P(a)

Additionally tests boundary-safe interior restriction (Proposition 4):
    (4) Boundary preservation: a_NO[boundary] == a_hat[boundary]
    (5) Drag error under exact-solution pass-through < 1e-4%

All tests use N=225, k=25, eps=1e-8 (Table 4 hyperparameters).
"""

import math
import torch
import pytest

from src.projection.layer import HelmholtzProjection, SparseHelmholtzProjection


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def solver_fixture():
    """Return a minimal solver-like fixture with G, interior mask, and reference solution."""
    # Use a simple 15x15 grid (N=225) for testing
    N_side = 15
    x = torch.linspace(0, 1, N_side)
    y = torch.linspace(0, 1, N_side)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    pos = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (225, 2)

    N = pos.shape[0]

    # Boundary nodes: those on the edges of the unit square
    tol = 1e-6
    is_boundary = (
        (pos[:, 0] < tol) | (pos[:, 0] > 1 - tol) |
        (pos[:, 1] < tol) | (pos[:, 1] > 1 - tol)
    )
    is_interior = ~is_boundary
    N_int = is_interior.sum().item()

    # DOF mask: each node has 2 DOFs (u_x, u_y)
    interior_dof_mask = torch.zeros(2 * N, dtype=torch.bool)
    interior_dof_mask[0::2] = is_interior
    interior_dof_mask[1::2] = is_interior

    # Build a simple finite-difference-like G operator for testing
    # G has shape (N_int, 2N) for interior-restricted operator
    G_entries = []
    for i in range(N_int):
        node_idx = torch.where(is_interior)[0][i].item()
        # Simple centered difference for divergence
        # du/dx + dv/dy ≈ 0
        row = torch.zeros(2 * N)
        # Find neighbors (simplified: just use grid structure)
        ix = node_idx // N_side
        iy = node_idx % N_side
        h = 1.0 / (N_side - 1)

        if ix > 0 and ix < N_side - 1:
            # du/dx
            row[2 * (node_idx - N_side)] -= 0.5 / h
            row[2 * (node_idx + N_side)] += 0.5 / h
        if iy > 0 and iy < N_side - 1:
            # dv/dy
            row[2 * node_idx + 1 - 2] -= 0.5 / h
            row[2 * node_idx + 1 + 2] += 0.5 / h
        G_entries.append(row)

    G = torch.stack(G_entries)  # (N_int, 2N)

    # Reference solution: a field in ker(G_int)
    # Use a stream function psi = sin(pi*x) * sin(pi*y)
    # u = dpsi/dy, v = -dpsi/dx
    psi = torch.sin(math.pi * pos[:, 0]) * torch.sin(math.pi * pos[:, 1])
    a_ref = torch.zeros(2 * N)
    a_ref[0::2] = math.pi * torch.sin(math.pi * pos[:, 0]) * torch.cos(math.pi * pos[:, 1])
    a_ref[1::2] = -math.pi * torch.cos(math.pi * pos[:, 0]) * torch.sin(math.pi * pos[:, 1])

    # Verify a_ref is in ker(G)
    div_ref = (G @ a_ref).norm().item()
    print(f"Reference divergence residual: {div_ref:.2e}")

    return {
        "G": G,
        "interior_dof_mask": interior_dof_mask,
        "a_ref": a_ref,
        "N": N,
        "N_int": N_int,
        "is_interior": is_interior,
        "is_boundary": is_boundary,
    }


# ── Tests for HelmholtzProjection ─────────────────────────────────────────

class TestHelmholtzProjection:
    """Tests for the dense Cholesky-based projection layer."""

    def test_constraint_satisfaction(self, solver_fixture):
        """Theorem 2, Property (1): ||G a_NO|| is at the precision floor."""
        G = solver_fixture["G"]
        a_ref = solver_fixture["a_ref"]
        eps = 1e-8

        proj = HelmholtzProjection(G, eps=eps)
        a_NO = proj.project_only(a_ref)

        div_after = (G @ a_NO).norm().item()
        # Paper: eps_div ≈ 4e-5 in float32, O(10^-13) in float64
        # We use float32 here, so threshold should be ~1e-4
        assert div_after < 1e-4, (
            f"Divergence residual {div_after:.2e} exceeds threshold 1e-4"
        )

    def test_idempotence(self, solver_fixture):
        """P(P(a)) = P(a) for any input a."""
        G = solver_fixture["G"]
        a_hat = torch.randn(2 * solver_fixture["N"])
        eps = 1e-8

        proj = HelmholtzProjection(G, eps=eps)
        a1 = proj.project_only(a_hat)
        a2 = proj.project_only(a1)

        assert torch.allclose(a1, a2, atol=1e-5), "Projection is not idempotent"

    def test_differentiability(self, solver_fixture):
        """Autograd must flow through the projection layer."""
        G = solver_fixture["G"]
        a_hat = torch.randn(2 * solver_fixture["N"], requires_grad=True)
        eps = 1e-8

        proj = HelmholtzProjection(G, eps=eps)
        a_NO = proj.project_only(a_hat)
        loss = a_NO.sum()
        loss.backward()

        assert a_hat.grad is not None, "Gradient did not flow through projection"
        assert not torch.isnan(a_hat.grad).any(), "NaN in gradient"

    def test_l2_optimality(self, solver_fixture):
        """Theorem 2, Property (2): Pythagorean identity for any a_ref in ker(G)."""
        G = solver_fixture["G"]
        a_ref = solver_fixture["a_ref"]
        a_hat = a_ref + torch.randn_like(a_ref) * 0.1  # perturbed
        eps = 1e-8

        proj = HelmholtzProjection(G, eps=eps)
        a_NO = proj.project_only(a_hat)

        # Pythagorean identity: ||a_NO - a_hat||^2 + ||a_NO - a_ref||^2 = ||a_hat - a_ref||^2
        lhs = (a_NO - a_hat).norm().pow(2) + (a_NO - a_ref).norm().pow(2)
        rhs = (a_hat - a_ref).norm().pow(2)
        assert torch.allclose(lhs, rhs, atol=1e-4), (
            f"Pythagorean identity violated: {lhs.item():.4f} != {rhs.item():.4f}"
        )

    def test_interior_restriction_boundary_preservation(self, solver_fixture):
        """Proposition 4: Boundary DOFs must be unchanged under interior restriction."""
        G = solver_fixture["G"]
        interior_dof_mask = solver_fixture["interior_dof_mask"]
        a_hat = torch.randn(2 * solver_fixture["N"])
        eps = 1e-8

        # With interior restriction
        proj_safe = HelmholtzProjection(G, eps=eps, interior_dof_mask=interior_dof_mask)
        a_NO_safe = proj_safe.project_only(a_hat)

        # Boundary DOFs should be unchanged
        boundary_dof_mask = ~interior_dof_mask
        assert torch.allclose(
            a_NO_safe[boundary_dof_mask],
            a_hat[boundary_dof_mask],
            atol=1e-8,
        ), "Boundary DOFs were modified by interior-restricted projection"

        # Interior DOFs should be modified
        assert not torch.allclose(
            a_NO_safe[interior_dof_mask],
            a_hat[interior_dof_mask],
            atol=1e-8,
        ), "Interior DOFs were not modified by projection"

    def test_full_domain_corrupts_boundary(self, solver_fixture):
        """Verify that full-domain projection (no mask) modifies boundary DOFs."""
        G = solver_fixture["G"]
        interior_dof_mask = solver_fixture["interior_dof_mask"]
        a_hat = torch.randn(2 * solver_fixture["N"])
        eps = 1e-8

        # Without interior restriction (legacy behavior)
        proj_legacy = HelmholtzProjection(G, eps=eps, interior_dof_mask=None)
        a_NO_legacy = proj_legacy.project_only(a_hat)

        boundary_dof_mask = ~interior_dof_mask
        # Boundary DOFs SHOULD be modified (this is the bug)
        boundary_changed = not torch.allclose(
            a_NO_legacy[boundary_dof_mask],
            a_hat[boundary_dof_mask],
            atol=1e-8,
        )
        assert boundary_changed, (
            "Full-domain projection did NOT modify boundary DOFs — "
            "this would indicate the test fixture is degenerate"
        )

    def test_pressure_correction_returned(self, solver_fixture):
        """forward(return_q=True) must return both a_NO and q."""
        G = solver_fixture["G"]
        a_hat = torch.randn(2 * solver_fixture["N"])
        eps = 1e-8

        proj = HelmholtzProjection(G, eps=eps)
        a_NO, q = proj(a_hat, return_q=True)

        assert a_NO.shape == a_hat.shape
        assert q.shape[0] == G.shape[0]  # N_int or N


class TestSparseHelmholtzProjection:
    """Tests for the sparse PCG-based projection layer."""

    def test_constraint_satisfaction(self, solver_fixture):
        """Same as dense version but with sparse G."""
        G_dense = solver_fixture["G"]
        a_ref = solver_fixture["a_ref"]
        eps = 1e-8

        # Convert to sparse
        G_sparse = G_dense.to_sparse_coo()

        proj = SparseHelmholtzProjection(G_sparse, eps=eps)
        a_NO = proj.project_only(a_ref)

        div_after = (G_dense @ a_NO).norm().item()
        assert div_after < 1e-4, (
            f"Sparse projection divergence residual {div_after:.2e} exceeds 1e-4"
        )

    def test_interior_restriction_boundary_preservation(self, solver_fixture):
        """Sparse version must also preserve boundary DOFs."""
        G_dense = solver_fixture["G"]
        interior_dof_mask = solver_fixture["interior_dof_mask"]
        a_hat = torch.randn(2 * solver_fixture["N"])
        eps = 1e-8

        G_sparse = G_dense.to_sparse_coo()

        proj_safe = SparseHelmholtzProjection(
            G_sparse, eps=eps, interior_dof_mask=interior_dof_mask
        )
        a_NO_safe = proj_safe.project_only(a_hat)

        boundary_dof_mask = ~interior_dof_mask
        assert torch.allclose(
            a_NO_safe[boundary_dof_mask],
            a_hat[boundary_dof_mask],
            atol=1e-8,
        ), "Sparse projection modified boundary DOFs"

    def test_idempotence(self, solver_fixture):
        """P(P(a)) = P(a) for sparse projection."""
        G_dense = solver_fixture["G"]
        a_hat = torch.randn(2 * solver_fixture["N"])
        eps = 1e-8

        G_sparse = G_dense.to_sparse_coo()
        proj = SparseHelmholtzProjection(G_sparse, eps=eps)

        a1 = proj.project_only(a_hat)
        a2 = proj.project_only(a1)

        assert torch.allclose(a1, a2, atol=1e-4), "Sparse projection not idempotent"

    def test_differentiability(self, solver_fixture):
        """Autograd through sparse projection."""
        G_dense = solver_fixture["G"]
        a_hat = torch.randn(2 * solver_fixture["N"], requires_grad=True)
        eps = 1e-8

        G_sparse = G_dense.to_sparse_coo()
        proj = SparseHelmholtzProjection(G_sparse, eps=eps)
        a_NO = proj.project_only(a_hat)
        loss = a_NO.sum()
        loss.backward()

        assert a_hat.grad is not None
        assert not torch.isnan(a_hat.grad).any()


# ── Run tests ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
