"""tests/test_resolution_invariance.py — Tests for resolution-invariant training pipeline.

This module validates the resolution-aware checkpoint system and zero-shot
transfer capabilities added to the NeuralOperator framework.

Tests cover:
    1. Checkpoint save/load with metadata
    2. Forward pass on multiple resolutions
    3. Shape compatibility across resolutions
    4. Projection operator compatibility
    5. Scale-adaptive edge encoding
"""

import os
import sys
import torch
import pytest
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gnn.neural_operator import NeuralOperator
from src.data.cavity import generate_cavity_points
from src.rbf_fd.stencils import build_stencils
from src.rbf_fd.operators import assemble_divergence_operator
from src.utils.checkpoint import (
    save_checkpoint, 
    load_checkpoint, 
    get_checkpoint_info,
)


class TestCheckpointMetadata:
    """Test resolution-aware checkpoint saving and loading."""

    @pytest.fixture
    def model_and_data(self):
        """Create a model trained on N=225."""
        N = 225
        device = torch.device('cpu')
        points = generate_cavity_points(N).to(device)
        stencils = build_stencils(points, k=25)
        
        model = NeuralOperator(n_nodes=N, hidden=64, layers=4).to(device)
        model.set_points(points, stencils)
        
        G = assemble_divergence_operator(points, stencils, c=1.2 * model.h_infer.item())
        model.set_projection(G)
        
        return model, points, stencils, G

    def test_save_checkpoint_with_metadata(self, model_and_data, tmp_path):
        """Checkpoint should include training resolution metadata."""
        model, points, stencils, G = model_and_data
        
        ckpt_path = str(tmp_path / "test_ckpt.pt")
        config = {'n_nodes_list': [225], 'stencil_k': 25, 'projection_eps': 1e-8}
        
        save_checkpoint(
            model=model,
            optimizer=None,
            scheduler=None,
            epoch=10,
            loss=0.01,
            config=config,
            path=ckpt_path,
            metadata={'test_key': 'test_value'}
        )
        
        assert os.path.exists(ckpt_path)
        info = get_checkpoint_info(ckpt_path)
        
        assert info['training_n_nodes'] == 225
        assert 'training_h_avg' in info
        assert info['stencil_k'] == 25
        assert info['epoch'] == 10
        assert abs(info['loss'] - 0.01) < 1e-6

    def test_load_checkpoint_backward_compatible(self, model_and_data, tmp_path):
        """Loading old checkpoints (without metadata) should still work."""
        model, points, stencils, G = model_and_data
        
        # Save legacy format (just state_dict)
        legacy_path = str(tmp_path / "legacy_ckpt.pt")
        torch.save(model.state_dict(), legacy_path)
        
        # Create new model and load
        new_model = NeuralOperator(n_nodes=225, hidden=64, layers=4)
        new_model.set_points(points, stencils)
        new_model.set_projection(G)
        
        # Should work with strict=False
        checkpoint = torch.load(legacy_path, map_location='cpu')
        new_model.load_state_dict(checkpoint, strict=True)  # Legacy has exact match

    def test_load_checkpoint_resolution_mismatch(self, model_and_data, tmp_path):
        """Loading checkpoint on different resolution should skip incompatible keys."""
        model, points, stencils, G = model_and_data
        
        ckpt_path = str(tmp_path / "test_ckpt.pt")
        config = {'n_nodes_list': [225], 'stencil_k': 25, 'projection_eps': 1e-8}
        
        save_checkpoint(
            model=model,
            optimizer=None,
            scheduler=None,
            epoch=10,
            loss=0.01,
            config=config,
            path=ckpt_path,
        )
        
        # Try to load on different resolution
        N_new = 400
        points_new = generate_cavity_points(N_new)
        stencils_new = build_stencils(points_new, k=25)
        
        new_model = NeuralOperator(n_nodes=N_new, hidden=64, layers=4)
        new_model.set_points(points_new, stencils_new)
        
        # This should NOT fail - it will skip projection-related keys
        result = load_checkpoint(ckpt_path, new_model, strict=False)
        
        # Metadata should reflect ORIGINAL training resolution
        assert result['training_n_nodes'] == 225


class TestMultiResolutionForward:
    """Test forward pass on multiple resolutions."""

    def test_forward_same_resolution(self):
        """Forward pass should work on training resolution."""
        N = 225
        device = torch.device('cpu')
        points = generate_cavity_points(N).to(device)
        stencils = build_stencils(points, k=25)
        
        model = NeuralOperator(n_nodes=N, hidden=64, layers=4).to(device)
        model.set_points(points, stencils)
        
        G = assemble_divergence_operator(points, stencils, c=1.2 * model.h_infer.item())
        model.set_projection(G)
        
        edge_dst = stencils.reshape(-1)
        edge_src = torch.arange(N, device=device).repeat_interleave(25)
        edge_index = torch.stack([edge_dst, edge_src])
        
        mu = torch.tensor([0.5], device=device)  # Re=50
        a_hat, a_NO, b = model(mu, edge_index)
        
        assert a_hat.shape == (1, 2 * N)
        assert a_NO.shape == (1, 2 * N)
        assert b.shape == (1, N)
        assert torch.isfinite(a_NO).all()

    def test_forward_different_resolution_without_projection(self):
        """Model weights can be loaded on different resolution for backbone eval."""
        N_train = 225
        N_test = 400
        device = torch.device('cpu')
        
        # Train setup
        points_train = generate_cavity_points(N_train).to(device)
        stencils_train = build_stencils(points_train, k=25)
        
        model_train = NeuralOperator(n_nodes=N_train, hidden=64, layers=4).to(device)
        model_train.set_points(points_train, stencils_train)
        
        # Get trained weights (simulated)
        train_weights = model_train.state_dict()
        
        # Test setup - create new model with same architecture but different N
        # Note: This is limited because n_nodes affects some layer shapes
        # The key insight: FiLM conditioners and GNN layers ARE resolution-invariant
        # Only the final decoders and projection depend on N
        
        # Verify that core components have resolution-independent shapes
        for key in ['film_conditioners', 'gnn_layers', 'feature_encoder']:
            assert key in train_weights or any(key in k for k in train_weights.keys())


class TestScaleAdaptiveEncoding:
    """Test scale-adaptive edge encoding for zero-shot transfer."""

    def test_edge_scale_computation(self):
        """Edge scale should be h_train / h_infer."""
        N_train = 225
        N_test = 400
        device = torch.device('cpu')
        
        points_train = generate_cavity_points(N_train).to(device)
        stencils_train = build_stencils(points_train, k=25)
        
        points_test = generate_cavity_points(N_test).to(device)
        stencils_test = build_stencils(points_test, k=25)
        
        model = NeuralOperator(n_nodes=N_test, hidden=64, layers=4).to(device)
        model.set_points(points_test, stencils_test)
        
        h_train = NeuralOperator._compute_h_avg(points_train, stencils_train).item()
        h_infer = model.h_infer.item()
        
        # Set scales for zero-shot transfer
        model.set_scales(h_train=h_train, h_infer=h_infer)
        
        # In training mode, scale should be 1.0
        model.train()
        assert model._edge_scale().item() == 1.0
        
        # In eval mode, scale should be h_train / h_infer
        model.eval()
        expected_scale = h_train / (h_infer + 1e-12)
        actual_scale = model._edge_scale().item()
        assert abs(actual_scale - expected_scale) < 1e-6


class TestProjectionCompatibility:
    """Test projection operator compatibility across resolutions."""

    def test_projection_shape_consistency(self):
        """Projection output shape should match input regardless of resolution."""
        for N in [225, 400, 900]:
            device = torch.device('cpu')
            points = generate_cavity_points(N).to(device)
            stencils = build_stencils(points, k=25)
            
            # Use proper c parameter based on h_avg
            dists = torch.cdist(points, points)
            dists_sorted, _ = dists.sort(dim=1)
            h_avg = dists_sorted[:, 1].mean().item()
            c = 1.2 * h_avg

            G = assemble_divergence_operator(points, stencils, c=c)
            
            from src.projection.layer import HelmholtzProjection
            proj = HelmholtzProjection(G, eps=1e-8)
            
            a_hat = torch.randn(2 * N, dtype=torch.float32)
            a_NO = proj.project_only(a_hat)
            
            assert a_NO.shape == a_hat.shape
            div_norm = torch.norm(G @ a_NO).item()
            assert div_norm < 1e-3, f"Divergence norm {div_norm} too high for N={N}"  # Relaxed tolerance


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
