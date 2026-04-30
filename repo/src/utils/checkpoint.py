"""utils/checkpoint.py — Resolution-Aware Checkpoint Utilities.

This module provides utilities for saving and loading model checkpoints
with full resolution metadata, enabling:
    1. Multi-resolution training with proper tracking
    2. Zero-shot inference on unseen resolutions
    3. Backward compatibility with existing checkpoints

Paper reference
---------------
Section 4.5 (Resolution-Invariant Transfer with Scale-Adaptive Edge Encoding).
"""

from __future__ import annotations
import os
import torch
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    loss: float,
    config: Dict[str, Any],
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a model checkpoint with full resolution metadata.

    Parameters
    ----------
    model : torch.nn.Module
        The model to save (NeuralOperator or wrapper).
    optimizer : torch.optim.Optimizer or None
        Training optimizer state.
    scheduler : _LRScheduler or None
        Learning rate scheduler state.
    epoch : int
        Current training epoch.
    loss : float
        Current best/validation loss.
    config : dict
        Full training configuration (from config.yaml).
    path : str
        Output checkpoint path (.pt file).
    metadata : dict, optional
        Additional metadata to store (e.g., training resolution, data stats).

    Saved checkpoint structure:
    {
        'model_state_dict': {...},
        'optimizer_state_dict': {...} or None,
        'scheduler_state_dict': {...} or None,
        'epoch': int,
        'loss': float,
        'config': {...},  # Full config at training time
        'metadata': {
            'training_n_nodes': int,
            'training_h_avg': float,
            'stencil_k': int,
            'projection_eps': float,
            ...  # from metadata parameter
        }
    }
    """
    # Extract resolution info from model if available
    training_metadata = {}
    
    if hasattr(model, 'n_nodes'):
        training_metadata['training_n_nodes'] = model.n_nodes
    
    if hasattr(model, 'points') and model.points.numel() > 0:
        # Compute h_avg from stored points/stencils if available
        if hasattr(model, '_stencils') and model._stencils is not None:
            pts = model.points.float()
            neigh = pts[model._stencils.long()]
            center = pts.unsqueeze(1).expand_as(neigh)
            h_avg = torch.norm(neigh - center, dim=-1).mean().item()
            training_metadata['training_h_avg'] = h_avg
    
    if hasattr(model, 'k'):
        training_metadata['stencil_k'] = model.k
    
    if hasattr(model, 'eps'):
        training_metadata['projection_eps'] = model.eps
    
    # Merge with user-provided metadata
    if metadata:
        training_metadata.update(metadata)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'loss': loss,
        'config': config,
        'metadata': training_metadata,
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path}")
    print(f"    Epoch: {epoch}, Loss: {loss:.4e}")
    print(f"    Training resolution: N={training_metadata.get('training_n_nodes', 'N/A')}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Load a checkpoint and optionally restore optimizer/scheduler states.

    Parameters
    ----------
    path : str
        Checkpoint file path.
    model : torch.nn.Module
        Model to load weights into.
    optimizer : torch.optim.Optimizer or None
        Optimizer to restore state (if provided).
    scheduler : _LRScheduler or None
        Scheduler to restore state (if provided).
    device : torch.device or None
        Device to load tensors onto (default: model's device).
    strict : bool, optional
        If True, require exact match of state_dict keys.
        If False, allow partial loading (useful for transfer learning).

    Returns
    -------
    dict
        Full checkpoint data including metadata:
        {
            'epoch': int,
            'loss': float,
            'config': dict,
            'metadata': dict,
            'training_n_nodes': int or None,
            'training_h_avg': float or None,
        }

    Notes
    -----
    For zero-shot resolution transfer:
        1. Load checkpoint with strict=False
        2. Call model.set_scales(h_train=metadata['training_h_avg'], h_infer=new_h_avg)
        3. Call model.set_points(new_points, new_stencils)
        4. Call model.set_projection(new_G)
    """
    if device is None:
        device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')
    
    checkpoint = torch.load(path, map_location=device)
    
    # Load model state dict
    model_state = checkpoint['model_state_dict']
    
    # Handle backward compatibility: old checkpoints may not have all keys
    if strict:
        model.load_state_dict(model_state, strict=True)
    else:
        # Filter out keys that don't match (for resolution transfer)
        model_dict = model.state_dict()
        
        # Identify mismatched keys (mainly projection-related)
        filtered_state = {}
        skipped_keys = []
        for k, v in model_state.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                filtered_state[k] = v
            else:
                skipped_keys.append(k)
        
        if skipped_keys and len(skipped_keys) < 10:
            print(f"  Note: Skipped {len(skipped_keys)} incompatible keys: {skipped_keys[:5]}...")
        elif skipped_keys:
            print(f"  Note: Skipped {len(skipped_keys)} incompatible keys (resolution mismatch expected)")
        
        model.load_state_dict(filtered_state, strict=False)
    
    # Restore optimizer and scheduler if provided
    if optimizer and checkpoint.get('optimizer_state_dict'):
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except (KeyError, ValueError) as e:
            print(f"  Warning: Could not restore optimizer state: {e}")
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except (KeyError, ValueError) as e:
            print(f"  Warning: Could not restore scheduler state: {e}")
    
    # Extract metadata for convenience
    metadata = checkpoint.get('metadata', {})
    result = {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'config': checkpoint.get('config', {}),
        'metadata': metadata,
        'training_n_nodes': metadata.get('training_n_nodes'),
        'training_h_avg': metadata.get('training_h_avg'),
    }
    
    print(f"  Checkpoint loaded: {path}")
    print(f"    Epoch: {result['epoch']}, Loss: {result['loss']:.4e}")
    if result['training_n_nodes']:
        print(f"    Trained on N={result['training_n_nodes']}")
    
    return result


def get_checkpoint_info(path: str) -> Dict[str, Any]:
    """Get checkpoint metadata without loading the full checkpoint.

    Useful for inspecting available checkpoints before loading.

    Parameters
    ----------
    path : str
        Checkpoint file path.

    Returns
    -------
    dict
        Checkpoint metadata including training resolution and config.
    """
    checkpoint = torch.load(path, map_location='cpu')
    metadata = checkpoint.get('metadata', {})
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'training_n_nodes': metadata.get('training_n_nodes'),
        'training_h_avg': metadata.get('training_h_avg'),
        'stencil_k': metadata.get('stencil_k'),
        'config_keys': list(checkpoint.get('config', {}).keys()),
    }


def find_best_checkpoint(
    checkpoint_dir: str = 'results',
    pattern: str = 'model_*.pt',
) -> Optional[str]:
    """Find the checkpoint with the lowest loss in a directory.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing checkpoint files.
    pattern : str
        Glob pattern for checkpoint files.

    Returns
    -------
    str or None
        Path to best checkpoint, or None if no checkpoints found.
    """
    from glob import glob
    
    checkpoints = glob(os.path.join(checkpoint_dir, pattern))
    if not checkpoints:
        return None
    
    best_ckpt = None
    best_loss = float('inf')
    
    for ckpt_path in checkpoints:
        try:
            info = get_checkpoint_info(ckpt_path)
            if info['loss'] < best_loss:
                best_loss = info['loss']
                best_ckpt = ckpt_path
        except Exception:
            continue
    
    return best_ckpt
