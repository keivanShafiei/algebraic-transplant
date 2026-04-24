"""Parametric cavity dataset utilities.

Note: ParametricCavityDataset is superseded by PrecomputedDataset in
scripts/train.py (fix T1). It is retained here for reference and for use
in lightweight tests that only need Re values (not solver outputs).
"""

import torch
from torch.utils.data import Dataset


class ParametricCavityDataset(Dataset):
    """Re parameter sampler — does NOT contain ground-truth flow fields.

    Suitable only for: listing Re values, building edge_index, lightweight tests.
    For training, use PrecomputedDataset in scripts/train.py which loads
    pre-generated (a_ref, b_ref) solver outputs (Eq. 23).

    D2 fix: added split parameter for consistent train/val/test splits.
      Paper Table 8: Ntest=80 implies 80/20 split on Ns=400 samples.
    D3 note: mu is returned as normalised Re/Re_max ∈ [0,1] for training
      stability; raw Re is also returned for logging.
    """

    RE_MIN: float = 10.0
    RE_MAX: float = 100.0

    def __init__(self, ns: int = 400, split: str = 'all',
                 val_frac: float = 0.2, seed: int = 42):
        """
        Args:
            ns       : total number of Re samples
            split    : 'all' | 'train' | 'val'
            val_frac : fraction reserved for validation (default 0.2 → 80 samples)
            seed     : RNG seed for reproducible split
        """
        re_all = torch.linspace(self.RE_MIN, self.RE_MAX, ns, dtype=torch.float32)

        # D2 fix: reproducible train/val split
        rng   = torch.Generator().manual_seed(seed)
        perm  = torch.randperm(ns, generator=rng)
        n_val = int(ns * val_frac)

        if split == 'train':
            idx = perm[n_val:]
        elif split == 'val':
            idx = perm[:n_val]
        else:
            idx = perm   # 'all'

        self.re = re_all[idx]
        # D3 fix: normalise to [0,1] for embed layer stability
        self.mu = (self.re / self.RE_MAX).unsqueeze(1)   # (M, 1)

    def __len__(self) -> int:
        return len(self.re)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu_normalised, re_raw)."""
        return self.mu[idx], self.re[idx]

# ──────────────────────────────────────────────────────────────
# PrecomputedDataset — used by train.py and train_baselines.py
# ──────────────────────────────────────────────────────────────

from torch.utils.data import Dataset
import os

class PrecomputedDataset(Dataset):
    """Loads pre-generated RBF-FD samples (a_ref, b_ref, mu)."""
    def __init__(self, sample_dir: str = 'data/samples'):
        self.files = sorted([
            os.path.join(sample_dir, f)
            for f in os.listdir(sample_dir) if f.endswith('.pt')
        ])
        if len(self.files) == 0:
            raise FileNotFoundError("No .pt files found. Run generate_data.py first.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location='cpu')
        return (
            data['mu'],
            data['a_ref'], data['a_mean'], data['a_std'],
            data['b_ref'], data['b_mean'], data['b_std']
        )
