import torch
import torch.nn as nn

class DeepONet(nn.Module):
    """DeepONet: branch (mu) + trunk (points)."""
    def __init__(self, hidden=64, layers=3):
        super().__init__()
        # Branch: parameter embedding
        branch_layers = [nn.Linear(1, hidden), nn.ReLU()]
        for _ in range(layers-1):
            branch_layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        self.branch = nn.Sequential(*branch_layers)

        # Trunk: position encoding
        trunk_layers = [nn.Linear(2, hidden), nn.ReLU()]
        for _ in range(layers-1):
            trunk_layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        self.trunk = nn.Sequential(*trunk_layers)

        self.decoder = nn.Linear(hidden, 2)   # u,v

    def forward(self, mu, points):
        b = self.branch(mu)                   # (hidden,)
        t = self.trunk(points)                # (N, hidden)
        out = (b * t).unsqueeze(0)            # (1, N, hidden) * broadcast
        a = self.decoder(out.squeeze(0)).reshape(-1)  # (2N,)
        return a, a
