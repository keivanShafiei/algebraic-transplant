import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierNeuralOperator(nn.Module):
    """Simple 2D FNO adapted to uniform 15×15 cavity grid (N=225)."""
    def __init__(self, modes=12, hidden=32, layers=4):
        super().__init__()
        self.modes = modes
        self.hidden = hidden
        self.layers = layers
        self.fc0 = nn.Linear(3, hidden)          # (x,y,mu)
        self.convs = nn.ModuleList([nn.Conv2d(hidden, hidden, 1) for _ in range(layers)])
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, 2)          # u,v per node

    def forward(self, mu, points):
        # points: (N,2) → grid 15×15
        grid_size = int(points.shape[0]**0.5)
        mu = mu.reshape(1, 1, 1, 1).expand(1, grid_size, grid_size, 1)
        x = torch.cat([points.reshape(grid_size, grid_size, 2), mu.squeeze(0)], dim=-1)
        x = self.fc0(x)                          # (15,15,hidden)
        x = x.permute(2, 0, 1).unsqueeze(0)     # (1, hidden, 15, 15)

        for conv in self.convs:
            x = F.relu(conv(x))

        x = x.squeeze(0).permute(1, 2, 0)       # (15,15,hidden)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)                          # (15,15,2)
        a = x.reshape(-1)                        # (2N,)
        return a, a  # (a_hat, a_NO) for compatibility with eval
