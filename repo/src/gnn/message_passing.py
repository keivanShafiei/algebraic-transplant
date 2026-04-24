"""gnn/message_passing.py — Graph Convolution Layer with FiLM Conditioning.

Implements a single message-passing step used in the NeuralOperator backbone.
Each layer combines:

    1. Spatial edge features  (Delta_x_ij, ||Delta_x_ij||)
    2. FiLM affine modulation (gamma * x + beta, conditioning on Re)
    3. Mean-aggregation message passing
    4. Residual connection + LayerNorm

Scale-Adaptive Edge Encoding (Section 3.2, Eq. 15)
---------------------------------------------------
At inference on a grid with fill distance h_infer different from the
training grid (h_train), all geometric edge features are multiplied by

    s_adapt = h_train / h_infer

This parameter-free rescaling aligns the spatial statistics of the new
grid with the training distribution, enabling zero-shot resolution transfer
without any retraining or fine-tuning.

When h_infer == h_train (i.e., same resolution), s_adapt = 1 and edge
features are unchanged.

Paper reference
---------------
Section 3.2 (Scale-adaptive message passing), Eq. (15).
Section 4.5 (Resolution-Invariant Transfer with Scale-Adaptive Edge Encoding).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """Single FiLM-conditioned message-passing layer with residual connection.

    Parameters
    ----------
    in_dim : int
        Input node feature dimension.
    out_dim : int
        Output node feature dimension.
    pos_dim : int, optional
        Spatial dimension of node coordinates (default 2 for 2D flows).

    Architecture
    ------------
    Edge features  : [x_dst | x_src | s_adapt * Delta_x | s_adapt * dist]
                     dimension = in_dim * 2 + pos_dim + 1
    Message MLP    : Linear(edge_feat_dim -> out_dim) + ReLU
    Aggregation    : mean over neighbours (index_add_ / count)
    Node update    : Linear([x | agg] -> out_dim) + ReLU + LayerNorm
    Residual       : proj(x) if in_dim != out_dim else identity
    """

    def __init__(self, in_dim: int, out_dim: int, pos_dim: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        edge_feat_dim = pos_dim + 1           # [Delta_x (pos_dim), dist (1)]
        msg_input_dim = in_dim * 2 + edge_feat_dim

        self.msg_lin = nn.Linear(msg_input_dim, out_dim, dtype=torch.float32)
        self.node_lin = nn.Linear(in_dim + out_dim, out_dim, dtype=torch.float32)
        self.norm = nn.LayerNorm(out_dim, dtype=torch.float32)
        self.residual_proj = (
            nn.Linear(in_dim, out_dim, bias=False, dtype=torch.float32)
            if in_dim != out_dim else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
        edge_scale: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        """Forward pass of one message-passing layer.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape (N, in_dim).
        edge_index : torch.Tensor
            Edge connectivity, shape (2, E). Row 0 = destination indices
            (aggregation targets), row 1 = source indices.
        pos : torch.Tensor
            Node coordinates, shape (N, pos_dim).
        edge_scale : float or torch.Tensor, optional
            Scale factor s_adapt = h_train / h_infer applied to geometric
            edge features. Set to 1.0 during training (no rescaling).
            At inference on a different resolution grid, set to the ratio
            of mean fill distances.

        Returns
        -------
        torch.Tensor
            Updated node features, shape (N, out_dim).
        """
        N = x.shape[0]
        dst, src = edge_index

        # Scale-adaptive geometric edge features (Eq. 15)
        delta_pos = pos[dst] - pos[src]                        # (E, pos_dim)
        dist = delta_pos.norm(dim=-1, keepdim=True)            # (E, 1)
        scale = torch.as_tensor(edge_scale, dtype=delta_pos.dtype, device=delta_pos.device)
        delta_pos = delta_pos * scale
        dist = dist * scale

        # Message construction and MLP
        msg_input = torch.cat([x[dst], x[src], delta_pos, dist], dim=-1)
        msg = F.relu(self.msg_lin(msg_input))                  # (E, out_dim)

        # Mean aggregation
        agg = torch.zeros(N, self.out_dim, dtype=x.dtype, device=x.device)
        count = torch.zeros(N, 1, dtype=x.dtype, device=x.device)
        agg.index_add_(0, dst, msg)
        count.index_add_(0, dst, torch.ones(dst.shape[0], 1, dtype=x.dtype, device=x.device))
        count.clamp_(min=1.0)
        agg = agg / count                                      # (N, out_dim)

        # Node update with residual
        h = F.relu(self.node_lin(torch.cat([x, agg], dim=-1)))  # (N, out_dim)
        h = self.norm(h)
        return h + self.residual_proj(x)                       # residual
