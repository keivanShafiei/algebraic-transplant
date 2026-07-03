"""gnn/neural_operator.py — Neural Operator with Algebraic Transplant Projection.

This module implements the 4-stage neural operator described in Section 3.2:

    Stage 1: Parameter Embedding (3-layer MLP + FiLM conditioning)
    Stage 2: Scale-Adaptive Message Passing (4 graph conv layers)
    Stage 3: Coefficient Prediction (velocity + pressure decoders)
    Stage 4: Algebraic Transplant Projection + Pressure Recovery

The projection layer is the key contribution: it enforces Ga_NO = 0
algebraically by reusing the solver's discrete divergence operator G
as a frozen, parameter-free layer (Theorem 2, Section 2.4).

For boundary-safe projection, the interior-restricted operator G_int
is used with interior_dof_mask to preserve Dirichlet boundary conditions
(Proposition 4, Section 4.9). Using the full-domain G_full corrupts
boundary velocities and produces ~74% drag error (Table 12).

Paper reference
---------------
Section 3.2: Neural Operator Architecture
Section 3.3: Algebraic Pressure Correction from Projection Byproduct
Section 4.9: Interior-Restricted Projection and Boundary Handling
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .message_passing import GraphConvLayer
from ..projection.layer import HelmholtzProjection, SparseHelmholtzProjection


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation (FiLM) for Reynolds-number conditioning.

    A 3-layer MLP maps the scalar parameter mu to per-layer affine
    coefficients (gamma, beta) that modulate node features after each
    graph convolution (Section 3.2, Stage 1).

    Parameters
    ----------
    param_dim : int
        Dimensionality of the parameter vector (e.g., 1 for scalar Re).
    hidden : int
        Hidden dimension of the MLP.
    num_layers : int
        Number of message-passing layers to condition.
    """

    def __init__(self, param_dim: int = 1, hidden: int = 64, num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.mlp = nn.Sequential(
            nn.Linear(param_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * num_layers * hidden),
        )

    def forward(self, mu: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return list of (gamma, beta) tuples, one per message-passing layer."""
        out = self.mlp(mu)
        chunks = torch.chunk(out, 2 * self.num_layers, dim=-1)
        return [
            (chunks[2 * i], chunks[2 * i + 1])
            for i in range(self.num_layers)
        ]


class NeuralOperator(nn.Module):
    """Graph Neural Operator with differentiable divergence-free projection.

    Architecture (Section 3.2):
        mu -> FiLM -> [GraphConv x layers] -> Decoder -> Projection -> Output

    Parameters
    ----------
    in_channels : int
        Input node feature dimension (e.g., 2 for x,y coordinates).
    hidden : int
        Hidden dimension for all layers (default 64, Table 4).
    layers : int
        Number of message-passing layers (default 4, Table 4).
    param_dim : int
        Dimensionality of the parameter vector (default 1 for scalar Re).
    eps : float
        Tikhonov regularisation for the projection layer (default 1e-8).

    Attributes
    ----------
    projection : HelmholtzProjection or SparseHelmholtzProjection
        The differentiable divergence-free projection layer.
    interior_dof_mask : torch.Tensor or None
        Boolean mask of shape (2N,) for interior velocity DOFs.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden: int = 64,
        layers: int = 4,
        param_dim: int = 1,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.eps = eps
        self.projection = None
        self.interior_dof_mask: Optional[torch.Tensor] = None
        self.interior_node_mask: Optional[torch.Tensor] = None

        # Stage 1: FiLM conditioning
        self.film = FiLMConditioner(param_dim=param_dim, hidden=hidden, num_layers=layers)

        # Stage 2: Graph convolution layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(hidden, hidden) for _ in range(layers)
        ])

        # Stage 3: Decoders
        self.decoder_vel = nn.Linear(hidden, 2)  # velocity coefficients
        self.decoder_p = nn.Linear(hidden, 1)    # pressure coefficients

    def set_projection(
        self,
        G: torch.Tensor,
        interior_dof_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Attach the differentiable projection layer.

        Parameters
        ----------
        G : torch.Tensor
            Discrete divergence operator from the solver.
            For boundary-safe projection, this should be G_int
            (interior-restricted, shape (N_int, 2N)).
            For legacy behavior, this can be G_full (shape (N, 2N)).
        interior_dof_mask : torch.Tensor, optional
            Boolean mask of shape (2N,) indicating interior velocity DOFs.
            When provided, the projection correction is applied ONLY to
            interior DOFs, preserving Dirichlet boundary conditions.
            **Strongly recommended** for problems with non-trivial
            Dirichlet boundaries (Section 4.9).
        """
        if G.is_sparse:
            self.projection = SparseHelmholtzProjection(
                G=G, eps=self.eps, interior_dof_mask=interior_dof_mask
            )
        else:
            self.projection = HelmholtzProjection(
                G=G, eps=self.eps, interior_dof_mask=interior_dof_mask
            )
        self.interior_dof_mask = interior_dof_mask

    def set_interior_mask(self, mask: torch.Tensor) -> None:
        """Set the interior node mask for pressure recovery.

        Parameters
        ----------
        mask : torch.Tensor
            Boolean mask of shape (N,) indicating interior nodes.
            Used to map q (N_int,) to full pressure field (N,).
        """
        self.interior_node_mask = mask.to(torch.bool)

    def forward(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        mu: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning raw velocity and pressure (before projection).

        Parameters
        ----------
        pos : torch.Tensor
            Node coordinates, shape (N, 2).
        edge_index : torch.Tensor
            Graph connectivity, shape (2, E).
        mu : torch.Tensor
            Parameter vector (e.g., Reynolds number), shape (param_dim,).
        edge_attr : torch.Tensor, optional
            Edge features (delta_x, ||delta_x||), shape (E, 3).
        edge_scale : float, optional
            Scale factor for edge features (s_adapt = h_train / h_infer).

        Returns
        -------
        a_NO : torch.Tensor
            Divergence-free velocity coefficients, shape (2N,).
        p_corr : torch.Tensor
            Algebraically corrected pressure, shape (N,).
        """
        x = pos  # (N, 2)
        film_params = self.film(mu)

        for i, conv in enumerate(self.gnn_layers):
            x = conv(x, edge_index, edge_attr=edge_attr, edge_scale=edge_scale)
            gamma, beta = film_params[i]
            x = gamma * x + beta
            if i < self.layers - 1:
                x = F.silu(x)

        # Stage 3: Decode velocity and pressure
        a_hat = self.decoder_vel(x).flatten()  # (2N,)
        b_pred = self.decoder_p(x).squeeze(-1)  # (N,)

        # Stage 4: Project velocity and recover pressure
        if self.projection is not None:
            a_NO, q = self.projection(a_hat, return_q=True)

            # Pressure recovery: p_corr = b_pred + q (Eq. 18)
            if self.interior_node_mask is not None:
                q_full = torch.zeros_like(b_pred)
                q_full[self.interior_node_mask] = q
            else:
                q_full = q
            p_corr = b_pred + q_full
        else:
            a_NO = a_hat
            p_corr = b_pred

        return a_NO, p_corr

    def predict(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        mu: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference wrapper — same as forward()."""
        return self.forward(pos, edge_index, mu, edge_attr, edge_scale)
