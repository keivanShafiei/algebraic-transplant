"""gnn/neural_operator.py — Neural Operator with Algebraic Transplant (HMG).

Implements the four-stage NeuralOperator architecture described in Section 3.2:

 Stage 1: FiLM Parameter Embedding
 3-layer MLP maps mu = Re/Re_max to (gamma_l, beta_l) for each GNN layer.

 Stage 2: Scale-Adaptive Message Passing (x4 layers)
 GraphConvLayer with FiLM modulation and scale-adaptive edge features.

 Stage 3: Coefficient Prediction
 Linear decoders for a_hat (velocity, 2N) and b_pred (pressure, N).

 Stage 4: Algebraic Transplant Projection + Pressure Recovery
 a_NO = a_hat - G_int^T (G_int G_int^T + eps*I)^{-1} G_int a_hat (Eq. 16)
 WHERE correction is applied ONLY to interior DOFs (Proposition 4)
 p_corr = b_pred + q (Eq. 18)

The key architectural invariant: the projection layer holds G as a frozen
(non-trainable) buffer, transplanted verbatim from the RBF-FD solver.
No gradient flows into G. The network learns to produce a_hat that,
after projection, best approximates the true divergence-free velocity.

Hybrid Manifold Guidance (HMG) training (Section 3.5)
\-\-\--------------------------------------------------
The training loss is:

 L(theta) = L_physics(theta) + lambda(e) * L_guidance(theta)

where:
 L_physics : variance-weighted MSE on the PROJECTED output a_NO
 L_guidance : variance-weighted MSE on the RAW decoder output a_hat
 lambda(e) : 0.1 for e < 150 (manifold-seeking), 0.01 for e >= 150

The dual-path loss means the network receives two gradient signals:
one from the post-projection physics loss (which can propagate through
the differentiable Cholesky solve) and one from the pre-projection
guidance loss (which provides a direct signal to the backbone).

Paper reference
\-\-\-------------
Section 3.2 (Neural Operator Architecture), Eqs. (15)-(18).
Section 3.5 (Hybrid Manifold Guidance Training), Eq. (21).
Table 4 (GNN hyperparameters).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn

from .message_passing import GraphConvLayer
from src.projection.layer import HelmholtzProjection, SparseHelmholtzProjection


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation (FiLM) conditioner on Reynolds number.

    Maps a log-normalised Reynolds number to per-layer affine parameters
    (gamma, beta) used to modulate GNN node features:

        x_mod = gamma * x + beta

    This enables Reynolds-number-conditional processing without modifying
    the graph structure. The cosine similarity between Re=10 and Re=100
    latent vectors (cos_theta_FiLM = 0.503 at epoch 200, Table 8) shows
    the model encodes both shared topology and Re-specific structure.

    Paper reference: Section 3.2, FiLM conditioning [22].

    Parameters
    ----------
    hidden_dim : int
        Feature dimension. Output is 2*hidden_dim (gamma, beta stacked).
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2, dtype=torch.float32),
        )
        # Initialise to near-identity: gamma ~ 1, beta ~ 0
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)
        with torch.no_grad():
            self.net[-1].bias[:hidden_dim].fill_(1.0)   # gamma bias = 1
            self.net[-1].bias[hidden_dim:].fill_(0.0)    # beta bias = 0

    def forward(
        self,
        mu_log: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (gamma, beta) from log-normalised Re.

        Parameters
        ----------
        mu_log : torch.Tensor
            Log-normalised Reynolds number, shape (B,) in [-1, 1].

        Returns
        -------
        gamma : torch.Tensor
            Scale parameters, shape (B, hidden_dim).
        beta : torch.Tensor
            Shift parameters, shape (B, hidden_dim).
        """
        out = self.net(mu_log.view(-1, 1))
        return out.chunk(2, dim=-1)


class NeuralOperator(nn.Module):
    """Graph Neural Operator with Algebraic Transplant projection.

    The complete 4-stage architecture for mapping Re -> (a, b):

    mu (Re/Re_max) -> FiLM embed -> 4x GraphConvLayer -> decoders
                                      -> Helmholtz projection
                                      -> (a_NO, b_pred, q)

    Usage
    -----
    After construction, ALWAYS call:
        model.set_points(points, stencils)          # registers node coords
        model.set_projection(G, interior_mask)      # transplants G_int from solver

    Then the forward() call returns (a_hat_raw, a_NO_projected, b_pred).
    Physical pressure is recovered externally as: p_corr = b_pred + q,
    where q is obtained by calling:
        a_NO, q = model.projection(a_hat, return_q=True)

    Parameters
    ----------
    n_nodes : int
        Number of nodes N. Must match the G operator shape.
    d : int, optional
        Spatial dimension (default 2).
    param_dim : int or None, optional
        Parameter dimension (unused, kept for interface compatibility).
    k : int, optional
        k-NN connectivity (default 25, must match stencil used for G).
    hidden : int, optional
        Node feature dimension (default 64, Table 4).
    layers : int, optional
        Number of GNN layers (default 4, Table 4).
    eps : float, optional
        Tikhonov regularisation for projection (default 1e-8; Table 4).
    """

    def __init__(
        self,
        n_nodes: int,
        d: int = 2,
        param_dim: int | None = None,
        k: int = 25,
        hidden: int = 64,
        layers: int = 4,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d = d
        self.k = k
        self.eps = eps
        self.hidden = hidden
        self.param_dim = param_dim

        # Stage 1: input encoding (coordinates + normalised Re)
        self.feature_encoder = nn.Sequential(
            nn.Linear(d + 1, hidden, dtype=torch.float32),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        # FiLM conditioners (one per GNN layer)
        self.film_conditioners = nn.ModuleList(
            [FiLMConditioner(hidden_dim=hidden) for _ in range(layers)]
        )

        # Stage 2: message-passing backbone
        self.gnn_layers = nn.ModuleList(
            [GraphConvLayer(hidden, hidden, pos_dim=d) for _ in range(layers)]
        )

        # Stage 3: coefficient decoders
        self.decoder_vel = nn.Linear(hidden, d, dtype=torch.float32)
        self.decoder_p = nn.Linear(hidden, 1, dtype=torch.float32)
        nn.init.xavier_uniform_(self.decoder_vel.weight, gain=0.1)
        nn.init.zeros_(self.decoder_vel.bias)
        nn.init.xavier_uniform_(self.decoder_p.weight, gain=0.1)
        nn.init.zeros_(self.decoder_p.bias)

        # Stage 4: projection layer (set later via set_projection)
        self.projection = None

        # Buffers for coordinate and scale information
        self.register_buffer(
            "points", torch.empty(0, d, dtype=torch.float32), persistent=False
        )
        self.register_buffer("h_train", torch.tensor(float("nan"), dtype=torch.float32))
        self.register_buffer("h_infer", torch.tensor(float("nan"), dtype=torch.float32))

        # Store stencils for checkpoint metadata (non-persistent, used for h_avg computation)
        self._stencils = None

    @staticmethod
    def _compute_h_avg(points: torch.Tensor, stencils: torch.Tensor) -> torch.Tensor:
        """Compute mean nearest-neighbour fill distance h_avg."""
        pts = points.float()
        neigh = pts[stencils.long()]
        center = pts.unsqueeze(1).expand_as(neigh)
        return torch.norm(neigh - center, dim=-1).mean()

    def set_points(self, points: torch.Tensor, stencils: torch.Tensor | None = None):
        """Register node coordinates and optionally compute h_avg.

        Must be called before forward(). If stencils are provided, h_infer
        is computed from the mesh geometry. On the first call, h_train is
        also set to h_infer (same grid as training).

        Parameters
        ----------
        points : torch.Tensor, shape (N, d).
        stencils : torch.Tensor or None, shape (N, k).
        """
        if not isinstance(points, torch.Tensor):
            raise TypeError(f"points must be torch.Tensor, got {type(points)}")
        if points.dim() != 2:
            raise ValueError(f"points must be 2D (N, d), got shape {points.shape}")
        if points.shape[1] != self.d:
            raise ValueError(
                f"points dimension ({points.shape[1]}) must match model d ({self.d})"
            )
        if not torch.isfinite(points).all():
            raise ValueError("points contains NaN or Inf")

        # Check points are in reasonable range for unit square
        if points.min() < -0.1 or points.max() > 1.1:
            import warnings
            warnings.warn(
                f"points range [{points.min():.3f}, {points.max():.3f}] outside [0,1]^2. "
                f"Expected unit square domain.",
                UserWarning,
            )

        self.points = points.detach().to(dtype=torch.float32)

        if stencils is not None:
            if stencils.dim() != 2:
                raise ValueError(f"stencils must be 2D (N, k), got shape {stencils.shape}")
            if stencils.shape[0] != points.shape[0]:
                raise ValueError(
                    f"stencils rows ({stencils.shape[0]}) must match points ({points.shape[0]})"
                )
            if stencils.shape[1] != self.k:
                raise ValueError(
                    f"stencils columns ({stencils.shape[1]}) must match k ({self.k})"
                )
            if stencils.min() < 0 or stencils.max() >= points.shape[0]:
                raise ValueError(
                    f"stencil indices out of range [0, {points.shape[0]}). "
                    f"Got min={stencils.min()}, max={stencils.max()}"
                )
            # Check for duplicate stencil entries
            for i in range(stencils.shape[0]):
                row = stencils[i]
                if len(row.unique()) != len(row):
                    raise ValueError(f"Duplicate indices in stencil row {i}")

            self._stencils = stencils.detach().to(dtype=torch.int64)
            h = self._compute_h_avg(self.points, self._stencils)
            self.h_infer = h.detach().to(dtype=torch.float32)
            if torch.isnan(self.h_train):
                self.h_train = self.h_infer.clone()

    def set_scales(
        self,
        h_train: float | torch.Tensor | None = None,
        h_infer: float | torch.Tensor | None = None,
        points: torch.Tensor | None = None,
        stencils: torch.Tensor | None = None,
    ):
        """Set h_train and h_infer for scale-adaptive edge encoding.

        For zero-shot resolution transfer, set h_train to the training-grid
        fill distance and h_infer to the inference-grid fill distance.
        The edge scale s_adapt = h_train / h_infer is computed automatically
        in _edge_scale() during forward().

        Parameters
        ----------
        h_train : float, optional
            Training-grid mean fill distance.
        h_infer : float, optional
            Inference-grid mean fill distance.
        points, stencils : optional
            If provided, h_infer (and optionally h_train) are computed
            automatically from the mesh geometry.
        """
        if points is not None:
            self.set_points(points, stencils)
        if points is not None and stencils is not None:
            h_geom = self._compute_h_avg(points.to(dtype=torch.float32), stencils)
            if h_train is None:
                h_train = h_geom
            if h_infer is None:
                h_infer = h_geom
        if h_train is not None:
            self.h_train = torch.as_tensor(
                h_train, dtype=torch.float32, device=self.points.device
            )
        if h_infer is not None:
            self.h_infer = torch.as_tensor(
                h_infer, dtype=torch.float32, device=self.points.device
            )

    def _edge_scale(self) -> torch.Tensor:
        """Compute scale-adaptive factor s_adapt = h_train / h_infer.

        Returns 1.0 during training (no rescaling).
        Returns h_train / h_infer at inference on a different grid.
        """
        device = (
            self.points.device if self.points.numel() else self.h_train.device
        )
        if self.training:
            return torch.tensor(1.0, dtype=torch.float32, device=device)
        if torch.isnan(self.h_train) or torch.isnan(self.h_infer):
            return torch.tensor(1.0, dtype=torch.float32, device=device)
        return self.h_train / (self.h_infer + 1e-12)

    def set_projection(
        self,
        G: torch.Tensor,
        interior_mask: torch.Tensor = None,
        interior_node_mask: torch.Tensor = None,
    ):
        """Transplant the discrete divergence operator G into the projection layer.

        Automatically selects HelmholtzProjection (dense Cholesky) for dense
        G, or SparseHelmholtzProjection (Jacobi-PCG) for sparse G.

        CRITICAL FIX: Added interior_mask parameter for boundary-safe projection
        per Proposition 4. When provided, the correction is applied ONLY to
        interior DOFs, preserving Dirichlet boundary conditions.

        Parameters
        ----------
        G : torch.Tensor
            Interior-restricted divergence operator G_int, from the solver.
            For the True Algebraic Transplant: use solver.G_int, not solver.G_full.
        interior_mask : torch.Tensor, optional
            Boolean mask of shape (2N,) indicating interior DOFs.
            When provided, enables boundary-safe projection (Proposition 4).
            When None, falls back to legacy full-domain behavior (not recommended).
        interior_node_mask : torch.Tensor, optional
            Boolean mask of shape (N,) for pressure nodes.
            When provided, enables pressure recovery mapping q from N_int to N.
        """
        if G.is_sparse:
            self.projection = SparseHelmholtzProjection(
                G=G, eps=self.eps, interior_mask=interior_mask
            )
        else:
            self.projection = HelmholtzProjection(
                G=G, eps=self.eps, interior_mask=interior_mask
            )

        if interior_node_mask is not None:
            self.set_interior_node_mask(interior_node_mask)

    def set_interior_node_mask(self, mask: torch.Tensor):
        """Register interior NODE mask (N,) for pressure recovery.

        The projection layer uses interior_dof_mask (2N,) for velocity DOFs.
        For pressure recovery, we need interior_node_mask (N,) to map
        q (N_int,) back to full N nodes.

        Parameters
        ----------
        mask : torch.Tensor
            Boolean mask of shape (N,), True for interior nodes.
        """
        self.register_buffer("interior_node_mask", mask.to(torch.bool))

    def set_interior_mask(self, mask: torch.Tensor):
        """Register interior DOF mask for boundary-safe projection.

        DEPRECATED: Use set_projection(G, interior_mask=mask) instead.
        Kept for backward compatibility.
        """
        self.interior_mask = mask
        if self.projection is not None:
            self.projection.interior_mask = mask

    def _log_re_normalized(
        self,
        mu: torch.Tensor,
        re_min: float = 10.0,
        re_max: float = 100.0,
    ) -> torch.Tensor:
        """Map mu = Re/Re_max to log-normalised value in [-1, 1].

        log_re_norm = (log(Re) - log(Re_min)) / (log(Re_max) - log(Re_min)) * 2 - 1
        """
        re = mu * re_max
        log_re = torch.log(re.clamp(min=1.0))
        lo, hi = math.log(re_min), math.log(re_max)
        return (log_re - lo) / (hi - lo) * 2.0 - 1.0

    def forward(
        self,
        mu: torch.Tensor,
        edge_index: torch.Tensor,
        inference: bool = False,
        use_projection: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through all four stages.

        Parameters
        ----------
        mu : torch.Tensor
            Normalised Reynolds number, shape (B,) or scalar.
            mu = Re / Re_max (= Re / 100 for the training range).
        edge_index : torch.Tensor
            GNN edge connectivity, shape (2, E). Must be the same
            k-NN stencil used to assemble G (stencil isomorphism).
        inference : bool, optional
            Unused (kept for interface compatibility). Scale adaptation
            is controlled by model.training flag.
        use_projection : bool, optional
            If False, skip the projection layer (returns (a_hat, a_hat, b)).
            Useful for ablation studies.

        Returns
        -------
        a_hat : torch.Tensor
            Raw velocity coefficients before projection, shape (B, 2N).
        a_NO : torch.Tensor
            Projected (divergence-free) velocity, shape (B, 2N).
            Boundary DOFs are unchanged when interior_mask was provided.
        b : torch.Tensor
            Raw pressure head, shape (B, N).
            Physical pressure: p_corr = b + q (use projection with return_q=True).
        """
        if self.points.numel() == 0:
            raise RuntimeError("Call set_points(points, stencils) before forward().")

        if self.projection is None and use_projection:
            raise RuntimeError(
                "Projection layer not set. Call set_projection(G, interior_mask) before forward()."
            )

        mu = mu.view(-1).float()
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            raise ValueError("mu contains NaN or Inf")
        if (mu < 0).any() or (mu > 1.5).any():  # Normalized Re should be in [0, 1.5] roughly
            import warnings
            warnings.warn(
                f"mu values outside expected range [0, 1.5]: min={mu.min():.3f}, max={mu.max():.3f}",
                UserWarning,
            )

        batch_size = mu.shape[0]
        mu_log = self._log_re_normalized(mu)

        # Stage 1: encode (coordinates, Re) -> node features
        pts = self.points.unsqueeze(0).expand(batch_size, -1, -1)
        re_field = mu_log.view(batch_size, 1, 1).expand(-1, self.n_nodes, -1)
        enc_input = torch.cat([pts, re_field], dim=-1)
        x = self.feature_encoder(enc_input)  # (B, N, hidden)

        edge_scale = self._edge_scale()

        # Stage 2: message passing with FiLM conditioning
        for film, gnn in zip(self.film_conditioners, self.gnn_layers):
            gamma, beta = film(mu_log)
            x_mod = gamma.unsqueeze(1) * x + beta.unsqueeze(1)
            x = torch.stack(
                [gnn(x_mod[b], edge_index, self.points, edge_scale=edge_scale)
                 for b in range(batch_size)],
                dim=0,
            )  # (B, N, hidden)

        # Stage 3: decode velocity and pressure
        a_hat = self.decoder_vel(x).reshape(batch_size, -1)  # (B, 2N)
        b = self.decoder_p(x).squeeze(-1)  # (B, N)

        # Stage 4: Algebraic Transplant projection
        if self.projection is not None and use_projection:
            a_proj_rows = []
            for i in range(batch_size):
                a_proj_i, _ = self.projection(a_hat[i], return_q=True)
                a_proj_rows.append(a_proj_i)
            a_NO = torch.stack(a_proj_rows, 0)
            return a_hat, a_NO, b
        else:
            return a_hat, a_hat, b

    def predict(
        self,
        mu: torch.Tensor,
        edge_index: torch.Tensor,
        return_raw: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict divergence-free velocity and pressure-corrected pressure.

        This is the PRIMARY inference API for end-users. It returns the
        physically meaningful fields: projected velocity a_NO and corrected
        pressure p_corr = b_pred + q (Eq. 18).

        Parameters
        ----------
        mu : torch.Tensor
            Normalised Reynolds number, shape (B,) or scalar.
        edge_index : torch.Tensor
            GNN edge connectivity, shape (2, E).
        return_raw : bool, optional
            If True, also return raw (pre-projection) velocity a_hat and
            raw pressure head b_pred for debugging.

        Returns
        -------
        a_NO : torch.Tensor
            Projected divergence-free velocity, shape (B, 2N).
        p_corr : torch.Tensor
            Pressure-consistent field, shape (B, N).
            p_corr = b_pred + q where q is the projection Lagrange multiplier.
        a_hat : torch.Tensor (only if return_raw=True)
            Raw decoder velocity output before projection.
        b_pred : torch.Tensor (only if return_raw=True)
            Raw decoder pressure output.

        Notes
        -----
        The pressure correction q is obtained from the projection solve:
            q = (G_int G_int^T + eps*I)^{-1} G_int a_hat
        and represents the discrete analogue of the pressure Poisson solve
        in the fractional-step method (Algorithm 1, Step 2).

        For engineering force evaluation, p_corr accuracy depends on upstream
        velocity accuracy (Section 4.10, Proposition 4). At 13.75% velocity
        error, drag error is ~30%; the pressure output should be interpreted
        as structurally aligned, not engineering-grade.

        Example
        -------
        >>> model.set_projection(G_int, interior_mask, interior_node_mask)
        >>> a_NO, p_corr = model.predict(mu, edge_index)
        >>> # For debugging: inspect raw vs projected
        >>> a_NO, p_corr, a_hat, b_pred = model.predict(mu, edge_index, return_raw=True)
        """
        # Run forward pass to get raw outputs
        a_hat, a_NO, b_pred = self.forward(mu, edge_index)

        batch_size = mu.shape[0]
        q_rows = []

        # Extract q from projection for each sample in batch
        for i in range(batch_size):
            _, q_i = self.projection(a_hat[i], return_q=True)
            q_rows.append(q_i)

        q = torch.stack(q_rows, 0)  # (B, N_int) or (B, N)

        # Pressure correction: p_corr = b_pred + q
        # q has shape (B, N_int) if interior-restricted, but b_pred is (B, N)
        # Need to map q back to full N nodes
        if q.shape[1] != b_pred.shape[1]:
            # Interior-restricted q: expand to full N nodes
            q_full = torch.zeros_like(b_pred)
            # q corresponds to interior nodes only
            # We need to know which nodes are interior
            # This requires storing interior_node_mask (N,) not just interior_dof_mask (2N,)
            if hasattr(self, 'interior_node_mask') and self.interior_node_mask is not None:
                q_full[:, self.interior_node_mask] = q
            else:
                # Fallback: if N_int == N (full domain), q is already full
                if q.shape[1] == b_pred.shape[1]:
                    q_full = q
                else:
                    raise RuntimeError(
                        f"Cannot align q shape {q.shape} with b_pred shape {b_pred.shape}. "
                        f"Set interior_node_mask via set_interior_node_mask()."
                    )
        else:
            q_full = q

        p_corr = b_pred + q_full

        if return_raw:
            return a_NO, p_corr, a_hat, b_pred
        return a_NO, p_corr
