"""rbf_fd/stencils.py — k-Nearest-Neighbour Stencil Construction.

The stencils built here serve a dual purpose that is central to the
Algebraic Transplant design:

1. **Solver stencils**: each node's k neighbours define the support set
   used to evaluate strong-form RBF-FD residuals (Section 2.2).

2. **GNN graph topology**: the identical neighbour lists are used as the
   message-passing edge set in the NeuralOperator (Principle 2, item (i),
   Section 2.5).

This stencil isomorphism (solver stencil = GNN graph) is what makes the
transplant of the discrete divergence operator G from the solver into the
GNN projection layer exact — the operator acts on the same connectivity.

Paper reference
---------------
Principle 2, item (i): "The message-passing graph is constructed with
k = 25 nearest neighbours at identical node coordinates to the solver
stencil."  Section 3.1 (Solver and Data Generation).
"""

import torch
from sklearn.neighbors import NearestNeighbors


def build_stencils(points: torch.Tensor, k: int = 25) -> torch.Tensor:
    """Build exact k-NN stencil indices for a set of scattered nodes.

    Uses a kd-tree for O(N log N) construction. Includes self (index 0
    in the returned array is the query node itself).

    Parameters
    ----------
    points : torch.Tensor
        Node coordinates, shape (N, d). Can be on CPU or GPU; the kd-tree
        computation is always performed on CPU via scikit-learn.
    k : int, optional
        Number of nearest neighbours per node (default 25, paper value).
        Includes the self-node (distance = 0).

    Returns
    -------
    torch.Tensor
        Stencil index matrix of shape (N, k), dtype int64.
        ``stencils[i, j]`` is the j-th nearest neighbour of node i.
        stencils[i, 0] == i always (self-neighbour at distance 0).
        Returned on the same device as ``points``.

    Notes
    -----
    **Sparsity**: with k = 25 and N = 225 nodes, each G row has exactly
    k * d = 50 nonzeros, giving 88.89% scalar sparsity (Appendix B).

    **Complexity**: kd-tree query is O(N k log N); operator assembly
    downstream is O(N k) — the key advantage over weak-form methods
    (Table 2 of the paper).
    """
    pts = points.cpu().numpy()
    nn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(pts)
    _, indices = nn.kneighbors(pts)
    return torch.from_numpy(indices).to(points.device)   # (N, k)
