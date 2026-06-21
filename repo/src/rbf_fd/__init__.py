"""RBF-FD discretisation: kernel, stencils, operators, and solver."""
from .kernel import mq_phi, mq_dphi_dr, mq_laplacian
from .stencils import build_stencils
from .operators import assemble_divergence_operator, assemble_phi_stencil, assemble_laplacian_stencil
from .solver import NavierStokesSolver

# Backward compatibility alias.
# NOTE: RBFFDSolver was an older class with a different API (scipy sparse matrices,
# explicit .assemble() method, .divergence_matrix attribute, etc.).
# Scripts using the old API will need updating. NavierStokesSolver uses PyTorch tensors
# and assembles operators in __init__.
RBFFDSolver = NavierStokesSolver
