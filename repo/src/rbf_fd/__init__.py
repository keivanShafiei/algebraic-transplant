"""RBF-FD discretisation: kernel, stencils, operators, and solver."""
from .kernel import mq_phi, mq_dphi_dr, mq_laplacian
from .stencils import build_stencils
from .operators import assemble_divergence_operator, assemble_phi_stencil, assemble_laplacian_stencil
from .solver import NavierStokesSolver
