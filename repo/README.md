# Algebraic Transplant of Meshless Discrete Operators into Graph Neural Architectures

**Numerically Consistent Neural Operators for Parametric Incompressible Flows**

> Shafiei, A. & Mosavi Nezhad, S. M. — *Preprint submitted to Journal of Computational Physics*, April 2026

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository contains the full implementation of the **Algebraic Transplant** framework — a co-designed system that combines:

- **Strong-form RBF-FD collocation** for efficient training-data generation (5–6× faster than FEM/spectral alternatives)
- **Graph Neural Operator (GNN)** with FiLM conditioning for Reynolds-number-parametric flow prediction
- **Differentiable Helmholtz projection layer** constructed from the solver's exact discrete divergence operator `G`, enforcing `G·a = 0` as a hard algebraic constraint at every forward pass
- **Hybrid Manifold Guidance (HMG)** two-phase training with variance-weighted loss
- **Scale-adaptive edge encoding** for zero-shot resolution transfer
- **Interior-restricted True Algebraic Transplant** that preserves Dirichlet boundary conditions and aerodynamic surface forces at solver precision

### Key Results (from the paper)

| Metric | Value | Notes |
|--------|-------|-------|
| Divergence residual `ε_div` (float32) | ≈ 4 × 10⁻⁵ | 6 orders below soft-penalty baselines |
| Divergence residual `ε_div` (float64 assembly) | O(10⁻¹³) | Machine-precision floor |
| Variance-weighted MSE | 3.728 × 10⁻² | 91.9% reduction vs. zero predictor |
| Warm-start speedup at Re = 500 | 3.2× wall-clock | 4.2× iteration reduction |
| Drag force discrepancy (True Algebraic Transplant) | 3.363 × 10⁻⁵ % | Solver-precision alignment |
| PCG projection (N = 100,000) | 3.08 s / 0.45 GB VRAM | Industrial-scale deployment |

---

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Project Structure](#project-structure)
- [Module Documentation](#module-documentation)
- [Configuration](#configuration)
- [Tests](#tests)
- [Citation](#citation)

---

## Architecture

```
Parameter μ (Re/Re_max)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1 — FiLM Parameter Embedding                │
│  3-layer MLP → (γ_ℓ, β_ℓ) for each GNN layer       │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2 — Scale-Adaptive Message Passing (×4)      │
│  GraphConvLayer  +  FiLM modulation                 │
│  Edge features rescaled by h_train/h_infer          │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  Stage 3 — Coefficient Decoders                     │
│  â  ∈ ℝ²ᴺ  (raw velocity)                          │
│  b_pred ∈ ℝᴺ  (raw pressure head)                   │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  Stage 4 — Algebraic Transplant Projection Layer    │
│  q = (Gᵢₙₜ Gᵢₙₜᵀ + εI)⁻¹ Gᵢₙₜ â                  │
│  a_NO = â − Gᵢₙₜᵀ q         ← divergence-free      │
│  p_corr = b_pred + q         ← physical pressure    │
└─────────────────────────────────────────────────────┘
```

The operator `G` is transplanted **verbatim** from the RBF-FD solver — not learned, not approximated.

---

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1 (with CUDA 12.1 recommended for GPU runs)
- scikit-learn ≥ 1.3
- scipy ≥ 1.11
- PyYAML ≥ 6.0
- matplotlib ≥ 3.7
- numpy ≥ 1.24
- pytest ≥ 7.4

### Install

```bash
git clone https://github.com/kshafiei/algebraic-transplant.git
cd algebraic-transplant
pip install -r requirements.txt
```

To install as an editable package:

```bash
pip install -e .
```

---

## Quick Start

### 1. Generate Training Data

```bash
python scripts/generate_data.py --n 225 --ns 400 --re-min 10 --re-max 100 --device cuda
```

This runs the RBF-FD projection solver for 400 Reynolds numbers and saves:
- `data/samples/sample_XXXX.pt` — individual flow solutions
- `data/fixed_G.pt` — the discrete divergence operator **G** (transplanted into the GNN)
- `data/interior_mask.pt` — interior DOF mask for boundary-safe projection

### 2. Train the Neural Operator

```bash
python scripts/train.py
```

Trains the GNN with Hybrid Manifold Guidance (200 epochs, cosine LR annealing). Best checkpoint saved to `results/model_best_v8.pt`.

### 3. Evaluate Projection Efficacy

```bash
python scripts/eval_projection.py --model results/model_best_v8.pt
```

Reports `r_before`, `r_after`, and reduction ratio ρ (reproduces Table 9 of the paper).

### 4. Zero-Shot Resolution Transfer

```bash
python scripts/eval_zeroshot.py \
    --model results/model_best_v8.pt \
    --coarse_sample data/samples/sample_0000.pt \
    --fine_sample data/fine_samples/sample_0000.pt
```

Tests resolution invariance with scale-adaptive edge encoding.

### 5. Run All Diagnostics

```bash
python scripts/diagnostic.py
```

---

## Reproducing Paper Results

All figures and tables from the paper can be regenerated with:

```bash
python scripts/generate_all_figures.py
```

Individual scripts for specific figures:

| Script | Paper Item |
|--------|-----------|
| `scripts/plot_fig9_constraint_enforcement.py` | Figure 3 — Projection layer efficacy |
| `scripts/plot_fig11_scaling.py` | Figure 1 — End-to-end timing decomposition |
| `scripts/plot_field_comparison.py` | Velocity/pressure field comparison |
| `scripts/plot_pressure_recovery.py` | Figure — p_corr = b_pred + q recovery |
| `scripts/plot_fallback_analysis.py` | Figure 7 — Adaptive fallback trigger |
| `scripts/test_scalability.py` | Table 17 — PCG scalability at N = 100,000 |
| `scripts/train_baselines.py` | Table 11 — FNO / DeepONet / POD-RBF baselines |

---

## Project Structure

```
algebraic-transplant/
│
├── config.yaml                    # All hyperparameters (Table 4 of the paper)
│
├── src/
│   ├── rbf_fd/
│   │   ├── kernel.py              # MQ RBF kernel and derivatives (Appendix B)
│   │   ├── stencils.py            # k-NN stencil construction (isomorphic to GNN graph)
│   │   ├── operators.py           # Divergence G, interpolation Φ, Laplacian matrices
│   │   └── solver.py              # Fractional-step Navier–Stokes solver (Algorithm 1)
│   │
│   ├── projection/
│   │   └── layer.py               # HelmholtzProjection (dense Cholesky) and
│   │                              #   SparseHelmholtzProjection (Jacobi-PCG)
│   │
│   ├── gnn/
│   │   ├── message_passing.py     # GraphConvLayer with FiLM + scale-adaptive edges
│   │   └── neural_operator.py    # Full NeuralOperator (4-stage architecture)
│   │
│   ├── data/
│   │   ├── cavity.py              # Lid-driven cavity node generation
│   │   ├── dataset.py             # ParametricCavityDataset & PrecomputedDataset
│   │   └── synthetic.py           # Analytical divergence-free fields (Appendix C)
│   │
│   ├── baselines/
│   │   ├── deeponet.py            # DeepONet baseline
│   │   ├── fno.py                 # Fourier Neural Operator baseline
│   │   └── pod_rbf.py             # POD-RBF baseline
│   │
│   └── utils/
│       ├── metrics.py             # Divergence residual, relative L₂ error
│       └── viz.py                 # Figure generation utilities
│
├── scripts/
│   ├── generate_data.py           # RBF-FD dataset generation
│   ├── train.py                   # HMG training (Algorithm 2 in paper)
│   ├── train_baselines.py         # Baseline model training
│   ├── eval_projection.py         # Projection efficacy measurement (Algorithm 3)
│   ├── eval_zeroshot.py           # Zero-shot resolution transfer (Section 4.5)
│   ├── diagnostic.py              # Full model diagnostics
│   ├── test_scalability.py        # Large-scale PCG test (N = 100,000)
│   ├── plot_*.py                  # Figure generation scripts
│   └── generate_all_figures.py    # Reproduce all paper figures
│
└── tests/
    ├── test_projection.py          # Unit tests for Theorem 2 properties
    └── test_consistency.py         # Numerical consistency with paper claims
```

---

## Module Documentation

### `src/rbf_fd/kernel.py`
Implements the **multiquadric (MQ) RBF** kernel and its derivatives used to assemble the discrete differential operators.

- `mq_phi(r, c)` — kernel value ϕ(r) = √(1 + (r/c)²)
- `mq_dphi_dr(r, c)` — first derivative dϕ/dr (basis of the divergence operator G)
- `mq_laplacian(r, c, d)` — Laplacian ∇²ϕ(r) with correct d-dimensional coefficient (Appendix B, ∇²ϕ(0) = d/c² ≈ 272.22 for N=225, d=2)

### `src/rbf_fd/stencils.py`
Builds the **k-nearest-neighbour stencils** that are simultaneously used as:
- The support set for RBF-FD operator assembly
- The message-passing graph topology of the GNN (Principle 2, item (i) — stencil isomorphism)

### `src/rbf_fd/operators.py`
Assembles the three key sparse matrices:
- **G** ∈ ℝᴺˣ²ᴺ — discrete divergence operator (Eq. 6); the central object of the Algebraic Transplant
- **Φ** ∈ ℝᴺˣᴺ — RBF interpolation matrix
- **Lap** ∈ ℝᴺˣᴺ — discrete Laplacian for the diffusion term

### `src/rbf_fd/solver.py`
Implements the **fractional-step Navier–Stokes solver** (Algorithm 1 of the paper):
1. Momentum solve: K(aⁿ) a* = F
2. Pressure correction: L_int b^(n+1) = G_int a*  (interior-restricted, float64)
3. Velocity correction: a^(n+1) = a* − G_intᵀ b^(n+1)

Exposes `solver.G` (full divergence operator) and `solver.G_int` (interior-restricted version, key to the True Algebraic Transplant).

### `src/projection/layer.py`
Two projection layer implementations:

**`HelmholtzProjection`** (dense Cholesky, for N ≤ ~10,000):
```
q = (GGᵀ + εI)⁻¹ G â        [Cholesky solve in float64]
a_NO = â − Gᵀ q               [divergence-free velocity]
p_corr = b_pred + q            [physical pressure recovery, Eq. 18]
```

**`SparseHelmholtzProjection`** (Jacobi-PCG, for industrial N ≤ 100,000+):
- Matrix-free: only sparse matrix–vector products
- O(N) memory footprint vs. O(N²) for dense Cholesky
- Jacobi preconditioner M = diag(GGᵀ + εI)
- Validated at N = 100,000: 3.08 s, 0.45 GB VRAM, ε_div < 1.89 × 10⁻⁴

### `src/gnn/message_passing.py`
**`GraphConvLayer`** — single message-passing layer implementing:
- Spatial edge features (Δx_ij, ‖Δx_ij‖)
- **Scale-adaptive edge encoding**: features multiplied by `h_train/h_infer` at inference time (Eq. 15), enabling zero-shot resolution transfer without retraining
- FiLM affine modulation (γ x + β) after convolution
- Residual connection with layer normalisation

### `src/gnn/neural_operator.py`
**`NeuralOperator`** — the full 4-stage architecture:
- `set_projection(G)` — transplants G from the solver into the frozen projection layer
- `set_scales(h_train, h_infer)` — registers the scale factors for adaptive edge encoding
- `forward(mu, edge_index)` — returns `(a_hat_raw, a_NO_projected, b_pressure_head)`

**`FiLMConditioner`** — 3-layer MLP mapping log-normalized Re to (γ, β) affine parameters.

### `src/data/`
- **`cavity.py`** — generates uniform node grids on the lid-driven cavity Ω = [0,1]²
- **`dataset.py`** — `PrecomputedDataset` loads pre-generated `.pt` solver outputs; `ParametricCavityDataset` for lightweight Re-sampling
- **`synthetic.py`** — generates analytically divergence-free fields from random stream functions (Appendix C; used only for architectural sanity checks)

---

## Configuration

All hyperparameters are in `config.yaml` (reproducing Table 4 of the paper):

```yaml
ns: 400                    # Training samples
re_min: 10
re_max: 100
n_nodes_list: [225, 1000, 5000, 10000]
stencil_k: 25              # k-NN neighbours (GNN graph = solver stencil)
rbf_c_factor: 1.2          # Shape parameter c = 1.2 × h_avg
projection_eps: 1.0e-8     # Tikhonov regularisation ε
gnn_layers: 6              # Message-passing layers
hidden_dim: 128            # Node feature dimension
batch_size: 16
epochs: 200
seed: 42
dtype: float32
```

**HMG λ-schedule** (hard-coded in `train.py`, Section 3.5):
- Epochs 1–149: λ = 0.1 (manifold-seeking phase)
- Epochs 150–200: λ = 0.01 (physics-refining phase)

---

## Tests

Run the full test suite:

```bash
pytest tests/ -v
```

**`tests/test_projection.py`** — verifies Theorem 2 mathematically:
- `test_mass_conservation` — ε_div < 5 × 10⁻⁵ (float32 floor, Remark 2)
- `test_divergence_reduction_ratio` — ρ > 10³ (paper reports ρ̄ ≈ 2.13 × 10⁵)
- `test_idempotency` — P_div ∘ P_div = P_div
- `test_minimum_energy_pythagorean` — Pythagorean identity ‖â‖² = ‖a_NO‖² + ‖â − a_NO‖²
- `test_differentiability` — gradient flows through `cholesky_solve`
- `test_finiteness_on_zero_input` — P_div(0) = 0

**`tests/test_consistency.py`** — verifies paper claims numerically:
- `test_laplacian_peak_value` — ∇²ϕ(0) = d/c² ≈ 272.22 (Appendix B)
- `test_G_shape` — G ∈ ℝᴺˣ²ᴺ (Eq. 6)
- `test_G_sparsity` — O(Nk) nonzeros
- `test_eps_div_float32_floor` — consistent ε_div < 5 × 10⁻⁵ over 10 random inputs
- `test_six_orders_of_magnitude_reduction` — log₁₀(ρ) > 4 (paper: 6 decades)

> **Note:** Tests that load `data/fixed_G.pt` require running `generate_data.py` first.

---

## Important Notes

### The Boundary Condition Paradox (Section 4.9)

Naively applying the Helmholtz projection over all N nodes (using `G_full`) corrupts hard Dirichlet boundary velocities and produces ≈74% error in aerodynamic drag. The **True Algebraic Transplant** resolves this by restricting projection to interior DOFs only (`G_int`), reducing the correction norm to ‖q‖₂ = 1.017 × 10⁻⁶ and drag discrepancy to 3.363 × 10⁻⁵%.

The solver automatically uses the interior-restricted operator. To use the full-domain operator (not recommended for force evaluation), access `solver.G` instead of `solver.G_int`.

### Precision Hierarchy (Remark 2)

Operator assembly (G, L = GGᵀ, Cholesky factorisation) is performed in **float64** for O(10⁻¹³) divergence residual. Neural network inference uses **float32** (standard PyTorch), raising ε_div to ≈ 4 × 10⁻⁵. This is an arithmetic consequence, not a methodological limitation.

### Force Recovery Accuracy Prerequisite (Section 5.3)

The Algebraic Transplant is a **necessary but not sufficient** condition for engineering-grade force accuracy. With smoothed neural noise, the 5% engineering tolerance for drag prediction requires upstream L₂ velocity error below approximately **3–4%**. The current model achieves ≈13.75% velocity error, which corresponds to ≈30% mean drag error under random noise (Table 18). Improving the neural backbone accuracy is identified as the primary path to engineering-grade force output.

---

## Citation

```bibtex
@article{shafiei2026algebraic,
  title   = {Algebraic Transplant of Meshless Discrete Operators into Graph Neural 
             Architectures: Numerically Consistent Neural Operators for Parametric 
             Incompressible Flows},
  author  = {Shafiei, Amirkeivan and Mosavi Nezhad, Seyed Mojtaba},
  journal = {},
  year    = {2026},
  note    = {University of Birjand}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Contact

- **Amirkeivan Shafiei** — k.shafiei@birjand.ac.ir — Computer Engineering, University of Birjand
- **Seyed Mojtaba Mosavi Nezhad** — mojtaba.mosavi@birjand.ac.ir — Civil Engineering, University of Birjand
