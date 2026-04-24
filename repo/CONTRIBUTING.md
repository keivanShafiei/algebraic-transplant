# Contributing

Thank you for your interest in the Algebraic Transplant project.

## Development Setup

```bash
git clone https://github.com/kshafiei/algebraic-transplant.git
cd algebraic-transplant
pip install -e ".[dev]"
```

## Running Tests

All tests require `data/fixed_G.pt` to exist. Generate it first:

```bash
python scripts/generate_data.py --n 225 --ns 10 --device cpu   # quick test run
pytest tests/ -v
```

## Code Style

- Format with `black` (line length 100)
- Sort imports with `isort`
- Follow existing docstring conventions (NumPy style)

## Reporting Issues

Please include:
- PyTorch version (`python -c "import torch; print(torch.__version__)"`)
- CUDA version if applicable
- Full traceback
- Minimal reproducing example
