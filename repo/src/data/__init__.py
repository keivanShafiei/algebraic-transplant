"""Data utilities: cavity mesh generation and dataset classes."""
from .cavity import generate_cavity_points
from .dataset import ParametricCavityDataset, PrecomputedDataset
from .synthetic import generate_synthetic_streamfunction
