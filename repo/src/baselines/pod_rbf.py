import torch
from sklearn.decomposition import PCA
from scipy.interpolate import RBFInterpolator
import numpy as np

class PODRBF:
    """POD + RBF regression on coefficients (very fast baseline)."""
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.pca = None
        self.rbf = None
        self.a_mean = None
        self.a_std = None

    def fit(self, mus, a_refs):
        mus = mus.numpy().reshape(-1, 1)
        a_refs = a_refs.numpy()
        self.a_mean = a_refs.mean(axis=0)
        self.a_std = a_refs.std(axis=0) + 1e-8
        a_norm = (a_refs - self.a_mean) / self.a_std

        self.pca = PCA(n_components=self.n_components)
        coeffs = self.pca.fit_transform(a_norm)

        self.rbf = RBFInterpolator(mus, coeffs, kernel='gaussian', epsilon=0.1)

    def predict(self, mu):
        mu = np.array([[mu.item()]])
        coeffs = self.rbf(mu)
        a_norm = self.pca.inverse_transform(coeffs)
        a = a_norm * self.a_std + self.a_mean
        return torch.from_numpy(a.flatten().astype(np.float32))
