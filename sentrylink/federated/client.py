"""Federated client: trains locally, returns a weight delta for secure aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ROUNDS_DEFAULT_LR
from .model import LogisticModel, sigmoid


@dataclass
class FederatedClient:
    org_id: str
    x: np.ndarray
    y: np.ndarray
    lr: float = ROUNDS_DEFAULT_LR
    l2: float = 1e-3

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64).reshape(-1)
        if len(self.x) != len(self.y):
            raise ValueError("x/y length mismatch")
        if len(self.x) == 0:
            raise ValueError("client has no data")
        self.dim = self.x.shape[1]

    @property
    def n_samples(self) -> int:
        return len(self.x)

    def local_train(self, global_model: LogisticModel, epochs: int = 2) -> np.ndarray:
        """SGD on private data; returns delta = local - global (flat vector)."""
        if global_model.dim != self.dim:
            raise ValueError("model dim mismatch")
        w = global_model.flat.copy()
        n = len(self.x)
        rng = np.random.default_rng(abs(hash(self.org_id)) % (2**32))
        for _ in range(epochs):
            order = rng.permutation(n)
            for idx in order:
                xi = self.x[idx]
                yi = self.y[idx]
                z = float(w[0] + w[1:] @ xi)
                p = float(sigmoid(np.asarray([z]))[0])
                err = p - yi
                grad = np.empty_like(w)
                grad[0] = err
                grad[1:] = err * xi + self.l2 * w[1:]
                w -= self.lr * grad
        return w - global_model.flat
