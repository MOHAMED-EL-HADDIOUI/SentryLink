"""Federated client: trains locally, returns a weight delta for secure aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ROUNDS_DEFAULT_LR
from ..errors import InvalidDataError, ModelDimensionMismatchError
from .model import LogisticModel, evaluate_sufficient_stats, sigmoid


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
            raise InvalidDataError("x/y length mismatch")
        if len(self.x) == 0:
            raise InvalidDataError("client has no data")
        if self.x.ndim != 2:
            raise InvalidDataError("x must be a 2-D feature matrix")
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.y)):
            raise InvalidDataError("features and labels must be finite (no NaN/inf)")
        if not set(np.unique(self.y)) <= {0.0, 1.0}:
            raise InvalidDataError("labels must be binary (0/1)")
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise InvalidDataError("learning rate must be positive and finite")
        if not np.isfinite(self.l2) or self.l2 < 0:
            raise InvalidDataError("l2 must be non-negative and finite")
        self.dim = self.x.shape[1]

    @property
    def n_samples(self) -> int:
        return len(self.x)

    def local_train(self, global_model: LogisticModel, epochs: int = 2) -> np.ndarray:
        """SGD on private data; returns delta = local - global (flat vector)."""
        if global_model.dim != self.dim:
            raise ModelDimensionMismatchError(
                f"model dim {global_model.dim} != client dim {self.dim}"
            )
        if epochs < 1:
            raise InvalidDataError("epochs must be >= 1")
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

    def eval_stats(self, global_model: LogisticModel) -> np.ndarray:
        """Local (n, sum_log_loss, n_correct) as a float vector for masked pooling.

        Computed client-side so per-client tallies never cross the boundary
        in the clear — only their masked sum is ever opened.
        """
        if global_model.dim != self.dim:
            raise ModelDimensionMismatchError(
                f"model dim {global_model.dim} != client dim {self.dim}"
            )
        n, loss_sum, correct = evaluate_sufficient_stats(global_model, self.x, self.y)
        return np.array([float(n), float(loss_sum), float(correct)], dtype=np.float64)
