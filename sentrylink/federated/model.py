"""Shared federated model: L2-regularized logistic regression.

Weights are flat float vectors so they plug directly into secure aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class LogisticModel:
    dim: int
    weights: np.ndarray
    bias: float = 0.0

    @staticmethod
    def zeros(dim: int) -> "LogisticModel":
        return LogisticModel(dim=dim, weights=np.zeros(dim, dtype=np.float64), bias=0.0)

    @property
    def flat(self) -> np.ndarray:
        return np.concatenate([[self.bias], self.weights])

    @staticmethod
    def from_flat(dim: int, flat: np.ndarray) -> "LogisticModel":
        flat = np.asarray(flat, dtype=np.float64)
        return LogisticModel(dim=dim, weights=flat[1:].copy(), bias=float(flat[0]))

    def logits(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(self.logits(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(np.int64)


def evaluate_sufficient_stats(
    model: LogisticModel, x: np.ndarray, y: np.ndarray
) -> tuple[int, float, int]:
    """Return (n, sum_log_loss, n_correct) — additive stats safe to secure-agg."""
    p = np.clip(model.predict_proba(x), 1e-9, 1.0 - 1e-9)
    y = y.astype(np.float64)
    loss_sum = float(-np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    correct = int(np.sum((p >= 0.5).astype(np.int64) == y.astype(np.int64)))
    return len(y), loss_sum, correct
