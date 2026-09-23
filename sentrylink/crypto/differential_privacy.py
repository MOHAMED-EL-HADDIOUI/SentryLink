"""Gaussian differential privacy: noise calibration, budgets, accountant."""

from __future__ import annotations

import math
import secrets
import time
from dataclasses import dataclass, field

import numpy as np


def gaussian_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """Classic analytic calibration: sigma >= sens * sqrt(2 ln(1.25/delta)) / eps."""
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def laplace_noise(
    shape, sensitivity: float, epsilon: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Pure ε-DP Laplace mechanism: scale = sensitivity / epsilon.

    Preferred for low-dimensional releases (histograms, scalars) where a
    δ-free guarantee is desirable and the Gaussian δ-term would dominate.
    """
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    rng = rng or np.random.default_rng()
    scale = sensitivity / epsilon
    return rng.laplace(0.0, scale, size=shape)


def gaussian_noise(
    shape, sensitivity: float, epsilon: float, delta: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    sigma = gaussian_sigma(sensitivity, epsilon, delta)
    return rng.normal(0.0, sigma, size=shape)


@dataclass(frozen=True)
class PrivacyBudget:
    epsilon: float
    delta: float

    def __post_init__(self):
        if self.epsilon < 0 or self.delta < 0:
            raise ValueError("budget components must be non-negative")

    @property
    def is_zero(self) -> bool:
        return self.epsilon == 0 and self.delta == 0

    def __add__(self, other: "PrivacyBudget") -> "PrivacyBudget":
        return PrivacyBudget(self.epsilon + other.epsilon, self.delta + other.delta)


@dataclass
class PrivacyAccountant:
    """Tracks epsilon/delta spend per purpose (basic composition)."""

    limit: PrivacyBudget
    spent: PrivacyBudget = field(default_factory=lambda: PrivacyBudget(0.0, 0.0))
    events: list[dict] = field(default_factory=list)

    def charge(self, cost: PrivacyBudget, purpose: str, subject: str) -> dict:
        if cost.epsilon < 0 or cost.delta < 0:
            raise ValueError("cost must be non-negative")
        projected = self.spent + cost
        if projected.epsilon > self.limit.epsilon + 1e-12:
            raise PermissionError(
                f"epsilon budget exceeded for {subject}: "
                f"{projected.epsilon:.4f} > {self.limit.epsilon:.4f}"
            )
        if projected.delta > self.limit.delta + 1e-12:
            raise PermissionError(
                f"delta budget exceeded for {subject}: "
                f"{projected.delta:.6g} > {self.limit.delta:.6g}"
            )
        self.spent = projected
        record = {
            "id": secrets.token_hex(8),
            "ts": time.time(),
            "purpose": purpose,
            "subject": subject,
            "epsilon_cost": cost.epsilon,
            "delta_cost": cost.delta,
            "spent_epsilon": self.spent.epsilon,
            "spent_delta": self.spent.delta,
        }
        self.events.append(record)
        return record

    @property
    def remaining(self) -> PrivacyBudget:
        return PrivacyBudget(
            max(0.0, self.limit.epsilon - self.spent.epsilon),
            max(0.0, self.limit.delta - self.spent.delta),
        )
