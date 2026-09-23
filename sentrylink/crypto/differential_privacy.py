"""Differential privacy: noise calibration, RDP moments accounting, budgets.

Two composition regimes are supported:
  - Basic composition (sums of epsilon/delta) — always tracked, and used to
    enforce limits whenever an event arrives without an RDP cost.
  - RDP moments accounting (Abadi et al.; Mironov) — per-event Renyi costs are
    summed per alpha order and converted to (epsilon, delta) on demand. When
    every event so far carries an RDP cost, limits are enforced on the tighter
    RDP-converted epsilon instead of the basic sum.
"""

from __future__ import annotations

import math
import secrets
import time
from dataclasses import dataclass, field

import numpy as np

from ..errors import PrivacyBudgetError


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


# Standard RDP order set (alpha > 1). Conversion optimizes over these.
RDP_ALPHAS: tuple[float, ...] = (
    1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 16.0, 32.0, 64.0,
)


def gaussian_rdp_cost(
    sensitivity: float, sigma: float, alphas=RDP_ALPHAS
) -> dict[float, float]:
    """RDP cost of one Gaussian release: RDP(alpha) = alpha * sens^2 / (2 sigma^2)."""
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    rho = (sensitivity / sigma) ** 2 / 2.0
    return {float(a): float(a) * rho for a in alphas}


def laplace_rdp_cost(epsilon: float, alphas=RDP_ALPHAS) -> dict[float, float]:
    """RDP cost of one Laplace release at pure-DP epsilon (Mironov 2017).

    Closed form, evaluated in a factored way so large alpha * epsilon never
    overflows: rdp = eps + log(w1 + w2 * exp(-(2a-1) * eps)) / (a - 1).
    Pure epsilon-DP implies RDP(alpha) <= epsilon, so clip there as a guard.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    out: dict[float, float] = {}
    for a in alphas:
        a = float(a)
        if a <= 1.0:
            raise ValueError("RDP orders must exceed 1")
        w1 = a / (2.0 * a - 1.0)
        w2 = (a - 1.0) / (2.0 * a - 1.0)
        val = epsilon + math.log(w1 + w2 * math.exp(-(2.0 * a - 1.0) * epsilon)) / (a - 1.0)
        out[a] = min(val, epsilon)
    return out


def rdp_to_epsilon(rdp: dict[float, float], delta: float) -> float:
    """Convert summed RDP costs to (epsilon, delta)-DP: min over orders."""
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    if not rdp:
        raise ValueError("no RDP costs to convert")
    return min(v + math.log(1.0 / delta) / (a - 1.0) for a, v in rdp.items())


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
    """Tracks epsilon/delta spend per purpose.

    Basic-composition sums are always tracked. Each charge may also carry an
    `rdp` cost dict (see gaussian_rdp_cost / laplace_rdp_cost); those are
    summed per alpha order. While every event so far has an RDP cost
    (`rdp_complete`), a charge is allowed when EITHER regime fits the limit:
    the basic sum, or the RDP-converted epsilon at limit.delta. Each regime
    is an independently valid upper bound, so fitting either one is sound
    (this matters both ways: RDP is tighter for many Gaussian compositions,
    basic is tighter for a few Laplace releases where conversion overhead
    dominates). The first event without an RDP cost permanently falls back
    to basic-composition enforcement (RDP totals are still accumulated for
    reporting).
    """

    limit: PrivacyBudget
    spent: PrivacyBudget = field(default_factory=lambda: PrivacyBudget(0.0, 0.0))
    events: list[dict] = field(default_factory=list)
    rdp_totals: dict[float, float] = field(default_factory=dict)
    rdp_complete: bool = True

    def preview(
        self, cost: PrivacyBudget, rdp: dict[float, float] | None = None
    ) -> tuple[bool, str]:
        """Non-mutating admission check: (allowed, accounting_regime).

        Mirrors charge() exactly but records nothing — used by privacy
        previews. Regime is "rdp" when RDP-tracked, else "basic".
        """
        if cost.epsilon < 0 or cost.delta < 0:
            raise ValueError("cost must be non-negative")
        projected = self.spent + cost
        if projected.delta > self.limit.delta + 1e-12:
            return False, "basic"
        if rdp is not None and self.rdp_complete:
            keys = set(self.rdp_totals) | set(rdp)
            proj_rdp = {
                a: self.rdp_totals.get(a, 0.0) + rdp.get(a, 0.0) for a in keys
            }
            rdp_eps = rdp_to_epsilon(proj_rdp, self.limit.delta)
            basic_fits = projected.epsilon <= self.limit.epsilon + 1e-12
            if rdp_eps <= self.limit.epsilon + 1e-12 or basic_fits:
                return True, "rdp"
            return False, "rdp"
        return projected.epsilon <= self.limit.epsilon + 1e-12, "basic"

    def charge(
        self,
        cost: PrivacyBudget,
        purpose: str,
        subject: str,
        *,
        rdp: dict[float, float] | None = None,
        ledger: dict | None = None,
    ) -> dict:
        if cost.epsilon < 0 or cost.delta < 0:
            raise ValueError("cost must be non-negative")
        projected = self.spent + cost
        before = {"epsilon": self.spent.epsilon, "delta": self.spent.delta}
        if projected.delta > self.limit.delta + 1e-12:
            raise PrivacyBudgetError(
                f"delta budget exceeded for {subject}: "
                f"{projected.delta:.6g} > {self.limit.delta:.6g}"
            )
        if rdp is not None and self.rdp_complete:
            keys = set(self.rdp_totals) | set(rdp)
            proj_rdp = {
                a: self.rdp_totals.get(a, 0.0) + rdp.get(a, 0.0) for a in keys
            }
            rdp_eps = rdp_to_epsilon(proj_rdp, self.limit.delta)
            basic_fits = projected.epsilon <= self.limit.epsilon + 1e-12
            if rdp_eps > self.limit.epsilon + 1e-12 and not basic_fits:
                raise PrivacyBudgetError(
                    f"epsilon budget exceeded for {subject}: "
                    f"basic {projected.epsilon:.4f} and RDP {rdp_eps:.4f} "
                    f"both exceed {self.limit.epsilon:.4f}"
                )
            self.rdp_totals = proj_rdp
        else:
            if projected.epsilon > self.limit.epsilon + 1e-12:
                raise PrivacyBudgetError(
                    f"epsilon budget exceeded for {subject}: "
                    f"{projected.epsilon:.4f} > {self.limit.epsilon:.4f}"
                )
            if rdp is not None:
                for a, v in rdp.items():
                    self.rdp_totals[a] = self.rdp_totals.get(a, 0.0) + v
            else:
                self.rdp_complete = False
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
            "rdp_tracked": rdp is not None,
            "ledger": dict(ledger or {}),
            "budget_before": before,
            "budget_after": {"epsilon": self.spent.epsilon, "delta": self.spent.delta},
        }
        self.events.append(record)
        return record

    def rdp_epsilon(self, delta: float | None = None) -> float:
        """RDP-converted epsilon spent so far (0.0 when nothing RDP-tracked)."""
        if not self.rdp_totals:
            return 0.0
        return rdp_to_epsilon(
            self.rdp_totals, self.limit.delta if delta is None else delta
        )

    @property
    def remaining(self) -> PrivacyBudget:
        return PrivacyBudget(
            max(0.0, self.limit.epsilon - self.spent.epsilon),
            max(0.0, self.limit.delta - self.spent.delta),
        )
