"""Federated server: orchestrates rounds with secure aggregation + optional DP."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import DEFAULT_DELTA, DEFAULT_EPSILON, UPDATE_CLIP
from ..crypto.differential_privacy import gaussian_sigma
from ..crypto.secure_aggregation import SecureAggregator
from .client import FederatedClient
from .model import LogisticModel, evaluate_sufficient_stats


@dataclass
class RoundResult:
    round_id: str
    participants: list[str]
    dropped: list[str]
    aggregate_delta: np.ndarray
    model: LogisticModel
    eval_stats: dict
    dp_applied: bool
    epsilon_used: float
    dp_sigma: float = 0.0  # Gaussian noise scale actually applied (0 when no DP)


@dataclass
class FederatedServer:
    dim: int
    model: LogisticModel | None = None
    history: list[RoundResult] = field(default_factory=list)

    def __post_init__(self):
        if self.model is None:
            self.model = LogisticModel.zeros(self.dim)
        elif self.model.dim != self.dim:
            raise ValueError("model dim mismatch")

    def run_round(
        self,
        clients: list[FederatedClient],
        *,
        drop: list[str] | None = None,
        epochs: int = 2,
        apply_dp: bool = True,
        epsilon: float = DEFAULT_EPSILON,
        delta: float = DEFAULT_DELTA,
        rng: np.random.Generator | None = None,
    ) -> RoundResult:
        drop = drop or []
        roster = [c.org_id for c in clients]
        if len(set(roster)) != len(roster):
            raise ValueError("duplicate org ids in roster")

        assert self.model is not None
        round_id = SecureAggregator.new_round_id()
        agg = SecureAggregator(round_id=round_id, roster=roster, dim=self.model.flat.shape[0])

        for c in clients:
            agg.client_register(c.org_id)

        active = [c for c in clients if c.org_id not in drop]
        if not active:
            raise RuntimeError("no active clients in round")

        deltas = {c.org_id: c.local_train(self.model, epochs=epochs) for c in active}

        for org_id, update in deltas.items():
            agg.client_mask_and_send(org_id, update)

        dropped = [oid for oid in roster if oid in drop]
        for survivor in deltas:
            for d in dropped:
                agg.client_reveal_seed(survivor, d)

        aggregate_sum = agg.finalize()

        n_active = len(active)
        # FedAvg: average the clipped updates (sensitivity scales with 1/k)
        aggregate = aggregate_sum / float(n_active)

        dp_applied = False
        eps_used = 0.0
        dp_sigma = 0.0
        if apply_dp:
            # Central Gaussian DP on the released mean update. Each client's
            # update is L2-clipped to UPDATE_CLIP; replace-one adjacency on
            # the mean has sensitivity 2C/k.
            sens = 2.0 * UPDATE_CLIP / n_active
            sigma = gaussian_sigma(sens, epsilon, delta)
            rng = rng or np.random.default_rng()
            aggregate = aggregate + rng.normal(0.0, sigma, size=aggregate.shape)
            dp_applied = True
            eps_used = epsilon
            dp_sigma = sigma

        self.model = LogisticModel.from_flat(self.dim, self.model.flat + aggregate)

        eval_stats = self._secure_eval(active, self.model)
        result = RoundResult(
            round_id=round_id,
            participants=sorted(deltas),
            dropped=sorted(dropped),
            aggregate_delta=aggregate,
            model=LogisticModel.from_flat(self.dim, self.model.flat.copy()),
            eval_stats=eval_stats,
            dp_applied=dp_applied,
            epsilon_used=eps_used,
            dp_sigma=dp_sigma,
        )
        self.history.append(result)
        return result

    @staticmethod
    def _secure_eval(clients: list[FederatedClient], model: LogisticModel) -> dict:
        """Pool additive sufficient stats across clients — per-client metrics
        are never exposed, only the cohort totals."""
        n_total = 0
        loss_total = 0.0
        correct_total = 0
        for c in clients:
            n, loss_sum, correct = evaluate_sufficient_stats(model, c.x, c.y)
            n_total += n
            loss_total += loss_sum
            correct_total += correct
        return {
            "n": n_total,
            "mean_loss": loss_total / n_total if n_total else float("nan"),
            "accuracy": correct_total / n_total if n_total else float("nan"),
            "n_correct": correct_total,
        }
