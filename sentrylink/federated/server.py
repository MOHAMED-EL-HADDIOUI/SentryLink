"""Federated server: orchestrates rounds with secure aggregation + optional DP."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import DEFAULT_DELTA, DEFAULT_EPSILON, UPDATE_CLIP
from ..crypto.differential_privacy import gaussian_sigma
from ..crypto.secure_aggregation import SecAggClient, SecAggServer, new_round_id
from ..errors import DuplicateRosterError, ModelDimensionMismatchError, ProtocolError
from .client import FederatedClient
from .model import LogisticModel


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
    delta_used: float = DEFAULT_DELTA


@dataclass
class FederatedServer:
    dim: int
    model: LogisticModel | None = None
    history: list[RoundResult] = field(default_factory=list)

    def __post_init__(self):
        if self.model is None:
            self.model = LogisticModel.zeros(self.dim)
        elif self.model.dim != self.dim:
            raise ModelDimensionMismatchError(
                f"model dim {self.model.dim} != server dim {self.dim}"
            )

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
            raise DuplicateRosterError("duplicate org ids in roster")

        assert self.model is not None
        round_id = new_round_id()
        agg = SecAggServer(round_id=round_id, roster=roster, dim=self.model.flat.shape[0])
        parties = {c.org_id: SecAggClient(c.org_id, round_id, roster, agg.dim) for c in clients}

        for oid, party in parties.items():
            agg.register(oid, party.public_key)
        keys = agg.public_keys()

        active = [c for c in clients if c.org_id not in drop]
        if not active:
            raise ProtocolError("no active clients in round")

        deltas = {c.org_id: c.local_train(self.model, epochs=epochs) for c in active}

        for org_id, update in deltas.items():
            agg.receive(parties[org_id].mask_and_send(update, keys))

        dropped = [oid for oid in roster if oid in drop]
        for survivor in deltas:
            for d in dropped:
                agg.add_recovery_seed(
                    survivor, d, parties[survivor].reveal_seed(d, keys[d])
                )

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
            delta_used=delta,
        )
        self.history.append(result)
        return result

    @staticmethod
    def _secure_eval(clients: list[FederatedClient], model: LogisticModel) -> dict:
        """Cohort totals via a masked sub-round over client-side tallies.

        Each client reduces its own rows to (n, loss_sum, n_correct) and only
        the masked sum is opened — per-client tallies never cross the boundary
        in the clear. Tallies skip L2 clipping (clip_bound=None): they are
        already bounded counts, rescaling them would corrupt the sums.
        """
        if not clients:
            return {
                "n": 0,
                "mean_loss": float("nan"),
                "accuracy": float("nan"),
                "n_correct": 0,
                "evaluation": {
                    "mechanism": "masked_sum",
                    "clip_bound": None,
                    "lossless_pre_dp": True,
                    "dp_applied": False,
                },
            }
        roster = [c.org_id for c in clients]
        sub = SecAggServer(round_id=new_round_id(), roster=roster, dim=3)
        parties = {c.org_id: SecAggClient(c.org_id, sub.round_id, roster, 3) for c in clients}
        for oid, party in parties.items():
            sub.register(oid, party.public_key)
        keys = sub.public_keys()
        for c in clients:
            sub.receive(parties[c.org_id].mask_and_send(c.eval_stats(model), keys, clip_bound=None))
        total = sub.finalize()  # [n, loss_sum, n_correct]
        n_total = int(round(float(total[0])))
        loss_total = float(total[1])
        correct_total = int(round(float(total[2])))
        return {
            "n": n_total,
            "mean_loss": loss_total / n_total if n_total else float("nan"),
            "accuracy": correct_total / n_total if n_total else float("nan"),
            "n_correct": correct_total,
            # Honest runtime metadata: tallies are masked-summed exactly
            # (never clipped) and released WITHOUT DP noise.
            "evaluation": {
                "mechanism": "masked_sum",
                "clip_bound": None,
                "lossless_pre_dp": True,
                "dp_applied": False,
            },
        }
