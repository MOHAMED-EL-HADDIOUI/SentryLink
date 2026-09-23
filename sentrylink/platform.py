"""Top-level orchestration: protected queries that combine policy + MPC + DP.

This is the single entry point organizations and the API use to obtain
aggregated intelligence. No raw row ever leaves an organization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from .config import DEFAULT_DELTA, MAX_ORG_CONTRIB
from .crypto.differential_privacy import (
    PrivacyAccountant,
    PrivacyBudget,
    laplace_noise,
)
from .federated.client import FederatedClient
from .federated.server import FederatedServer, RoundResult
from .mpc.stats import run_correlation, run_histogram, run_variance
from .governance.audit import AuditLog
from .governance.policy import PolicyEngine, QueryRequest, default_policy_for
from .governance.registry import Organization, Registry

NODE_IDS = ["node-a", "node-b"]


@dataclass
class QueryResult:
    query_id: str
    metric: str
    value: object
    raw_value: object
    participants: list[str]
    epsilon: float
    delta: float
    noise_sigma: float
    purpose: str
    ts: float


@dataclass
class SentryLinkPlatform:
    """In-process platform wiring registry + policy + audit + compute."""

    registry: Registry = field(default_factory=Registry)
    policy: PolicyEngine | None = None
    audit: AuditLog = field(default_factory=AuditLog)
    accountant: PrivacyAccountant | None = None
    node_ids: list[str] = field(default_factory=lambda: list(NODE_IDS))
    servers: dict[str, FederatedServer] = field(default_factory=dict)
    last_round: dict[str, RoundResult] = field(default_factory=dict)

    def __post_init__(self):
        if self.policy is None:
            self.policy = PolicyEngine(self.registry)
        if self.accountant is None:
            from .config import DEFAULT_EPSILON

            self.accountant = PrivacyAccountant(
                limit=PrivacyBudget(DEFAULT_EPSILON * 100, DEFAULT_DELTA * 100)
            )

    # ---- onboarding --------------------------------------------------
    def join(self, name: str, domain: str, sector_group: str) -> Organization:
        org = self.registry.register(name, domain, sector_group)
        self.policy.set_policy(org.org_id, default_policy_for(org))
        self.audit.record("org.join", org_id=org.org_id, name=name, domain=domain)
        return org

    def set_consent(self, org_id: str, allowed_metrics: set[str]) -> None:
        # Replace only the allow-list; preserve the org's other policy terms
        # (e.g. healthcare's stricter max_epsilon_per_query).
        org = self.registry.get(org_id)  # existence check (KeyError if unknown)
        base = self.policy.policies.get(org_id, default_policy_for(org))
        self.policy.set_policy(
            org_id, replace(base, allowed_metrics=frozenset(allowed_metrics))
        )
        self.audit.record("consent.update", org_id=org_id, metrics=sorted(allowed_metrics))

    # ---- federated intelligence -------------------------------------
    def run_federated_round(
        self,
        sector_group: str,
        domain: str,
        clients_by_org: dict[str, FederatedClient],
        *,
        epsilon: float = 1.0,
        delta: float = DEFAULT_DELTA,
        epochs: int = 2,
        drop: list[str] | None = None,
        purpose: str = "federated model improvement",
    ) -> RoundResult:
        req = QueryRequest(
            metric="federated_model_round",
            sector_group=sector_group,
            domain=domain,
            epsilon=epsilon,
            delta=delta,
            purpose=purpose,
        )
        decision = self.policy.evaluate(req)
        decision.raise_if_denied()

        roster_clients = [
            clients_by_org[oid] for oid in decision.participants if oid in clients_by_org
        ]
        if len(roster_clients) < len(decision.participants):
            missing = set(decision.participants) - {c.org_id for c in roster_clients}
            raise ValueError(f"missing client data for orgs: {sorted(missing)}")

        key = f"{sector_group}:{domain}"
        if key not in self.servers:
            self.servers[key] = FederatedServer(dim=roster_clients[0].dim)
        server = self.servers[key]

        self.accountant.charge(req.budget, purpose=purpose, subject=key)
        result = server.run_round(
            roster_clients, drop=drop, epochs=epochs, apply_dp=True, epsilon=epsilon, delta=delta
        )
        self.last_round[key] = result
        self.audit.record(
            "federated.round",
            round_id=result.round_id,
            participants=result.participants,
            dropped=result.dropped,
            epsilon=epsilon,
            delta=delta,
            eval=result.eval_stats,
        )
        return result

    # ---- protected aggregate queries --------------------------------
    def histogram(
        self,
        sector_group: str,
        domain: str,
        org_buckets: dict[str, list[int]],
        *,
        labels: list[str] | None = None,
        epsilon: float = 1.0,
        delta: float = DEFAULT_DELTA,
        purpose: str = "distribution insight",
    ) -> QueryResult:
        req = QueryRequest(
            metric="histogram",
            sector_group=sector_group,
            domain=domain,
            epsilon=epsilon,
            delta=delta,
            purpose=purpose,
        )
        decision = self.policy.evaluate(req)
        decision.raise_if_denied()

        contrib = {
            oid: self.policy.cap_contributions(org_buckets[oid])
            for oid in decision.participants
            if oid in org_buckets
        }
        if len(contrib) < len(decision.participants):
            raise ValueError("all consented orgs must supply bucket counts")

        raw = run_histogram(contrib, self.node_ids)

        # Add/remove sensitivity of the joint vector: one org's contribution
        # is L2-projected onto the MAX_ORG_CONTRIB ball before sharing.
        sens = float(MAX_ORG_CONTRIB)
        # Pure ε-DP (Laplace): charge epsilon only, delta stays 0.
        self.accountant.charge(
            PrivacyBudget(req.epsilon, 0.0), purpose=purpose, subject=f"hist:{sector_group}"
        )
        noise = laplace_noise(len(raw), sens, epsilon)
        noisy = [int(v) for v in np.rint(np.asarray(raw, dtype=np.float64) + noise)]

        if labels is not None and len(labels) != len(noisy):
            raise ValueError("labels length must match buckets")

        value = (
            {labels[i]: noisy[i] for i in range(len(noisy))}
            if labels
            else noisy
        )
        self.audit.record(
            "query.histogram",
            query_id=decision.query_id,
            participants=decision.participants,
            epsilon=epsilon,
            delta=delta,
            buckets=len(noisy),
        )
        return QueryResult(
            query_id=decision.query_id,
            metric="histogram",
            value=value,
            raw_value=raw,
            participants=decision.participants,
            epsilon=epsilon,
            delta=delta,
            noise_sigma=float(noise.std()) if len(noise) else 0.0,
            purpose=purpose,
            ts=time.time(),
        )

    def variance(
        self,
        sector_group: str,
        domain: str,
        org_values: dict[str, list[float]],
        *,
        epsilon: float = 1.0,
        delta: float = DEFAULT_DELTA,
        purpose: str = "dispersion insight",
    ) -> QueryResult:
        req = QueryRequest(
            metric="variance",
            sector_group=sector_group,
            domain=domain,
            epsilon=epsilon,
            delta=delta,
            purpose=purpose,
        )
        decision = self.policy.evaluate(req)
        decision.raise_if_denied()
        vals = {oid: org_values[oid] for oid in decision.participants}
        if len(vals) != len(decision.participants):
            raise ValueError("all consented orgs must supply values")

        raw = run_variance(vals, self.node_ids)
        # variance is released with unit-scale sensitivity after bounded
        # org contributions; production would use smooth sensitivity here.
        self.accountant.charge(
            PrivacyBudget(req.epsilon, 0.0), purpose=purpose, subject=f"var:{sector_group}"
        )
        noise = laplace_noise((), 1.0, epsilon)
        noisy = max(0.0, float(raw) + float(noise))

        self.audit.record(
            "query.variance",
            query_id=decision.query_id,
            participants=decision.participants,
            epsilon=epsilon,
            delta=delta,
        )
        return QueryResult(
            query_id=decision.query_id,
            metric="variance",
            value=noisy,
            raw_value=raw,
            participants=decision.participants,
            epsilon=epsilon,
            delta=delta,
            noise_sigma=float(np.std(noise)) if np.ndim(noise) else abs(float(noise)),
            purpose=purpose,
            ts=time.time(),
        )

    def correlation(
        self,
        sector_group: str,
        domain: str,
        org_pairs: dict[str, tuple[list[float], list[float]]],
        *,
        epsilon: float = 1.0,
        delta: float = DEFAULT_DELTA,
        purpose: str = "association insight",
    ) -> QueryResult:
        req = QueryRequest(
            metric="correlation",
            sector_group=sector_group,
            domain=domain,
            epsilon=epsilon,
            delta=delta,
            purpose=purpose,
        )
        decision = self.policy.evaluate(req)
        decision.raise_if_denied()
        pairs = {oid: org_pairs[oid] for oid in decision.participants}
        if len(pairs) != len(decision.participants):
            raise ValueError("all consented orgs must supply (x, y) pairs")

        raw = run_correlation(pairs, self.node_ids)
        # r ∈ [-1, 1]: add/remove sensitivity ≤ 2; Laplace keeps the release
        # informative at practical epsilon values.
        self.accountant.charge(
            PrivacyBudget(req.epsilon, 0.0), purpose=purpose, subject=f"corr:{sector_group}"
        )
        noise = laplace_noise((), 2.0, epsilon)
        noisy = float(np.clip(raw + float(noise), -1.0, 1.0))

        self.audit.record(
            "query.correlation",
            query_id=decision.query_id,
            participants=decision.participants,
            epsilon=epsilon,
            delta=delta,
        )
        return QueryResult(
            query_id=decision.query_id,
            metric="correlation",
            value=noisy,
            raw_value=raw,
            participants=decision.participants,
            epsilon=epsilon,
            delta=delta,
            noise_sigma=abs(float(noise)),
            purpose=purpose,
            ts=time.time(),
        )

    def budget_report(self) -> dict:
        return {
            "limit_epsilon": self.accountant.limit.epsilon,
            "limit_delta": self.accountant.limit.delta,
            "spent_epsilon": self.accountant.spent.epsilon,
            "spent_delta": self.accountant.spent.delta,
            "remaining_epsilon": self.accountant.remaining.epsilon,
            "remaining_delta": self.accountant.remaining.delta,
            "events": len(self.accountant.events),
        }
