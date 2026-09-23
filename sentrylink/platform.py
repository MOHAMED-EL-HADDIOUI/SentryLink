"""Top-level orchestration: protected queries that combine policy + MPC + DP.

This is the single entry point organizations and the API use to obtain
aggregated intelligence. No raw row ever leaves an organization.
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from .config import DEFAULT_DELTA, DEFAULT_EPSILON, MAX_ORG_CONTRIB, UPDATE_CLIP
from .crypto.differential_privacy import (
    PrivacyAccountant,
    PrivacyBudget,
    gaussian_rdp_cost,
    laplace_noise,
    laplace_rdp_cost,
)
from .federated.client import FederatedClient
from .federated.server import FederatedServer, RoundResult
from .mpc.stats import run_correlation, run_histogram, run_variance
from .governance.audit import AuditLog
from .governance.policy import PolicyEngine, QueryRequest, default_policy_for
from .governance.registry import Organization, Registry
from .storage import InMemoryStateStore, StateStore, StoreTx, codec

NODE_IDS = ["node-a", "node-b"]


def _locked(method):
    """Serialize mutating platform calls (in-memory objects + store commit)."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


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
    store: StateStore = field(default_factory=InMemoryStateStore, repr=False, compare=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def __post_init__(self):
        if self.policy is None:
            self.policy = PolicyEngine(self.registry)
        if self.accountant is None:
            self.accountant = PrivacyAccountant(
                limit=PrivacyBudget(DEFAULT_EPSILON * 100, DEFAULT_DELTA * 100)
            )
        # Deterministic recovery: a non-empty store replaces the live objects.
        # Explicit constructor args act as seed defaults for an empty store.
        self._restore()

    # ---- persistence -------------------------------------------------
    def _restore(self) -> bool:
        """Rebuild live objects from the store; False when the store is empty."""
        state = self.store.load()
        if state is None:
            return False
        registry = Registry()
        for raw_org in state.orgs:
            registry.add_restored(codec.decode_org(raw_org))
        policy = PolicyEngine(registry)
        for org_id, raw_policy in state.policies.items():
            try:
                policy.set_policy(org_id, codec.decode_policy(raw_policy))
            except KeyError:
                continue  # policy for an unknown org: skip defensively
        self.registry = registry
        self.policy = policy
        if state.budget is not None:
            self.accountant = codec.decode_budget(state.budget)
        self.audit = AuditLog()
        self.audit.restore(state.audit)
        servers: dict[str, FederatedServer] = {}
        for key, raw_server in state.servers.items():
            servers[key] = codec.decode_server(key, raw_server, state.rounds.get(key, []))
        self.servers = servers
        self.last_round = {k: s.history[-1] for k, s in servers.items() if s.history}
        return True

    def _reset_empty(self) -> None:
        self.registry = Registry()
        self.policy = PolicyEngine(self.registry)
        self.accountant = PrivacyAccountant(
            limit=PrivacyBudget(DEFAULT_EPSILON * 100, DEFAULT_DELTA * 100)
        )
        self.audit = AuditLog()
        self.servers = {}
        self.last_round = {}

    def _commit(self, persist: Callable[[StoreTx], None]) -> None:
        """Commit one atomic store bundle; reconverge memory on failure."""
        try:
            with self.store.transaction() as tx:
                persist(tx)
        except Exception:
            if not self._restore():
                self._reset_empty()
            raise

    # ---- onboarding --------------------------------------------------
    @_locked
    def join(self, name: str, domain: str, sector_group: str) -> Organization:
        org = self.registry.register(name, domain, sector_group)
        self.policy.set_policy(org.org_id, default_policy_for(org))
        entry = self.audit.record("org.join", org_id=org.org_id, name=name, domain=domain)
        policy = self.policy.policies[org.org_id]

        def _persist(tx: StoreTx) -> None:
            tx.save_org(codec.encode_org(org))
            tx.save_policy(org.org_id, codec.encode_policy(policy))
            tx.append_audit(entry)

        self._commit(_persist)
        return org

    @_locked
    def set_consent(self, org_id: str, allowed_metrics: set[str]) -> None:
        # Replace only the allow-list; preserve the org's other policy terms
        # (e.g. healthcare's stricter max_epsilon_per_query).
        org = self.registry.get(org_id)  # existence check (KeyError if unknown)
        base = self.policy.policies.get(org_id, default_policy_for(org))
        self.policy.set_policy(
            org_id, replace(base, allowed_metrics=frozenset(allowed_metrics))
        )
        entry = self.audit.record("consent.update", org_id=org_id, metrics=sorted(allowed_metrics))
        policy = self.policy.policies[org_id]

        def _persist(tx: StoreTx) -> None:
            tx.save_policy(org_id, codec.encode_policy(policy))
            tx.append_audit(entry)

        self._commit(_persist)

    # ---- federated intelligence -------------------------------------
    @_locked
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

        result = server.run_round(
            roster_clients, drop=drop, epochs=epochs, apply_dp=True, epsilon=epsilon, delta=delta
        )
        # Charge after a successful round (failed rounds consume no budget).
        # The released mean has replace-one sensitivity 2C/k (see server).
        sens = 2.0 * UPDATE_CLIP / len(result.participants)
        self.accountant.charge(
            req.budget,
            purpose=purpose,
            subject=key,
            rdp=gaussian_rdp_cost(sens, result.dp_sigma) if result.dp_applied else None,
        )
        self.last_round[key] = result
        entry = self.audit.record(
            "federated.round",
            round_id=result.round_id,
            participants=result.participants,
            dropped=result.dropped,
            epsilon=epsilon,
            delta=delta,
            eval=result.eval_stats,
        )

        def _persist(tx: StoreTx) -> None:
            tx.save_budget(codec.encode_budget(self.accountant))
            tx.save_server(codec.encode_server(key, server))
            tx.append_round(key, codec.encode_round(key, result))
            tx.append_audit(entry)

        self._commit(_persist)
        return result

    # ---- protected aggregate queries --------------------------------
    @_locked
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
            PrivacyBudget(req.epsilon, 0.0),
            purpose=purpose,
            subject=f"hist:{sector_group}",
            rdp=laplace_rdp_cost(req.epsilon),
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
        entry = self.audit.record(
            "query.histogram",
            query_id=decision.query_id,
            participants=decision.participants,
            epsilon=epsilon,
            delta=delta,
            buckets=len(noisy),
        )

        def _persist_hist(tx: StoreTx) -> None:
            tx.save_budget(codec.encode_budget(self.accountant))
            tx.append_audit(entry)

        self._commit(_persist_hist)
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

    @_locked
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
            PrivacyBudget(req.epsilon, 0.0),
            purpose=purpose,
            subject=f"var:{sector_group}",
            rdp=laplace_rdp_cost(req.epsilon),
        )
        noise = laplace_noise((), 1.0, epsilon)
        noisy = max(0.0, float(raw) + float(noise))

        entry = self.audit.record(
            "query.variance",
            query_id=decision.query_id,
            participants=decision.participants,
            epsilon=epsilon,
            delta=delta,
        )

        def _persist_var(tx: StoreTx) -> None:
            tx.save_budget(codec.encode_budget(self.accountant))
            tx.append_audit(entry)

        self._commit(_persist_var)
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

    @_locked
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
            PrivacyBudget(req.epsilon, 0.0),
            purpose=purpose,
            subject=f"corr:{sector_group}",
            rdp=laplace_rdp_cost(req.epsilon),
        )
        noise = laplace_noise((), 2.0, epsilon)
        noisy = float(np.clip(raw + float(noise), -1.0, 1.0))

        entry = self.audit.record(
            "query.correlation",
            query_id=decision.query_id,
            participants=decision.participants,
            epsilon=epsilon,
            delta=delta,
        )

        def _persist_corr(tx: StoreTx) -> None:
            tx.save_budget(codec.encode_budget(self.accountant))
            tx.append_audit(entry)

        self._commit(_persist_corr)
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
            "rdp_epsilon_spent": self.accountant.rdp_epsilon(),
            "rdp_complete": self.accountant.rdp_complete,
        }
