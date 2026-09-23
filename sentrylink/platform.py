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

from .config import (
    DEFAULT_DELTA,
    DEFAULT_EPSILON,
    MAX_ORG_CONTRIB,
    PROTOCOL_VERSION,
    QUANT_SCALE,
    UPDATE_CLIP,
)
from .crypto.differential_privacy import (
    RDP_ALPHAS,
    PrivacyAccountant,
    PrivacyBudget,
    gaussian_rdp_cost,
    gaussian_sigma,
    laplace_noise,
    laplace_rdp_cost,
)
from .federated.client import FederatedClient
from .federated.server import FederatedServer, RoundResult
from .mpc.stats import run_correlation, run_histogram, run_variance
from .errors import InvalidDataError, StorageError
from .governance.audit import AuditLog
from .governance.policy import PolicyEngine, QueryRequest, default_policy_for
from .governance.registry import Organization, Registry
from .privacy import PrivacyLedgerEntry, build_release_card
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
    privacy_card: dict | None = None


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
        except StorageError:
            raise
        except Exception as exc:
            if not self._restore():
                self._reset_empty()
            raise StorageError(f"persistent commit failed: {exc}") from exc

    def _audit(self, request_id: str | None, action: str, **details):
        """Audit helper: attaches the request ID when the caller has one."""
        if request_id is not None:
            details["request_id"] = request_id
        return self.audit.record(action, **details)

    @property
    def _policy(self) -> PolicyEngine:
        # Set in __post_init__ (or rebuilt by _restore); the assert turns the
        # Optional field into a statically non-None accessor for method bodies.
        assert self.policy is not None, "platform policy not initialized"
        return self.policy

    @property
    def _accountant(self) -> PrivacyAccountant:
        assert self.accountant is not None, "platform accountant not initialized"
        return self.accountant

    def _accounting(self) -> str:
        return "rdp" if self._accountant.rdp_complete else "basic"

    def _ledger_entry(
        self,
        *,
        scope: str,
        query_type: str,
        mechanism: str,
        epsilon: float,
        delta: float,
        sensitivity: float,
        cohort_size: int,
        clip_bound: float | None,
        purpose: str,
        request_id: str | None,
    ) -> dict:
        return PrivacyLedgerEntry(
            scope=scope,
            query_type=query_type,
            mechanism=mechanism,
            epsilon=float(epsilon),
            delta=float(delta),
            rdp_orders=[float(a) for a in RDP_ALPHAS],
            sensitivity=float(sensitivity),
            cohort_size=int(cohort_size),
            clip_bound=None if clip_bound is None else float(clip_bound),
            purpose=purpose,
            request_id=request_id,
        ).to_dict()

    @staticmethod
    def _preview_spec(metric: str, cohort_size: int) -> tuple[str, float]:
        if metric == "histogram":
            return "laplace", float(MAX_ORG_CONTRIB)
        if metric == "variance":
            return "laplace", 1.0
        if metric == "correlation":
            return "laplace", 2.0
        if metric == "federated_model_round":
            return "gaussian", 2.0 * UPDATE_CLIP / max(1, cohort_size)
        return "unknown", 0.0

    # ---- onboarding --------------------------------------------------
    @_locked
    def join(
        self, name: str, domain: str, sector_group: str, *,
        request_id: str | None = None,
    ) -> Organization:
        org = self.registry.register(name, domain, sector_group)
        self._policy.set_policy(org.org_id, default_policy_for(org))
        entry = self._audit(request_id, "org.join", org_id=org.org_id, name=name, domain=domain)
        policy = self._policy.policies[org.org_id]

        def _persist(tx: StoreTx) -> None:
            tx.save_org(codec.encode_org(org))
            tx.save_policy(org.org_id, codec.encode_policy(policy))
            tx.append_audit(entry)

        self._commit(_persist)
        return org

    @_locked
    def set_consent(
        self, org_id: str, allowed_metrics: set[str], *,
        request_id: str | None = None,
    ) -> None:
        # Replace only the allow-list; preserve the org's other policy terms
        # (e.g. healthcare's stricter max_epsilon_per_query).
        org = self.registry.get(org_id)  # existence check (KeyError if unknown)
        base = self._policy.policies.get(org_id, default_policy_for(org))
        self._policy.set_policy(
            org_id, replace(base, allowed_metrics=frozenset(allowed_metrics))
        )
        entry = self._audit(request_id, "consent.update", org_id=org_id, metrics=sorted(allowed_metrics))
        policy = self._policy.policies[org_id]

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
        request_id: str | None = None,
    ) -> RoundResult:
        req = QueryRequest(
            metric="federated_model_round",
            sector_group=sector_group,
            domain=domain,
            epsilon=epsilon,
            delta=delta,
            purpose=purpose,
        )
        decision = self._policy.evaluate(req)
        decision.raise_if_denied()

        roster_clients = [
            clients_by_org[oid] for oid in decision.participants if oid in clients_by_org
        ]
        if len(roster_clients) < len(decision.participants):
            missing = set(decision.participants) - {c.org_id for c in roster_clients}
            raise InvalidDataError(f"missing client data for orgs: {sorted(missing)}")

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
        self._accountant.charge(
            req.budget,
            purpose=purpose,
            subject=key,
            rdp=gaussian_rdp_cost(sens, result.dp_sigma) if result.dp_applied else None,
            ledger=self._ledger_entry(
                scope=key,
                query_type="federated_model_round",
                mechanism="gaussian",
                epsilon=epsilon,
                delta=delta,
                sensitivity=sens,
                cohort_size=len(result.participants),
                clip_bound=UPDATE_CLIP,
                purpose=purpose,
                request_id=request_id,
            ),
        )
        self.last_round[key] = result
        entry = self._audit(
            request_id,
            "federated.round",
            round_id=result.round_id,
            participants=result.participants,
            dropped=result.dropped,
            epsilon=epsilon,
            delta=delta,
            eval=result.eval_stats,
        )

        def _persist(tx: StoreTx) -> None:
            tx.save_budget(codec.encode_budget(self._accountant))
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
        request_id: str | None = None,
    ) -> QueryResult:
        req = QueryRequest(
            metric="histogram",
            sector_group=sector_group,
            domain=domain,
            epsilon=epsilon,
            delta=delta,
            purpose=purpose,
        )
        decision = self._policy.evaluate(req)
        decision.raise_if_denied()

        contrib = {
            oid: self._policy.cap_contributions(org_buckets[oid])
            for oid in decision.participants
            if oid in org_buckets
        }
        if len(contrib) < len(decision.participants):
            raise InvalidDataError("all consented orgs must supply bucket counts")

        try:
            raw = run_histogram(contrib, self.node_ids)
        except ValueError as exc:
            raise InvalidDataError(str(exc)) from exc

        # Add/remove sensitivity of the joint vector: one org's contribution
        # is L2-projected onto the MAX_ORG_CONTRIB ball before sharing.
        sens = float(MAX_ORG_CONTRIB)
        # Pure ε-DP (Laplace): charge epsilon only, delta stays 0.
        self._accountant.charge(
            PrivacyBudget(req.epsilon, 0.0),
            purpose=purpose,
            subject=f"hist:{sector_group}",
            rdp=laplace_rdp_cost(req.epsilon),
            ledger=self._ledger_entry(
                scope=f"hist:{sector_group}",
                query_type="histogram",
                mechanism="laplace",
                epsilon=req.epsilon,
                delta=0.0,
                sensitivity=sens,
                cohort_size=len(decision.participants),
                clip_bound=MAX_ORG_CONTRIB,
                purpose=purpose,
                request_id=request_id,
            ),
        )
        noise = laplace_noise(len(raw), sens, epsilon)
        noisy = [int(v) for v in np.rint(np.asarray(raw, dtype=np.float64) + noise)]

        if labels is not None and len(labels) != len(noisy):
            raise InvalidDataError("labels length must match buckets")

        value = (
            {labels[i]: noisy[i] for i in range(len(noisy))}
            if labels
            else noisy
        )
        entry = self._audit(
            request_id,
            "query.histogram",
            query_id=decision.query_id,
            participants=decision.participants,
            epsilon=epsilon,
            delta=delta,
            buckets=len(noisy),
        )

        def _persist_hist(tx: StoreTx) -> None:
            tx.save_budget(codec.encode_budget(self._accountant))
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
            privacy_card=build_release_card(
                release="histogram",
                domain=domain,
                cohort_size=len(decision.participants),
                mechanism="laplace",
                epsilon=epsilon,
                delta=0.0,
                sensitivity=float(MAX_ORG_CONTRIB),
                accounting=self._accounting(),
                preprocessing=["per-org L2 contribution cap", "local bucketing"],
                computation="2-node additive secret sharing",
            ),
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
        request_id: str | None = None,
    ) -> QueryResult:
        req = QueryRequest(
            metric="variance",
            sector_group=sector_group,
            domain=domain,
            epsilon=epsilon,
            delta=delta,
            purpose=purpose,
        )
        decision = self._policy.evaluate(req)
        decision.raise_if_denied()
        vals = {oid: org_values[oid] for oid in decision.participants}
        if len(vals) != len(decision.participants):
            raise InvalidDataError("all consented orgs must supply values")

        try:
            raw = run_variance(vals, self.node_ids)
        except ValueError as exc:
            raise InvalidDataError(str(exc)) from exc
        # variance is released with unit-scale sensitivity after bounded
        # org contributions; production would use smooth sensitivity here.
        self._accountant.charge(
            PrivacyBudget(req.epsilon, 0.0),
            purpose=purpose,
            subject=f"var:{sector_group}",
            rdp=laplace_rdp_cost(req.epsilon),
            ledger=self._ledger_entry(
                scope=f"var:{sector_group}",
                query_type="variance",
                mechanism="laplace",
                epsilon=req.epsilon,
                delta=0.0,
                sensitivity=1.0,
                cohort_size=len(decision.participants),
                clip_bound=None,
                purpose=purpose,
                request_id=request_id,
            ),
        )
        noise = laplace_noise((), 1.0, epsilon)
        noisy = max(0.0, float(raw) + float(noise))

        entry = self._audit(
            request_id,
            "query.variance",
            query_id=decision.query_id,
            participants=decision.participants,
            epsilon=epsilon,
            delta=delta,
        )

        def _persist_var(tx: StoreTx) -> None:
            tx.save_budget(codec.encode_budget(self._accountant))
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
            privacy_card=build_release_card(
                release="variance",
                domain=domain,
                cohort_size=len(decision.participants),
                mechanism="laplace",
                epsilon=epsilon,
                delta=0.0,
                sensitivity=1.0,
                accounting=self._accounting(),
                preprocessing=["local sufficient statistics", "output floored at 0"],
                computation="2-node additive secret sharing",
            ),
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
        request_id: str | None = None,
    ) -> QueryResult:
        req = QueryRequest(
            metric="correlation",
            sector_group=sector_group,
            domain=domain,
            epsilon=epsilon,
            delta=delta,
            purpose=purpose,
        )
        decision = self._policy.evaluate(req)
        decision.raise_if_denied()
        pairs = {oid: org_pairs[oid] for oid in decision.participants}
        if len(pairs) != len(decision.participants):
            raise InvalidDataError("all consented orgs must supply (x, y) pairs")

        try:
            raw = run_correlation(pairs, self.node_ids)
        except ValueError as exc:
            raise InvalidDataError(str(exc)) from exc
        # r ∈ [-1, 1]: add/remove sensitivity ≤ 2; Laplace keeps the release
        # informative at practical epsilon values.
        self._accountant.charge(
            PrivacyBudget(req.epsilon, 0.0),
            purpose=purpose,
            subject=f"corr:{sector_group}",
            rdp=laplace_rdp_cost(req.epsilon),
            ledger=self._ledger_entry(
                scope=f"corr:{sector_group}",
                query_type="correlation",
                mechanism="laplace",
                epsilon=req.epsilon,
                delta=0.0,
                sensitivity=2.0,
                cohort_size=len(decision.participants),
                clip_bound=None,
                purpose=purpose,
                request_id=request_id,
            ),
        )
        noise = laplace_noise((), 2.0, epsilon)
        noisy = float(np.clip(raw + float(noise), -1.0, 1.0))

        entry = self._audit(
            request_id,
            "query.correlation",
            query_id=decision.query_id,
            participants=decision.participants,
            epsilon=epsilon,
            delta=delta,
        )

        def _persist_corr(tx: StoreTx) -> None:
            tx.save_budget(codec.encode_budget(self._accountant))
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
            privacy_card=build_release_card(
                release="correlation",
                domain=domain,
                cohort_size=len(decision.participants),
                mechanism="laplace",
                epsilon=epsilon,
                delta=0.0,
                sensitivity=2.0,
                accounting=self._accounting(),
                preprocessing=["local sufficient statistics", "output clipped to [-1, 1]"],
                computation="2-node additive secret sharing",
            ),
        )

    # ---- privacy preview (no spend, no mutation, no audit) ------------
    @_locked
    def preview(
        self,
        metric: str,
        sector_group: str,
        domain: str,
        *,
        epsilon: float = 1.0,
        delta: float = DEFAULT_DELTA,
        request_id: str | None = None,
    ) -> dict:
        """Inspect what a query would consume before executing it.

        Pure read path: evaluates governance, projects budget admission, and
        reports the outcome. Records nothing, charges nothing, persists
        nothing.
        """
        req = QueryRequest(
            metric=metric,
            sector_group=sector_group,
            domain=domain,
            epsilon=epsilon,
            delta=delta,
            purpose="privacy preview",
        )
        decision = self._policy.evaluate(req)
        mechanism, sens = self._preview_spec(metric, len(decision.participants))
        remaining_before = max(
            0.0, self._accountant.limit.epsilon - self._accountant.spent.epsilon
        )
        base: dict = {
            "allowed": False,
            "reason_code": decision.code,
            "reasons": list(decision.reasons),
            "cohort_size": 0,
            "mechanism": mechanism,
            "epsilon_requested": epsilon,
            "delta": delta if mechanism == "gaussian" else 0.0,
            "estimated_cost": epsilon,
            "remaining_budget_before": remaining_before,
            "remaining_budget_after": remaining_before,
            "accounting_regime": "rdp" if self._accountant.rdp_complete else "basic",
            "sensitivity": sens,
            "request_id": request_id,
            "governance": {
                "cohort_floor_satisfied": decision.code
                not in ("COHORT_TOO_SMALL", "HEALTHCARE_COHORT_TOO_SMALL"),
                "sector_scope_valid": decision.code
                not in ("SECTOR_MISMATCH", "DOMAIN_MISMATCH"),
                "consent_valid": decision.code
                not in ("CONSENT_REQUIRED", "QUERY_NOT_ALLOWED"),
            },
        }
        if not decision.allowed:
            return base
        delta_charge = delta if mechanism == "gaussian" else 0.0
        cost = PrivacyBudget(epsilon, delta_charge)
        if mechanism == "laplace":
            rdp = laplace_rdp_cost(epsilon)
        elif mechanism == "gaussian":
            rdp = gaussian_rdp_cost(sens, gaussian_sigma(sens, epsilon, delta))
        else:
            rdp = None
        admitted, regime = self._accountant.preview(cost, rdp)
        base.update(
            allowed=admitted,
            reason_code=None if admitted else "BUDGET_EXHAUSTED",
            reasons=[] if admitted else ["privacy budget exhausted"],
            cohort_size=len(decision.participants),
            remaining_budget_after=max(0.0, remaining_before - epsilon),
            accounting_regime=regime,
        )
        return base

    # ---- model release metadata --------------------------------------
    @_locked
    def model_release_metadata(self, key: str) -> dict:
        """Traceable envelope for a released federated model (all derived)."""
        server = self.servers[key]  # KeyError when no round has run
        last = self.last_round.get(key)
        if last is None and server.history:
            last = server.history[-1]
        participants = list(last.participants) if last else []
        dropped = list(last.dropped) if last else []
        k = len(participants)
        return {
            "model_version": len(server.history),
            "algorithm": "logistic_regression",
            "aggregation": "FedAvg",
            "participants": k,
            "dropouts": len(dropped),
            "participants_total": k + len(dropped),
            "recovery_used": bool(dropped),
            "feature_dimension": server.dim,
            "dp": {
                "mechanism": "gaussian",
                "epsilon": last.epsilon_used if last else 0.0,
                "delta": last.delta_used if last else 0.0,
                "accounting": self._accounting(),
                "sensitivity": 2.0 * UPDATE_CLIP / k if k else 0.0,
            },
            "secure_aggregation": {
                "quantization_scale": QUANT_SCALE,
                "update_clip": UPDATE_CLIP,
                "masked_evaluation": True,
            },
            "protocol_version": PROTOCOL_VERSION,
        }

    @_locked
    def federated_release_card(self, key: str, domain: str) -> dict:
        """Privacy Card for a released federated model (derived, not stored)."""
        meta = self.model_release_metadata(key)
        dp = meta["dp"]
        return build_release_card(
            release="federated_model",
            domain=domain,
            cohort_size=meta["participants"],
            mechanism="gaussian",
            epsilon=dp["epsilon"],
            delta=dp["delta"],
            sensitivity=dp["sensitivity"],
            accounting=dp["accounting"],
            preprocessing=[
                "per-client L2 update clipping",
                "pairwise-masked secure aggregation",
                "masked evaluation tallies",
            ],
            computation="pairwise-mask secure aggregation",
        )

    def budget_report(self) -> dict:
        return {
            "limit_epsilon": self._accountant.limit.epsilon,
            "limit_delta": self._accountant.limit.delta,
            "spent_epsilon": self._accountant.spent.epsilon,
            "spent_delta": self._accountant.spent.delta,
            "remaining_epsilon": self._accountant.remaining.epsilon,
            "remaining_delta": self._accountant.remaining.delta,
            "events": len(self._accountant.events),
            "rdp_epsilon_spent": self._accountant.rdp_epsilon(),
            "rdp_complete": self._accountant.rdp_complete,
            "ledger": self._accountant.events,
        }
