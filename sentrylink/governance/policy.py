"""Consent policy engine: which queries may run, over whom, at what cost."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field

from ..config import MAX_ORG_CONTRIB, MIN_PARTICIPANTS
from ..crypto.differential_privacy import PrivacyBudget
from .registry import Organization, Registry

ALLOWED_METRICS = {
    "histogram",
    "variance",
    "correlation",
    "federated_model_round",
}


@dataclass(frozen=True)
class ConsentPolicy:
    """Per-organization opt-in rules."""

    allowed_metrics: frozenset[str] = frozenset()
    min_participants: int = MIN_PARTICIPANTS
    max_epsilon_per_query: float = 25.0
    purpose: str = "cross-org intelligence"

    @staticmethod
    def default(allowed: set[str] | None = None) -> "ConsentPolicy":
        return ConsentPolicy(allowed_metrics=frozenset(allowed or set(ALLOWED_METRICS)))


@dataclass
class QueryRequest:
    metric: str
    sector_group: str
    domain: str
    epsilon: float
    delta: float = 1e-5
    purpose: str = "aggregate insight"
    requester: str = "consortium"
    extra: dict = field(default_factory=dict)

    @property
    def budget(self) -> PrivacyBudget:
        return PrivacyBudget(self.epsilon, self.delta)


@dataclass
class QueryDecision:
    allowed: bool
    query_id: str
    participants: list[str]
    reasons: list[str]
    budget: PrivacyBudget

    def raise_if_denied(self):
        if not self.allowed:
            raise PermissionError("; ".join(self.reasons))


class PolicyEngine:
    def __init__(self, registry: Registry, policies: dict[str, ConsentPolicy] | None = None):
        self.registry = registry
        self.policies: dict[str, ConsentPolicy] = policies or {}

    def set_policy(self, org_id: str, policy: ConsentPolicy) -> None:
        self.registry.get(org_id)  # existence check
        self.policies[org_id] = policy

    def evaluate(self, req: QueryRequest) -> QueryDecision:
        reasons: list[str] = []
        candidates = self.registry.cohort(req.sector_group, req.domain)
        participants: list[str] = []

        if req.metric not in ALLOWED_METRICS:
            reasons.append(f"metric '{req.metric}' not in platform allow-list")

        for org in candidates:
            pol = self.policies.get(org.org_id, ConsentPolicy.default())
            if req.metric not in pol.allowed_metrics:
                continue
            if req.epsilon > pol.max_epsilon_per_query:
                continue
            participants.append(org.org_id)

        # Cohort floor: global minimum, raised to the strictest
        # min_participants among contributing orgs (e.g. healthcare k>=4).
        required = MIN_PARTICIPANTS
        for oid in participants:
            pol = self.policies.get(oid, ConsentPolicy.default())
            required = max(required, pol.min_participants)
        if len(participants) < required:
            reasons.append(
                f"cohort too small: {len(participants)} consented "
                f"< min_participants={required}"
            )
        if req.epsilon <= 0:
            reasons.append("epsilon must be positive")
        if req.delta <= 0 or req.delta >= 1:
            reasons.append("delta must be in (0, 1)")

        return QueryDecision(
            allowed=not reasons,
            query_id=secrets.token_hex(8),
            participants=sorted(participants),
            reasons=reasons,
            budget=req.budget,
        )

    @staticmethod
    def cap_contributions(
        values: list[int], radius: float | None = None
    ) -> list[int]:
        """Project an org's contribution onto an L2 ball (DP sensitivity bound).
        Integer rounding is inward so the projected norm stays ≤ radius."""
        import numpy as np

        from ..config import MAX_ORG_CONTRIB

        r = float(MAX_ORG_CONTRIB if radius is None else radius)
        v = np.asarray(values, dtype=np.float64)
        norm = float(np.linalg.norm(v))
        if norm > r and norm > 0:
            v = v * (r / norm)
        return [int(x) for x in np.rint(v)]


def default_policy_for(org: Organization) -> ConsentPolicy:
    if org.domain == "healthcare":
        return ConsentPolicy(
            allowed_metrics=frozenset({"histogram", "correlation", "federated_model_round"}),
            min_participants=max(MIN_PARTICIPANTS, 4),
            max_epsilon_per_query=10.0,
            purpose="treatment-response research",
        )
    return ConsentPolicy.default()
