"""Consent policy engine: which queries may run, over whom, at what cost."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field

from ..config import MAX_ORG_CONTRIB, MIN_PARTICIPANTS
from ..crypto.differential_privacy import PrivacyBudget
from ..verticals import policy_for_domain
from ..errors import (
    ConsentDeniedError,
    DomainMismatchError,
    EpsilonTooHighError,
    GovernanceError,
    HealthcareCohortTooSmallError,
    HealthcareEpsilonTooHighError,
    CohortTooSmallError,
    InvalidDataError,
    QueryNotAllowedError,
    ScopeMismatchError,
)
from .registry import Organization, Registry

_CODE_TO_ERROR = {
    "QUERY_NOT_ALLOWED": QueryNotAllowedError,
    "COHORT_TOO_SMALL": CohortTooSmallError,
    "HEALTHCARE_COHORT_TOO_SMALL": HealthcareCohortTooSmallError,
    "EPSILON_TOO_HIGH": EpsilonTooHighError,
    "HEALTHCARE_EPSILON_TOO_HIGH": HealthcareEpsilonTooHighError,
    "CONSENT_REQUIRED": ConsentDeniedError,
    "SECTOR_MISMATCH": ScopeMismatchError,
    "DOMAIN_MISMATCH": DomainMismatchError,
    "INVALID_DATA": InvalidDataError,
}


def _error_for_code(code: str | None):
    return _CODE_TO_ERROR.get(code or "", GovernanceError)

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
    code: str | None = None
    policy_snapshot: dict = field(default_factory=dict)

    def raise_if_denied(self):
        if not self.allowed:
            raise _error_for_code(self.code)("; ".join(self.reasons))


class PolicyEngine:
    def __init__(self, registry: Registry, policies: dict[str, ConsentPolicy] | None = None):
        self.registry = registry
        self.policies: dict[str, ConsentPolicy] = policies or {}

    def set_policy(self, org_id: str, policy: ConsentPolicy) -> None:
        self.registry.get(org_id)  # existence check
        self.policies[org_id] = policy

    def evaluate(self, req: QueryRequest) -> QueryDecision:
        reasons: list[str] = []
        code: str | None = None
        candidates = self.registry.cohort(req.sector_group, req.domain)
        participants: list[str] = []

        def snapshot(floor: int) -> dict:
            caps = [
                self.policies.get(oid, ConsentPolicy.default()).max_epsilon_per_query
                for oid in participants
            ]
            return {
                "metric": req.metric,
                "floor_required": floor,
                "candidates": len(candidates),
                "epsilon_cap_min": min(caps) if caps else None,
            }

        def denied(reason: str, reason_code: str, floor: int = MIN_PARTICIPANTS) -> QueryDecision:
            reasons.append(reason)
            return QueryDecision(
                allowed=False,
                query_id=secrets.token_hex(8),
                participants=[],
                reasons=reasons,
                budget=req.budget,
                code=reason_code,
                policy_snapshot=snapshot(floor),
            )

        if req.metric not in ALLOWED_METRICS:
            return denied(
                f"metric '{req.metric}' not in platform allow-list", "QUERY_NOT_ALLOWED"
            )
        if req.epsilon <= 0:
            return denied("epsilon must be positive", "INVALID_DATA")
        if req.delta <= 0 or req.delta >= 1:
            return denied("delta must be in (0, 1)", "INVALID_DATA")

        if not candidates:
            if not self.registry.cohort(req.sector_group):
                return denied(
                    f"unknown sector_group '{req.sector_group}'", "SECTOR_MISMATCH"
                )
            return denied(
                f"no organizations for domain '{req.domain}' "
                f"in sector_group '{req.sector_group}'",
                "DOMAIN_MISMATCH",
            )

        epsilon_blocked = 0
        for org in candidates:
            pol = self.policies.get(org.org_id, ConsentPolicy.default())
            if req.metric not in pol.allowed_metrics:
                continue
            if req.epsilon > pol.max_epsilon_per_query:
                epsilon_blocked += 1
                continue
            participants.append(org.org_id)

        if not participants and epsilon_blocked == len(candidates):
            variant = (
                "HEALTHCARE_EPSILON_TOO_HIGH"
                if req.domain == "healthcare"
                else "EPSILON_TOO_HIGH"
            )
            return denied(
                f"epsilon {req.epsilon} exceeds every consented per-query cap",
                variant,
            )
        if not participants:
            return denied(
                "no organization consented to this metric at this budget",
                "CONSENT_REQUIRED",
            )

        # Cohort floor: global minimum, raised to the strictest
        # min_participants among contributing orgs (e.g. healthcare k>=4).
        required = MIN_PARTICIPANTS
        for oid in participants:
            pol = self.policies.get(oid, ConsentPolicy.default())
            required = max(required, pol.min_participants)
        if len(participants) < required:
            variant = (
                "HEALTHCARE_COHORT_TOO_SMALL"
                if req.domain == "healthcare" and required >= 4
                else "COHORT_TOO_SMALL"
            )
            return denied(
                f"cohort too small: {len(participants)} consented "
                f"< min_participants={required}",
                variant,
                floor=required,
            )

        return QueryDecision(
            allowed=True,
            query_id=secrets.token_hex(8),
            participants=sorted(participants),
            reasons=reasons,
            budget=req.budget,
            code=None,
            policy_snapshot=snapshot(required),
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
        # Truncate toward zero (never round up): the projected norm is
        # guaranteed <= radius, which the DP sensitivity bound requires.
        return [int(x) for x in np.fix(v)]


def default_policy_for(org: Organization) -> ConsentPolicy:
    """Build the default consent policy from the domain's vertical policy."""
    vp = policy_for_domain(org.domain)
    return ConsentPolicy(
        allowed_metrics=frozenset(vp.allowed_metrics),
        min_participants=max(MIN_PARTICIPANTS, vp.min_participants),
        max_epsilon_per_query=vp.max_epsilon_per_query,
        purpose=vp.purpose,
    )
