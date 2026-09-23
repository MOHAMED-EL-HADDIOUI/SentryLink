"""Privacy release artifacts: ledger entries, previews, transparency cards.

Every artifact here is derived from actual runtime metadata — the mechanism
really used, the epsilon/delta really charged, the cohort really evaluated.
Nothing in this module touches raw data, keys, or unmasked values, so its
outputs are safe to display, persist, and serve over the API.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class PrivacyLedgerEntry:
    """One immutable privacy-spend record (stored inside the budget events)."""

    scope: str  # budget subject, e.g. "hist:retail-consortium"
    query_type: str  # histogram | variance | correlation | federated_model_round
    mechanism: str  # laplace | gaussian
    epsilon: float
    delta: float
    rdp_orders: list[float]
    sensitivity: float
    cohort_size: int
    clip_bound: float | None
    purpose: str
    request_id: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


def build_release_card(
    *,
    release: str,
    domain: str,
    cohort_size: int,
    mechanism: str,
    epsilon: float,
    delta: float,
    sensitivity: object,
    accounting: str,
    preprocessing: list[str],
    computation: str,
) -> dict:
    """Sanitized Privacy Card: cohort SIZE, never organization identities."""
    return {
        "release": release,
        "domain": domain,
        "cohort_size": cohort_size,
        "mechanism": mechanism,
        "privacy": {"epsilon": epsilon, "delta": delta},
        "accounting": accounting,
        "sensitivity": sensitivity,
        "preprocessing": list(preprocessing),
        "secure_computation": computation,
        "raw_data": "never leaves organization",
        "persistent_raw_records": "none",
        "audit": "hash-chained",
        "security_note": "2-node collusion is out of scope",
    }


def render_card_text(card: dict) -> str:
    p = card["privacy"]
    lines = [
        "SENTRYLINK PRIVACY CARD",
        "",
        f"Release: {card['release']}",
        f"Domain: {card['domain']}",
        f"Cohort: {card['cohort_size']} organizations",
        "",
        "Mechanism:",
        f"  {card['mechanism']}",
        "",
        "Privacy:",
        f"  epsilon = {p['epsilon']}",
        f"  delta   = {p['delta']}",
        "",
        "Accounting:",
        f"  {card['accounting']}",
        "",
        "Sensitivity:",
        f"  {card['sensitivity']}",
        "",
        "Pre-processing:",
    ]
    lines += [f"  {step}" for step in card["preprocessing"]]
    lines += [
        "",
        "Secure computation:",
        f"  {card['secure_computation']}",
        "",
        "Raw data:",
        f"  {card['raw_data']}",
        "",
        "Persistent raw records:",
        f"  {card['persistent_raw_records']}",
        "",
        "Audit:",
        f"  {card['audit']}",
        "",
        "Security note:",
        f"  {card['security_note']}",
    ]
    return "\n".join(lines)
