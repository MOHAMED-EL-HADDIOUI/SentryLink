"""Privacy artifacts: ledger entries, previews, cards, release metadata."""

import numpy as np
import pytest

from sentrylink.crypto.differential_privacy import PrivacyAccountant, PrivacyBudget
from sentrylink.platform import SentryLinkPlatform
from sentrylink.privacy import PrivacyLedgerEntry, build_release_card, render_card_text


def _platform(n=3, domain="retail", group="g-p"):
    p = SentryLinkPlatform()
    ids = [p.join(f"O{i}", domain, group).org_id for i in range(n)]
    return p, ids


def test_ledger_entry_recorded_with_before_after():
    p, ids = _platform()
    p.histogram("g-p", "retail", {oid: [4, 2] for oid in ids}, epsilon=1.0)
    (record,) = p.accountant.events
    assert record["budget_before"] == {"epsilon": 0.0, "delta": 0.0}
    assert record["budget_after"]["epsilon"] == pytest.approx(1.0)
    ledger = record["ledger"]
    assert ledger["query_type"] == "histogram"
    assert ledger["mechanism"] == "laplace"
    assert ledger["epsilon"] == 1.0 and ledger["delta"] == 0.0
    assert ledger["sensitivity"] == 100.0
    assert ledger["cohort_size"] == 3
    assert ledger["clip_bound"] == 100.0
    assert ledger["scope"] == "hist:g-p"
    assert ledger["timestamp"] > 0 and isinstance(ledger["rdp_orders"], list)
    # sanitized: no org ids, keys, or raw values anywhere in the entry
    blob = str(record)
    for oid in ids:
        assert oid not in blob


def test_ledger_entry_dataclass_roundtrip():
    entry = PrivacyLedgerEntry(
        scope="s", query_type="histogram", mechanism="laplace", epsilon=1.0,
        delta=0.0, rdp_orders=[2.0], sensitivity=100.0, cohort_size=3,
        clip_bound=100.0, purpose="p",
    )
    d = entry.to_dict()
    assert d["query_type"] == "histogram" and d["request_id"] is None


def test_preview_spends_nothing_and_mutates_nothing():
    p, ids = _platform()
    audit_before = len(p.audit.entries)
    prev = p.preview("histogram", "g-p", "retail", epsilon=1.0)
    assert prev["allowed"] is True
    assert prev["cohort_size"] == 3
    assert prev["mechanism"] == "laplace"
    assert prev["epsilon_requested"] == 1.0
    assert prev["remaining_budget_after"] == pytest.approx(
        prev["remaining_budget_before"] - 1.0
    )
    assert prev["governance"] == {
        "cohort_floor_satisfied": True,
        "sector_scope_valid": True,
        "consent_valid": True,
    }
    assert p.budget_report()["spent_epsilon"] == 0.0
    assert len(p.audit.entries) == audit_before


def test_preview_denied_cohort_reports_code():
    p, _ = _platform(n=2)
    prev = p.preview("histogram", "g-p", "retail", epsilon=1.0)
    assert prev["allowed"] is False
    assert prev["reason_code"] == "COHORT_TOO_SMALL"
    assert prev["governance"]["cohort_floor_satisfied"] is False
    assert p.budget_report()["spent_epsilon"] == 0.0


def test_preview_denied_budget_reports_code():
    p, ids = _platform()
    p.accountant = PrivacyAccountant(limit=PrivacyBudget(0.5, 1e-3))
    # re-join not needed: accountant swap is test-only; policy state intact
    prev = p.preview("histogram", "g-p", "retail", epsilon=1.0)
    assert prev["allowed"] is False
    assert prev["reason_code"] == "BUDGET_EXHAUSTED"


def test_preview_unknown_metric():
    p, ids = _platform()
    prev = p.preview("raw_export", "g-p", "retail", epsilon=1.0)
    assert prev["allowed"] is False
    assert prev["reason_code"] == "QUERY_NOT_ALLOWED"


def test_histogram_card_matches_runtime():
    p, ids = _platform()
    res = p.histogram("g-p", "retail", {oid: [4, 2] for oid in ids}, epsilon=2.0)
    card = res.privacy_card
    assert card["release"] == "histogram"
    assert card["mechanism"] == "laplace"
    assert card["privacy"] == {"epsilon": 2.0, "delta": 0.0}
    assert card["cohort_size"] == 3
    assert card["sensitivity"] == 100.0
    assert "org_id" not in str(card) and all(oid not in str(card) for oid in ids)
    text = render_card_text(card)
    assert "SENTRYLINK PRIVACY CARD" in text and "2-node collusion is out of scope" in text


def test_release_metadata_before_any_round():
    p, _ = _platform()
    with pytest.raises(KeyError):
        p.model_release_metadata("g-p:retail")


def test_release_metadata_derived_from_actuals(tmp_path):
    from sentrylink.federated.client import FederatedClient
    from sentrylink.storage import SQLiteStateStore
    from sentrylink.usecases import make_retail_cohort

    p = SentryLinkPlatform(store=SQLiteStateStore(str(tmp_path / "m.db")))
    try:
        orgs = make_retail_cohort(n_orgs=3, n_per_org=60, seed=7)
        ids = [p.join(o.name, "retail", "g-m").org_id for o in orgs]
        clients = {oid: FederatedClient(org_id=oid, x=o.x, y=o.y) for oid, o in zip(ids, orgs)}
        result = p.run_federated_round("g-m", "retail", clients, epsilon=8.0, epochs=1)
        meta = p.model_release_metadata("g-m:retail")
        assert meta["model_version"] == 1
        assert meta["algorithm"] == "logistic_regression"
        assert meta["participants"] == 3 and meta["dropouts"] == 0
        assert meta["feature_dimension"] == 4
        assert meta["dp"]["mechanism"] == "gaussian"
        assert meta["dp"]["epsilon"] == result.epsilon_used == 8.0
        assert meta["dp"]["delta"] == result.delta_used
        assert meta["dp"]["sensitivity"] == pytest.approx(2.0 * 1.0 / 3)
        assert meta["secure_aggregation"]["quantization_scale"] == 1_000_000
        assert meta["protocol_version"] == "1"
        card = p.federated_release_card("g-m:retail", "retail")
        assert card["release"] == "federated_model"
        assert card["privacy"]["epsilon"] == 8.0
    finally:
        p.store.close()


def test_transparency_and_timeline_platform_level():
    p, ids = _platform()
    p.histogram("g-p", "retail", {oid: [1, 1] for oid in ids}, epsilon=1.0)
    assert p.audit.verify_chain()
    timeline = p.audit.timeline()
    stages = {t["action"]: t["stage"] for t in timeline}
    assert stages["query.histogram"] == "release"
    assert any(t["stage"] == "governance" for t in timeline)
    assert [t["seq"] for t in timeline] == sorted(t["seq"] for t in timeline)
