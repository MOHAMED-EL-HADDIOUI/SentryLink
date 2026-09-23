"""Restart/recovery tests: deterministic state survival across process restarts."""

import numpy as np
import pytest

from sentrylink.crypto.differential_privacy import PrivacyAccountant, PrivacyBudget
from sentrylink.platform import SentryLinkPlatform
from sentrylink.storage import InMemoryStateStore, SQLiteStateStore
from sentrylink.usecases import make_retail_cohort
from sentrylink.federated.client import FederatedClient


def _platform(path, **kwargs):
    return SentryLinkPlatform(store=SQLiteStateStore(str(path)), **kwargs)


def _retail_round(p, group="rc", n_orgs=3, n_per_org=60, seed=7, epochs=1):
    orgs = make_retail_cohort(n_orgs=n_orgs, n_per_org=n_per_org, seed=seed)
    ids = [p.join(o.name, domain="retail", sector_group=group).org_id for o in orgs]
    clients = {oid: FederatedClient(org_id=oid, x=o.x, y=o.y) for oid, o in zip(ids, orgs)}
    result = p.run_federated_round(group, "retail", clients, epsilon=8.0, epochs=epochs)
    return p, ids, result


def test_restart_recovery_full(tmp_path):
    db = tmp_path / "restart.db"
    p, ids, r1 = _retail_round(_platform(db))
    p.histogram("rc", "retail", {oid: [5, 3] for oid in ids}, epsilon=1.0)
    spent_before = p.budget_report()["spent_epsilon"]
    audit_before = len(p.audit.entries)
    weights_before = r1.model.flat.copy()
    p.store.close()

    p2 = _platform(db)
    try:
        assert sorted(o.org_id for o in p2.registry.list()) == sorted(ids)
        assert p2.budget_report()["spent_epsilon"] == pytest.approx(spent_before)
        assert len(p2.audit.entries) == audit_before
        assert p2.audit.verify_chain()
        key = "rc:retail"
        assert key in p2.servers and key in p2.last_round
        np.testing.assert_array_equal(p2.servers[key].model.flat, weights_before)
        assert len(p2.servers[key].history) == 1
        # cohort governance still enforced from restored state
        h = p2.histogram("rc", "retail", {oid: [1, 1] for oid in ids}, epsilon=1.0)
        assert set(h.participants) == set(ids)
    finally:
        p2.store.close()


def test_budget_exhaustion_across_restart(tmp_path):
    db = tmp_path / "budget.db"
    acc = PrivacyAccountant(limit=PrivacyBudget(2.0, 1e-3))
    p = _platform(db, accountant=acc)
    ids = [p.join(f"O{i}", "retail", "g-b").org_id for i in range(3)]
    buckets = {oid: [4, 2] for oid in ids}
    p.histogram("g-b", "retail", buckets, epsilon=1.0)
    p.histogram("g-b", "retail", buckets, epsilon=1.0)
    with pytest.raises(PermissionError):
        p.histogram("g-b", "retail", buckets, epsilon=1.0)
    p.store.close()

    p2 = _platform(db)  # stored budget (limit 2.0) wins over fresh defaults
    try:
        assert p2.budget_report()["spent_epsilon"] == pytest.approx(2.0)
        with pytest.raises(PermissionError):
            p2.histogram("g-b", "retail", buckets, epsilon=1.0)
    finally:
        p2.store.close()


def test_denied_query_leaves_no_trace(tmp_path):
    db = tmp_path / "notrace.db"
    p = _platform(db)
    ids = [p.join(f"O{i}", "retail", "g-n").org_id for i in range(3)]
    audit_before = len(p.audit.entries)
    spent_before = p.budget_report()["spent_epsilon"]
    with pytest.raises(PermissionError):  # over per-query cap
        p.histogram("g-n", "retail", {oid: [1] for oid in ids}, epsilon=999.0)
    assert len(p.audit.entries) == audit_before
    assert p.budget_report()["spent_epsilon"] == spent_before
    p.store.close()
    p2 = _platform(db)
    try:
        assert len(p2.audit.entries) == audit_before
        assert p2.budget_report()["spent_epsilon"] == spent_before
    finally:
        p2.store.close()


def test_consent_caps_survive_restart(tmp_path):
    db = tmp_path / "consent.db"
    p = _platform(db)
    ids = [p.join(f"H{i}", "healthcare", "hg").org_id for i in range(4)]
    p.set_consent(ids[0], {"histogram", "correlation", "federated_model_round"})
    p.store.close()

    p2 = _platform(db)
    try:
        pol = p2.policy.policies[ids[0]]
        assert pol.allowed_metrics == frozenset({"histogram", "correlation", "federated_model_round"})
        assert pol.max_epsilon_per_query == 10.0  # no cap reset through persistence
        assert pol.min_participants == 4
        with pytest.raises(PermissionError):  # variance still denied for healthcare
            p2.variance("hg", "healthcare", {oid: [1.0, 2.0] for oid in ids})
    finally:
        p2.store.close()


def test_model_persistence_and_continued_training(tmp_path):
    db = tmp_path / "model.db"
    p, ids, r1 = _retail_round(_platform(db), group="mc")
    w1 = r1.model.flat.copy()
    p.store.close()

    p2 = _platform(db)
    try:
        key = "mc:retail"
        np.testing.assert_array_equal(p2.servers[key].model.flat, w1)
        assert len(p2.servers[key].history) == 1
        orgs = make_retail_cohort(n_orgs=3, n_per_org=60, seed=7)
        clients = {oid: FederatedClient(org_id=oid, x=o.x, y=o.y) for oid, o in zip(ids, orgs)}
        r2 = p2.run_federated_round("mc", "retail", clients, epsilon=8.0, epochs=1)
        assert len(p2.servers[key].history) == 2
        assert set(r2.participants) == set(ids)
    finally:
        p2.store.close()


def test_restored_api_key_authenticates(tmp_path):
    db = tmp_path / "auth.db"
    p = _platform(db)
    info = p.join("Acme", "retail", "g-a")
    api_key = info.api_key
    p.store.close()

    p2 = _platform(db)
    try:
        assert p2.registry.authenticate(info.org_id, api_key).name == "Acme"
        with pytest.raises(PermissionError):
            p2.registry.authenticate(info.org_id, "wrong")
    finally:
        p2.store.close()


def test_failed_commit_reconverges_memory():
    class _FailingStore(InMemoryStateStore):
        def transaction(self):
            raise RuntimeError("disk gone")

    p = SentryLinkPlatform(store=_FailingStore())
    with pytest.raises(RuntimeError):
        p.join("Acme", "retail", "g-f")
    assert p.registry.list() == []  # in-memory rolled back to last committed
    assert p.audit.entries == []
