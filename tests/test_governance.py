import pytest

from sentrylink.governance.audit import AuditLog
from sentrylink.governance.policy import ConsentPolicy, PolicyEngine, QueryRequest
from sentrylink.governance.registry import Registry
from sentrylink.platform import SentryLinkPlatform


def _registry_with(n=4, domain="retail", group="g1"):
    r = Registry()
    for i in range(n):
        r.register(name=f"Org {i}", domain=domain, sector_group=group, org_id=f"o{i}")
    return r


def test_registry_register_and_auth():
    r = Registry()
    org = r.register("Acme", "retail", "g1")
    assert r.authenticate(org.org_id, org.api_key).name == "Acme"
    with pytest.raises(PermissionError):
        r.authenticate(org.org_id, "wrong")


def test_policy_blocks_when_cohort_too_small():
    reg = _registry_with(n=2)
    eng = PolicyEngine(reg)
    for oid in ["o0", "o1"]:
        eng.set_policy(oid, ConsentPolicy.default())
    dec = eng.evaluate(
        QueryRequest(metric="histogram", sector_group="g1", domain="retail", epsilon=1.0)
    )
    assert not dec.allowed
    assert any("cohort too small" in r for r in dec.reasons)


def test_policy_respects_consent_opt_out():
    reg = _registry_with(n=4)
    eng = PolicyEngine(reg)
    eng.set_policy("o0", ConsentPolicy.default())
    eng.set_policy("o1", ConsentPolicy(allowed_metrics=frozenset({"correlation"})))
    eng.set_policy("o2", ConsentPolicy.default())
    eng.set_policy("o3", ConsentPolicy.default())
    dec = eng.evaluate(
        QueryRequest(metric="histogram", sector_group="g1", domain="retail", epsilon=1.0)
    )
    assert dec.allowed
    assert "o1" not in dec.participants
    assert dec.participants == ["o0", "o2", "o3"]


def test_policy_rejects_unknown_metric():
    reg = _registry_with(n=5)
    eng = PolicyEngine(reg)
    for i in range(5):
        eng.set_policy(f"o{i}", ConsentPolicy.default())
    dec = eng.evaluate(
        QueryRequest(metric="raw_export", sector_group="g1", domain="retail", epsilon=1.0)
    )
    assert not dec.allowed


def test_policy_rejects_excessive_epsilon():
    reg = _registry_with(n=5)
    eng = PolicyEngine(reg)
    for i in range(5):
        eng.set_policy(f"o{i}", ConsentPolicy(max_epsilon_per_query=1.0))
    dec = eng.evaluate(
        QueryRequest(metric="histogram", sector_group="g1", domain="retail", epsilon=9.0)
    )
    # nobody consents at this epsilon => cohort too small
    assert not dec.allowed


def test_audit_hash_chain():
    log = AuditLog()
    log.record("a", x=1)
    log.record("b", y=2)
    log.record("c", z=3)
    assert log.verify_chain()
    log.entries[1]["action"] = "tampered"
    assert not log.verify_chain()


def test_platform_join_and_budget():
    p = SentryLinkPlatform()
    org = p.join("A", "retail", "g1")
    assert org.org_id in [o.org_id for o in p.registry.list()]
    assert p.audit.verify_chain()
    report = p.budget_report()
    assert report["spent_epsilon"] == 0.0
