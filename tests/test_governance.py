import pytest

from sentrylink.errors import (
    CohortTooSmallError,
    ConsentDeniedError,
    DomainMismatchError,
    EpsilonTooHighError,
    GovernanceError,
    HealthcareCohortTooSmallError,
    HealthcareEpsilonTooHighError,
    InvalidDataError,
    QueryNotAllowedError,
    ScopeMismatchError,
)
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


def _dec(engine, metric="histogram", group="g1", domain="retail", epsilon=1.0):
    return engine.evaluate(
        QueryRequest(metric=metric, sector_group=group, domain=domain, epsilon=epsilon)
    )


def test_decision_codes_cohort_floor():
    reg = _registry_with(n=2)
    eng = PolicyEngine(reg)
    for oid in ["o0", "o1"]:
        eng.set_policy(oid, ConsentPolicy.default())
    dec = _dec(eng)
    assert dec.code == "COHORT_TOO_SMALL"
    with pytest.raises(CohortTooSmallError):
        dec.raise_if_denied()
    with pytest.raises(PermissionError):  # back-compat: still a PermissionError
        dec.raise_if_denied()


def test_decision_codes_unknown_metric_and_scope():
    reg = _registry_with(n=4)
    eng = PolicyEngine(reg)
    for i in range(4):
        eng.set_policy(f"o{i}", ConsentPolicy.default())
    dec = _dec(eng, metric="raw_export")
    assert dec.code == "QUERY_NOT_ALLOWED"
    with pytest.raises(QueryNotAllowedError):
        dec.raise_if_denied()
    assert _dec(eng, group="nope").code == "SECTOR_MISMATCH"
    with pytest.raises(ScopeMismatchError):
        _dec(eng, group="nope").raise_if_denied()
    assert _dec(eng, domain="nope").code == "DOMAIN_MISMATCH"
    with pytest.raises(DomainMismatchError):
        _dec(eng, domain="nope").raise_if_denied()


def test_decision_codes_consent_and_epsilon():
    reg = _registry_with(n=3)
    eng = PolicyEngine(reg)
    for i in range(3):
        eng.set_policy(f"o{i}", ConsentPolicy(allowed_metrics=frozenset()))
    dec = _dec(eng)
    assert dec.code == "CONSENT_REQUIRED"
    with pytest.raises(ConsentDeniedError):
        dec.raise_if_denied()

    eng2 = PolicyEngine(_registry_with(n=3))
    for i in range(3):
        eng2.set_policy(
            f"o{i}",
            ConsentPolicy(
                allowed_metrics=frozenset({"histogram"}), max_epsilon_per_query=1.0
            ),
        )
    dec = _dec(eng2, epsilon=9.0)
    assert dec.code == "EPSILON_TOO_HIGH"
    with pytest.raises(EpsilonTooHighError):
        dec.raise_if_denied()


def test_decision_codes_healthcare_variants():
    from sentrylink.governance.policy import default_policy_for

    reg = Registry()
    for i in range(3):
        reg.register(name=f"H{i}", domain="healthcare", sector_group="hg", org_id=f"h{i}")
    eng = PolicyEngine(reg)
    for i in range(3):
        eng.set_policy(f"h{i}", default_policy_for(reg.get(f"h{i}")))
    dec = _dec(eng, group="hg", domain="healthcare")
    assert dec.code == "HEALTHCARE_COHORT_TOO_SMALL"
    with pytest.raises(HealthcareCohortTooSmallError):
        dec.raise_if_denied()
    assert issubclass(HealthcareCohortTooSmallError, GovernanceError)
    assert issubclass(HealthcareEpsilonTooHighError, GovernanceError)
    assert issubclass(InvalidDataError, ValueError)


def test_allowed_decision_carries_snapshot():
    reg = _registry_with(n=3)
    eng = PolicyEngine(reg)
    for i in range(3):
        eng.set_policy(f"o{i}", ConsentPolicy.default())
    dec = _dec(eng)
    assert dec.allowed and dec.code is None
    assert dec.policy_snapshot["floor_required"] == 3
    assert dec.policy_snapshot["candidates"] == 3
    assert dec.policy_snapshot["epsilon_cap_min"] == 25.0
