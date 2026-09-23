"""Safe local adversarial self-test for SentryLink.

This is NOT an offensive-security tool. It drives the platform through
attacks and failure modes it must reject or contain, and reports each as
[PASS]/[FAIL]. Every check uses tiny synthetic fixtures and runs in-process.

Run:  python -m sentrylink.redteam
"""

from __future__ import annotations

import tempfile

import numpy as np


def _result(name, passed, detail=""):
    return {"name": name, "passed": bool(passed), "detail": detail}


def check_bad_credentials():
    from sentrylink.platform import SentryLinkPlatform

    p = SentryLinkPlatform()
    org = p.join("Acme", "retail", "g-rt")
    try:
        p.registry.authenticate(org.org_id, "wrong-key")
        return _result("bad credentials rejected", False, "wrong key accepted")
    except PermissionError:
        pass
    try:
        p.registry.authenticate("ghost", "whatever")
        return _result("bad credentials rejected", False, "unknown org accepted")
    except KeyError:
        pass
    return _result("bad credentials rejected", True, "wrong key + unknown org denied")


def check_non_member_access():
    from sentrylink.api.app import require_cohort_member
    from sentrylink.errors import AuthorizationError
    from sentrylink.platform import SentryLinkPlatform

    p = SentryLinkPlatform()
    for i in range(3):
        p.join(f"In{i}", "retail", "g-in")
    outsider = p.join("Out", "retail", "g-out")
    try:
        require_cohort_member(p, "g-in", "retail", outsider.org_id, outsider.api_key)
        return _result("non-member access blocked", False, "outsider authorized")
    except AuthorizationError as exc:
        if exc.code == "ORG_NOT_MEMBER":
            return _result("non-member access blocked", True, "code=ORG_NOT_MEMBER")
        return _result("non-member access blocked", False, f"code={exc.code}")
    except Exception as exc:  # noqa: BLE001 - asserting on the mapped status
        if getattr(exc, "status_code", None) == 403:
            return _result("non-member access blocked", True, "outsider got 403")
        return _result("non-member access blocked", False, f"unexpected: {exc!r}")


def check_cross_domain_query():
    from sentrylink.platform import SentryLinkPlatform

    p = SentryLinkPlatform()
    ids = [p.join(f"O{i}", "retail", "g-rt").org_id for i in range(3)]
    try:
        p.histogram("g-rt", "manufacturing", {oid: [1] for oid in ids}, epsilon=1.0)
        return _result("cross-domain query blocked", False, "wrong domain accepted")
    except PermissionError as exc:
        code = getattr(exc, "code", "")
        if code == "DOMAIN_MISMATCH":
            return _result("cross-domain query blocked", True, f"code={code}")
        return _result("cross-domain query blocked", False, f"code={code}")


def check_insufficient_cohort():
    from sentrylink.platform import SentryLinkPlatform

    p = SentryLinkPlatform()
    ids = [p.join(f"O{i}", "retail", "g-small").org_id for i in range(2)]
    try:
        p.histogram("g-small", "retail", {oid: [1] for oid in ids}, epsilon=1.0)
        return _result("insufficient cohort blocked", False, "k=2 accepted")
    except PermissionError as exc:
        if getattr(exc, "code", "") == "COHORT_TOO_SMALL":
            return _result("insufficient cohort blocked", True, "k=2 denied")
        return _result("insufficient cohort blocked", False, f"code={getattr(exc, 'code', '')}")


def check_healthcare_policy_violations():
    from sentrylink.platform import SentryLinkPlatform

    p = SentryLinkPlatform()
    ids = [p.join(f"H{i}", "healthcare", "hg").org_id for i in range(3)]
    try:
        p.variance("hg", "healthcare", {oid: [1.0, 2.0] for oid in ids})
        return _result("healthcare variance blocked", False, "variance accepted")
    except PermissionError:
        pass
    try:
        p.histogram("hg", "healthcare", {oid: [1] for oid in ids}, epsilon=25.0)
        return _result("healthcare epsilon cap enforced", False, "eps=25 accepted")
    except PermissionError as exc:
        if getattr(exc, "code", "") != "HEALTHCARE_EPSILON_TOO_HIGH":
            return _result("healthcare epsilon cap enforced", False, "wrong code")
    try:
        p.correlation(
            "hg", "healthcare",
            {oid: ([1.0, 2.0], [0.0, 1.0]) for oid in ids},
        )
        return _result("healthcare k-floor enforced", False, "k=3 accepted")
    except PermissionError as exc:
        if getattr(exc, "code", "") != "HEALTHCARE_COHORT_TOO_SMALL":
            return _result("healthcare k-floor enforced", False, "wrong code")
    return _result("healthcare policy violations blocked", True, "variance/cap/floor denied")


def check_duplicate_roster():
    from sentrylink.errors import DuplicateRosterError
    from sentrylink.federated.client import FederatedClient
    from sentrylink.federated.server import FederatedServer

    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 4))
    y = (rng.uniform(size=20) > 0.5).astype(float)
    clients = [FederatedClient(org_id="a", x=x, y=y), FederatedClient(org_id="a", x=x, y=y)]
    try:
        FederatedServer(dim=4).run_round(clients, epochs=1, apply_dp=False)
        return _result("duplicate roster rejected", False, "duplicates accepted")
    except DuplicateRosterError:
        return _result("duplicate roster rejected", True, "DuplicateRosterError")


def check_nan_payload():
    from sentrylink.errors import InvalidDataError
    from sentrylink.federated.client import FederatedClient

    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 4))
    y = (rng.uniform(size=20) > 0.5).astype(float)
    try:
        FederatedClient(org_id="a", x=np.full((20, 4), np.nan), y=y)
        return _result("NaN payload rejected", False, "NaN accepted")
    except InvalidDataError:
        return _result("NaN payload rejected", True, "InvalidDataError")


def check_infinite_payload():
    from sentrylink.errors import InvalidDataError
    from sentrylink.federated.client import FederatedClient

    rng = np.random.default_rng(2)
    y = (rng.uniform(size=20) > 0.5).astype(float)
    try:
        FederatedClient(org_id="a", x=np.full((20, 4), np.inf), y=y)
        return _result("infinite payload rejected", False, "inf accepted")
    except InvalidDataError:
        return _result("infinite payload rejected", True, "InvalidDataError")


def check_dimension_mismatch():
    from sentrylink.errors import ModelDimensionMismatchError
    from sentrylink.federated.client import FederatedClient
    from sentrylink.federated.model import LogisticModel

    rng = np.random.default_rng(3)
    c = FederatedClient(
        org_id="a", x=rng.normal(size=(20, 4)),
        y=(rng.uniform(size=20) > 0.5).astype(float),
    )
    try:
        c.local_train(LogisticModel.zeros(3), epochs=1)
        return _result("dimension mismatch rejected", False, "wrong dim accepted")
    except ModelDimensionMismatchError:
        return _result("dimension mismatch rejected", True, "ModelDimensionMismatchError")


def check_budget_exhaustion():
    from sentrylink.crypto.differential_privacy import PrivacyAccountant, PrivacyBudget
    from sentrylink.errors import PrivacyBudgetError
    from sentrylink.platform import SentryLinkPlatform

    p = SentryLinkPlatform(accountant=PrivacyAccountant(limit=PrivacyBudget(1.0, 1e-3)))
    ids = [p.join(f"O{i}", "retail", "g-b").org_id for i in range(3)]
    buckets = {oid: [2, 1] for oid in ids}
    p.histogram("g-b", "retail", buckets, epsilon=1.0)
    try:
        p.histogram("g-b", "retail", buckets, epsilon=1.0)
        return _result("budget exhaustion enforced", False, "over-budget accepted")
    except PrivacyBudgetError:
        return _result("budget exhaustion enforced", True, "second charge denied")


def check_database_write_failure():
    from sentrylink.errors import StorageError
    from sentrylink.platform import SentryLinkPlatform
    from sentrylink.storage import InMemoryStateStore

    class _BrokenStore(InMemoryStateStore):
        def transaction(self):
            raise RuntimeError("disk gone")

    p = SentryLinkPlatform(store=_BrokenStore())
    try:
        p.join("Acme", "retail", "g-f")
        return _result("db write failure contained", False, "no error raised")
    except StorageError:
        if p.registry.list() or p.audit.entries:
            return _result("db write failure contained", False, "state diverged")
        return _result("db write failure contained", True, "raised + reconverged")


def check_private_key_retention():
    import dataclasses

    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

    from sentrylink.crypto.secure_aggregation import SecAggClient, SecAggServer, new_round_id

    roster = ["a", "b", "c"]
    round_id = new_round_id()
    server = SecAggServer(round_id=round_id, roster=roster, dim=4)
    clients = {oid: SecAggClient(oid, round_id, roster, 4) for oid in roster}
    for oid, c in clients.items():
        server.register(oid, c.public_key)
    keys = server.public_keys()
    rng = np.random.default_rng(4)
    for oid, c in clients.items():
        server.receive(c.mask_and_send(rng.normal(size=4), keys))
    server.finalize()

    seen: set[int] = set()

    def _walk(obj):
        if id(obj) in seen:
            return False
        seen.add(id(obj))
        if isinstance(obj, X25519PrivateKey):
            return True
        if isinstance(obj, dict):
            return any(_walk(v) for v in obj.values())
        if isinstance(obj, (list, tuple, set, frozenset)):
            return any(_walk(v) for v in obj)
        if hasattr(obj, "__dataclass_fields__"):
            return any(_walk(getattr(obj, f.name)) for f in dataclasses.fields(obj))
        if hasattr(obj, "__dict__"):
            return _walk(vars(obj))
        return False

    if _walk(server):
        return _result("private-key retention blocked", False, "key reachable on server")
    return _result("private-key retention blocked", True, "server graph clean")


def check_raw_data_persistence():
    from sentrylink.federated.client import FederatedClient
    from sentrylink.platform import SentryLinkPlatform
    from sentrylink.storage import SQLiteStateStore
    from sentrylink.usecases import make_retail_cohort
    from tests.sentinel_scan import scan_sqlite

    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/rt.db"
        p = SentryLinkPlatform(store=SQLiteStateStore(db))
        orgs = make_retail_cohort(n_orgs=3, n_per_org=30, seed=9)
        ids = [p.join(o.name, "retail", "g-rt").org_id for o in orgs]
        keys = [p.registry.get(oid).api_key for oid in ids]
        clients = {oid: FederatedClient(org_id=oid, x=o.x, y=o.y) for oid, o in zip(ids, orgs)}
        p.run_federated_round("g-rt", "retail", clients, epsilon=8.0, epochs=1)
        p.store.close()
        sentinels = {
            "api_key": [k.encode() for k in keys],
            "raw_features": [o.x.tobytes() for o in orgs],
        }
        findings = scan_sqlite(db, sentinels)
        if findings:
            return _result("raw-data persistence blocked", False, str(findings[:2]))
        return _result("raw-data persistence blocked", True, "keys + feature bytes absent")


def check_audit_tampering():
    from sentrylink.platform import SentryLinkPlatform

    p = SentryLinkPlatform()
    ids = [p.join(f"O{i}", "retail", "g-a").org_id for i in range(3)]
    p.histogram("g-a", "retail", {oid: [1, 1] for oid in ids}, epsilon=1.0)
    if not p.audit.verify_chain():
        return _result("audit tampering detected", False, "clean chain fails")
    p.audit.entries[0]["details"]["name"] = "mallory"
    if p.audit.verify_chain():
        return _result("audit tampering detected", False, "modification missed")
    p.audit.entries.pop(0)
    if p.audit.verify_chain():
        return _result("audit tampering detected", False, "deletion missed")
    return _result("audit tampering detected", True, "modification + deletion caught")


def check_dropout_consistency():
    from sentrylink.crypto.secure_aggregation import (
        SecAggClient,
        SecAggServer,
        clip_l2,
        dequantize,
        new_round_id,
        quantize,
    )
    from sentrylink.errors import ProtocolError

    roster = ["a", "b", "c", "d"]
    dim = 8
    rng = np.random.default_rng(5)
    updates = {oid: rng.normal(size=dim) for oid in roster}

    def _run(active, reveal):
        rid = new_round_id()
        server = SecAggServer(round_id=rid, roster=roster, dim=dim)
        clients = {oid: SecAggClient(oid, rid, roster, dim) for oid in roster}
        for oid, c in clients.items():
            server.register(oid, c.public_key)
        keys = server.public_keys()
        for oid in active:
            server.receive(clients[oid].mask_and_send(updates[oid], keys))
        if reveal:
            for s in active:
                for d in set(roster) - set(active):
                    server.add_recovery_seed(s, d, clients[s].reveal_seed(d, keys[d]))
        return server.finalize()

    # missing recovery seeds must fail loudly, not silently unmask wrong
    try:
        _run(["a", "b", "c"], reveal=False)
        return _result("dropout inconsistency detected", False, "missing seeds accepted")
    except ProtocolError:
        pass
    # N-1 and multi-dropout recover exactly
    got = _run(["a", "b", "c"], reveal=True)
    exp = dequantize(
        sum((quantize(clip_l2(updates[o])) for o in ["a", "b", "c"]),
            np.zeros(dim, dtype=np.int64))
    )
    if not np.allclose(got, exp, atol=1e-9):
        return _result("dropout inconsistency detected", False, "N-1 sum wrong")
    got2 = _run(["a", "b"], reveal=True)
    exp2 = dequantize(
        sum((quantize(clip_l2(updates[o])) for o in ["a", "b"]),
            np.zeros(dim, dtype=np.int64))
    )
    if not np.allclose(got2, exp2, atol=1e-9):
        return _result("dropout inconsistency detected", False, "multi-dropout sum wrong")
    return _result("dropout inconsistency detected", True, "missing seeds fail; recovery exact")


def check_honest_round_succeeds():
    from sentrylink.federated.client import FederatedClient
    from sentrylink.platform import SentryLinkPlatform
    from sentrylink.usecases import make_retail_cohort

    p = SentryLinkPlatform()
    orgs = make_retail_cohort(n_orgs=3, n_per_org=60, seed=11)
    ids = [p.join(o.name, "retail", "g-ok").org_id for o in orgs]
    clients = {oid: FederatedClient(org_id=oid, x=o.x, y=o.y) for oid, o in zip(ids, orgs)}
    result = p.run_federated_round("g-ok", "retail", clients, epsilon=8.0, epochs=1)
    if len(result.participants) != 3 or not p.audit.verify_chain():
        return _result("honest round succeeds", False, "bad participants/chain")
    return _result("honest round succeeds", True, "3 participants, chain valid")


CHECKS = [
    check_bad_credentials,
    check_non_member_access,
    check_cross_domain_query,
    check_insufficient_cohort,
    check_healthcare_policy_violations,
    check_duplicate_roster,
    check_nan_payload,
    check_infinite_payload,
    check_dimension_mismatch,
    check_budget_exhaustion,
    check_database_write_failure,
    check_private_key_retention,
    check_raw_data_persistence,
    check_audit_tampering,
    check_dropout_consistency,
    check_honest_round_succeeds,
]


def run_all() -> list[dict]:
    results = []
    for check in CHECKS:
        try:
            results.append(check())
        except Exception as exc:  # noqa: BLE001 - a crashing check is a failure
            results.append(_result(check.__name__, False, f"crashed: {exc!r}"))
    return results


def main() -> int:
    results = run_all()
    failed = 0
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            failed += 1
        detail = f" — {r['detail']}" if r["detail"] else ""
        print(f"[{mark}] {r['name']}{detail}")
    print(f"\n{len(results) - failed}/{len(results)} red-team checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
