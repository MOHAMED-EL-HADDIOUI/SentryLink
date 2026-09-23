"""Versioned release report: tests, security, privacy, persistence, demo.

Run:  python -m sentrylink.report [--json]
"""

from __future__ import annotations

import argparse
import json
import platform as _platform_mod
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _versions() -> dict:
    from sentrylink import __version__ as pkg_version
    from sentrylink.config import PROTOCOL_VERSION
    from sentrylink.storage import CURRENT_SCHEMA_VERSION

    return {
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": CURRENT_SCHEMA_VERSION,
        "python_version": _platform_mod.python_version(),
        "package_version": pkg_version,
    }


def _pytest_section() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    # NOTE: search the whole output, not the last line — warnings and
    # plugin chatter on stderr can trail the summary line.
    text = proc.stdout + proc.stderr
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", text)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", text)) else 0
    skipped = int(m.group(1)) if (m := re.search(r"(\d+) skipped", text)) else 0
    if proc.returncode != 0 and passed == 0 and failed == 0:
        failed = 1
    return {"passed": passed, "failed": failed, "skipped": skipped}


def _walk_privkey(obj, _seen=None) -> bool:
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

    _seen = _seen if _seen is not None else set()
    if id(obj) in _seen:
        return False
    _seen.add(id(obj))
    if isinstance(obj, X25519PrivateKey):
        return True
    if isinstance(obj, dict):
        return any(_walk_privkey(v, _seen) for v in obj.values())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return any(_walk_privkey(v, _seen) for v in obj)
    if hasattr(obj, "__dataclass_fields__"):
        import dataclasses

        return any(
            _walk_privkey(getattr(obj, f.name), _seen)
            for f in dataclasses.fields(obj)
        )
    if hasattr(obj, "__dict__"):
        return _walk_privkey(vars(obj), _seen)
    return False


def _security_section() -> dict:
    import numpy as np

    from sentrylink import redteam
    from sentrylink.crypto.secure_aggregation import (
        SecAggClient,
        SecAggServer,
        new_round_id,
    )
    from sentrylink.platform import SentryLinkPlatform
    from sentrylink.storage import SQLiteStateStore

    results = redteam.run_all()
    red_ok = sum(1 for r in results if r["passed"])

    roster = ["a", "b"]
    rid = new_round_id()
    server = SecAggServer(round_id=rid, roster=roster, dim=4)
    clients = {oid: SecAggClient(oid, rid, roster, 4) for oid in roster}
    for oid, c in clients.items():
        server.register(oid, c.public_key)
    keys = server.public_keys()
    for oid, c in clients.items():
        server.receive(c.mask_and_send(np.ones(4), keys))
    server.finalize()

    with tempfile.TemporaryDirectory() as tmp:
        from tests.sentinel_scan import scan_sqlite

        db = f"{tmp}/rep.db"
        p = SentryLinkPlatform(store=SQLiteStateStore(db))
        ids = [p.join(f"O{i}", "retail", "g-rep").org_id for i in range(3)]
        api_keys = [p.registry.get(oid).api_key for oid in ids]
        chain_ok = p.audit.verify_chain()
        p.store.close()
        scan_ok = scan_sqlite(db, {"api_key": [k.encode() for k in api_keys]}) == []

    return {
        "private_key_retention": "PASS" if not _walk_privkey(server) else "FAIL",
        "sensitive_storage_scan": "PASS" if scan_ok else "FAIL",
        "audit_verification": "PASS" if chain_ok else "FAIL",
        "redteam": f"{red_ok}/{len(results)}",
    }


def _privacy_section() -> dict:
    from sentrylink.crypto.differential_privacy import (
        PrivacyAccountant,
        PrivacyBudget,
        gaussian_rdp_cost,
        gaussian_sigma,
    )
    from sentrylink.platform import SentryLinkPlatform

    sigma = gaussian_sigma(1.0, 0.5, 1e-6)
    totals: dict[float, float] = {}
    for _ in range(30):
        for a, v in gaussian_rdp_cost(1.0, sigma).items():
            totals[a] = totals.get(a, 0.0) + v
    from sentrylink.crypto.differential_privacy import rdp_to_epsilon

    rdp_ok = rdp_to_epsilon(totals, 1e-3) < 15.0

    acc = PrivacyAccountant(limit=PrivacyBudget(1.0, 1e-3))
    p = SentryLinkPlatform(accountant=acc)
    ids = [p.join(f"O{i}", "retail", "g-rep").org_id for i in range(3)]
    buckets = {oid: [2, 1] for oid in ids}
    p.histogram("g-rep", "retail", buckets, epsilon=1.0)
    try:
        p.histogram("g-rep", "retail", buckets, epsilon=1.0)
        budget_ok = False
    except PermissionError:
        budget_ok = True

    hp = SentryLinkPlatform()
    hids = [hp.join(f"H{i}", "healthcare", "g-hrep").org_id for i in range(3)]
    try:
        hp.variance("g-hrep", "healthcare", {oid: [1.0] for oid in hids})
        healthcare_ok = False
    except PermissionError:
        healthcare_ok = True

    return {
        "rdp_accountant": "PASS" if rdp_ok else "FAIL",
        "budget_enforcement": "PASS" if budget_ok else "FAIL",
        "healthcare_restrictions": "PASS" if healthcare_ok else "FAIL",
    }


def _persistence_section() -> dict:
    import numpy as np

    from sentrylink.federated.client import FederatedClient
    from sentrylink.platform import SentryLinkPlatform
    from sentrylink.storage import InMemoryStateStore, SQLiteStateStore
    from sentrylink.usecases import make_retail_cohort

    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/rep.db"
        p = SentryLinkPlatform(store=SQLiteStateStore(db))
        orgs = make_retail_cohort(n_orgs=3, n_per_org=30, seed=5)
        ids = [p.join(o.name, "retail", "g-rep").org_id for o in orgs]
        clients = {
            oid: FederatedClient(org_id=oid, x=o.x, y=o.y) for oid, o in zip(ids, orgs)
        }
        r1 = p.run_federated_round("g-rep", "retail", clients, epsilon=8.0, epochs=1)
        spent = p.budget_report()["spent_epsilon"]
        n_audit = len(p.audit.entries)
        p.store.close()
        p2 = SentryLinkPlatform(store=SQLiteStateStore(db))
        try:
            replayed_model = p2.servers["g-rep:retail"].model
            assert replayed_model is not None
            replay_ok = (
                sorted(o.org_id for o in p2.registry.list()) == sorted(ids)
                and p2.budget_report()["spent_epsilon"] == spent
                and len(p2.audit.entries) == n_audit
                and p2.audit.verify_chain()
                and bool(np.allclose(replayed_model.flat, r1.model.flat))
            )
        finally:
            p2.store.close()

    class _BrokenStore(InMemoryStateStore):
        def transaction(self):
            raise RuntimeError("disk gone")

    p3 = SentryLinkPlatform(store=_BrokenStore())
    try:
        p3.join("Acme", "retail", "g-f")
        atomic_ok = False
    except RuntimeError:
        atomic_ok = p3.registry.list() == [] and p3.audit.entries == []

    return {
        "restart_replay": "PASS" if replay_ok else "FAIL",
        "atomicity": "PASS" if atomic_ok else "FAIL",
    }


def _demo_section() -> dict:
    from sentrylink.simulate import run_simulation

    out = {}
    for vertical in ("retail", "manufacturing", "healthcare"):
        rep = run_simulation(vertical)
        out[vertical] = (
            "PASS" if rep["audit_valid"] and rep["cohort_size"] == 4 else "FAIL"
        )
    return out


def run_report(*, include_tests: bool = True, include_demo: bool = True) -> dict:
    report = {"title": "SentryLink Release Report", **_versions()}
    report["tests"] = _pytest_section() if include_tests else {
        "passed": 0, "failed": 0, "skipped": 0}
    report["security"] = _security_section()
    report["privacy"] = _privacy_section()
    report["persistence"] = _persistence_section()
    report["demo"] = _demo_section() if include_demo else {}
    return report


def _overall(report: dict) -> bool:
    tests = report["tests"]
    if tests.get("failed", 1) != 0 or tests.get("passed", 0) == 0:
        return False
    for section in ("security", "privacy", "persistence", "demo"):
        for value in report.get(section, {}).values():
            if value == "FAIL":
                return False
    try:
        ok, total = (int(x) for x in report["security"]["redteam"].split("/"))
    except (ValueError, KeyError):
        return False
    return total > 0 and ok == total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SentryLink release report")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = run_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("SentryLink Release Report")
        print("-------------------------")
        print()
        print(f"Protocol version  : {report['protocol_version']}")
        print(f"Schema version    : {report['schema_version']}")
        print(f"Python version    : {report['python_version']}")
        print()
        for section in ("tests", "security", "privacy", "persistence", "demo"):
            print(f"{section.capitalize()}")
            print("-" * len(section))
            for key, value in report[section].items():
                print(f"  {key:24s} : {value}")
            print()
    return 0 if _overall(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
