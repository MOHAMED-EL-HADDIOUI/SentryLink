"""Lightweight latency benchmarks (tiny synthetic fixtures, no claims).

Run:  python -m sentrylink.bench [--json]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .crypto.secret_sharing import reconstruct, share_vector
from .crypto.secure_aggregation import SecAggClient, SecAggServer, new_round_id
from .federated.client import FederatedClient
from .governance.audit import AuditLog
from .platform import SentryLinkPlatform
from .storage import SQLiteStateStore
from .usecases import make_retail_cohort


def _timed(fn, repeats=1):
    best = min(
        (t1 := time.perf_counter(), fn(), time.perf_counter())[2] - t1
        for _ in range(repeats)
    )
    return best


def run_benchmarks() -> dict[str, float]:
    out: dict[str, float] = {}
    out["registration"] = _timed(_bench_registration)
    out["histogram"] = _timed(_bench_histogram)
    out["variance"] = _timed(_bench_variance)
    out["correlation"] = _timed(_bench_correlation)
    out["fl_round"] = _timed(_bench_fl_round)
    out["secagg_mask_finalize"] = _timed(_bench_secagg)
    out["mpc_reconstruct"] = _timed(_bench_mpc)
    out["sqlite_transaction"] = _timed(_bench_sqlite)
    out["audit_append"] = _timed(_bench_audit)
    return out


def _bench_registration():
    p = SentryLinkPlatform()
    for i in range(3):
        p.join(f"B{i}", "retail", "g-bench")


def _bench_histogram():
    p = SentryLinkPlatform()
    ids = [p.join(f"B{i}", "retail", "g-bench").org_id for i in range(3)]
    p.histogram("g-bench", "retail", {oid: [5, 3, 1] for oid in ids}, epsilon=1.0)


def _bench_variance():
    p = SentryLinkPlatform()
    ids = [p.join(f"B{i}", "retail", "g-bench").org_id for i in range(3)]
    rng = np.random.default_rng(0)
    p.variance(
        "g-bench", "retail",
        {oid: rng.normal(size=50).tolist() for oid in ids}, epsilon=1.0,
    )


def _bench_correlation():
    p = SentryLinkPlatform()
    ids = [p.join(f"B{i}", "retail", "g-bench").org_id for i in range(3)]
    rng = np.random.default_rng(1)
    x = rng.normal(size=50).tolist()
    y = rng.normal(size=50).tolist()
    p.correlation(
        "g-bench", "retail", {oid: (x, y) for oid in ids}, epsilon=1.0
    )


def _bench_fl_round():
    orgs = make_retail_cohort(n_orgs=3, n_per_org=60, seed=2)
    p = SentryLinkPlatform()
    ids = [p.join(o.name, "retail", "g-bench").org_id for o in orgs]
    clients = {oid: FederatedClient(org_id=oid, x=o.x, y=o.y) for oid, o in zip(ids, orgs)}
    p.run_federated_round("g-bench", "retail", clients, epsilon=8.0, epochs=1)


def _bench_secagg():
    roster = [f"c{i}" for i in range(4)]
    dim = 64
    rng = np.random.default_rng(3)
    rid = new_round_id()
    server = SecAggServer(round_id=rid, roster=roster, dim=dim)
    clients = {oid: SecAggClient(oid, rid, roster, dim) for oid in roster}
    for oid, c in clients.items():
        server.register(oid, c.public_key)
    keys = server.public_keys()
    for oid, c in clients.items():
        server.receive(c.mask_and_send(rng.normal(size=dim), keys))
    server.finalize()


def _bench_mpc():
    shares = share_vector([int(v) for v in np.arange(1000)], ["node-a", "node-b"])
    reconstruct([shares["node-a"], shares["node-b"]])


def _bench_sqlite():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteStateStore(f"{tmp}/b.db")
        log = AuditLog()
        with store.transaction() as tx:
            entry = log.record("bench", i=1)
            tx.append_audit(entry, expect_prev=entry["prev_hash"])
        store.close()


def _bench_audit():
    log = AuditLog()
    for i in range(20):
        log.record("bench", i=i)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SentryLink benchmarks")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    results = run_benchmarks()
    if args.json:
        print(json.dumps({"seconds": results}, indent=2))
    else:
        print("SENTRYLINK BENCHMARKS (seconds, tiny fixtures)")
        for key, value in results.items():
            print(f"  {key:22s} : {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
