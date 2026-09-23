"""Local privacy simulator: cohort -> queries -> report (synthetic data only).

Run:  python -m sentrylink.simulate --vertical manufacturing [--json]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .config import MAX_ORG_CONTRIB
from .crypto.secure_aggregation import (
    SecAggClient,
    SecAggServer,
    clip_l2,
    dequantize,
    new_round_id,
    quantize,
)
from .federated.client import FederatedClient
from .platform import SentryLinkPlatform
from .verticals import DATASETS


def _quantization_error() -> float:
    v = np.array([0.0, -3.25, 7.5, 123456.789, -0.000001])
    return float(np.max(np.abs(dequantize(quantize(v)) - v)))


def _aggregation_error() -> float:
    roster = ["a", "b", "c", "d"]
    dim = 8
    rng = np.random.default_rng(7)
    updates = {oid: rng.normal(size=dim) for oid in roster}
    rid = new_round_id()
    server = SecAggServer(round_id=rid, roster=roster, dim=dim)
    clients = {oid: SecAggClient(oid, rid, roster, dim) for oid in roster}
    for oid, c in clients.items():
        server.register(oid, c.public_key)
    keys = server.public_keys()
    for oid, u in updates.items():
        server.receive(clients[oid].mask_and_send(u, keys))
    got = server.finalize()
    expected = dequantize(
        sum((quantize(clip_l2(u)) for u in updates.values()),
            np.zeros(dim, dtype=np.int64))
    )
    return float(np.max(np.abs(got - expected)))


def run_simulation(vertical: str, n_orgs: int = 4, n_per_org: int = 50) -> dict:
    t0 = time.perf_counter()
    dataset = DATASETS[vertical]
    group = f"sim-{vertical}"
    domain = vertical
    platform = SentryLinkPlatform()
    orgs = dataset.generate(n_orgs=n_orgs, n_per_org=n_per_org, seed=21)
    ids = [platform.join(o.name, domain=domain, sector_group=group).org_id for o in orgs]
    col = 0 if vertical != "healthcare" else 2
    pairs = {oid: (o.x[:, col].tolist(), o.y.tolist()) for oid, o in zip(ids, orgs)}
    corr = platform.correlation(group, domain, pairs, epsilon=2.0)
    clients = {oid: FederatedClient(org_id=oid, x=o.x, y=o.y) for oid, o in zip(ids, orgs)}
    dropped = [ids[-1]]
    result = platform.run_federated_round(
        group, domain, clients, epsilon=8.0, epochs=1, drop=dropped
    )
    report = platform.budget_report()
    return {
        "vertical": vertical,
        "cohort_size": n_orgs,
        "query": "correlation",
        "sensitivity": 2.0,
        "mechanism": "laplace",
        "epsilon": 2.0,
        "delta": 0.0,
        "released_correlation": round(float(corr.value), 4),
        "federated_rounds": 1,
        "dropout_rate": round(len(dropped) / n_orgs, 4),
        "round_accuracy": round(float(result.eval_stats["accuracy"]), 4),
        "remaining_budget": round(report["remaining_epsilon"], 4),
        "accounting_regime": "rdp" if report["rdp_complete"] else "basic",
        "quantization_error": _quantization_error(),
        "aggregation_error": _aggregation_error(),
        "audit_valid": platform.audit.verify_chain(),
        "audit_entries": len(platform.audit.entries),
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SentryLink privacy simulator")
    parser.add_argument("--vertical", required=True,
                        choices=sorted(DATASETS),
                        help="vertical to simulate")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    report = run_simulation(args.vertical)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"SENTRYLINK SIMULATION — {args.vertical}")
        for key, value in report.items():
            if key != "vertical":
                print(f"  {key:22s} : {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
