"""Property-style, boundary, and edge-case tests (stdlib randomness only)."""

import numpy as np
import pytest

from sentrylink.config import MAX_ORG_CONTRIB
from sentrylink.crypto.differential_privacy import gaussian_sigma
from sentrylink.crypto.secret_sharing import reconstruct, share_vector
from sentrylink.crypto.secure_aggregation import (
    SecAggClient,
    SecAggServer,
    clip_l2,
    dequantize,
    new_round_id,
    quantize,
)
from sentrylink.governance.policy import PolicyEngine
from sentrylink.mpc.stats import run_correlation, run_histogram, run_variance
from sentrylink.platform import SentryLinkPlatform

NODES = ["node-a", "node-b"]


def test_sharing_roundtrip_random_values():
    field_p = (1 << 127) - 1
    rng = np.random.default_rng(11)
    for trial in range(10):
        n = int(rng.integers(1, 20))
        vals = [int(v) for v in rng.integers(0, 1 << 62, size=n)]
        vals[0] = 0  # zero element included
        vals[-1] = field_p - 1 - trial  # field edge included
        shares = share_vector(vals, NODES)
        assert reconstruct([shares[m] for m in NODES]) == vals
    # callers must mod-map first: out-of-field inputs are rejected loudly
    with pytest.raises(ValueError):
        share_vector([-1, 2], NODES)
    with pytest.raises(ValueError):
        share_vector([field_p], NODES)


def test_mask_cancellation_roster_sizes():
    rng = np.random.default_rng(12)
    for size in (2, 3, 4, 5, 6):
        roster = [f"o{i}" for i in range(size)]
        dim = 5
        rid = new_round_id()
        server = SecAggServer(round_id=rid, roster=roster, dim=dim)
        clients = {oid: SecAggClient(oid, rid, roster, dim) for oid in roster}
        for oid, c in clients.items():
            server.register(oid, c.public_key)
        keys = server.public_keys()
        updates = {oid: rng.normal(size=dim) for oid in roster}
        for oid, u in updates.items():
            server.receive(clients[oid].mask_and_send(u, keys))
        expected = dequantize(
            sum((quantize(clip_l2(u)) for u in updates.values()),
                np.zeros(dim, dtype=np.int64))
        )
        np.testing.assert_allclose(server.finalize(), expected, atol=1e-9)


def test_quantize_tolerance_documented():
    # rint() rounds to nearest ulp: |err| <= 0.5 / QUANT_SCALE < 1e-6,
    # including negatives, zeros, sparse and large permitted magnitudes.
    cases = [
        [0.0, 0.0, 0.0],
        [-3.25, 0.0, 7.5, 0.0],
        [123456.789, -0.000001, 1e9, -1e9],
    ]
    for values in cases:
        v = np.asarray(values, dtype=np.float64)
        assert np.max(np.abs(dequantize(quantize(v)) - v)) < 1e-6


def test_single_survivor_reveals_only_its_own_sum():
    # With one survivor the aggregate necessarily equals its (clipped,
    # quantized) update — asserting exactness proves nothing extra leaks.
    roster = ["a", "b"]
    dim = 6
    rid = new_round_id()
    server = SecAggServer(round_id=rid, roster=roster, dim=dim)
    clients = {oid: SecAggClient(oid, rid, roster, dim) for oid in roster}
    for oid, c in clients.items():
        server.register(oid, c.public_key)
    keys = server.public_keys()
    update = np.random.default_rng(13).normal(size=dim)
    server.receive(clients["a"].mask_and_send(update, keys))
    server.add_recovery_seed("a", "b", clients["a"].reveal_seed("b", keys["b"]))
    np.testing.assert_allclose(server.finalize(), clip_l2(update), atol=1e-6)


def test_mpc_degenerate_inputs():
    assert run_variance({"a": [5.0, 5.0, 5.0]}, NODES) == pytest.approx(0.0)
    assert run_variance({"a": [1.0]}, NODES) == 0.0
    assert run_correlation({"a": ([1.0, 1.0], [0.0, 1.0])}, NODES) == 0.0
    assert run_correlation({"a": ([1.0], [2.0])}, NODES) == 0.0
    with pytest.raises(ValueError):
        run_histogram({}, NODES)
    with pytest.raises(ValueError):
        run_correlation({"a": ([1.0, 2.0], [1.0])}, NODES)


def test_contribution_cap_boundary():
    big = [1000] * 10
    capped = PolicyEngine.cap_contributions(big)
    assert float(np.linalg.norm(np.asarray(capped))) <= MAX_ORG_CONTRIB + 1e-9
    small = [1, 2, 3]
    assert PolicyEngine.cap_contributions(small) == small  # inside: untouched


def test_dp_sigma_matches_sensitivity_contract():
    # FL mean sensitivity 2C/k with C=1.0: sigma is exactly calibrated.
    from sentrylink.config import UPDATE_CLIP
    from sentrylink.federated.client import FederatedClient
    from sentrylink.usecases import make_retail_cohort

    orgs = make_retail_cohort(n_orgs=4, n_per_org=60, seed=7)
    p = SentryLinkPlatform()
    ids = [p.join(o.name, "retail", "g-s").org_id for o in orgs]
    clients = {oid: FederatedClient(org_id=oid, x=o.x, y=o.y) for oid, o in zip(ids, orgs)}
    result = p.run_federated_round("g-s", "retail", clients, epsilon=2.0,
                                   delta=1e-5, epochs=1)
    expected = gaussian_sigma(2.0 * UPDATE_CLIP / 4, 2.0, 1e-5)
    assert result.dp_sigma == pytest.approx(expected)
    assert result.delta_used == 1e-5


def test_epsilon_cap_boundary_exact():
    p = SentryLinkPlatform()
    ids = [p.join(f"H{i}", "healthcare", "hg").org_id for i in range(4)]
    buckets = {oid: [2, 1] for oid in ids}
    # epsilon == cap is allowed (exclusion is strictly-greater-than)
    assert p.histogram("hg", "healthcare", buckets, epsilon=10.0).participants
    with pytest.raises(PermissionError):
        p.histogram("hg", "healthcare", buckets, epsilon=10.5)


def test_audit_tamper_variants():
    from sentrylink.governance.audit import AuditLog

    log = AuditLog()
    for i in range(4):
        log.record("op", i=i)
    assert log.verify_chain()

    def _clone():
        clone = AuditLog()
        clone.restore([dict(e) for e in log.entries])
        return clone

    tampered = _clone()
    tampered.entries[2]["ts"] += 1.0
    assert not tampered.verify_chain()

    tampered = _clone()
    tampered.entries[1]["prev_hash"] = "f" * 64
    assert not tampered.verify_chain()

    # Tail truncation yields a self-consistent PREFIX, so the chain alone
    # cannot catch it — that is exactly why head/count checkpoints exist.
    head, count = log.entries[-1]["hash"], len(log.entries)
    truncated = _clone()
    truncated.entries.pop()
    assert truncated.verify_chain()  # prefix consistency holds...
    assert (truncated.entries[-1]["hash"], len(truncated.entries)) != (head, count)

    tampered = _clone()
    tampered.entries[0], tampered.entries[1] = tampered.entries[1], tampered.entries[0]
    assert not tampered.verify_chain()

    forged = _clone()
    forged.entries.append({**forged.entries[-1], "prev_hash": "0" * 64})
    assert not forged.verify_chain()
