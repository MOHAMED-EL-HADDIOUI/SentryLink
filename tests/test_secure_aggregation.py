import numpy as np

from sentrylink.crypto.secure_aggregation import SecureAggregator, clip_l2, quantize, dequantize
from sentrylink.config import UPDATE_CLIP


def _make_round(roster, dim, round_id="r1"):
    return SecureAggregator(round_id=round_id, roster=roster, dim=dim)


def test_quantize_roundtrip():
    v = np.array([0.1, -0.25, 3.5])
    q = quantize(v)
    assert q.dtype == np.int64
    np.testing.assert_allclose(dequantize(q), v, atol=1e-6)


def test_clip_l2():
    v = np.array([3.0, 4.0])  # norm 5
    c = clip_l2(v, bound=1.0)
    assert np.linalg.norm(c) == pytest_approx(1.0)


def pytest_approx(x, rel=1e-6):
    class A:
        def __eq__(self, other):
            return abs(other - x) <= rel * max(1.0, abs(x))
    return A()


def test_masks_cancel_without_dropout():
    roster = ["a", "b", "c"]
    dim = 8
    agg = _make_round(roster, dim, round_id="t1")
    rng = np.random.default_rng(0)
    updates = {oid: rng.normal(size=dim) for oid in roster}
    for oid in roster:
        agg.client_register(oid)
    for oid, u in updates.items():
        agg.client_mask_and_send(oid, u)

    got = agg.finalize()
    expected = dequantize(sum((quantize(clip_l2(u)) for u in updates.values()), np.zeros(dim, dtype=np.int64)))
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_masks_cancel_with_dropout():
    roster = ["a", "b", "c", "d"]
    dim = 16
    agg = _make_round(roster, dim, round_id="t2")
    rng = np.random.default_rng(1)
    updates = {oid: rng.normal(size=dim) for oid in roster}
    for oid in roster:
        agg.client_register(oid)
    active = ["a", "b", "c"]
    for oid in active:
        agg.client_mask_and_send(oid, updates[oid])
    for survivor in active:
        agg.client_reveal_seed(survivor, "d")

    got = agg.finalize()
    expected_vec = sum(
        (quantize(clip_l2(updates[oid])) for oid in active), np.zeros(dim, dtype=np.int64)
    )
    np.testing.assert_allclose(got, dequantize(expected_vec), atol=1e-9)


def test_updates_clipped_before_masking():
    roster = ["a", "b"]
    agg = _make_round(roster, 4, round_id="t3")
    for oid in roster:
        agg.client_register(oid)
    big = np.full(4, 100.0)
    agg.client_mask_and_send("a", big)
    agg.client_mask_and_send("b", np.zeros(4))
    got = agg.finalize()
    clipped = clip_l2(big, UPDATE_CLIP)
    np.testing.assert_allclose(got, clipped, atol=1e-6)
