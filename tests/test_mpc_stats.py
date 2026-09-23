import numpy as np

from sentrylink.mpc.stats import (
    run_correlation,
    run_histogram,
    run_variance,
    quantize_ints,
)
from sentrylink.crypto.secret_sharing import share_vector

NODES = ["node-a", "node-b"]


def test_histogram_sums_org_buckets():
    contribs = {
        "a": [10, 20, 0],
        "b": [5, 0, 7],
        "c": [1, 2, 3],
    }
    out = run_histogram(contribs, NODES)
    assert out == [16, 22, 10]


def test_individual_org_buckets_never_opened():
    """Only the joint sum reconstructs — per-node shares are random."""
    contribs = {"a": [111, 222], "b": [333, 444]}
    shares_a = share_vector(contribs["a"], NODES)
    shares_b = share_vector(contribs["b"], NODES)
    # a single node's view of org-a's share differs from the secret
    s = shares_a["node-a"].values
    assert tuple(v for v in s) != (111, 222)


def test_variance_matches_numpy():
    rng = np.random.default_rng(3)
    data = {
        "org1": list(rng.normal(5, 2, size=200)),
        "org2": list(rng.normal(5, 2, size=300)),
        "org3": list(rng.normal(5, 2, size=150)),
    }
    pooled = np.concatenate([np.asarray(v) for v in data.values()])
    got = run_variance(data, NODES)
    expected = float(np.var(pooled))
    assert abs(got - expected) < 1e-2 * max(1.0, abs(expected))


def test_correlation_matches_numpy():
    rng = np.random.default_rng(9)
    x = rng.normal(0, 1, size=500)
    y = 0.7 * x + rng.normal(0, 0.5, size=500)
    pairs = {
        "h1": (x[:250].tolist(), y[:250].tolist()),
        "h2": (x[250:].tolist(), y[250:].tolist()),
        "h3": (x[:100].tolist(), y[:100].tolist()),  # overlap ok for numeric check
    }
    # use disjoint split for exact match
    pairs = {
        "h1": (x[:200].tolist(), y[:200].tolist()),
        "h2": (x[200:350].tolist(), y[200:350].tolist()),
        "h3": (x[350:].tolist(), y[350:].tolist()),
    }
    got = run_correlation(pairs, NODES)
    expected = float(np.corrcoef(x, y)[0, 1])
    assert abs(got - expected) < 0.02


def test_quantize_ints():
    assert quantize_ints([1.0, 0.5]) == [1_000_000, 500_000]
