"""Vertical policies and dataset descriptors."""

import numpy as np

from sentrylink.governance.policy import ALLOWED_METRICS, default_policy_for
from sentrylink.governance.registry import Registry
from sentrylink.verticals import (
    DATASETS,
    HEALTHCARE,
    MANUFACTURING,
    RETAIL,
    HealthcareDataset,
    ManufacturingDataset,
    RetailDataset,
    policy_for_domain,
)


def test_builtin_vertical_values():
    assert HEALTHCARE.min_participants == 4
    assert HEALTHCARE.max_epsilon_per_query == 10.0
    assert HEALTHCARE.allowed_metrics == frozenset(
        {"histogram", "correlation", "federated_model_round"}
    )
    for vp in (RETAIL, MANUFACTURING):
        assert vp.min_participants == 3
        assert vp.max_epsilon_per_query == 25.0
        assert vp.allowed_metrics == frozenset(ALLOWED_METRICS)
    assert all(vp.description for vp in (RETAIL, MANUFACTURING, HEALTHCARE))
    assert policy_for_domain("unknown-domain").key == "default"


def test_default_policy_delegates_to_verticals():
    reg = Registry()
    healthcare = reg.register("H", "healthcare", "hg", org_id="h")
    retail = reg.register("R", "retail", "rt", org_id="r")
    other = reg.register("O", "brand-new-domain", "xx", org_id="o")
    hp, rp, op = (default_policy_for(x) for x in (healthcare, retail, other))
    assert (hp.min_participants, hp.max_epsilon_per_query) == (4, 10.0)
    assert "variance" not in hp.allowed_metrics
    assert (rp.min_participants, rp.max_epsilon_per_query) == (3, 25.0)
    assert set(rp.allowed_metrics) == set(ALLOWED_METRICS)
    assert (op.min_participants, op.max_epsilon_per_query) == (3, 25.0)


def test_dataset_metadata_and_determinism():
    assert [f[0] for f in RetailDataset.FEATURES] == [
        "promo_depth", "price_index", "weather_index", "foot_traffic",
    ]
    assert len(ManufacturingDataset.FEATURES) == 4
    assert len(HealthcareDataset.FEATURES) == 4
    assert RetailDataset.LABEL and HealthcareDataset.LABEL
    a = RetailDataset.generate(n_orgs=2, n_per_org=50, seed=7)
    b = RetailDataset.generate(n_orgs=2, n_per_org=50, seed=7)
    assert len(a) == 2 and all(
        np.array_equal(x.x, y.x) and np.array_equal(x.y, y.y) for x, y in zip(a, b)
    )
    assert set(DATASETS) == {"retail", "manufacturing", "healthcare"}
    for cls in DATASETS.values():
        orgs = cls.generate(n_orgs=2, n_per_org=30, seed=1)
        assert all(o.x.shape[1] == 4 and len(o.x) == 30 for o in orgs)
