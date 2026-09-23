import numpy as np

from sentrylink.platform import SentryLinkPlatform
from sentrylink.federated.client import FederatedClient
from sentrylink.usecases import (
    make_healthcare_cohort,
    make_manufacturing_cohort,
    make_retail_cohort,
)


def _setup(platform, orgs, domain, group):
    regs = [platform.join(o.name, domain=domain, sector_group=group) for o in orgs]
    ids = [r.org_id for r in regs]
    return ids


def test_retail_end_to_end():
    p = SentryLinkPlatform()
    orgs = make_retail_cohort(n_orgs=4, n_per_org=250, seed=7)
    ids = _setup(p, orgs, "retail", "rc")

    clients = {oid: FederatedClient(org_id=oid, x=o.x, y=o.y) for oid, o in zip(ids, orgs)}
    r = p.run_federated_round("rc", "retail", clients, epsilon=8.0, epochs=2)
    # a few more rounds so the cohort model clearly beats chance
    for _ in range(5):
        r = p.run_federated_round("rc", "retail", clients, epsilon=8.0, epochs=2)
    assert len(r.participants) == 4
    assert r.eval_stats["accuracy"] > 0.6

    buckets = {}
    for oid, o in zip(ids, orgs):
        binned = np.digitize(o.x[:, 3], [0.0])
        buckets[oid] = [int(np.sum((binned == 0) & (o.y == 1))),
                        int(np.sum((binned == 1) & (o.y == 1)))]
    h = p.histogram("rc", "retail", buckets, labels=["low", "high"], epsilon=1.0)
    assert set(h.participants) == set(ids)
    assert isinstance(h.value, dict)

    pairs = {oid: (o.x[:, 0].tolist(), o.y.tolist()) for oid, o in zip(ids, orgs)}
    c = p.correlation("rc", "retail", pairs, epsilon=1.0)
    assert -1.0 <= c.value <= 1.0
    assert p.audit.verify_chain()


def test_manufacturing_end_to_end():
    p = SentryLinkPlatform()
    orgs = make_manufacturing_cohort(n_orgs=4, n_per_org=200, seed=11)
    ids = _setup(p, orgs, "manufacturing", "mc")
    buckets = {oid: [5, 3, 2, 1] for oid in ids}
    h = p.histogram("mc", "manufacturing", buckets, epsilon=1.0, purpose="defect bands")
    assert len(h.participants) == 4
    values = {oid: o.y.tolist() for oid, o in zip(ids, orgs)}
    v = p.variance("mc", "manufacturing", values, epsilon=1.0)
    assert v.raw_value >= 0


def test_healthcare_end_to_end():
    p = SentryLinkPlatform()
    orgs = make_healthcare_cohort(n_orgs=4, n_per_org=220, seed=23)
    ids = _setup(p, orgs, "healthcare", "hc")
    pairs = {oid: (o.x[:, 2].tolist(), o.y.tolist()) for oid, o in zip(ids, orgs)}
    c = p.correlation("hc", "healthcare", pairs, epsilon=1.0)
    assert -1.0 <= c.value <= 1.0
    # healthcare policy: variance not allowed for hospitals
    try:
        p.variance("hc", "healthcare", {oid: o.y.tolist() for oid, o in zip(ids, orgs)})
        raised = False
    except PermissionError:
        raised = True
    assert raised


def test_consent_revocation_blocks_query():
    p = SentryLinkPlatform()
    orgs = make_retail_cohort(n_orgs=4, n_per_org=100, seed=1)
    ids = _setup(p, orgs, "retail", "rev")
    p.set_consent(ids[0], set())  # opts out of everything
    buckets = {oid: [1, 2] for oid in ids}
    try:
        p.histogram("rev", "retail", buckets, epsilon=1.0)
        # still allowed if 3 others consent
        ok = True
    except PermissionError:
        ok = False
    assert ok
    # opt out two more -> below cohort floor
    p.set_consent(ids[1], set())
    p.set_consent(ids[2], set())
    try:
        p.histogram("rev", "retail", buckets, epsilon=1.0)
        blocked = False
    except PermissionError:
        blocked = True
    assert blocked
