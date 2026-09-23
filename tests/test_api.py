import numpy as np
import pytest
from fastapi.testclient import TestClient

from sentrylink.api.app import app, get_platform, PLATFORM
from sentrylink.platform import SentryLinkPlatform


@pytest.fixture()
def client():
    # fresh platform per test session to avoid cross-test state
    fresh = SentryLinkPlatform()
    import sentrylink.api.app as app_mod

    app_mod.PLATFORM = fresh
    with TestClient(app) as c:
        yield c
    app_mod.PLATFORM = PLATFORM


def _join(client, name, domain, group):
    r = client.post(
        "/orgs", json={"name": name, "domain": domain, "sector_group": group}
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["audit_verified"] is True


def test_join_and_list(client):
    info = _join(client, "Acme", "retail", "g-api")
    assert "org_id" in info and "api_key" in info
    orgs = client.get("/orgs").json()
    assert any(o["name"] == "Acme" for o in orgs)


def test_consent_requires_valid_key(client):
    info = _join(client, "Acme", "retail", "g-api")
    bad = client.post(
        "/consent",
        json={
            "org_id": info["org_id"],
            "api_key": "nope",
            "allowed_metrics": ["histogram"],
        },
    )
    assert bad.status_code == 401
    good = client.post(
        "/consent",
        json={
            "org_id": info["org_id"],
            "api_key": info["api_key"],
            "allowed_metrics": ["histogram", "correlation"],
        },
    )
    assert good.status_code == 200


def test_histogram_requires_min_cohort(client):
    for i in range(2):
        _join(client, f"Org{i}", "retail", "g-small")
    r = client.post(
        "/queries/histogram",
        json={
            "sector_group": "g-small",
            "domain": "retail",
            "org_buckets": {"x": [1, 2]},
            "epsilon": 1.0,
        },
    )
    assert r.status_code == 403


def test_histogram_happy_path(client):
    orgs = []
    for i in range(4):
        orgs.append(_join(client, f"Org{i}", "retail", "g-hist"))
    buckets = {o["org_id"]: [10 + i, 5, 1] for i, o in enumerate(orgs)}
    r = client.post(
        "/queries/histogram",
        json={
            "sector_group": "g-hist",
            "domain": "retail",
            "org_buckets": buckets,
            "labels": ["b0", "b1", "b2"],
            "epsilon": 1.0,
            "delta": 1e-5,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["metric"] == "histogram"
    assert set(body["participants"]) == {o["org_id"] for o in orgs}
    # released values are noisy ints
    assert all(isinstance(v, int) for v in body["value"].values())


def test_federated_round_api(client):
    orgs = [_join(client, f"F{i}", "retail", "g-fl") for i in range(4)]
    rng = np.random.default_rng(0)
    clients_payload = []
    for o in orgs:
        x = rng.normal(size=(80, 4)).tolist()
        y = (rng.uniform(size=80) > 0.5).astype(int).tolist()
        clients_payload.append({"org_id": o["org_id"], "x": x, "y": y})
    r = client.post(
        "/federated/round",
        json={
            "sector_group": "g-fl",
            "domain": "retail",
            "clients": clients_payload,
            "epsilon": 1.0,
            "epochs": 1,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dp_applied"] is True
    assert len(body["weights"]) == 5
    assert body["eval_stats"]["n"] == 320


def test_audit_and_budget_endpoints(client):
    _join(client, "A", "retail", "g-x")
    audit = client.get("/audit").json()
    assert audit["verified"] is True
    budgets = client.get("/budgets").json()
    assert "spent_epsilon" in budgets
