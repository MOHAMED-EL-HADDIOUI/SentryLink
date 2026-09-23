import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sentrylink.api.app import app, get_platform, PLATFORM
from sentrylink.platform import SentryLinkPlatform


def _app_module():
    # NOTE: `import sentrylink.api.app as x` binds the FastAPI instance, not
    # the module (the package attribute shadows the submodule); only
    # sys.modules reaches the PLATFORM global the routes actually read.
    return sys.modules["sentrylink.api.app"]


@pytest.fixture()
def client():
    # fresh platform per test session to avoid cross-test state
    fresh = SentryLinkPlatform()
    mod = _app_module()
    mod.PLATFORM = fresh
    with TestClient(app) as c:
        yield c
    mod.PLATFORM = PLATFORM


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
    members = [_join(client, f"Org{i}", "retail", "g-small") for i in range(2)]
    r = client.post(
        "/queries/histogram",
        json={
            "org_id": members[0]["org_id"],
            "api_key": members[0]["api_key"],
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
            "org_id": orgs[0]["org_id"],
            "api_key": orgs[0]["api_key"],
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


def _hist_payload(org, group, domain="retail", buckets=None, key=None):
    return {
        "org_id": org["org_id"],
        "api_key": key if key is not None else org["api_key"],
        "sector_group": group,
        "domain": domain,
        "org_buckets": buckets or {org["org_id"]: [1, 2]},
        "epsilon": 1.0,
    }


def test_histogram_missing_credentials_rejected(client):
    a = _join(client, "A", "retail", "g-auth")
    payload = _hist_payload(a, "g-auth")
    del payload["org_id"]
    del payload["api_key"]
    assert client.post("/queries/histogram", json=payload).status_code == 422


def test_histogram_bad_and_unknown_credentials_rejected(client):
    a = _join(client, "A", "retail", "g-auth")
    _join(client, "B", "retail", "g-auth")
    _join(client, "C", "retail", "g-auth")
    assert client.post(
        "/queries/histogram", json=_hist_payload(a, "g-auth", key="wrong")
    ).status_code == 401
    ghost = dict(a, org_id="ghost")
    assert client.post(
        "/queries/histogram", json=_hist_payload(ghost, "g-auth")
    ).status_code == 401


def test_histogram_non_member_rejected(client):
    insider = _join(client, "In", "retail", "g-in")
    _join(client, "In2", "retail", "g-in")
    _join(client, "In3", "retail", "g-in")
    outsider = _join(client, "Out", "retail", "g-out")
    buckets = {o: [1, 2] for o in [insider["org_id"], "x", "y"]}
    # valid key, but the org belongs to another cohort
    r = client.post(
        "/queries/histogram", json=_hist_payload(outsider, "g-in", buckets=buckets)
    )
    assert r.status_code == 403
    # and the insider passes auth (governance may still deny on data, not auth)
    r = client.post(
        "/queries/histogram", json=_hist_payload(insider, "g-in", buckets=buckets)
    )
    assert r.status_code != 401


def test_federated_model_requires_member_headers(client):
    r = client.get("/federated/model", params={"sector_group": "g-m", "domain": "retail"})
    assert r.status_code == 401
    a = _join(client, "A", "retail", "g-m")
    r = client.get(
        "/federated/model",
        params={"sector_group": "g-m", "domain": "retail"},
        headers={"X-Org-Id": a["org_id"], "X-API-Key": "wrong"},
    )
    assert r.status_code == 401
    # authenticated member, but no model ran yet for this cohort
    r = client.get(
        "/federated/model",
        params={"sector_group": "g-m", "domain": "retail"},
        headers={"X-Org-Id": a["org_id"], "X-API-Key": a["api_key"]},
    )
    assert r.status_code == 404


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
            "org_id": orgs[0]["org_id"],
            "api_key": orgs[0]["api_key"],
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
