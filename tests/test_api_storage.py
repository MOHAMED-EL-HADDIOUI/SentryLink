"""API behavior against the SQLite backend (plus readiness probes)."""

import sys

import pytest
from fastapi.testclient import TestClient

from sentrylink.api.app import app
from sentrylink.platform import SentryLinkPlatform
from sentrylink.storage import SQLiteStateStore


def _app_module():
    # NOTE: `import sentrylink.api.app as x` binds the FastAPI instance, not
    # the module (the package attribute shadows the submodule). sys.modules
    # is the only reliable handle for swapping the PLATFORM global.
    return sys.modules["sentrylink.api.app"]


@pytest.fixture()
def sqlite_client(tmp_path):
    fresh = SentryLinkPlatform(store=SQLiteStateStore(str(tmp_path / "api.db")))
    mod = _app_module()
    prev = mod.PLATFORM
    mod.PLATFORM = fresh
    try:
        with TestClient(app) as c:
            yield c
    finally:
        mod.PLATFORM = prev
        fresh.store.close()


@pytest.fixture()
def memory_client():
    fresh = SentryLinkPlatform()
    mod = _app_module()
    prev = mod.PLATFORM
    mod.PLATFORM = fresh
    try:
        with TestClient(app) as c:
            yield c
    finally:
        mod.PLATFORM = prev


def _join(client, name, domain, group):
    r = client.post("/orgs", json={"name": name, "domain": domain, "sector_group": group})
    assert r.status_code == 200, r.text
    return r.json()


def test_sqlite_backed_api_flow(sqlite_client):
    orgs = [_join(sqlite_client, f"O{i}", "retail", "g-db") for i in range(3)]
    buckets = {o["org_id"]: [6, 2] for o in orgs}
    r = sqlite_client.post(
        "/queries/histogram",
        json={"org_id": orgs[0]["org_id"], "api_key": orgs[0]["api_key"],
              "sector_group": "g-db", "domain": "retail", "org_buckets": buckets, "epsilon": 1.0},
    )
    assert r.status_code == 200, r.text
    audit = sqlite_client.get("/audit").json()
    assert audit["verified"] is True
    budgets = sqlite_client.get("/budgets").json()
    assert budgets["spent_epsilon"] == pytest.approx(1.0)
    assert budgets["rdp_complete"] is True
    ready = sqlite_client.get("/ready").json()
    assert ready["ready"] is True and ready["storage"] == "sqlite"
    health = sqlite_client.get("/health").json()
    assert health["storage"] == "sqlite" and health["audit_verified"] is True


def test_readiness_memory_backend(memory_client):
    ready = memory_client.get("/ready").json()
    assert ready["ready"] is True and ready["storage"] == "memory"
    assert memory_client.get("/health").json()["storage"] == "memory"


def test_api_state_survives_platform_restart(tmp_path):
    db = str(tmp_path / "restart-api.db")
    mod = _app_module()
    first = SentryLinkPlatform(store=SQLiteStateStore(db))
    prev = mod.PLATFORM
    mod.PLATFORM = first
    try:
        with TestClient(app) as c:
            acme = _join(c, "Acme", "retail", "g-rc")
            _join(c, "B", "retail", "g-rc")
            _join(c, "C", "retail", "g-rc")
        first.store.close()
        # "restart": brand-new platform object over the same database file
        second = SentryLinkPlatform(store=SQLiteStateStore(db))
        mod.PLATFORM = second
        try:
            with TestClient(app) as c2:
                orgs = c2.get("/orgs").json()
                assert any(o["name"] == "Acme" for o in orgs)
                assert c2.get("/ready").json()["ready"] is True
                # original API key still authenticates (hash path) after restart
                buckets = {o["org_id"]: [2, 1] for o in orgs}
                r = c2.post(
                    "/queries/histogram",
                    json={"org_id": acme["org_id"], "api_key": acme["api_key"],
                          "sector_group": "g-rc", "domain": "retail",
                          "org_buckets": buckets, "epsilon": 1.0},
                )
                assert r.status_code == 200, r.text
        finally:
            second.store.close()
    finally:
        mod.PLATFORM = prev
