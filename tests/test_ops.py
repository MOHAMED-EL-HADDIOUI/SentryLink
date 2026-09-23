"""Settings, rate limiting, redaction, request IDs, structured errors."""

import sys
import time

import pytest
from fastapi.testclient import TestClient

from sentrylink.api.app import app
from sentrylink.observability import log_event, redact
from sentrylink.platform import SentryLinkPlatform
from sentrylink.ratelimit import InMemoryTokenBucket
from sentrylink.settings import Settings, parse_rate_limit


def _app_module():
    return sys.modules["sentrylink.api.app"]


def test_settings_defaults_and_env():
    s = Settings.from_env({})
    assert (s.db, s.log_level, s.rate_limit, s.env) == (
        None, "INFO", "300/minute", "dev")
    s = Settings.from_env({
        "SENTRYLINK_DB": "x.db",
        "SENTRYLINK_LOG_LEVEL": "debug",
        "SENTRYLINK_RATE_LIMIT": "10/hour",
        "SENTRYLINK_ENV": "prod",
    })
    assert (s.db, s.log_level, s.env) == ("x.db", "DEBUG", "prod")
    assert s.rate_limit_parsed() == (10, 3600.0)


def test_settings_rejects_bad_values():
    with pytest.raises(ValueError):
        Settings.from_env({"SENTRYLINK_LOG_LEVEL": "VERBOSE"})
    with pytest.raises(ValueError):
        Settings.from_env({"SENTRYLINK_ENV": "staging"})
    for bad in ("", "abc", "0/minute", "10/day", "10/"):
        with pytest.raises(ValueError):
            parse_rate_limit(bad)


def test_token_bucket_allow_deny_refill():
    ticks = [0.0]
    bucket = InMemoryTokenBucket(2, 60.0, _clock=lambda: ticks[0])
    assert bucket.allow("k") and bucket.allow("k")
    assert not bucket.allow("k")
    assert bucket.allow("other")  # independent keys
    ticks[0] += 61.0
    assert bucket.allow("k")  # window slid past: refilled
    with pytest.raises(ValueError):
        InMemoryTokenBucket(0, 60.0)


def test_redact_strips_credential_shapes():
    payload = {
        "api_key": "secret",
        "nested": {"private_key": "x", "public": 1},
        "tokens": [{"token": "t"}],
        "monkey": "stays",
        "org_id": "abc",
        "epsilon": 1.0,
    }
    clean = redact(payload)
    assert clean["api_key"] == "[REDACTED]"
    assert clean["nested"]["private_key"] == "[REDACTED]"
    assert clean["nested"]["public"] == 1
    assert clean["tokens"] == [{"token": "[REDACTED]"}]
    assert clean["monkey"] == "stays" and clean["org_id"] == "abc"
    # input not mutated
    assert payload["api_key"] == "secret"


def test_log_event_never_emits_secrets(caplog):
    import json
    import logging

    with caplog.at_level(logging.INFO, logger="sentrylink"):
        log_event("test.event", request_id="r1", api_key="zzz", status=200)
    line = next(m for m in caplog.messages if "test.event" in m)
    body = json.loads(line)
    assert body["api_key"] == "[REDACTED]" and body["status"] == 200
    assert "zzz" not in line


def _fresh_client():
    fresh = SentryLinkPlatform()
    mod = _app_module()
    prev = mod.PLATFORM
    mod.PLATFORM = fresh
    return fresh, mod, prev


def test_request_id_echo_and_audit_propagation():
    fresh, mod, prev = _fresh_client()
    try:
        with TestClient(app) as c:
            a = c.post("/orgs", json={"name": "A", "domain": "retail",
                                      "sector_group": "g-r"}).json()
            r = c.post(
                "/queries/histogram",
                json={"org_id": a["org_id"], "api_key": a["api_key"],
                      "sector_group": "g-x", "domain": "retail",
                      "org_buckets": {a["org_id"]: [1]}, "epsilon": 1.0},
                headers={"X-Request-Id": "demo-req-1"},
            )
            assert r.headers["x-request-id"] == "demo-req-1"
            generated = c.post("/orgs", json={"name": "B", "domain": "retail",
                                              "sector_group": "g-r"}).headers["x-request-id"]
            assert generated and generated != "demo-req-1"
    finally:
        mod.PLATFORM = prev


def test_request_id_reaches_audit_entries():
    fresh, mod, prev = _fresh_client()
    try:
        orgs = [fresh.join(f"O{i}", "retail", "g-r") for i in range(3)]
        fresh.histogram("g-r", "retail", {o.org_id: [2, 1] for o in orgs},
                        epsilon=1.0, request_id="req-42")
        entry = fresh.audit.by_action("query.histogram")[-1]
        assert entry["details"]["request_id"] == "req-42"
    finally:
        mod.PLATFORM = prev


def test_structured_error_bodies_carry_codes():
    fresh, mod, prev = _fresh_client()
    try:
        with TestClient(app) as c:
            a = c.post("/orgs", json={"name": "A", "domain": "retail",
                                      "sector_group": "g-e"}).json()
            b = c.post("/orgs", json={"name": "B", "domain": "retail",
                                      "sector_group": "g-e"}).json()
            r = c.post(
                "/queries/histogram",
                json={"org_id": a["org_id"], "api_key": a["api_key"],
                      "sector_group": "g-e", "domain": "retail",
                      "org_buckets": {a["org_id"]: [1], b["org_id"]: [1]},
                      "epsilon": 1.0},
            )
            assert r.status_code == 403
            body = r.json()["error"]
            assert body["code"] == "COHORT_TOO_SMALL"
            assert body["request_id"]
            r = c.post(
                "/queries/histogram",
                json={"org_id": a["org_id"], "api_key": "wrong",
                      "sector_group": "g-e", "domain": "retail",
                      "org_buckets": {}, "epsilon": 1.0},
            )
            assert r.status_code == 401
            assert r.json()["error"]["code"] == "BAD_CREDENTIAL"
    finally:
        mod.PLATFORM = prev


def test_rate_limit_distinct_from_budget():
    fresh, mod, prev = _fresh_client()
    old_limiter = mod.RATE_LIMITER
    mod.RATE_LIMITER = InMemoryTokenBucket(1, 3600.0)
    try:
        with TestClient(app) as c:
            orgs = [c.post("/orgs", json={"name": f"O{i}", "domain": "retail",
                                          "sector_group": "g-rl"}).json()
                    for i in range(3)]
            payload = {"org_id": orgs[0]["org_id"], "api_key": orgs[0]["api_key"],
                       "sector_group": "g-rl", "domain": "retail",
                       "org_buckets": {o["org_id"]: [2, 1] for o in orgs},
                       "epsilon": 1.0}
            assert c.post("/queries/histogram", json=payload).status_code == 200
            r = c.post("/queries/histogram", json=payload)
            assert r.status_code == 429
            assert r.json()["error"]["code"] == "RATE_LIMITED"
            # the rejected call spent no budget: only the first charge landed
            assert c.get("/budgets").json()["spent_epsilon"] == 1.0
    finally:
        mod.RATE_LIMITER = old_limiter
        mod.PLATFORM = prev


def test_privacy_review_clean_on_real_tree():
    import sys

    sys.path.insert(0, "scripts")
    from privacy_review import review

    assert review() == []


def test_malformed_request_ids_regenerated():
    fresh, mod, prev = _fresh_client()
    try:
        with TestClient(app) as c:
            r = c.get("/health", headers={"X-Request-Id": "has spaces!!"})
            assert r.headers["x-request-id"] != "has spaces!!"
            assert r.headers["x-request-id"]
    finally:
        mod.PLATFORM = prev
