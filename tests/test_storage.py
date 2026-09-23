"""Storage-layer tests: schema, codec round-trips, transactions, safety."""

import threading

import numpy as np
import pytest
from sqlalchemy import inspect

from sentrylink.crypto.differential_privacy import (
    PrivacyAccountant,
    PrivacyBudget,
    gaussian_rdp_cost,
    laplace_rdp_cost,
)
from sentrylink.federated.model import LogisticModel
from sentrylink.federated.server import RoundResult
from sentrylink.governance.audit import AuditLog
from sentrylink.governance.policy import ConsentPolicy, default_policy_for
from sentrylink.governance.registry import Organization, Registry, hash_api_key
from sentrylink.storage import (
    CURRENT_SCHEMA_VERSION,
    InMemoryStateStore,
    SQLiteStateStore,
    codec,
    get_schema_version,
    init_db,
    migrate,
)
from sentrylink.storage.engine import create_engine_for


def _sqlite(tmp_path):
    return SQLiteStateStore(str(tmp_path / "test.db"))


def test_schema_version_and_idempotent_migrate(tmp_path):
    store = _sqlite(tmp_path)
    try:
        assert get_schema_version(store.engine) == CURRENT_SCHEMA_VERSION == 2
        assert migrate(store.engine) == 2  # re-run is a no-op
        tables = set(inspect(store.engine).get_table_names())
        assert {
            "schema_meta", "orgs", "policies", "privacy_budget",
            "federated_servers", "federated_rounds", "audit_log",
        } <= tables
    finally:
        store.close()


def test_migrate_from_empty_engine(tmp_path):
    from sentrylink.storage.engine import normalize_url

    engine = create_engine_for(normalize_url(str(tmp_path / "bare.db")))
    try:
        assert get_schema_version(engine) == 0
        assert migrate(engine) == 2
    finally:
        engine.dispose()


def test_migrate_v1_to_v2_preserves_data(tmp_path):
    """Simulate a v1 database, upgrade, and prove data + defaults survive."""
    from sqlalchemy import text

    from sentrylink.storage.engine import normalize_url

    engine = create_engine_for(normalize_url(str(tmp_path / "v1.db")))
    try:
        from sentrylink.storage.models import Base

        Base.metadata.create_all(engine)
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE orgs DROP COLUMN key_salt"))
            conn.execute(text("ALTER TABLE federated_rounds DROP COLUMN delta_used"))
            conn.execute(text("DELETE FROM schema_meta"))
            conn.execute(text("INSERT INTO schema_meta (id, version) VALUES (1, 1)"))
            conn.execute(
                text(
                    "INSERT INTO orgs (org_id, name, domain, sector_group, "
                    "api_key_hash, key_algo, active) VALUES "
                    "('o1', 'Acme', 'retail', 'g', 'h', 'pbkdf2-sha256', 1)"
                )
            )
        assert get_schema_version(engine) == 1
        assert migrate(engine) == 2
        # data intact, new columns present with safe defaults
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT org_id, key_salt FROM orgs WHERE org_id = 'o1'")
            ).first()
            assert row[0] == "o1" and row[1] == ""
            assert migrate(engine) == 2  # restart-safe re-run
    finally:
        engine.dispose()


def test_org_policy_roundtrip_sqlite(tmp_path):
    store = _sqlite(tmp_path)
    try:
        reg = Registry()
        org = reg.register("Hospital", "healthcare", "hg", org_id="h0")
        pol = default_policy_for(org)
        with store.transaction() as tx:
            tx.save_org(codec.encode_org(org))
            tx.save_policy(org.org_id, codec.encode_policy(pol))
        state = store.load()
        assert state is not None
        back_org = codec.decode_org(state.orgs[0])
        back_pol = codec.decode_policy(state.policies["h0"])
        assert (back_org.org_id, back_org.domain, back_org.sector_group) == ("h0", "healthcare", "hg")
        assert back_pol.min_participants == 4
        assert back_pol.max_epsilon_per_query == 10.0
        assert back_pol.allowed_metrics == pol.allowed_metrics
    finally:
        store.close()


def test_api_key_hash_and_verify():
    reg = Registry()
    org = reg.register("Acme", "retail", "g1")
    assert org.api_key_hash.startswith("pbkdf2-sha256$")
    assert org.verify_key(org.api_key)
    assert not org.verify_key("wrong")
    # restored orgs carry no plaintext but still authenticate via hash
    restored = Organization(
        org_id=org.org_id, name=org.name, domain=org.domain,
        sector_group=org.sector_group, api_key="", api_key_hash=org.api_key_hash,
        key_salt=org.key_salt,
    )
    assert restored.api_key == ""
    assert restored.verify_key(org.api_key)
    assert not restored.verify_key("wrong")
    assert hash_api_key("k", salt="s1") == hash_api_key("k", salt="s1")  # deterministic
    assert hash_api_key("k", salt="s1") != hash_api_key("k", salt="s2")  # salt matters
    assert org.key_salt and len(org.key_salt) >= 32  # random per-org salt


def test_budget_snapshot_roundtrip_with_float_rdp_keys():
    acc = PrivacyAccountant(limit=PrivacyBudget(10.0, 1e-3))
    acc.charge(PrivacyBudget(1.0, 0.0), purpose="h", subject="s", rdp=laplace_rdp_cost(1.0))
    acc.charge(PrivacyBudget(0.5, 1e-6), purpose="f", subject="s",
               rdp=gaussian_rdp_cost(0.5, 2.0))
    back = codec.decode_budget(codec.encode_budget(acc))
    assert back.spent == acc.spent
    assert back.limit == acc.limit
    assert back.events == acc.events
    assert back.rdp_complete is True
    assert set(back.rdp_totals) == set(acc.rdp_totals)
    assert all(isinstance(k, float) for k in back.rdp_totals)  # JSON str keys restored
    assert back.rdp_epsilon() == pytest.approx(acc.rdp_epsilon())


def test_server_round_codec_roundtrip():
    dim = 4
    model = LogisticModel.zeros(dim)
    result = RoundResult(
        round_id="r1", participants=["a", "b"], dropped=[],
        aggregate_delta=np.arange(dim + 1, dtype=np.float64),
        model=model, eval_stats={"n": 10, "mean_loss": 0.5, "accuracy": 0.8, "n_correct": 8},
        dp_applied=True, epsilon_used=1.0, dp_sigma=0.25,
    )
    from sentrylink.federated.server import FederatedServer

    server = FederatedServer(dim=dim)
    server.model = model
    server.history = [result]
    raw_server = codec.encode_server("g:retail", server)
    raw_round = codec.encode_round("g:retail", result)
    back = codec.decode_server("g:retail", raw_server, [raw_round])
    assert back.dim == dim
    np.testing.assert_array_equal(back.model.flat, model.flat)
    assert len(back.history) == 1
    np.testing.assert_array_equal(back.history[0].aggregate_delta, result.aggregate_delta)
    assert back.history[0].eval_stats == result.eval_stats
    assert back.history[0].dp_sigma == 0.25


def test_audit_append_order_and_chain(tmp_path):
    store = _sqlite(tmp_path)
    try:
        log = AuditLog()
        with store.transaction() as tx:
            for i in range(3):
                tx.append_audit(log.record("op", i=i))
        state = store.load()
        assert [e["details"]["i"] for e in state.audit] == [0, 1, 2]
        restored = AuditLog()
        restored.restore(state.audit)
        assert restored.verify_chain()
    finally:
        store.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_transaction_atomicity_on_failure(tmp_path, backend):
    store = InMemoryStateStore() if backend == "memory" else _sqlite(tmp_path)
    try:
        with pytest.raises(RuntimeError):
            with store.transaction() as tx:
                tx.save_org({"org_id": "x", "name": "X", "domain": "d",
                             "sector_group": "g", "api_key_hash": "h", "key_algo": "a",
                             "active": True})
                tx.append_audit({"id": "1", "ts": 0.0, "action": "a", "details": {},
                                 "prev_hash": "p", "hash": "h"})
                raise RuntimeError("boom mid-bundle")
        assert store.load() is None  # no partial rows survived
    finally:
        store.close()


def test_concurrent_store_writes_serialized(tmp_path):
    store = _sqlite(tmp_path)
    errors: list[BaseException] = []
    try:
        def worker(tid: int):
            try:
                for i in range(5):
                    log = AuditLog()
                    with store.transaction() as tx:
                        tx.append_audit(log.record("op", tid=tid, i=i))
            except BaseException as exc:  # noqa: BLE001 - collected, asserted below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        state = store.load()
        assert state is not None and len(state.audit) == 40
    finally:
        store.close()


EXPECTED_COLUMNS = {
    "schema_meta": {"id", "version"},
    "orgs": {"org_id", "name", "domain", "sector_group", "api_key_hash", "key_algo", "key_salt", "active"},
    "policies": {"org_id", "allowed_metrics", "min_participants", "max_epsilon_per_query", "purpose"},
    "privacy_budget": {"id", "limit_epsilon", "limit_delta", "spent_epsilon", "spent_delta",
                       "rdp_totals", "rdp_complete", "events"},
    "federated_servers": {"key", "dim", "bias", "weights", "rounds"},
    "federated_rounds": {"id", "server_key", "round_id", "participants", "dropped",
                         "aggregate_delta", "weights", "eval_stats", "dp_applied",
                         "epsilon_used", "dp_sigma", "delta_used"},
    "audit_log": {"seq", "entry"},
}


def test_no_sensitive_data_persisted(tmp_path):
    from pathlib import Path

    db = tmp_path / "safe.db"
    store = SQLiteStateStore(str(db))
    try:
        reg = Registry()
        orgs = [reg.register(f"Org{i}", "retail", "g-safe") for i in range(2)]
        pol = ConsentPolicy.default()
        with store.transaction() as tx:
            for o in orgs:
                tx.save_org(codec.encode_org(o))
                tx.save_policy(o.org_id, codec.encode_policy(pol))
        store.close()
        raw = Path(str(db)).read_bytes()
        for o in orgs:
            assert o.api_key.encode() not in raw  # plaintext keys never touch disk
        insp = inspect(init_db(str(db)))
        for table, cols in EXPECTED_COLUMNS.items():
            names = {c["name"] for c in insp.get_columns(table)}
            assert names == cols
            for n in names:
                assert "raw" not in n and "private" not in n and n != "api_key"
    finally:
        store.close()
