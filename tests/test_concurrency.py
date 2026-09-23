"""Concurrency: parallel mutations must not lose updates or corrupt state."""

import multiprocessing as mp
import threading

import pytest

from sentrylink.crypto.differential_privacy import PrivacyAccountant, PrivacyBudget
from sentrylink.errors import ConcurrentWriteError
from sentrylink.federated.client import FederatedClient
from sentrylink.platform import SentryLinkPlatform
from sentrylink.storage import SQLiteStateStore
from sentrylink.usecases import make_retail_cohort


def _mp_histogram_worker(db_path, buckets, group, domain, epsilon, attempts, queue):
    """Separate OS process: charges budget against the shared SQLite file."""
    from sentrylink.errors import ConcurrentWriteError as _Conflict
    from sentrylink.platform import SentryLinkPlatform as _Platform
    from sentrylink.storage import SQLiteStateStore as _Store

    store = _Store(db_path)
    platform = _Platform(store=store)
    outcome = {"ok": 0, "denied": 0, "conflicted": 0}
    try:
        for _ in range(attempts):
            try:
                platform.histogram(group, domain, buckets, epsilon=epsilon)
                outcome["ok"] += 1
            except PermissionError:
                outcome["denied"] += 1
            except _Conflict:
                outcome["conflicted"] += 1
    except BaseException as exc:  # noqa: BLE001 - surfaced to the parent
        queue.put({"error": repr(exc)})
    else:
        queue.put(outcome)
    finally:
        store.close()


def _platform(path):
    return SentryLinkPlatform(store=SQLiteStateStore(str(path)))


def _run_threads(n, fn):
    errors: list[BaseException] = []

    def _wrap(*args):
        try:
            fn(*args)
        except BaseException as exc:  # noqa: BLE001 - collected below
            errors.append(exc)

    threads = [threading.Thread(target=_wrap, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_parallel_registrations(tmp_path):
    p = _platform(tmp_path / "c1.db")
    try:
        errors = _run_threads(8, lambda i: p.join(f"O{i}", "retail", "g-c"))
        assert not errors
        assert len(p.registry.list()) == 8
        assert len(p.audit.by_action("org.join")) == 8
    finally:
        p.store.close()


def test_parallel_budget_queries_no_lost_charges(tmp_path):
    p = _platform(tmp_path / "c2.db")
    try:
        ids = [p.join(f"O{i}", "retail", "g-c").org_id for i in range(3)]
        buckets = {oid: [4, 2] for oid in ids}
        errors = _run_threads(
            6, lambda i: p.histogram("g-c", "retail", buckets, epsilon=1.0))
        assert not errors
        assert p.budget_report()["spent_epsilon"] == pytest.approx(6.0)
        assert len(p.audit.by_action("query.histogram")) == 6
    finally:
        p.store.close()


def test_parallel_federated_rounds_consistent(tmp_path):
    orgs = make_retail_cohort(n_orgs=3, n_per_org=50, seed=3)
    p = _platform(tmp_path / "c3.db")
    try:
        ids = [p.join(o.name, "retail", "g-c").org_id for o in orgs]
        clients = {oid: FederatedClient(org_id=oid, x=o.x, y=o.y) for oid, o in zip(ids, orgs)}
        errors = _run_threads(
            3, lambda i: p.run_federated_round(
                "g-c", "retail", clients, epsilon=8.0, epochs=1))
        assert not errors
        server = p.servers["g-c:retail"]
        assert len(server.history) == 3
        assert p.audit.verify_chain()
    finally:
        p.store.close()


def test_budget_exhaustion_during_concurrency_no_double_spend(tmp_path):
    acc = PrivacyAccountant(limit=PrivacyBudget(4.0, 1e-3))
    p = SentryLinkPlatform(store=SQLiteStateStore(str(tmp_path / "c4.db")),
                           accountant=acc)
    try:
        ids = [p.join(f"O{i}", "retail", "g-c").org_id for i in range(3)]
        buckets = {oid: [4, 2] for oid in ids}
        outcomes: list[bool] = []
        lock = threading.Lock()

        def _try(i):
            try:
                p.histogram("g-c", "retail", buckets, epsilon=1.0)
                ok = True
            except PermissionError:
                ok = False
            with lock:
                outcomes.append(ok)

        errors = _run_threads(8, lambda i: (_try(i), _try(i)))
        assert not errors
        assert sum(outcomes) == 4  # exactly floor(4.0 / 1.0) succeed
        assert p.budget_report()["spent_epsilon"] == pytest.approx(4.0)
    finally:
        p.store.close()


def test_multiprocess_no_double_spend(tmp_path):
    # Four OS processes fight over the last 3.0 of a 4.0 budget (parent
    # spent 1.0 first). Any interleaving must yield exactly 3 successes,
    # spent == limit, and only budget/conflict failures — never lost charges.
    db = str(tmp_path / "mp.db")
    parent = SentryLinkPlatform(
        store=SQLiteStateStore(db),
        accountant=PrivacyAccountant(limit=PrivacyBudget(4.0, 1e-3)),
    )
    try:
        ids = [parent.join(f"O{i}", "retail", "g-mp").org_id for i in range(3)]
        buckets = {oid: [4, 2] for oid in ids}
        parent.histogram("g-mp", "retail", buckets, epsilon=1.0)
        parent.store.close()

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        procs = [
            ctx.Process(target=_mp_histogram_worker,
                        args=(db, buckets, "g-mp", "retail", 1.0, 3, queue))
            for _ in range(4)
        ]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(timeout=240)
        assert all(proc.exitcode == 0 for proc in procs)
        results = [queue.get(timeout=30) for _ in procs]
        assert not [r for r in results if "error" in r]
        assert sum(r["ok"] for r in results) == 3
        assert sum(r["ok"] + r["denied"] + r["conflicted"] for r in results) == 12

        check = SentryLinkPlatform(store=SQLiteStateStore(db))
        try:
            assert check.budget_report()["spent_epsilon"] == pytest.approx(4.0)
            assert len(check.audit.by_action("query.histogram")) == 4
            assert check.audit.verify_chain()
        finally:
            check.store.close()
    finally:
        try:
            parent.store.close()
        except Exception:  # noqa: BLE001 - already closed above in-flow
            pass


def test_parallel_read_write_activity(tmp_path):
    p = _platform(tmp_path / "c5.db")
    try:
        ids = [p.join(f"O{i}", "retail", "g-c").org_id for i in range(3)]
        buckets = {oid: [4, 2] for oid in ids}

        def _mixed(i):
            if i % 2:
                p.histogram("g-c", "retail", buckets, epsilon=1.0)
            else:
                assert p.budget_report()["spent_epsilon"] >= 0.0
                assert p.audit.verify_chain()

        errors = _run_threads(8, _mixed)
        assert not errors
        assert p.audit.verify_chain()
        # state survives a reload after concurrent mutation
        p.store.close()
        p2 = _platform(tmp_path / "c5.db")
        try:
            assert len(p2.registry.list()) == 3
            assert p2.audit.verify_chain()
        finally:
            p2.store.close()
    finally:
        try:
            p.store.close()
        except Exception:  # noqa: BLE001 - already closed above in-flow
            pass
