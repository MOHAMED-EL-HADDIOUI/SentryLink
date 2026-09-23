"""Repository-level persistence for SentryLink platform state.

Layering (nothing here imports domain logic — only plain dicts/bytes cross):

    domain objects --codec--> plain dicts --store--> SQLite / memory

Two backends implement StateStore:
  - InMemoryStateStore: no I/O. Default for unit tests; objects still pass
    through the codec so encode/decode paths stay identical to production.
  - SQLiteStateStore: SQLAlchemy 2.0 over a file (or :memory:) database with
    WAL concurrency, per-operation sessions, and one-transaction-per-mutation
    bundles via transaction().

Every platform mutation commits its store writes as a single bundle; if the
commit fails the platform rebuilds its live objects from the last committed
state (see SentryLinkPlatform._commit), so in-memory and stored state cannot
 silently diverge.
"""

from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterator

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from ..errors import ConcurrentWriteError
from . import codec
from .engine import db_health, init_db
from .models import AuditRow, BudgetRow, OrgRow, PolicyRow, RoundRow, ServerRow


def _merge_events(
    existing: list, incoming: list
) -> list:
    """Union event lists by id: concurrent committers never lose rows."""
    seen = {e.get("id") for e in existing}
    return list(existing) + [e for e in incoming if e.get("id") not in seen]


@dataclass
class PersistedState:
    orgs: list[dict[str, Any]] = field(default_factory=list)
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    budget: dict[str, Any] | None = None
    servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    rounds: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    audit: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return (
            not self.orgs
            and self.budget is None
            and not self.audit
            and not self.servers
            and not self.rounds
        )


class StoreTx(ABC):
    """One atomic bundle of store writes (commit on clean exit).

    Freshness proofs (`expect_*`) implement optimistic concurrency: a bundle
    computed from stale state fails loudly with ConcurrentWriteError instead
    of silently overwriting another writer's commit. Single-process callers
    always present matching proofs, so behavior there is unchanged.
    """

    @abstractmethod
    def save_org(self, org: dict[str, Any]) -> None: ...

    @abstractmethod
    def save_policy(self, org_id: str, policy: dict[str, Any]) -> None: ...

    @abstractmethod
    def save_budget(
        self, budget: dict[str, Any], *, expect_spent: tuple[float, float]
    ) -> None: ...

    @abstractmethod
    def save_server(
        self, server: dict[str, Any], *, expect_rounds: int
    ) -> None: ...

    @abstractmethod
    def append_round(self, server_key: str, round: dict[str, Any]) -> None: ...

    @abstractmethod
    def append_audit(self, entry: dict[str, Any], *, expect_prev: str) -> None: ...


class StateStore(ABC):
    backend_name: str = "abstract"

    @abstractmethod
    def load(self) -> PersistedState | None:
        """Full persisted state, or None when nothing was ever committed."""

    @abstractmethod
    def transaction(self) -> Any:
        """Context manager yielding a StoreTx committed atomically on exit."""

    @abstractmethod
    def ping(self) -> bool:
        """True when the backend is reachable."""

    def close(self) -> None:
        pass


class _InMemoryTx(StoreTx):
    def __init__(self, state: PersistedState, guard: Lock):
        self._ops: list[Any] = []
        self._state = state
        self._guard = guard

    def save_org(self, org: dict[str, Any]) -> None:
        self._ops.append(("org", copy.deepcopy(org)))

    def save_policy(self, org_id: str, policy: dict[str, Any]) -> None:
        self._ops.append(("policy", org_id, copy.deepcopy(policy)))

    def save_budget(
        self, budget: dict[str, Any], *, expect_spent: tuple[float, float]
    ) -> None:
        self._ops.append(("budget", copy.deepcopy(budget), tuple(expect_spent)))

    def save_server(
        self, server: dict[str, Any], *, expect_rounds: int
    ) -> None:
        self._ops.append(("server", copy.deepcopy(server), int(expect_rounds)))

    def append_round(self, server_key: str, round: dict[str, Any]) -> None:
        self._ops.append(("round", server_key, copy.deepcopy(round)))

    def append_audit(self, entry: dict[str, Any], *, expect_prev: str) -> None:
        self._ops.append(("audit", codec.normalize_audit_entry(entry), expect_prev))

    def __enter__(self) -> "_InMemoryTx":
        return self

    def __exit__(self, exc_type: Any, _e: Any, _t: Any) -> None:
        if exc_type is not None:
            self._ops.clear()  # discard the whole bundle
            return
        # One guard for check+apply makes concurrent bundles atomic here too.
        with self._guard:
            self._apply()

    def _apply(self) -> None:
        tip = self._state.audit[-1]["hash"] if self._state.audit else "0" * 64
        for op in self._ops:
            kind = op[0]
            if kind == "org":
                self._state.orgs = [o for o in self._state.orgs if o["org_id"] != op[1]["org_id"]] + [op[1]]
            elif kind == "policy":
                self._state.policies[op[1]] = op[2]
            elif kind == "budget":
                _, data, expect = op
                cur = self._state.budget
                if cur is not None and (
                    cur["spent"]["epsilon"], cur["spent"]["delta"]
                ) != (expect[0], expect[1]):
                    raise ConcurrentWriteError("budget changed since read")
                if cur is not None:
                    data = {**data, "events": _merge_events(
                        cur.get("events", []), data.get("events", []))}
                self._state.budget = data
            elif kind == "server":
                _, data, expect_rounds = op
                cur = self._state.servers.get(data["key"])
                if cur is not None and cur["rounds"] != expect_rounds:
                    raise ConcurrentWriteError("server model changed since read")
                self._state.servers[data["key"]] = data
            elif kind == "round":
                self._state.rounds.setdefault(op[1], []).append(op[2])
            elif kind == "audit":
                _, data, expect_prev = op
                if tip != expect_prev:
                    raise ConcurrentWriteError("audit tip moved since read")
                self._state.audit.append(data)
                tip = data["hash"]
        self._ops.clear()


class InMemoryStateStore(StateStore):
    backend_name = "memory"

    def __init__(self) -> None:
        self._state = PersistedState()
        self._guard = Lock()

    def load(self) -> PersistedState | None:
        with self._guard:
            if self._state.is_empty:
                return None
            return copy.deepcopy(self._state)

    def transaction(self) -> _InMemoryTx:
        return _InMemoryTx(self._state, self._guard)

    def ping(self) -> bool:
        return True


class _SQLiteTx(StoreTx):
    def __init__(self, session: Session):
        self._s = session

    def save_org(self, org: dict[str, Any]) -> None:
        self._s.merge(
            OrgRow(
                org_id=org["org_id"],
                name=org["name"],
                domain=org["domain"],
                sector_group=org["sector_group"],
                api_key_hash=org.get("api_key_hash", ""),
                key_algo=org.get("key_algo", ""),
                key_salt=org.get("key_salt", ""),
                active=bool(org.get("active", True)),
            )
        )

    def save_policy(self, org_id: str, policy: dict[str, Any]) -> None:
        self._s.merge(
            PolicyRow(
                org_id=org_id,
                allowed_metrics=list(policy["allowed_metrics"]),
                min_participants=int(policy["min_participants"]),
                max_epsilon_per_query=float(policy["max_epsilon_per_query"]),
                purpose=policy.get("purpose", ""),
            )
        )

    def save_budget(
        self, budget: dict[str, Any], *, expect_spent: tuple[float, float]
    ) -> None:
        row = self._s.get(BudgetRow, 1)
        if row is None:
            try:
                self._s.add(
                    BudgetRow(
                        id=1,
                        limit_epsilon=float(budget["limit"]["epsilon"]),
                        limit_delta=float(budget["limit"]["delta"]),
                        spent_epsilon=float(budget["spent"]["epsilon"]),
                        spent_delta=float(budget["spent"]["delta"]),
                        rdp_totals={str(k): float(v) for k, v in budget.get("rdp_totals", {}).items()},
                        rdp_complete=bool(budget.get("rdp_complete", True)),
                        events=list(budget.get("events", [])),
                    )
                )
                self._s.flush()
            except IntegrityError as exc:
                raise ConcurrentWriteError(
                    "budget row created concurrently") from exc
            return
        if (row.spent_epsilon, row.spent_delta) != (
            float(expect_spent[0]), float(expect_spent[1])):
            raise ConcurrentWriteError("budget changed since read")
        row.limit_epsilon = float(budget["limit"]["epsilon"])
        row.limit_delta = float(budget["limit"]["delta"])
        row.spent_epsilon = float(budget["spent"]["epsilon"])
        row.spent_delta = float(budget["spent"]["delta"])
        row.rdp_totals = {str(k): float(v) for k, v in budget.get("rdp_totals", {}).items()}
        row.rdp_complete = bool(budget.get("rdp_complete", True))
        row.events = _merge_events(list(row.events), list(budget.get("events", [])))

    def save_server(
        self, server: dict[str, Any], *, expect_rounds: int
    ) -> None:
        row = self._s.get(ServerRow, server["key"])
        if row is None:
            try:
                self._s.add(
                    ServerRow(
                        key=server["key"],
                        dim=int(server["dim"]),
                        bias=float(server["bias"]),
                        weights=bytes(server["weights"]),
                        rounds=int(server["rounds"]),
                    )
                )
                self._s.flush()
            except IntegrityError as exc:
                raise ConcurrentWriteError(
                    "server row created concurrently") from exc
            return
        if row.rounds != int(expect_rounds):
            raise ConcurrentWriteError("server model changed since read")
        row.dim = int(server["dim"])
        row.bias = float(server["bias"])
        row.weights = bytes(server["weights"])
        row.rounds = int(server["rounds"])

    def append_round(self, server_key: str, round: dict[str, Any]) -> None:
        self._s.add(
            RoundRow(
                server_key=server_key,
                round_id=round["round_id"],
                participants=list(round["participants"]),
                dropped=list(round["dropped"]),
                aggregate_delta=bytes(round["aggregate_delta"]),
                weights=bytes(round["weights"]),
                eval_stats=dict(round["eval_stats"]),
                dp_applied=bool(round["dp_applied"]),
                epsilon_used=float(round["epsilon_used"]),
                dp_sigma=float(round.get("dp_sigma", 0.0)),
                delta_used=float(round.get("delta_used", 1e-5)),
            )
        )

    def append_audit(self, entry: dict[str, Any], *, expect_prev: str) -> None:
        data = codec.normalize_audit_entry(entry)
        # Single-statement conditional INSERT: the tip check and the insert
        # are atomic, so concurrent bundles serialize here even though the
        # connection runs DEFERRED. Zero inserted rows = lost the race.
        # Every platform bundle ends with exactly one audit append, which
        # makes this gate serialize ALL bundles (rollback undoes the rest).
        result = self._s.execute(
            text(
                "INSERT INTO audit_log (entry) "
                "SELECT :entry WHERE COALESCE("
                "  (SELECT json_extract(entry, '$.hash') FROM audit_log "
                "   ORDER BY seq DESC LIMIT 1), "
                "  '0000000000000000000000000000000000000000000000000000000000000000'"
                ") = :expect_prev"
            ),
            {"entry": json.dumps(data), "expect_prev": expect_prev},
        )
        # NOTE: pysqlite populates rowcount for INSERT...SELECT; the stubs
        # do not model it, hence the ignore (runtime-proven by the
        # concurrent CAS-retry test, which only passes if 0-row races
        # are detected here).
        if result.rowcount != 1:  # type: ignore[attr-defined]
            raise ConcurrentWriteError("audit tip moved since read")


class _SQLiteTxContext:
    def __init__(self, factory: sessionmaker[Session]):
        self._factory = factory
        self._session: Session | None = None

    def __enter__(self) -> _SQLiteTx:
        self._session = self._factory()
        return _SQLiteTx(self._session)

    def __exit__(self, exc_type: Any, _e: Any, _t: Any) -> None:
        assert self._session is not None
        try:
            if exc_type is None:
                self._session.commit()
            else:
                self._session.rollback()
        finally:
            self._session.close()


class SQLiteStateStore(StateStore):
    backend_name = "sqlite"

    def __init__(self, path_or_url: str):
        from sqlalchemy.engine import Engine

        self.path_or_url = path_or_url
        self.engine: Engine = init_db(path_or_url)
        self._sessions = sessionmaker(bind=self.engine, expire_on_commit=False)

    def load(self) -> PersistedState | None:
        with self._sessions() as s:
            orgs = [
                {
                    "org_id": r.org_id,
                    "name": r.name,
                    "domain": r.domain,
                    "sector_group": r.sector_group,
                    "api_key_hash": r.api_key_hash,
                    "key_algo": r.key_algo,
                    "key_salt": r.key_salt,
                    "active": r.active,
                }
                for r in s.query(OrgRow).order_by(OrgRow.org_id).all()
            ]
            policies = {
                r.org_id: {
                    "allowed_metrics": list(r.allowed_metrics),
                    "min_participants": r.min_participants,
                    "max_epsilon_per_query": r.max_epsilon_per_query,
                    "purpose": r.purpose,
                }
                for r in s.query(PolicyRow).all()
            }
            brow = s.get(BudgetRow, 1)
            budget = (
                None
                if brow is None
                else {
                    "limit": {"epsilon": brow.limit_epsilon, "delta": brow.limit_delta},
                    "spent": {"epsilon": brow.spent_epsilon, "delta": brow.spent_delta},
                    "events": list(brow.events),
                    "rdp_totals": {str(k): float(v) for k, v in brow.rdp_totals.items()},
                    "rdp_complete": brow.rdp_complete,
                }
            )
            servers = {
                r.key: {
                    "key": r.key,
                    "dim": r.dim,
                    "bias": r.bias,
                    "weights": bytes(r.weights),
                    "rounds": r.rounds,
                }
                for r in s.query(ServerRow).all()
            }
            rounds: dict[str, list[dict[str, Any]]] = {}
            for r in s.query(RoundRow).order_by(RoundRow.id).all():
                rounds.setdefault(r.server_key, []).append(
                    {
                        "server_key": r.server_key,
                        "round_id": r.round_id,
                        "participants": list(r.participants),
                        "dropped": list(r.dropped),
                        "aggregate_delta": bytes(r.aggregate_delta),
                        "weights": bytes(r.weights),
                        "eval_stats": dict(r.eval_stats),
                        "dp_applied": r.dp_applied,
                        "epsilon_used": r.epsilon_used,
                        "dp_sigma": r.dp_sigma,
                        "delta_used": r.delta_used,
                    }
                )
            audit = [dict(r.entry) for r in s.query(AuditRow).order_by(AuditRow.seq).all()]
        state = PersistedState(
            orgs=orgs, policies=policies, budget=budget,
            servers=servers, rounds=rounds, audit=audit,
        )
        return None if state.is_empty else state

    def transaction(self) -> _SQLiteTxContext:
        return _SQLiteTxContext(self._sessions)

    def ping(self) -> bool:
        return db_health(self.engine)

    def close(self) -> None:
        self.engine.dispose()
