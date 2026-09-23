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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

from sqlalchemy.orm import Session, sessionmaker

from . import codec
from .engine import db_health, init_db
from .models import AuditRow, BudgetRow, OrgRow, PolicyRow, RoundRow, ServerRow


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
        return not self.orgs and self.budget is None and not self.audit


class StoreTx(ABC):
    """One atomic bundle of store writes (commit on clean exit)."""

    @abstractmethod
    def save_org(self, org: dict[str, Any]) -> None: ...

    @abstractmethod
    def save_policy(self, org_id: str, policy: dict[str, Any]) -> None: ...

    @abstractmethod
    def save_budget(self, budget: dict[str, Any]) -> None: ...

    @abstractmethod
    def save_server(self, server: dict[str, Any]) -> None: ...

    @abstractmethod
    def append_round(self, server_key: str, round: dict[str, Any]) -> None: ...

    @abstractmethod
    def append_audit(self, entry: dict[str, Any]) -> None: ...


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
    def __init__(self, state: PersistedState):
        self._ops: list[Any] = []
        self._state = state

    def save_org(self, org: dict[str, Any]) -> None:
        self._ops.append(("org", copy.deepcopy(org)))

    def save_policy(self, org_id: str, policy: dict[str, Any]) -> None:
        self._ops.append(("policy", org_id, copy.deepcopy(policy)))

    def save_budget(self, budget: dict[str, Any]) -> None:
        self._ops.append(("budget", copy.deepcopy(budget)))

    def save_server(self, server: dict[str, Any]) -> None:
        self._ops.append(("server", copy.deepcopy(server)))

    def append_round(self, server_key: str, round: dict[str, Any]) -> None:
        self._ops.append(("round", server_key, copy.deepcopy(round)))

    def append_audit(self, entry: dict[str, Any]) -> None:
        self._ops.append(("audit", codec.normalize_audit_entry(entry)))

    def __enter__(self) -> "_InMemoryTx":
        return self

    def __exit__(self, exc_type: Any, _e: Any, _t: Any) -> None:
        if exc_type is not None:
            self._ops.clear()  # discard the whole bundle
            return
        for op in self._ops:
            kind = op[0]
            if kind == "org":
                self._state.orgs = [o for o in self._state.orgs if o["org_id"] != op[1]["org_id"]] + [op[1]]
            elif kind == "policy":
                self._state.policies[op[1]] = op[2]
            elif kind == "budget":
                self._state.budget = op[1]
            elif kind == "server":
                self._state.servers[op[1]["key"]] = op[1]
            elif kind == "round":
                self._state.rounds.setdefault(op[1], []).append(op[2])
            elif kind == "audit":
                self._state.audit.append(op[1])
        self._ops.clear()


class InMemoryStateStore(StateStore):
    backend_name = "memory"

    def __init__(self) -> None:
        self._state = PersistedState()

    def load(self) -> PersistedState | None:
        if self._state.is_empty:
            return None
        return copy.deepcopy(self._state)

    def transaction(self) -> _InMemoryTx:
        return _InMemoryTx(self._state)

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

    def save_budget(self, budget: dict[str, Any]) -> None:
        self._s.merge(
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

    def save_server(self, server: dict[str, Any]) -> None:
        self._s.merge(
            ServerRow(
                key=server["key"],
                dim=int(server["dim"]),
                bias=float(server["bias"]),
                weights=bytes(server["weights"]),
                rounds=int(server["rounds"]),
            )
        )

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

    def append_audit(self, entry: dict[str, Any]) -> None:
        self._s.add(AuditRow(entry=codec.normalize_audit_entry(entry)))


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
