"""SQLite table definitions for SentryLink persistent state.

Privacy rule for this layer: tables hold governance artifacts (org identities,
consent allow-lists, budget counters, aggregate model weights, audit events).
They must NEVER hold raw organizational rows, per-row features, private
cryptographic keys, or unmasked client updates. API keys are stored only as
salted hashes (see governance.registry.hash_api_key).
"""

from __future__ import annotations

from sqlalchemy import JSON, LargeBinary, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SchemaMeta(Base):
    """Single-row schema version marker (id is always 1)."""

    __tablename__ = "schema_meta"

    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[int] = mapped_column(nullable=False)


class OrgRow(Base):
    __tablename__ = "orgs"

    org_id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(Text, nullable=False)
    sector_group: Mapped[str] = mapped_column(Text, nullable=False)
    api_key_hash: Mapped[str] = mapped_column(Text, nullable=False, default="")
    key_algo: Mapped[str] = mapped_column(Text, nullable=False, default="")
    active: Mapped[bool] = mapped_column(nullable=False, default=True)


class PolicyRow(Base):
    __tablename__ = "policies"

    org_id: Mapped[str] = mapped_column(Text, primary_key=True)
    allowed_metrics: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    min_participants: Mapped[int] = mapped_column(nullable=False)
    max_epsilon_per_query: Mapped[float] = mapped_column(nullable=False)
    purpose: Mapped[str] = mapped_column(Text, nullable=False)


class BudgetRow(Base):
    """Singleton row (id is always 1) with the privacy accountant snapshot."""

    __tablename__ = "privacy_budget"

    id: Mapped[int] = mapped_column(primary_key=True)
    limit_epsilon: Mapped[float] = mapped_column(nullable=False)
    limit_delta: Mapped[float] = mapped_column(nullable=False)
    spent_epsilon: Mapped[float] = mapped_column(nullable=False)
    spent_delta: Mapped[float] = mapped_column(nullable=False)
    rdp_totals: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    rdp_complete: Mapped[bool] = mapped_column(nullable=False)
    events: Mapped[list[dict[str, object]]] = mapped_column(JSON, nullable=False)


class ServerRow(Base):
    """Latest federated model per cohort key (sector_group:domain)."""

    __tablename__ = "federated_servers"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    dim: Mapped[int] = mapped_column(nullable=False)
    bias: Mapped[float] = mapped_column(nullable=False)
    weights: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    rounds: Mapped[int] = mapped_column(nullable=False, default=0)


class RoundRow(Base):
    """Append-only per-round records (model snapshots + eval stats)."""

    __tablename__ = "federated_rounds"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    server_key: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    round_id: Mapped[str] = mapped_column(Text, nullable=False)
    participants: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    dropped: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    aggregate_delta: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    weights: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    eval_stats: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    dp_applied: Mapped[bool] = mapped_column(nullable=False)
    epsilon_used: Mapped[float] = mapped_column(nullable=False)
    dp_sigma: Mapped[float] = mapped_column(nullable=False)


class AuditRow(Base):
    """Append-only hash-chained audit entries (full entry JSON per row)."""

    __tablename__ = "audit_log"

    seq: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    entry: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
