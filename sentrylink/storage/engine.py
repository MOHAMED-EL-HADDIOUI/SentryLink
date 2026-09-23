"""SQLite engine factory: thread-safe sessions, WAL concurrency, startup init."""

from __future__ import annotations

import os
from sqlalchemy import event
from sqlalchemy.engine import Engine, create_engine

from .migrations import get_schema_version, migrate


def normalize_url(path_or_url: str) -> str:
    """Accept a bare file path, :memory:, or a full sqlite:/// URL."""
    if path_or_url == ":memory:":
        return "sqlite:///:memory:"
    if "://" in path_or_url:
        if not path_or_url.startswith("sqlite:"):
            raise ValueError("only sqlite URLs are supported")
        return path_or_url
    return "sqlite:///" + os.path.abspath(path_or_url)


def create_engine_for(url: str) -> Engine:
    engine = create_engine(
        url,
        connect_args={"check_same_thread": False, "timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def _wal_mode(dbapi_conn, _record):  # pragma: no cover - driver hook
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    return engine


def init_db(path_or_url: str) -> Engine:
    """Create the engine and migrate the schema to the current version."""
    engine = create_engine_for(normalize_url(path_or_url))
    migrate(engine)
    return engine


def db_health(engine: Engine) -> bool:
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
