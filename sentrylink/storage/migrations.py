"""Schema versioning for the SentryLink SQLite store.

Version history is a plain append-only sequence: MIGRATIONS maps each target
version to the function that upgrades the previous version to it. migrate()
reads the current version (0 when no schema_meta table exists) and applies
every pending step in order, stamping the version after each step.
"""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from .models import Base, SchemaMeta

CURRENT_SCHEMA_VERSION = 1


def _migrate_to_1(engine: Engine) -> None:
    Base.metadata.create_all(engine, checkfirst=True)


MIGRATIONS = {
    1: _migrate_to_1,
}


def get_schema_version(engine: Engine) -> int:
    if not inspect(engine).has_table(SchemaMeta.__tablename__):
        return 0
    with engine.connect() as conn:
        row = conn.execute(text("SELECT version FROM schema_meta WHERE id = 1")).first()
    return int(row[0]) if row else 0


def migrate(engine: Engine) -> int:
    """Apply pending migrations in order; return the resulting version."""
    version = get_schema_version(engine)
    while version < CURRENT_SCHEMA_VERSION:
        nxt = version + 1
        MIGRATIONS[nxt](engine)
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO schema_meta (id, version) VALUES (1, :v) "
                    "ON CONFLICT(id) DO UPDATE SET version = :v"
                ),
                {"v": nxt},
            )
        version = nxt
    return version
