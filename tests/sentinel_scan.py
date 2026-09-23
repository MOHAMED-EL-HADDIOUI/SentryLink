"""Reusable sensitive-value scanner for storage tests (not collected as tests).

Given named byte-sentinels that must never persist (API keys, raw feature
bytes, private key material, unmasked updates), scans a SQLite database both
as raw file bytes and cell-by-cell through SQL. Returns a list of findings;
an empty list means clean.

The scanner cannot prove a negative in general — it proves that the exact
sensitive values a test controls never reached storage. Detection is by
byte-substring, never by field name alone.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SECRET_SENTINEL = "SENTRYLINK_NEVER_PERSIST_THIS"


def _walk_cell(value, _depth=0):
    if _depth > 6:
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        yield bytes(value)
    elif isinstance(value, str):
        yield value.encode()
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return
        yield from _walk_cell(parsed, _depth + 1)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _walk_cell(v, _depth + 1)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk_cell(v, _depth + 1)


def scan_sqlite(db_path: str, sentinels: dict[str, list[bytes]]) -> list[dict]:
    """Scan file bytes and every cell for any sentinel value."""
    findings: list[dict] = []
    base = Path(db_path)
    blobs = []
    if base.exists():
        blobs.append(("main", base.read_bytes()))
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(base) + suffix)
        if sidecar.exists():
            blobs.append((suffix, sidecar.read_bytes()))
    for category, values in sentinels.items():
        for value in values:
            if not value:
                continue
            for location, blob in blobs:
                if value in blob:
                    findings.append(
                        {"location": f"file:{location}", "category": category,
                         "length": len(value)}
                    )
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        for table in tables:
            cursor = conn.execute(f'SELECT * FROM "{table}"')
            columns = [d[0] for d in cursor.description]
            for row_idx, row in enumerate(cursor):
                for col, cell in zip(columns, row):
                    for chunk in _walk_cell(cell):
                        for category, values in sentinels.items():
                            for value in values:
                                if value and value in chunk:
                                    findings.append(
                                        {"location": f"{table}[{row_idx}].{col}",
                                         "category": category,
                                         "length": len(value)}
                                    )
    finally:
        conn.close()
    return findings
