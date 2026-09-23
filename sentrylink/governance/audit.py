"""Append-only, hash-chained audit log of every protected operation."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field


def _entry_hash(prev_hash: str, payload: dict) -> str:
    blob = json.dumps({"prev": prev_hash, "payload": payload}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class AuditLog:
    entries: list[dict] = field(default_factory=list)
    _prev_hash: str = "0" * 64

    def record(self, action: str, **details) -> dict:
        payload = {
            "id": secrets.token_hex(8),
            "ts": time.time(),
            "action": action,
            "details": details,
        }
        h = _entry_hash(self._prev_hash, payload)
        entry = {**payload, "prev_hash": self._prev_hash, "hash": h}
        self.entries.append(entry)
        self._prev_hash = h
        return entry

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for e in self.entries:
            payload = {k: e[k] for k in ("id", "ts", "action", "details")}
            expected = _entry_hash(prev, payload)
            if e["hash"] != expected or e["prev_hash"] != prev:
                return False
            prev = e["hash"]
        return True

    def by_action(self, action: str) -> list[dict]:
        return [e for e in self.entries if e["action"] == action]
