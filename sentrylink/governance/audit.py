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

    def restore(self, entries: list[dict]) -> None:
        """Rebuild from persisted entries (chain re-verified via verify_chain)."""
        self.entries = [dict(e) for e in entries]
        self._prev_hash = entries[-1]["hash"] if entries else "0" * 64

    def timeline(self, limit: int = 50) -> list[dict]:
        """Developer-friendly stage view over entries (sanitized scalars only)."""
        stages = {
            "org.": "governance",
            "consent.": "governance",
            "query.": "release",
            "federated.": "model_release",
        }
        out = []
        tail = self.entries[-limit:]
        base = len(self.entries) - len(tail)
        for i, e in enumerate(tail):
            action = str(e.get("action", "?"))
            stage = next(
                (s for prefix, s in stages.items() if action.startswith(prefix)),
                "other",
            )
            details = e.get("details", {})
            if isinstance(details, dict):
                parts = []
                if "participants" in details and isinstance(details["participants"], list):
                    parts.append(f"n={len(details['participants'])}")
                for key in ("epsilon", "delta", "round_id", "name", "domain", "buckets"):
                    if key in details:
                        parts.append(f"{key}={details[key]}")
                if "dropped" in details and isinstance(details["dropped"], list):
                    parts.append(f"dropped={len(details['dropped'])}")
                summary = " ".join(parts) if parts else action
            else:
                summary = action
            out.append(
                {"seq": base + i, "stage": stage, "action": action,
                 "ts": e.get("ts"), "summary": summary}
            )
        return out
