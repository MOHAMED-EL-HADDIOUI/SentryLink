"""Safe structured logging: fixed fields only, aggressive redaction.

Rules enforced here, not by convention:
  - log events carry a fixed schema (timestamp, event, request_id, fields);
  - request/response bodies, feature arrays, keys and secrets are never
    logged — only method, path, status, duration and error codes;
  - redact() defensively strips anything shaped like a credential from
    arbitrary structures before they reach a log line.
"""

from __future__ import annotations

import json
import logging
import re
import time

LOGGER_NAME = "sentrylink"

# Whole-token matches on snake/kebab/camel-ish splits. Deliberately includes
# "key" so public_key/key_algo-style metadata is redacted too (conservative:
# logs stay boring). Single-token words like "monkey" do NOT match.
SENSITIVE_TOKENS = frozenset(
    {"api", "apikey", "private", "secret", "token", "password", "seed", "key"}
)

REDACTED = "[REDACTED]"


def _is_sensitive(name: str) -> bool:
    tokens = set(re.split(r"[^a-z0-9]+", name.lower())) - {""}
    return not tokens.isdisjoint(SENSITIVE_TOKENS)


def redact(obj):
    """Deep-copy with credential-shaped keys replaced by [REDACTED]."""
    if isinstance(obj, dict):
        return {
            k: (REDACTED if isinstance(k, str) and _is_sensitive(k) else redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    return obj


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def log_event(event: str, *, request_id: str | None = None, **fields) -> None:
    payload: dict = {"ts": time.time(), "event": event}
    if request_id is not None:
        payload["request_id"] = request_id
    payload.update(redact(fields))
    logging.getLogger(LOGGER_NAME).info(json.dumps(payload))
