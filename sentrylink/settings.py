"""Centralized runtime configuration, read from the environment and validated.

Supported variables:

    SENTRYLINK_DB          SQLite path (or sqlite:/// URL); unset = in-memory
    SENTRYLINK_LOG_LEVEL   DEBUG | INFO | WARNING | ERROR (default INFO)
    SENTRYLINK_RATE_LIMIT  "<n>/<second|minute|hour>" (default "300/minute")
    SENTRYLINK_ENV         dev | test | prod (default dev)

Invalid security-sensitive configuration fails fast with ValueError at
startup instead of silently degrading. Tests pass explicit dicts, so no
environment manipulation can make them flaky.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})
VALID_ENVS = frozenset({"dev", "test", "prod"})
_PERIOD_SECONDS = {"second": 1.0, "minute": 60.0, "hour": 3600.0}


def parse_rate_limit(spec: str) -> tuple[int, float]:
    """Parse "<n>/<period>" into (max_calls, window_seconds)."""
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(second|minute|hour)\s*", spec or "")
    if not match:
        raise ValueError(
            f"invalid rate limit {spec!r}: expected '<n>/<second|minute|hour>'"
        )
    limit = int(match.group(1))
    if limit < 1:
        raise ValueError(f"invalid rate limit {spec!r}: n must be >= 1")
    return limit, _PERIOD_SECONDS[match.group(2)]


@dataclass(frozen=True)
class Settings:
    db: str | None = None
    log_level: str = "INFO"
    rate_limit: str = "300/minute"
    env: str = "dev"

    @classmethod
    def from_env(cls, env: dict | None = None) -> "Settings":
        src = env if env is not None else os.environ
        return cls(
            db=src.get("SENTRYLINK_DB") or None,
            log_level=(src.get("SENTRYLINK_LOG_LEVEL") or "INFO").upper(),
            rate_limit=src.get("SENTRYLINK_RATE_LIMIT") or "300/minute",
            env=(src.get("SENTRYLINK_ENV") or "dev").lower(),
        ).validated()

    def validated(self) -> "Settings":
        if self.log_level not in VALID_LOG_LEVELS:
            raise ValueError(
                f"invalid SENTRYLINK_LOG_LEVEL {self.log_level!r}: "
                f"expected one of {sorted(VALID_LOG_LEVELS)}"
            )
        if self.env not in VALID_ENVS:
            raise ValueError(
                f"invalid SENTRYLINK_ENV {self.env!r}: "
                f"expected one of {sorted(VALID_ENVS)}"
            )
        parse_rate_limit(self.rate_limit)  # validates format now, fails fast
        return self

    def rate_limit_parsed(self) -> tuple[int, float]:
        return parse_rate_limit(self.rate_limit)


def get_settings(env: dict | None = None) -> Settings:
    return Settings.from_env(env)
