"""Application-level rate limiting for expensive privacy-consuming calls.

Deliberately separate from privacy-budget accounting: hitting the rate limit
means "too many requests" (429 RATE_LIMITED), never "budget exhausted".
Backends are injectable behind the RateLimiter protocol so a distributed
store can replace the in-memory bucket later without touching call sites.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod


class RateLimiter(ABC):
    @abstractmethod
    def allow(self, key: str) -> bool:
        """True when one call for key may proceed (consuming one token)."""


class InMemoryTokenBucket(RateLimiter):
    """Sliding-window limiter: at most `limit` calls per `window_seconds`."""

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        *,
        _clock=time.monotonic,
    ):
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.limit = limit
        self.window = float(window_seconds)
        self._clock = _clock
        self._lock = threading.Lock()
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            recent = [t for t in self._hits.get(key, []) if t > now - self.window]
            if len(recent) >= self.limit:
                self._hits[key] = recent
                return False
            recent.append(now)
            self._hits[key] = recent
            return True
