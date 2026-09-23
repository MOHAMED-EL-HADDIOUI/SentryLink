"""HMAC-SHA256 based pseudo-random generator for pairwise masks."""

from __future__ import annotations

import hashlib
import hmac

import numpy as np

from ..config import MASK_BOUND


def expand_seed(seed: bytes, n_bytes: int) -> bytes:
    """Expand a 32-byte seed into n_bytes of PRG output (counter-mode HMAC)."""
    if n_bytes < 0:
        raise ValueError("n_bytes must be >= 0")
    out = bytearray()
    counter = 0
    while len(out) < n_bytes:
        out += hmac.new(seed, counter.to_bytes(8, "big"), hashlib.sha256).digest()
        counter += 1
    return bytes(out[:n_bytes])


def int_masks(seed: bytes, length: int) -> np.ndarray:
    """Deterministic int64 mask vector in [-MASK_BOUND, MASK_BOUND]."""
    if length <= 0:
        return np.zeros(0, dtype=np.int64)
    raw = expand_seed(seed, length * 8)
    u = np.frombuffer(raw, dtype=">u8").astype(np.uint64)
    span = np.uint64(2 * MASK_BOUND + 1)
    centered = (u % span).astype(np.int64) - MASK_BOUND
    return centered.astype(np.int64)
