"""Additive (Shamir-free) secret sharing over the prime field FIELD_P.

Used by the MPC layer: each organization splits its local aggregate into
shares held by independent compute nodes; only the reconstructed aggregate
is ever opened (and only after policy checks + optional DP noise).
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

import numpy as np

from ..config import FIELD_P


@dataclass(frozen=True)
class Share:
    """One party's additive share of a vector (values mod FIELD_P)."""

    node_id: str
    values: tuple[int, ...]

    def __add__(self, other: "Share") -> "Share":
        if self.node_id != other.node_id or len(self.values) != len(other.values):
            raise ValueError("shares must belong to the same node and length")
        return Share(
            self.node_id,
            tuple((a + b) % FIELD_P for a, b in zip(self.values, other.values)),
        )

    def __sub__(self, other: "Share") -> "Share":
        if self.node_id != other.node_id or len(self.values) != len(other.values):
            raise ValueError("shares must belong to the same node and length")
        return Share(
            self.node_id,
            tuple((a - b) % FIELD_P for a, b in zip(self.values, other.values)),
        )

    def scale(self, k: int) -> "Share":
        return Share(self.node_id, tuple((k * v) % FIELD_P for v in self.values))


def _as_ints(values) -> list[int]:
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    flat = arr.astype(object).reshape(-1)
    ints = [int(x) for x in flat]
    for x in ints:
        if not (0 <= x < FIELD_P):
            raise ValueError(f"value {x} outside field [0, {FIELD_P})")
    return ints


def share_vector(values, node_ids: list[str]) -> dict[str, Share]:
    """Split a vector into |node_ids| additive shares that sum (mod p) to it."""
    if len(node_ids) < 2:
        raise ValueError("need at least 2 compute nodes")
    ints = _as_ints(values)
    n = len(node_ids)
    shares: dict[str, Share] = {}
    acc = [0] * len(ints)
    for node_id in node_ids[:-1]:
        vals = tuple(secrets.randbelow(FIELD_P) for _ in ints)
        shares[node_id] = Share(node_id, vals)
        for i, v in enumerate(vals):
            acc[i] = (acc[i] + v) % FIELD_P
    last = node_ids[-1]
    shares[last] = Share(last, tuple((ints[i] - acc[i]) % FIELD_P for i in range(len(ints))))
    return shares


def share_scalar(value: int, node_ids: list[str]) -> dict[str, Share]:
    return share_vector([int(value)], node_ids)


def reconstruct(shares: list[Share]) -> list[int]:
    if not shares:
        raise ValueError("no shares")
    length = len(shares[0].values)
    if any(len(s.values) != length for s in shares):
        raise ValueError("share length mismatch")
    return [sum(s.values[i] for s in shares) % FIELD_P for i in range(length)]
