"""Pairwise-mask secure aggregation (Bonawitz-style) for federated updates.

Trust boundaries are enforced by construction, not just documented:
  - SecAggClient holds its own X25519 private key. It never leaves the client.
  - SecAggServer sees only public keys, masked updates, and revealed
    dropout-recovery seeds. It holds no private key material.

Flow per round:
  1. Server announces the roster of participating organization ids.
  2. Each client derives an ephemeral X25519 keypair, exchanges public keys
     via the server, and computes a shared seed with every peer.
  3. For each unordered pair (i, j), i < j: client i adds PRG(seed_ij) and
     client j subtracts PRG(seed_ij) to/from its quantized update. All masks
     cancel in the sum, so the server only ever learns the aggregate.
  4. On dropout, survivors reveal seeds shared with dropped clients; the
     server subtracts the non-cancelling residual mask terms.

Transport confidentiality of public keys is assumed (TLS in deployment);
the server never learns pairwise seeds for surviving pairs.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from ..config import QUANT_SCALE, UPDATE_CLIP
from ..errors import InvalidDataError, ProtocolError
from .prg import int_masks


def new_round_id() -> str:
    return secrets.token_hex(8)


def quantize(vec: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(vec, dtype=np.float64) * QUANT_SCALE).astype(np.int64)


def dequantize(vec: np.ndarray) -> np.ndarray:
    return np.asarray(vec, dtype=np.float64) / QUANT_SCALE


def clip_l2(vec: np.ndarray, bound: float = UPDATE_CLIP) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm > bound and norm > 0:
        v = v * (bound / norm)
    return v


def _pairwise_seed(
    private: X25519PrivateKey, peer_public_raw: bytes, round_id: str, i: str, j: str
) -> bytes:
    shared = private.exchange(X25519PublicKey.from_public_bytes(peer_public_raw))
    a, b = sorted((i, j))
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=hashlib.sha256(round_id.encode()).digest(),
        info=f"sentrylink/agg/v1/{a}/{b}".encode(),
    ).derive(shared)


@dataclass
class Contribution:
    org_id: str
    masked_update: np.ndarray  # int64
    public_key: bytes


class SecAggClient:
    """One organization's side of a round. Owns its private key exclusively."""

    def __init__(self, org_id: str, round_id: str, roster: list[str], dim: int):
        if org_id not in roster:
            raise PermissionError(f"{org_id} not in roster")
        self.org_id = org_id
        self.round_id = round_id
        self.roster = list(roster)
        self.dim = dim
        self._private_key = X25519PrivateKey.generate()
        self.public_key: bytes = self._private_key.public_key().public_bytes_raw()

    def mask_and_send(
        self,
        update: np.ndarray,
        peer_keys: dict[str, bytes],
        *,
        clip_bound: float | None = UPDATE_CLIP,
    ) -> Contribution:
        """Quantize (optionally L2-clip), mask, and package our update.

        Pass clip_bound=None for already-bounded payloads (e.g. eval tallies)
        that must not be rescaled.
        """
        if set(peer_keys) != set(self.roster):
            raise ProtocolError("waiting for all roster keys before masking")
        v = np.asarray(update, dtype=np.float64)
        if not np.all(np.isfinite(v)):
            raise InvalidDataError("update must be finite (no NaN/inf)")
        if clip_bound is not None:
            v = clip_l2(v, clip_bound)
        q = quantize(v)
        if q.shape != (self.dim,):
            raise InvalidDataError(f"update dim {q.shape} != ({self.dim},)")

        mask = np.zeros(self.dim, dtype=np.int64)
        for peer in self.roster:
            if peer == self.org_id:
                continue
            seed = _pairwise_seed(
                self._private_key, peer_keys[peer], self.round_id, self.org_id, peer
            )
            stream = int_masks(seed, self.dim)
            if self.org_id < peer:
                mask = mask + stream
            else:
                mask = mask - stream
        return Contribution(self.org_id, q + mask, self.public_key)

    def reveal_seed(self, dropped_id: str, dropped_public_raw: bytes) -> bytes:
        """Release our pairwise seed with a dropped peer (dropout recovery)."""
        if dropped_id not in self.roster:
            raise ValueError("dropped id not in roster")
        if dropped_id == self.org_id:
            raise ValueError("cannot reveal a seed with ourselves")
        return _pairwise_seed(
            self._private_key, dropped_public_raw, self.round_id, self.org_id, dropped_id
        )


@dataclass
class SecAggServer:
    """Coordinator side of a round.

    Holds public keys, masked contributions, and revealed recovery seeds —
    never private key material.
    """

    round_id: str
    roster: list[str]
    dim: int
    _keys: dict[str, bytes] = field(default_factory=dict)
    _contributions: dict[str, Contribution] = field(default_factory=dict)
    _revealed_seeds: dict[tuple[str, str], bytes] = field(default_factory=dict)

    def register(self, org_id: str, public_key: bytes) -> None:
        if org_id not in self.roster:
            raise PermissionError(f"{org_id} not in roster")
        # Fail fast on malformed keys rather than at seed-derivation time.
        X25519PublicKey.from_public_bytes(bytes(public_key))
        self._keys[org_id] = bytes(public_key)

    def public_keys(self) -> dict[str, bytes]:
        return dict(self._keys)

    def receive(self, contrib: Contribution) -> None:
        if contrib.org_id not in self.roster:
            raise PermissionError(f"{contrib.org_id} not in roster")
        q = np.asarray(contrib.masked_update, dtype=np.int64)
        if q.shape != (self.dim,):
            raise ValueError(f"contribution dim {q.shape} != ({self.dim},)")
        self._contributions[contrib.org_id] = Contribution(
            contrib.org_id, q, bytes(contrib.public_key)
        )

    def add_recovery_seed(self, survivor: str, dropped: str, seed: bytes) -> None:
        if survivor not in self.roster or dropped not in self.roster:
            raise ValueError("survivor/dropped must be roster members")
        first, second = sorted((survivor, dropped))
        self._revealed_seeds[(first, second)] = bytes(seed)

    def finalize(self) -> np.ndarray:
        """Return the summed (dequantized) aggregate from received contributions."""
        if not self._contributions:
            raise ProtocolError("no contributions")
        received = set(self._contributions)
        dropped = [oid for oid in self.roster if oid not in received]

        residual = np.zeros(self.dim, dtype=np.int64)
        for survivor in received:
            for drop in dropped:
                first, second = sorted((survivor, drop))
                key = (first, second)
                seed = self._revealed_seeds.get(key)
                if seed is None:
                    raise ProtocolError(
                        f"missing recovery seed for pair {key}; cannot unmask dropout"
                    )
                stream = int_masks(seed, self.dim)
                if survivor < drop:
                    residual = residual + stream
                else:
                    residual = residual - stream

        total = np.zeros(self.dim, dtype=np.int64)
        for c in self._contributions.values():
            total = total + c.masked_update
        total = total - residual
        return dequantize(total)

    @property
    def participants(self) -> list[str]:
        return sorted(self._contributions)
