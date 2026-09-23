"""Pairwise-mask secure aggregation (Bonawitz-style) for federated updates.

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
from .prg import int_masks


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
class ClientContext:
    org_id: str
    private_key: X25519PrivateKey
    public_raw: bytes


@dataclass
class Contribution:
    org_id: str
    masked_update: np.ndarray  # int64
    public_key: bytes


@dataclass
class SecureAggregator:
    """Server-side orchestrator for one secure-aggregation round."""

    round_id: str
    roster: list[str]
    dim: int
    _privates: dict[str, ClientContext] = field(default_factory=dict)
    _contributions: dict[str, Contribution] = field(default_factory=dict)
    _revealed_seeds: dict[tuple[str, str], bytes] = field(default_factory=dict)

    # ---- client side -------------------------------------------------
    def client_register(self, org_id: str) -> bytes:
        if org_id not in self.roster:
            raise PermissionError(f"{org_id} not in roster")
        priv = X25519PrivateKey.generate()
        pub = priv.public_key().public_bytes_raw()
        self._privates[org_id] = ClientContext(org_id, priv, pub)
        return pub

    def public_keys(self) -> dict[str, bytes]:
        return {oid: ctx.public_raw for oid, ctx in self._privates.items()}

    def client_mask_and_send(self, org_id: str, update: np.ndarray) -> Contribution:
        ctx = self._privates.get(org_id)
        if ctx is None:
            raise PermissionError(f"{org_id} has not registered a key")
        keys = self.public_keys()
        if len(keys) != len(self.roster):
            raise RuntimeError("waiting for all roster keys before masking")

        clipped = clip_l2(np.asarray(update, dtype=np.float64))
        q = quantize(clipped)
        if q.shape != (self.dim,):
            raise ValueError(f"update dim {q.shape} != ({self.dim},)")

        mask = np.zeros(self.dim, dtype=np.int64)
        for peer in self.roster:
            if peer == org_id:
                continue
            seed = _pairwise_seed(ctx.private_key, keys[peer], self.round_id, org_id, peer)
            stream = int_masks(seed, self.dim)
            if org_id < peer:
                mask = mask + stream
            else:
                mask = mask - stream
        contrib = Contribution(org_id, q + mask, ctx.public_raw)
        self._contributions[org_id] = contrib
        return contrib

    def client_reveal_seed(self, org_id: str, dropped_id: str) -> bytes:
        """Survivor releases its pairwise seed with a dropped peer (recovery)."""
        ctx = self._privates.get(org_id)
        if ctx is None:
            raise PermissionError(f"{org_id} has not registered")
        if dropped_id not in self.roster:
            raise ValueError("dropped id not in roster")
        seed = _pairwise_seed(
            ctx.private_key, self._privates[dropped_id].public_raw, self.round_id, org_id, dropped_id
        )
        self._revealed_seeds[tuple(sorted((org_id, dropped_id)))] = seed
        return seed

    # ---- server side -------------------------------------------------
    def finalize(self) -> np.ndarray:
        """Return the summed (dequantized) aggregate from received contributions."""
        if not self._contributions:
            raise RuntimeError("no contributions")
        received = set(self._contributions)
        dropped = [oid for oid in self.roster if oid not in received]

        residual = np.zeros(self.dim, dtype=np.int64)
        for survivor in received:
            for drop in dropped:
                key = tuple(sorted((survivor, drop)))
                seed = self._revealed_seeds.get(key)
                if seed is None:
                    raise RuntimeError(
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

    @staticmethod
    def new_round_id() -> str:
        return secrets.token_hex(8)
