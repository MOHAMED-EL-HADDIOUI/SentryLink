"""Encode/decode between live domain objects and JSON-safe persisted forms.

This module is the only place that translates domain objects for storage, so
the privacy boundary is auditable in one file:

  - encode_org DROPS the plaintext API key and keeps only its salted hash.
  - Weight vectors are stored as raw float64 bytes (aggregate model state).
  - Audit entries pass through strict JSON: anything not JSON-serializable
    (e.g. a numpy value or key material smuggled into details) raises
    TypeError instead of being silently persisted.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..config import DEFAULT_DELTA
from ..crypto.differential_privacy import PrivacyAccountant, PrivacyBudget
from ..federated.model import LogisticModel
from ..federated.server import FederatedServer, RoundResult
from ..governance.policy import ConsentPolicy
from ..governance.registry import Organization, hash_api_key

_VECTOR_DTYPE = np.float64


def encode_vector(vec: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(vec, dtype=_VECTOR_DTYPE)).tobytes()


def decode_vector(blob: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(bytes(blob), dtype=_VECTOR_DTYPE).copy()
    if arr.shape != (dim,):
        raise ValueError(f"stored vector dim {arr.shape} != ({dim},)")
    return arr


def encode_org(org: Organization) -> dict[str, Any]:
    key_hash = org.api_key_hash
    salt = org.key_salt
    algo = ""
    if key_hash:
        algo = "pbkdf2-sha256"
    elif org.api_key:
        # Defensive path (register() always sets both): hash with the
        # per-org salt, falling back to the legacy org_id salt.
        salt = salt or org.org_id
        key_hash = hash_api_key(org.api_key, salt=salt)
        algo = "pbkdf2-sha256"
    return {
        "org_id": org.org_id,
        "name": org.name,
        "domain": org.domain,
        "sector_group": org.sector_group,
        "api_key_hash": key_hash,
        "key_algo": algo,
        "key_salt": salt,
        "active": org.active,
    }


def decode_org(data: dict[str, Any]) -> Organization:
    return Organization(
        org_id=data["org_id"],
        name=data["name"],
        domain=data["domain"],
        sector_group=data["sector_group"],
        api_key="",
        active=bool(data.get("active", True)),
        api_key_hash=data.get("api_key_hash", ""),
        key_salt=data.get("key_salt", ""),
    )


def encode_policy(policy: ConsentPolicy) -> dict[str, Any]:
    return {
        "allowed_metrics": sorted(policy.allowed_metrics),
        "min_participants": policy.min_participants,
        "max_epsilon_per_query": policy.max_epsilon_per_query,
        "purpose": policy.purpose,
    }


def decode_policy(data: dict[str, Any]) -> ConsentPolicy:
    return ConsentPolicy(
        allowed_metrics=frozenset(data["allowed_metrics"]),
        min_participants=int(data["min_participants"]),
        max_epsilon_per_query=float(data["max_epsilon_per_query"]),
        purpose=data.get("purpose", "cross-org intelligence"),
    )


def encode_budget(accountant: PrivacyAccountant) -> dict[str, Any]:
    def _budget(b: PrivacyBudget) -> dict[str, float]:
        return {"epsilon": b.epsilon, "delta": b.delta}

    return {
        "limit": _budget(accountant.limit),
        "spent": _budget(accountant.spent),
        "events": json.loads(json.dumps(accountant.events)),
        "rdp_totals": {str(k): float(v) for k, v in accountant.rdp_totals.items()},
        "rdp_complete": accountant.rdp_complete,
    }


def decode_budget(data: dict[str, Any]) -> PrivacyAccountant:
    return PrivacyAccountant(
        limit=PrivacyBudget(float(data["limit"]["epsilon"]), float(data["limit"]["delta"])),
        spent=PrivacyBudget(float(data["spent"]["epsilon"]), float(data["spent"]["delta"])),
        events=list(data.get("events", [])),
        rdp_totals={float(k): float(v) for k, v in data.get("rdp_totals", {}).items()},
        rdp_complete=bool(data.get("rdp_complete", True)),
    )


def encode_server(key: str, server: FederatedServer) -> dict[str, Any]:
    assert server.model is not None
    return {
        "key": key,
        "dim": server.dim,
        "bias": float(server.model.bias),
        "weights": encode_vector(server.model.flat),
        "rounds": len(server.history),
    }


def encode_round(server_key: str, result: RoundResult) -> dict[str, Any]:
    return {
        "server_key": server_key,
        "round_id": result.round_id,
        "participants": list(result.participants),
        "dropped": list(result.dropped),
        "aggregate_delta": encode_vector(result.aggregate_delta),
        "weights": encode_vector(result.model.flat),
        "eval_stats": json.loads(json.dumps(result.eval_stats)),
        "dp_applied": result.dp_applied,
        "epsilon_used": result.epsilon_used,
        "dp_sigma": result.dp_sigma,
        "delta_used": result.delta_used,
    }


def decode_server(
    key: str, raw_server: dict[str, Any], raw_rounds: list[dict[str, Any]]
) -> FederatedServer:
    dim = int(raw_server["dim"])
    server = FederatedServer(dim=dim)
    server.model = LogisticModel.from_flat(dim, decode_vector(raw_server["weights"], dim + 1))
    server.history = [decode_round(r, dim) for r in raw_rounds]
    assert len(server.history) == int(raw_server.get("rounds", len(server.history)))
    return server


def decode_round(data: dict[str, Any], dim: int) -> RoundResult:
    model = LogisticModel.from_flat(dim, decode_vector(data["weights"], dim + 1))
    return RoundResult(
        round_id=data["round_id"],
        participants=list(data["participants"]),
        dropped=list(data["dropped"]),
        aggregate_delta=decode_vector(data["aggregate_delta"], dim + 1),
        model=model,
        eval_stats=dict(data["eval_stats"]),
        dp_applied=bool(data["dp_applied"]),
        epsilon_used=float(data["epsilon_used"]),
        dp_sigma=float(data.get("dp_sigma", 0.0)),
        delta_used=float(data.get("delta_used", DEFAULT_DELTA)),
    )


def normalize_audit_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Strict JSON round-trip: copy detached from live objects, reject junk."""
    return json.loads(json.dumps(entry))
