"""Organization registry: identities, roles, and data-domain membership."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from typing import List

from ..config import MIN_PARTICIPANTS

# API-key hash parameters. Only the salted hash is ever persisted; the
# plaintext key exists solely in process memory (returned once at join).
KEY_HASH_ALGO = "pbkdf2-sha256"
KEY_HASH_ITERATIONS = 100_000


def hash_api_key(api_key: str, org_id: str) -> str:
    """Salted API-key hash (salt = org_id). Stored; plaintext never is."""
    digest = hashlib.pbkdf2_hmac(
        "sha256", api_key.encode(), org_id.encode(), KEY_HASH_ITERATIONS
    ).hex()
    return f"{KEY_HASH_ALGO}${KEY_HASH_ITERATIONS}${digest}"


@dataclass
class Organization:
    org_id: str
    name: str
    domain: str  # e.g. "retail", "manufacturing", "healthcare"
    sector_group: str  # supply-chain / consortium identifier
    api_key: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    active: bool = True
    api_key_hash: str = ""

    def public(self) -> dict:
        return {
            "org_id": self.org_id,
            "name": self.name,
            "domain": self.domain,
            "sector_group": self.sector_group,
            "active": self.active,
        }

    def verify_key(self, api_key: str) -> bool:
        # Same-process path: plaintext key still resident in memory.
        if self.api_key and hmac.compare_digest(api_key, self.api_key):
            return True
        # Restored path: only the salted hash survived the restart.
        if self.api_key_hash and hmac.compare_digest(
            hash_api_key(api_key, self.org_id), self.api_key_hash
        ):
            return True
        return False


class Registry:
    def __init__(self):
        self._orgs: dict[str, Organization] = {}

    def register(
        self,
        name: str,
        domain: str,
        sector_group: str,
        org_id: str | None = None,
    ) -> Organization:
        org_id = org_id or secrets.token_hex(4)
        if org_id in self._orgs:
            raise ValueError(f"org {org_id} already registered")
        org = Organization(org_id=org_id, name=name, domain=domain, sector_group=sector_group)
        org.api_key_hash = hash_api_key(org.api_key, org.org_id)
        self._orgs[org.org_id] = org
        return org

    def add_restored(self, org: Organization) -> Organization:
        """Insert an organization rebuilt from storage (no key regeneration)."""
        if org.org_id in self._orgs:
            raise ValueError(f"org {org.org_id} already registered")
        self._orgs[org.org_id] = org
        return org

    def get(self, org_id: str) -> Organization:
        if org_id not in self._orgs:
            raise KeyError(f"unknown org {org_id}")
        return self._orgs[org_id]

    def authenticate(self, org_id: str, api_key: str) -> Organization:
        org = self.get(org_id)
        if not org.verify_key(api_key):
            raise PermissionError("invalid API key")
        return org

    def list(self) -> List[Organization]:
        # NOTE: List (not list) — the method name shadows the builtin here.
        return list(self._orgs.values())

    def cohort(self, sector_group: str, domain: str | None = None) -> List[Organization]:
        out: List[Organization] = []
        for org in self._orgs.values():
            if not org.active or org.sector_group != sector_group:
                continue
            if domain is not None and org.domain != domain:
                continue
            out.append(org)
        return out

    @staticmethod
    def min_cohort() -> int:
        return MIN_PARTICIPANTS
