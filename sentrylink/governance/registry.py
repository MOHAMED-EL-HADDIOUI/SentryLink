"""Organization registry: identities, roles, and data-domain membership."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field

from ..config import MIN_PARTICIPANTS


@dataclass
class Organization:
    org_id: str
    name: str
    domain: str  # e.g. "retail", "manufacturing", "healthcare"
    sector_group: str  # supply-chain / consortium identifier
    api_key: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    active: bool = True

    def public(self) -> dict:
        return {
            "org_id": self.org_id,
            "name": self.name,
            "domain": self.domain,
            "sector_group": self.sector_group,
            "active": self.active,
        }


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
        self._orgs[org.org_id] = org
        return org

    def get(self, org_id: str) -> Organization:
        if org_id not in self._orgs:
            raise KeyError(f"unknown org {org_id}")
        return self._orgs[org_id]

    def authenticate(self, org_id: str, api_key: str) -> Organization:
        org = self.get(org_id)
        if org.api_key != api_key:
            raise PermissionError("invalid API key")
        return org

    def list(self) -> list[Organization]:
        return list(self._orgs.values())

    def cohort(self, sector_group: str, domain: str | None = None) -> list[Organization]:
        out = []
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
