"""Pydantic request/response schemas for the SentryLink REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class JoinRequest(BaseModel):
    name: str
    domain: str = Field(description="retail | manufacturing | healthcare | ...")
    sector_group: str = Field(description="consortium / supply-chain identifier")


class JoinResponse(BaseModel):
    org_id: str
    name: str
    api_key: str
    domain: str
    sector_group: str


class ConsentRequest(BaseModel):
    org_id: str
    api_key: str
    allowed_metrics: list[str]


class HistogramQuery(BaseModel):
    sector_group: str
    domain: str
    org_buckets: dict[str, list[int]]
    labels: list[str] | None = None
    epsilon: float = 1.0
    delta: float = 1e-5
    purpose: str = "distribution insight"


class CorrelationQuery(BaseModel):
    sector_group: str
    domain: str
    org_pairs: dict[str, tuple[list[float], list[float]]]
    epsilon: float = 1.0
    delta: float = 1e-5
    purpose: str = "association insight"


class VarianceQuery(BaseModel):
    sector_group: str
    domain: str
    org_values: dict[str, list[float]]
    epsilon: float = 1.0
    delta: float = 1e-5
    purpose: str = "dispersion insight"


class FederatedClientData(BaseModel):
    org_id: str
    x: list[list[float]]
    y: list[float]


class FederatedRoundRequest(BaseModel):
    sector_group: str
    domain: str
    clients: list[FederatedClientData]
    epsilon: float = 1.0
    delta: float = 1e-5
    epochs: int = 2
    drop: list[str] = Field(default_factory=list)
    purpose: str = "federated model improvement"


class QueryResultResponse(BaseModel):
    query_id: str
    metric: str
    value: object
    participants: list[str]
    epsilon: float
    delta: float
    noise_sigma: float
    purpose: str


class RoundResponse(BaseModel):
    round_id: str
    participants: list[str]
    dropped: list[str]
    dp_applied: bool
    epsilon_used: float
    eval_stats: dict
    weights: list[float]


class HealthResponse(BaseModel):
    status: str
    version: str
    orgs: int
    audit_verified: bool
