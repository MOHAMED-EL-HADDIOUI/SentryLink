"""FastAPI application exposing the SentryLink platform."""

from __future__ import annotations

import os

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from .. import __version__
from ..federated.client import FederatedClient
from ..platform import SentryLinkPlatform
from ..storage import SQLiteStateStore
from .schemas import (
    ConsentRequest,
    CorrelationQuery,
    FederatedRoundRequest,
    HealthResponse,
    HistogramQuery,
    JoinRequest,
    JoinResponse,
    PreviewRequest,
    QueryResultResponse,
    RoundResponse,
    VarianceQuery,
)

app = FastAPI(
    title="SentryLink",
    description="Privacy-preserving cross-organization intelligence layer",
    version=__version__,
)

def build_platform() -> SentryLinkPlatform:
    """Startup factory: SQLite backend when SENTRYLINK_DB is set, else memory.

    The SQLite store runs schema migrations at construction, so by the time
    the app serves traffic the database is initialized and migrated.
    """
    db = os.environ.get("SENTRYLINK_DB")
    if db:
        return SentryLinkPlatform(store=SQLiteStateStore(db))
    return SentryLinkPlatform()


PLATFORM = build_platform()


def get_platform() -> SentryLinkPlatform:
    return PLATFORM


def require_cohort_member(
    platform: SentryLinkPlatform,
    sector_group: str,
    domain: str,
    org_id: str | None,
    api_key: str | None,
):
    """Authenticate the caller and authorize it for this cohort.

    401 = missing/unknown credential or bad key; 403 = valid credential for
    an org outside the queried cohort. Runs before any budget is spent.
    """
    if not org_id or not api_key:
        raise HTTPException(status_code=401, detail="missing credentials")
    try:
        org = platform.registry.authenticate(org_id, api_key)
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=401, detail="invalid credentials") from exc
    cohort_ids = {o.org_id for o in platform.registry.cohort(sector_group, domain)}
    if org.org_id not in cohort_ids:
        raise HTTPException(status_code=403, detail="org is not a member of this cohort")
    return org


@app.exception_handler(RuntimeError)
def _storage_failure(_request, exc: RuntimeError):
    return JSONResponse(status_code=500, content={"detail": f"storage failure: {exc}"})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    p = get_platform()
    return HealthResponse(
        status="ok",
        version=__version__,
        orgs=len(p.registry.list()),
        audit_verified=p.audit.verify_chain(),
        storage=p.store.backend_name,
    )


@app.get("/ready")
def readiness():
    """Readiness probe: store reachable AND audit chain intact."""
    p = get_platform()
    store_ok = p.store.ping()
    chain_ok = p.audit.verify_chain()
    ready = bool(store_ok and chain_ok)
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "ready": ready,
            "storage": p.store.backend_name,
            "store_reachable": store_ok,
            "audit_verified": chain_ok,
        },
    )


@app.post("/orgs", response_model=JoinResponse)
def join_org(req: JoinRequest) -> JoinResponse:
    p = get_platform()
    try:
        org = p.join(req.name, req.domain, req.sector_group)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JoinResponse(
        org_id=org.org_id,
        name=org.name,
        api_key=org.api_key,
        domain=org.domain,
        sector_group=org.sector_group,
    )


@app.get("/orgs")
def list_orgs() -> list[dict]:
    return [o.public() for o in get_platform().registry.list()]


@app.post("/consent")
def update_consent(req: ConsentRequest) -> dict:
    p = get_platform()
    try:
        p.registry.authenticate(req.org_id, req.api_key)
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    p.set_consent(req.org_id, set(req.allowed_metrics))
    return {"org_id": req.org_id, "allowed_metrics": sorted(req.allowed_metrics)}


@app.post("/queries/histogram", response_model=QueryResultResponse)
def query_histogram(req: HistogramQuery) -> QueryResultResponse:
    p = get_platform()
    require_cohort_member(p, req.sector_group, req.domain, req.org_id, req.api_key)
    try:
        res = p.histogram(
            req.sector_group,
            req.domain,
            req.org_buckets,
            labels=req.labels,
            epsilon=req.epsilon,
            delta=req.delta,
            purpose=req.purpose,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return QueryResultResponse(
        query_id=res.query_id,
        metric=res.metric,
        value=res.value,
        participants=res.participants,
        epsilon=res.epsilon,
        delta=res.delta,
        noise_sigma=res.noise_sigma,
        purpose=res.purpose,
        privacy_card=res.privacy_card,
    )


@app.post("/queries/variance", response_model=QueryResultResponse)
def query_variance(req: VarianceQuery) -> QueryResultResponse:
    p = get_platform()
    require_cohort_member(p, req.sector_group, req.domain, req.org_id, req.api_key)
    try:
        res = p.variance(
            req.sector_group,
            req.domain,
            req.org_values,
            epsilon=req.epsilon,
            delta=req.delta,
            purpose=req.purpose,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return QueryResultResponse(
        query_id=res.query_id,
        metric=res.metric,
        value=res.value,
        participants=res.participants,
        epsilon=res.epsilon,
        delta=res.delta,
        noise_sigma=res.noise_sigma,
        purpose=res.purpose,
        privacy_card=res.privacy_card,
    )


@app.post("/queries/correlation", response_model=QueryResultResponse)
def query_correlation(req: CorrelationQuery) -> QueryResultResponse:
    p = get_platform()
    require_cohort_member(p, req.sector_group, req.domain, req.org_id, req.api_key)
    try:
        pairs = {k: (list(v[0]), list(v[1])) for k, v in req.org_pairs.items()}
        res = p.correlation(
            req.sector_group,
            req.domain,
            pairs,
            epsilon=req.epsilon,
            delta=req.delta,
            purpose=req.purpose,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return QueryResultResponse(
        query_id=res.query_id,
        metric=res.metric,
        value=res.value,
        participants=res.participants,
        epsilon=res.epsilon,
        delta=res.delta,
        noise_sigma=res.noise_sigma,
        purpose=res.purpose,
        privacy_card=res.privacy_card,
    )


@app.post("/queries/preview")
def preview_query(req: PreviewRequest) -> dict:
    """Privacy preview: projected cost/admission without spending anything."""
    p = get_platform()
    require_cohort_member(p, req.sector_group, req.domain, req.org_id, req.api_key)
    return p.preview(
        req.metric,
        req.sector_group,
        req.domain,
        epsilon=req.epsilon,
        delta=req.delta,
    )


@app.post("/federated/round", response_model=RoundResponse)
def federated_round(req: FederatedRoundRequest) -> RoundResponse:
    p = get_platform()
    require_cohort_member(p, req.sector_group, req.domain, req.org_id, req.api_key)
    clients = [
        FederatedClient(org_id=c.org_id, x=np.asarray(c.x, dtype=np.float64), y=np.asarray(c.y))
        for c in req.clients
    ]
    try:
        result = p.run_federated_round(
            req.sector_group,
            req.domain,
            {c.org_id: c for c in clients},
            epsilon=req.epsilon,
            delta=req.delta,
            epochs=req.epochs,
            drop=req.drop,
            purpose=req.purpose,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    key = f"{req.sector_group}:{req.domain}"
    return RoundResponse(
        round_id=result.round_id,
        participants=result.participants,
        dropped=result.dropped,
        dp_applied=result.dp_applied,
        epsilon_used=result.epsilon_used,
        eval_stats=result.eval_stats,
        weights=result.model.flat.tolist(),
        metadata=p.model_release_metadata(key),
        privacy_card=p.federated_release_card(key, req.domain),
    )


@app.get("/federated/model")
def get_model(
    sector_group: str,
    domain: str,
    x_org_id: str | None = Header(default=None, alias="X-Org-Id"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict:
    p = get_platform()
    require_cohort_member(p, sector_group, domain, x_org_id, x_api_key)
    key = f"{sector_group}:{domain}"
    if key not in p.servers:
        raise HTTPException(status_code=404, detail="no model for this cohort")
    server = p.servers[key]
    assert server.model is not None
    return {
        "key": key,
        "dim": server.dim,
        "weights": server.model.flat.tolist(),
        "rounds": len(server.history),
        "metadata": p.model_release_metadata(key),
        "privacy_card": p.federated_release_card(key, domain),
    }


@app.get("/audit")
def audit_log(limit: int = 50) -> dict:
    p = get_platform()
    entries = p.audit.entries
    return {
        "verified": p.audit.verify_chain(),
        "count": len(entries),
        "head": entries[-1]["hash"] if entries else "0" * 64,
        "entries": entries[-limit:],
    }


@app.get("/audit/timeline")
def audit_timeline(limit: int = 50) -> dict:
    p = get_platform()
    return {
        "verified": p.audit.verify_chain(),
        "count": len(p.audit.entries),
        "timeline": p.audit.timeline(limit),
    }


@app.get("/budgets")
def budgets() -> dict:
    return get_platform().budget_report()


@app.get("/transparency")
def transparency() -> dict:
    """Sanitized observatory view: health, budgets, audit, model releases."""
    from ..config import PROTOCOL_VERSION
    from ..storage import CURRENT_SCHEMA_VERSION

    p = get_platform()
    entries = p.audit.entries
    return {
        "status": "ok",
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": CURRENT_SCHEMA_VERSION,
        "storage": p.store.backend_name,
        "orgs": len(p.registry.list()),
        "budget": {
            "spent_epsilon": p.accountant.spent.epsilon,
            "remaining_epsilon": p.accountant.remaining.epsilon,
            "rdp_epsilon_spent": p.accountant.rdp_epsilon(),
            "events": len(p.accountant.events),
        },
        "audit": {
            "valid": p.audit.verify_chain(),
            "entries": len(entries),
            "head": entries[-1]["hash"] if entries else "0" * 64,
        },
        "models": {
            key: p.model_release_metadata(key) for key in sorted(p.servers)
        },
    }
