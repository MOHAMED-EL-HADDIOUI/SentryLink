"""FastAPI application exposing the SentryLink platform."""

from __future__ import annotations

import os

import numpy as np
from fastapi import FastAPI, HTTPException
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
    )


@app.post("/queries/variance", response_model=QueryResultResponse)
def query_variance(req: VarianceQuery) -> QueryResultResponse:
    p = get_platform()
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
    )


@app.post("/queries/correlation", response_model=QueryResultResponse)
def query_correlation(req: CorrelationQuery) -> QueryResultResponse:
    p = get_platform()
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
    )


@app.post("/federated/round", response_model=RoundResponse)
def federated_round(req: FederatedRoundRequest) -> RoundResponse:
    p = get_platform()
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
    return RoundResponse(
        round_id=result.round_id,
        participants=result.participants,
        dropped=result.dropped,
        dp_applied=result.dp_applied,
        epsilon_used=result.epsilon_used,
        eval_stats=result.eval_stats,
        weights=result.model.flat.tolist(),
    )


@app.get("/federated/model")
def get_model(sector_group: str, domain: str) -> dict:
    p = get_platform()
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
    }


@app.get("/audit")
def audit_log(limit: int = 50) -> dict:
    p = get_platform()
    return {
        "verified": p.audit.verify_chain(),
        "count": len(p.audit.entries),
        "entries": p.audit.entries[-limit:],
    }


@app.get("/budgets")
def budgets() -> dict:
    return get_platform().budget_report()
