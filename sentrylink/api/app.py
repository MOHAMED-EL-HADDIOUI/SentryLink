"""FastAPI application exposing the SentryLink platform."""

from __future__ import annotations

import secrets
import time

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..errors import (
    AuthenticationError,
    AuthorizationError,
    RateLimitedError,
    SentryLinkError,
)
from ..federated.client import FederatedClient
from ..observability import configure_logging, log_event
from ..platform import SentryLinkPlatform
from ..ratelimit import InMemoryTokenBucket, RateLimiter
from ..settings import get_settings
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

SETTINGS = get_settings()
configure_logging(SETTINGS.log_level)


def build_platform(settings=SETTINGS) -> SentryLinkPlatform:
    """Startup factory: SQLite backend when SENTRYLINK_DB is set, else memory.

    The SQLite store runs schema migrations at construction, so by the time
    the app serves traffic the database is initialized and migrated.
    Invalid settings fail fast here instead of degrading at runtime.
    """
    if settings.db:
        return SentryLinkPlatform(store=SQLiteStateStore(settings.db))
    return SentryLinkPlatform()


def build_rate_limiter(settings=SETTINGS) -> RateLimiter:
    limit, window = settings.rate_limit_parsed()
    return InMemoryTokenBucket(limit, window)


PLATFORM = build_platform()
RATE_LIMITER: RateLimiter = build_rate_limiter()


def get_platform() -> SentryLinkPlatform:
    return PLATFORM


def _request_id_from_headers(scope) -> str:
    headers = {
        k.decode("latin-1").lower(): v.decode("latin-1")
        for k, v in scope.get("headers", [])
    }
    rid = (headers.get("x-request-id") or "").strip()
    if rid and len(rid) <= 64 and all(ch.isalnum() or ch in "-_" for ch in rid):
        return rid
    return secrets.token_hex(8)


class RequestContextMiddleware:
    """Assign/echo X-Request-Id and log one fixed-schema line per request.

    Only method, path, status and duration are logged — never bodies,
    credentials, or results.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = _request_id_from_headers(scope)
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            state["request_id"] = request_id
        start = time.perf_counter()
        status: dict = {}

        async def _send(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                message.setdefault("headers", []).append(
                    (b"x-request-id", request_id.encode())
                )
            await send(message)

        try:
            await self.app(scope, receive, _send)
        finally:
            log_event(
                "http.request",
                request_id=request_id,
                method=scope.get("method", "?"),
                path=scope.get("path", "?"),
                status=status.get("code", 0),
                duration_ms=round((time.perf_counter() - start) * 1000, 3),
            )


app.add_middleware(RequestContextMiddleware)


def _request_id(request: Request) -> str | None:
    return getattr(getattr(request, "state", None), "request_id", None)


def _error_body(code: str, message: str, request: Request) -> dict:
    return {"error": {"code": code, "message": message,
                      "request_id": _request_id(request)}}


@app.exception_handler(SentryLinkError)
def _domain_error(request: Request, exc: SentryLinkError):
    return JSONResponse(
        status_code=exc.http_status,
        content=_error_body(exc.code, str(exc), request),
    )


_STATUS_CODES = {
    400: "INVALID_REQUEST",
    401: "BAD_CREDENTIAL",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    503: "SERVICE_UNAVAILABLE",
}


def _http_error_body(request: Request, status: int, detail: object) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=_error_body(
            _STATUS_CODES.get(status, "HTTP_ERROR"), str(detail), request),
    )


@app.exception_handler(HTTPException)
def _http_exception(request: Request, exc: HTTPException):
    return _http_error_body(request, exc.status_code, exc.detail)


@app.exception_handler(StarletteHTTPException)
def _starlette_http_exception(request: Request, exc: StarletteHTTPException):
    return _http_error_body(request, exc.status_code, exc.detail)


@app.exception_handler(RequestValidationError)
def _validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=_error_body("VALIDATION_ERROR", "malformed request", request),
    )


@app.exception_handler(Exception)
def _unhandled(request: Request, exc: Exception):
    log_event("http.internal_error", request_id=_request_id(request),
              error=type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content=_error_body("INTERNAL_ERROR", "internal error", request),
    )


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
        raise AuthenticationError("missing credentials")
    try:
        org = platform.registry.authenticate(org_id, api_key)
    except (KeyError, PermissionError) as exc:
        raise AuthenticationError("invalid credentials") from exc
    cohort_ids = {o.org_id for o in platform.registry.cohort(sector_group, domain)}
    if org.org_id not in cohort_ids:
        raise AuthorizationError("org is not a member of this cohort")
    return org


def _check_rate_limit(org_id: str) -> None:
    # Rate limiting is independent of privacy budgets: 429 means
    # "too many requests", never "budget exhausted".
    if not RATE_LIMITER.allow(f"spend:{org_id}"):
        raise RateLimitedError(f"rate limit exceeded for org {org_id}")


@app.exception_handler(RuntimeError)
def _storage_failure(_request, exc: RuntimeError):
    return JSONResponse(status_code=500, content={"detail": f"storage failure: {exc}"})


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Always-open. Reports build version, org count, audit validity and storage backend.",
)
def health() -> HealthResponse:
    p = get_platform()
    return HealthResponse(
        status="ok",
        version=__version__,
        orgs=len(p.registry.list()),
        audit_verified=p.audit.verify_chain(),
        storage=p.store.backend_name,
    )


@app.get(
    "/ready",
    summary="Readiness probe",
    description="200 when the store answers and the audit chain verifies, else 503. "
    "Error code READY_NOT_READY is internal-only; the body carries the checks.",
    responses={503: {"description": "Not ready"}},
)
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


@app.post(
    "/orgs",
    response_model=JoinResponse,
    summary="Onboard an organization",
    description="Registers an org and returns its API key exactly once. "
    "Enrollment is open; intelligence endpoints require the credential.",
    responses={400: {"description": "Invalid data"}, 422: {"description": "Malformed body"}},
)
def join_org(req: JoinRequest, request: Request) -> JoinResponse:
    p = get_platform()
    try:
        org = p.join(req.name, req.domain, req.sector_group,
                     request_id=_request_id(request))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JoinResponse(
        org_id=org.org_id,
        name=org.name,
        api_key=org.api_key,
        domain=org.domain,
        sector_group=org.sector_group,
    )


@app.get(
    "/orgs",
    summary="List organizations (public)",
    description="Public roster. Never includes API keys or hashes.",
)
def list_orgs() -> list[dict]:
    return [o.public() for o in get_platform().registry.list()]


@app.post(
    "/consent",
    summary="Set metric allow-list",
    description="Replaces only the allow-list; caps and floors are preserved. "
    "Requires a valid API key.",
    responses={401: {"description": "Bad credentials"}, 422: {"description": "Malformed body"}},
)
def update_consent(req: ConsentRequest, request: Request) -> dict:
    p = get_platform()
    try:
        p.registry.authenticate(req.org_id, req.api_key)
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    p.set_consent(req.org_id, set(req.allowed_metrics),
                  request_id=_request_id(request))
    return {"org_id": req.org_id, "allowed_metrics": sorted(req.allowed_metrics)}


@app.post(
    "/queries/histogram",
    response_model=QueryResultResponse,
    summary="MPC + DP histogram",
    description="Member-only. Spends epsilon (Laplace) and returns a Privacy Card. "
    "Auth is verified before any budget is spent.",
    responses={401: {"description": "Bad credentials"}, 403: {"description": "Not a member / governance / budget"},
               429: {"description": "Rate limited"}, 422: {"description": "Malformed body"}},
)
def query_histogram(req: HistogramQuery, request: Request) -> QueryResultResponse:
    p = get_platform()
    require_cohort_member(p, req.sector_group, req.domain, req.org_id, req.api_key)
    _check_rate_limit(req.org_id)
    try:
        res = p.histogram(
            req.sector_group,
            req.domain,
            req.org_buckets,
            labels=req.labels,
            epsilon=req.epsilon,
            delta=req.delta,
            purpose=req.purpose,
            request_id=_request_id(request),
        )
    except SentryLinkError:
        raise
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


@app.post(
    "/queries/variance",
    response_model=QueryResultResponse,
    summary="MPC + DP variance",
    description="Member-only. Spends epsilon (Laplace) and returns a Privacy Card. "
    "Denied for healthcare cohorts by policy.",
    responses={401: {"description": "Bad credentials"}, 403: {"description": "Not a member / governance / budget"},
               429: {"description": "Rate limited"}, 422: {"description": "Malformed body"}},
)
def query_variance(req: VarianceQuery, request: Request) -> QueryResultResponse:
    p = get_platform()
    require_cohort_member(p, req.sector_group, req.domain, req.org_id, req.api_key)
    _check_rate_limit(req.org_id)
    try:
        res = p.variance(
            req.sector_group,
            req.domain,
            req.org_values,
            epsilon=req.epsilon,
            delta=req.delta,
            purpose=req.purpose,
            request_id=_request_id(request),
        )
    except SentryLinkError:
        raise
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


@app.post(
    "/queries/correlation",
    response_model=QueryResultResponse,
    summary="MPC + DP correlation",
    description="Member-only. Spends epsilon (Laplace) and returns a Privacy Card.",
    responses={401: {"description": "Bad credentials"}, 403: {"description": "Not a member / governance / budget"},
               429: {"description": "Rate limited"}, 422: {"description": "Malformed body"}},
)
def query_correlation(req: CorrelationQuery, request: Request) -> QueryResultResponse:
    p = get_platform()
    require_cohort_member(p, req.sector_group, req.domain, req.org_id, req.api_key)
    _check_rate_limit(req.org_id)
    try:
        pairs = {k: (list(v[0]), list(v[1])) for k, v in req.org_pairs.items()}
        res = p.correlation(
            req.sector_group,
            req.domain,
            pairs,
            epsilon=req.epsilon,
            delta=req.delta,
            purpose=req.purpose,
            request_id=_request_id(request),
        )
    except SentryLinkError:
        raise
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


@app.post(
    "/queries/preview",
    summary="Privacy preview (no spend)",
    description="Member-only. Projects cost and admission for a query without "
    "spending budget, mutating state, or writing audit charges.",
    responses={401: {"description": "Bad credentials"}, 403: {"description": "Not a member"},
               422: {"description": "Malformed body"}},
)
def preview_query(req: PreviewRequest, request: Request) -> dict:
    """Privacy preview: projected cost/admission without spending anything."""
    p = get_platform()
    require_cohort_member(p, req.sector_group, req.domain, req.org_id, req.api_key)
    return p.preview(
        req.metric,
        req.sector_group,
        req.domain,
        epsilon=req.epsilon,
        delta=req.delta,
        request_id=_request_id(request),
    )


@app.post(
    "/federated/round",
    response_model=RoundResponse,
    summary="Secure-aggregated FL round",
    description="Member-only. Masked federated round with central DP, dropout "
    "recovery, release metadata and a Privacy Card.",
    responses={401: {"description": "Bad credentials"}, 403: {"description": "Not a member / governance / budget"},
               429: {"description": "Rate limited"}, 422: {"description": "Malformed body"}},
)
def federated_round(req: FederatedRoundRequest, request: Request) -> RoundResponse:
    p = get_platform()
    require_cohort_member(p, req.sector_group, req.domain, req.org_id, req.api_key)
    _check_rate_limit(req.org_id)
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
            request_id=_request_id(request),
        )
    except SentryLinkError:
        raise
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


@app.get(
    "/federated/model",
    summary="Global cohort model",
    description="Member-only (X-Org-Id/X-API-Key headers). Returns weights plus "
    "release metadata and a Privacy Card. 404 until a round has run.",
    responses={401: {"description": "Bad credentials"}, 403: {"description": "Not a member"},
               404: {"description": "No model yet"}},
)
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


@app.get(
    "/audit",
    summary="Hash-chained audit log",
    description="Always-open transparency log. Entries carry sanitized metadata "
    "only — no keys, rows, or unmasked values.",
)
def audit_log(limit: int = 50) -> dict:
    p = get_platform()
    entries = p.audit.entries
    return {
        "verified": p.audit.verify_chain(),
        "count": len(entries),
        "head": entries[-1]["hash"] if entries else "0" * 64,
        "entries": entries[-limit:],
    }


@app.get(
    "/audit/timeline",
    summary="Audit stage timeline",
    description="Developer-friendly governance → release → model_release stage view.",
)
def audit_timeline(limit: int = 50) -> dict:
    p = get_platform()
    return {
        "verified": p.audit.verify_chain(),
        "count": len(p.audit.entries),
        "timeline": p.audit.timeline(limit),
    }


@app.get(
    "/budgets",
    summary="Privacy budget ledger",
    description="Global spend vs remaining, RDP totals, and the per-event ledger "
    "(mechanism, sensitivity, cohort size — no private data).",
)
def budgets() -> dict:
    return get_platform().budget_report()


@app.get(
    "/transparency",
    summary="Sanitized observatory view",
    description="Machine-readable health, budget, audit and model-release status. "
    "No credentials, rows, or per-org private content.",
)
def transparency() -> dict:
    """Sanitized observatory view: health, budgets, audit, model releases."""
    from ..config import PROTOCOL_VERSION
    from ..storage import CURRENT_SCHEMA_VERSION

    p = get_platform()
    entries = p.audit.entries
    accountant = p.accountant
    assert accountant is not None
    return {
        "status": "ok",
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": CURRENT_SCHEMA_VERSION,
        "storage": p.store.backend_name,
        "orgs": len(p.registry.list()),
        "budget": {
            "spent_epsilon": accountant.spent.epsilon,
            "remaining_epsilon": accountant.remaining.epsilon,
            "rdp_epsilon_spent": accountant.rdp_epsilon(),
            "events": len(accountant.events),
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
