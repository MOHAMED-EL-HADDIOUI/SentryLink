"""Domain exceptions with stable machine-readable reason codes.

Design rules:
  - Exceptions raised from existing platform/governance/crypto paths subclass
    the builtin they replace (PermissionError / ValueError / RuntimeError),
    so every existing `pytest.raises(...)` keeps passing.
  - Every domain exception carries a stable `.code` from the documented set.
    The API boundary translates codes to HTTP statuses without leaking
    internals (unknown org and bad key both surface as BAD_CREDENTIAL).
  - Codes are part of the contract: do not rename them; only append.
"""

from __future__ import annotations


class SentryLinkError(Exception):
    """Base class. Carries a stable reason code plus an HTTP status hint."""

    code = "INTERNAL_ERROR"
    http_status = 500

    def __init__(self, message: str = "", *, code: str | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code


# ---- authentication / authorization (raised at the API boundary) ----

class AuthenticationError(SentryLinkError):
    code = "BAD_CREDENTIAL"
    http_status = 401


class AuthorizationError(SentryLinkError):
    code = "ORG_NOT_MEMBER"
    http_status = 403


class RateLimitedError(SentryLinkError):
    code = "RATE_LIMITED"
    http_status = 429


class ConcurrentWriteError(SentryLinkError):
    """Optimistic-concurrency conflict: committed state moved under us.

    Never silent: the caller reconverges from the store and the client may
    retry the (idempotent-cheap) operation. Maps to HTTP 409 Conflict.
    """

    code = "CONCURRENT_WRITE"
    http_status = 409


# ---- governance (raised from policy evaluation; still PermissionError) ----

class GovernanceError(PermissionError, SentryLinkError):
    code = "GOVERNANCE_DENIED"
    http_status = 403


class CohortTooSmallError(GovernanceError):
    code = "COHORT_TOO_SMALL"


class HealthcareCohortTooSmallError(GovernanceError):
    code = "HEALTHCARE_COHORT_TOO_SMALL"


class EpsilonTooHighError(GovernanceError):
    code = "EPSILON_TOO_HIGH"


class HealthcareEpsilonTooHighError(GovernanceError):
    code = "HEALTHCARE_EPSILON_TOO_HIGH"


class ConsentDeniedError(GovernanceError):
    code = "CONSENT_REQUIRED"


class QueryNotAllowedError(GovernanceError):
    code = "QUERY_NOT_ALLOWED"


class ScopeMismatchError(GovernanceError):
    code = "SECTOR_MISMATCH"


class DomainMismatchError(ScopeMismatchError):
    code = "DOMAIN_MISMATCH"


# ---- privacy budget (still PermissionError) ----

class PrivacyBudgetError(PermissionError, SentryLinkError):
    code = "BUDGET_EXHAUSTED"
    http_status = 403


# ---- invalid data / protocol (still ValueError / RuntimeError) ----

class InvalidDataError(ValueError, SentryLinkError):
    code = "INVALID_DATA"
    http_status = 400


class ModelDimensionMismatchError(InvalidDataError):
    code = "MODEL_DIMENSION_MISMATCH"


class DuplicateRosterError(InvalidDataError):
    code = "DUPLICATE_ROSTER"


class ProtocolError(RuntimeError, SentryLinkError):
    code = "PROTOCOL_ERROR"
    http_status = 400


# ---- storage / audit integrity (still RuntimeError) ----

class StorageError(RuntimeError, SentryLinkError):
    code = "STORAGE_ERROR"
    http_status = 500


class AuditIntegrityError(RuntimeError, SentryLinkError):
    code = "AUDIT_CORRUPT"
    http_status = 500
