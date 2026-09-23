# SENTRYLINK — MASTER IMPLEMENTATION PROMPT

You are the **lead software architect, security engineer, privacy engineer, ML engineer, backend engineer, and open-source maintainer** responsible for taking the existing **SentryLink** repository from its current implementation to a highly polished, security-conscious, production-oriented open-source project.

Repository:

`https://github.com/MOHAMED-EL-HADDIOUI/SentryLink`

Project identity:

> **SentryLink — Privacy-Preserving Cross-Organization Intelligence**

SentryLink allows organizations within the same vertical/domain to collaboratively compute useful intelligence without exposing raw organizational data. Organizations contribute only masked/aggregated information, and released statistics/models are protected with differential privacy.

The implementation must remain **honest, auditable, dependency-light, modular, deterministic where possible, and security-first**.

---

# 0. NON-NEGOTIABLE ENGINEERING PRINCIPLES

Before changing anything:

1. Inspect the existing repository completely.
2. Treat the existing implementation and tests as the source of truth.
3. Do not rewrite working architecture merely for stylistic reasons.
4. Preserve all currently passing behavior unless the specification explicitly requires a change.
5. Never weaken an existing privacy/security invariant to make a feature easier.
6. Never persist:
   - raw organization data
   - plaintext API keys
   - private encryption keys
   - unmasked federated updates
   - reconstructed sensitive statistics
   - arbitrary request payloads containing organizational data
7. Never log secrets.
8. Never claim cryptographic or differential-privacy guarantees that have not actually been mathematically or experimentally validated.
9. Every new feature must have tests.
10. Every security/privacy-sensitive behavior must have negative tests as well as success tests.
11. Prefer standard-library solutions and existing dependencies before introducing new dependencies.
12. Keep Python compatibility at **Python >= 3.11**.
13. Maintain clean typing and run `mypy`.
14. Maintain deterministic behavior wherever deterministic behavior is useful for testing/replay.
15. Avoid unnecessary abstraction layers.
16. Do not introduce distributed infrastructure that is not required by the current architecture.
17. Clearly distinguish:
   - privacy guarantee
   - cryptographic protection
   - governance policy
   - operational security
   - deployment security.

The result must be a real engineering project, not a conceptual prototype.

---

# 1. CURRENT SYSTEM CONTRACT

The current SentryLink architecture includes:

## Core identity

Privacy-preserving cross-organization intelligence layer.

Example verticals:

- retail
- manufacturing
- healthcare

Primary package:

`sentrylink/`

Primary orchestration class:

`sentrylink/platform.py::SentryLinkPlatform`

The orchestration layer currently owns:

- organization registry
- governance/policies
- federated learning
- secure aggregation
- MPC statistics
- differential privacy
- budget accounting
- audit chain
- persistence

`api/app.py` must remain a thin API adapter around the platform.

---

# 2. ARCHITECTURAL RULES

Preserve and strengthen this architecture:

```text
                   ┌─────────────────────────┐
                   │       FastAPI API        │
                   │   api/app.py             │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │    SentryLinkPlatform   │
                   │                         │
                   │ Registry                │
                   │ Governance              │
                   │ DP Accounting           │
                   │ Secure Aggregation      │
                   │ MPC                     │
                   │ Federated Learning      │
                   │ Audit                   │
                   │ Persistence             │
                   └────────────┬────────────┘
                                │
                  ┌─────────────┼─────────────┐
                  ▼             ▼             ▼
            Governance      Privacy       Computation
                  │             │             │
                  └─────────────┼─────────────┘
                                ▼
                     storage/codec.py
                                │
                                ▼
                          store.py
                                │
                     ┌──────────┴─────────┐
                     ▼                    ▼
               InMemoryStore        SQLiteStore
```

Important rule:

```text
domain objects
      ↓
storage/codec.py
      ↓
plain dict / bytes
      ↓
store.py
      ↓
SQL
```

`storage/codec.py` is the **sole translation/privacy boundary**.

No application/domain layer may directly construct persistence SQL payloads.

---

# 3. TRANSACTIONAL CONSISTENCY

`SentryLinkPlatform` currently serializes mutations with a single `RLock`.

Preserve the invariant:

> Memory state and persistent state must never silently diverge.

For every mutation:

```text
validate
   ↓
governance check
   ↓
privacy computation
   ↓
budget check
   ↓
prepare mutation
   ↓
atomic persistence
   ↓
commit in-memory state
   ↓
audit
```

If persistence fails:

```text
rollback/rebuild live state from store
```

Do not allow a partially committed in-memory state.

Add explicit tests for:

- failed transaction
- database exception
- rollback
- reconstruction
- restart
- concurrent mutations
- budget exhaustion during concurrency
- duplicate roster
- conflicting model dimensions
- audit chain consistency after failure.

---

# 4. EXISTING TECHNOLOGY CONTRACT

Preserve this dependency philosophy:

### Numerical computation

Use:

```text
NumPy
```

for:

- SGD
- vector operations
- quantization
- mask arithmetic
- noise generation
- sufficient statistics
- aggregation.

### Cryptography

Use:

```text
cryptography
```

for:

- X25519
- HKDF-SHA256.

Use the existing HMAC-SHA256 PRG for deterministic expansion of secure seeds.

### Persistence

Use:

```text
SQLAlchemy 2.x
SQLite
WAL mode
```

### API

Use:

```text
FastAPI
Pydantic
Uvicorn
```

### Testing

Use:

```text
pytest
mypy
```

Do not introduce a large framework ecosystem.

---

# 5. SECURITY MODEL

Threat model:

### Adversary A — Curious central server

The server must not receive:

- raw rows
- individual private datasets
- individual unmasked updates
- private X25519 keys
- plaintext API credentials.

The server sees only what the protocol requires.

### Adversary B — Curious MPC node

A single MPC node sees only its secret shares.

### Out of scope

Colluding MPC nodes in the current 2-node design.

Do not pretend the current architecture protects against collusion.

### Transport

TLS is deployment responsibility.

Do not present HTTP/TLS as a cryptographic feature implemented by SentryLink itself.

### Poisoning

Current protection is limited.

The current design includes:

- clipping
- DP noise.

Do not claim Byzantine robustness.

If adding poisoning defenses, ensure they do not silently break the masking model.

---

# 6. HARD PRIVACY INVARIANTS

Create explicit invariant checks and tests for all of the following.

## Organization isolation

Every operation must respect:

```text
sector_group
+
domain
```

Never aggregate across incompatible scopes.

## Consent

Only explicitly consented organizations can participate.

## Cohort minimum

General:

```text
k >= 3
```

Healthcare:

```text
k >= 4
```

## Healthcare constraints

Healthcare must enforce:

```text
k >= 4
epsilon <= 10
no variance query
```

These restrictions must survive consent edits.

Changing consent allow-lists must not accidentally reset privacy restrictions.

## Contribution bounds

Ensure FL contributions remain within the configured L2 contribution limit.

## Budget enforcement

A request is accepted only if the applicable privacy-accounting regime allows it.

The current accounting behavior supports:

```text
RDP accounting
OR
basic composition fallback
```

A charge is permitted if either valid accounting regime permits it.

Do not replace this logic with a simplistic epsilon counter.

---

# 7. CRYPTOGRAPHIC SECURE AGGREGATION

Preserve the split:

```text
SecAggClient
SecAggServer
```

The client owns the X25519 private key.

The server must NEVER contain or indirectly reference client private keys.

There is an existing test that walks the server object graph.

Keep it.

Strengthen it.

Add tests to ensure:

- no private key object
- no private key byte representation
- no secret seed retained accidentally
- no object reference path from server to private key
- no private key serialization
- no hidden key field added later.

Use:

```text
X25519
+
HKDF-SHA256
+
HMAC-SHA256 PRG
```

for pairwise masking.

Mask generation must use the existing signed/int64-compatible range:

```text
±2^40
```

Preserve:

```text
QUANT_SCALE = 1_000_000
UPDATE_CLIP = 1.0
```

Do not change these constants without a documented reason and updated tests.

---

# 8. FIXED-POINT SECURE AGGREGATION

Federated updates are quantized before masking.

Implement explicit stages:

```text
float vector
    ↓
L2 clipping
    ↓
fixed-point quantization
    ↓
int64 masking
    ↓
secure aggregation
    ↓
unmask/recover
    ↓
dequantization
```

Tests must verify:

- bounded quantization error
- deterministic behavior with deterministic seeds in tests
- overflow safety
- negative values
- zero vector
- sparse vectors
- large permitted vectors
- dropout recovery
- survivor seed reveal
- duplicate roster rejection
- missing participant rejection.

---

# 9. FEDERATED LEARNING

The model remains:

```text
L2-regularized logistic regression
```

Each organization trains locally.

Each organization returns:

```text
weight delta
```

The server performs FedAvg.

Model dimension must be inferred from the first client.

Released weights:

```text
bias + feature weights
```

so output dimension is:

```text
model_dim + 1
```

Do not accidentally interpret bias as another feature.

Implement strong validation around:

- feature dimensionality
- labels
- finite values
- NaN
- infinity
- empty datasets
- class validity
- clipping
- learning-rate bounds
- regularization bounds
- duplicate organizations.

---

# 10. DROPOUT RECOVERY

Preserve full-roster API semantics.

The caller supplies:

```text
full client dictionary
+
drop=[organization IDs]
```

Survivors reveal the required recovery seeds.

Test:

```text
N participants
N-1 survives
```

and:

```text
multiple dropped clients
```

Verify that:

- dropped updates do not contribute
- survivor masks cancel correctly
- server cannot reconstruct private information from a single survivor
- accounting remains correct
- audit metadata records dropout.

Add metrics:

```text
participants_total
participants_active
participants_dropped
recovery_used
```

These may be exposed as transparent metadata but must never expose sensitive organization-level content.

---

# 11. MASKED EVALUATION

Preserve pooled federated evaluation via a masked dimension-3 sub-round.

Important invariant:

```text
clip_bound = None
```

for evaluation tallies.

Evaluation tallies must remain exact/lossless before DP release.

Do not accidentally clip evaluation counts.

Expose transparent metadata:

```json
{
  "evaluation": {
    "mechanism": "masked_sum",
    "dp_applied": true,
    "clip_bound": null,
    "lossless_pre_dp": true
  }
}
```

---

# 12. MPC STATISTICS

Current design is intentionally:

```text
linear-only MPC
```

Organizations locally compute sufficient statistics.

Example:

```text
sum(x)
sum(x²)
sum(xy)
count
```

The MPC layer only performs share addition.

Do NOT add Beaver triples or multiplication machinery unless the entire protocol is redesigned and formally documented.

Preserve this principle:

> Non-linear work stays local. MPC performs only linear aggregation of already-reduced sufficient statistics.

Test:

- histogram
- variance
- correlation
- negative values
- zero values
- empty local contribution
- one malformed share
- wrong modulus
- share reconstruction
- incompatible domain.

Use:

```text
modulus = 2^127 - 1
```

unless a protocol review proves another choice is necessary.

---

# 13. DIFFERENTIAL PRIVACY

Preserve both:

## Laplace mechanism

For sensitivity `sens` and privacy budget `epsilon`:

```text
scale = sens / epsilon
```

Use for appropriate aggregate queries.

## Gaussian mechanism

For `(epsilon, delta)` DP.

Preserve the analytic Gaussian calibration.

Never replace it with a simplistic:

```text
sigma = 1 / epsilon
```

formula.

The exact implemented equation must be documented in code comments/docstrings.

---

# 14. EXISTING SENSITIVITY CONTRACT

Preserve:

```text
FL mean sensitivity = 2C / k
histogram sensitivity = MAX_ORG_CONTRIB
variance sensitivity = 1
correlation sensitivity <= 2
```

with correlation inputs clipped to:

```text
[-1, 1]
```

Existing:

```text
MAX_ORG_CONTRIB = 100.0
```

must remain enforced.

Test boundary conditions.

---

# 15. RDP ACCOUNTANT

Strengthen the RDP implementation.

For every supported event define:

```text
mechanism
orders
cost
epsilon
delta
metadata
```

Support composition by summing per-order RDP costs.

Convert the final RDP vector into an epsilon bound for requested delta.

If an event does not contain a valid RDP cost, use the basic-composition fallback.

The decision logic must remain:

```text
RDP valid and within budget
        OR
basic composition valid and within budget
```

Do not accidentally implement:

```text
RDP AND basic composition
```

which would unnecessarily reject valid operations.

Add exact tests for:

- single event
- two events
- multiple orders
- delta changes
- epsilon cap
- fallback event
- mixed RDP/non-RDP history
- budget exhaustion
- restart persistence
- deterministic replay.

---

# 16. PRIVACY BUDGET LEDGER — NEW FEATURE

Add a polished privacy-budget ledger.

Every privacy-consuming event should have an immutable logical record:

```text
event_id
organization scope
query/model type
mechanism
epsilon
delta
RDP orders
sensitivity
cohort size
clip bound
timestamp
request ID
budget before
budget after
decision
```

Do NOT store raw data.

Create a clean domain object such as:

```text
PrivacyLedgerEntry
```

and expose a sanitized representation.

This turns privacy spending into an auditable first-class concept.

The ledger must make it possible to answer:

> “Why did this request consume this much privacy budget?”

without requiring access to private organization data.

---

# 17. PRIVACY PREVIEW — NEW FEATURE

Add a **privacy preview** capability.

The client should be able to inspect what a query would consume before executing it.

Example conceptual response:

```json
{
  "allowed": true,
  "cohort_size": 5,
  "mechanism": "laplace",
  "epsilon_requested": 2.0,
  "delta": 0.0,
  "estimated_cost": 2.0,
  "remaining_budget_before": 11.0,
  "remaining_budget_after": 9.0,
  "governance": {
    "cohort_floor_satisfied": true,
    "sector_scope_valid": true,
    "consent_valid": true
  }
}
```

Important:

A preview must not spend budget.

A preview must not mutate persistent state.

A preview must not write to the audit chain as a privacy charge.

Do optionally record a non-charging observability event only if it cannot expose sensitive inputs.

---

# 18. PRIVACY TRANSPARENCY CARDS — NEW FEATURE

For every released statistic/model, generate a sanitized “Privacy Card”.

Example:

```text
SENTRYLINK PRIVACY CARD

Release: correlation
Domain: manufacturing
Cohort: 4 organizations

Mechanism:
  Gaussian DP

Privacy:
  epsilon = 2.4
  delta   = 1e-5

Accounting:
  RDP

Sensitivity:
  <= 2

Pre-processing:
  local clipping
  local sufficient statistics

Secure computation:
  2-node additive secret sharing

Raw data:
  never leaves organization

Persistent raw records:
  none

Audit:
  hash-chained

Security note:
  2-node collusion is out of scope
```

The card must be generated from actual runtime metadata, not hard-coded marketing text.

Do not display information that could identify private participating organizations unless policy explicitly permits it.

---

# 19. GOVERNANCE ENGINE

Preserve:

```text
PBKDF2 + random salt
```

for API-key hashes.

Plaintext API keys:

- shown exactly once
- memory only
- never logged
- never persisted.

Consent model:

```text
set_consent replaces only allow-list
```

It must NOT reset:

- epsilon cap
- cohort floor
- healthcare restrictions
- sector/domain boundaries.

Preserve:

```text
epsilon cap 25.0
healthcare epsilon cap 10.0
```

as configured policy values.

Make policy decisions deterministic and testable.

Prefer a dedicated object:

```text
PolicyDecision
```

containing:

```text
allowed
reason_code
human_reason
policy_snapshot
```

This improves API diagnostics without weakening security.

---

# 20. GOVERNANCE DECISION CODES — NEW FEATURE

Instead of returning only free-form error messages, introduce stable internal reason codes.

Examples:

```text
BAD_CREDENTIAL
ORG_NOT_MEMBER
CONSENT_REQUIRED
SECTOR_MISMATCH
DOMAIN_MISMATCH
COHORT_TOO_SMALL
HEALTHCARE_COHORT_TOO_SMALL
HEALTHCARE_EPSILON_TOO_HIGH
QUERY_NOT_ALLOWED
BUDGET_EXHAUSTED
MODEL_DIMENSION_MISMATCH
DUPLICATE_ROSTER
INVALID_DATA
```

API clients should be able to rely on these codes.

Human-readable messages should still be supplied.

Never reveal internal secrets or policy implementation details that help bypass controls.

---

# 21. AUDIT CHAIN

Preserve the append-only hash chain.

Each entry should conceptually contain:

```text
sequence
timestamp
event_type
sanitized metadata
previous_hash
current_hash
```

Implement chain verification.

Verification should detect:

- deletion
- modification
- reordering
- forged predecessor
- malformed entry.

Create a clean API response:

```json
{
  "valid": true,
  "entries": 128,
  "head": "..."
}
```

The audit endpoint must never return secret material.

---

# 22. AUDIT EXPLORER — NEW FEATURE

Add a developer-friendly audit representation.

For example:

```text
ROUND_STARTED
       ↓
POLICY_APPROVED
       ↓
SECURE_AGGREGATION
       ↓
DP_CALIBRATED
       ↓
BUDGET_CHARGED
       ↓
MODEL_RELEASED
```

Expose only sanitized information.

Never expose:

- API keys
- private keys
- raw vectors
- raw rows
- unmasked updates.

---

# 23. STORAGE

Current tables:

```text
schema_meta
orgs
policies
privacy_budget
federated_servers
federated_rounds
audit_log
```

Preserve these.

Use:

```text
SQLite
WAL
transaction boundaries
```

Strengthen migration support.

Migrations must be:

- deterministic
- versioned
- restart-safe
- testable.

Never use unrestricted serialization such as:

```text
pickle
```

for persistent domain state.

Use explicit codecs.

---

# 24. STORAGE SENSITIVE-DATA FIREWALL

Strengthen the existing storage sensitive-data tests.

Build a reusable sensitive-value scanner for tests.

It should inspect:

- SQL rows
- serialized values
- dictionaries
- bytes
- nested structures.

Detect obvious leakage of:

```text
api_key
private_key
secret
raw_data
feature matrix
training rows
unmasked updates
```

Do not rely solely on field names.

Where practical, compare against known test fixtures and sentinel secret values.

Example test:

```text
SECRET_SENTINEL = "SENTRYLINK_NEVER_PERSIST_THIS"
```

The secret must not appear anywhere in SQLite storage.

---

# 25. API CONTRACT

Preserve:

```text
POST /orgs
POST /consent

POST /queries/histogram
POST /queries/variance
POST /queries/correlation

POST /federated/round
GET  /federated/model

GET /orgs
GET /audit
GET /budgets
GET /health
GET /ready
```

Authentication:

Either:

```text
body:
{
  "org_id": "...",
  "api_key": "..."
}
```

or:

```text
X-Org-Id
X-API-Key
```

Authentication must be verified BEFORE budget spending.

---

# 26. HTTP ERROR SEMANTICS

Preserve:

```text
400 = invalid data/request semantics
401 = authentication failed
403 = authenticated but not authorized / governance / budget violation
404 = model/resource does not exist
422 = malformed request/schema
503 = readiness failure
```

Do not leak whether a secret credential exists beyond the intended authentication contract.

Create consistent structured errors:

```json
{
  "error": {
    "code": "BUDGET_EXHAUSTED",
    "message": "Privacy budget is insufficient for this operation",
    "request_id": "..."
  }
}
```

---

# 27. REQUEST IDs — NEW FEATURE

Introduce request IDs.

Every API request should have a request ID.

Behavior:

- accept incoming request ID when valid
- otherwise generate one
- return it in the response
- include it in safe audit/operational metadata
- never log secrets or raw data.

Do not make request IDs influence privacy accounting.

---

# 28. RATE LIMITING — SAFE NEXT STEP

Add a lightweight application-level rate limit concept for expensive privacy-consuming operations.

Do not introduce Redis or external infrastructure just for this.

For the current single-process architecture, an in-memory token bucket or sliding-window implementation is sufficient.

Rate limiting must be separate from privacy budget accounting.

The system must never confuse:

```text
too many requests
```

with:

```text
privacy budget exhausted
```

Expose distinct error codes.

Make the rate limiter injectable so a distributed backend can be introduced later.

---

# 29. VERTICAL PLUG-IN ARCHITECTURE

Improve the vertical-specific logic without making the core domain messy.

Create a lightweight abstraction such as:

```text
VerticalPolicy
```

or equivalent.

Implement built-in policies:

```text
RetailPolicy
ManufacturingPolicy
HealthcarePolicy
```

Each can define:

- cohort floor
- allowed statistics
- epsilon cap
- prohibited statistics
- contribution rules
- default demo datasets
- human-readable explanation.

Do not hard-code every rule throughout the API.

Core governance should consume policy objects.

---

# 30. VERTICAL DEMO DATASET SYSTEM

Turn the demo into reusable dataset generators.

Create:

```text
RetailDataset
ManufacturingDataset
HealthcareDataset
```

Each dataset generator should:

- use deterministic random seeds
- document feature meanings
- generate realistic but synthetic values
- never use real personal/medical records.

Example:

```text
retail
  demand
  promotion
  price
  store_traffic

manufacturing
  temperature
  pressure
  vibration
  defect_signal

healthcare
  treatment_signal
  response_score
  age_bucket
  outcome_proxy
```

For healthcare, explicitly keep the dataset synthetic and non-identifying.

---

# 31. DEMO EXPERIENCE

Upgrade:

```bash
python -m sentrylink.demo
```

into a compelling terminal demonstration.

The demo should tell the complete story:

```text
╔══════════════════════════════════════╗
║          SENTRYLINK DEMO             ║
║ Privacy-Preserving Intelligence      ║
╚══════════════════════════════════════╝
```

Then execute:

```text
1. Register organizations
2. Issue credentials
3. Configure governance
4. Establish consent
5. Generate synthetic local data
6. Run histogram
7. Run variance/correlation where allowed
8. Run federated training
9. Simulate dropout
10. Aggregate model
11. Apply DP
12. Charge privacy budget
13. Write audit event
14. Verify audit chain
15. Restart from SQLite
16. Authenticate using original API key
17. Verify model continuity
18. Show privacy ledger
19. Show Privacy Card
20. Show final budget
```

The demo should clearly show:

```text
RAW DATA LEFT ORGANIZATIONS: YES/NO
PRIVATE KEYS LEFT ORGANIZATIONS: NO
UNMASKED UPDATES AT SERVER: NO
DP APPLIED: YES
AUDIT CHAIN VALID: YES
```

Do not print raw confidential values.

---

# 32. “RED TEAM” DEMO MODE — CREATIVE FEATURE

Add a safe local testing mode:

```bash
python -m sentrylink.redteam
```

This is not an offensive-security tool.

It is a self-test suite demonstrating that SentryLink rejects or contains:

```text
bad credentials
non-member access
cross-domain query
insufficient cohort
healthcare policy violation
duplicate participant
NaN payload
infinite payload
model dimension mismatch
budget exhaustion
database write failure
server-side private-key retention
attempted raw-data persistence
audit tampering
dropout inconsistency
```

Output:

```text
[PASS] raw-data persistence blocked
[PASS] private-key retention blocked
[PASS] cross-domain query blocked
[PASS] healthcare k-floor enforced
[PASS] audit tampering detected
...
```

This gives the project a strong security/testing identity.

---

# 33. PRIVACY SIMULATOR — CREATIVE FEATURE

Add a local simulation utility.

Concept:

```bash
python -m sentrylink.simulate --vertical manufacturing
```

It should generate a report showing:

```text
cohort size
query
sensitivity
mechanism
epsilon
delta
remaining budget
accounting regime
dropout rate
quantization error
aggregation error
audit status
```

Use synthetic data only.

Add a machine-readable JSON mode:

```bash
python -m sentrylink.simulate ... --json
```

This is particularly useful for README examples and CI.

---

# 34. BENCHMARK SUITE

Add lightweight benchmarks.

Measure:

```text
registration latency
histogram latency
variance latency
correlation latency
FL aggregation latency
secure aggregation latency
MPC reconstruction latency
SQLite transaction latency
audit append latency
```

Do not make absolute latency claims in the README without running benchmarks.

Generate a reproducible benchmark command:

```bash
python -m sentrylink.bench
```

Output both human-readable and JSON.

---

# 35. QUALITY METRICS

Add internal metrics for:

```text
secure aggregation participants
dropout count
quantization error
DP epsilon
DP delta
budget remaining
round duration
storage duration
audit sequence
```

Metrics must be privacy-safe.

Never expose per-organization private information unless policy explicitly permits it.

---

# 36. MODEL RELEASE METADATA

Every federated model release should have a metadata envelope.

Example:

```json
{
  "model_version": 7,
  "algorithm": "logistic_regression",
  "aggregation": "FedAvg",
  "participants": 5,
  "dropouts": 1,
  "feature_dimension": 12,
  "dp": {
    "mechanism": "gaussian",
    "epsilon": 3.2,
    "delta": 1e-5,
    "accounting": "rdp"
  },
  "secure_aggregation": {
    "quantization_scale": 1000000,
    "update_clip": 1.0
  }
}
```

The metadata must be derived from the actual computation.

Never hard-code claimed privacy values.

---

# 37. VERSIONING

Introduce explicit versions where beneficial:

```text
protocol_version
schema_version
model_version
policy_version
audit_version
```

Model releases must be traceable to the protocol/policy configuration that produced them.

Do not break existing APIs unnecessarily.

---

# 38. RESTART AND REPLAY

Preserve the deterministic restart guarantee.

After restarting:

```text
organizations restored
budgets restored
models restored
round history restored
audit chain restored
credentials still authenticate
```

The system must behave consistently after restart.

Write explicit end-to-end tests:

```text
start
→ mutate
→ stop
→ restart
→ authenticate
→ query
→ inspect model
→ inspect budget
→ verify audit
```

---

# 39. CONCURRENCY

The existing system supports 8-thread concurrency testing.

Keep this behavior.

Create concurrency tests for:

```text
parallel registrations
parallel budget-consuming queries
parallel federated rounds
parallel read/write activity
audit append concurrency
SQLite persistence
```

Verify:

- no lost updates
- no double-spending
- no broken audit sequence
- no inconsistent models
- no SQLite corruption.

---

# 40. TEST TARGET

Current target:

```text
82 tests green
```

Do not reduce coverage to make development easier.

Grow the suite substantially where appropriate.

Aim for broad security coverage rather than an arbitrary number.

Recommended categories:

```text
API
API/storage integration
storage
recovery
DP
RDP accounting
federated learning
governance
MPC
secret sharing
secure aggregation
concurrency
vertical policies
privacy ledger
privacy preview
audit integrity
red-team scenarios
demo smoke tests
```

Every bug fixed must receive a regression test.

---

# 41. PROPERTY-BASED TESTING

Where practical, add property-style tests without adding a huge dependency.

Examples:

## Secret sharing

For valid values:

```text
reconstruct(split(x)) == x
```

## Mask cancellation

For compatible participants:

```text
sum(masked_updates)
→ valid aggregate after recovery
```

## Quantization

For values inside the clipping region:

```text
dequantize(quantize(x))
≈ x
```

within a documented tolerance.

## Audit

Appending N valid events must yield:

```text
verify_chain() == true
```

Tampering with any event must produce:

```text
verify_chain() == false
```

---

# 42. STATIC PRIVACY REVIEW

Add a CI check that searches source code and tests for dangerous persistence/logging patterns.

Examples:

```text
pickle
logging api_key
logging private_key
INSERT raw_rows
INSERT api_key plaintext
```

This should be conservative enough to avoid false positives becoming impossible to manage.

Document intentional exceptions.

---

# 43. DOCUMENTATION

Rewrite the README into a professional open-source landing page.

Suggested structure:

```text
SentryLink
Privacy-Preserving Cross-Organization Intelligence

[badges]

Why SentryLink?
Architecture
Threat Model
Privacy Model
Security Model
Supported Verticals
Quick Start
API
Federated Learning
Secure Aggregation
MPC
Differential Privacy
RDP Accounting
Governance
Privacy Budget
Auditability
Persistence
Demo
Red-Team Mode
Benchmarks
Testing
Limitations
Roadmap
Contributing
License
```

Do not use vague marketing claims such as:

```text
"military-grade"
"zero-risk"
"perfectly private"
"fully secure"
```

Prefer measurable statements.

---

# 44. ARCHITECTURE DIAGRAM

Create a Mermaid architecture diagram for the README.

Include:

```text
Organization A
Organization B
Organization C
        │
        ▼
 Local computation
        │
        ▼
 clipping / sufficient statistics
        │
        ▼
 secure aggregation / MPC
        │
        ▼
 policy + accounting
        │
        ▼
 differential privacy
        │
        ▼
 released aggregate/model
        │
        ▼
 audit + privacy ledger
```

Also create a Mermaid sequence diagram for:

```text
POST federated round
```

and:

```text
POST histogram
```

---

# 45. API DOCUMENTATION

Improve OpenAPI descriptions.

Every endpoint needs:

- summary
- description
- authentication requirement
- request schema
- response schema
- error codes
- security notes
- privacy/budget behavior.

Do not expose implementation internals unnecessarily.

---

# 46. EXAMPLES

Create executable examples such as:

```text
examples/register_org.py
examples/run_histogram.py
examples/run_federated_round.py
examples/restart_sqlite.py
examples/privacy_preview.py
```

Examples should be minimal and copy/paste friendly.

Never put a real API key in source control.

Use placeholders and generated demo credentials.

---

# 47. CONFIGURATION

Centralize safe configuration defaults.

Examples:

```text
SENTRYLINK_DB
SENTRYLINK_LOG_LEVEL
SENTRYLINK_RATE_LIMIT
SENTRYLINK_ENV
```

Validate configuration at startup.

Fail clearly on invalid security-sensitive configuration.

Avoid environment-dependent behavior that makes tests flaky.

---

# 48. OBSERVABILITY

Introduce safe structured logging.

Each log event may include:

```text
timestamp
request_id
event
duration_ms
result
policy code
privacy mechanism
```

Never include:

```text
api_key
private key
raw rows
feature arrays
unmasked updates
secrets
```

Consider a privacy-safe log redaction utility and test it.

---

# 49. FAILURE MODES

Every failure must be explicit.

Do not catch broad exceptions and silently continue.

Bad:

```python
try:
    ...
except Exception:
    pass
```

Use specific exception classes.

Introduce domain exceptions such as:

```text
AuthenticationError
AuthorizationError
GovernanceError
PrivacyBudgetError
CohortError
ProtocolError
StorageError
AuditIntegrityError
InvalidDataError
```

Translate these into API responses at the API boundary.

---

# 50. CODE ORGANIZATION

Prefer this conceptual structure:

```text
sentrylink/
    api/
        app.py
        schemas.py
        errors.py
    core/
        exceptions.py
        config.py
        request_context.py
    governance/
        registry.py
        consent.py
        policies.py
        decisions.py
    crypto/
        prg.py
        secure_aggregation.py
        differential_privacy.py
        rdp.py
    federated/
        client.py
        server.py
        model.py
        aggregation.py
    mpc/
        stats.py
        sharing.py
    privacy/
        ledger.py
        preview.py
        cards.py
    storage/
        codec.py
        store.py
        sqlite.py
        memory.py
        migrations/
    audit/
        chain.py
    verticals/
        base.py
        retail.py
        manufacturing.py
        healthcare.py
    demo.py
    redteam.py
    simulate.py
    bench.py
```

Do not create empty abstractions simply to match this structure.

Only introduce modules where they improve cohesion.

---

# 51. TYPE SAFETY

Run:

```bash
mypy sentrylink/storage
```

and then expand coverage to other modules where reasonable.

Avoid:

```python
Any
cast(...)
# type: ignore
```

unless justified.

Document unavoidable dynamic behavior.

---

# 52. LINT / STYLE

Follow idiomatic modern Python.

Prefer:

```text
dataclasses
typing
Protocol
Enum
frozen structures where useful
context managers
explicit exceptions
```

Avoid unnecessary metaprogramming.

Keep functions small enough to reason about.

Security-sensitive functions should be especially readable.

---

# 53. TEST EXECUTION

At the end of implementation, run:

```bash
pytest -q
```

Then:

```bash
mypy sentrylink/storage
```

Then:

```bash
python -m sentrylink.demo
```

Then:

```bash
python -m sentrylink.redteam
```

Then:

```bash
python -m sentrylink.simulate --vertical retail
python -m sentrylink.simulate --vertical manufacturing
python -m sentrylink.simulate --vertical healthcare
```

Then:

```bash
python -m sentrylink.bench
```

If any command fails, fix the root cause rather than weakening tests.

---

# 54. CI

Maintain Python matrix:

```text
3.11
3.12
```

Run:

```text
pytest
mypy
demo smoke test
storage restart test
```

Where practical also run:

```text
red-team smoke test
```

Keep CI dependency-light.

---

# 55. SECURITY SELF-CHECK

Before declaring completion, perform a manual security review covering:

## Credential security

- API keys hashed
- salt unique
- plaintext only shown once
- plaintext never persisted
- plaintext never logged

## Cryptography

- correct X25519 usage
- HKDF context separation
- secure random seeds
- correct mask cancellation
- no private-key retention
- no unsafe serialization

## DP

- correct sensitivity
- correct clipping
- correct mechanism
- correct sigma
- correct composition
- budget charged exactly once
- no budget charge after failed persistence

## Governance

- consent
- cohort
- vertical
- sector/domain
- healthcare restrictions
- epsilon caps

## Storage

- no sensitive fields
- migration correctness
- rollback correctness
- restart correctness

## API

- authentication
- authorization
- error handling
- request IDs
- input validation

## Audit

- append only
- hash chain
- verification
- failure behavior

---

# 56. DO NOT OVER-ENGINEER

Do NOT add:

```text
Kubernetes
Kafka
Redis
Postgres
Celery
RabbitMQ
service mesh
microservices
Terraform
cloud dependencies
```

to the local core implementation merely to make the project look enterprise-grade.

SentryLink's strength should be:

```text
small core
strong invariants
clear architecture
real privacy mechanisms
real cryptography
real tests
real persistence
real auditability
```

The project should remain easy to clone and run.

---

# 57. CREATIVE PRODUCT POSITIONING

Turn SentryLink into more than a collection of privacy primitives.

The product narrative should become:

```text
SentryLink
     │
     ├── Intelligence
     │     ├── Federated Learning
     │     ├── Histograms
     │     ├── Variance
     │     └── Correlation
     │
     ├── Privacy
     │     ├── Secure Aggregation
     │     ├── MPC
     │     ├── Differential Privacy
     │     └── RDP Accounting
     │
     ├── Governance
     │     ├── Consent
     │     ├── Cohort Policies
     │     ├── Vertical Policies
     │     └── Privacy Budgets
     │
     ├── Trust
     │     ├── Audit Chain
     │     ├── Privacy Ledger
     │     ├── Privacy Cards
     │     └── Restart Replay
     │
     └── Verification
           ├── Red Team
           ├── Simulation
           ├── Benchmarks
           └── Security Tests
```

This should make the project attractive to engineers interested in:

```text
AI
privacy engineering
cryptography
federated learning
data engineering
distributed systems
security
MLOps
governance
```

---

# 58. OPTIONAL “OBSERVATORY” MODE

If it can be implemented without materially increasing dependency complexity, add a small local observability interface.

Possible:

```text
GET /transparency
```

returning a sanitized machine-readable representation of:

```text
system health
registered org count
active policy versions
privacy budget status
audit chain status
latest model release
latest round
```

A lightweight browser UI may be added only if it remains simple.

Do not create a massive frontend application.

A polished local dashboard is desirable, but privacy/security correctness has priority over visual polish.

---

# 59. VERSIONED RELEASE REPORT

Add a command:

```bash
python -m sentrylink.report
```

that generates a sanitized report:

```text
SentryLink Release Report
-------------------------

Protocol version
Schema version
Python version

Tests
-----
passed
failed
skipped

Security
--------
private-key retention: PASS
sensitive-storage scan: PASS
audit verification: PASS

Privacy
-------
RDP accountant: PASS
budget enforcement: PASS
healthcare restrictions: PASS

Persistence
-----------
restart replay: PASS
atomicity: PASS

Demo
----
retail: PASS
manufacturing: PASS
healthcare: PASS
```

Also support:

```bash
python -m sentrylink.report --json
```

---

# 60. DOCUMENT LIMITATIONS EXPLICITLY

The README and documentation must clearly retain:

```text
single-process assumption
2-node collusion out of scope
transport TLS is deployment-side
poisoning defense is limited
SQLite is suitable for the current architecture but not a replacement
for a distributed coordination layer
```

Do not hide these limitations.

Also describe future work:

```text
multi-process deployment
stronger Byzantine/poisoning defenses
larger MPC topology
distributed storage
rate limiting backend
formal protocol verification
```

---

# 61. IMPLEMENTATION ORDER

Implement in this order:

## Phase 1 — Baseline

- inspect repository
- run all existing tests
- inspect architecture
- identify current gaps
- do not modify code yet unless required for build/test issues.

## Phase 2 — Core correctness

- exceptions
- policy decisions
- transaction consistency
- request IDs
- validation
- storage invariants.

## Phase 3 — Privacy

- privacy ledger
- privacy preview
- Privacy Cards
- RDP/accounting hardening.

## Phase 4 — Security

- secure aggregation review
- private-key isolation
- secret persistence scanner
- audit hardening
- red-team suite.

## Phase 5 — Product layer

- vertical policy objects
- deterministic demo datasets
- improved demo
- simulation
- benchmark
- release report.

## Phase 6 — API and documentation

- OpenAPI improvements
- examples
- README
- Mermaid diagrams
- limitations
- roadmap.

## Phase 7 — Final verification

Run:

```bash
pytest -q
mypy sentrylink/storage
python -m sentrylink.demo
python -m sentrylink.redteam
python -m sentrylink.simulate --vertical retail
python -m sentrylink.simulate --vertical manufacturing
python -m sentrylink.simulate --vertical healthcare
python -m sentrylink.bench
python -m sentrylink.report
```

---

# 62. GIT DISCIPLINE

Before editing:

```bash
git status
```

After each meaningful milestone:

```bash
git diff
git status
```

Do not destroy unrelated local changes.

Do not reset or checkout files containing user work unless explicitly necessary.

Keep commits logically organized when asked to commit.

Suggested commit categories:

```text
feat:
fix:
security:
privacy:
test:
docs:
refactor:
perf:
```

---

# 63. FINAL ACCEPTANCE CRITERIA

Do not declare the implementation complete until all of these are true:

### Correctness

- existing tests remain green
- new features have tests
- no silent state divergence
- restart works
- migrations work
- concurrency tests work.

### Privacy

- raw organizational rows never leave organizations in supported flows
- raw rows never persist
- unmasked updates never persist
- private keys never persist
- plaintext API keys never persist
- privacy budget is enforced
- healthcare restrictions are preserved
- privacy metadata reflects actual runtime behavior.

### Cryptography

- secure aggregation works
- dropout recovery works
- private-key isolation test passes
- secret-sharing tests pass
- mask cancellation tests pass.

### Governance

- consent works
- cohort floor works
- sector/domain isolation works
- epsilon caps work
- stable policy reason codes work.

### Storage

- memory backend works
- SQLite backend works
- WAL mode works
- migration works
- restart replay works
- storage sensitive-data scan passes.

### API

- auth matrix works
- HTTP semantics are correct
- structured errors work
- request IDs work
- readiness works.

### Product

- demo is compelling
- red-team mode works
- simulator works
- benchmark works
- privacy cards work
- privacy ledger works
- README explains the architecture clearly.

### Honesty

The documentation must clearly state what SentryLink does NOT protect against.

---

# 64. IMPORTANT AGENT BEHAVIOR

When you encounter a design ambiguity:

1. Prefer the existing architecture.
2. Prefer the stricter privacy interpretation.
3. Prefer explicit validation.
4. Prefer failing closed.
5. Prefer deterministic behavior.
6. Prefer minimal dependencies.
7. Prefer a regression test.
8. Never invent cryptographic guarantees.
9. Never silently weaken an invariant.
10. Never hide a limitation.

When implementing security or privacy-sensitive functionality, explain the invariant directly in code comments/docstrings.

Do not use comments like:

```text
"trust me"
"secure enough"
"probably safe"
```

Use precise engineering statements.

---

# 65. FINAL OUTPUT FROM THE CODING AGENT

At the end, provide a concise engineering report containing:

```text
1. What was implemented
2. Files added/changed
3. Security/privacy improvements
4. New APIs
5. New tests
6. Test results
7. Demo results
8. Benchmark results
9. Known limitations
10. Recommended next milestone
```

For each metric, report actual observed values.

Never fabricate test counts, benchmarks, security results, or privacy parameters.

The final implementation should make SentryLink feel like a serious open-source platform combining:

```text
Privacy Engineering
+
Applied Cryptography
+
Federated Learning
+
MPC
+
Differential Privacy
+
AI/Data Infrastructure
+
Governance
+
Auditable Software Engineering
```

The goal is not to make the repository larger.

The goal is to make the existing design **more coherent, safer, easier to understand, easier to demonstrate, and significantly more impressive to experienced engineers and technical reviewers while remaining honest about its guarantees and limitations.**