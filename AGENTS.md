# AGENTS.md — SentryLink

## Setup & commands
- Requires Python `>=3.11`. Install: `pip install -r requirements.txt` (Windows venv: `.venv\Scripts\activate`).
- End-to-end check: `python -m sentrylink.demo` — 20-step narrative (retail + compact mfg/healthcare, SQLite restart inside).
- Full suite: `python -m pytest -q` (~149 tests, ~1 min, no external services).
- Single test: `python -m pytest tests/test_<name>.py -q` (e.g. `test_secure_aggregation.py`, `test_api.py`); focused: `python -m pytest tests/test_api.py -k <substring> -q`.
- Security self-check: `python -m sentrylink.redteam` (17 checks, must all print `[PASS]`).
- Simulator/bench/report: `python -m sentrylink.simulate --vertical <name> [--json]`, `python -m sentrylink.bench [--json]`, `python -m sentrylink.report [--json]`.
- Static privacy review: `python scripts/privacy_review.py` (must report clean).
- Examples: `python -m examples.<register_org|run_histogram|run_federated_round|restart_sqlite|privacy_preview>` from repo root.
- API dev server: `uvicorn sentrylink.api.app:app --reload` → docs at `/docs`.
- CI (`.github/workflows/ci.yml`) runs pytest + demo + redteam + all simulates + bench + report + privacy review + examples + repo-wide mypy on push/PR (matrix 3.11/3.12).

## Architecture (entrypoints)
- `sentrylink/platform.py::SentryLinkPlatform` is the single orchestration entry — policy + MPC + DP + audit. `sentrylink/api/app.py` is a thin wrapper over it; add new queries there by delegating to a `platform.py` method.
- `sentrylink/config.py` holds coupled crypto/DP constants — `FIELD_P = 2**127-1`, `QUANT_SCALE = 1e6`, `MASK_BOUND = 2**40` (int64-safe), `UPDATE_CLIP = 1.0`, `MAX_ORG_CONTRIB = 100.0`. Changing any one breaks masking/DP sensitivity assumptions elsewhere.
- Privacy split: **Laplace (pure ε-DP)** for `histogram`/`variance`/`correlation` (accountant charged with `delta=0`); **Gaussian ((ε,δ)-DP)** only for federated rounds. Keep this split when adding releases.
- Accounting is RDP moments-based: every `accountant.charge` site must pass an `rdp=` cost (`laplace_rdp_cost` / `gaussian_rdp_cost`), or enforcement silently falls back to basic composition from the first cost-less event on (`rdp_complete=False`). While fully RDP-tracked, a charge is allowed when **either** the basic sum or the RDP-converted ε fits the limit (each is independently valid — RDP wins on many-Gaussian workloads, basic on few-Laplace ones). FL rounds read the applied noise from `RoundResult.dp_sigma` — keep that field truthful.
- MPC is 2 non-colluding nodes (`NODE_IDS` in `platform.py`); collusion is explicitly out of scope. The protocol is deliberately linear-only (orgs pre-reduce to sufficient statistics) — there is no triple machinery; don't reintroduce products without a no-leak design.
- `api/app.py::PLATFORM` is a module-level singleton built by `build_platform()` from settings (`SENTRYLINK_DB` selects SQLite, else memory). With SQLite, state replays on restart; `GET /federated/model` 404s until a round has run for that `sector_group:domain` key.

## Governance gotchas (cause most test/API failures)
- Cohort floor is **≥3 consented orgs** (`MIN_PARTICIPANTS`), raised to the strictest `min_participants` among contributing orgs — healthcare enforces **≥4** via `default_policy_for`. Healthcare's other enforced differences: allow-list `{histogram, correlation, federated_model_round}` (`variance` denied) and `max_epsilon_per_query=10.0` (25.0 elsewhere).
- Per-query caps: `epsilon > 0`, `delta in (0, 1)`, `epsilon <= max_epsilon_per_query`. Denials carry stable codes on `QueryDecision.code` (`COHORT_TOO_SMALL`, `HEALTHCARE_*`, `EPSILON_TOO_HIGH`, `CONSENT_REQUIRED`, `SECTOR/DOMAIN_MISMATCH`, `QUERY_NOT_ALLOWED`, `BUDGET_EXHAUSTED`, …); `raise_if_denied` raises coded subclasses of the original builtins, so old `pytest.raises(PermissionError/ValueError)` still passes. API maps codes to structured bodies `{"error": {"code", "message", "request_id"}}`: 400 invalid data · 401 bad credential · 403 non-member/governance/budget · 404 no model · 409 concurrent-write conflict (retry) · 422 malformed · 429 rate-limited (never budget) · 500 internal · 503 not ready. Never invent new codes without adding them to `errors.py` + a negative test.
- Auth rule: every budget-spending endpoint (`/queries/*`, `/federated/round`, `/federated/model`) requires a cohort-member credential — `org_id`+`api_key` body fields on POSTs (via `schemas.MemberAuth`), `X-Org-Id`/`X-API-Key` headers on `GET /federated/model` (keys never go in URLs). Verified by `require_cohort_member` **before** any budget is spent. Transparency endpoints (`/orgs`, `/audit`, `/budgets`, `/health`, `/ready`) and enrollment stay open.
- Cohort is scoped by **both** `sector_group` and `domain` (`Registry.cohort`); joining with a mismatched string yields zero candidates → `"cohort too small"` denial.
- `platform.set_consent` replaces only the metric allow-list (via `dataclasses.replace`) — other terms like healthcare's 10.0 ε cap are preserved.
- Every consented org must supply input (`platform.py` raises `"all consented orgs must supply..."` otherwise). Pass exactly `decision.participants` worth of data.
- Per-org histogram vectors are L2-projected onto the `MAX_ORG_CONTRIB` ball (`PolicyEngine.cap_contributions`) before sharing — sensitivity of the joint release equals that radius.
- Own-data products stay local: orgs may square/`xy` their **own** rows before sharing; only cross-org combination happens on secret shares.
- Accountant default limit is `100× (DEFAULT_EPSILON, DEFAULT_DELTA)`; demo epsilons (8.0/4.0/1.0) spend fast — check `GET /budgets` when adding queries.

## Federated / MPC / test quirks
- FL round infers model dim from the first client (`FederatedServer(dim=clients[0].dim)`); all clients must share that feature dim or `local_train` raises `"model dim mismatch"`. Released `weights` length is `dim+1` (bias + features). Roster with duplicate org ids is rejected.
- Dropout recovery: still pass the **full** clients dict and list dropouts in `drop=[org_id]` — the platform handles seed reveal internally (`demo.py` pattern). `result.participants` lists survivors only.
- Trust boundary is structural: `SecAggClient` owns its X25519 private key; `SecAggServer` must never hold key material (`tests/test_secure_aggregation.py` enforces this by walking the server object). Keep all masking client-side.
- FL eval stats are pooled through a masked dim-3 sub-round (`clip_bound=None` — tallies must not be L2-clipped). `FederatedServer._secure_eval` takes the same `(clients, model)` args; per-client tallies never cross in the clear.
- `mean` is **not** a servable metric: it was removed from `ALLOWED_METRICS` (no `platform.mean()` / `/queries/mean` ever existed). `mpc/stats.py::run_mean` remains only as an unused pooled helper.
- DP noise is unseeded per call: assert on types/ranges (`isinstance`, `>= 0`, `[-1, 1]`) or deterministic `raw_value`, never exact noisy values.
- `api/app.py::PLATFORM` is a process-global singleton — API tests must swap in a fresh `SentryLinkPlatform` per test and restore it after (see `tests/test_api.py::client` fixture), or state leaks across tests. NOTE: `import sentrylink.api.app as x` binds the FastAPI instance, not the module (package attribute shadowing) — swap via `sys.modules["sentrylink.api.app"].PLATFORM`.

## Storage (SQLite + SQLAlchemy 2.0)
- Layering is strict: domain objects → `storage/codec.py` → plain dicts/bytes → `storage/store.py` → SQL tables. `codec.py` is the only translation point and the privacy boundary — nothing outside it may touch persisted forms.
- Backends: `InMemoryStateStore` (default; tests, zero I/O) and `SQLiteStateStore(path)` (file/`:memory:`, WAL + 30s busy timeout, per-op sessions). Select via `SENTRYLINK_DB` env var (path or `sqlite:///` URL); `api/app.py::build_platform` reads it at startup.
- `SentryLinkPlatform(store=...)` replays the store on construction (stored state wins over constructor args); every mutation commits one atomic bundle (`_commit`) and rebuilds live objects from the store if the commit fails. All six mutating methods hold a process-wide `RLock` — keep new mutations under it.
- Cross-process writers are guarded by freshness proofs, not locks: `save_budget` takes `expect_spent`, `save_server` takes `expect_rounds`, `append_audit` takes `expect_prev` (== the entry's own `prev_hash`). A stale bundle fails with `ConcurrentWriteError` (code `CONCURRENT_WRITE` → HTTP 409) and the platform reconverges — never silent overwrites or double-spends. The audit conditional-INSERT is a single atomic statement, so it serializes all bundles. Clients retry on 409; single-process behavior is unchanged.
- Migrations: `storage/migrations.py` (`CURRENT_SCHEMA_VERSION`, sequential `MIGRATIONS`, `schema_meta` table); `init_db()` migrates at engine creation. Add future schema changes as new versioned steps, never by editing v1.
- Never persisted: plaintext API keys (only `pbkdf2-sha256` hashes via `registry.hash_api_key`), raw rows/features, private keys, unmasked updates. `tests/test_storage.py::EXPECTED_COLUMNS` whitelists every column — extend it if you add tables.
- Type gate: `python -m mypy sentrylink` (whole repo) must stay clean — CI enforces it. `platform.py` uses `_policy`/`_accountant` narrowing properties for the Optional fields; external code keeps using `p.policy`/`p.accountant` directly.

## Ops (request IDs, rate limits, settings, logging)
- Every response carries `X-Request-Id` (accepted inbound when `[\w-]{1,64}`, else generated). Pass `request_id=` into the six platform mutations so audit entries and ledger records carry it. Log lines are fixed-schema only (`observability.log_event`) — never bodies/keys/results; `observability.redact` strips credential-shaped keys (conservative: `public_key`-style metadata is redacted too).
- Spend endpoints (`/queries/*`, `/federated/round`) check `RATE_LIMITER` (module global, built from settings; swap it in tests like `PLATFORM`). Limits are per-org (`spend:{org_id}`), injected via the `RateLimiter` ABC — never confuse 429 `RATE_LIMITED` with budget exhaustion.
- Runtime config lives in `sentrylink/settings.py` (`SENTRYLINK_DB/LOG_LEVEL/RATE_LIMIT/ENV`), validated at startup with fail-fast `ValueError`. Tests pass explicit dicts to `Settings.from_env` — never depend on ambient env.

## Product modules
- `sentrylink/verticals.py`: `VerticalPolicy` objects + `DATASETS` (metadata over `usecases/synthetic.py`, which stays the single generator implementation). `policy.default_policy_for` delegates to them — behavior must stay identical (pinned by `tests/test_verticals.py`).
- `sentrylink/privacy.py`: `PrivacyLedgerEntry` (persisted inside budget events), `platform.preview()` (pure reads only — no charge/audit/store), `build_release_card`/`render_card_text` (runtime values only, cohort sizes never identities), `platform.model_release_metadata()` (traceable envelope incl. `PROTOCOL_VERSION`).
- `sentrylink/{redteam,simulate,bench,report}.py`: CLIs with `main(argv) -> int`; tests call the functions, CI runs the CLIs. Keep fixtures tiny (suite budget matters).
- `examples/`: runnable via `python -m examples.<name>` from root (package is not installed; plain `python examples/x.py` fails on `sys.path`).
- Never use `pickle` for domain state (explicit codecs only) — `scripts/privacy_review.py` enforces this plus log/credential patterns in CI.
