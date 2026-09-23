# AGENTS.md — SentryLink

## Setup & commands
- Requires Python `>=3.11`. Install: `pip install -r requirements.txt` (Windows venv: `.venv\Scripts\activate`).
- End-to-end check: `python -m sentrylink.demo` — runs all three verticals (retail, manufacturing, healthcare) on one in-process platform.
- Full suite: `python -m pytest -q` (~47 tests, ~15s, no external services).
- Single test: `python -m pytest tests/test_<name>.py -q` (e.g. `test_secure_aggregation.py`, `test_api.py`); focused: `python -m pytest tests/test_api.py -k <substring> -q`.
- API dev server: `uvicorn sentrylink.api.app:app --reload` → docs at `/docs`.
- No lint / typecheck / CI config exists — `pytest` (`testpaths=["tests"]` in `pyproject.toml`) is the only verification gate.

## Architecture (entrypoints)
- `sentrylink/platform.py::SentryLinkPlatform` is the single orchestration entry — policy + MPC + DP + audit. `sentrylink/api/app.py` is a thin wrapper over it; add new queries there by delegating to a `platform.py` method.
- `sentrylink/config.py` holds coupled crypto/DP constants — `FIELD_P = 2**127-1`, `QUANT_SCALE = 1e6`, `MASK_BOUND = 2**40` (int64-safe), `UPDATE_CLIP = 1.0`, `MAX_ORG_CONTRIB = 100.0`. Changing any one breaks masking/DP sensitivity assumptions elsewhere.
- Privacy split: **Laplace (pure ε-DP)** for `histogram`/`variance`/`correlation` (accountant charged with `delta=0`); **Gaussian ((ε,δ)-DP)** only for federated rounds. Keep this split when adding releases.
- MPC is 2 non-colluding nodes (`NODE_IDS` in `platform.py`); collusion is explicitly out of scope. Beaver triples come from a preprocessing dealer (demo-only, not malicious-secure).
- `api/app.py::PLATFORM` is a module-level in-memory singleton — no DB. State (registry, servers, audit, accountant) resets on process restart; `GET /federated/model` 404s until a round has run for that `sector_group:domain` key.

## Governance gotchas (cause most test/API failures)
- Cohort floor is always **≥3 consented orgs** (`MIN_PARTICIPANTS`): `PolicyEngine.evaluate` checks the global, never the per-policy `min_participants` field — healthcare's `min_participants=4` in `default_policy_for` is dead config. Healthcare's enforced difference is only its allow-list `{histogram, correlation, federated_model_round}` (`variance` denied) and `max_epsilon_per_query=10.0` (25.0 elsewhere).
- Per-query caps: `epsilon > 0`, `delta in (0, 1)`, `epsilon <= max_epsilon_per_query`. Violations raise `PermissionError` → API maps to **403**; missing-org data / bad shapes raise `ValueError`/`KeyError` → **400**; bad `api_key` on `/consent` → **401**. Budget exhaustion (`PrivacyAccountant.charge`, basic composition by summation) also raises `PermissionError` → 403.
- Cohort is scoped by **both** `sector_group` and `domain` (`Registry.cohort`); joining with a mismatched string yields zero candidates → `"cohort too small"` denial.
- `platform.set_consent` **replaces** the whole policy with defaults — calling it on a healthcare org resets `max_epsilon_per_query` to 25.0, silently dropping the stricter 10.0 cap.
- Every consented org must supply input (`platform.py` raises `"all consented orgs must supply..."` otherwise). Pass exactly `decision.participants` worth of data.
- Per-org histogram vectors are L2-projected onto the `MAX_ORG_CONTRIB` ball (`PolicyEngine.cap_contributions`) before sharing — sensitivity of the joint release equals that radius.
- Own-data products stay local: orgs may square/`xy` their **own** rows before sharing; only cross-org combination happens on secret shares.
- Accountant default limit is `100× (DEFAULT_EPSILON, DEFAULT_DELTA)`; demo epsilons (8.0/4.0/1.0) spend fast — check `GET /budgets` when adding queries.

## Federated / MPC / test quirks
- FL round infers model dim from the first client (`FederatedServer(dim=clients[0].dim)`); all clients must share that feature dim or `local_train` raises `"model dim mismatch"`. Released `weights` length is `dim+1` (bias + features). Roster with duplicate org ids is rejected.
- Dropout recovery: still pass the **full** clients dict and list dropouts in `drop=[org_id]` — the platform handles seed reveal internally (`demo.py` pattern). `result.participants` lists survivors only.
- `mean` is in `ALLOWED_METRICS` and `mpc/stats.py::run_mean` exists, but there is **no** `platform.mean()` and no `/queries/mean` endpoint — don't request it; wire a new handler first if needed.
- DP noise is unseeded per call: assert on types/ranges (`isinstance`, `>= 0`, `[-1, 1]`) or deterministic `raw_value`, never exact noisy values.
- `api/app.py::PLATFORM` is a process-global singleton — API tests must swap in a fresh `SentryLinkPlatform` per test and restore it after (see `tests/test_api.py::client` fixture), or state leaks across tests.
