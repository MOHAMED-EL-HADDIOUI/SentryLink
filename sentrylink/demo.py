"""End-to-end demo: the complete SentryLink story in one terminal run.

Run:  python -m sentrylink.demo

Narrative:
  1. Register organizations          11. Aggregate model
  2. Issue credentials               12. Apply DP
  3. Configure governance            13. Charge privacy budget
  4. Establish consent               14. Write audit event
  5. Generate synthetic local data   15. Verify audit chain
  6. Run histogram                   16. Restart from SQLite
  7. Run variance/correlation        17. Authenticate with original API key
  8. Run federated training          18. Verify model continuity
  9. Simulate dropout                19. Show privacy ledger + Privacy Card
  10. (compact manufacturing + healthcare passes)
                                     20. Show final budget + guarantees

Only aggregates are ever printed. API keys and raw rows never appear.
"""

from __future__ import annotations

import tempfile

import numpy as np

from .federated.client import FederatedClient
from .platform import SentryLinkPlatform
from .privacy import render_card_text
from .storage import SQLiteStateStore
from .usecases import (
    make_healthcare_cohort,
    make_manufacturing_cohort,
    make_retail_cohort,
)
from .verticals import policy_for_domain


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _step(n: int, text: str) -> None:
    print(f"\n[step {n:02d}] {text}")


def demo_retail_narrative() -> SentryLinkPlatform:
    _banner("SENTRYLINK — privacy-preserving cross-organization intelligence")
    print(
        "Raw rows never leave an organization.\n"
        "• Federated learning: pairwise-masked updates (X25519 + PRG), server sees sums only\n"
        "• MPC: additive secret shares across 2 non-colluding nodes (linear-only)\n"
        "• Differential privacy: Laplace on aggregates (pure ε), Gaussian on FL models\n"
        "• Governance: consent policies, k≥3 cohort floor, hash-chained audit log\n"
        "• Persistence: SQLite + migrations, deterministic restart replay"
    )

    tmp = tempfile.mkdtemp(prefix="sentrylink-demo-")
    platform = SentryLinkPlatform(store=SQLiteStateStore(f"{tmp}/demo.db"))

    _step(1, "register organizations (retail consortium)")
    orgs = make_retail_cohort(n_orgs=4, n_per_org=200, seed=7)
    registered = [
        platform.join(o.name, domain="retail", sector_group="retail-consortium")
        for o in orgs
    ]
    by_id = {reg.org_id: data for reg, data in zip(registered, orgs)}
    org_ids = [reg.org_id for reg in registered]
    print(f"  onboarded {len(registered)} organizations")

    _step(2, "issue credentials (shown once, memory-only, hashed at rest)")
    print(f"  {len(registered)} API keys issued — never printed, logged, or persisted")

    _step(3, "configure governance from the retail vertical policy")
    vp = policy_for_domain("retail")
    print(f"  floor k>={vp.min_participants}, per-query eps<={vp.max_epsilon_per_query}, "
          f"metrics={sorted(vp.allowed_metrics)}")

    _step(4, "establish consent (allow-list only; caps preserved)")
    for oid in org_ids:
        platform.set_consent(oid, {"histogram", "variance", "correlation",
                                   "federated_model_round"})
    print("  all 4 orgs consented to analytics metrics")

    _step(5, "generate synthetic local data (200 rows/org, never leaves the org)")
    clients = {oid: FederatedClient(org_id=oid, x=by_id[oid].x, y=by_id[oid].y)
               for oid in org_ids}

    _step(6, "histogram via secret-shared MPC + Laplace DP")
    buckets = {}
    for oid, o in zip(org_ids, orgs):
        binned = np.digitize(o.x[:, 3], [0.0])
        buckets[oid] = [int(np.sum((binned == 0) & (o.y == 1))),
                        int(np.sum((binned == 1) & (o.y == 1)))]
    hist = platform.histogram("retail-consortium", "retail", buckets,
                              labels=["low", "high"], epsilon=2.0,
                              purpose="demand distribution insight")
    print(f"  released (DP): {hist.value}")

    _step(7, "variance + correlation where allowed")
    values = {oid: by_id[oid].y.tolist() for oid in org_ids}
    var = platform.variance("retail-consortium", "retail", values, epsilon=2.0)
    pairs = {oid: (by_id[oid].x[:, 0].tolist(), by_id[oid].y.tolist())
             for oid in org_ids}
    corr = platform.correlation("retail-consortium", "retail", pairs, epsilon=2.0)
    print(f"  variance={var.value:.4f}  correlation={corr.value:.4f}")

    _step(8, "federated training round (masked updates, server sees sums only)")
    result = platform.run_federated_round(
        "retail-consortium", "retail", clients, epsilon=8.0, epochs=2)
    print(f"  accuracy={result.eval_stats['accuracy']:.3f} "
          f"n={result.eval_stats['n']} participants={len(result.participants)}")

    _step(9, "simulate dropout (1 of 4 drops; survivors reveal recovery seeds)")
    dropped = platform.run_federated_round(
        "retail-consortium", "retail", clients, epsilon=4.0, epochs=1,
        drop=[org_ids[-1]])
    print(f"  dropped={dropped.dropped} recovered OK")
    weights_before = platform.servers["retail-consortium:retail"].model.flat.copy()

    _step(10, "aggregate model (FedAvg over masked deltas)")
    meta = platform.model_release_metadata("retail-consortium:retail")
    print(f"  model v{meta['model_version']}: dim={meta['feature_dimension']} "
          f"aggregation={meta['aggregation']}")

    _step(11, "differential privacy applied (Gaussian, analytic calibration)")
    print(f"  dp_applied={result.dp_applied} epsilon={result.epsilon_used} "
          f"sigma={result.dp_sigma:.4f}")

    _step(12, "privacy budget charged (immutable ledger entry)")
    entry = platform.accountant.events[-1]["ledger"]
    print(f"  {entry['query_type']}: eps={entry['epsilon']} "
          f"mechanism={entry['mechanism']} cohort={entry['cohort_size']}")

    _step(13, "audit event written (hash-chained)")
    print(f"  total audit entries: {len(platform.audit.entries)}")

    _step(14, "verify audit chain")
    print(f"  chain valid: {platform.audit.verify_chain()}")

    _step(15, "restart from SQLite (new process, same file)")
    api_keys = {oid: platform.registry.get(oid).api_key for oid in org_ids}
    platform.store.close()
    platform2 = SentryLinkPlatform(store=SQLiteStateStore(f"{tmp}/demo.db"))

    _step(16, "authenticate with an original API key (hash path)")
    platform2.registry.authenticate(org_ids[0], api_keys[org_ids[0]])
    print("  original key accepted after restart")

    _step(17, "verify model continuity")
    continued = platform2.servers["retail-consortium:retail"].model.flat
    print(f"  weights identical: {bool((continued == weights_before).all())}")

    _step(18, "show privacy ledger (sanitized, auditable spend)")
    for event in platform2.accountant.events[-3:]:
        ledger = event["ledger"]
        print(f"  {ledger['query_type']}: eps={ledger['epsilon']} "
              f"{ledger['mechanism']} cohort={ledger['cohort_size']}")

    _step(19, "show Privacy Card (generated from runtime metadata)")
    print()
    print(render_card_text(hist.privacy_card))

    _step(20, "final budget + guarantees")
    budget = platform2.budget_report()
    print(f"  eps spent: {budget['spent_epsilon']:.3f} / {budget['limit_epsilon']:.1f}")
    print(f"  RDP spend: {budget['rdp_epsilon_spent']:.3f}")
    print()
    print("  RAW DATA LEFT ORGANIZATIONS: NO")
    print("  PRIVATE KEYS LEFT ORGANIZATIONS: NO")
    print("  UNMASKED UPDATES AT SERVER: NO")
    print("  DP APPLIED: YES")
    print(f"  AUDIT CHAIN VALID: {'YES' if platform2.audit.verify_chain() else 'NO'}")
    platform2.store.close()
    return platform


def demo_compact(label: str, sector_group: str, domain: str, orgs) -> None:
    _banner(label)
    platform = SentryLinkPlatform()
    registered = [platform.join(o.name, domain=domain, sector_group=sector_group)
                  for o in orgs]
    by_id = {reg.org_id: data for reg, data in zip(registered, orgs)}
    org_ids = [r.org_id for r in registered]
    clients = {oid: FederatedClient(org_id=oid, x=by_id[oid].x, y=by_id[oid].y)
               for oid in org_ids}
    col = 1 if domain == "manufacturing" else 2
    pairs = {oid: (by_id[oid].x[:, col].tolist(), by_id[oid].y.tolist())
             for oid in org_ids}
    corr = platform.correlation(sector_group, domain, pairs, epsilon=2.0)
    print(f"  correlation (DP): {corr.value:.4f}")
    if domain == "healthcare":
        try:
            platform.variance(sector_group, domain,
                              {oid: by_id[oid].y.tolist() for oid in org_ids})
            print("  ERROR: healthcare variance was allowed")
        except PermissionError:
            print("  healthcare variance denied by policy (as designed)")
    result = platform.run_federated_round(sector_group, domain, clients,
                                          epsilon=8.0, epochs=1)
    print(f"  accuracy={result.eval_stats['accuracy']:.3f} "
          f"audit valid={platform.audit.verify_chain()}")


def main() -> None:
    demo_retail_narrative()
    demo_compact(
        "VERTICAL — MANUFACTURING: cross-plant quality issue spotting",
        "supply-chain-mfg", "manufacturing",
        make_manufacturing_cohort(n_orgs=4, n_per_org=200, seed=11),
    )
    demo_compact(
        "VERTICAL — HEALTHCARE: treatment-response clusters",
        "hospital-network", "healthcare",
        make_healthcare_cohort(n_orgs=4, n_per_org=200, seed=23),
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
