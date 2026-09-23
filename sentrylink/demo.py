"""End-to-end demo: three verticals on one SentryLink deployment.

Run:  python -m sentrylink.demo
"""

from __future__ import annotations

import numpy as np

from .federated.client import FederatedClient
from .platform import SentryLinkPlatform
from .usecases import (
    make_healthcare_cohort,
    make_manufacturing_cohort,
    make_retail_cohort,
)


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _run_vertical(
    platform: SentryLinkPlatform,
    *,
    label: str,
    sector_group: str,
    domain: str,
    orgs,
    histogram_fn,
    corr_fn,
) -> None:
    _banner(label)

    registered = [
        platform.join(o.name, domain=domain, sector_group=sector_group) for o in orgs
    ]
    by_id = {reg.org_id: data for reg, data in zip(registered, orgs)}

    # map synthetic ids to registry ids
    org_ids = [reg.org_id for reg in registered]
    print(f"Consortium '{sector_group}': {len(registered)} organizations onboarded")
    for reg in registered:
        print(f"  - {reg.name} ({reg.org_id}) domain={reg.domain}")

    # ---- federated round (secure aggregation + DP) ----
    clients = {
        oid: FederatedClient(org_id=oid, x=by_id[oid].x, y=by_id[oid].y)
        for oid in org_ids
    }
    result = platform.run_federated_round(
        sector_group,
        domain,
        clients,
        epsilon=8.0,
        delta=1e-5,
        epochs=2,
    )
    print("\n[Federated round]")
    print(f"  round_id        : {result.round_id}")
    print(f"  participants    : {result.participants}")
    print(f"  dropped         : {result.dropped or 'none'}")
    print(f"  DP noise applied: {result.dp_applied} (epsilon={result.epsilon_used})")
    print(
        f"  cohort metrics  : acc={result.eval_stats['accuracy']:.3f} "
        f"loss={result.eval_stats['mean_loss']:.3f} n={result.eval_stats['n']}"
    )
    print("  per-client updates were pairwise-masked; server saw only the sum.")

    # dropout resilience round
    if len(org_ids) >= 4:
        drop_id = org_ids[-1]
        r2 = platform.run_federated_round(
            sector_group, domain, clients, epsilon=4.0, epochs=1, drop=[drop_id]
        )
        print(f"\n[Dropout round] dropped={r2.dropped} → recovery seeds unmasked OK")

    # ---- histogram query (MPC + DP) ----
    buckets = histogram_fn(orgs, org_ids)
    labels = [f"b{i}" for i in range(len(next(iter(buckets.values()))))]
    hres = platform.histogram(
        sector_group,
        domain,
        buckets,
        labels=labels,
        epsilon=8.0,
        delta=1e-5,
        purpose=f"{domain} distribution insight",
    )
    print("\n[Histogram via secret-shared MPC + DP]")
    print(f"  participants    : {hres.participants}")
    print(f"  raw joint totals: {hres.raw_value}  (never attributed per org)")
    print(f"  released (DP)   : {hres.value}")
    print(f"  noise sigma     : {hres.noise_sigma:.1f}")

    # ---- correlation query ----
    pairs = corr_fn(orgs, org_ids)
    cres = platform.correlation(
        sector_group,
        domain,
        pairs,
        epsilon=8.0,
        delta=1e-5,
        purpose=f"{domain} association insight",
    )
    print("\n[Correlation via secret-shared sufficient statistics + DP]")
    print(f"  raw ρ (pooled)  : {cres.raw_value:.4f}")
    print(f"  released ρ (DP) : {cres.value:.4f}")


def demo_retail(platform: SentryLinkPlatform) -> None:
    orgs = make_retail_cohort()

    def hist(orgs, ids):
        out = {}
        for oid, o in zip(ids, orgs):
            # demand-lift buckets by foot-traffic quartile
            q = np.quantile(o.x[:, 3], [0.25, 0.5, 0.75])
            binned = np.digitize(o.x[:, 3], q)
            counts = [int(np.sum((binned == k) & (o.y == 1))) for k in range(4)]
            out[oid] = counts
        return out

    def corr(orgs, ids):
        return {oid: (o.x[:, 0].tolist(), o.y.tolist()) for oid, o in zip(ids, orgs)}

    _run_vertical(
        platform,
        label="VERTICAL 1 — RETAIL: emerging demand pattern detection",
        sector_group="retail-consortium",
        domain="retail",
        orgs=orgs,
        histogram_fn=hist,
        corr_fn=corr,
    )


def demo_manufacturing(platform: SentryLinkPlatform) -> None:
    orgs = make_manufacturing_cohort()

    def hist(orgs, ids):
        out = {}
        for oid, o in zip(ids, orgs):
            # defect counts by vibration severity band
            edges = [-1.0, -0.25, 0.25, 1.0, 10.0]
            binned = np.digitize(o.x[:, 0], edges[1:-1])
            counts = [int(np.sum((binned == k) & (o.y == 1))) for k in range(4)]
            out[oid] = counts
        return out

    def corr(orgs, ids):
        return {oid: (o.x[:, 1].tolist(), o.y.tolist()) for oid, o in zip(ids, orgs)}

    _run_vertical(
        platform,
        label="VERTICAL 2 — MANUFACTURING: cross-plant quality issue spotting",
        sector_group="supply-chain-mfg",
        domain="manufacturing",
        orgs=orgs,
        histogram_fn=hist,
        corr_fn=corr,
    )


def demo_healthcare(platform: SentryLinkPlatform) -> None:
    orgs = make_healthcare_cohort()

    def hist(orgs, ids):
        out = {}
        for oid, o in zip(ids, orgs):
            # response counts by treatment arm (col 3) + severity band (col 1)
            arm = np.rint((o.x[:, 3] > 0).astype(int)).astype(int)
            sev = (o.x[:, 1] > 0).astype(int)
            counts = [int(np.sum((arm == a) & (sev == s) & (o.y == 1))) for a in (0, 1) for s in (0, 1)]
            out[oid] = counts
        return out

    def corr(orgs, ids):
        # biomarker (col 2) vs response
        return {oid: (o.x[:, 2].tolist(), o.y.tolist()) for oid, o in zip(ids, orgs)}

    _run_vertical(
        platform,
        label="VERTICAL 3 — HEALTHCARE: treatment-response clusters",
        sector_group="hospital-network",
        domain="healthcare",
        orgs=orgs,
        histogram_fn=hist,
        corr_fn=corr,
    )


def main() -> None:
    platform = SentryLinkPlatform()

    _banner("SENTRYLINK — privacy-preserving cross-organization intelligence")
    print(
        "Raw rows never leave an organization.\n"
        "• Federated learning: pairwise-masked updates (X25519 + PRG), server sees sums only\n"
        "• MPC: additive secret shares across 2 non-colluding nodes (linear-only)\n"
        "• Differential privacy: Laplace on aggregates (pure ε), Gaussian on FL models\n"
        "• Governance: consent policies, k≥3 cohort floor, hash-chained audit log\n"
    )

    demo_retail(platform)
    demo_manufacturing(platform)
    demo_healthcare(platform)

    _banner("PRIVACY ACCOUNTING & AUDIT")
    budget = platform.budget_report()
    print(f"  ε spent   : {budget['spent_epsilon']:.3f} / {budget['limit_epsilon']:.1f}")
    print(f"  ε remaining: {budget['remaining_epsilon']:.3f}")
    print(f"  audit entries : {budget['events']} charged events")
    print(f"  audit chain OK: {platform.audit.verify_chain()}")
    print(f"  total audit entries: {len(platform.audit.entries)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
