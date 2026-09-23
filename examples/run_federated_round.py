"""Run one secure-aggregated federated round (with dropout recovery)."""

from sentrylink.federated.client import FederatedClient
from sentrylink.platform import SentryLinkPlatform
from sentrylink.usecases import make_retail_cohort


def main() -> None:
    platform = SentryLinkPlatform()
    orgs = make_retail_cohort(n_orgs=4, n_per_org=200, seed=7)
    ids = [
        platform.join(o.name, domain="retail",
                      sector_group="retail-consortium").org_id
        for o in orgs
    ]
    clients = {
        oid: FederatedClient(org_id=oid, x=o.x, y=o.y) for oid, o in zip(ids, orgs)
    }
    result = platform.run_federated_round(
        "retail-consortium", "retail", clients,
        epsilon=8.0, epochs=1, drop=[ids[-1]],
    )
    print(f"participants: {result.participants}")
    print(f"dropped (recovered): {result.dropped}")
    print(f"accuracy: {result.eval_stats['accuracy']:.3f}")
    meta = platform.model_release_metadata("retail-consortium:retail")
    print(f"model version: {meta['model_version']}, dp epsilon: {meta['dp']['epsilon']}")


if __name__ == "__main__":
    main()
