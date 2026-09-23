"""Register organizations and issue credentials (shown once, memory-only)."""

from sentrylink.platform import SentryLinkPlatform


def main() -> None:
    platform = SentryLinkPlatform()
    for name in ("Retailer A", "Retailer B", "Retailer C"):
        org = platform.join(name, domain="retail", sector_group="retail-consortium")
        # The API key is returned exactly once: store it securely, it is
        # never shown, logged, or persisted in plaintext again.
        print(f"joined {org.name}: org_id={org.org_id}")
        print("  api_key=<issued once, keep secret>")
    print(f"cohort size: {len(platform.registry.list())}")


if __name__ == "__main__":
    main()
