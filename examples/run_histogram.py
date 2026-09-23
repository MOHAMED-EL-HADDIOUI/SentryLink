"""Run a privacy-protected histogram query across a cohort."""

from sentrylink.platform import SentryLinkPlatform
from sentrylink.privacy import render_card_text


def main() -> None:
    platform = SentryLinkPlatform()
    ids = [
        platform.join(f"Retailer {i}", domain="retail",
                      sector_group="retail-consortium").org_id
        for i in range(3)
    ]
    buckets = {oid: [12, 7, 3] for oid in ids}
    result = platform.histogram(
        "retail-consortium", "retail", buckets, epsilon=2.0,
        purpose="demand distribution insight",
    )
    print("released (DP-noised):", result.value)
    print("participants:", len(result.participants))
    print()
    print(render_card_text(result.privacy_card))


if __name__ == "__main__":
    main()
