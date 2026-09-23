"""Preview what a query would cost — without spending anything."""

import json

from sentrylink.platform import SentryLinkPlatform


def main() -> None:
    platform = SentryLinkPlatform()
    for i in range(3):
        platform.join(f"Retailer {i}", domain="retail",
                      sector_group="retail-consortium")
    preview = platform.preview(
        "histogram", "retail-consortium", "retail", epsilon=2.0)
    print(json.dumps(preview, indent=2))
    assert platform.budget_report()["spent_epsilon"] == 0.0
    print("budget untouched: preview spends nothing")


if __name__ == "__main__":
    main()
