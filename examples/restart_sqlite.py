"""Persist to SQLite, restart, and prove deterministic recovery."""

import tempfile

from sentrylink.platform import SentryLinkPlatform
from sentrylink.storage import SQLiteStateStore


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = f"{tmp}/state.db"
        p1 = SentryLinkPlatform(store=SQLiteStateStore(db))
        ids = [
            p1.join(f"Retailer {i}", domain="retail",
                    sector_group="retail-consortium").org_id
            for i in range(3)
        ]
        keys = {oid: p1.registry.get(oid).api_key for oid in ids}
        p1.histogram("retail-consortium", "retail",
                     {oid: [5, 2] for oid in ids}, epsilon=1.0)
        spent = p1.budget_report()["spent_epsilon"]
        p1.store.close()

        # "restart": a brand-new platform over the same file
        p2 = SentryLinkPlatform(store=SQLiteStateStore(db))
        assert len(p2.registry.list()) == 3
        assert p2.budget_report()["spent_epsilon"] == spent
        assert p2.audit.verify_chain()
        # original credentials still authenticate (hash path)
        p2.registry.authenticate(ids[0], keys[ids[0]])
        print("restart replay OK: orgs, budget, audit chain, credentials")
        p2.store.close()


if __name__ == "__main__":
    main()
