"""Static privacy review: fail CI on dangerous persistence/logging patterns.

Conservative by design: scans sentrylink/ source (tests intentionally handle
fixture credentials, so they are out of scope) for:
  - pickle usage for domain state (explicit codecs required instead)
  - logging/printing anything shaped like a credential or key
  - raw-SQL persistence of sensitive identifiers
  - weak hashes (md5/sha1) anywhere near credentials

Documented exceptions live in ALLOWLIST with justifications.
Run:  python scripts/privacy_review.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "sentrylink"

# (pattern, message). Checked against every .py file under sentrylink/.
PATTERNS = [
    (r"^\s*import pickle\b|^\s*from pickle import|pickle\.(loads|dumps|load|dump)\(",
     "pickle usage: persistent domain state requires explicit codecs"),
    (r"print\(.*api_key|print\(.*private_key",
     "printing credential material"),
    (r"log(?:ger|ging)?\.(?:info|debug|warning|error|exception)\(.*(?:api_key|private_key)",
     "logging credential material"),
    (r"INSERT INTO\s+(?:raw_rows|api_keys|private_keys|features)\b",
     "raw SQL persistence of sensitive identifiers"),
    (r"hashlib\.(?:md5|sha1)\(",
     "weak hash near credentials (use sha256+ / HKDF)"),
    (r"CREATE TABLE\s+(?:raw_rows|api_keys|private_keys)\b",
     "sensitive table definition"),
]

# (file suffix, pattern, justification). Keep this list empty-or-justified.
ALLOWLIST = [
    # ("observability.py", "private_key", "redaction token list (never logged values)"),
]

FAIL_HINT = "If a finding is a false positive, add a documented ALLOWLIST entry."


def review() -> list[str]:
    findings = []
    files = sorted(SRC.rglob("*.py"))
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for pattern, message in PATTERNS:
            for i, line in enumerate(text.splitlines(), start=1):
                if not re.search(pattern, line):
                    continue
                if any(rel.endswith(f) and re.search(p, line) for f, p in ALLOWLIST):
                    continue
                findings.append(f"{rel}:{i}: {message}: {line.strip()[:120]}")
    return findings


def main() -> int:
    findings = review()
    if findings:
        print(f"{len(findings)} privacy-review finding(s):")
        for finding in findings:
            print(f"  [FAIL] {finding}")
        print(FAIL_HINT)
        return 1
    print(f"privacy review clean ({len(list(SRC.rglob('*.py')))} files scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
