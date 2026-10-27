"""Verify that the committed data files match ``data/MANIFEST.sha256``.

The manifest is in ``sha256sum`` format, so this script is a thin, dependency
free cross-check that also works where ``sha256sum`` is unavailable (Windows).

Usage:
    python tools/verify_data_manifest.py [--manifest data/MANIFEST.sha256]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_manifest(path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, target = line.partition("  ")
        if not target:
            raise ValueError(f"malformed manifest line: {line!r}")
        entries.append((digest, ROOT / target))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "MANIFEST.sha256")
    args = parser.parse_args()

    entries = parse_manifest(args.manifest)
    missing: list[Path] = []
    mismatched: list[tuple[Path, str, str]] = []
    total = 0

    for expected, path in entries:
        if not path.exists():
            missing.append(path)
            continue
        total += path.stat().st_size
        actual = sha256(path)
        if actual != expected:
            mismatched.append((path, expected, actual))

    for path in missing:
        print(f"MISSING   {path.relative_to(ROOT)}")
    for path, expected, actual in mismatched:
        print(f"MISMATCH  {path.relative_to(ROOT)}\n          expected {expected}\n          actual   {actual}")

    checked = len(entries) - len(missing) - len(mismatched)
    print(
        f"\nchecked {len(entries)} files, {total / 1048576:.1f} MB: "
        f"{checked} ok, {len(mismatched)} mismatched, {len(missing)} missing"
    )
    if missing or mismatched:
        print("DATA MANIFEST FAILED")
        return 1
    print("DATA MANIFEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
