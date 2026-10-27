"""Validate and package the current paper inputs for a manual Overleaf compile.

No PDF is compiled or included. The main file is main.tex at the archive root;
all referenced input, graphics and bibliography paths are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from validate_paper_static import ROOT, collect_sources, validate


def build_package(root: Path, output: Path, main: Path = Path("paper/main.tex")) -> dict[str, object]:
    root = root.resolve()
    main = (main if main.is_absolute() else root / main).resolve()
    report = validate(root, main)
    if report["errors"]:
        raise ValueError("Paper source validation failed: " + "; ".join(report["errors"]))
    sources = collect_sources(root, main)
    paths = set(sources.tex) | set(sources.bib) | sources.assets
    contents = {"main.tex" if path == main else path.relative_to(root).as_posix(): path.read_bytes() for path in paths}
    if len(contents) != len(paths):
        raise ValueError("Archive name collision with main.tex")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stable entry timestamps make identical source inputs produce identical ZIPs.
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or any(archive.read(name) != data for name, data in contents.items()):
            raise ValueError("Archive integrity check failed")
    return {"zip_file": str(output), "file_count": len(contents),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(contents.items())},
            "source_validation": report, "pdf_compiled": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--main", type=Path, default=Path("paper/main.tex"))
    parser.add_argument("--output", type=Path, default=ROOT / "paper_overleaf_revision.zip")
    args = parser.parse_args()
    print(json.dumps(build_package(args.root, args.output, args.main), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
