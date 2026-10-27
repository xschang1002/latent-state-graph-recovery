"""Single local entry for verifying or rebuilding the supplied paper artifact.

Default/--verify is read-only. --rebuild regenerates grouping analyses from the
retained old-fit cache and all manuscript tables/figures, then checks numbers.
--compile additionally compiles the manuscript using tectonic or latexmk on PATH.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def verify_bundle(root: Path) -> int:
    manifest_path = root / "PACKAGE_MANIFEST.json"
    if not manifest_path.exists():
        return 0  # Repository checkout; the exported archive carries the manifest.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Invalid or missing bundle payload: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    return len(manifest["sha256"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Read-only verification (default)")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild from supplied results and retained fit outputs")
    parser.add_argument("--compile", action="store_true", help="Also compile with tectonic or latexmk from PATH")
    args = parser.parse_args()
    if args.verify and (args.rebuild or args.compile):
        parser.error("Use --verify alone for a read-only run")
    count = verify_bundle(ROOT)
    print(f"Verified {count} exported payload checksums" if count else "Repository checkout: no exported bundle manifest", flush=True)
    if args.rebuild:
        for source in ("scripts/run_grouping_controls.py", "tools/build_revision2_tables.py",
                       "tools/build_information_sciences_assets.py",
                       "tools/build_information_sciences_supplement.py",
                       "tools/audit_generator_outputs.py",
                       "tools/build_reviewer_controls_assets.py",
                       "tools/build_evaluation_sensitivity_assets.py",
                       "tools/build_narrative_assets.py"):
            subprocess.run([sys.executable, "-B", str(ROOT / source)], cwd=ROOT, check=True)
    for source in ("tools/verify_revision2_numbers.py", "tools/verify_temporal_grid.py",
                   "tools/verify_grouping_controls.py", "tools/verify_information_sciences.py",
                   "tools/validate_paper_static.py"):
        command = [sys.executable, "-B", str(ROOT / source)]
        if args.rebuild and source == "tools/verify_information_sciences.py":
            # Exported bytes were checked before rebuilding; retain all scientific
            # integrity checks while allowing regenerated display timestamps.
            command.append("--skip-bundle")
        subprocess.run(command, cwd=ROOT, check=True)
    if args.compile:
        if engine := shutil.which("tectonic"):
            command = [engine, "-Z", f"search-path={ROOT}", "--keep-logs", "paper/main.tex"]
        elif engine := shutil.which("latexmk"):
            command = [engine, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-outdir=paper", "paper/main.tex"]
        else:
            raise SystemExit("Compilation requires tectonic or latexmk on PATH; verified sources/results are retained.")
        subprocess.run(command, cwd=ROOT, check=True)
        if (ROOT / "paper/supplement.tex").exists():
            second = [part.replace("paper/main.tex", "paper/supplement.tex") for part in command]
            subprocess.run(second, cwd=ROOT, check=True)
    print("Paper verification complete. Rebuilt PDFs can differ in build timestamps; numeric checks use source CSVs.")


if __name__ == "__main__":
    main()
