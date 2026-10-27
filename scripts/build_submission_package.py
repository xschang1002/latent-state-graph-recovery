"""Build a local anonymous reviewer-facing submission artifact package."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import hashlib
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PACKAGE_INCLUDE_PREFIXES = (
    "analysis/",
    "audit/",
    "configs/",
    "data/external/essdata/",
    "mechanism_research/",
    "notes/",
    "results/aggregation/",
    "results/baselines/",
    "results/confirmatory/",
    "results/figures/",
    "results/mechanism/",
    "results/revision2/",
    "results/tables/",
    "results/theory/",
    "scripts/",
    "src/",
    "tests/",
    "tools/",
)

PACKAGE_INCLUDE_FILES = {
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "docs/anonymous_artifact_readme.md",
    "docs/real_data_ingestion_checklist.md",
    "docs/reproduction_manifest.md",
    "docs/third_party_notices.md",
    "docs/revision_specialist_protocol_2026-09-13.md",
    "docs/revision_real_protocol_2026-09-13.md",
    "docs/revision_gate_protocol_2026-09-13.md",
    "docs/change_point_gate_protocol_2026-10-21.md",
    "docs/gate_identifiability_audit_2026-10-21.md",
    "docs/estimands.md",
    "paper/legacy/main_pre_revision.tex",
    "results/original_reproduction.csv",
    "paper/main.tex",
    "paper/references.bib",
    "submission_materials/mlj_contribution_information/main.tex",
    "submission_materials/mlj_contribution_information/main.pdf",
    "pyproject.toml",
    "requirements.txt",
    "requirements-revision.txt",
}

SPECIALIST_METADATA_FILES = {
    f"results/tables/revision_specialist_{name}.json"
    for name in ("protocol", "storage_amendment", "schedule_amendment",
                 "schedule_16_amendment", "schedule_drain", "interrupted_status", "analysis_provenance")
}
SPECIALIST_RECORD_PATTERN = re.compile(
    r"results/tables/revision_specialist_parts(?:_history)?/"
    r"(?:stress|timing)_[0-9a-f]{12}_seed[0-9]{3,}_[a-z0-9]+\.(?:json|log)$"
)
REAL_PROVENANCE_FILES = {
    "experiments/logs/revision_real_validation/executed_manifest.json",
    "experiments/logs/revision_real_validation/reporting_manifest.json",
}

PACKAGE_PLACEHOLDER_FILES = {
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "experiments/logs/.gitkeep",
}

PACKAGE_EXCLUDE_FILES = {
    "results/tables/real_benchmark_run_plan.md",
    "results/tables/real_data_inventory.csv",
    "results/tables/real_data_inventory.md",
    # Internal repository audit: quotes absolute paths as evidence and is not
    # part of the reviewer-facing artifact.
    "audit/repo_audit_2026-10-22.md",
}

IDENTITY_PATTERNS = {
    "local_windows_path": re.compile(r"\b[A-Za-z]:[\\/]+[^\s\"'<>]+", re.IGNORECASE),
    "local_posix_home": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "orcid": re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b"),
    "github_profile": re.compile(r"https?://(?:www\.)?github\.com/[A-Za-z0-9_.-]+/?(?=[\s\"'<>)]|$)"),
    "tex_non_anonymous_author": re.compile(r"\\author\{(?!\s*Anonymous\b)[^}]{1,200}\}"),
}

ALLOWED_MATCHES = {
    "anonymous@example.com",
}

PDF_METADATA_KEYS = ("Author", "Creator", "Producer")

TEXT_SUFFIXES = {
    ".bib",
    ".cfg",
    ".csv",
    ".json",
    ".log",
    ".md",
    ".py",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class PackageEntry:
    source: Path
    archive_name: str
    content: bytes | None = None
    original_sha256: str | None = None

    def read_bytes(self) -> bytes:
        return self.content if self.content is not None else self.source.read_bytes()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submission_package")
    parser.add_argument("--name", default="stcg_anonymous_artifact")
    parser.add_argument("--no-pdf", action="store_true", help="Do not include paper/main.pdf")
    parser.add_argument("--require-complete-specialist", action="store_true",
                        help="Require every frozen stress and exclusive-timing job record before export")
    return parser.parse_args()


def git_ls_files(root: Path = ROOT) -> list[str]:
    """Read the current working tree, including new nonignored source files.

An extracted artifact has no .git directory; use the same whitelist there.
Ignored data, environments and generated scratch outputs are never walked.
"""
    if not (root / ".git").exists():
        candidates = set(PACKAGE_INCLUDE_FILES | PACKAGE_PLACEHOLDER_FILES)
        for prefix in PACKAGE_INCLUDE_PREFIXES:
            directory = root / prefix
            if directory.is_dir():
                candidates.update(p.relative_to(root).as_posix() for p in directory.rglob("*") if p.is_file())
        return sorted(name for name in candidates if (root / name).is_file())
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return sorted({item for item in result.stdout.decode("utf-8").split("\0") if item})


def should_include(path: str) -> bool:
    parts = Path(path).parts
    if any(part in {"__pycache__", ".pytest_cache", ".venv"} for part in parts):
        return False
    if Path(path).suffix.lower() in {".pyc", ".pyo"}:
        return False
    if path in SPECIALIST_METADATA_FILES or path in REAL_PROVENANCE_FILES:
        return True
    if SPECIALIST_RECORD_PATTERN.fullmatch(path):
        return path.endswith(".json") or "/revision_specialist_parts_history/" in path
    if path.startswith("results/tables/") and Path(path).name != ".gitkeep":
        if Path(path).suffix.lower() not in {".csv", ".md", ".tex"}:
            return False
    if path.startswith("results/figures/") and Path(path).name != ".gitkeep":
        if Path(path).suffix.lower() not in {".png", ".jpg", ".jpeg"} and path != "results/figures/rev2_temporal_grid.pdf":
            return False
    if path in PACKAGE_EXCLUDE_FILES:
        return False
    if path in PACKAGE_INCLUDE_FILES or path in PACKAGE_PLACEHOLDER_FILES:
        return True
    return any(path.startswith(prefix) for prefix in PACKAGE_INCLUDE_PREFIXES)


def specialist_candidates(root: Path) -> set[str]:
    """Explicitly collect only approved ignored evidence, never a raw-data tree."""
    names = {name for name in SPECIALIST_METADATA_FILES | REAL_PROVENANCE_FILES if (root / name).is_file()}
    for directory in ("revision_specialist_parts", "revision_specialist_parts_history"):
        for path in (root / "results/tables" / directory).glob("*"):
            name = path.relative_to(root).as_posix()
            if path.is_file() and SPECIALIST_RECORD_PATTERN.fullmatch(name) and should_include(name):
                names.add(name)
    return names


def anonymize_diagnostic(text: str) -> str:
    """Remove machine-specific prefixes only in exported diagnostic copies."""
    def replace(match: re.Match[str]) -> str:
        value = match.group(0).replace("\\", "/")
        if "/stcg-revision/" in value:
            return "<artifact>/" + value.split("/stcg-revision/", 1)[1]
        if "/Lib/" in value:
            return "<python-runtime>/Lib/" + value.split("/Lib/", 1)[1]
        return "<local-path>/" + value.rsplit("/", 1)[-1]
    return IDENTITY_PATTERNS["local_windows_path"].sub(replace, text)


def snapshot_entry(source: Path, archive_name: str) -> PackageEntry:
    original = source.read_bytes()
    content = original
    if SPECIALIST_RECORD_PATTERN.fullmatch(archive_name):
        if archive_name.endswith(".log"):
            content = anonymize_diagnostic(original.decode("utf-8")).encode("utf-8")
        else:
            record = json.loads(original)
            changed = False
            for key in ("error", "traceback"):
                if isinstance(record.get(key), str):
                    safe = anonymize_diagnostic(record[key])
                    changed |= safe != record[key]
                    record[key] = safe
            if changed:
                content = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    original_hash = hashlib.sha256(original).hexdigest() if original != content else None
    if SPECIALIST_RECORD_PATTERN.fullmatch(archive_name) and original_hash is None:
        # The previous export may live at the repository root or under
        # submission_package/; both are legitimate sources for prior redaction
        # hashes, and a fresh checkout has neither until the first export.
        candidates = [parent / "PACKAGE_MANIFEST.json" for parent in source.parents]
        candidates.append(ROOT / "submission_package" / "PACKAGE_MANIFEST.json")
        candidates.extend(sorted((ROOT / "submission_package").glob("*_manifest.json")))
        manifest_path = next((path for path in candidates if path.is_file()), None)
        if manifest_path is not None:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            redaction = prior.get("diagnostic_redactions", {}).get(archive_name, {})
            if redaction.get("exported_sha256") == hashlib.sha256(content).hexdigest():
                original_hash = redaction.get("original_sha256")
    return PackageEntry(source, archive_name, content, original_hash)


def specialist_snapshot(entries: list[PackageEntry], require_complete: bool = False) -> dict | None:
    """Validate recorded job identities/configurations using the frozen JSON only."""
    by_name = {entry.archive_name: entry for entry in entries}
    protocol_name = "results/tables/revision_specialist_protocol.json"
    parts = [name for name in by_name if name.startswith("results/tables/revision_specialist_parts/")]
    if protocol_name not in by_name:
        if parts or require_complete:
            raise ValueError("Specialist evidence requires the frozen protocol JSON")
        return None
    required = {"docs/revision_specialist_protocol_2026-09-13.md",
                "scripts/run_revision_specialist_stress.py", "tools/summarize_revision_specialist.py"}
    if missing := required - by_name.keys():
        raise ValueError(f"Specialist offline analysis dependencies are missing: {sorted(missing)}")
    frozen = json.loads(by_name[protocol_name].read_bytes())
    spec = frozen["specification"]
    calculated = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if calculated != frozen["protocol_hash"]:
        raise ValueError("Specialist frozen protocol hash mismatch")
    hashes = dict(spec["source_hashes"])
    hashes["docs/revision_specialist_protocol_2026-09-13.md"] = spec["protocol_document_sha256"]
    amendment_name = "results/tables/revision_specialist_storage_amendment.json"
    if amendment_name in by_name:
        amendment = json.loads(by_name[amendment_name].read_bytes())
        runner = "scripts/run_revision_specialist_stress.py"
        if (amendment["protocol_hash"] != frozen["protocol_hash"]
                or amendment["old_runner_sha256"] != hashes[runner]):
            raise ValueError("Specialist storage amendment does not match frozen source")
        hashes[runner] = amendment["new_runner_sha256"]
        history = "results/tables/revision_specialist_parts_history/"
        for retry in amendment.get("serialization_retries", []) + amendment.get("interrupted_retries", []):
            checks = {history + retry["job"] + ".log": retry["original_log_sha256"]}
            if "original_record" in retry:
                checks["results/tables/" + retry["original_record"].replace("\\", "/")] = retry["original_record_sha256"]
            for name, expected_hash in checks.items():
                entry = by_name.get(name)
                if entry is None:
                    raise ValueError(f"Specialist amendment history is missing: {name}")
                original_hash = entry.original_sha256 or hashlib.sha256(entry.read_bytes()).hexdigest()
                if original_hash != expected_hash:
                    raise ValueError(f"Specialist original diagnostic history hash mismatch: {name}")
    for name, expected in hashes.items():
        if name not in by_name or hashlib.sha256(by_name[name].read_bytes()).hexdigest() != expected:
            raise ValueError(f"Specialist frozen source dependency mismatch: {name}")
    schedule_name = "results/tables/revision_specialist_schedule_16_amendment.json"
    if schedule_name in by_name:
        schedule = json.loads(by_name[schedule_name].read_bytes())
        scheduler = by_name.get("tools/run_revision_specialist_schedule.py")
        if (schedule["protocol_hash"] != frozen["protocol_hash"] or scheduler is None
                or hashlib.sha256(scheduler.read_bytes()).hexdigest() != schedule["scheduling_tool_sha256"]):
            raise ValueError("Specialist scheduling amendment/source mismatch")
    settings = {cell["setting_id"]: cell["config"] for cell in spec["display_cells"]}
    base = next(cell["setting_id"] for cell in spec["display_cells"] if cell["axis"] == "obs_noise" and cell["value"] == 0)
    expected = {}
    for phase, phase_settings in (("stress", settings), ("timing", {base: settings[base]})):
        for setting, config in phase_settings.items():
            for seed in spec["seeds"]:
                for method in spec["methods"]:
                    name = f"results/tables/revision_specialist_parts/{phase}_{setting}_seed{seed:03d}_{method.replace('-', '').lower()}.json"
                    expected[name] = dict(phase=phase, setting_id=setting, config=config, seed=seed,
                                          method=method, protocol_hash=frozen["protocol_hash"])
    for name in parts:
        record = json.loads(by_name[name].read_bytes())
        if name not in expected or any(record.get(key) != value for key, value in expected[name].items()):
            raise ValueError(f"Specialist checkpoint identity/configuration mismatch: {name}")
        if record.get("status") not in {"ok", "failed", "timeout", "crashed"}:
            raise ValueError(f"Specialist checkpoint is incomplete: {name}")
    for name, entry in by_name.items():
        if not name.startswith("results/tables/revision_specialist_parts_history/"):
            continue
        current_name = name.replace("_parts_history/", "_parts/").removesuffix(".log")
        if name.endswith(".log"):
            current_name += ".json"
        if current_name not in expected:
            raise ValueError(f"Specialist history does not belong to a frozen job: {name}")
        if name.endswith(".json"):
            record = json.loads(entry.read_bytes())
            if any(record.get(key) != value for key, value in expected[current_name].items()):
                raise ValueError(f"Specialist history identity/configuration mismatch: {name}")
    missing = sorted(expected.keys() - set(parts))
    if require_complete and missing:
        raise ValueError(f"Specialist checkpoint is incomplete: {len(missing)} of {len(expected)} job records missing")
    provenance = by_name.get("results/tables/revision_specialist_analysis_provenance.json")
    summarized = json.loads(provenance.read_bytes())["summary"]["stored_fits"] if provenance else None
    if require_complete and summarized != len(parts):
        raise ValueError("Specialist summaries are missing/stale; run tools/summarize_revision_specialist.py before release")
    return {"protocol_hash": frozen["protocol_hash"], "expected_job_records": len(expected),
            "included_job_records": len(parts), "missing_job_records": len(missing),
            "stored_summary_fit_count": summarized, "summary_count_matches_checkpoint": summarized == len(parts),
            "complete": not missing, "scope": "frozen stress and exclusive-timing phases",
            "note": "Failed/timeout records are retained; completeness does not imply every fit succeeded."}


def collect_entries(include_pdf: bool, root: Path = ROOT) -> list[PackageEntry]:
    root = root.resolve()
    readme = root / "docs" / "anonymous_artifact_readme.md"
    if not readme.is_file():
        readme = root / "README.md"
    entries = [
        snapshot_entry(readme, "README.md"),
    ]
    for rel_path in sorted(set(git_ls_files(root)) | specialist_candidates(root)):
        if not should_include(rel_path):
            continue
        source = root / rel_path
        if not source.resolve().is_relative_to(root):
            raise ValueError(f"Source escapes package root: {rel_path}")
        if source.is_file():
            entries.append(snapshot_entry(source, rel_path.replace("\\", "/")))

    if include_pdf:
        pdf = root / "paper" / "main.pdf"
        if not pdf.exists():
            raise FileNotFoundError("paper/main.pdf is missing; build the paper or use --no-pdf")
        entries.append(snapshot_entry(pdf, "paper/main.pdf"))

    required = {
        "README.md", "requirements.txt", "requirements-revision.txt", "pyproject.toml",
        "tools/build_revision_tables.py", "tools/verify_paper_numbers.py",
        "tools/validate_paper_static.py", "tools/build_overleaf_package.py",
        "docs/reproduction_manifest.md", "paper/main.tex", "paper/references.bib",
        "scripts/build_submission_package.py",
    }
    missing = required - {entry.archive_name for entry in entries if entry.source.is_file()}
    if missing:
        raise FileNotFoundError(f"Required artifact files are missing: {sorted(missing)}")
    return sorted(entries, key=lambda entry: entry.archive_name)


def is_text_archive_name(name: str) -> bool:
    return Path(name).suffix.lower() in TEXT_SUFFIXES


def scan_text(text: str, archive_name: str) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern_name, pattern in IDENTITY_PATTERNS.items():
            for match in pattern.finditer(line):
                value = match.group(0)
                if value in ALLOWED_MATCHES:
                    continue
                findings.append(
                    {
                        "file": archive_name,
                        "line": lineno,
                        "pattern": pattern_name,
                        "match": value,
                    }
                )
    return findings


_PDF_INFO_ENTRY = re.compile(
    rb"/(?P<key>Author|Creator|Producer)\s*(?:\((?P<literal>(?:[^()\\]|\\.)*)\)|<(?P<hex>[0-9A-Fa-f\s]+)>)"
)


def _decode_pdf_string(value: bytes) -> str:
    if value.startswith(b"\xfe\xff"):
        return value[2:].decode("utf-16-be", errors="ignore")
    if value.startswith(b"\xff\xfe"):
        return value[2:].decode("utf-16-le", errors="ignore")
    return value.decode("latin-1", errors="ignore")


def scan_pdf_metadata(entry: PackageEntry) -> list[dict[str, object]]:
    """Check the plain-text PDF info dictionary; binary streams are not parsed."""
    findings: list[dict[str, object]] = []
    raw = entry.read_bytes()
    for match in _PDF_INFO_ENTRY.finditer(raw):
        key = match.group("key").decode("ascii")
        if key not in PDF_METADATA_KEYS:
            continue
        if match.group("hex") is not None:
            digits = re.sub(rb"\s+", b"", match.group("hex"))
            try:
                value = _decode_pdf_string(bytes.fromhex(digits.decode("ascii")))
            except ValueError:
                continue
        else:
            value = _decode_pdf_string(match.group("literal").replace(b"\\(", b"(").replace(b"\\)", b")"))
        value = value.strip()
        if not value or value.lower().startswith("anonymous"):
            continue
        findings.append(
            {
                "file": entry.archive_name,
                "line": None,
                "pattern": f"pdf_{key.lower()}",
                "match": value,
            }
        )
    return findings


def scan_identity(entries: list[PackageEntry]) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for entry in entries:
        if entry.archive_name.lower().endswith(".pdf"):
            findings.extend(scan_pdf_metadata(entry))
            continue
        if not is_text_archive_name(entry.archive_name):
            continue
        text = entry.read_bytes().decode("utf-8", errors="ignore")
        findings.extend(scan_text(text, entry.archive_name))
    return findings


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def zip_package(entries: list[PackageEntry], zip_path: Path, manifest: dict[str, object]) -> tuple[int, Path]:
    if zip_path.exists():
        try:
            zip_path.chmod(0o666)
            zip_path.unlink()
        except PermissionError:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_path = zip_path.with_name(f"{zip_path.stem}_{stamp}{zip_path.suffix}")
            manifest["zip_file"] = zip_path.name
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in entries:
            archive.writestr(entry.archive_name, entry.read_bytes())
        archive.writestr(
            "PACKAGE_MANIFEST.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
    return len(entries) + 1, zip_path


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    zip_path = output_dir / f"{args.name}.zip"

    entries = collect_entries(include_pdf=not args.no_pdf)
    checkpoint = specialist_snapshot(entries, args.require_complete_specialist)
    findings = scan_identity(entries)
    manifest = {
        "package_name": args.name,
        "zip_file": f"{args.name}.zip",
        "included_files": [entry.archive_name for entry in entries],
        "included_file_count": len(entries),
        "source": "working-tree snapshot at collection time, including untracked but non-ignored files; not a git archive. See identity_scan_scope for what the anonymity check covers.",
        "identity_scan_scope": {
            "text_suffixes": sorted(TEXT_SUFFIXES),
            "text_patterns": sorted(IDENTITY_PATTERNS),
            "pdf_metadata_keys": list(PDF_METADATA_KEYS),
            "binary_payloads": "not decoded; only PDF info dictionaries are inspected",
        },
        "sha256": {entry.archive_name: hashlib.sha256(entry.read_bytes()).hexdigest() for entry in entries},
        "specialist_checkpoint": checkpoint,
        "diagnostic_redactions": {entry.archive_name: {"original_sha256": entry.original_sha256,
            "exported_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
            "change": "Local path prefixes removed from diagnostic text only; original files unchanged"}
            for entry in entries if entry.original_sha256},
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    findings.extend(scan_text(manifest_text, "PACKAGE_MANIFEST.json"))
    manifest["identity_findings"] = findings
    manifest["identity_findings_count"] = len(findings)
    if findings:
        write_json(output_dir / f"{args.name}_identity_scan.json", findings)
        print(json.dumps({"identity_findings_count": len(findings), "zip_written": False}))
        return 1

    zipped_count, zip_path = zip_package(entries, zip_path, manifest)
    write_json(output_dir / f"{args.name}_manifest.json", manifest)
    write_json(output_dir / f"{args.name}_identity_scan.json", findings)

    result = {
        "identity_findings_count": len(findings),
        "manifest": str(output_dir / f"{args.name}_manifest.json"),
        "zip_file_count": zipped_count,
        "zip_path": str(zip_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
