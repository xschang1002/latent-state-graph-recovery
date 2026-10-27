"""Prepare a local, identified Information Sciences reproducibility archive.

This does not deposit data, allocate a DOI, publish a release or submit a paper.
Collection/scan helpers can inspect a draft; the command-line release path
requires completed studies, current article/supplement PDFs and verification.
The earlier anonymous package and its builder are left unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import build_submission_package as legacy

ROOT = SCRIPTS.parent
AUTHOR = "Xuansong Chang"
EMAIL = "alec.xs.chang@gmail.com"
AFFILIATION = "Stony Brook Institute at Anhui University, Hefei, China"
POSTAL_AFFILIATION_LINES = (
    "Stony Brook Institute at Anhui University",
    "Anhui University, 3 Feixi Road, Shushan District,",
    "Hefei, Anhui 230039, China",
)
DEFAULT_NAME = "information_sciences_reproducibility"
MATERIALS = "submission_materials/information_sciences/"
EXTRA_FILES = {
    "paper/supplement.tex",
    "docs/research_guide.md",
    "docs/narrative_cleanup_2026-10-27.md",
    "docs/information_sciences_artifact_readme.md",
    "docs/information_sciences_requirements_2026-10-24.md",
    "docs/information_sciences_related_work_2026-10-24.md",
    "docs/information_sciences_aggregation_literature_2026-10-24.md",
    "docs/information_sciences_evidence_index_2026-10-24.md",
    "docs/reviewer_controls_evidence_index_2026-10-25.md",
    "docs/reviewer_response_2026-10-25.md",
    "analysis/reviewer_revision_plan_2026-10-25.md",
    "analysis/evaluation_revision_plan_2026-10-25.md",
    "docs/evaluation_sensitivity_literature_2026-10-25.md",
    "docs/evaluation_revision_evidence_index_2026-10-25.md",
    "docs/evaluation_revision_response_2026-10-25.md",
}
NUMERICAL_SUFFIXES = {".csv", ".json", ".npz", ".npy", ".pdf", ".png", ".tex", ".bib", ".md", ".txt", ".yaml", ".yml"}
TEXT_SUFFIXES = legacy.TEXT_SUFFIXES | {".cff"}
IDENTITY_PATTERNS = dict(legacy.IDENTITY_PATTERNS)
# A confirmed postal address can exceed the anonymous export's short-block
# limit. Inspect the complete block before applying the exact author allowlist.
IDENTITY_PATTERNS["tex_non_anonymous_author"] = re.compile(r"\\author\{(?!\s*Anonymous\b)[^}]+\}")
PAPER_NAMES = {"paper/main.tex", "paper/supplement.tex", "paper/main.pdf", "paper/supplement.pdf"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def within_root(root: Path, name: str) -> Path:
    root = root.resolve()
    target = (root / name).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError(f"Missing or escaping package source: {name}")
    return target


def extra_candidates(root: Path) -> set[str]:
    """Collect complete new-study evidence even when numerical arrays are ignored."""
    names = set(EXTRA_FILES)
    for prefix in ("results/information_sciences", MATERIALS.rstrip("/"),
                   "paper/sections", "paper/supplement_sections"):
        folder = root / prefix
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if not path.is_file() or any(p in {"__pycache__", ".pytest_cache"} for p in path.parts):
                continue
            if path.suffix.lower() in {".pyc", ".pyo"}:
                continue
            if prefix.startswith("paper/") and path.suffix.lower() != ".tex":
                continue
            if prefix.startswith("results/") and path.suffix.lower() not in NUMERICAL_SUFFIXES:
                raise ValueError(f"Unexpected new-study file type: {path.relative_to(root).as_posix()}")
            names.add(path.relative_to(root).as_posix())
    return names


def collect_entries(include_pdf: bool = True, root: Path = ROOT) -> list[legacy.PackageEntry]:
    """Reuse the established working-tree whitelist and diagnostic redactions.

    include_pdf=False is for inspecting draft collection/privacy only. It is
    not exposed as a way to publish an incomplete archive from the CLI.
    """
    root = root.resolve()
    entries = {}
    for entry in legacy.collect_entries(include_pdf=False, root=root):
        if entry.archive_name in {"README.md", "docs/anonymous_artifact_readme.md"}:
            continue
        if entry.archive_name.startswith("submission_materials/mlj_contribution_information/"):
            continue
        entries[entry.archive_name] = entry
    for name in sorted(extra_candidates(root)):
        path = within_root(root, name)
        entries[name] = legacy.snapshot_entry(path, name)
    readme = within_root(root, "docs/information_sciences_artifact_readme.md")
    entries["README.md"] = legacy.snapshot_entry(readme, "README.md")
    citation = within_root(root, MATERIALS + "CITATION.cff")
    entries["CITATION.cff"] = legacy.snapshot_entry(citation, "CITATION.cff")
    # Published source and data seals bind exact bytes. Repository checkouts
    # must not normalise the mixed line endings in these saved inputs.
    attributes = within_root(root, MATERIALS + "repository.gitattributes")
    entries[".gitattributes"] = legacy.snapshot_entry(attributes, ".gitattributes")
    if include_pdf:
        for name in ("paper/main.pdf", "paper/supplement.pdf"):
            entries[name] = legacy.snapshot_entry(within_root(root, name), name)
    return [entries[name] for name in sorted(entries)]


def author_context(name: str) -> bool:
    return (name in PAPER_NAMES or name in {"README.md", "CITATION.cff", "PACKAGE_MANIFEST.json",
            "docs/information_sciences_artifact_readme.md",
            "docs/information_sciences_requirements_2026-10-24.md"}
            or name.startswith(MATERIALS)
            or (name.startswith(("scripts/", "src/", "tools/", "tests/")) and name.endswith(".py")))


def approved_author_match(pattern: str, value: str, name: str) -> bool:
    if not author_context(name):
        return False
    if pattern == "email":
        return value == EMAIL
    if pattern == "tex_non_anonymous_author":
        normal = re.sub(r"\s+", " ", value)
        author_start = chr(92) + "author" + chr(123)
        simple = author_start + AUTHOR + chr(125)
        full = author_start + AUTHOR + "\\\\ \\small " + AFFILIATION + "\\\\ \\small\\texttt{" + EMAIL + "}"
        postal = author_start + AUTHOR + "\\\\ \\small " + "\\\\ \\small ".join(POSTAL_AFFILIATION_LINES) + "\\\\ \\small\\texttt{" + EMAIL + "}"
        return normal in {simple, full, postal}
    return False


def scan_text(text: str, archive_name: str) -> list[dict]:
    # Inspect whole author blocks before line-wise scanning: the canonical title
    # page spans several lines, whereas an unknown co-author must still fail.
    findings = []
    for pattern_name, pattern in IDENTITY_PATTERNS.items():
        for match in pattern.finditer(text):
            value = match.group(0)
            if value in legacy.ALLOWED_MATCHES or approved_author_match(pattern_name, value, archive_name):
                continue
            findings.append({"file": archive_name, "line": text.count("\n", 0, match.start()) + 1,
                             "pattern": pattern_name, "match": value})
    return findings


def scan_pdf_metadata(entry: legacy.PackageEntry) -> list[dict]:
    findings = []
    for match in legacy._PDF_INFO_ENTRY.finditer(entry.read_bytes()):
        key = match.group("key").decode("ascii")
        if match.group("hex") is not None:
            value = legacy._decode_pdf_string(bytes.fromhex(re.sub(rb"\s+", b"", match.group("hex")).decode("ascii")))
        else:
            value = legacy._decode_pdf_string(match.group("literal").replace(b"\\(", b"(").replace(b"\\)", b")"))
        value = value.strip()
        if not value or value.lower().startswith("anonymous"):
            continue
        if key == "Author":
            allowed = entry.archive_name in {"paper/main.pdf", "paper/supplement.pdf"} and value == AUTHOR
        else:
            # Known renderer metadata are not personal identities. An arbitrary
            # author or unrecognised metadata value is not silently accepted.
            allowed = bool(re.fullmatch(r"(?:LaTeX with hyperref|TeX|LaTeX|tectonic(?: [\w.\-]+)?|"
                                        r"XeTeX(?: [\w.\-]+)?|LuaTeX(?: [\w.\-]+)?|pdfTeX(?: [\w.\-]+)?|"
                                        r"xdvipdfmx(?:[\s\w().\-]+)?|Matplotlib v[\w.\-]+(?:, https://matplotlib.org)?/?)(?:;.*)?", value))
        embedded = scan_text(value, entry.archive_name)
        if embedded or not allowed:
            findings.append({"file": entry.archive_name, "line": None, "pattern": "pdf_" + key.lower(), "match": value})
    return findings


def scan_identity(entries: list[legacy.PackageEntry]) -> list[dict]:
    """Single-anonymized author allowance, not a global personal-data exemption."""
    findings = []
    for entry in entries:
        if entry.archive_name.lower().endswith(".pdf"):
            findings.extend(scan_pdf_metadata(entry))
        elif Path(entry.archive_name).suffix.lower() in TEXT_SUFFIXES:
            findings.extend(scan_text(entry.read_bytes().decode("utf-8", errors="replace"), entry.archive_name))
    return findings


def verify_hash_map(folder: Path, mapping: dict[str, str], root: Path) -> None:
    if not mapping:
        raise ValueError(f"An empty integrity map is not a complete study: {folder.name}")
    for name, expected in mapping.items():
        target = (folder / name).resolve()
        if not target.is_relative_to(folder.resolve()) or not target.is_relative_to(root.resolve()) or not target.is_file():
            raise ValueError(f"Invalid retained-output path: {name}")
        if sha(target.read_bytes()) != expected:
            raise ValueError(f"Retained-output checksum mismatch: {name}")


def pdf_dependencies(root: Path, source: Path) -> list[Path]:
    # Follow section inputs recursively so a changed nested table/figure also
    # invalidates an older PDF; fail on missing dependencies before export.
    paths = {within_root(root, "paper/references.bib")}
    pending = [source]
    while pending:
        current = pending.pop()
        if current in paths:
            continue
        paths.add(current)
        if current.suffix != ".tex":
            continue
        for match in re.finditer(r"\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}", current.read_text(encoding="utf-8")):
            pending.append(within_root(root, match.group(1)))
    return sorted(paths)


def check_release_ready(root: Path = ROOT) -> dict:
    root = root.resolve()
    folder_a = root / "results/information_sciences/duration_confirmation"
    a = json.loads(within_root(root, "results/information_sciences/duration_confirmation/COMPLETE.json").read_text())
    if a.get("n_repeats") != 100 or a.get("n_detail_rows") != 2400:
        raise ValueError("Study A is not the completed frozen 100-repeat study")
    verify_hash_map(folder_a, a["sha256"], root)
    folder_b = root / "results/information_sciences/schedule_robustness"
    b = json.loads(within_root(root, "results/information_sciences/schedule_robustness/schedule_robustness_outputs.json").read_text())
    verify_hash_map(folder_b, b["sha256"], root)
    cases = list((folder_b / "parts").glob("repeat_*_D*.json"))
    if len(cases) != 300:
        raise ValueError(f"Study B requires all 300 planned cell records, found {len(cases)}")
    assets = json.loads(within_root(root, "results/information_sciences/assets_manifest.json").read_text())
    verify_hash_map(root, assets["source_sha256"], root)
    verify_hash_map(root / "results/information_sciences", assets["output_sha256"], root)
    for name in ("main", "supplement"):
        source = within_root(root, f"paper/{name}.tex")
        text = source.read_text(encoding="utf-8")
        author_blocks = list(IDENTITY_PATTERNS["tex_non_anonymous_author"].finditer(text))
        if len(author_blocks) != 1 or not approved_author_match(
                "tex_non_anonymous_author", author_blocks[0].group(0), f"paper/{name}.tex"):
            raise ValueError(f"{name} must identify the sole declared author before packaging")
        pdf = within_root(root, f"paper/{name}.pdf")
        if not pdf.read_bytes().startswith(b"%PDF-"):
            raise ValueError(f"Invalid {name} PDF")
        if pdf.stat().st_mtime_ns < max(p.stat().st_mtime_ns for p in pdf_dependencies(root, source)):
            raise ValueError(f"{name} PDF predates its current source/input; compile it before packaging")
    verifier = within_root(root, "tools/verify_information_sciences.py")
    subprocess.run([sys.executable, "-B", str(verifier)], cwd=root, check=True)
    return {"study_a_repeats": 100, "study_b_cell_records": 300, "study_c_cell_records": 300, "main_tables": 5,
            "main_figures": 6, "pdfs": ["paper/main.pdf", "paper/supplement.pdf"],
            "verification": "tools/verify_information_sciences.py completed before collection"}


def repository_metadata(root: Path = ROOT) -> dict:
    metadata = json.loads(within_root(root, MATERIALS + "deposit_metadata.json").read_text(encoding="utf-8"))
    return {"status": metadata["repository_status"], "doi": metadata["doi"],
            "url": metadata["public_url"], "release_url": metadata.get("release_url"),
            "version": metadata["version"]}


def build_manifest(entries: list[legacy.PackageEntry], name: str, checkpoint: dict, ready: dict) -> dict:
    return {
        "package_name": name, "zip_file": name + ".zip", "created_utc": datetime.now(timezone.utc).isoformat(),
        "article_title": "State Aggregation in Dynamic Graph Recovery: Separating Structural Ranking from Temporal Discrimination",
        "author": AUTHOR, "correspondence": EMAIL, "review_model": "single-anonymized",
        "repository": repository_metadata(),
        "source": "Working-tree snapshot including approved retained numerical outputs; not a git archive or public deposit",
        "included_files": [e.archive_name for e in entries], "included_file_count": len(entries),
        "sha256": {e.archive_name: sha(e.read_bytes()) for e in entries},
        "specialist_checkpoint": checkpoint, "release_checks": ready,
        "diagnostic_redactions": {e.archive_name: {"original_sha256": e.original_sha256,
                                  "exported_sha256": sha(e.read_bytes()),
                                  "change": "Machine path prefixes removed from approved diagnostic exports only"}
                                 for e in entries if e.original_sha256},
        "privacy_scan_scope": {"text_suffixes": sorted(TEXT_SUFFIXES), "patterns": sorted(legacy.IDENTITY_PATTERNS),
                               "declared_author_exception": "Only supplied author/correspondence in canonical paper, source and submission contexts",
                               "pdf_metadata": list(legacy.PDF_METADATA_KEYS),
                               "binary_limit": "PDF info dictionaries inspected; other binary streams are not a general personal-data scan. Numerical arrays are validated by study verifiers."},
        "privacy_findings_count": 0,
    }


def write_archive(entries: list[legacy.PackageEntry], manifest: dict, target: Path, overwrite: bool = False) -> None:
    if "anonymous" in target.stem.lower():
        raise ValueError("The earlier anonymous artifact is protected")
    if target.exists() and not overwrite:
        raise FileExistsError("The new archive already exists; use a new name or explicitly --overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=target.stem + "_", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for entry in entries:
                current = legacy.snapshot_entry(entry.source, entry.archive_name)
                if current.read_bytes() != entry.read_bytes() or current.original_sha256 != entry.original_sha256:
                    raise RuntimeError(f"Source changed after collection: {entry.archive_name}")
                archive.writestr(entry.archive_name, entry.read_bytes())
            archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("New archive failed its ZIP integrity check")
            for name, expected in manifest["sha256"].items():
                if sha(archive.read(name)) != expected:
                    raise RuntimeError(f"New archive payload hash mismatch: {name}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submission_package")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--overwrite", action="store_true", help="Replace this identified archive only; never the old anonymous archive")
    parser.add_argument("--check", action="store_true", help="Validate complete release and privacy without writing an archive")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name) or "anonymous" in args.name.lower():
        parser.error("Use a simple identified archive name; anonymous artifacts are protected")
    target = args.output_dir.resolve() / (args.name + ".zip")
    protected = ROOT / "submission_package/stcg_anonymous_artifact.zip"
    if target.resolve() == protected.resolve():
        raise ValueError("The earlier anonymous archive is protected")
    ready = check_release_ready()
    entries = collect_entries(include_pdf=True)
    checkpoint = legacy.specialist_snapshot(entries, require_complete=True)
    if checkpoint is None or checkpoint["included_job_records"] != 1400:
        raise ValueError("The upload-ready archive must retain all 1,400 frozen specialist records")
    findings = scan_identity(entries)
    manifest = build_manifest(entries, args.name, checkpoint, ready)
    findings.extend(scan_text(json.dumps(manifest, indent=2), "PACKAGE_MANIFEST.json"))
    if findings:
        print(json.dumps({"archive_written": False, "privacy_findings": findings}, indent=2))
        return 1
    if args.check:
        print(json.dumps({"archive_written": False, "complete_payloads": len(entries), "privacy_findings_count": 0}, indent=2))
        return 0
    write_archive(entries, manifest, target, overwrite=args.overwrite)
    legacy.write_json(target.with_name(args.name + "_manifest.json"), manifest)
    legacy.write_json(target.with_name(args.name + "_privacy_scan.json"), findings)
    print(json.dumps({"zip_path": str(target), "payloads": len(entries), "zip_members": len(entries) + 1,
                      "privacy_findings_count": 0, "published": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
