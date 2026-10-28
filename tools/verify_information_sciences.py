"""Read-only consistency gate for the Information Sciences manuscript/artifact.

Replay retained studies, check presentation/interpretation provenance and exact
reported records, then validate the current main and supplementary sources.
There is no hard-coded supplement hash, experiment run, file write or refit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
from pathlib import Path, PureWindowsPath
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from validate_paper_static import clean_source, collect_sources, group_end, validate

FIGURES = {"fig_target_example", "fig_confirmation", "fig_schedule", "fig_gate"}
TABLES = {"tab_protocol", "tab_main_results", "tab_factors", "tab_ess_compact"}
MAIN_FIGURES = {"fig_target_example", "fig_confirmation", "fig_fallback", "fig_symmetric", "fig_boundary_sensitivity"}
MAIN_TABLES = {"tab_protocol", "tab_study_a_combined", "tab_external_framework"}
SUPPLEMENT_OUTPUTS = {
    "tab_supp_a_contrasts.tex", "tab_supp_b_gate_factors.tex", "tab_supp_diagnostics_ols.tex",
    "supp_a_contrasts_source.csv", "supp_b_gate_source.csv", "supp_b_factors_source.csv",
    "supp_fit_diagnostics_source.csv", "supp_ols_absolute_source.csv",
}
GENERATOR_AUDIT_OUTPUTS = {
    "mechanisms.csv", "observed_trajectories.csv", "study_summary.csv", "tab_generator_diagnostics.tex",
    "frozen_before_sha256.json", "frozen_after_sha256.json", "frozen_sources_before_sha256.json",
    "frozen_sources_after_sha256.json", "diagnostic_summary.json",
}
PROVENANCE_EXPORTS = {
    "A_freeze_stage.json", "A_completion_stage.json", "A_run_stdout.txt", "A_verify_report.txt",
    "B_execution_stage.json", "B_run_stdout.txt", "B_run_stderr.txt", "B_verify_report.json",
}
AUTHOR_NAME = "Xuansong Chang"
AUTHOR_AFFILIATION = "Stony Brook Institute at Anhui University"
AUTHOR_EMAIL = "alec.xs.chang@gmail.com"


class VerificationError(ValueError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(base: Path, name: str) -> Path:
    base = base.resolve()
    relative = Path(name)
    windows = PureWindowsPath(name)
    if relative.is_absolute() or relative.drive or windows.drive or ".." in relative.parts or ".." in windows.parts:
        raise VerificationError(f"Invalid manifest/dependency path: {name}")
    target = (base / relative).resolve()
    if not target.is_relative_to(base):
        raise VerificationError(f"Path escapes its root: {name}")
    return target


def check_hashes(entries: dict, base: Path, label: str) -> int:
    if not isinstance(entries, dict) or not entries:
        raise VerificationError(f"Missing hash mapping: {label}")
    for name, expected in entries.items():
        path = inside(base, name)
        if not path.is_file() or sha(path) != expected:
            raise VerificationError(f"{label} checksum mismatch: {name}")
    return len(entries)


def verify_manifests(root: Path) -> dict:
    output = root / "results/information_sciences"
    assets = json.loads((output / "assets_manifest.json").read_text(encoding="utf-8"))
    interpretation = json.loads((output / "interpretation_manifest.json").read_text(encoding="utf-8"))
    counts = {"asset_sources": check_hashes(assets["source_sha256"], root, "Asset source"),
              "asset_outputs": check_hashes(assets["output_sha256"], output, "Asset output"),
              "interpretation_sources": check_hashes(interpretation["source_sha256"], root, "Interpretation source"),
              "interpretation_outputs": check_hashes(interpretation["output_sha256"], output, "Interpretation output")}
    if sha(root / "tools/build_information_sciences_assets.py") != assets["asset_builder_sha256"]:
        raise VerificationError("Presentation builder changed since assets were generated")
    if assets.get("main_figure_count") != 4 or assets.get("main_table_count") != 4:
        raise VerificationError("The main asset set must contain exactly four figures and four tables")
    if {Path(row["pdf"]).stem for row in assets["figures"]} != FIGURES:
        raise VerificationError("Unexpected main figure inventory")
    if not all(f"{name}.tex" in assets["output_sha256"] for name in TABLES):
        raise VerificationError("Incomplete four-table asset inventory")
    if assets.get("experiments_fitted") != 0 or not assets.get("retained_source_hashes_unchanged"):
        raise VerificationError("Asset provenance must preserve the frozen study outputs")
    counts.update(verify_supplement_manifest(root))
    return counts


def verify_supplement_manifest(root: Path) -> dict:
    """Require the complete three-table/five-CSV closure, without a fixed Sup hash."""
    output = root / "results/information_sciences"
    manifest = json.loads((output / "supplement_assets_manifest.json").read_text(encoding="utf-8"))
    if set(manifest["sha256"]) != SUPPLEMENT_OUTPUTS:
        raise VerificationError("Supplement asset inventory must retain three tables and five source CSVs")
    sources = check_hashes(manifest["frozen_source_sha256"], root, "Supplement frozen source")
    outputs = check_hashes(manifest["sha256"], output, "Supplement output")
    return {"supplement_frozen_sources": sources, "supplement_outputs": outputs}


def verify_generator_audit(root: Path) -> dict:
    """Recompute retained-X diagnostics only; never call the audit writer or simulator."""
    output = root / "results/information_sciences/generator_audit"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest["sha256"]) != GENERATOR_AUDIT_OUTPUTS:
        raise VerificationError("Generator audit must retain all nine diagnostic outputs")
    if sha(root / "tools/audit_generator_outputs.py") != manifest["audit_builder_sha256"]:
        raise VerificationError("Generator audit builder changed since diagnostics were generated")
    source_count = check_hashes(manifest["frozen_source_sha256"], root, "Generator frozen source")
    output_count = check_hashes(manifest["sha256"], output, "Generator audit output")
    audit = importlib.import_module("audit_generator_outputs")
    for stem, current in (("frozen", audit.frozen_hashes(root)),
                          ("frozen_sources", audit.frozen_source_hashes(root))):
        before = json.loads((output / f"{stem}_before_sha256.json").read_text(encoding="utf-8"))
        after = json.loads((output / f"{stem}_after_sha256.json").read_text(encoding="utf-8"))
        if before != after or after != current:
            raise VerificationError(f"Generator {stem} before/after snapshots differ from retained files")
        check_hashes(after, root, f"Generator {stem} snapshot")
    if manifest["frozen_source_sha256"] != audit.frozen_source_hashes(root):
        raise VerificationError("Generator frozen-source inventory is incomplete")
    if (manifest.get("frozen_files_unchanged") != len(audit.frozen_hashes(root))
            or manifest.get("frozen_source_files_unchanged") != source_count
            or manifest.get("before_after_frozen_hashes_equal") is not True
            or manifest.get("before_after_frozen_source_hashes_equal") is not True):
        raise VerificationError("Generator audit preservation counts/flags are inconsistent")
    audit.check_declared_sources(root)
    mechanisms, trajectories = audit.collect(root)
    summary = audit.summaries(mechanisms, trajectories)
    for name, expected in (("mechanisms.csv", mechanisms), ("observed_trajectories.csv", trajectories),
                           ("study_summary.csv", summary)):
        pd.testing.assert_frame_equal(pd.read_csv(output / name), expected, check_dtype=False,
                                      check_exact=False, atol=1e-12, rtol=1e-12)
    if (output / "tab_generator_diagnostics.tex").read_text(encoding="utf-8").strip() != audit.table(summary).strip():
        raise VerificationError("Generator diagnostic table differs from saved observed trajectories")
    report = json.loads((output / "diagnostic_summary.json").read_text(encoding="utf-8"))
    compare_record_lists(report["study_summary"], summary, ("study",), "Generator summary")
    check_hashes(report["source_sha256"], root, "Generator report source")
    if (report.get("new_simulations") != 0 or report.get("new_fits") != 0
            or report.get("frozen_files_unchanged") != manifest["frozen_files_unchanged"]
            or report.get("frozen_source_files_unchanged") != source_count):
        raise VerificationError("Generator report must preserve the read-only diagnostic scope")
    return {"outputs": output_count, "frozen_sources": source_count,
            "frozen_files": manifest["frozen_files_unchanged"],
            "mechanisms_recomputed": len(mechanisms), "observed_trajectories_recomputed": len(trajectories),
            "summary_rows": len(summary), "tables_reconstructed": 1}


def verify_provenance_exports(root: Path) -> tuple[dict, dict]:
    """Require distributable copies; private original logs are optional local checks."""
    relative = "results/information_sciences/provenance/"
    manifest = json.loads((root / relative / "export_manifest.json").read_text(encoding="utf-8"))
    rows = manifest["exports"]
    names = [row["exported_relative_path"] for row in rows]
    if len(names) != 8 or set(names) != {relative + name for name in PROVENANCE_EXPORTS}:
        raise VerificationError("Provenance export inventory must retain all eight execution records")
    originals_checked = 0
    for row in rows:
        exported = inside(root, row["exported_relative_path"])
        check_hashes({row["exported_relative_path"]: row["exported_sha256"]}, root, "Provenance export")
        if exported.stat().st_size != row["bytes"]:
            raise VerificationError("Provenance exported byte count differs from its manifest")
        original = inside(root, row["original_relative_path"])
        if original.is_file():
            check_hashes({row["original_relative_path"]: row["original_sha256"]}, root, "Optional local provenance original")
            originals_checked += 1
    derived = manifest["derived_chronology"]
    if derived["path"] != relative + "chronology.json":
        raise VerificationError("Unexpected chronology dependency")
    check_hashes({derived["path"]: derived["sha256"]}, root, "Provenance chronology")
    return manifest, {"exported_records": len(rows), "optional_private_originals_checked": originals_checked}


def verify_chronology(root: Path, chronology: dict, exported_paths: set[str]) -> int:
    expected = set()
    sources = {}
    for name in exported_paths:
        if Path(name).name not in {"A_freeze_stage.json", "A_completion_stage.json", "B_execution_stage.json"}:
            continue
        sources[name] = json.loads(inside(root, name).read_text(encoding="utf-8"))
        expected.update((name, field) for field, value in sources[name].items()
                        if field.endswith("_utc") and isinstance(value, str))
    if chronology.get("timezone") != "Asia/Shanghai":
        raise VerificationError("Chronology timezone differs from its declared conversion")
    seen = set()
    for event in chronology["events"]:
        identity = (event["source"], event["field"])
        if identity not in expected or identity in seen:
            raise VerificationError("Unknown/duplicate chronology source field")
        seen.add(identity)
        if event["utc"] != sources[identity[0]][identity[1]]:
            raise VerificationError("Chronology UTC differs from its retained source field")
        utc = datetime.fromisoformat(event["utc"])
        local = datetime.fromisoformat(event["asia_shanghai"])
        if (utc.tzinfo is None or local.tzinfo is None or utc != local
                or local.utcoffset() != timedelta(hours=8)
                or event["local_date"] != utc.astimezone(timezone(timedelta(hours=8))).date().isoformat()):
            raise VerificationError("Chronology local conversion differs from its retained UTC")
    if seen != expected:
        raise VerificationError("Chronology omits a retained source clock field")
    return len(seen)


def _template_paths(template: str, seeds: list[int]) -> list[str]:
    paths = [template.replace("{seed}", str(seed)) for seed in seeds]
    while any(re.search(r"\{[^{}]+\}", name) for name in paths):
        expanded = []
        for name in paths:
            match = re.search(r"\{([^{}]+)\}", name)
            if not match:
                expanded.append(name)
            else:
                choices = match[1].split("|")
                if len(choices) < 2:
                    raise VerificationError("Unknown evidence-index record template placeholder")
                expanded.extend(name[:match.start()] + value + name[match.end():] for value in choices)
        paths = expanded
    if any("{" in name or "}" in name for name in paths):
        raise VerificationError("Malformed evidence-index record template")
    return paths


def verify_provenance(root: Path) -> dict:
    output = root / "results/information_sciences/provenance"
    index = json.loads((output / "evidence_index.json").read_text(encoding="utf-8"))
    exports, counts = verify_provenance_exports(root)
    counts["frozen_sources"] = check_hashes(index["frozen_source_sha256"], root, "Evidence-index frozen source")
    evidence = index["protocol_and_evidence_sha256"]
    counts["protocol_and_evidence_files"] = check_hashes(evidence, root, "Evidence-index protocol/evidence")
    exported_paths = {row["exported_relative_path"] for row in exports["exports"]}
    for row in exports["exports"]:
        if evidence.get(row["exported_relative_path"]) != row["exported_sha256"]:
            raise VerificationError("Evidence index omits or changes an exported execution record")
    for name in ("export_manifest.json", "chronology.json"):
        path = (output / name).relative_to(root).as_posix()
        if path not in evidence:
            raise VerificationError("Evidence index omits a provenance manifest dependency")
    chronology = json.loads((output / "chronology.json").read_text(encoding="utf-8"))
    counts["chronology_source_fields"] = verify_chronology(root, chronology, exported_paths)
    if set(index["studies"]) != {"A", "B"}:
        raise VerificationError("Evidence index must preserve both paired studies")
    record_paths = set()
    for study, description in index["studies"].items():
        directory = inside(root, description["output_directory"])
        for field in ("protocol", "runner", "read_only_verifier", "seal", "protocol_copy"):
            if field not in description:
                if field == "protocol_copy" and study == "B":
                    continue
                raise VerificationError(f"Evidence index omits {study}/{field}")
            name = description[field]
            if not inside(root, name).is_file() or name not in {**evidence, **index["frozen_source_sha256"]}:
                raise VerificationError(f"Evidence-index dependency is not packaged and hash-bound: {name}")
        protocol = json.loads(inside(root, description["protocol"]).read_text(encoding="utf-8"))
        seeds = protocol["repeat_seeds"] if study == "A" else protocol["repeats"]
        if description["seeds"] != {"count": len(seeds), "first": min(seeds), "last": max(seeds)}:
            raise VerificationError("Evidence-index seed range differs from its frozen protocol")
        seal = json.loads(inside(root, description["seal"]).read_text(encoding="utf-8"))
        check_hashes(seal["sha256"], directory, f"Indexed {study} seal")
        for name in description["summaries"]:
            if name not in seal["sha256"] or not inside(directory, name).is_file():
                raise VerificationError("Evidence-index summary is absent from the study seal")
        if len(description["record_templates"]) != (5 if study == "A" else 4):
            raise VerificationError("Evidence-index record-template inventory is incomplete")
        for template in description["record_templates"]:
            for name in _template_paths(template, seeds):
                path = inside(directory, name)
                if not path.is_file() or name not in seal["sha256"]:
                    raise VerificationError(f"Indexed repeat record is not packaged and sealed: {name}")
                record_paths.add(path)
    supplement = expand_inputs((root / "paper/supplement.tex").read_text(encoding="utf-8"), root)
    labels = set(re.findall(r"\\label\s*\{([^{}]+)\}", supplement))
    locators = index["supplement_locators"]
    if any(row["label"] not in labels for row in locators):
        raise VerificationError("An indexed supplement source label is missing")
    counts.update({"indexed_studies": 2, "indexed_repeat_records": len(record_paths), "supplement_source_labels": len(locators)})
    return counts


def verify_bundle(root: Path, *, skip_bundle: bool = False) -> int:
    """Exported bytes are checked before a rebuild; scientific checks always run."""
    manifest = root / "PACKAGE_MANIFEST.json"
    if skip_bundle or not manifest.is_file():
        return 0
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    return check_hashes(entries["sha256"], root, "Exported bundle")


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))


def _key_value(value):
    return None if _is_missing(value) else value.item() if isinstance(value, np.generic) else value


def compare_record_lists(records: list[dict], expected: pd.DataFrame, keys: tuple[str, ...], label: str) -> int:
    """Check every stored field, including tiny p values and missing auxiliary p."""
    if len(records) != len(expected):
        raise VerificationError(f"{label} record count differs from the retained CSV")
    lookup = {}
    for row in expected.to_dict("records"):
        identity = tuple(_key_value(row[key]) for key in keys)
        if identity in lookup:
            raise VerificationError(f"Duplicate retained {label} identity: {identity}")
        lookup[identity] = row
    seen = set()
    for row in records:
        if set(row) != set(expected.columns):
            raise VerificationError(f"{label} fields differ from the retained CSV")
        identity = tuple(_key_value(row[key]) for key in keys)
        if identity not in lookup or identity in seen:
            raise VerificationError(f"Unknown/duplicate {label} record: {identity}")
        seen.add(identity)
        for field, target in lookup[identity].items():
            actual = row[field]
            if _is_missing(target) or _is_missing(actual):
                correct = _is_missing(target) and _is_missing(actual)
            elif isinstance(target, (bool, np.bool_, str)):
                correct = actual == target
            else:
                # Absolute tolerances must not accept zero for a tiny p value.
                absolute = 0 if field.startswith("p_") else 1e-14
                correct = bool(np.isclose(float(actual), float(target), rtol=1e-12, atol=absolute))
            if not correct:
                raise VerificationError(f"{label} numeric/text drift at {identity}/{field}: {actual} != {target}")
    return len(seen)


def verify_interpretation(root: Path) -> dict:
    output = root / "results/information_sciences"
    values = json.loads((output / "interpretation_numbers.json").read_text(encoding="utf-8"))
    if set(values) != {"A", "B"}:
        raise VerificationError("Interpretation numbers must preserve separate A and B evidence")
    a = pd.read_csv(output / "duration_confirmation/contrasts.csv")
    b = pd.read_csv(output / "schedule_robustness/schedule_robustness_secondary_tests.csv")
    counts = {"A_records": compare_record_lists(values["A"], a, ("scorer", "contrast"), "A interpretation"),
              "B_records": compare_record_lists(values["B"], b, ("family", "schedule", "duration_mean"), "B interpretation")}
    if int(a.primary.sum()) != 4 or not (a.loc[a.primary, "scorer"] == "lasso").all():
        raise VerificationError("A primary family must retain its four Lasso-only comparisons")
    if b.groupby("family").size().to_dict() != {"B_duration_gamma": 3, "B_forced_minus_raw": 6}:
        raise VerificationError("B secondary families must retain six contrasts and three paired duration differences")
    return counts


def expand_inputs(text: str, root: Path, stack=()) -> str:
    """Expand TeX inputs for semantic counts, keeping dependencies within root."""
    pattern = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")
    def substitute(match):
        name = match[1]
        if not Path(name).suffix:
            name += ".tex"
        path = inside(root, name)
        if path in stack:
            raise VerificationError("Cyclic TeX input while checking manuscript")
        if not path.is_file():
            raise VerificationError(f"Missing manuscript input: {name}")
        return expand_inputs(clean_source(path.read_text(encoding="utf-8")), root, (*stack, path))
    return pattern.sub(substitute, clean_source(text))


def abstract_words(text: str) -> int:
    text = re.sub(r"\$[^$]*\$", " VALUE ", clean_source(text))
    text = re.sub(r"\\[A-Za-z]+\*?", " ", text)
    return len(text.replace("{", " ").replace("}", " ").split())


def verify_manuscript(root: Path) -> dict:
    path = root / "paper/main.tex"
    source = clean_source(path.read_text(encoding="utf-8"))
    expanded = expand_inputs(source, root, (path.resolve(),))
    figures = len(re.findall(r"\\begin\s*\{figure\*?\}", expanded))
    tables = len(re.findall(r"\\begin\s*\{table\*?\}", expanded))
    if (figures, tables) != (len(MAIN_FIGURES), len(MAIN_TABLES)):
        raise VerificationError(f"Expected main {len(MAIN_FIGURES)} figures/{len(MAIN_TABLES)} tables, found {figures}/{tables}")
    graphics = {Path(name).stem for name in re.findall(
        r"\\includegraphics(?:\s*\[[^]]*\])*\s*\{([^{}]+)\}", expanded)}
    if graphics != MAIN_FIGURES:
        raise VerificationError("Main figure references differ from the verified asset inventory")
    table_refs = {Path(name).stem for name in re.findall(r"\\input\s*\{([^{}]+)\}", source)
                  if Path(name).stem.startswith("tab_")}
    if table_refs != MAIN_TABLES:
        raise VerificationError("Main table references differ from the verified asset inventory")
    match = re.search(r"\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}", expanded)
    if not match:
        raise VerificationError("The manuscript abstract could not be found after input expansion")
    count = abstract_words(match[1])
    if not 180 <= count <= 195:
        raise VerificationError(f"Abstract has {count} words; expected the agreed 180--195 range")
    introduction = re.search(r"\\section\{Introduction\}([\s\S]*?)\\section\{", source)
    items = re.search(r"\\begin\{itemize\}([\s\S]*?)\\end\{itemize\}", introduction[1] if introduction else "")
    contributions = len(re.findall(r"\\item\b", items[1] if items else ""))
    if contributions != 3:
        raise VerificationError("The introduction must retain the three substantive contributions")
    author = re.search(r"\\author\s*", source)
    end = group_end(source, author.end()) if author else None
    author_text = source[author.end():end] if end else ""
    if not all(value in author_text for value in (AUTHOR_NAME, AUTHOR_AFFILIATION, AUTHOR_EMAIL)):
        raise VerificationError("The current main author block must match the supplied real author details")
    if re.search(r"anonymous|placeholder", author_text, re.I):
        raise VerificationError("Anonymous/placeholder authors are not the current submission identity")
    return {"main_figures": figures, "main_tables": tables, "abstract_words": count,
            "contributions": contributions, "real_author_details": True}


def verify_asset_numbers(root: Path) -> dict:
    """Reconstruct presentation data/tables in memory; never call build/export."""
    builder = importlib.import_module("build_information_sciences_assets")
    output = root / "results/information_sciences"
    data = {
        "a_detail": pd.read_csv(output / "duration_confirmation/detail.csv"),
        "a_summary": pd.read_csv(output / "duration_confirmation/summary.csv"),
        "a_tests": pd.read_csv(output / "duration_confirmation/contrasts.csv"),
        "b_detail": pd.read_csv(output / "schedule_robustness/schedule_robustness_detail.csv"),
        "b_diagnostics": pd.read_csv(output / "schedule_robustness/schedule_robustness_diagnostics.csv"),
        "b_paired": pd.read_csv(output / "schedule_robustness/schedule_robustness_paired.csv"),
        "b_tests": pd.read_csv(output / "schedule_robustness/schedule_robustness_secondary_tests.csv"),
        "b_gate": pd.read_csv(output / "schedule_robustness/schedule_robustness_gate_summary.csv"),
        "ess": pd.read_csv(root / "results/tables/ess_alignment.csv"),
    }
    factors = builder.factor_data(data)
    expected_tables = {"tab_protocol.tex": builder.table_protocol(), "tab_main_results.tex": builder.table_main(data),
                       "tab_factors.tex": builder.table_factors(factors), "tab_ess_compact.tex": builder.table_ess(data)}
    for name, expected in expected_tables.items():
        if (output / name).read_text(encoding="utf-8").strip() != expected.strip():
            raise VerificationError(f"Generated table no longer matches retained data: {name}")
    for name, expected in (("factor_table_source.csv", factors), ("fig_schedule_source.csv", builder.b_plot_data(data)),
                           ("fig_gate_source.csv", builder.b_plot_data(data))):
        pd.testing.assert_frame_equal(pd.read_csv(output / name), expected, check_dtype=False,
                                      check_exact=False, atol=1e-12, rtol=0)
    example = pd.read_csv(output / "target_example.csv")
    if len(example) != 32 or example.centre.nunique() != 8 or example.edge.nunique() != 4:
        raise VerificationError("The finite target illustration must retain all 4x8 cells")
    true_states = example.drop_duplicates("centre").sort_values("centre").true_state.to_numpy()
    np.testing.assert_array_equal(true_states, [0] * 6 + [1] * 2)
    truth = example.pivot(index="edge", columns="centre", values="true_active").to_numpy()
    scores = example.pivot(index="edge", columns="centre", values="static_occupancy_score").to_numpy()
    np.testing.assert_array_equal(truth, np.stack([np.ones(8), true_states == 0, true_states == 1, np.zeros(8)]))
    np.testing.assert_array_equal(scores, np.repeat(truth.mean(axis=1, keepdims=True), 8, axis=1))
    from sklearn.metrics import roc_auc_score
    metrics = pd.read_csv(output / "target_example_metrics.csv").iloc[0]
    if metrics.pooled_auroc != roc_auc_score(truth.reshape(-1), scores.reshape(-1)):
        raise VerificationError("Finite illustration pooled AUROC does not match its saved cells")
    if metrics.changing_edge_temporal_auroc != np.mean([roc_auc_score(truth[k], scores[k]) for k in (1, 2)]):
        raise VerificationError("Finite illustration changing-edge AUROC does not match its saved cells")
    return {"tables_reconstructed": 4, "factor_rows": len(factors), "B_display_cells": 6, "finite_example_cells": 32}


def _replay_studies(root: Path) -> dict:
    output = root / "results/information_sciences"
    a_protocol = json.loads((output / "duration_confirmation/protocol.json").read_text(encoding="utf-8"))
    b_protocol = json.loads((output / "schedule_robustness/schedule_robustness_protocol.json").read_text(encoding="utf-8"))
    check_hashes(a_protocol["source_sha256"], root, "Frozen A source")
    check_hashes(b_protocol["source_sha256"], root, "Frozen B source")
    a_module = importlib.import_module("verify_duration_confirmation")
    b_module = importlib.import_module("verify_schedule_robustness")
    return {"A": a_module.verify(output / "duration_confirmation"),
            "B": b_module.verify(output=output / "schedule_robustness", root=root)}


def tracked_snapshot(root: Path) -> dict:
    """Track the actual current source closure and evidence, without fixed Sup hashes."""
    paths = set()
    output = root / "results/information_sciences"
    paths.update(path for path in output.rglob("*") if path.is_file())
    for main in (Path("paper/main.tex"), Path("paper/supplement.tex")):
        closure = collect_sources(root, main)
        paths.update(closure.tex); paths.update(closure.bib); paths.update(closure.assets)
    for manifest_name in ("assets_manifest.json", "interpretation_manifest.json"):
        manifest = json.loads((output / manifest_name).read_text(encoding="utf-8"))
        paths.update(inside(root, name) for name in manifest["source_sha256"])
    supplement_manifest = output / "supplement_assets_manifest.json"
    if supplement_manifest.is_file():
        manifest = json.loads(supplement_manifest.read_text(encoding="utf-8"))
        paths.update(inside(root, name) for name in manifest["frozen_source_sha256"])
    audit_manifest = output / "generator_audit/manifest.json"
    if audit_manifest.is_file():
        manifest = json.loads(audit_manifest.read_text(encoding="utf-8"))
        paths.add(root / "tools/audit_generator_outputs.py")
        paths.update(inside(root, name) for name in manifest["frozen_source_sha256"])
    evidence_index = output / "provenance/evidence_index.json"
    if evidence_index.is_file():
        index = json.loads(evidence_index.read_text(encoding="utf-8"))
        for mapping in ("frozen_source_sha256", "protocol_and_evidence_sha256"):
            paths.update(inside(root, name) for name in index[mapping])
    if (root / "PACKAGE_MANIFEST.json").is_file():
        paths.add(root / "PACKAGE_MANIFEST.json")
    return {path.relative_to(root).as_posix(): sha(path) for path in sorted(paths) if path.is_file()}


def verify(root: Path = ROOT, *, replay: bool = True, skip_bundle: bool = False) -> dict:
    root = Path(root).resolve()
    bundle_count = verify_bundle(root, skip_bundle=skip_bundle)
    before = tracked_snapshot(root)
    manifest_report = verify_manifests(root)
    number_report = verify_interpretation(root)
    manuscript = verify_manuscript(root)
    assets = verify_asset_numbers(root)
    generator_audit = verify_generator_audit(root)
    provenance = verify_provenance(root)
    reviewer = importlib.import_module("verify_reviewer_controls").verify(root=root, replay=replay)
    sensitivity = importlib.import_module("verify_evaluation_sensitivity").verify(root=root, replay=replay)
    narrative = importlib.import_module("build_narrative_assets").verify(
        root=root, output=root / "results/information_sciences/narrative_assets")
    static = {}
    for name in ("main", "supplement"):
        result = validate(root, Path(f"paper/{name}.tex"))
        if result["errors"]:
            raise VerificationError(f"{name} static validation failed: " + "; ".join(result["errors"][:5]))
        static[name] = {"tex_sources": result["tex_file_count"], "assets": result["asset_count"], "errors": 0}
    studies = _replay_studies(root) if replay else None
    after = tracked_snapshot(root)
    if before != after:
        raise VerificationError("Source/evidence files changed during the read-only verification")
    return {"status": "passed", "read_only": True, "files_unchanged": len(before),
            "manifests": manifest_report, "interpretation": number_report, "manuscript": manuscript,
            "presentation": assets, "static": static, "cache_replay_performed": replay,
            "generator_audit": generator_audit, "provenance": provenance,
            "reviewer_controls": reviewer,
            "evaluation_sensitivity": sensitivity,
            "publication_assets": narrative,
            "cache_replay": studies, "new_fits": 0, "bundle_payloads_verified": bundle_count,
            "original_bundle_byte_check_skipped": skip_bundle}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-bundle", action="store_true",
                        help="After the entry point checked original exported bytes, skip only that byte check during a derived rebuild")
    args = parser.parse_args()
    try:
        report = verify(args.root, skip_bundle=args.skip_bundle)
    except (ValueError, AssertionError, KeyError, FileNotFoundError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, separators=(",", ":")))
        return 1
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
