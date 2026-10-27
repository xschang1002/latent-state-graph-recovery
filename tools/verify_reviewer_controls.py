"""Pure read-only gate for all post-review analyses and their presentation.

Missing modules, outputs or planned cases fail closed. This never invokes a
runner, fit, writer or simulator. replay=False omits the expensive retained-array
replays only; manifests, coverage, formula contracts and displayed sources still
must pass. Original A/B repeats are not re-fitted or redundantly replayed here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PureWindowsPath
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("results/information_sciences")
CORE_METHODS = {"raw_window", "native_hard", "forced_hard", "oracle_label", "static_mean"}
CONTROL_METHODS = {"raw_window", "forced_hard", "native_hard", "raw_fallback",
    "shrink_l025", "shrink_l050", "shrink_l075", "shrink_l100", "residual_hard_l000",
    "residual_hard_l025", "residual_hard_l050", "residual_hard_l075", "residual_hard_l100"}
LOCAL_METHODS = CONTROL_METHODS | {"oracle_label", "static_mean"}
LOCAL_METRICS = {"boundary_positive_change_ap", "boundary_slot_f1", "stable_false_change_rate",
                 "stable_any_change_edge_fraction", "persistent_false_change_rate", "absent_false_change_rate"}
CASE_FIELDS = ["study", "scorer", "repeat_seed", "duration", "cell"]
GROUP_FIELDS = ["study", "scorer", "duration", "cell"]
SPECS = (
    ("score_preserving_controls", "scripts/analyze_score_preserving_controls.py", "controls", False),
    ("temporal_localisation", "scripts/analyze_temporal_localisation.py", "localisation", False),
    ("symmetric_sensitivity", "scripts/run_symmetric_sensitivity.py", "core", True),
    ("symmetric_controls", "scripts/analyze_symmetric_controls.py", "controls", True),
    ("symmetric_localisation", "scripts/analyze_symmetric_localisation.py", "localisation", True),
)
PROVENANCE_EXPORTS = {
    "score_controls_execution.json", "score_controls_amendment.json",
    "score_controls_amended_execution.json", "score_controls_completion_summary.json",
    "symmetric_controls_execution.json", "temporal_localisation_freeze.json",
    "temporal_localisation_run.json", "symmetric_localisation_freeze.json",
    "symmetric_localisation_run.json", "symmetric_localisation_QA_amendment.json",
    "symmetric_localisation_QA_numeric_invariance.json",
    "initial_v1__outputs__protocol.json", "initial_v1__outputs__manifest.json",
    "initial_v1__scripts__analyze_score_preserving_controls.py.txt",
    "temporal_localisation_C_initial_v1__symmetric_localisation__protocol.json",
    "temporal_localisation_C_initial_v1__symmetric_localisation__manifest.json",
    "temporal_localisation_C_initial_v1__analyze_symmetric_localisation.py.txt",
}
JSON_NORMALIZATION = "JSON canonicalization and workspace-prefix redaction; code snapshot unchanged as text"
CODE_NORMALIZATION = "unchanged code bytes; .txt suffix for historical QA source"


class VerificationError(ValueError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_target(base: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise VerificationError("Empty/non-text dependency path")
    base = Path(base).resolve(); part = Path(relative); windows = PureWindowsPath(relative)
    if part.is_absolute() or part.drive or windows.drive or ".." in part.parts or ".." in windows.parts:
        raise VerificationError(f"Unsafe dependency path: {relative}")
    target = (base / part).resolve()
    if not target.is_relative_to(base):
        raise VerificationError(f"Dependency escapes root: {relative}")
    return target


def inside(base: Path, relative: str) -> Path:
    target = relative_target(base, relative)
    if not target.is_file():
        raise VerificationError(f"Required file missing: {relative}")
    return target


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationError(f"Expected JSON object: {path.name}")
    return value


def check_hashes(base: Path, mapping: dict, label: str) -> set[Path]:
    if not isinstance(mapping, dict) or not mapping:
        raise VerificationError(f"Empty/missing hash mapping: {label}")
    paths = set()
    for name, expected in mapping.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise VerificationError(f"Invalid digest: {label}/{name}")
        path = inside(base, name)
        if sha(path) != expected:
            raise VerificationError(f"Checksum mismatch: {label}/{name}")
        paths.add(path)
    return paths


def _root_maps(value, root: Path, closure: set[Path]) -> None:
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        if isinstance(item, dict):
            if key.endswith("sha256") and item and all(isinstance(v, str) for v in item.values()):
                closure.update(check_hashes(root, item, key))
            else:
                _root_maps(item, root, closure)


def check_manifest(root: Path, relative: str, manifest_name="manifest.json", excludes=()) -> set[Path]:
    output = root / relative
    manifest_path = inside(output, manifest_name); manifest = read_json(manifest_path)
    closure = check_hashes(output, manifest.get("sha256"), relative)
    existing = {p.resolve() for p in output.rglob("*") if p.is_file()}
    expected = closure | {manifest_path} | {inside(output, name) for name in excludes}
    if existing != expected:
        raise VerificationError(f"Output inventory drift: {relative}")
    for field in ("source_sha256", "analysis_source_sha256"):
        if field in manifest:
            closure.update(check_hashes(root, manifest[field], relative + "/" + field))
    protocol_name = "symmetric_sensitivity_protocol.json" if manifest_name != "manifest.json" else "protocol.json"
    if "protocol_sha256" in manifest:
        protocol_file = inside(output, protocol_name)
        if sha(protocol_file) != manifest["protocol_sha256"]:
            raise VerificationError(f"Protocol digest mismatch: {relative}")
    for name in (protocol_name, "input_manifest.json", "freeze_record.json"):
        path = output / name
        if path.is_file():
            _root_maps(read_json(path), root, closure)
            closure.add(path.resolve())
    if manifest_name == "symmetric_sensitivity_outputs.json":
        protocol = read_json(inside(output, protocol_name))
        config = inside(root, "configs/symmetric_sensitivity.yaml")
        if sha(config) != protocol.get("configuration_sha256"):
            raise VerificationError("Study C configuration checksum mismatch")
        closure.add(config)
    closure.update(existing)
    return closure


def resolve_module(root: Path, relative: str):
    path = inside(root, relative)
    name = "_reviewer_gate_" + path.stem + "_" + hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:10]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise VerificationError(f"Cannot resolve required module: {relative}")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise VerificationError(f"Module resolved outside requested root: {relative}")
    return module


def check_control_contract(module) -> None:
    """Small independent mathematical contracts, no model fits or simulations."""
    raw = np.zeros((5, 1, 2, 2)); raw[:, 0, 0, 1] = [.1, .3, .2, .6, .4]
    raw[:, 0, 1, 0] = [.6, .2, .4, .1, .3]
    hard = raw[::-1].copy(); native = np.repeat(raw.max(axis=0)[None], 5, axis=0)
    truth = np.array([0, 1, 0, 1, 1])
    for accepted in (False, True):
        emitted = module.build_control_emissions(raw, hard, native, accepted)
        if set(emitted) != CONTROL_METHODS:
            raise VerificationError("Incomplete fixed score-control methods")
        np.testing.assert_array_equal(emitted["raw_fallback"], hard if accepted else raw,
                                      err_msg="raw-fallback contract")
        for name in ("raw_window", "shrink_l100", "residual_hard_l100"):
            np.testing.assert_array_equal(emitted[name], raw, err_msg="raw endpoint contract")
        np.testing.assert_array_equal(emitted["residual_hard_l000"], hard)
        for lam, label in ((.25, "025"), (.5, "050"), (.75, "075")):
            shrink = emitted["shrink_l" + label]
            np.testing.assert_array_equal(shrink, lam * raw + (1 - lam) * raw.mean(axis=0))
            np.testing.assert_array_equal(emitted["residual_hard_l" + label], lam * raw + (1 - lam) * hard)
            for source, target in ((0, 1), (1, 0)):
                if roc_auc_score(truth, raw[:, 0, source, target]) != roc_auc_score(truth, shrink[:, 0, source, target]):
                    raise VerificationError("Positive affine shrink changed within-edge ordering")


def check_local_contract(module) -> None:
    ap = module.positive_change_ap
    if ap(np.array([0, 1]), np.zeros(2)) != 0 or ap(np.array([0, 1]), np.ones(2)) != .5:
        raise VerificationError("Positive-change AP must abstain at zero and keep tied groups")
    if not np.isnan(ap(np.zeros(2, bool), np.ones(2))) or not np.isnan(ap(np.ones(2, bool), np.ones(2))):
        raise VerificationError("Ineligible AP target must remain missing")


def expected_cases(symmetric=False) -> set:
    if symmetric:
        return {("C", "lasso", seed, d, f"D{d}_h{h}_s{s}") for seed in range(70000, 70030)
                for d in (300, 600) for h, s in [(200, 100), (300, 150), (400, 200), (300, 75), (300, 300)]}
    return {("A", scorer, seed, d, f"D{d}_fixed_phase") for seed in range(50000, 50100)
            for scorer in ("lasso", "ols") for d in (300, 600)} | {
            ("B", "lasso", seed, d, f"D{d}_{kind}") for seed in range(60000, 60050)
            for d in (300, 600) for kind in ("fixed_phase", "random_phase", "irregular")}


def normalize_detail(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "repeat_seed" not in frame:
        frame["repeat_seed"] = frame.seed
    if "duration" not in frame:
        frame["duration"] = frame.D
    return frame


def check_coverage(frame: pd.DataFrame, kind: str, symmetric: bool) -> dict:
    frame = normalize_detail(frame)
    methods = CORE_METHODS if kind == "core" else CONTROL_METHODS if kind == "controls" else LOCAL_METHODS
    needed = CASE_FIELDS + ["method"]
    if not set(needed) <= set(frame) or frame[needed].isna().any().any() or frame.duplicated(needed).any():
        raise VerificationError("Missing, duplicate or unidentified planned method row")
    cases = set(frame[CASE_FIELDS].itertuples(index=False, name=None))
    if cases != expected_cases(symmetric):
        raise VerificationError("Planned repeat/cell coverage differs")
    if any(set(group.method) != methods for _, group in frame.groupby(CASE_FIELDS, sort=False)):
        raise VerificationError("A planned case silently lost or gained a method")
    metrics = LOCAL_METRICS if kind == "localisation" else {"temporal_delta_auroc", "pooled_temporal_auroc"}
    if not metrics <= set(frame):
        raise VerificationError("Required metric coverage is missing")
    for metric in metrics:
        numeric = pd.to_numeric(frame[metric], errors="coerce")
        if (frame[metric].notna() & ~np.isfinite(numeric)).any():
            raise VerificationError("Metric entries must be finite numbers or explicitly missing")
    status_name = "fit_status" if kind == "core" else "status"
    if status_name not in frame or frame[status_name].isna().any():
        raise VerificationError("Failures/undefined targets lack explicit status")
    failed = frame[status_name].astype(str).str.contains("fail", case=False)
    if frame.loc[failed, list(metrics)].notna().any().any():
        raise VerificationError("A retained fit failure was imputed to a metric")
    return {"cases": len(cases), "method_rows": len(frame), "failed_method_rows": int(failed.sum()),
            "missing_metric_entries": {metric: int(frame[metric].isna().sum()) for metric in sorted(metrics)}}


def check_control_rows(frame: pd.DataFrame) -> None:
    frame = normalize_detail(frame)
    groups = frame.groupby(CASE_FIELDS, sort=False)
    for _, group in groups:
        if (group.status == "failed").any():
            continue
        rows = group.set_index("method")
        gate = rows.loc["raw_window", "gate_accepted"]
        if not isinstance(gate, (bool, np.bool_)):
            raise VerificationError("Missing/invalid gate decision")
        for metric in ("temporal_delta_auroc", "pooled_temporal_auroc"):
            raw = rows.loc["raw_window", metric]; hard = rows.loc["forced_hard", metric]
            np.testing.assert_allclose(rows.loc["raw_fallback", metric], hard if gate else raw, atol=2e-12, rtol=0,
                                       equal_nan=True, err_msg="raw-fallback rule")
            for method in ("shrink_l100", "residual_hard_l100"):
                np.testing.assert_allclose(rows.loc[method, metric], raw, atol=2e-12, rtol=0, equal_nan=True)
            np.testing.assert_allclose(rows.loc["residual_hard_l000", metric], hard, atol=2e-12, rtol=0, equal_nan=True)
        for method in ("shrink_l025", "shrink_l050", "shrink_l075"):
            np.testing.assert_allclose(rows.loc[method, "temporal_delta_auroc"], rows.loc["raw_window", "temporal_delta_auroc"],
                atol=2e-12, rtol=0, equal_nan=True, err_msg="reported score-preserving invariance")


def check_summary(detail: pd.DataFrame, summary: pd.DataFrame, paired=False) -> None:
    detail = normalize_detail(detail)
    for _, row in summary.iterrows():
        group = detail
        for field in GROUP_FIELDS:
            if field in row.index:
                group = group[group[field] == row[field]]
        selected = group[group.method == row.method].set_index("repeat_seed")
        values = pd.to_numeric(selected[row.metric], errors="coerce")
        if paired or row.get("comparison", "absolute") == "minus_raw":
            ref = row.get("reference", "raw_window")
            reference = group[group.method == ref].set_index("repeat_seed")[row.metric]
            values = values - reference.reindex(selected.index)
        n = int(values.notna().sum())
        if len(values) != int(row.n_planned):
            raise VerificationError("Summary dropped a planned repeat")
        if "n_failed" in row.index:
            status = "fit_status" if "fit_status" in selected else "status"
            n_failed = int(selected[status].astype(str).str.contains("fail", case=False).sum())
            if int(row.n_failed) != n_failed:
                raise VerificationError("Summary retained-failure denominator differs")
        for field in ("n_valid", "n_finite", "n_evaluable", "n_paired", "n"):
            if field in row.index and int(row[field]) != n:
                raise VerificationError("Summary finite-pair denominator differs")
        for field in ("n_missing", "n_missing_pair", "n_missing_pairs"):
            if field in row.index and int(row[field]) != len(values) - n:
                raise VerificationError("Summary missing-target denominator differs")
        np.testing.assert_allclose(row["mean"], values.mean(), atol=2e-12, rtol=0, equal_nan=True)
        if n < 5 and not (pd.isna(row.ci95_low) and pd.isna(row.ci95_high)):
            raise VerificationError("Small/empty subgroup has an unsupported interval")
        if n >= 5 and not (np.isfinite(row.ci95_low) and np.isfinite(row.ci95_high) and row.ci95_low <= row.ci95_high):
            raise VerificationError("Eligible summary lacks its pointwise interval")


def check_summary_coverage(detail: pd.DataFrame, summary: pd.DataFrame, kind: str, paired=False) -> None:
    """Require the complete prespecified summary grid even after a re-seal."""
    detail = normalize_detail(detail)
    metadata = ["duration", "cell"] if kind == "core" else GROUP_FIELDS
    identifiers = metadata + ["method", "metric"]
    if kind == "core":
        identifiers += ["comparison"]
        combinations = {(method, metric, comparison) for method in CORE_METHODS
            for metric, comparison in (("temporal_delta_auroc", "absolute"),
                ("pooled_temporal_auroc", "absolute"), ("union_auroc", "absolute"),
                ("temporal_delta_auroc", "minus_raw"))}
    else:
        methods = CONTROL_METHODS if kind == "controls" else LOCAL_METHODS
        metrics = {"temporal_delta_auroc", "pooled_temporal_auroc"} if kind == "controls" else LOCAL_METRICS
        if kind == "controls" and paired:
            identifiers += ["reference"]
            combinations = {(method, metric, reference) for method in methods for metric in metrics
                            for reference in ("raw_window", "forced_hard")}
        else:
            combinations = {(method, metric) for method in methods for metric in metrics}
    if not set(identifiers) <= set(summary) or summary[identifiers].isna().any().any():
        raise VerificationError("Incomplete summary identifiers")
    if summary.duplicated(identifiers).any():
        raise VerificationError("Duplicate summary method/metric row")
    groups = set(detail[metadata].itertuples(index=False, name=None))
    expected = {group + combination for group in groups for combination in combinations}
    actual = set(summary[identifiers].itertuples(index=False, name=None))
    if actual != expected:
        raise VerificationError("Planned summary method/metric coverage differs")


def preservation(root: Path) -> tuple[dict, set[Path]]:
    baseline = read_json(inside(root, str(BASE / "reviewer_assets/prior_results_sha256.json")))
    if len(baseline) != 242 or not all(name.endswith(".csv") for name in baseline):
        raise VerificationError("Prior-stage CSV baseline must contain all 242 CSVs")
    closure = check_hashes(root, baseline, "prior-stage CSVs")
    guard = read_json(inside(root, str(BASE / "score_preserving_controls/input_manifest.json")))
    for key, size in (("frozen_A_B_sha256", 1915), ("frozen_source_sha256", 12)):
        if len(guard.get(key, {})) != size:
            raise VerificationError("Incomplete original A/B preservation guard")
        closure.update(check_hashes(root, guard[key], key))
    loc = read_json(inside(root, str(BASE / "temporal_localisation/protocol.json")))
    if loc.get("frozen_input_sha256") != guard["frozen_A_B_sha256"] or loc.get("frozen_source_sha256") != guard["frozen_source_sha256"]:
        raise VerificationError("Independent original-study preservation maps disagree")
    return {"prior_stage_csvs": 242, "frozen_A_B_files": 1915, "frozen_sources": 12}, closure


def check_provenance(root: Path) -> tuple[dict, set[Path]]:
    """Require portable QA/clock exports, without requiring private originals."""
    output = root / BASE / "reviewer_provenance"
    manifest = inside(output, "export_manifest.json")
    value = read_json(manifest)
    records = value.get("records")
    if not isinstance(records, list) or len(records) != len(PROVENANCE_EXPORTS):
        raise VerificationError("Incomplete reviewer-provenance record set")
    if value.get("external_time_attestation") is not False or any(
            not isinstance(value.get(key), str) or not value[key].strip()
            for key in ("scope", "no_hidden_change", "reviewer_material", "C_original_records")):
        raise VerificationError("Missing reviewer-provenance scope/processing declarations")
    closure = {manifest}; names = set(); origins = set(); originals_checked = 0
    for record in records:
        if not isinstance(record, dict):
            raise VerificationError("Invalid reviewer-provenance record")
        name = record.get("exported_file"); original_name = record.get("original_relative_path")
        exported = inside(output, name); original = relative_target(root, original_name)
        if name in names or original_name in origins:
            raise VerificationError("Duplicate reviewer-provenance record")
        names.add(name); origins.add(original_name); closure.add(exported)
        for key in ("exported_sha256", "original_sha256"):
            if not isinstance(record.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", record[key]):
                raise VerificationError("Invalid reviewer-provenance digest")
        if sha(exported) != record["exported_sha256"]:
            raise VerificationError("Reviewer-provenance export checksum mismatch")
        if original.exists():
            if not original.is_file() or sha(original) != record["original_sha256"]:
                raise VerificationError("Reviewer-provenance original checksum mismatch")
            originals_checked += 1
        if name.endswith(".json"):
            if not original_name.endswith(".json") or record.get("normalization") != JSON_NORMALIZATION:
                raise VerificationError("Undeclared reviewer-provenance JSON processing")
            read_json(exported)  # Strict UTF-8 and valid JSON object; never execute retained data.
        elif name.endswith(".py.txt"):
            if (not original_name.endswith(".py") or record.get("normalization") != CODE_NORMALIZATION
                    or record["exported_sha256"] != record["original_sha256"]):
                raise VerificationError("Historical code export must retain its original bytes")
            compile(exported.read_text(encoding="utf-8"), name, "exec")  # Parse only, never import/run.
        else:
            raise VerificationError("Unexpected reviewer-provenance export type")
    existing = {p.resolve() for p in output.rglob("*") if p.is_file()}
    if names != PROVENANCE_EXPORTS or existing != closure:
        raise VerificationError("Reviewer-provenance inventory drift")
    return {"records": len(records), "files": len(closure), "local_originals_checked": originals_checked,
            "private_originals_required": False, "external_time_attestation": False}, closure


def paper_closure(root: Path) -> set[Path]:
    validator = resolve_module(root, "tools/validate_paper_static.py")
    paths = set()
    for name in ("paper/main.tex", "paper/supplement.tex"):
        source = inside(root, name); found = validator.collect_sources(root, source)
        if found.errors:
            raise VerificationError("Invalid paper dependency closure: " + "; ".join(found.errors))
        paths.update(found.tex); paths.update(found.bib); paths.update(found.assets)
    return paths


def snapshot(root: Path, closure: set[Path]) -> dict:
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(closure)}


def verify(root: Path = ROOT, replay=True) -> dict:
    root = Path(root).resolve(); previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    scripts, tools = str(root / "scripts"), str(root / "tools")
    sys.path[:0] = [scripts, tools]
    try:
        # Resolve all five before executing any verifier; no optional/missing pass.
        modules = {prefix: resolve_module(root, module) for prefix, module, _, _ in SPECS}
        asset_module = resolve_module(root, "tools/build_reviewer_controls_assets.py")
        closure = paper_closure(root)
        kept, guards = preservation(root); closure.update(guards)
        provenance, exported_records = check_provenance(root); closure.update(exported_records)
        coverage = {}
        for prefix, module, kind, symmetric in SPECS:
            output = root / BASE / prefix
            seal = "symmetric_sensitivity_outputs.json" if kind == "core" else "manifest.json"
            closure.update(check_manifest(root, str(BASE / prefix), seal))
            closure.add(inside(root, module))
            detail_name = "symmetric_sensitivity_detail.csv" if kind == "core" else "detail.csv"
            detail = pd.read_csv(inside(output, detail_name))
            coverage[prefix] = check_coverage(detail, kind, symmetric)
            if kind == "controls":
                check_control_rows(detail)
            summary_name = "symmetric_sensitivity_summary.csv" if kind == "core" else "summary.csv"
            summary = pd.read_csv(inside(output, summary_name))
            check_summary_coverage(detail, summary, kind)
            check_summary(detail, summary)
            if kind != "core":
                compare_name = "contrasts.csv" if kind == "controls" else "paired_vs_raw.csv"
                paired = pd.read_csv(inside(output, compare_name))
                check_summary_coverage(detail, paired, kind, paired=True)
                check_summary(detail, paired, paired=True)
        closure.update(check_manifest(root, str(BASE / "reviewer_assets"), excludes=("prior_results_sha256.json",)))
        closure.add(inside(root, "tools/build_reviewer_controls_assets.py"))
        before = snapshot(root, closure)
        check_control_contract(modules["score_preserving_controls"])
        check_local_contract(modules["temporal_localisation"])
        reports = {}
        if replay:
            for prefix, _, kind, _ in SPECS:
                report = modules[prefix].verify(root=root, output=root / BASE / prefix)
                if not (report.get("status") == "passed" or report.get("passed") is True):
                    raise VerificationError(f"Numerical replay failed: {prefix}")
                if report.get("new_fits") != 0 or report.get("new_simulations") != 0:
                    raise VerificationError(f"Replay reports generation/fitting: {prefix}")
                reports[prefix] = report
        assets = asset_module.verify_assets(root=root)
        if assets.get("new_fits") != 0:
            raise VerificationError("Asset verification must never fit")
        after_paths = set(closure)
        for prefix, _, _, _ in SPECS:
            after_paths.update(p.resolve() for p in (root / BASE / prefix).rglob("*") if p.is_file())
        after_paths.update(p.resolve() for p in (root / BASE / "reviewer_assets").rglob("*") if p.is_file())
        after_paths.update(p.resolve() for p in (root / BASE / "reviewer_provenance").rglob("*") if p.is_file())
        after_paths.update(paper_closure(root))
        if snapshot(root, after_paths) != before:
            raise VerificationError("Read-only review gate changed its source/output closure")
        return {"status": "passed", "read_only": True, "replay": bool(replay), "new_fits": 0, "new_simulations": 0,
            "preservation": kept, "provenance": provenance, "coverage": coverage, "analyses": reports, "assets": assets,
            "files_unchanged": len(before)}
    finally:
        sys.dont_write_bytecode = previous
        for entry in (scripts, tools):
            if entry in sys.path:
                sys.path.remove(entry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, replay=not args.no_replay), indent=2, allow_nan=False))
