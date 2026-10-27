"""Read-only, fail-closed gate for evaluation sensitivity and external validation.

replay=False skips retained-array numerical replays only. It still checks both
complete blocks, safe sealed inventories, frozen inputs, every planned setting,
summary denominators and presentation sources. No paper dependency is required.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path, PureWindowsPath
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("results/information_sciences")
CASE = ["study", "scorer", "duration", "schedule", "cell", "repeat_seed"]
GROUP = CASE[:-1]
METHODS = {"raw_window", "forced_hard", "native_hard", "raw_fallback", "shrink_l025", "shrink_l050",
    "shrink_l075", "shrink_l100", "residual_hard_l000", "residual_hard_l025", "residual_hard_l050",
    "residual_hard_l075", "residual_hard_l100", "oracle_label", "static_mean"}
QUANTILES = (.8, .9, .95, .975, .99)
LAMBDAS = (0., 1e-8, .25, .5, .75, 1.)
AP_METRICS = ("boundary_positive_change_ap", "boundary_standard_ap", "boundary_zero_group_increment")
THRESHOLD_METRICS = ("boundary_slot_f1", "stable_false_change_rate", "persistent_false_change_rate",
                     "absent_false_change_rate", "stable_any_change_edge_fraction")
PAIR_METRICS = ("hard_state_auroc", "residual_state_auroc", "within_auc_delta", "across_auc_delta", "total_auc_delta")
EXTERNAL_METHODS = {"raw_window", "forced_hard", "pelt_var_adapter"}
EXTERNAL_METRICS = ("temporal_delta_auroc", "pooled_temporal_auroc", "boundary_positive_change_ap", "boundary_standard_ap",
    "boundary_zero_group_increment", "boundary_slot_f1_raw_q95", "boundary_slot_f1_method_q95",
    "stable_false_change_rate_raw_q95", "stable_false_change_rate_method_q95", "persistent_false_change_rate_raw_q95",
    "persistent_false_change_rate_method_q95", "absent_false_change_rate_raw_q95", "absent_false_change_rate_method_q95")
BOUNDARY_SOURCES = {"scripts/analyze_boundary_sensitivity.py", "tests/test_boundary_sensitivity.py",
    "scripts/analyze_temporal_localisation.py", "scripts/analyze_score_preserving_controls.py", "scripts/run_symmetric_sensitivity.py"}
EXTERNAL_SOURCES = {"scripts/run_segmentation_validation.py", "tests/test_segmentation_validation.py",
    "scripts/analyze_boundary_sensitivity.py", "scripts/analyze_temporal_localisation.py", "scripts/analyze_score_preserving_controls.py",
    "scripts/aggregation_study_common.py", "scripts/run_duration_confirmation.py", "scripts/run_symmetric_sensitivity.py",
    "src/stcg/segmented.py", "src/stcg/mechanism.py", "src/stcg/baselines.py", "src/stcg/self_supervised.py",
    "src/stcg/evaluation.py", "src/stcg/__init__.py", "configs/symmetric_sensitivity.yaml", "requirements-revision.txt"}
BOUNDARY_CSVS = {f"{part}_{kind}.csv" for part, kinds in (("ap", ("detail", "summary", "paired_vs_raw")),
    ("threshold", ("detail", "summary", "paired_vs_raw")), ("residual", ("detail", "summary", "paired_vs_hard"))) for kind in kinds}
FIGURES = {"fig_boundary_sensitivity", "fig_threshold_curves", "fig_residual_pairs"}
TABLES = {"tab_external_framework", "tab_ap_sensitivity", "tab_residual_pairs", "tab_external_contrasts"}
PROSE = {"results_boundary.tex", "results_external.tex", "results_residual.tex"}
EXPORTS = {"boundary_freeze.json", "boundary_run.json", "boundary_rebuild.json", "boundary_preservation.json",
           "segmentation_isolated_environment.json"}


class VerificationError(ValueError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(base, name):
    base = Path(base).resolve()
    if (not isinstance(name, str) or not name or PureWindowsPath(name).drive
            or PureWindowsPath(name).is_absolute() or Path(name).is_absolute()):
        raise VerificationError(f"Unsafe dependency path: {name!r}")
    normalized = name.replace("\\", "/")
    if any(part in ("", ".", "..") for part in normalized.split("/")):
        raise VerificationError(f"Unsafe dependency path: {name!r}")
    result = (base / normalized).resolve()
    if not result.is_relative_to(base):
        raise VerificationError("Dependency escapes root: " + name)
    return result


def inside(base, name):
    path = safe_path(base, name)
    if not path.is_file():
        raise VerificationError("Required file missing: " + name)
    return path


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationError("Expected JSON object: " + Path(path).name)
    return value


def check_hashes(base, mapping, label):
    if not isinstance(mapping, dict) or not mapping:
        raise VerificationError("Missing hash map: " + label)
    closure = set()
    for name, digest in mapping.items():
        if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
            raise VerificationError("Invalid digest: " + label)
        path = inside(base, name)
        if path in closure:
            raise VerificationError("Duplicate normalized dependency path: " + label)
        if sha(path) != digest:
            raise VerificationError("Checksum mismatch: " + label + "/" + name)
        closure.add(path)
    return closure


def inventory(directory):
    return {p.relative_to(directory).as_posix() for p in Path(directory).rglob("*") if p.is_file()}


def check_manifest(root, block, expected=None):
    output = Path(root) / BASE / block
    manifest_path = inside(output, "manifest.json"); value = read_json(manifest_path)
    closure = check_hashes(output, value.get("sha256"), block)
    names = inventory(output)
    if names != set(value["sha256"]) | {"manifest.json"} or (expected is not None and names != expected):
        raise VerificationError("Output inventory drift: " + block)
    protocol = inside(output, "protocol.json")
    if value.get("protocol_sha256") != sha(protocol):
        raise VerificationError("Protocol checksum mismatch: " + block)
    return value, closure | {manifest_path, protocol}


def expected_cases():
    cases = set()
    for scorer in ("lasso", "ols"):
        for seed in range(50000, 50100):
            for d in (300, 600):
                cases.add(("A", scorer, d, "fixed_phase", f"D{d}_fixed_phase", seed))
    for seed in range(60000, 60050):
        for d in (300, 600):
            for schedule in ("fixed_phase", "random_phase", "irregular"):
                cases.add(("B", "lasso", d, schedule, f"D{d}_{schedule}", seed))
    for seed in range(70000, 70030):
        for d in (300, 600):
            for h, stride in ((200, 100), (300, 150), (400, 200), (300, 75), (300, 300)):
                cases.add(("C", "lasso", d, "fixed_phase", f"D{d}_h{h}_s{stride}", seed))
    return cases


def boundary_files():
    files = {"manifest.json", "protocol.json", "input_manifest.json", "diagnostics.json"} | BOUNDARY_CSVS
    files.update(f"records/{study}_{scorer}_{seed}_{cell}.npz" for study, scorer, d, schedule, cell, seed in expected_cases())
    return files


def external_cases():
    return {(seed, d) for seed in range(70000, 70030) for d in (300, 600)}


def fresh_cases():
    return {(study, seed, d) for study, seeds in (("A", (50000, 50001)), ("C", (70000, 70001))) for seed in seeds for d in (300, 600)}


def external_files():
    files = {"manifest.json", "protocol.json", "detail.csv", "summary.csv", "contrasts.csv", "fresh_comparisons.csv", "diagnostics.json"}
    files.update("execution/" + n for n in ("environment.json", "freeze_record.json", "run_record.json"))
    files.update(f"parts/repeat_{seed}_D{d}.{suffix}" for seed, d in external_cases() for suffix in ("json", "npz"))
    for study, seed, d in fresh_cases():
        prefix = f"fresh/{study}_{seed}_D{d}"
        files.add(prefix + ".json")
        files.update(prefix + "_" + kind + ".npz" for kind in ("inputs", "series", "fit"))
    return files


def require_grid(frame, identifiers, combinations, label):
    if not set(identifiers) <= set(frame) or frame[identifiers].isna().any().any():
        raise VerificationError("Missing coverage identifiers: " + label)
    if frame.duplicated(identifiers).any():
        raise VerificationError("Duplicate coverage row: " + label)
    if set(frame[identifiers].itertuples(index=False, name=None)) != combinations:
        raise VerificationError("Planned coverage lost or gained: " + label)


def finite_or_missing(frame, metrics, label):
    for metric in metrics:
        values = frame[metric].to_numpy(float)
        if np.isinf(values).any():
            raise VerificationError("Infinite metric: " + label + "/" + metric)


def close(actual, expected, label):
    try:
        np.testing.assert_allclose(np.asarray(actual, float), np.asarray(expected, float), atol=2e-12, rtol=1e-12, equal_nan=True)
    except (AssertionError, TypeError, ValueError) as exc:
        raise VerificationError("Numerical contract differs: " + label) from exc


def check_boundary_coverage(ap, threshold, residual):
    cases = expected_cases()
    require_grid(ap, CASE + ["method"], {c + (m,) for c in cases for m in METHODS}, "boundary AP")
    require_grid(threshold, CASE + ["method", "q", "threshold_mode"],
        {c + (m, q, mode) for c in cases for m in METHODS for q in QUANTILES for mode in ("raw_shared", "method_self")}, "boundary threshold")
    require_grid(residual, CASE + ["method", "lambda"],
        {c + (f"residual_{weight:g}", weight) for c in cases for weight in LAMBDAS}, "residual pairs")
    for frame, metrics, label in ((ap, AP_METRICS, "AP"), (threshold, THRESHOLD_METRICS, "threshold"), (residual, PAIR_METRICS, "pairs")):
        finite_or_missing(frame, metrics, label)
        if set(frame.status) != {"ok"}:
            raise VerificationError("Frozen successful cases changed status: " + label)
    missing = (ap.study == "C") & (ap.cell == "D300_h300_s300")
    for metric in AP_METRICS:
        if not np.array_equal(ap[metric].isna().to_numpy(), missing.to_numpy()):
            raise VerificationError("AP missing/all-positive coverage differs: " + metric)
    if not ((ap.loc[missing, "boundary_ap_edges_used"] == 0).all() and (ap.loc[~missing, "boundary_ap_edges_used"] > 0).all()
            and (ap.loc[missing, "changing_edges_all_slots_boundary"] > 0).all()):
        raise VerificationError("AP eligible-edge denominator differs")
    close(ap.boundary_standard_ap - ap.boundary_positive_change_ap, ap.boundary_zero_group_increment, "AP zero group")
    if threshold[list(THRESHOLD_METRICS)].isna().any().any() or (threshold.boundary_f1_edges_used <= 0).any():
        raise VerificationError("Threshold metric/edge denominator missing")
    for prefix in ("stable", "persistent", "absent"):
        events, opportunities = threshold[prefix + "_false_events"], threshold[prefix + "_event_opportunities"]
        if (opportunities <= 0).any() or (events < 0).any() or (events > opportunities).any():
            raise VerificationError("False-change count denominator differs")
        close(threshold[prefix + "_false_change_rate"], events / opportunities, prefix + " exceedance")
    close(threshold.stable_event_opportunities, threshold.persistent_event_opportunities + threshold.absent_event_opportunities, "stable opportunities")
    close(threshold.stable_false_events, threshold.persistent_false_events + threshold.absent_false_events, "stable events")
    for name in ("threshold", "raw_threshold", "method_threshold"):
        if not np.isfinite(threshold[name]).all() or (threshold[name] < 0).any():
            raise VerificationError("Invalid fixed-quantile threshold")
    selected = np.where(threshold.threshold_mode == "raw_shared", threshold.raw_threshold, threshold.method_threshold)
    close(threshold.threshold, selected, "selected threshold mode")
    if not np.array_equal(threshold.threshold_zero.astype(bool), threshold.threshold == 0):
        raise VerificationError("Zero-threshold strict event convention differs")
    if residual[list(PAIR_METRICS)].isna().any().any() or (residual.n_evaluable_edges <= 0).any():
        raise VerificationError("Residual evaluable-edge denominator differs")
    close(residual.within_auc_delta + residual.across_auc_delta, residual.total_auc_delta, "pair partition")
    close(residual.residual_state_auroc - residual.hard_state_auroc, residual.total_auc_delta, "pair AUROC")
    close(residual.within_positive_negative_pairs + residual.across_positive_negative_pairs, residual.positive_negative_pairs, "positive-negative pair counts")
    return {"cases": len(cases), "AP_rows": len(ap), "threshold_rows": len(threshold), "residual_rows": len(residual),
            "AP_missing_rows": int(missing.sum()), "threshold_missing_rows": 0, "retained_failures": 0}


def check_summary(detail, summary, group, metrics, extra=(), *, paired=False, references=("raw_window",), reference_extra=None):
    identifiers = [*group, "method", *extra, "metric", *(["reference"] if paired else [])]
    conditions = set(detail[[*group, "method", *extra]].itertuples(index=False, name=None))
    combinations = {c + (metric,) + ((reference,) if paired else ()) for c in conditions for metric in metrics
                    for reference in (references if paired else (None,))}
    require_grid(summary, identifiers, combinations, "paired summary" if paired else "summary")
    # Indexed groups avoid rescanning 150k rows for each of the 15k settings.
    groups = {key if isinstance(key, tuple) else (key,): frame for key, frame in detail.groupby(list(group), sort=False)}
    indexed = {}
    for identity, frame in groups.items():
        for setting, part in frame.groupby(["method", *extra], sort=False):
            key = setting if isinstance(setting, tuple) else (setting,)
            indexed[identity + key] = part.set_index("repeat_seed").sort_index()
    for row in summary.to_dict("records"):
        identity = tuple(row[field] for field in group)
        setting = tuple(row[field] for field in ("method", *extra))
        part = indexed[identity + setting]
        values = part[row["metric"]].to_numpy(float)
        if paired:
            ref_params = tuple(reference_extra) if reference_extra is not None else setting[1:]
            baseline = indexed[identity + (row["reference"], *ref_params)].reindex(part.index)
            values = values - baseline[row["metric"]].to_numpy(float)
        valid = np.isfinite(values); count = int(valid.sum())
        total = len(values)
        if int(row["n_planned"]) != total or int(row["n_paired" if paired else "n_finite"]) != count:
            raise VerificationError("Summary effective denominator differs")
        missing_name = "n_missing_pairs" if "n_missing_pairs" in summary else "n_missing_pair"
        if int(row[missing_name if paired else "n_missing"]) != total - count:
            raise VerificationError("Summary missing denominator differs")
        close(row["mean"], float(values[valid].mean()) if count else np.nan, "summary mean")
        ci = np.asarray([row["ci95_low"], row["ci95_high"]], float)
        if count == 0:
            if not np.isnan(ci).all():
                raise VerificationError("Undefined target gained an interval")
        elif not (np.isfinite(ci).all() and ci[0] <= ci[1]):
            raise VerificationError("Finite target lacks a valid pointwise interval")
        elif np.all(values[valid] == values[valid][0]):
            close(ci, [values[valid][0]] * 2, "constant target interval")


def check_boundary_settings(p, manifest):
    if (set(p.get("methods", [])) != METHODS or len(p["methods"]) != len(METHODS)
            or p.get("quantiles") != list(QUANTILES) or p.get("residual_lambdas") != list(LAMBDAS)
            or p.get("thresholds", {}).get("modes") != ["raw_shared", "method_self"]
            or p["thresholds"].get("labels_used") is not False or p["thresholds"].get("event") != "strict strength>threshold"
            or p.get("statistics", {}).get("draws") != 10000 or p["statistics"].get("seed") != 20261025
            or p["statistics"].get("bootstrap_metrics") != {"AP": list(AP_METRICS), "threshold": list(THRESHOLD_METRICS), "residual": list(PAIR_METRICS)}
            or p.get("new_fits") != 0 or p.get("new_simulations") != 0
            or manifest.get("new_fits") != 0 or manifest.get("new_simulations") != 0):
        raise VerificationError("Boundary fixed sensitivity configuration differs")


def check_boundary_protocol(root, output, manifest):
    p = read_json(inside(output, "protocol.json")); path = inside(output, "input_manifest.json")
    inputs = read_json(path); closure = {path}
    if (set(p.get("source_sha256", {})) != BOUNDARY_SOURCES or set(manifest.get("source_sha256", {})) != BOUNDARY_SOURCES
            or p["source_sha256"] != manifest["source_sha256"]):
        raise VerificationError("Boundary frozen source inventory differs")
    closure.update(check_hashes(root, p["source_sha256"], "boundary sources"))
    if p.get("input_manifest_sha256") != sha(path) or manifest.get("input_manifest_sha256") != sha(path):
        raise VerificationError("Boundary input-manifest checksum differs")
    check_boundary_settings(p, manifest)
    ab = inputs.get("original_AB_guard", {})
    maps = [(ab.get("frozen_A_B_sha256"), 1915), (ab.get("frozen_source_sha256"), 12),
            (inputs.get("original_C_files_sha256"), 790), (inputs.get("original_C_sources_sha256"), 10),
            (inputs.get("previous_analysis_files_sha256"), 1030)]
    for mapping, count in maps:
        if not isinstance(mapping, dict) or len(mapping) != count:
            raise VerificationError("Incomplete protected boundary input inventory")
        closure.update(check_hashes(root, mapping, "protected boundary inputs"))
    old_ab_path = inside(root, (BASE / "score_preserving_controls/input_manifest.json").as_posix()); old = read_json(old_ab_path)
    if old != ab:
        raise VerificationError("Boundary original A/B guard disagrees with retained guard")
    previous_names = {p.relative_to(root).as_posix() for name in ("score_preserving_controls", "temporal_localisation", "symmetric_controls", "symmetric_localisation")
                      for p in (root / BASE / name).rglob("*") if p.is_file()}
    c_names = {p.relative_to(root).as_posix() for p in (root / BASE / "symmetric_sensitivity").rglob("*") if p.is_file()}
    if previous_names != set(inputs["previous_analysis_files_sha256"]) or c_names != set(inputs["original_C_files_sha256"]):
        raise VerificationError("Protected prior output inventory differs")
    closure.add(old_ab_path)
    return inputs, closure


def check_external_protocol(root, output, manifest, boundary_inputs):
    p = read_json(inside(output, "protocol.json")); c = p.get("configuration", {})
    if set(p.get("source_sha256", {})) != EXTERNAL_SOURCES:
        raise VerificationError("External frozen source inventory differs")
    closure = check_hashes(root, p["source_sha256"], "external sources")
    if (c.get("seeds") != list(range(70000, 70030)) or c.get("durations") != [300, 600]
            or set(c.get("methods", [])) != EXTERNAL_METHODS or len(c["methods"]) != 3
            or c.get("window_size") != 300 or c.get("stride") != 150
            or c.get("pelt") != {"penalty_factor": 1., "n_pcs": 5, "min_size": 2, "jump": 1, "model": "l2", "n_bkps": None}
            or c.get("metrics", {}).get("names") != list(EXTERNAL_METRICS)
            or c.get("statistics", {}).get("bootstrap_seed") != 20261025 or c["statistics"].get("bootstrap_draws") != 10000
            or c.get("fresh", {}).get("A_seeds") != [50000, 50001] or c["fresh"].get("C_seeds") != [70000, 70001]
            or c["fresh"].get("durations") != [300, 600] or c["fresh"].get("fresh_cells") != 8
            or c.get("no_Kmeans_in_adapter") is not True or c.get("no_gate_in_adapter") is not True
            or p.get("internal_freeze_only") is not True or p.get("public_preregistration") is not False):
        raise VerificationError("External bounded sensitivity configuration differs")
    protected = p.get("protected", {})
    expected = {"A_B_files_sha256": boundary_inputs["original_AB_guard"]["frozen_A_B_sha256"],
        "C_files_sha256": boundary_inputs["original_C_files_sha256"],
        "old_sources_sha256": {**boundary_inputs["original_AB_guard"]["frozen_source_sha256"], **boundary_inputs["original_C_sources_sha256"]}}
    if protected != expected:
        raise VerificationError("External protected input maps disagree with boundary maps")
    for name, mapping in protected.items():
        closure.update(check_hashes(root, mapping, "external " + name))
    for seed, duration in external_cases():
        name = f"parts/repeat_{seed}_D{duration}"
        record = read_json(inside(output, name + ".json"))
        if (record.get("repeat_seed"), record.get("duration")) != (seed, duration):
            raise VerificationError("External part identity differs")
        source = inside(root, record.get("source"))
        if sha(source) != record.get("source_sha256") or sha(inside(output, name + ".npz")) != record.get("arrays_sha256"):
            raise VerificationError("External retained part/input digest differs")
        closure.add(source)
    for study, seed, duration in fresh_cases():
        record = read_json(inside(output, f"fresh/{study}_{seed}_D{duration}.json"))
        if (record.get("study"), record.get("repeat_seed"), record.get("duration")) != (study, seed, duration):
            raise VerificationError("Fresh reproduction record identity differs")
        for name, reference in record.get("reference_paths", {}).items():
            closure.add(inside(root, reference))
        if set(record.get("reference_paths", {})) != {"inputs", "series", "fit"}:
            raise VerificationError("Fresh reproduction reference inventory differs")
    return closure


def check_external_coverage(detail, output):
    require_grid(detail, ["repeat_seed", "duration", "method"],
        {(seed, duration, method) for seed, duration in external_cases() for method in EXTERNAL_METHODS}, "external adapter")
    finite_or_missing(detail, EXTERNAL_METRICS, "external metrics")
    if set(detail.status) != {"ok"} or detail[list(EXTERNAL_METRICS)].isna().any().any():
        raise VerificationError("Frozen successful external method coverage differs")
    comparisons = pd.read_csv(inside(output, "fresh_comparisons.csv"))
    actual = set(comparisons[["study", "repeat_seed", "duration"]].itertuples(index=False, name=None))
    if actual != fresh_cases() or not comparisons.within_tolerance.all():
        raise VerificationError("Fresh cached reproduction coverage or tolerance differs")
    return {"adapter_cases": len(external_cases()), "method_rows": len(detail), "fresh_reproduction_cases": len(fresh_cases()),
            "fresh_comparison_rows": len(comparisons), "retained_adapter_failures": 0}


def check_compatibility(root, ap=None, threshold=None):
    """Independently compare unchanged old AP+/raw-q95 per-repeat measurements."""
    output = root / BASE / "boundary_sensitivity"
    ap = pd.read_csv(inside(output, "ap_detail.csv")) if ap is None else ap
    threshold = pd.read_csv(inside(output, "threshold_detail.csv")) if threshold is None else threshold
    paths = {inside(root, (BASE / block / "detail.csv").as_posix()) for block in ("temporal_localisation", "symmetric_localisation")}
    old_frames = [pd.read_csv(path) for path in sorted(paths)]
    old = pd.concat([frame for frame in old_frames if len(frame)] or old_frames, ignore_index=True)
    keys = [*CASE, "method"]
    expected = {case + (method,) for case in expected_cases() for method in METHODS}
    require_grid(old, keys, expected, "original boundary compatibility")
    require_grid(ap, keys, expected, "new AP compatibility")
    shared = threshold[(threshold.q == .95) & (threshold.threshold_mode == "raw_shared")]
    require_grid(shared, keys, expected, "new raw-q95 compatibility")
    paired_ap = ap.merge(old, on=keys, suffixes=("_new", "_old"), validate="1:1")
    close(paired_ap.boundary_positive_change_ap_new, paired_ap.boundary_positive_change_ap_old, "unchanged original AP+")
    close(paired_ap.boundary_ap_edges_used_new, paired_ap.boundary_ap_edges_used_old, "unchanged AP eligible counts")
    paired_threshold = shared.merge(old, on=keys, suffixes=("_new", "_old"), validate="1:1")
    for metric in THRESHOLD_METRICS:
        close(paired_threshold[metric + "_new"], paired_threshold[metric + "_old"], "unchanged raw-q95 " + metric)
    close(paired_threshold.boundary_f1_edges_used_new, paired_threshold.boundary_f1_edges_used_old, "unchanged F1 eligible counts")
    expected_missing = (ap.study == "C") & (ap.cell == "D300_h300_s300")
    for metric in ("boundary_standard_ap", "boundary_positive_change_ap"):
        if not np.array_equal(ap[metric].isna(), expected_missing):
            raise VerificationError("Compatibility AP missing coverage differs")
    gap = np.abs(paired_ap.boundary_positive_change_ap_new - paired_ap.boundary_positive_change_ap_old)
    return {"method_rows_compared": len(ap), "AP_missing_rows_each_definition": int(expected_missing.sum()),
        "AP_plus_max_abs_gap": float(gap.max()), "raw_q95_metrics": list(THRESHOLD_METRICS),
        "sample_or_metric_definition_changed": False,
        "CI_stream_change": "Old streams include group/method/metric/absolute-or-paired; the new fixed stream includes only cell and finite-repeat mask, reused across settings. Small CI endpoint differences are bootstrap Monte Carlo variation; previous CIs remain retained."}, paths


def check_provenance(root):
    output = root / BASE / "evaluation_provenance"
    manifest_path = inside(output, "export_manifest.json"); manifest = read_json(manifest_path)
    records = manifest.get("records")
    if (not isinstance(records, list) or len(records) != len(EXPORTS)
            or manifest.get("external_time_attestation") is not False or manifest.get("private_originals_required") is not False
            or not isinstance(manifest.get("scope"), str) or not manifest["scope"].strip()):
        raise VerificationError("Evaluation provenance declarations or record count differ")
    closure = {manifest_path}; names, origins = set(), set()
    values = {}
    for record in records:
        if not isinstance(record, dict):
            raise VerificationError("Invalid evaluation provenance record")
        name = record.get("exported_file"); origin = record.get("original_relative_path")
        exported = inside(output, name); safe_path(root, origin)  # Historical private originals are optional.
        if name in names or origin in origins or name not in EXPORTS:
            raise VerificationError("Duplicate or unexpected provenance record")
        names.add(name); origins.add(origin)
        if (not isinstance(record.get("normalization"), str) or not record["normalization"].strip()
                or not isinstance(record.get("original_sha256"), str) or re.fullmatch("[0-9a-f]{64}", record["original_sha256"]) is None):
            raise VerificationError("Invalid evaluation provenance processing/digest")
        closure.update(check_hashes(output, {name: record.get("exported_sha256")}, "provenance export"))
        values[name] = read_json(exported)
    if names != EXPORTS or inventory(output) != EXPORTS | {"export_manifest.json"}:
        raise VerificationError("Evaluation provenance inventory differs")
    boundary = root / BASE / "boundary_sensitivity"
    protocol = read_json(inside(boundary, "protocol.json")); seal = read_json(inside(boundary, "manifest.json"))
    freeze, run, rebuild, preserved = [values[name] for name in ("boundary_freeze.json", "boundary_run.json", "boundary_rebuild.json", "boundary_preservation.json")]
    if (freeze.get("protocol_sha256") != sha(boundary / "protocol.json") or freeze.get("source_sha256") != protocol["source_sha256"]
            or freeze.get("input_manifest_sha256") != sha(boundary / "input_manifest.json")
            or run.get("manifest_sha256") != sha(boundary / "manifest.json")):
        raise VerificationError("Boundary execution/protocol digest linkage differs")
    def time(value, key):
        try:
            stamp = datetime.fromisoformat(value[key])
            if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
                raise ValueError("not UTC")
            return stamp
        except (KeyError, TypeError, ValueError) as exc:
            raise VerificationError("Invalid internal UTC chronology: " + key) from exc
    stamps = [time(freeze, "before_freeze_utc"), time(freeze, "after_freeze_before_computation_utc"),
        time(run, "started_utc"), time(run, "completed_utc"), time(rebuild, "started_utc"), time(rebuild, "completed_utc")]
    if stamps != sorted(stamps):
        raise VerificationError("Internal freeze/run/rebuild chronology is out of order")
    actual = {name: sha(inside(boundary, name)) for name in inventory(boundary)}
    for field in ("before_sha256", "after_sha256"):
        if rebuild.get(field) != actual:
            raise VerificationError("Rebuild byte-invariance inventory differs")
        closure.update(check_hashes(boundary, rebuild[field], "internal rebuild evidence"))
    if (rebuild.get("status") != "passed" or rebuild.get("byte_identical_files") != len(actual)
            or preserved.get("before_after_inputs_equal") is not True
            or preserved.get("frozen_input_manifest_sha256") != sha(boundary / "input_manifest.json")
            or preserved.get("after_input_snapshot_sha256") != sha(boundary / "input_manifest.json")):
        raise VerificationError("Rebuild/preservation evidence differs")
    for item in (freeze, run, rebuild, preserved):
        if item.get("new_fits") != 0 or item.get("new_simulations") != 0:
            raise VerificationError("Boundary cached analysis execution reports new fitting")
    return {"exported_records": len(records), "private_originals_required": False, "external_time_attestation": False,
            "chronology": "Internal local source-lock and computation records only"}, closure


def check_assets(root):
    output = root / BASE / "evaluation_assets"; manifest = read_json(inside(output, "manifest.json"))
    generated = {name + "." + ext for name in FIGURES for ext in ("pdf", "png")}
    generated |= {name + ".tex" for name in TABLES} | {"display_source.csv"}
    if (set(manifest.get("output_sha256", {})) != generated or set(manifest.get("figures", [])) != FIGURES
            or set(manifest.get("tables", [])) != TABLES or manifest.get("new_fits") != 0 or manifest.get("new_simulations") != 0
            or inventory(output) != generated | PROSE | {"manifest.json"}):
        raise VerificationError("Evaluation asset inventory differs")
    closure = check_hashes(output, manifest["output_sha256"], "evaluation assets")
    closure.update(check_hashes(root, manifest.get("source_sha256"), "evaluation asset sources"))
    if not {(BASE / "evaluation_assets" / name).as_posix() for name in PROSE} <= set(manifest["source_sha256"]):
        raise VerificationError("Evaluation prose source seal missing")
    closure.add(inside(output, "manifest.json"))
    return closure


def resolve_module(root, relative):
    path = inside(root, relative)
    name = "_evaluation_gate_" + path.stem + "_" + hashlib.sha256(str(root).encode()).hexdigest()[:10]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise VerificationError("Cannot load required pure verifier: " + relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise VerificationError("Verifier resolved outside requested root")
    return module


FORBIDDEN_CALLS = {"fit", "fit_transform", "fit_predict", "partial_fit", "pelt_on_features", "segmented_lasso_scores",
    "fit_aggregation", "_window_scores", "draw_base_inputs", "simulate_schedule", "draw_inputs", "simulate_full",
    "simulate_mechanism", "generate_switching_var", "run_adapter_case", "run_fresh_case", "run", "freeze", "rebuild",
    "write_derived", "seal", "build", "write_json"}


@contextmanager
def no_fitting():
    previous = sys.getprofile()
    def guard(frame, event, argument):
        if event == "call" and frame.f_code.co_name in FORBIDDEN_CALLS:
            raise VerificationError("Read-only replay attempted fitting/generation/writing: " + frame.f_code.co_name)
    sys.setprofile(guard)
    try:
        yield
    finally:
        sys.setprofile(previous)


def guarded_replay(module, root, output):
    with no_fitting():
        report = module.verify(root=root, output=output)
    if report.get("status") != "passed" or report.get("read_only") is not True:
        raise VerificationError("Pure numerical replay did not pass")
    if report.get("new_fits") != 0 or report.get("new_simulations") != 0:
        raise VerificationError("Numerical replay reports new fitting/generation")
    return report


def snapshot(root, closure):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(closure)}


def verify(root=ROOT, replay=True):
    root = Path(root).resolve(); previous = sys.dont_write_bytecode; sys.dont_write_bytecode = True
    entries = [str(root / "scripts"), str(root / "tools")]; sys.path[:0] = entries
    try:
        boundary_seal, closure = check_manifest(root, "boundary_sensitivity", boundary_files())
        external_seal, paths = check_manifest(root, "segmentation_validation", external_files()); closure.update(paths)
        boundary_output, external_output = (root / BASE / block for block in ("boundary_sensitivity", "segmentation_validation"))
        inputs, paths = check_boundary_protocol(root, boundary_output, boundary_seal); closure.update(paths)
        closure.update(check_external_protocol(root, external_output, external_seal, inputs))
        frames = {name: pd.read_csv(inside(boundary_output, name)) for name in BOUNDARY_CSVS}
        coverage = {"boundary_sensitivity": check_boundary_coverage(frames["ap_detail.csv"], frames["threshold_detail.csv"], frames["residual_detail.csv"])}
        compatibility, paths = check_compatibility(root, frames["ap_detail.csv"], frames["threshold_detail.csv"]); closure.update(paths)
        provenance, paths = check_provenance(root); closure.update(paths)
        for part, metrics, extra in (("ap", AP_METRICS, ()), ("threshold", THRESHOLD_METRICS, ("q", "threshold_mode")), ("residual", PAIR_METRICS, ("lambda",))):
            detail = frames[part + "_detail.csv"]
            check_summary(detail, frames[part + "_summary.csv"], GROUP, metrics, extra)
            paired_name = part + ("_paired_vs_hard.csv" if part == "residual" else "_paired_vs_raw.csv")
            check_summary(detail, frames[paired_name], GROUP, metrics, extra, paired=True,
                references=("residual_0",) if part == "residual" else ("raw_window",), reference_extra=(0.,) if part == "residual" else None)
        external_detail = pd.read_csv(inside(external_output, "detail.csv"))
        coverage["segmentation_validation"] = check_external_coverage(external_detail, external_output)
        check_summary(external_detail, pd.read_csv(inside(external_output, "summary.csv")), ["duration"], EXTERNAL_METRICS)
        check_summary(external_detail, pd.read_csv(inside(external_output, "contrasts.csv")), ["duration"], EXTERNAL_METRICS,
            paired=True, references=("raw_window", "forced_hard"))
        closure.update(check_assets(root))
        module_names = {"boundary_sensitivity": "scripts/analyze_boundary_sensitivity.py",
            "segmentation_validation": "scripts/run_segmentation_validation.py", "assets": "tools/build_evaluation_sensitivity_assets.py"}
        modules = {key: resolve_module(root, name) for key, name in module_names.items()}
        closure.update(inside(root, name) for name in module_names.values())
        closure.update(inside(root, name) for name in ("tools/verify_evaluation_sensitivity.py", "tests/test_evaluation_sensitivity_verification.py"))
        before = snapshot(root, closure)
        reports = {}
        if replay:
            for block in ("boundary_sensitivity", "segmentation_validation"):
                reports[block] = guarded_replay(modules[block], root, root / BASE / block)
        with no_fitting():
            assets = modules["assets"].verify_assets(root=root)
        if assets.get("status") != "passed" or assets.get("new_fits") != 0:
            raise VerificationError("Asset replay did not pass without fitting")
        after_closure = set(closure)
        for block in ("boundary_sensitivity", "segmentation_validation", "evaluation_assets"):
            after_closure.update(p.resolve() for p in (root / BASE / block).rglob("*") if p.is_file())
        if before != snapshot(root, after_closure):
            raise VerificationError("Read-only sensitivity gate changed its input/source/output closure")
        return {"status": "passed", "read_only": True, "replay": bool(replay), "coverage": coverage,
            "analyses": reports, "assets": assets, "compatibility": compatibility, "provenance": provenance,
            "files_unchanged": len(before), "new_fits": 0, "new_simulations": 0}
    finally:
        sys.dont_write_bytecode = previous
        for entry in entries:
            if entry in sys.path:
                sys.path.remove(entry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT); parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args(); print(json.dumps(verify(args.root, replay=not args.no_replay), indent=2, allow_nan=False))
