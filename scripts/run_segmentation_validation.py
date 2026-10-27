"""Source-locked PELT+VAR adapter and bounded fresh-fit reproduction.

This separate post hoc study never edits or refreshes Studies A/B/C. PELT uses
retained raw window features, an unknown segment count and disjoint time-range
Lasso refits. Fresh A/C fits are explicitly labelled reproduction, not new seeds.
--verify is read-only and never calls a detector, simulator or scorer.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import aggregation_study_common as common
import run_symmetric_sensitivity as symmetric
from stcg.evaluation import nonself_mask
from stcg.mechanism import MechanismConfig
from stcg.segmented import pelt_on_features, segmented_lasso_scores

OUT = ROOT / "results/information_sciences/segmentation_validation"
BASE = Path("results/information_sciences")
METHODS = ("raw_window", "forced_hard", "pelt_var_adapter")
METRICS = ("temporal_delta_auroc", "pooled_temporal_auroc", "boundary_positive_change_ap",
           "boundary_standard_ap", "boundary_zero_group_increment", "boundary_slot_f1_raw_q95",
           "boundary_slot_f1_method_q95", "stable_false_change_rate_raw_q95",
           "stable_false_change_rate_method_q95", "persistent_false_change_rate_raw_q95",
           "persistent_false_change_rate_method_q95", "absent_false_change_rate_raw_q95",
           "absent_false_change_rate_method_q95")
SOURCES = ("scripts/run_segmentation_validation.py", "tests/test_segmentation_validation.py",
           "scripts/analyze_boundary_sensitivity.py", "scripts/analyze_temporal_localisation.py",
           "scripts/analyze_score_preserving_controls.py", "scripts/aggregation_study_common.py",
           "scripts/run_duration_confirmation.py", "scripts/run_symmetric_sensitivity.py",
           "src/stcg/segmented.py", "src/stcg/mechanism.py", "src/stcg/baselines.py",
           "src/stcg/self_supervised.py", "src/stcg/evaluation.py", "src/stcg/__init__.py",
           "configs/symmetric_sensitivity.yaml", "requirements-revision.txt")
DERIVED = ("detail.csv", "summary.csv", "contrasts.csv", "fresh_comparisons.csv", "diagnostics.json")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def json_bytes(value):
    return (json.dumps(clean(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_json(path, value):
    Path(path).write_bytes(json_bytes(value))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def tree_hashes(path, exclude=()):
    return {p.relative_to(path).as_posix(): sha(p) for p in sorted(Path(path).rglob("*"))
            if p.is_file() and p.relative_to(path).as_posix() not in exclude}


def check_hashes(root, mapping):
    for name, expected in mapping.items():
        p = (Path(root) / name).resolve()
        if not p.is_relative_to(Path(root).resolve()) or not p.is_file() or sha(p) != expected:
            raise ValueError(f"Source/input/output checksum mismatch: {name}")


def config():
    return {"study": "post_review_segmentation_validation_v1", "role": "post hoc descriptive alternative framework and exact-seed reproduction",
        "seeds": list(range(70000, 70030)), "durations": [300, 600], "window_size": 300, "stride": 150,
        "pelt": {"penalty_factor": 1.0, "n_pcs": 5, "min_size": 2, "jump": 1, "model": "l2", "n_bkps": None},
        "refit": {"lasso_alpha": .005, "max_lag": 3, "ranges": "existing segmented_lasso_scores; midpoint of adjacent integer window centres; nonoverlapping observed-time ranges; existing short-range merge",
                  "emission": "final refit range containing each integer window centre; no merging recurrent segments into states"},
        "features": "retained raw window absolute nonself scores; existing pelt_on_features standardisation/PCA",
        "methods": list(METHODS), "no_Kmeans_in_adapter": True, "no_gate_in_adapter": True,
        "metrics": {"names": list(METRICS), "implementation": "analyze_boundary_sensitivity.evaluate_methods; q95 only",
                    "boundary_truth": "actual support changes in adjacent integer-window-centre slots",
                    "AP_subset": "same changing edges with both positive and negative covered slots for AP+ and standard AP",
                    "AP_plus": "exact tie groups; zero change abstains", "standard_AP": "all strengths including zero tie group",
                    "thresholds": "linear q95 of all nonself edge-slot absolute score changes per case; shared raw or method-specific; strict >",
                    "stable_edges": "always-present and always-absent separately; false change is descriptive, not calibrated FPR"},
        "statistics": {"bootstrap_seed": 20261025, "bootstrap_draws": 10000, "unit": "independent paired mechanism repeat within duration", "interval": "pointwise95percentile", "tests": "none; no original A/B family assignment"},
        "fresh": {"A_seeds": [50000, 50001], "C_seeds": [70000, 70001], "durations": [300, 600],
                  "scorer": "lasso", "window_size": 300, "stride": 150, "fresh_cells": 8,
                  "rtol": 1e-10, "atol": 1e-12, "claim": "new generation and fitting of existing seed labels, not independent replication seeds"},
        "threads": 1, "failures": "retain all planned cells, failed method rows and diagnostics; no retry/replacement/parameter changes",
        "warning_policy": "PELT segment-refit ConvergenceWarning fails that adapter case; finite warned arrays retained under failed_ keys but excluded from metrics; other warnings retained; fresh fits follow their original scorer policy",
        "scope": "PELT plus a VAR-Lasso adapter, not an off-the-shelf complete causal method; C maximum radius approximately .67 and no observed scaling trigger do not test the scale-active case"}


def protection(root):
    ab = read_json(root / BASE / "score_preserving_controls/input_manifest.json")
    c = read_json(root / BASE / "symmetric_controls/input_manifest.json")
    return {"A_B_files_sha256": ab["frozen_A_B_sha256"], "C_files_sha256": c["study_C_sha256"],
            "old_sources_sha256": {**ab["frozen_source_sha256"], **c["study_C_frozen_sources_sha256"]}}


def protocol(root=ROOT):
    root = Path(root).resolve()
    guard = protection(root)
    for mapping in guard.values():
        check_hashes(root, mapping)
    return {"configuration": config(), "source_sha256": {n: sha(root / n) for n in SOURCES},
            "protected": guard, "internal_freeze_only": True, "public_preregistration": False}


def freeze(root=ROOT, output=OUT, stage=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Freeze needs a new empty destination")
    value = protocol(root); before = utc_now()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "protocol.json", value)
    # Per-output execution records avoid historical provenance-path collisions.
    execution = output / "execution"; execution.mkdir()
    record = {"before_protocol_write_utc": before, "after_protocol_write_before_fitting_utc": utc_now(),
              "protocol_sha256": sha(output / "protocol.json"), "fits_started": False}
    write_json(execution / "freeze_record.json", record)
    if stage is not None:
        stage = Path(stage); stage.mkdir(parents=True, exist_ok=True)
        target = stage / "segmentation_freeze.json"
        if target.exists():
            raise ValueError("Refuse to overwrite an execution record; choose a new stage")
        write_json(target, record)
    return record


def check_protocol(root, output):
    value = read_json(output / "protocol.json")
    if value != protocol(root):
        raise ValueError("Frozen segmentation protocol/source/input drift")
    return value


def range_emission(segment_maps, ranges, bounds, total):
    """Map final disjoint refit ranges, including any merged segments, to centres."""
    ranges = np.asarray(ranges, dtype=int); bounds = np.asarray(bounds, dtype=int)
    maps = np.asarray(segment_maps)
    if (ranges.ndim != 2 or ranges.shape[1] != 2 or not len(ranges) or len(maps) != len(ranges)
            or ranges[0, 0] != 0 or ranges[-1, 1] != total or np.any(ranges[:, 1] <= ranges[:, 0])
            or np.any(ranges[1:, 0] != ranges[:-1, 1])):
        raise ValueError("Refit ranges must partition observed time exactly")
    centres = (bounds[:, 0] + bounds[:, 1] - 1) // 2
    labels = np.searchsorted(ranges[:, 1], centres, side="right")
    if np.any(labels >= len(ranges)) or np.any(centres < ranges[labels, 0]):
        raise ValueError("Window centres must fall inside refit ranges")
    return maps[labels], labels


def fit_pelt(x, raw, bounds):
    features = np.asarray(raw)[:, nonself_mask(raw.shape[1:])]
    result = pelt_on_features(features, penalty_factor=1., n_pcs=5, min_size=2, n_bkps=None)
    maps, ranges = segmented_lasso_scores(x, bounds, result.labels, max_lag=3, alpha=.005)
    emitted, final_labels = range_emission(maps, ranges, bounds, len(x))
    if not np.isfinite(maps).all():
        raise ValueError("Nonfinite PELT segment scores")
    return {"pelt_labels": result.labels, "pelt_breakpoints": np.asarray(result.breakpoints, dtype=int),
            "segment_maps": maps, "segment_ranges": np.asarray(ranges, dtype=int),
            "refit_window_labels": final_labels, "emission_pelt_var_adapter": emitted}, {
                "penalty": result.penalty, "n_pcs": result.n_pcs,
                "n_detected_segments": len(np.unique(result.labels)), "n_refit_segments": len(ranges)}


def evaluate(emissions, raw, bounds, regimes, support):
    from analyze_boundary_sensitivity import evaluate_methods
    return evaluate_methods(emissions, raw, bounds, regimes, support)


def retained_case(root, seed, duration):
    base = root / BASE / "symmetric_sensitivity"
    part = base / "parts" / f"repeat_{seed}_D{duration}_h300_s150.npz"
    values = arrays(part)
    return values, part.relative_to(root).as_posix()


def run_adapter_case(root, output, seed, duration):
    values, source = retained_case(root, seed, duration)
    path = output / "parts" / f"repeat_{seed}_D{duration}"
    saved = {k: values[k] for k in ("window_scores", "bounds", "regimes", "state_support",
                                    "emission_raw_window", "emission_forced_hard")}
    before = perf_counter(); status, error, info = "ok", "", {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fitted, info = fit_pelt(values["observed_series"], values["window_scores"], values["bounds"])
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                saved.update({"failed_" + key: value for key, value in fitted.items()})
                raise common.ScorerFailure("PELT segment refit emitted ConvergenceWarning; no retry", info)
            saved.update(fitted)
        except Exception as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            info = {**info, **getattr(exc, "diagnostic", {}), "traceback": traceback.format_exc()}
    np.savez_compressed(path.with_suffix(".npz"), **saved)
    record = {"repeat_seed": seed, "duration": duration, "status": status, "error": error,
              "source": source, "source_sha256": sha(root / source), "diagnostic": clean(info),
              "warnings": [str(w.message) for w in caught], "fit_seconds": perf_counter() - before,
              "arrays_sha256": sha(path.with_suffix(".npz")), "new_fitting": True}
    write_json(path.with_suffix(".json"), record)
    return record


def comparison_rows(new, old, identity, category, rtol=1e-10, atol=1e-12):
    """Report true array equality separately from the frozen float tolerance."""
    rows = []
    for key in sorted(set(new) | set(old)):
        a, b = new.get(key), old.get(key)
        shape = a is not None and b is not None and a.shape == b.shape
        dtype = shape and a.dtype == b.dtype
        numeric = shape and a.dtype.kind in "biufc" and b.dtype.kind in "biufc"
        exact = bool(shape and np.array_equal(a, b, equal_nan=True))
        tolerant = bool(dtype and (np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
                                   if numeric and a.dtype.kind in "fc" else exact))
        finite = numeric and np.isfinite(a).all() and np.isfinite(b).all()
        difference = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if finite and a.size else (0. if exact else np.nan)
        rows.append({**identity, "category": category, "array": key, "same_shape": bool(shape),
            "same_dtype": bool(dtype), "exact_equal": exact and bool(dtype), "within_tolerance": tolerant,
            "max_abs_difference": difference, "classification": "exact" if exact and dtype else ("tolerance" if tolerant else "different")})
    return rows


def run_fresh_case(root, output, study, seed, duration):
    """Generate RNG inputs and trajectories and fit afresh; never initialise from saved arrays."""
    folder = output / "fresh"; prefix = f"{study}_{seed}_D{duration}"
    started = perf_counter(); identity = {"study": study, "repeat_seed": seed, "duration": duration}
    refs = {}; status, error, rows, fit_arrays = "ok", "", [], {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if study == "A":
                original = root / BASE / "duration_confirmation"; p = read_json(original / "protocol.json")
                setting = replace(MechanismConfig(**p["base_config"]), seed=seed, switch_period=duration)
                inputs = common.draw_base_inputs(setting, seed)
                schedule = common.fixed_schedule(setting.timesteps + setting.burn_in, duration)
                dataset = common.simulate_schedule(inputs, schedule, setting)
                series = {"x": dataset.x, "regimes": dataset.regimes, "regimes_full": schedule}
                refs = {"inputs": (original / "records" / f"repeat_{seed}_inputs.npz").relative_to(root).as_posix(),
                        "series": (original / "records" / f"repeat_{seed}_D{duration}_data.npz").relative_to(root).as_posix(),
                        "fit": (original / "records" / f"repeat_{seed}_D{duration}_lasso.npz").relative_to(root).as_posix()}
                exact_random, cluster_seed = True, seed
            else:
                original = root / BASE / "symmetric_sensitivity"; c = symmetric.load_config(root / "configs/symmetric_sensitivity.yaml")
                inputs = symmetric.draw_inputs(c, seed)
                series, dataset = symmetric.simulate_full(inputs, duration, c)
                refs = {"inputs": (original / "inputs" / f"repeat_{seed}.npz").relative_to(root).as_posix(),
                        "series": (original / "series" / f"repeat_{seed}_D{duration}.npz").relative_to(root).as_posix(),
                        "fit": (original / "parts" / f"repeat_{seed}_D{duration}_h300_s150.npz").relative_to(root).as_posix()}
                exact_random, cluster_seed = False, symmetric.domain_seed(seed, "clustering", bits=32)
            # Persist new inputs/trajectory before the scorer is called.
            np.savez_compressed(folder / f"{prefix}_inputs.npz", **inputs)
            np.savez_compressed(folder / f"{prefix}_series.npz", **series)
            rows, fit_arrays, numerical = common.fit_aggregation(dataset, "lasso", cluster_seed, exact_random=exact_random,
                window_size=300, stride=150, lasso_alpha=.005, n_init=10)
            if study == "C":
                fit_arrays = {"observed_series": dataset.x, "regimes": dataset.regimes, "state_support": inputs["support"],
                              "coefficients": inputs["coefficients"], **fit_arrays}
            np.savez_compressed(folder / f"{prefix}_fit.npz", **fit_arrays)
        except Exception as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            numerical = {"traceback": traceback.format_exc()}
    record = {**identity, "status": status, "error": error, "reference_paths": refs, "metrics": rows,
              "warnings": [str(w.message) for w in caught], "diagnostic": numerical, "new_generation": True,
              "new_fitting": True, "elapsed_seconds": perf_counter() - started}
    write_json(folder / f"{prefix}.json", record)
    return record


def fresh_comparisons(root, output):
    rows = []
    for path in sorted((output / "fresh").glob("*.json")):
        record = read_json(path); identity = {k: record[k] for k in ("study", "repeat_seed", "duration")}
        prefix = path.stem
        if record["status"] != "ok":
            rows.append({**identity, "category": "failure", "array": "all", "classification": "failed",
                         "exact_equal": False, "within_tolerance": False, "error": record["error"]})
            continue
        for kind, reference in record["reference_paths"].items():
            rows.extend(comparison_rows(arrays(output / "fresh" / f"{prefix}_{kind}.npz"), arrays(root / reference), identity, kind))
        # Score metrics are also compared with the original fitted record, not merely array identity.
        if record["study"] == "A":
            old = read_json(root / BASE / "duration_confirmation/records" / f"repeat_{record['repeat_seed']}.json")["rows"]
            old = [x for x in old if x["D"] == record["duration"] and x["scorer"] == "lasso"]
        else:
            old = read_json((root / record["reference_paths"]["fit"]).with_suffix(".json"))["methods"]
        old = {x["method"]: x for x in old}
        new_inputs = arrays(output / "fresh" / f"{prefix}_inputs.npz")
        new_series = arrays(output / "fresh" / f"{prefix}_series.npz")
        new_fit = arrays(output / "fresh" / f"{prefix}_fit.npz")
        support = new_inputs["support"]
        dataset = common.MechanismDataset(x=new_series.get("x", new_series.get("observed_series")),
            coefficients=new_inputs["coefficients"], state_support=support, union_support=support.max(axis=0),
            regimes=new_series["regimes"], spectral_radius=new_inputs.get("spectral_radius", new_inputs.get("rho_final")),
            overlap_realised=np.nan)
        for row in record["metrics"]:
            if "emission_" + row["method"] in new_fit:
                recomputed = common.evaluate_predictions(new_fit["emission_" + row["method"]], new_fit["bounds"], dataset)
            elif row["method"] == "matched_random":
                recomputed = {metric: float(new_fit["matched_random_" + metric].mean()) for metric in
                              ("temporal_delta_auroc", "pooled_temporal_auroc", "union_auroc")}
            else:
                raise ValueError("Missing emission for a fresh-fit metric")
            for metric in ("temporal_delta_auroc", "pooled_temporal_auroc", "union_auroc"):
                rows.extend(comparison_rows({metric: np.asarray(row[metric], dtype=float)},
                    {metric: np.asarray(old[row["method"]][metric], dtype=float)}, identity, "metric_" + row["method"]))
                rows.extend(comparison_rows({metric: np.asarray(recomputed[metric], dtype=float)},
                    {metric: np.asarray(row[metric], dtype=float)}, identity, "recomputed_metric_" + row["method"]))
    return pd.DataFrame(rows)


def interval(values, identity):
    v = np.asarray(values, dtype=float); v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan
    if np.all(v == v[0]):
        return float(v[0]), float(v[0])
    words = np.frombuffer(hashlib.sha256(json.dumps(identity, default=str).encode()).digest()[:16], dtype="<u4")
    rng = np.random.default_rng(np.random.SeedSequence([20261025, *map(int, words)]))
    means = v[rng.integers(len(v), size=(10000, len(v)))].mean(axis=1)
    return tuple(map(float, np.quantile(means, [.025, .975])))


def summaries(detail):
    absolute, paired = [], []
    for duration, frame in detail.groupby("duration", sort=True):
        methods = {m: f.set_index("repeat_seed").sort_index() for m, f in frame.groupby("method", sort=False)}
        raw = methods["raw_window"]
        for method, rows in methods.items():
            if rows.index.has_duplicates or not rows.index.equals(raw.index):
                raise ValueError("Missing or duplicate paired method/repeat")
            for metric in METRICS:
                values = rows[metric].to_numpy(float); valid = np.isfinite(values)
                low, high = interval(values[valid], (duration, method, metric, "absolute"))
                absolute.append({"duration": duration, "method": method, "metric": metric, "n_planned": len(rows),
                    "n_finite": int(valid.sum()), "n_missing": int((~valid).sum()), "mean": float(values[valid].mean()) if valid.any() else np.nan,
                    "ci95_low": low, "ci95_high": high})
                for reference in ("raw_window", "forced_hard"):
                    ref = methods[reference].loc[rows.index, metric].to_numpy(float); usable = valid & np.isfinite(ref)
                    delta = values[usable] - ref[usable]
                    low, high = interval(delta, (duration, method, reference, metric, "paired"))
                    paired.append({"duration": duration, "method": method, "reference": reference, "metric": metric,
                        "n_planned": len(rows), "n_paired": int(usable.sum()), "n_missing_pair": int((~usable).sum()),
                        "mean": float(delta.mean()) if len(delta) else np.nan, "ci95_low": low, "ci95_high": high})
    return pd.DataFrame(absolute), pd.DataFrame(paired)


def derived(root, output):
    rows, records = [], []
    for path in sorted((output / "parts").glob("*.json")):
        record = read_json(path); records.append(record); values = arrays(path.with_suffix(".npz"))
        emitted = {m: values["emission_" + m] for m in METHODS if "emission_" + m in values}
        evaluated = evaluate(emitted, values["window_scores"], values["bounds"], values["regimes"], values["state_support"])
        methods = {x["method"]: x for x in evaluated}
        for method in METHODS:
            row = methods.get(method)
            if row is None:
                # Target opportunities remain known when a method fit fails.
                row = {key: value for key, value in methods["raw_window"].items()
                       if key.startswith(("n_", "changing_edges_", "covered_", "raw_change_threshold"))}
                row.update({"method": method, "status": "fit_failed", "boundary_ap_edges_used": 0,
                            "boundary_f1_edges_used": 0, **{m: np.nan for m in METRICS}})
            rows.append({"repeat_seed": record["repeat_seed"], "duration": record["duration"],
                **row, "fit_status": record["status"] if method == "pelt_var_adapter" else "retained_reference",
                "fit_error": record["error"] if method == "pelt_var_adapter" else "",
                "n_detected_segments": record["diagnostic"].get("n_detected_segments", np.nan),
                "n_refit_segments": record["diagnostic"].get("n_refit_segments", np.nan)})
    detail = pd.DataFrame(rows); summary, contrasts = summaries(detail)
    fresh = fresh_comparisons(root, output)
    diagnostics = {"role": config()["role"], "planned_adapter_cases": 60, "retained_adapter_cases": len(records),
        "failed_adapter_cases": sum(r["status"] != "ok" for r in records), "method_rows": len(detail),
        "detected_segment_counts": {str(d): [r["diagnostic"].get("n_detected_segments") for r in records if r["duration"] == d] for d in (300, 600)},
        "new_adapter_detector_fits": len(records), "new_adapter_segment_refits": sum(r["diagnostic"].get("n_refit_segments", 0) for r in records),
        "fresh_cases": len(list((output / "fresh").glob("*.json"))),
        "fresh_comparisons": fresh.classification.value_counts().to_dict(),
        "fresh_all_within_tolerance": bool(fresh.within_tolerance.all()), "no_new_independent_seeds": True}
    return {"detail.csv": detail, "summary.csv": summary, "contrasts.csv": contrasts,
            "fresh_comparisons.csv": fresh, "diagnostics.json": diagnostics}


def write_derived(root, output):
    for name, value in derived(root, output).items():
        if name.endswith(".csv"):
            value.to_csv(output / name, index=False)
        else:
            write_json(output / name, value)


def seal(output):
    write_json(output / "manifest.json", {"sha256": tree_hashes(output, ("manifest.json",)),
        "protocol_sha256": sha(output / "protocol.json"), "role": config()["role"]})


def run(root=ROOT, output=OUT, stage=None):
    root, output = Path(root).resolve(), Path(output).resolve(); check_protocol(root, output)
    if set(tree_hashes(output)) != {"protocol.json", "execution/freeze_record.json"}:
        raise ValueError("Run requires protocol/freeze records only; no retries or overwrites")
    for name in ("parts", "fresh"):
        (output / name).mkdir()
    environment = {"python": platform.python_version(), "platform": platform.platform(), "threads": 1,
        "versions": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scikit-learn", "scipy", "ruptures", "threadpoolctl")}}
    write_json(output / "execution/environment.json", environment)
    record = {"before_fitting_utc": utc_now(), "protocol_sha256": sha(output / "protocol.json"),
              "new_fitting": True, "fresh_generation": True}
    write_json(output / "execution/run_record.json", record); started = perf_counter()
    with threadpool_limits(limits=1):
        for seed in config()["seeds"]:
            for duration in config()["durations"]:
                item = run_adapter_case(root, output, seed, duration)
                print(f"PELT retained case {seed} D{duration}: {item['status']}", flush=True)
        for study, seeds in (("A", [50000, 50001]), ("C", [70000, 70001])):
            for seed in seeds:
                for duration in (300, 600):
                    item = run_fresh_case(root, output, study, seed, duration)
                    print(f"Fresh reproduction {study} {seed} D{duration}: {item['status']}", flush=True)
        write_derived(root, output)
    check_protocol(root, output)
    record.update(after_fitting_and_aggregation_utc=utc_now(), elapsed_seconds=perf_counter() - started, complete=True)
    write_json(output / "execution/run_record.json", record)
    if stage is not None:
        stage = Path(stage); stage.mkdir(parents=True, exist_ok=True)
        target = stage / "segmentation_run.json"
        if target.exists():
            raise ValueError("Refuse to overwrite a stage execution record")
        write_json(target, record)
    seal(output)
    return read_json(output / "diagnostics.json")


def verify(root=ROOT, output=OUT):
    root, output = Path(root).resolve(), Path(output).resolve(); check_protocol(root, output)
    before = tree_hashes(output); manifest = read_json(output / "manifest.json")
    if {k: v for k, v in before.items() if k != "manifest.json"} != manifest["sha256"]:
        raise ValueError("Segmentation output inventory/hash mismatch")
    expected = {(s, d) for s in range(70000, 70030) for d in (300, 600)}; seen = set()
    for path in sorted((output / "parts").glob("*.json")):
        rec = read_json(path); key = (rec["repeat_seed"], rec["duration"])
        if key in seen:
            raise ValueError("Duplicate adapter case")
        seen.add(key); v = arrays(path.with_suffix(".npz")); old, source = retained_case(root, *key)
        if source != rec["source"] or sha(root / source) != rec["source_sha256"] or sha(path.with_suffix(".npz")) != rec["arrays_sha256"]:
            raise ValueError("Adapter reference/part provenance drift")
        for name in ("window_scores", "bounds", "regimes", "state_support", "emission_raw_window", "emission_forced_hard"):
            np.testing.assert_array_equal(v[name], old[name])
        if rec["status"] == "ok":
            emitted, labels = range_emission(v["segment_maps"], v["segment_ranges"], v["bounds"], len(old["observed_series"]))
            np.testing.assert_array_equal(emitted, v["emission_pelt_var_adapter"])
            np.testing.assert_array_equal(labels, v["refit_window_labels"])
            bp = v["pelt_breakpoints"]
            if bp[-1] != len(v["bounds"]) or np.any(np.diff(np.r_[0, bp]) < 2):
                raise ValueError("PELT breakpoints violate min_size or final endpoint")
            expected_labels = np.repeat(np.arange(len(bp)), np.diff(np.r_[0, bp]))
            np.testing.assert_array_equal(v["pelt_labels"], expected_labels)
            if rec["diagnostic"]["n_detected_segments"] != len(bp) or rec["diagnostic"]["n_refit_segments"] != len(v["segment_ranges"]):
                raise ValueError("Saved segmentation counts disagree with arrays")
    if seen != expected:
        raise ValueError("Incomplete planned adapter cases")
    fresh_records = [read_json(p) for p in sorted((output / "fresh").glob("*.json"))]
    planned = {(st, s, d) for st, seeds in (("A", [50000, 50001]), ("C", [70000, 70001])) for s in seeds for d in (300, 600)}
    if {(x["study"], x["repeat_seed"], x["duration"]) for x in fresh_records} != planned or len(fresh_records) != 8:
        raise ValueError("Fresh reproduction case inventory differs")
    with threadpool_limits(limits=1):
        rebuilt = derived(root, output)
    for name, value in rebuilt.items():
        if name.endswith(".csv"):
            actual = pd.read_csv(output / name, keep_default_na=False)
            expected_frame = pd.read_csv(io.StringIO(value.to_csv(index=False)), keep_default_na=False)
            pd.testing.assert_frame_equal(actual, expected_frame, check_exact=True)
        elif json_bytes(value) != (output / name).read_bytes():
            raise ValueError(f"Derived diagnostic mismatch: {name}")
    if before != tree_hashes(output):
        raise AssertionError("Read-only verification changed an output")
    return {"status": "passed", "read_only": True, "adapter_cases": len(seen), "fresh_reproduction_cases": 8,
            "files_unchanged": len(before), "new_fits": 0, "new_simulations": 0, **rebuilt["diagnostics.json"]}


def rebuild(root=ROOT, output=OUT):
    root, output = Path(root).resolve(), Path(output).resolve()
    verify(root, output)  # Integrity must hold before any deterministic display/CSV rewrite.
    write_derived(root, output); seal(output)
    return {"derived_files": list(DERIVED), "new_fits": 0, "new_simulations": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    for mode in ("freeze", "run", "verify", "rebuild"):
        modes.add_argument("--" + mode, action="store_true")
    p.add_argument("--root", type=Path, default=ROOT); p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--stage-dir", type=Path)
    a = p.parse_args()
    if a.freeze:
        result = freeze(a.root, a.output_dir, a.stage_dir)
    elif a.run:
        result = run(a.root, a.output_dir, a.stage_dir)
    elif a.verify:
        result = verify(a.root, a.output_dir)
    else:
        result = rebuild(a.root, a.output_dir)
    print(json.dumps(clean(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
