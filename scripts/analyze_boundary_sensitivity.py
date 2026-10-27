"""Post hoc boundary AP/threshold sensitivity and residual pair accounting.

This extension reads retained A/B/C predictions only. All quantiles and mixing
weights are fixed after seeing previous results, with no truth-based choice of
a best threshold and no new false-alarm calibration or confirmatory family.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_temporal_localisation as loc
import analyze_score_preserving_controls as controls

OUT = ROOT / "results/information_sciences/boundary_sensitivity"
LOG = ROOT / "experiments/logs/boundary_sensitivity_2026-10-25"
QUANTILES = (.80, .90, .95, .975, .99)
LAMBDAS = (0., 1e-8, .25, .5, .75, 1.)
BASE = ["study", "scorer", "duration", "schedule", "cell"]
AP_METRICS = ["boundary_positive_change_ap", "boundary_standard_ap", "boundary_zero_group_increment"]
DETECTION_METRICS = ["boundary_slot_f1", "stable_false_change_rate", "persistent_false_change_rate",
                     "absent_false_change_rate", "stable_any_change_edge_fraction"]
PAIR_METRICS = ["hard_state_auroc", "residual_state_auroc", "within_auc_delta", "across_auc_delta", "total_auc_delta"]


def average_precision_components(truth, strength) -> dict:
    """Noninterpolated tied-group AP, its positive-only part, and zero-group term."""
    y, score = np.asarray(truth, bool), np.asarray(strength, float)
    if y.ndim != 1 or score.shape != y.shape:
        raise ValueError("AP needs aligned one-dimensional boundary slots")
    if not np.isfinite(score).all() or np.any(score < 0):
        return {key: np.nan for key in ("standard_ap", "positive_ap", "prevalence", "positive_recall", "zero_increment", "identity_gap")}
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        return {key: np.nan for key in ("standard_ap", "positive_ap", "prevalence", "positive_recall", "zero_increment", "identity_gap")}
    order = np.argsort(-score, kind="stable")
    values, labels = score[order], y[order]
    ends = np.r_[np.flatnonzero(values[:-1] != values[1:]) + 1, len(values)]
    standard, positive, previous_hits = 0., 0., 0
    for stop in ends:
        hits = int(labels[:stop].sum())
        contribution = (hits - previous_hits) / positives * hits / int(stop)
        standard += contribution
        if values[stop - 1] > 0:
            positive += contribution
        previous_hits = hits
    prevalence = positives / len(y)
    recall = float(y[score > 0].sum() / positives)
    increment = prevalence * (1 - recall)
    return {"standard_ap": float(standard), "positive_ap": float(positive), "prevalence": prevalence,
            "positive_recall": recall, "zero_increment": increment,
            "identity_gap": float(abs(standard - positive - increment))}


def detection_metrics(detected: np.ndarray, grid: dict) -> tuple[dict, dict]:
    """Macro changing-edge F1 and stable-edge exceedances at one declared threshold."""
    detected = np.asarray(detected, bool)
    truth = grid["truth_boundary_slots"]
    if detected.shape != truth.shape:
        raise ValueError("Detected events must match boundary slots")
    slots = len(truth)
    count = truth.sum(axis=0)
    eligible = grid["changing_edges"] & (count > 0)
    tp = (detected & truth).sum(axis=0); fp = (detected & ~truth).sum(axis=0); fn = (~detected & truth).sum(axis=0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.full(len(count), np.nan), where=(2 * tp + fp + fn) > 0)
    events = detected.sum(axis=0)
    persistent, absent = grid["persistent_edges"], grid["absent_edges"]
    stable = persistent | absent
    ratio = lambda a, b: float(a / b) if b else np.nan
    row = {"boundary_slot_f1": float(f1[eligible].mean()) if eligible.any() else np.nan,
           "boundary_f1_edges_used": int(eligible.sum()), "boundary_tp": int(tp[eligible].sum()),
           "boundary_fp": int(fp[eligible].sum()), "boundary_fn": int(fn[eligible].sum()),
           "stable_false_change_rate": ratio(events[stable].sum(), int(stable.sum()) * slots),
           "persistent_false_change_rate": ratio(events[persistent].sum(), int(persistent.sum()) * slots),
           "absent_false_change_rate": ratio(events[absent].sum(), int(absent.sum()) * slots),
           "stable_any_change_edge_fraction": ratio((events[stable] > 0).sum(), int(stable.sum())),
           "stable_false_events": int(events[stable].sum()), "stable_event_opportunities": int(stable.sum()) * slots,
           "persistent_false_events": int(events[persistent].sum()), "persistent_event_opportunities": int(persistent.sum()) * slots,
           "absent_false_events": int(events[absent].sum()), "absent_event_opportunities": int(absent.sum()) * slots,
           "stable_edges": int(stable.sum()), "persistent_edges": int(persistent.sum()), "absent_edges": int(absent.sum())}
    return row, {"tp": tp, "fp": fp, "fn": fn, "events": events}


def evaluate_scores(emissions: dict, raw, bounds, regimes, state_support, *, quantiles=QUANTILES):
    """Pure boundary evaluator returning AP rows, threshold rows and per-edge arrays."""
    grid = loc.boundary_grid(bounds, regimes, state_support)
    raw = np.asarray(raw, float)
    shape = (len(bounds), *np.asarray(state_support).shape[1:])
    if raw.shape != shape or not emissions or any(not 0 < q < 1 for q in quantiles):
        raise ValueError("Invalid raw grid, method inventory or quantile")
    mask = grid["mask"]; truth = grid["truth_boundary_slots"]
    raw_strength = np.abs(np.diff(raw[:, mask], axis=0))
    if not np.isfinite(raw_strength).all():
        raise ValueError("Nonfinite raw scores cannot define a shared threshold")
    true_count = truth.sum(axis=0)
    eligible = grid["changing_edges"] & (true_count > 0) & (true_count < len(truth))
    rows, threshold_rows = [], []
    per_edge = {key: [] for key in ("positive_ap", "standard_ap", "prevalence", "positive_recall", "zero_increment", "identity_gap")}
    for method, prediction in emissions.items():
        prediction = np.asarray(prediction, float)
        if prediction.shape != shape or not np.isfinite(prediction[:, mask]).all():
            raise ValueError(f"Nonfinite/misaligned method emission: {method}")
        strength = np.abs(np.diff(prediction[:, mask], axis=0))
        components = {key: np.full(truth.shape[1], np.nan) for key in per_edge}
        for edge in np.flatnonzero(eligible):
            component = average_precision_components(truth[:, edge], strength[:, edge])
            for key in components:
                components[key][edge] = component[key]
        mean = lambda key: float(components[key][eligible].mean()) if eligible.any() else np.nan
        row = {"method": method, "status": "ok", "n_windows": len(bounds), "n_slots": len(truth),
               "changing_edges_total": int(grid["changing_edges"].sum()), "boundary_ap_edges_used": int(eligible.sum()),
               "changing_edges_no_covered_boundary": int((grid["changing_edges"] & (true_count == 0)).sum()),
               "changing_edges_all_slots_boundary": int((grid["changing_edges"] & (true_count == len(truth))).sum()),
               "boundary_positive_change_ap": mean("positive_ap"), "boundary_standard_ap": mean("standard_ap"),
               "boundary_prevalence_mean": mean("prevalence"), "boundary_positive_recall_mean": mean("positive_recall"),
               "boundary_zero_group_increment": mean("zero_increment"),
               "boundary_ap_identity_max_gap": float(np.nanmax(components["identity_gap"])) if eligible.any() else np.nan,
               "support_events_collapsed": grid["support_events_collapsed"], "covered_start": grid["covered_start"],
               "covered_stop_inclusive": grid["covered_stop_inclusive"]}
        rows.append(row)
        for key in per_edge:
            per_edge[key].append(components[key])
        for q in quantiles:
            raw_threshold = float(np.quantile(raw_strength, q, method="linear"))
            method_threshold = float(np.quantile(strength, q, method="linear"))
            for mode, threshold in (("raw_shared", raw_threshold), ("method_self", method_threshold)):
                metrics, _ = detection_metrics(strength > threshold, grid)
                threshold_rows.append({"method": method, "status": "ok", "q": float(q), "threshold_mode": mode,
                    "threshold": threshold, "raw_threshold": raw_threshold, "method_threshold": method_threshold,
                    "threshold_zero": threshold == 0, "n_slots": len(truth), **metrics})
    arrays = {key: np.stack(values) for key, values in per_edge.items()}
    arrays.update({"method_names": np.asarray(list(emissions)), "edge_indices": grid["edge_indices"],
                   "truth_event_counts": grid["truth_event_counts"], "persistent_edges": grid["persistent_edges"],
                   "absent_edges": grid["absent_edges"], "centres": grid["centres"]})
    return rows, threshold_rows, arrays


def evaluate_methods(emissions: dict, raw, bounds, regimes, state_support, *, quantiles=(.95,)) -> list[dict]:
    """Reusable external-framework API; no fitting, simulation or output writes."""
    ap_rows, threshold_rows, _ = evaluate_scores(emissions, raw, bounds, regimes, state_support, quantiles=quantiles)
    support = np.asarray(state_support)
    centres = (np.asarray(bounds)[:, 0] + np.asarray(bounds)[:, 1] - 1) // 2
    active = support[np.asarray(regimes)[centres]]
    changing = np.any(support != support[0], axis=0)
    output = []
    for row in ap_rows:
        method = row["method"]
        result = {**row, **controls.score_metrics(emissions[method], active, changing)}
        for q in quantiles:
            tag = "q" + (f"{q * 100:g}".replace(".", "p"))
            for mode, label in (("raw_shared", "raw"), ("method_self", "method")):
                selected = [r for r in threshold_rows if r["method"] == method and r["q"] == q and r["threshold_mode"] == mode][0]
                for metric in DETECTION_METRICS:
                    result[f"{metric}_{label}_{tag}"] = selected[metric]
                result[f"{label}_change_threshold_{tag}"] = selected["threshold"]
                result["boundary_f1_edges_used"] = selected["boundary_f1_edges_used"]
        output.append(result)
    return output


def residual_pair_decomposition(raw, hard, candidate_labels, active_truth, changing_mask, *, lambdas=LAMBDAS):
    """Exact positive/negative pair credit accounting, normalised per edge first."""
    raw, hard = np.asarray(raw, float), np.asarray(hard, float)
    y = np.asarray(active_truth, bool); labels = np.asarray(candidate_labels)
    if raw.shape != hard.shape or y.shape != raw.shape or raw.ndim != 4 or labels.shape != (len(raw),):
        raise ValueError("Residual accounting needs aligned raw/hard/truth and candidate labels")
    mask = controls.nonself(raw.shape[1:]); raw = raw[:, mask]; hard = hard[:, mask]; y = y[:, mask]
    changing = np.asarray(changing_mask, bool)[mask]
    if not np.isfinite(raw).all() or not np.isfinite(hard).all():
        raise ValueError("Pair accounting requires finite scores")
    # A fixed map within each candidate group is required for its interpretation.
    for group in np.unique(labels):
        if not np.all(hard[labels == group] == hard[labels == group][0]):
            raise ValueError("Hard emission must be constant within each candidate group")
    pairs = y[:, None, :] & ~y[None, :, :]
    denominator = pairs.sum(axis=(0, 1)); eligible = changing & (denominator > 0)
    within = (labels[:, None] == labels[None, :])[:, :, None]
    sign = lambda values: (values[:, None, :] > values[None, :, :]).astype(np.int8) - (values[:, None, :] < values[None, :, :]).astype(np.int8)
    hard_sign, raw_sign = sign(hard), sign(raw)
    hard_credit = ((.5 + .5 * hard_sign) * pairs).sum(axis=(0, 1))
    rows, arrays = [], {key: [] for key in ("within_auc_delta", "across_auc_delta", "total_auc_delta", "hard_state_auroc", "residual_state_auroc",
        "within_lost_ties", "across_lost_ties", "within_new_ties", "across_new_ties", "within_reversed_pairs", "across_reversed_pairs", "within_raw_order_mismatches")}
    for lam in lambdas:
        value = hard.copy() if lam == 0 else (raw.copy() if lam == 1 else lam * raw + (1 - lam) * hard)
        current = sign(value); change = (current - hard_sign) * .5
        numerators = {"within_auc_delta": (change * pairs * within).sum(axis=(0, 1)),
                      "across_auc_delta": (change * pairs * ~within).sum(axis=(0, 1)),
                      "total_auc_delta": (change * pairs).sum(axis=(0, 1)),
                      "hard_state_auroc": hard_credit,
                      "residual_state_auroc": ((.5 + .5 * current) * pairs).sum(axis=(0, 1))}
        components = {key: np.divide(numerator, denominator, out=np.full(len(denominator), np.nan), where=eligible)
                      for key, numerator in numerators.items()}
        events = {"lost_ties": (hard_sign == 0) & (current != 0), "new_ties": (hard_sign != 0) & (current == 0),
                  "reversed_pairs": hard_sign * current == -1}
        for scope, selection in (("within", within), ("across", ~within)):
            for key, event in events.items():
                components[f"{scope}_{key}"] = (pairs * selection * event).sum(axis=(0, 1))
        components["within_raw_order_mismatches"] = (pairs * within * (current != raw_sign)).sum(axis=(0, 1))
        row = {"method": f"residual_{lam:g}", "lambda": float(lam), "status": "ok", "n_evaluable_edges": int(eligible.sum()),
               "n_changing_edges": int(changing.sum()), "positive_negative_pairs": int(denominator[eligible].sum()),
               "within_positive_negative_pairs": int((pairs * within)[:, :, eligible].sum()),
               "across_positive_negative_pairs": int((pairs * ~within)[:, :, eligible].sum())}
        for key, values in components.items():
            row[key] = float(values[eligible].mean()) if key in PAIR_METRICS and eligible.any() else (np.nan if key in PAIR_METRICS else int(values[eligible].sum()))
            arrays[key].append(values)
        row["decomposition_max_gap"] = float(np.nanmax(abs(components["total_auc_delta"] - components["within_auc_delta"] - components["across_auc_delta"]))) if eligible.any() else np.nan
        row["auroc_difference_max_gap"] = float(np.nanmax(abs(components["total_auc_delta"] - components["residual_state_auroc"] + components["hard_state_auroc"]))) if eligible.any() else np.nan
        if eligible.any():
            assert row["decomposition_max_gap"] < 2e-14 and row["auroc_difference_max_gap"] < 2e-14
        rows.append(row)
    arrays = {key: np.stack(value) for key, value in arrays.items()}
    arrays.update({"lambdas": np.asarray(lambdas), "eligible_edges": eligible, "positive_negative_pairs": denominator,
                   "edge_indices": np.array(np.nonzero(mask)).T})
    return rows, arrays


def iter_cases(root: Path = ROOT):
    """Original 1000 retained cases only; independent external fits stay separate."""
    root = Path(root)
    for case in controls.iter_retained_cases(root):
        if case["status"] == "ok":
            name = (f"duration_confirmation/records/repeat_{case['repeat_seed']}_D{case['duration']}_{case['scorer']}.npz"
                    if case["study"] == "A" else f"schedule_robustness/parts/repeat_{case['repeat_seed']}_D{case['duration']}_{case['schedule']}.npz")
            with np.load(root / "results/information_sciences" / name, allow_pickle=False) as saved:
                candidate = saved["candidate_labels"]
            case = {**case, "candidate_labels": candidate}
        yield case
    import run_symmetric_sensitivity as symmetric
    for item in symmetric.iter_cases(root):
        identity, arrays = item["identity"], item["arrays"]
        case = {"study": "C", "scorer": "lasso", "repeat_seed": identity["seed"], "duration": identity["duration"],
                "schedule": "fixed_phase", "cell": identity["cell"], "status": item["fit_status"],
                "error": item["record"]["diagnostic"].get("error") or ""}
        if item["fit_status"] == "ok":
            centres = (arrays["bounds"][:, 0] + arrays["bounds"][:, 1] - 1) // 2
            support = arrays["state_support"]
            case.update({"raw": arrays["window_scores"], "forced": arrays["emission_forced_hard"],
                         "native": arrays["emission_native_hard"], "oracle": arrays["emission_oracle_label"],
                         "bounds": arrays["bounds"], "regimes": arrays["regimes"], "state_support": support,
                         "active_truth": support[arrays["regimes"][centres]], "changing_mask": np.any(support != support[0], axis=0),
                         "candidate_labels": arrays["candidate_labels"], "gate_accepted": item["gate_accepted"]})
        yield case


def compute_case(case: dict, *, quantiles=QUANTILES, lambdas=LAMBDAS):
    """Pure one-case A/B/C analysis; reusable by separately reported frameworks."""
    identity = {key: case[key] for key in [*BASE, "repeat_seed"]}
    if case["status"] != "ok":
        ap = [{"method": method, "status": "retained_fit_failure", **{k: np.nan for k in AP_METRICS}} for method in loc.METHODS]
        threshold = [{"method": method, "q": q, "threshold_mode": mode, "status": "retained_fit_failure",
                      **{k: np.nan for k in DETECTION_METRICS}} for method in loc.METHODS for q in quantiles for mode in ["raw_shared", "method_self"]]
        pair = [{"method": f"residual_{lam:g}", "lambda": lam, "status": "retained_fit_failure",
                 **{k: np.nan for k in PAIR_METRICS}} for lam in lambdas]
        return {"ap": [{**identity, **row} for row in ap], "threshold": [{**identity, **row} for row in threshold],
                "residual": [{**identity, **row} for row in pair]}, {}
    raw = case["raw"]
    identity.update({"h": int(case["bounds"][0, 1] - case["bounds"][0, 0]),
                     "stride": int(case["bounds"][1, 0] - case["bounds"][0, 0]), "M": len(raw)})
    emissions = controls.build_control_emissions(raw, case["forced"], case["native"], case["gate_accepted"])
    emissions["oracle_label"] = case["oracle"]
    emissions["static_mean"] = np.repeat(raw.mean(axis=0)[None], len(raw), axis=0)
    ap, threshold, ap_arrays = evaluate_scores(emissions, raw, case["bounds"], case["regimes"], case["state_support"], quantiles=quantiles)
    pair, pair_arrays = residual_pair_decomposition(raw, case["forced"], case["candidate_labels"], case["active_truth"], case["changing_mask"], lambdas=lambdas)
    rows = {name: [{**identity, **row} for row in table] for name, table in [("ap", ap), ("threshold", threshold), ("residual", pair)]}
    arrays = {**{"ap_" + key: value for key, value in ap_arrays.items()}, **{"residual_" + key: value for key, value in pair_arrays.items()}}
    return rows, arrays


def _bootstrap_columns(values, identity, cache):
    """Reuse one 10k repeat-index matrix per cell/finite mask for every column."""
    values = np.asarray(values, float)
    means = np.full(values.shape[1], np.nan); low = means.copy(); high = means.copy()
    finite = np.isfinite(values); n = finite.sum(axis=0)
    patterns = {}
    for column in range(values.shape[1]):
        patterns.setdefault(finite[:, column].tobytes(), []).append(column)
    for mask_bytes, columns in patterns.items():
        mask = finite[:, columns[0]]
        if not mask.any():
            continue
        subset = values[mask][:, columns]
        means[columns] = subset.mean(axis=0)
        constant = np.all(subset == subset[0], axis=0)
        constants = np.asarray(columns)[constant]
        low[constants] = high[constants] = means[constants]
        if constant.all():
            continue
        key = (tuple(map(str, identity)), mask_bytes)
        if key not in cache:
            token = json.dumps([list(map(str, identity)), mask.tolist()], separators=(",", ":")).encode()
            words = np.frombuffer(hashlib.sha256(token).digest()[:16], dtype="<u4")
            rng = np.random.default_rng(np.random.SeedSequence([20261025, *map(int, words)]))
            size = int(mask.sum())
            index = rng.integers(size, size=(10000, size))
            counts = np.bincount((index + np.arange(10000)[:, None] * size).ravel(), minlength=10000 * size)
            cache[key] = counts.reshape(10000, size).astype(float) / size
        columns_variable = np.asarray(columns)[~constant]
        bootstrap = cache[key] @ subset[:, ~constant]
        bounds = np.quantile(bootstrap, [.025, .975], axis=0)
        low[columns_variable], high[columns_variable] = bounds
    return means, low, high, n


def summarise_block(detail: pd.DataFrame, metrics, extra=(), *, reference="raw_window", cache=None):
    cache = {} if cache is None else cache
    absolute, paired = [], []
    for identity, frame in detail.groupby(BASE, sort=True, dropna=False):
        seeds = sorted(frame.repeat_seed.unique())
        records, columns = [], []
        index = frame.set_index(["repeat_seed", "method", *extra])
        if index.index.has_duplicates:
            raise ValueError("Duplicate repeat/method/setting in sensitivity analysis")
        for setting, rows in frame.groupby(["method", *extra], sort=True, dropna=False):
            setting = setting if isinstance(setting, tuple) else (setting,)
            method, *params = setting
            rows = rows.set_index("repeat_seed").reindex(seeds)
            if rows.method.isna().any():
                raise ValueError("Sensitivity analysis omits a planned method row")
            baseline_setting = (reference, *([0.] if reference == "residual_0" else params))
            baseline = frame
            for key, value in zip(["method", *extra], baseline_setting):
                baseline = baseline[baseline[key] == value]
            baseline = baseline.set_index("repeat_seed").reindex(seeds)
            if baseline.method.isna().any():
                raise ValueError("Missing paired reference in sensitivity analysis")
            for metric in metrics:
                raw_values = rows[metric].to_numpy(float)
                difference = raw_values - baseline[metric].to_numpy(float)
                record = {**dict(zip(BASE, identity)), "method": method, **dict(zip(extra, params)), "metric": metric,
                          "n_planned": len(seeds), "reference": reference}
                records.append(record)
                columns.extend([raw_values, difference])
        matrix = np.stack(columns, axis=1)
        means, lower, upper, counts = _bootstrap_columns(matrix, identity, cache)
        for number, record in enumerate(records):
            for offset, destination in [(0, absolute), (1, paired)]:
                column = 2 * number + offset
                destination.append({**record, "n_finite" if offset == 0 else "n_paired": int(counts[column]),
                                    "n_missing" if offset == 0 else "n_missing_pairs": int(len(seeds) - counts[column]),
                                    "mean": means[column], "ci95_low": lower[column], "ci95_high": upper[column]})
    return pd.DataFrame(absolute), pd.DataFrame(paired)


def summarize(frames: dict) -> dict:
    cache = {}
    with threadpool_limits(limits=1):
        ap, ap_pair = summarise_block(frames["ap_detail.csv"], AP_METRICS, cache=cache)
        threshold, threshold_pair = summarise_block(frames["threshold_detail.csv"], DETECTION_METRICS, ["q", "threshold_mode"], cache=cache)
        residual, residual_pair = summarise_block(frames["residual_detail.csv"], PAIR_METRICS, ["lambda"], reference="residual_0", cache=cache)
    return {"ap_summary.csv": ap, "ap_paired_vs_raw.csv": ap_pair,
            "threshold_summary.csv": threshold, "threshold_paired_vs_raw.csv": threshold_pair,
            "residual_summary.csv": residual, "residual_paired_vs_hard.csv": residual_pair}


def compute(root: Path = ROOT, *, write_records: Path | None = None, verify_records: Path | None = None):
    rows = {"ap": [], "threshold": [], "residual": []}; cases = failures = 0
    max_ap_gap = max_pair_gap = 0.
    for case in iter_cases(root):
        part, arrays = compute_case(case)
        for name in rows:
            rows[name].extend(part[name])
        filename = f"{case['study']}_{case['scorer']}_{case['repeat_seed']}_{case['cell']}.npz"
        if write_records is not None and arrays:
            np.savez_compressed(write_records / filename, **arrays)
        if verify_records is not None and arrays:
            with np.load(verify_records / filename, allow_pickle=False) as stored:
                if set(stored.files) != set(arrays):
                    raise ValueError("Sensitivity per-edge cache inventory drift")
                for key, value in arrays.items():
                    if value.dtype.kind in "US":
                        np.testing.assert_array_equal(stored[key], value)
                    else:
                        np.testing.assert_allclose(stored[key], value, rtol=1e-12, atol=2e-14, equal_nan=True)
        for row in part["ap"]:
            if np.isfinite(row.get("boundary_ap_identity_max_gap", np.nan)):
                max_ap_gap = max(max_ap_gap, row["boundary_ap_identity_max_gap"])
        for row in part["residual"]:
            if np.isfinite(row.get("decomposition_max_gap", np.nan)):
                max_pair_gap = max(max_pair_gap, row["decomposition_max_gap"], row["auroc_difference_max_gap"])
        cases += 1; failures += int(case["status"] != "ok")
        if write_records is not None and cases % 50 == 0:
            print(f"retained sensitivity cases processed: {cases}", flush=True)
    frames = {name + "_detail.csv": pd.DataFrame(value) for name, value in rows.items()}
    frames.update(summarize(frames))
    return frames, {"cases": cases, "retained_failures": failures, "ap_rows": len(rows["ap"]),
                    "threshold_rows": len(rows["threshold"]), "residual_rows": len(rows["residual"]),
                    "AP_identity_max_gap": max_ap_gap, "residual_identity_max_gap": max_pair_gap,
                    "new_fits": 0, "new_simulations": 0}


def directory_hashes(directory: Path) -> dict:
    return {p.relative_to(directory).as_posix(): loc.sha(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def input_snapshot(root: Path) -> dict:
    root = Path(root)
    original = controls.input_snapshot(root)
    c_dir = root / "results/information_sciences/symmetric_sensitivity"
    c_protocol = json.loads((c_dir / "symmetric_sensitivity_protocol.json").read_text(encoding="utf-8"))
    c_sources = {**c_protocol["source_sha256"], "configs/symmetric_sensitivity.yaml": c_protocol["configuration_sha256"]}
    for name, digest in c_sources.items():
        if loc.sha(root / name) != digest:
            raise ValueError("Original C frozen source changed: " + name)
    c_seal = json.loads((c_dir / "symmetric_sensitivity_outputs.json").read_text(encoding="utf-8"))
    for name, digest in c_seal["sha256"].items():
        if loc.sha(c_dir / name) != digest:
            raise ValueError("Original C sealed output changed: " + name)
    previous = {}
    for directory in ("score_preserving_controls", "temporal_localisation", "symmetric_controls", "symmetric_localisation"):
        base = root / "results/information_sciences" / directory
        previous.update({p.relative_to(root).as_posix(): loc.sha(p) for p in base.rglob("*") if p.is_file()})
    return {"original_AB_guard": original, "original_C_files_sha256": {p.relative_to(root).as_posix(): loc.sha(p) for p in c_dir.rglob("*") if p.is_file()},
            "original_C_sources_sha256": c_sources, "previous_analysis_files_sha256": previous}


SOURCES = ("scripts/analyze_boundary_sensitivity.py", "tests/test_boundary_sensitivity.py", "scripts/analyze_temporal_localisation.py",
           "scripts/analyze_score_preserving_controls.py", "scripts/run_symmetric_sensitivity.py")


def protocol(input_digest: str, root: Path = ROOT) -> dict:
    return {"analysis": "posthoc_boundary_sensitivity_and_residual_pairs_v1",
            "role": "Post hoc extension after observing all preceding A/B/C outcomes; not an independent confirmation",
            "source_sha256": {name: loc.sha(Path(root) / name) for name in SOURCES}, "input_manifest_sha256": input_digest,
            "cases": "Original A400 fits (Lasso/OLS), B300 and C300; 1000 fits from paired repeats. External validation fits are separate.",
            "methods": list(loc.METHODS), "quantiles": list(QUANTILES), "residual_lambdas": list(LAMBDAS),
            "AP": {"standard": "Noninterpolated AP over every distinct score including final zero-strength group, whole-group ties",
                   "positive": "Old positive-change AP with zero-strength abstention, unchanged",
                   "identity": "For each edge AP_standard=AP_plus+prevalence*(1-positive_strength_recall). Average the product, not marginal means.",
                   "coverage": "No positive or no negative slot remains missing for BOTH AP conventions, including C D300/stride300"},
            "thresholds": {"modes": ["raw_shared", "method_self"], "pool": "All nonself edge-slot abs differences within one fit",
                           "quantile_interpolation": "linear", "event": "strict strength>threshold", "labels_used": False,
                           "interpretation": "Fixed descriptive amplitude or method-relative reference, not independently calibrated false-positive rates; no best-q selection"},
            "residual": {"arithmetic": "lambda*raw+(1-lambda)*forced, exact endpoint copies", "pairs": "positive/negative time pairs per nonself changing edge",
                         "credit": "1 for >, 0 for <, half for exact ties", "partition": "candidate labels equal vs unequal; each component divided by that edge's ALL positive-negative pair count, then macro-averaged",
                         "identity": "within_delta+across_delta=residual_state_AUROC-hard_state_AUROC, floating gaps retained",
                         "near_zero": "1e-8 is fixed finite-precision/tie-breaking diagnostic, not a fitted model choice"},
            "statistics": {"draws": 10000, "seed": 20261025, "scope": "95% pointwise paired-repeat percentile intervals; no tests/families or multiplicity claim",
                           "reuse": "One repeat index matrix per cell/finite-mask, reused across metrics/methods/q/absolute and paired estimates via equivalent count weights",
                           "streams": "SeedSequence(seed, first4 little-endian u32 of SHA256(JSON(cell_identity, finite_mask)))",
                           "bootstrap_metrics": {"AP": AP_METRICS, "threshold": DETECTION_METRICS, "residual": PAIR_METRICS},
                           "counts": "Retained exactly, not bootstrapped; failures/undefined pairs remain explicit missing denominators"},
            "new_fits": 0, "new_simulations": 0}


def freeze(root: Path = ROOT, output: Path = OUT, log: Path = LOG):
    root, output, log = Path(root).resolve(), Path(output).resolve(), Path(log).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Freeze requires a new empty sensitivity output directory")
    started = datetime.now(timezone.utc).isoformat()
    snapshot = input_snapshot(root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "input_manifest.json").write_bytes(loc.json_bytes(snapshot))
    p = protocol(loc.sha(output / "input_manifest.json"), root)
    (output / "protocol.json").write_bytes(loc.json_bytes(p))
    log.mkdir(parents=True, exist_ok=True)
    (log / "freeze.json").write_bytes(loc.json_bytes({"before_freeze_utc": started, "after_freeze_before_computation_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": loc.sha(output / "protocol.json"), "input_manifest_sha256": loc.sha(output / "input_manifest.json"),
        "source_sha256": p["source_sha256"], "new_fits": 0, "new_simulations": 0}))
    return p


def validate_frozen(root: Path, output: Path):
    before = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    if before != input_snapshot(root):
        raise ValueError("Sensitivity original inputs, sources or preceding analyses changed")
    p = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if p != protocol(loc.sha(output / "input_manifest.json"), root):
        raise ValueError("Sensitivity source/protocol changed")
    return p


def _write(root: Path, output: Path):
    p = validate_frozen(root, output)
    records = output / "records"; records.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        frames, report = compute(root, write_records=records)
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False, lineterminator="\n")
    validate_frozen(root, output)
    (output / "diagnostics.json").write_bytes(loc.json_bytes(report))
    seal = {"protocol_sha256": loc.sha(output / "protocol.json"), "input_manifest_sha256": loc.sha(output / "input_manifest.json"),
            "source_sha256": p["source_sha256"], "sha256": {k: v for k, v in directory_hashes(output).items() if k != "manifest.json"},
            "new_fits": 0, "new_simulations": 0}
    (output / "manifest.json").write_bytes(loc.json_bytes(seal))
    return report


def run(root: Path = ROOT, output: Path = OUT, log: Path = LOG):
    root, output, log = Path(root).resolve(), Path(output).resolve(), Path(log).resolve()
    if set(directory_hashes(output)) != {"input_manifest.json", "protocol.json"}:
        raise ValueError("Run refuses to overwrite existing sensitivity results")
    started = datetime.now(timezone.utc).isoformat()
    report = _write(root, output)
    log.mkdir(parents=True, exist_ok=True)
    (log / "run.json").write_bytes(loc.json_bytes({"started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": loc.sha(output / "manifest.json"), **report}))
    return report


def verify(root: Path = ROOT, output: Path = OUT):
    root, output = Path(root).resolve(), Path(output).resolve()
    p = validate_frozen(root, output)
    before = directory_hashes(output)
    seal = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if seal["source_sha256"] != p["source_sha256"] or seal["protocol_sha256"] != before["protocol.json"] or seal["input_manifest_sha256"] != before["input_manifest.json"]:
        raise ValueError("Sensitivity source/protocol/input seal drift")
    if set(seal["sha256"]) != set(before) - {"manifest.json"} or any(before[name] != digest for name, digest in seal["sha256"].items()):
        raise ValueError("Sensitivity output inventory/hash drift")
    with threadpool_limits(limits=1):
        frames, report = compute(root, verify_records=output / "records")
    for name, expected in frames.items():
        # Retain the same CSV float/missing-value convention in independent replay.
        import io
        expected = pd.read_csv(io.StringIO(expected.to_csv(index=False)))
        pd.testing.assert_frame_equal(pd.read_csv(output / name), expected, check_dtype=False, check_exact=False, atol=2e-14, rtol=1e-12)
    if json.loads((output / "diagnostics.json").read_text(encoding="utf-8")) != loc.clean(report):
        raise ValueError("Sensitivity diagnostic counts changed")
    validate_frozen(root, output)
    if before != directory_hashes(output):
        raise AssertionError("Read-only sensitivity verification wrote outputs")
    return {"status": "passed", "read_only": True, "files_unchanged": len(before), **report}


def rebuild(root: Path = ROOT, output: Path = OUT):
    root, output = Path(root).resolve(), Path(output).resolve()
    verify(root, output)
    return _write(root, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ["freeze", "run", "verify", "rebuild"]:
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--log-dir", type=Path, default=LOG)
    args = parser.parse_args()
    if args.freeze:
        p = freeze(args.root, args.output_dir, args.log_dir)
        result = {"status": "frozen", "methods": len(p["methods"]), "quantiles": p["quantiles"], "lambdas": p["residual_lambdas"]}
    elif args.run:
        result = run(args.root, args.output_dir, args.log_dir)
    elif args.rebuild:
        result = rebuild(args.root, args.output_dir)
    else:
        result = verify(args.root, args.output_dir)
    print(json.dumps(loc.clean(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
