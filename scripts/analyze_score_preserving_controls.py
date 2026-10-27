"""Post hoc score-preserving controls using retained predictions, never fits.

The public array API is also usable by separately defined studies. CLI --freeze
records this analysis before --run; --verify only recomputes retained controls.
No result is added to the original A/B inferential families.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/information_sciences/score_preserving_controls"
LAMBDAS = (.25, .5, .75, 1.)
STAT_SEED = 20261025
BOOTSTRAP_DRAWS = 10000
METRICS = ("temporal_delta_auroc", "pooled_temporal_auroc")
GROUP = ["study", "scorer", "duration", "schedule", "cell"]
NEW_SOURCES = ("scripts/analyze_score_preserving_controls.py", "tests/test_score_preserving_controls.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def nonself(shape: tuple) -> np.ndarray:
    if len(shape) != 3 or shape[-1] != shape[-2]:
        raise ValueError("Expected lag/source/target edge dimensions")
    return np.broadcast_to(~np.eye(shape[-1], dtype=bool), shape)


def method_spec(lambdas=LAMBDAS) -> list[dict]:
    rows = [{"method": "raw_window", "family": "reference", "lambda": 1., "duplicate_of": ""},
            {"method": "forced_hard", "family": "reference", "lambda": 0., "duplicate_of": ""},
            {"method": "native_hard", "family": "reference", "lambda": np.nan, "duplicate_of": ""},
            {"method": "raw_fallback", "family": "raw_fallback", "lambda": np.nan, "duplicate_of": ""}]
    for family in ("shrink", "residual_hard"):
        for lam in ((0., *lambdas) if family == "residual_hard" else lambdas):
            rows.append({"method": f"{family}_l{round(lam * 100):03d}", "family": family, "lambda": lam,
                         "duplicate_of": "raw_window" if lam == 1 else ("forced_hard" if lam == 0 else "")})
    return rows


def build_control_emissions(raw: np.ndarray, forced: np.ndarray, native: np.ndarray,
                            gate_accepted: bool, lambdas=LAMBDAS) -> dict[str, np.ndarray]:
    """Actual float64 arithmetic; no rank substitution or correction of new ties.

    Endpoint copies explicitly preserve raw/forced arrays. Other lambdas use the
    stated convex combination, including its finite-precision limitations.
    """
    raw, forced, native = (np.asarray(x, dtype=np.float64) for x in (raw, forced, native))
    if raw.ndim != 4 or forced.shape != raw.shape or native.shape != raw.shape or not len(raw):
        raise ValueError("raw/forced/native must have identical nonempty [window,lag,source,target] shape")
    nonself(raw.shape[1:])
    if not isinstance(gate_accepted, (bool, np.bool_)):
        raise ValueError("gate_accepted must be boolean")
    if tuple(lambdas) != LAMBDAS:
        raise ValueError("The frozen lambda set is .25,.5,.75,1; do not select lambdas")
    result = {"raw_window": raw.copy(), "forced_hard": forced.copy(), "native_hard": native.copy(),
              "raw_fallback": (forced if gate_accepted else raw).copy()}
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        mean = raw.mean(axis=0, keepdims=True)
        for lam in lambdas:
            result[f"shrink_l{round(lam * 100):03d}"] = raw.copy() if lam == 1 else lam * raw + (1 - lam) * mean
        result["residual_hard_l000"] = forced.copy()
        for lam in lambdas:
            result[f"residual_hard_l{round(lam * 100):03d}"] = raw.copy() if lam == 1 else lam * raw + (1 - lam) * forced
    return result


def edge_aurocs(active: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Every column is one edge. Undefined class targets remain NaN."""
    active = np.asarray(active, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if active.shape != scores.shape or active.ndim != 2:
        raise ValueError("Expected equally sized [window,edge] truth/scores")
    positives = active.sum(axis=0)
    valid = (positives > 0) & (positives < len(active)) & np.isfinite(scores).all(axis=0)
    result = np.full(active.shape[1], np.nan)
    if valid.any():
        y, s = active[:, valid], scores[:, valid]
        pairs = y[:, None, :] & ~y[None, :, :]
        # Comparisons avoid overflow from subtracting extreme finite scores.
        credit = (s[:, None, :] > s[None, :, :]) + .5 * (s[:, None, :] == s[None, :, :])
        result[valid] = (credit * pairs).sum(axis=(0, 1)) / pairs.sum(axis=(0, 1))
    return result


def score_metrics(predictions: np.ndarray, active_truth: np.ndarray, changing_mask: np.ndarray,
                  *, mask=None) -> dict:
    predictions, truth = np.asarray(predictions), np.asarray(active_truth)
    changed = np.asarray(changing_mask, dtype=bool)
    if predictions.shape != truth.shape or predictions.ndim != 4 or changed.shape != predictions.shape[1:]:
        raise ValueError("Prediction, active_truth and changing_mask shape mismatch")
    if not np.isin(truth, (0, 1)).all():
        raise ValueError("active_truth must be binary")
    mask = nonself(predictions.shape[1:]) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != changed.shape or not mask.any():
        raise ValueError("Invalid edge mask")
    y, s, changed = truth[:, mask].astype(bool), predictions[:, mask], changed[mask]
    usable = changed & (y.sum(axis=0) > 0) & (y.sum(axis=0) < len(y))
    result = {"temporal_delta_edges": int(usable.sum()), "temporal_delta_edges_total": int(changed.sum()),
              "scored_edges": int(mask.sum()), "temporal_delta_auroc": np.nan, "pooled_temporal_auroc": np.nan}
    if not np.isfinite(s).all():
        return {**result, "status": "failed", "undefined_reason": "nonfinite_scored_predictions"}
    if usable.any(): result["temporal_delta_auroc"] = float(edge_aurocs(y[:, usable], s[:, usable]).mean())
    if np.unique(y).size == 2: result["pooled_temporal_auroc"] = float(roc_auc_score(y.ravel(), s.ravel()))
    reasons = []
    if not usable.any(): reasons.append("no_support_change" if not changed.any() else "no_observed_flip")
    if not np.isfinite(result["pooled_temporal_auroc"]): reasons.append("pooled_single_class")
    return {**result, "status": "undefined" if reasons else "ok", "undefined_reason": ";".join(reasons)}


def affine_checks(raw: np.ndarray, shrunk: np.ndarray, active_truth: np.ndarray,
                  changing_mask: np.ndarray, lam: float, *, mask=None) -> list[dict]:
    """Check every scored edge and unordered time pair, retaining failures/ties."""
    mask = nonself(raw.shape[1:]) if mask is None else np.asarray(mask, dtype=bool)
    indices = np.argwhere(mask)
    r, s, y = raw[:, mask], shrunk[:, mask], np.asarray(active_truth[:, mask], dtype=bool)
    pos = y.sum(axis=0)
    evaluable = changing_mask[mask] & (pos > 0) & (pos < len(y))
    ar, ass = edge_aurocs(y, r), edge_aurocs(y, s)
    i, j = np.triu_indices(len(r), k=1)
    rr = (r[i] > r[j]).astype(np.int8) - (r[i] < r[j]).astype(np.int8)
    ss = (s[i] > s[j]).astype(np.int8) - (s[i] < s[j]).astype(np.int8)
    records = []
    for k, (lag, source, target) in enumerate(indices):
        finite = bool(np.isfinite(r[:, k]).all() and np.isfinite(s[:, k]).all())
        records.append({"lambda": lam, "lag": int(lag + 1), "source": int(source), "target": int(target),
            "time_pairs": len(i), "finite": finite, "changing_edge": bool(changing_mask[mask][k]),
            "temporal_evaluable": bool(evaluable[k] and finite),
            "raw_tied_pairs": int((rr[:, k] == 0).sum()) if finite else -1,
            "shrink_tied_pairs": int((ss[:, k] == 0).sum()) if finite else -1,
            "new_tied_pairs": int(((rr[:, k] != 0) & (ss[:, k] == 0)).sum()) if finite else -1,
            "lost_tied_pairs": int(((rr[:, k] == 0) & (ss[:, k] != 0)).sum()) if finite else -1,
            "reversed_pairs": int((rr[:, k] * ss[:, k] == -1).sum()) if finite else -1,
            "raw_edge_auroc": float(ar[k]), "shrink_edge_auroc": float(ass[k]),
            "edge_auroc_difference": float(ass[k] - ar[k]),
            "undefined_reason": "" if evaluable[k] and finite else (
                "nonfinite_scores" if not finite else ("stable_edge_excluded_from_temporal_target" if not changing_mask[mask][k] else "no_observed_flip"))})
    return records


def evaluate_controls(raw, forced, native, gate_accepted, active_truth, changing_mask,
                      *, mask=None) -> tuple[list[dict], dict[str, np.ndarray], list[dict]]:
    """Shared pure API: method metrics, explicit emissions, per-edge affine QA."""
    emitted = build_control_emissions(raw, forced, native, gate_accepted)
    rows = [{**spec, **score_metrics(emitted[spec["method"]], active_truth, changing_mask, mask=mask),
             "gate_accepted": bool(gate_accepted),
             "equals_raw": bool(np.array_equal(emitted[spec["method"]], raw)),
             "equals_forced": bool(np.array_equal(emitted[spec["method"]], forced))}
            for spec in method_spec()]
    checks = [row for lam in LAMBDAS for row in affine_checks(np.asarray(raw), emitted[f"shrink_l{round(lam * 100):03d}"],
              np.asarray(active_truth), np.asarray(changing_mask), lam, mask=mask)]
    return rows, emitted, checks


def _npz(path: Path, keys) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in keys}


def iter_retained_cases(root: Path = ROOT) -> Iterator[dict]:
    """Yield one A/B case at a time; never load series/innovations unnecessarily.

    Arrays are top-level keys raw/forced/native/oracle/bounds/state_support/
    window_truth/active_truth/changing_mask/regimes. Failed planned cases are
    yielded with status/error rather than omitted. No scoring or fit is run.
    """
    root = Path(root)
    base = root / "results/information_sciences"
    a, b = base / "duration_confirmation", base / "schedule_robustness"
    ap = json.loads((a / "protocol.json").read_text(encoding="utf-8"))
    bp = json.loads((b / "schedule_robustness_protocol.json").read_text(encoding="utf-8"))
    for study, protocol in (("A", ap), ("B", bp)):
        seeds = protocol["repeat_seeds"] if study == "A" else protocol["repeats"]
        durations = protocol["durations"] if study == "A" else protocol["configuration"]["durations"]
        scorers = protocol["scorers"] if study == "A" else ["lasso"]
        schedules = ["fixed_phase"] if study == "A" else protocol["configuration"]["schedules"]
        for seed in seeds:
            for duration in durations:
                for schedule in schedules:
                    for scorer in scorers:
                        case = {"study": study, "scorer": scorer, "seed": seed, "repeat_seed": seed,
                                "duration": duration, "schedule": schedule, "cell": f"D{duration}_{schedule}"}
                        try:
                            if study == "A":
                                path = a / "records" / f"repeat_{seed}_D{duration}_{scorer}.npz"
                                meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                                if meta["status"] != "success": raise ValueError("retained_fit_failure:" + str(meta.get("error", "")))
                                support = _npz(a / "records" / f"repeat_{seed}_inputs.npz", ["support"])["support"]
                                regimes = _npz(a / "records" / f"repeat_{seed}_D{duration}_data.npz", ["regimes"])["regimes"]
                            else:
                                path = b / "parts" / f"repeat_{seed}_D{duration}_{schedule}.npz"
                                meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                                if meta["diagnostic"]["status"] != "ok": raise ValueError("retained_fit_failure:" + str(meta["diagnostic"].get("error", "")))
                                other = _npz(path, ["state_support", "regimes"])
                                support, regimes = other["state_support"], other["regimes"]
                            keys = ["window_scores", "bounds", "true_window_labels", "emission_forced_hard", "emission_native_hard", "emission_oracle_label"]
                            arr = _npz(path, keys)
                            bounds = arr["bounds"]
                            truth = regimes[(bounds[:, 0] + bounds[:, 1] - 1) // 2]
                            np.testing.assert_array_equal(arr["true_window_labels"], truth)
                            yield {**case, "status": "ok", "error": "", "raw": arr["window_scores"],
                                   "forced": arr["emission_forced_hard"], "native": arr["emission_native_hard"],
                                   "oracle": arr["emission_oracle_label"], "bounds": bounds, "window_truth": truth,
                                   "state_support": support, "active_truth": support[truth].astype(bool),
                                   "changing_mask": np.any(support != support[0], axis=0), "regimes": regimes,
                                   "gate_accepted": bool(meta["diagnostic"]["hard_gate_accepted"])}
                        except (ValueError, KeyError, OSError, AssertionError) as error:
                            yield {**case, "status": "failed", "error": str(error)}


def paired_bootstrap(values, *, seed=STAT_SEED, draws=BOOTSTRAP_DRAWS) -> tuple:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values): return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if len(values) < 5: return mean, np.nan, np.nan
    # Resetting the fixed stream preserves paired resample indices for groups
    # with the same sorted repeat ids and finite target set.
    indices = np.random.default_rng(seed).integers(0, len(values), (draws, len(values)))
    low, high = np.quantile(values[indices].mean(axis=1), [.025, .975])
    return mean, float(low), float(high)


def summarise(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, contrasts = [], []
    for keys, group in detail.groupby(GROUP, sort=True):
        identity = dict(zip(GROUP, keys))
        for method, rows in group.groupby("method", sort=True):
            rows = rows.sort_values("repeat_seed")
            for metric in METRICS:
                values = rows[metric].to_numpy(float)
                mean, low, high = paired_bootstrap(values)
                summaries.append({**identity, "method": method, "metric": metric, "n_planned": len(rows),
                    "n_valid": int(np.isfinite(values).sum()), "n_failed": int((rows.status == "failed").sum()),
                    "n_missing": int((~np.isfinite(values)).sum()),
                    "mean": mean, "ci95_low": low, "ci95_high": high, "interval_scope": "pointwise descriptive; not simultaneous"})
                for reference in ("raw_window", "forced_hard"):
                    ref = group[group.method == reference].set_index("repeat_seed")[metric]
                    difference = values - ref.reindex(rows.repeat_seed).to_numpy(float)
                    mean, low, high = paired_bootstrap(difference)
                    contrasts.append({**identity, "method": method, "reference": reference, "metric": metric,
                        "n_planned": len(rows), "n_paired": int(np.isfinite(difference).sum()),
                        "n_missing_pairs": int((~np.isfinite(difference)).sum()),
                        "mean": mean, "ci95_low": low, "ci95_high": high,
                        "role": "post hoc descriptive; no p value or original-family assignment"})
    return pd.DataFrame(summaries), pd.DataFrame(contrasts)


def directory_hashes(path: Path) -> dict:
    return {p.relative_to(path).as_posix(): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def input_snapshot(root: Path) -> dict:
    base = root / "results/information_sciences"
    frozen = {p.relative_to(root).as_posix(): sha(p) for name in ("duration_confirmation", "schedule_robustness")
              for p in sorted((base / name).rglob("*")) if p.is_file()}
    sources = {"configs/schedule_robustness.yaml"}
    for name in ("duration_confirmation/protocol.json", "schedule_robustness/schedule_robustness_protocol.json"):
        path = base / name
        proto = json.loads(path.read_text(encoding="utf-8"))
        for source, expected in proto["source_sha256"].items():
            if sha(root / source) != expected: raise ValueError("Frozen source changed: " + source)
        sources.update(proto["source_sha256"])
        seal_name = "COMPLETE.json" if "duration_confirmation" in name else "schedule_robustness_outputs.json"
        seal = json.loads((path.parent / seal_name).read_text(encoding="utf-8"))
        for relative, expected in seal["sha256"].items():
            target = (path.parent / relative).resolve()
            if not target.is_relative_to(path.parent.resolve()) or sha(target) != expected:
                raise ValueError("Frozen output checksum mismatch: " + relative)
    return {"frozen_A_B_sha256": frozen, "frozen_source_sha256": {p: sha(root / p) for p in sorted(sources)}}


def protocol(input_sha: str, root: Path = ROOT) -> dict:
    return {"analysis": "score_preserving_controls_v1", "role": "post hoc descriptive retained-output analysis; not new confirmation or public preregistration",
        "lambda_positive": list(LAMBDAS), "residual_endpoints": [0., 1.], "no_lambda_selection": True,
        "formulas": {"shrink": "lambda * raw + (1-lambda) * raw.mean(axis=0)",
                     "residual_hard": "lambda * raw + (1-lambda) * forced_candidate",
                     "raw_fallback": "forced candidate if gate accepts, raw otherwise"},
        "arithmetic": "float64; retain endpoints as byte-exact copies; record new/lost ties, inversions and per-edge AUROC changes without repair",
        "targets": list(METRICS), "excluded": "self edges excluded from both targets; stable edges excluded only from within-changing-edge target",
        "studies": {"A": {"seeds": [50000, 50099], "n": 100, "scorers": ["lasso", "ols"], "durations": [300, 600]},
                    "B": {"seeds": [60000, 60049], "n": 50, "scorers": ["lasso"], "durations": [300, 600], "schedules": ["fixed_phase", "random_phase", "irregular"]}},
        "methods": [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()} for row in method_spec()],
        "statistics": {"unit": "independent repeat; pair methods within the same repeat", "draws": BOOTSTRAP_DRAWS,
                       "seed": STAT_SEED, "interval": "95% pointwise paired-repeat percentile bootstrap, minimum n5, finite complete pairs",
                       "references": ["raw_window", "forced_hard"], "hypothesis_tests": "none", "original_A_B_families": "unchanged"},
        "failure_policy": "retain all planned case/method rows; no replacement, no 0.5 imputation; counts/undefined reasons reported",
        "threads": 1, "input_manifest_sha256": input_sha, "source_sha256": {p: sha(root / p) for p in NEW_SOURCES},
        "qa_amendment": {"previous_protocol_sha256": "9a7acfd19faa96ac80d2c415de15391e8cd75320a72f504f71c1c0126c67415f",
                         "previous_script_sha256": "6c4ff19b9e7148cac979d868060113066d018dd6438334bab6c8d57e8acde5a0",
                         "v1_complete_results_observed": True,
                         "reason": "QA-only event filter correction: undefined NaN AUROC differences are not nonzero events; new/lost ties and reversals still count independently",
                         "unchanged": "lambdas, metrics, samples, comparisons, pointwise bootstrap, detail/summary/contrasts/affine_checks CSVs; only affine_events and its diagnostic count change; no result selection",
                         "timing": "Internal amendment/freeze clocks and preserved v1 evidence are retained in the execution stage; no public preregistration claimed"},
        "new_fits": 0, "new_simulations": 0}


def freeze(output: Path = OUT, root: Path = ROOT) -> dict:
    output, root = Path(output), Path(root)
    if output.exists() and any(output.iterdir()): raise ValueError("Freeze requires a new empty output directory")
    snapshot = input_snapshot(root)
    assert len(snapshot["frozen_A_B_sha256"]) == 1915 and len(snapshot["frozen_source_sha256"]) == 12
    output.mkdir(parents=True, exist_ok=True)
    (output / "input_manifest.json").write_bytes(json_bytes(snapshot))
    p = protocol(sha(output / "input_manifest.json"), root)
    (output / "protocol.json").write_bytes(json_bytes(p))
    return p


def compute(root: Path = ROOT) -> tuple[dict[str, pd.DataFrame], dict]:
    details, checks = [], []
    n_cases = failed = 0
    for case in iter_retained_cases(root):
        n_cases += 1
        identity = {k: case[k] for k in (*GROUP, "repeat_seed")}
        if case["status"] == "failed":
            failed += 1
            details.extend({**identity, **spec, **{metric: np.nan for metric in METRICS},
                            "status": "failed", "undefined_reason": case["error"], "gate_accepted": None,
                            "temporal_delta_edges": 0, "temporal_delta_edges_total": 0, "scored_edges": 0,
                            "equals_raw": False, "equals_forced": False} for spec in method_spec())
            continue
        rows, _, edge_checks = evaluate_controls(case["raw"], case["forced"], case["native"], case["gate_accepted"],
                                                case["active_truth"], case["changing_mask"])
        details.extend({**identity, **row} for row in rows)
        checks.extend({**identity, **row} for row in edge_checks)
    detail = pd.DataFrame(details)
    affine_columns = [*GROUP, "repeat_seed", "lambda", "lag", "source", "target", "time_pairs", "finite",
                      "changing_edge", "temporal_evaluable", "raw_tied_pairs", "shrink_tied_pairs", "new_tied_pairs",
                      "lost_tied_pairs", "reversed_pairs", "raw_edge_auroc", "shrink_edge_auroc", "edge_auroc_difference", "undefined_reason"]
    affine = pd.DataFrame(checks, columns=affine_columns)
    for name in ("finite", "temporal_evaluable"):
        affine[name] = affine[name].astype(bool)
    detail = detail.sort_values(GROUP + ["repeat_seed", "method"]).reset_index(drop=True)
    affine = affine.sort_values(GROUP + ["repeat_seed", "lambda", "lag", "source", "target"]).reset_index(drop=True)
    summary, contrasts = summarise(detail)
    finite = affine[affine.finite]
    auc_valid = finite[finite.temporal_evaluable]
    auc_difference = finite.edge_auroc_difference.astype(float)
    changed_auc = np.isfinite(auc_difference) & (auc_difference != 0)
    events = finite[(finite.new_tied_pairs > 0) | (finite.lost_tied_pairs > 0) | (finite.reversed_pairs > 0) | changed_auc]
    report = {"role": "post hoc descriptive; no confirmatory-family assignments", "cases": n_cases, "failed_cases": failed,
              "method_rows": len(detail), "method_failed_rows": int((detail.status == "failed").sum()),
              "method_undefined_rows": int((detail.status == "undefined").sum()), "affine_edge_checks": len(affine),
              "affine_nonfinite_checks": int((~affine.finite).sum()), "affine_event_rows": len(events),
              "new_tied_pairs": int(finite.new_tied_pairs.sum()), "lost_tied_pairs": int(finite.lost_tied_pairs.sum()),
              "reversed_pairs": int(finite.reversed_pairs.sum()),
              "temporal_evaluable_edge_checks": len(auc_valid),
              "changed_edge_auc_count": int((auc_valid.edge_auroc_difference != 0).sum()),
              "max_abs_edge_auc_difference": float(auc_valid.edge_auroc_difference.abs().max()) if len(auc_valid) else None,
              "new_fits": 0, "new_simulations": 0}
    return {"detail.csv": detail, "summary.csv": summary, "contrasts.csv": contrasts,
            "affine_checks.csv": affine, "affine_events.csv": events.reset_index(drop=True)}, report


def check_reference_metrics(detail: pd.DataFrame, root: Path) -> int:
    """Check the recomputed raw/native/forced baselines against frozen row CSVs."""
    base = root / "results/information_sciences"
    a = pd.read_csv(base / "duration_confirmation/detail.csv").rename(columns={"D": "duration"})
    a["study"], a["schedule"] = "A", "fixed_phase"
    b = pd.read_csv(base / "schedule_robustness/schedule_robustness_detail.csv").rename(
        columns={"seed": "repeat_seed", "duration_mean": "duration"})
    b["study"] = "B"
    originals = pd.concat([a, b], ignore_index=True)
    keys = [*GROUP, "repeat_seed", "method"]
    a["cell"] = a.duration.map(lambda d: f"D{d}_fixed_phase")
    originals.loc[originals.study == "A", "cell"] = originals.loc[originals.study == "A", "duration"].map(lambda d: f"D{d}_fixed_phase")
    selected = detail[detail.method.isin(["raw_window", "forced_hard", "native_hard"])]
    merged = selected.merge(originals, on=keys, how="left", suffixes=("_new", "_original"), validate="one_to_one", indicator=True)
    if not (merged._merge == "both").all(): raise AssertionError("Original reference rows missing")
    for metric in METRICS:
        np.testing.assert_allclose(merged[metric + "_new"], merged[metric + "_original"], rtol=0, atol=2e-12, equal_nan=True)
    return len(merged)


def validate_frozen(output: Path, root: Path) -> dict:
    p = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    original = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    if input_snapshot(root) != original: raise ValueError("Frozen scientific inputs changed")
    if p != protocol(sha(output / "input_manifest.json"), root): raise ValueError("Analysis source/protocol changed since freeze")
    return p


def _write_derived(output: Path, root: Path, p: dict) -> dict:
    with threadpool_limits(limits=1): tables, report = compute(root)
    report["reference_method_rows_checked"] = check_reference_metrics(tables["detail.csv"], root)
    for name, frame in tables.items(): frame.to_csv(output / name, index=False, lineterminator="\n")
    (output / "diagnostics.json").write_bytes(json_bytes(report))
    validate_frozen(output, root)
    manifest = {"protocol_sha256": sha(output / "protocol.json"), "source_sha256": p["source_sha256"],
                "input_manifest_sha256": sha(output / "input_manifest.json"),
                "sha256": {k: v for k, v in directory_hashes(output).items() if k != "manifest.json"},
                "new_fits": 0, "new_simulations": 0}
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    return report


def run(output: Path = OUT, root: Path = ROOT) -> dict:
    output, root = Path(output), Path(root)
    p = validate_frozen(output, root)
    if set(directory_hashes(output)) != {"protocol.json", "input_manifest.json"}:
        raise ValueError("Run requires the frozen-only directory; refuse to overwrite results")
    return _write_derived(output, root, p)


def rebuild(output: Path = OUT, root: Path = ROOT) -> dict:
    """Regenerate derived controls only, preserving protocol and input manifest.

    Neither these deterministic outputs nor their seal contain clock fields;
    any external historical pre-run stage record remains outside this command.
    """
    output, root = Path(output), Path(root)
    p = validate_frozen(output, root)
    seal = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    before = directory_hashes(output)
    if seal["sha256"] != {k: v for k, v in before.items() if k != "manifest.json"}:
        raise ValueError("Existing control output checksums fail; refuse to rebuild over altered evidence")
    result = _write_derived(output, root, p)
    assert before["protocol.json"] == sha(output / "protocol.json")
    assert before["input_manifest.json"] == sha(output / "input_manifest.json")
    return result


def verify(output: Path = OUT, root: Path = ROOT) -> dict:
    """Pure read-only recomputation, including bootstrap tables and float ties."""
    output, root = Path(output), Path(root)
    before = directory_hashes(output)
    p = validate_frozen(output, root)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["sha256"] != {k: v for k, v in before.items() if k != "manifest.json"}:
        raise ValueError("Control output checksum mismatch")
    assert manifest["protocol_sha256"] == sha(output / "protocol.json")
    assert manifest["source_sha256"] == p["source_sha256"]
    with threadpool_limits(limits=1): tables, report = compute(root)
    report["reference_method_rows_checked"] = check_reference_metrics(tables["detail.csv"], root)
    for name, frame in tables.items():
        # Match CSV's blank fields/dtype inference while preserving file bytes.
        expected = pd.read_csv(io.StringIO(frame.to_csv(index=False, lineterminator="\n")))
        actual = pd.read_csv(output / name)
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=0, atol=2e-12)
    assert json.loads((output / "diagnostics.json").read_text(encoding="utf-8")) == report
    validate_frozen(output, root)
    assert before == directory_hashes(output), "Read-only verifier modified control files"
    return {"status": "passed", "read_only": True, "files_unchanged": len(before), **report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--verify", action="store_true")
    action.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    report = freeze(args.output_dir, args.root) if args.freeze else (
        run(args.output_dir, args.root) if args.run else (
            rebuild(args.output_dir, args.root) if args.rebuild else verify(args.output_dir, args.root)))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
