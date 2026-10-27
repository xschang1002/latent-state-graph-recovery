"""Post-review, retained-output boundary-slot localisation and stable-edge changes.

This module neither generates data nor fits a model. Positive-change AP excludes
zero-strength predictions, and fixed-amplitude detections share an unlabelled
raw-score reference within each case. The time resolution is the centre-to-centre
slot, rather than a claim about sample-level boundary error.
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/information_sciences/temporal_localisation"
STAGE = ROOT / "experiments/logs/reviewer_controls_2026-10-25"
sys.path.insert(0, str(ROOT / "scripts"))
METHODS = ("raw_window", "forced_hard", "native_hard", "raw_fallback",
           "shrink_l025", "shrink_l050", "shrink_l075", "shrink_l100",
           "residual_hard_l000", "residual_hard_l025", "residual_hard_l050",
           "residual_hard_l075", "residual_hard_l100", "oracle_label", "static_mean")
METRICS = ("boundary_positive_change_ap", "boundary_slot_f1", "stable_false_change_rate",
           "stable_any_change_edge_fraction", "persistent_false_change_rate", "absent_false_change_rate")
GROUP = ("study", "scorer", "duration", "schedule", "cell")
BOOTSTRAP_SEED = 20261025


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def json_bytes(value) -> bytes:
    return (json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def positive_change_ap(truth: np.ndarray, strength: np.ndarray) -> float:
    """Tied-level AP; zero strength abstains; no positive/negative target is missing."""
    truth, strength = np.asarray(truth, bool), np.asarray(strength, float)
    if truth.ndim != 1 or strength.shape != truth.shape:
        raise ValueError("AP requires aligned one-dimensional slots")
    if not np.isfinite(strength).all() or np.any(strength < 0):
        return np.nan
    positives = int(truth.sum())
    if positives == 0 or positives == len(truth):
        return np.nan
    selected = strength > 0
    order = np.argsort(-strength[selected], kind="stable")
    scores, labels = strength[selected][order], truth[selected][order]
    if not len(scores):
        return 0.0
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]) + 1, len(scores)]
    hits, previous_hits, ap = 0, 0, 0.0
    for stop in ends:
        hits = int(labels[:stop].sum())
        ap += ((hits - previous_hits) / positives) * (hits / int(stop))
        previous_hits = hits
    return float(ap)


def boundary_grid(bounds, regimes, state_support) -> dict:
    bounds = np.asarray(bounds, dtype=int)
    regimes, support = np.asarray(regimes, dtype=int), np.asarray(state_support)
    if (bounds.ndim != 2 or bounds.shape[1] != 2 or len(bounds) < 2 or regimes.ndim != 1
            or support.ndim != 4 or support.shape[-1] != support.shape[-2]):
        raise ValueError("Need two or more windows, full observed regimes and [K,L,N,N] support")
    if (np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] > len(regimes))
            or np.any(bounds[:, 1] <= bounds[:, 0]) or not np.isin(support, [0, 1]).all()
            or np.any(regimes < 0) or np.any(regimes >= len(support))):
        raise ValueError("Bounds, regime labels or binary support are invalid")
    centres = (bounds[:, 0] + bounds[:, 1] - 1) // 2
    if np.any(np.diff(centres) <= 0):
        raise ValueError("Window centres must increase strictly")
    mask = np.ones(support.shape[1:], dtype=bool)
    for node in range(mask.shape[-1]):
        mask[:, node, node] = False
    edge_support = support[:, mask].astype(bool)
    changes = np.flatnonzero(regimes[1:] != regimes[:-1]) + 1
    covered = changes[(changes > centres[0]) & (changes <= centres[-1])]
    counts = np.zeros((len(centres) - 1, int(mask.sum())), dtype=np.int64)
    for time in covered:
        slot = int(np.searchsorted(centres, time, side="left") - 1)
        counts[slot] += edge_support[regimes[time]] != edge_support[regimes[time - 1]]
    present, absent = edge_support.all(axis=0), ~edge_support.any(axis=0)
    edge_indices = np.array(np.nonzero(mask)).T
    return {"centres": centres, "mask": mask, "truth_event_counts": counts,
            "truth_boundary_slots": counts > 0, "persistent_edges": present, "absent_edges": absent,
            "changing_edges": ~(present | absent), "edge_indices": edge_indices,
            "actual_state_changes_total": len(changes), "actual_state_changes_covered": len(covered),
            "covered_start": int(centres[0]), "covered_stop_inclusive": int(centres[-1]),
            "support_events_collapsed": int((counts - (counts > 0)).sum())}


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


def evaluate_localisation(emissions: dict, raw, bounds, regimes, state_support, *, quantile: float = .95):
    """Return repeat-method rows, small per-edge arrays, and grid coverage metadata.

    No labels enter raw-threshold calibration. AP is over actual change-bearing
    centre-to-centre slots and cannot resolve location within such a slot.
    """
    if not emissions:
        raise ValueError("At least one method emission is required")
    grid = boundary_grid(bounds, regimes, state_support)
    raw = np.asarray(raw, float)
    expected_shape = (len(bounds), *np.asarray(state_support).shape[1:])
    if raw.shape != expected_shape or not 0 < quantile < 1:
        raise ValueError("Raw scores and quantile must match the declared evaluation grid")
    raw_finite = bool(np.isfinite(raw[:, grid["mask"]]).all())
    raw_strength = np.abs(np.diff(raw[:, grid["mask"]], axis=0))
    threshold = float(np.quantile(raw_strength, quantile, method="linear")) if raw_finite else np.nan
    truth = grid["truth_boundary_slots"]
    slots, edges = truth.shape
    true_count = truth.sum(axis=0)
    ap_eligible = grid["changing_edges"] & (true_count > 0) & (true_count < slots)
    f1_eligible = grid["changing_edges"] & (true_count > 0)
    stable = grid["persistent_edges"] | grid["absent_edges"]
    rows, ap_values, tp_values, fp_values, fn_values, event_values = [], [], [], [], [], []
    for method, prediction in emissions.items():
        prediction = np.asarray(prediction, float)
        if prediction.shape != expected_shape:
            raise ValueError(f"Emission shape differs from raw: {method}")
        finite = raw_finite and bool(np.isfinite(prediction[:, grid["mask"]]).all())
        ap = np.full(edges, np.nan); tp = np.full(edges, np.nan); fp = tp.copy(); fn = tp.copy(); events = tp.copy()
        row = {"method": method, "status": "ok" if finite else "nonfinite_scores", "raw_change_threshold": threshold,
               "threshold_zero": bool(threshold == 0) if raw_finite else False, "n_windows": slots + 1,
               "n_slots": slots, "n_nonself_edges": edges, "changing_edges_total": int(grid["changing_edges"].sum()),
               "boundary_ap_edges_eligible": int(ap_eligible.sum()), "boundary_f1_edges_eligible": int(f1_eligible.sum()),
               "boundary_ap_edges_used": int(ap_eligible.sum()) if finite else 0,
               "boundary_f1_edges_used": int(f1_eligible.sum()) if finite else 0,
               "changing_edges_no_covered_boundary": int((grid["changing_edges"] & (true_count == 0)).sum()),
               "changing_edges_all_slots_boundary": int((grid["changing_edges"] & (true_count == slots)).sum()),
               "stable_edges": int(stable.sum()), "persistent_edges": int(grid["persistent_edges"].sum()),
               "absent_edges": int(grid["absent_edges"].sum()), "truth_boundary_slots": int(truth.sum()),
               "actual_support_events_covered": int(grid["truth_event_counts"].sum()),
               "actual_state_changes_total": grid["actual_state_changes_total"],
               "actual_state_changes_covered": grid["actual_state_changes_covered"],
               "support_events_collapsed": grid["support_events_collapsed"],
               "covered_start": grid["covered_start"], "covered_stop_inclusive": grid["covered_stop_inclusive"]}
        row.update({metric: np.nan for metric in METRICS})
        row.update({name: np.nan for name in ("boundary_tp", "boundary_fp", "boundary_fn", "stable_false_events",
                                            "stable_event_opportunities", "persistent_false_events", "absent_false_events",
                                            "stable_edges_any_change", "stable_false_changes_per_1000_samples")})
        if finite:
            strength = np.abs(np.diff(prediction[:, grid["mask"]], axis=0))
            detected = strength > threshold
            tp = (detected & truth).sum(axis=0); fp = (detected & ~truth).sum(axis=0)
            fn = (~detected & truth).sum(axis=0); events = detected.sum(axis=0)
            for edge in np.flatnonzero(ap_eligible):
                ap[edge] = positive_change_ap(truth[:, edge], strength[:, edge])
            f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.full(edges, np.nan), where=(2 * tp + fp + fn) > 0)
            row.update({"boundary_positive_change_ap": float(ap[ap_eligible].mean()) if ap_eligible.any() else np.nan,
                        "boundary_slot_f1": float(f1[f1_eligible].mean()) if f1_eligible.any() else np.nan,
                        "boundary_tp": int(tp[f1_eligible].sum()), "boundary_fp": int(fp[f1_eligible].sum()),
                        "boundary_fn": int(fn[f1_eligible].sum()), "stable_false_events": int(events[stable].sum()),
                        "stable_event_opportunities": int(slots * stable.sum()),
                        "persistent_false_events": int(events[grid["persistent_edges"]].sum()),
                        "absent_false_events": int(events[grid["absent_edges"]].sum()),
                        "stable_edges_any_change": int((events[stable] > 0).sum())})
            row["stable_false_change_rate"] = _rate(row["stable_false_events"], row["stable_event_opportunities"])
            row["stable_any_change_edge_fraction"] = _rate(row["stable_edges_any_change"], int(stable.sum()))
            row["persistent_false_change_rate"] = _rate(row["persistent_false_events"], int(slots * grid["persistent_edges"].sum()))
            row["absent_false_change_rate"] = _rate(row["absent_false_events"], int(slots * grid["absent_edges"].sum()))
            row["stable_false_changes_per_1000_samples"] = 1000 * _rate(row["stable_false_events"], int(stable.sum()) * int(grid["centres"][-1] - grid["centres"][0]))
        rows.append(row); ap_values.append(ap); tp_values.append(tp); fp_values.append(fp); fn_values.append(fn); event_values.append(events)
    edge_arrays = {"method_names": np.asarray(list(emissions)), "edge_indices": grid["edge_indices"],
                   "centres": grid["centres"], "truth_event_counts": grid["truth_event_counts"],
                   "persistent_edges": grid["persistent_edges"], "absent_edges": grid["absent_edges"],
                   "method_edge_ap": np.stack(ap_values), "method_edge_tp": np.stack(tp_values),
                   "method_edge_fp": np.stack(fp_values), "method_edge_fn": np.stack(fn_values),
                   "method_edge_predicted_changes": np.stack(event_values), "raw_change_threshold": np.asarray(threshold)}
    metadata = {name: value for name, value in grid.items() if not isinstance(value, np.ndarray)}
    return rows, edge_arrays, metadata


def frozen_hashes(root: Path) -> dict:
    base = root / "results/information_sciences"
    return {p.relative_to(root).as_posix(): sha(p) for name in ("duration_confirmation", "schedule_robustness")
            for p in sorted((base / name).rglob("*")) if p.is_file()}


def frozen_sources(root: Path) -> dict:
    base = root / "results/information_sciences"
    names = {"configs/schedule_robustness.yaml"}
    for name in ("duration_confirmation/protocol.json", "schedule_robustness/schedule_robustness_protocol.json"):
        names.update(json.loads((base / name).read_text(encoding="utf-8"))["source_sha256"])
    return {name: sha(root / name) for name in sorted(names)}


def protocol(root: Path) -> dict:
    return {"study": "post_review_boundary_slot_localisation_v1", "role": "post hoc descriptive retained-output analysis",
            "methods": list(METHODS), "input_studies": ["A", "B"], "new_fits": 0, "new_simulations": 0,
            "boundary_grid": "c=(start+stop-1)//2; slot(c_w,c_{w+1}]; actual edge support events; multiple events collapsed and counted",
            "boundary_AP": "positive-change AP with zero-strength abstention; exact float ties in whole groups; no true boundary or no negative slot -> missing; macro changing edges per repeat",
            "boundary_F1": "strict abs(adjacent emitted score difference)>shared raw q95; per changing edge with >=1 true boundary; constant ->0; macro edges",
            "stable_metric": "strict abs(adjacent difference)>same threshold, separately on persistent/absent supports; rates per edge-slot; not a calibrated FPR or binary-support threshold",
            "threshold": {"quantile": .95, "method": "linear", "pool": "all raw nonself edges and adjacent slots within case/scorer; labels unused", "comparison": ">"},
            "interpretation": "stride-resolution boundary-slot localisation; not sample error; compare each cell relative to raw, not absolute AP across strides; AUROC is temporal state discrimination",
            "statistics": {"bootstrap_draws": 10000, "bootstrap_seed": BOOTSTRAP_SEED, "unit": "independent paired repeat", "interval": "ordinary pointwise95percentile", "tests": "none",
                           "streams": "SeedSequence([seed, first4 little-endian uint32 words of SHA256(JSON(group,method,metric,absolute_or_paired))])"},
            "frozen_input_sha256": frozen_hashes(root), "frozen_source_sha256": frozen_sources(root),
            "analysis_source_sha256": {name: sha(root / name) for name in ("scripts/analyze_temporal_localisation.py", "scripts/analyze_score_preserving_controls.py")}}


def freeze(root: Path = ROOT, output: Path = OUT, stage: Path = STAGE) -> dict:
    root, output, stage = Path(root).resolve(), Path(output).resolve(), Path(stage).resolve()
    value = protocol(root)
    if output.exists() and any(output.iterdir()):
        existing = output / "protocol.json"
        if not existing.is_file() or existing.read_bytes() != json_bytes(value):
            raise ValueError("Refusing to overwrite an existing/different localisation protocol or result")
        return value
    before = datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_bytes(json_bytes(value))
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "temporal_localisation_freeze.json").write_bytes(json_bytes({"before_protocol_write_utc": before,
        "after_protocol_write_before_analysis_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha(output / "protocol.json"),
        "input_files": len(value["frozen_input_sha256"]), "source_files": len(value["frozen_source_sha256"]), "new_fits": 0, "new_simulations": 0}))
    return value


def _check_protocol(root: Path, output: Path) -> dict:
    value = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if value != protocol(root):
        raise ValueError("Frozen localisation protocol/source/input hashes differ; no overwrite or silent refresh")
    return value


def retained_cases(root: Path):
    from analyze_score_preserving_controls import build_control_emissions, iter_retained_cases
    for case in iter_retained_cases(root):
        meta = {key: case[key] for key in ("study", "scorer", "repeat_seed", "duration", "schedule", "cell", "status", "error")}
        if case["status"] != "ok":
            yield meta, None, None, None
            continue
        emissions = build_control_emissions(case["raw"], case["forced"], case["native"], case["gate_accepted"])
        emissions["oracle_label"] = case["oracle"]
        emissions["static_mean"] = np.repeat(case["raw"].mean(axis=0)[None], len(case["raw"]), axis=0)
        if tuple(emissions) != METHODS:
            raise ValueError("Control-emission adapter method inventory/order drift")
        rows, edge_arrays, grid = evaluate_localisation(emissions, case["raw"], case["bounds"], case["regimes"], case["state_support"])
        yield meta, rows, edge_arrays, grid


def _case_name(meta: dict) -> str:
    return f"{meta['study']}_{meta['scorer']}_{meta['repeat_seed']}_D{meta['duration']}_{meta['schedule']}"


def _interval(values: np.ndarray, identity: tuple, draws: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, float)
    if not len(values):
        return np.nan, np.nan
    if np.all(values == values[0]):
        return float(values[0]), float(values[0])
    words = np.frombuffer(hashlib.sha256(json.dumps(identity, default=str).encode()).digest()[:16], dtype="<u4")
    rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, *map(int, words)]))
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    return tuple(map(float, np.quantile(values[indices].mean(axis=1), [.025, .975])))


def summarise(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, paired = [], []
    for identity, frame in detail.groupby(list(GROUP), sort=True, dropna=False):
        base = dict(zip(GROUP, identity))
        raw = frame[frame.method == "raw_window"].set_index("repeat_seed")
        if raw.index.has_duplicates:
            raise ValueError("Duplicate repeat in a localisation cell")
        for method, rows in frame.groupby("method", sort=False):
            rows = rows.set_index("repeat_seed").sort_index()
            if rows.index.has_duplicates or set(rows.index) != set(raw.index):
                raise ValueError("Missing/duplicate method row in a paired localisation cell")
            for metric in METRICS:
                values = rows[metric].to_numpy(float)
                valid = np.isfinite(values)
                low, high = _interval(values[valid], (*identity, method, metric, "absolute"))
                summaries.append({**base, "method": method, "metric": metric, "n_planned": len(rows),
                    "n_finite": int(valid.sum()), "n_missing": int((~valid).sum()),
                    "mean": float(values[valid].mean()) if valid.any() else np.nan, "ci95_low": low, "ci95_high": high})
                baseline = raw.loc[rows.index, metric].to_numpy(float)
                matched = valid & np.isfinite(baseline)
                delta = values[matched] - baseline[matched]
                low, high = _interval(delta, (*identity, method, metric, "paired_minus_raw"))
                paired.append({**base, "method": method, "metric": metric, "n_planned": len(rows),
                    "n_paired": int(matched.sum()), "n_missing_pair": int((~matched).sum()),
                    "mean": float(delta.mean()) if len(delta) else np.nan, "ci95_low": low, "ci95_high": high})
    return pd.DataFrame(summaries), pd.DataFrame(paired)


def run(root: Path = ROOT, output: Path = OUT, stage: Path = STAGE) -> dict:
    root, output, stage = Path(root).resolve(), Path(output).resolve(), Path(stage).resolve()
    locked = _check_protocol(root, output)
    if any(path.name != "protocol.json" for path in output.iterdir()):
        raise ValueError("Refusing to overwrite existing localisation outputs")
    started = datetime.now(timezone.utc).isoformat()
    records = output / "records"; records.mkdir()
    details = []; n_cases = failed = 0
    for meta, rows, edge_arrays, grid in retained_cases(root):
        if rows is None:
            failed += 1
            rows = [{"method": method, "status": "retained_fit_failure", **{key: np.nan for key in METRICS}} for method in METHODS]
        details.extend({**meta, **row} for row in rows)
        if edge_arrays is not None:
            np.savez_compressed(records / (_case_name(meta) + ".npz"), **edge_arrays)
        n_cases += 1
        if n_cases % 50 == 0:
            print(f"retained cases processed: {n_cases}", flush=True)
    detail = pd.DataFrame(details)
    detail.to_csv(output / "detail.csv", index=False)
    summary, paired = summarise(detail)
    summary.to_csv(output / "summary.csv", index=False)
    paired.to_csv(output / "paired_vs_raw.csv", index=False)
    if frozen_hashes(root) != locked["frozen_input_sha256"] or frozen_sources(root) != locked["frozen_source_sha256"]:
        raise AssertionError("Frozen studies/sources changed during retained-output analysis")
    (output / "preservation.json").write_bytes(json_bytes({"before_after_frozen_files_equal": True,
        "before_after_frozen_sources_equal": True, "frozen_files": len(locked["frozen_input_sha256"]),
        "frozen_sources": len(locked["frozen_source_sha256"]), "planned_cases_retained": n_cases, "failed_cases_retained": failed,
        "detail_rows": len(detail), "new_simulations": 0, "new_fits": 0}))
    hashes = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "manifest.json").write_bytes(json_bytes({"sha256": hashes, "protocol_sha256": sha(output / "protocol.json"),
        "analysis_source_sha256": locked["analysis_source_sha256"], "frozen_files_unchanged": len(locked["frozen_input_sha256"]),
        "frozen_source_files_unchanged": len(locked["frozen_source_sha256"]), "new_simulations": 0, "new_fits": 0}))
    report = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "cases": n_cases,
        "failed_cases": failed, "detail_rows": len(detail), "summary_rows": len(summary), "paired_rows": len(paired),
        "protocol_sha256": sha(output / "protocol.json"), "manifest_sha256": sha(output / "manifest.json"),
        "new_simulations": 0, "new_fits": 0}
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "temporal_localisation_run.json").write_bytes(json_bytes(report))
    return report


def verify(root: Path = ROOT, output: Path = OUT) -> dict:
    """Pure retained-output recomputation; no audit writer, RNG simulation or fitting."""
    root, output = Path(root).resolve(), Path(output).resolve()
    locked = _check_protocol(root, output)
    before = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected = set(before) - {"manifest.json"}
    if set(manifest["sha256"]) != expected or any(before[name] != digest for name, digest in manifest["sha256"].items()):
        raise ValueError("Localisation output inventory/checksum drift")
    if manifest["protocol_sha256"] != before["protocol.json"] or manifest["analysis_source_sha256"] != locked["analysis_source_sha256"]:
        raise ValueError("Localisation source/protocol manifest drift")
    detail = pd.read_csv(output / "detail.csv")
    expected_rows = []; cases = failed = 0
    for meta, rows, edge_arrays, grid in retained_cases(root):
        if rows is None:
            failed += 1
            rows = [{"method": method, "status": "retained_fit_failure", **{key: np.nan for key in METRICS}} for method in METHODS]
        expected_rows.extend({**meta, **row} for row in rows)
        if edge_arrays is not None:
            with np.load(output / "records" / (_case_name(meta) + ".npz"), allow_pickle=False) as saved:
                if set(saved.files) != set(edge_arrays):
                    raise ValueError("Localisation per-edge cache inventory drift")
                for key, value in edge_arrays.items():
                    if value.dtype.kind in "US":
                        np.testing.assert_array_equal(saved[key], value)
                    else:
                        np.testing.assert_allclose(saved[key], value, rtol=1e-12, atol=1e-14, equal_nan=True)
        cases += 1
    expected_detail = pd.DataFrame(expected_rows)
    # CSV empty strings are read as missing; preserve that serialization convention.
    for column in expected_detail.select_dtypes(include="object"):
        expected_detail[column] = expected_detail[column].map(lambda value: np.nan if isinstance(value, str) and value == "" else value)
    pd.testing.assert_frame_equal(detail, expected_detail, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-14)
    summary, paired = summarise(detail)
    for name, expected_frame in (("summary.csv", summary), ("paired_vs_raw.csv", paired)):
        pd.testing.assert_frame_equal(pd.read_csv(output / name), expected_frame, check_dtype=False,
                                      check_exact=False, rtol=1e-12, atol=1e-14)
    after = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()}
    if before != after or frozen_hashes(root) != locked["frozen_input_sha256"] or frozen_sources(root) != locked["frozen_source_sha256"]:
        raise AssertionError("Read-only localisation verification changed retained files")
    return {"status": "passed", "read_only": True, "cases": cases, "retained_failures": failed,
            "method_rows_recomputed": len(detail), "summary_rows_recomputed": len(summary),
            "paired_rows_recomputed": len(paired), "files_unchanged": len(before),
            "frozen_files_unchanged": len(locked["frozen_input_sha256"]), "frozen_source_files_unchanged": len(locked["frozen_source_sha256"]),
            "new_simulations": 0, "new_fits": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "run", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--stage-dir", type=Path, default=STAGE)
    args = parser.parse_args()
    if args.mode == "freeze":
        value = freeze(args.root, args.output_dir, args.stage_dir)
        report = {"status": "frozen", "methods": len(value["methods"]), "input_files": len(value["frozen_input_sha256"]), "source_files": len(value["frozen_source_sha256"])}
    elif args.mode == "run":
        report = run(args.root, args.output_dir, args.stage_dir)
    else:
        report = verify(args.root, args.output_dir)
    print(json.dumps(clean(report), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
