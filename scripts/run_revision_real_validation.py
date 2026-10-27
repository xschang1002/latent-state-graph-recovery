"""Frozen label-independent intervals and train-only forward real-data checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_real_state_label_prediction as base
import run_real_state_label_prediction_forward as forward
from _common import align_labels
from stcg import STCGStateConfig, rpcmci_scores

DATASETS = {
    "HydraulicSystems": {
        "columns": ["TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"],
        "labels": ["cooler_condition", "valve_condition"],
        "boundary_column": "cycle",
    },
    "AReMActivities": {
        "columns": ["avg_rss12", "var_rss12", "avg_rss13", "var_rss13", "avg_rss23", "var_rss23"],
        "labels": ["activity"],
        "boundary_column": "sequence_id",
    },
}
FORWARD_METHODS = ["Train Majority", "Sensor Centroid", "Graph Centroid", "Sensor Logistic", "STCG Fixed K2", "STCG Train Silhouette K"]
SEGMENT_METHODS = ["STCG Fixed K2", "Sensor KMeans K2", "Regime-PCMCI K2"]
SEEDS = [0, 1, 2, 3, 4]
WINDOW = 300
LAG = 3
PARTS = ROOT / "experiments" / "logs" / "revision_real_validation"
OUT = ROOT / "results" / "tables"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval_plan(n_rows: int) -> list[dict]:
    """Coordinates depend only on the observation count, never on labels."""
    if n_rows < 3000:
        raise ValueError("At least 3000 rows are required for this frozen real-data grid")
    plan = []
    cuts = [int(n_rows * i / 5) for i in range(1, 6)]
    for i in range(4):
        plan.append(dict(protocol="forward", interval_id=i, train_start=0,
                         train_end=cuts[i], start=cuts[i], end=cuts[i + 1]))
    for i, fraction in enumerate([0.10, 0.35, 0.60, 0.85]):
        start = min(max(0, int(n_rows * fraction) - 750), n_rows - 1500)
        plan.append(dict(protocol="segment", interval_id=i, train_start=-1,
                         train_end=-1, start=start, end=start + 1500))
    return plan


def complete_windows(start: int, end: int, window: int = WINDOW):
    return tuple((left, left + window) for left in range(start, end - window + 1, window))


def load_sensors(dataset):
    path = ROOT / "data" / "raw" / f"{dataset}.csv"
    frame = pd.read_csv(path)[DATASETS[dataset]["columns"]].apply(pd.to_numeric, errors="coerce").ffill()
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{dataset}: initial missing or non-finite sensor values")
    return values


def config(seed, k=2):
    return STCGStateConfig(n_states=k, base_model="lasso", lasso_alpha=0.005,
                           window_size=WINDOW, stride=WINDOW, contrast_mix=0.0,
                           auto_states=True, seed=95000 + seed)


def train_only_states(features, train_indices, seed, choose_k=False):
    """Optional K choice and all across-window scaling use training data only."""
    chosen_k = 2
    criterion = math.nan
    if choose_k:
        training = features[train_indices]
        scaled = StandardScaler().fit_transform(training)
        best = (-math.inf, -2)
        for k in range(2, min(6, len(training) - 1) + 1):
            model = KMeans(n_clusters=k, n_init=10, random_state=95000 + seed)
            states = model.fit_predict(scaled)
            if 1 < np.unique(states).size < len(states):
                score = float(silhouette_score(scaled, states))
                if (score, -k) > best:
                    best = (score, -k)
                    chosen_k, criterion = k, score
    states, improvement, coherence, accepted = forward.assign_train_fitted_states(
        features, np.asarray(train_indices, dtype=int), config(seed, chosen_k)
    )
    return states, dict(supplied_k=chosen_k, selection_score=criterion,
                        state_improvement=improvement, temporal_coherence=coherence,
                        states_accepted=accepted,
                        inferred_train_k=int(np.unique(states[train_indices]).size))


def predict_from_training(sensor, graph, labels, train, test, seed):
    """Training labels never include test labels; features may contain test windows."""
    train_labels = [labels[int(i)] for i in train]
    majority = base.majority_label(train_labels)[0]
    predictions = {"Train Majority": [majority] * len(test)}
    predictions["Sensor Centroid"] = base.centroid_predict(sensor, labels, train, test)
    predictions["Graph Centroid"] = base.centroid_predict(graph, labels, train, test)
    if len(set(train_labels)) == 1:
        predictions["Sensor Logistic"] = [majority] * len(test)
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=5000, solver="lbfgs"))
        model.fit(sensor[train], train_labels)
        predictions["Sensor Logistic"] = model.predict(sensor[test]).tolist()
    metadata = {}
    for method, choose_k in [("STCG Fixed K2", False), ("STCG Train Silhouette K", True)]:
        states, info = train_only_states(graph, train, seed, choose_k=choose_k)
        mapping, fallback = base.fit_state_label_mapping(states, labels, train)
        predictions[method] = base.predict_from_state_mapping(states, mapping, fallback, test)
        info.update(state_mapping=json.dumps(mapping, sort_keys=True),
                    inferred_test_k=int(np.unique(states[test]).size))
        metadata[method] = info
    return predictions, metadata


def counts(values):
    return json.dumps(pd.Series(values).value_counts().sort_index().to_dict(), sort_keys=True)


def evaluate_forward(dataset, x, labels_frame, plan):
    rows, predictions = [], []
    for interval in [row for row in plan if row["protocol"] == "forward"]:
        bounds = complete_windows(0, interval["train_end"]) + complete_windows(interval["start"], interval["end"])
        n_train = len(complete_windows(0, interval["train_end"]))
        train = np.arange(n_train)
        test = np.arange(n_train, len(bounds))
        started = time.perf_counter()
        sensor = base.sensor_window_features(x, bounds)
        graph = forward.window_features(x, bounds, LAG, config(0))
        feature_seconds = time.perf_counter() - started
        recording_crossings = 0
        if dataset == "AReMActivities":
            recording_crossings = sum(labels_frame["sequence_id"].iloc[bounds[i][0]:bounds[i][1]].nunique() > 1 for i in test)
        for label_column in DATASETS[dataset]["labels"]:
            true, purities = base.window_majority_labels(labels_frame[label_column].astype(str).tolist(), bounds)
            training = [true[i] for i in train]
            testing = [true[i] for i in test]
            unseen_fraction = float(np.mean([label not in set(training) for label in testing]))
            for seed in SEEDS:
                started = time.perf_counter()
                predicted, info = predict_from_training(sensor, graph, true, train, test, seed)
                fit_seconds = time.perf_counter() - started
                for method in FORWARD_METHODS:
                    row = dict(dataset=dataset, label_column=label_column, interval_id=interval["interval_id"],
                               seed=seed, method=method, status="ok", train_start=0,
                               train_end=interval["train_end"], test_start=interval["start"],
                               test_end=interval["end"], train_windows=len(train), test_windows=len(test),
                               train_label_counts=counts(training), test_label_counts=counts(testing),
                               true_test_k=len(set(testing)), unseen_class_fraction=unseen_fraction,
                               test_majority_purity=float(np.mean(np.asarray(purities)[test])),
                               test_recording_boundary_windows=recording_crossings,
                               feature_seconds=feature_seconds, all_methods_fit_seconds=fit_seconds)
                    row.update(info.get(method, {}))
                    row.update(base.metric_values(testing, predicted[method]))
                    rows.append(row)
                    for idx, window_id in enumerate(test):
                        start, end = bounds[int(window_id)]
                        predictions.append(dict(dataset=dataset, label_column=label_column,
                                                interval_id=interval["interval_id"], seed=seed, method=method,
                                                start=start, end=end, true_label=testing[idx],
                                                predicted_label=predicted[method][idx],
                                                label_purity=purities[int(window_id)]))
        print(f"Forward completed: {dataset}, origin {interval['interval_id'] + 1}/4", flush=True)
    return rows, predictions


def segment_worker(dataset, interval_id, output):
    x = load_sensors(dataset)
    interval = [p for p in interval_plan(len(x)) if p["protocol"] == "segment"][interval_id]
    segment = x[interval["start"]:interval["end"]]
    started = time.perf_counter()
    try:
        _, info = rpcmci_scores(segment, LAG, num_regimes=2, num_iterations=20,
                                max_anneal=10, max_transitions=10, switch_thres=0.05,
                                pc_alpha=0.05, alpha_level=0.01, seed=0, n_jobs=1, return_info=True)
        # Wrapper gamma covers the full segment; the first lag positions are unscored.
        result = dict(status="ok", sequence=np.asarray(info["regime_sequence"]).tolist(),
                      offset=int(info["effective_offset"]),
                      error_free_annealings=int(info["error_free_annealings"]))
    except Exception as exc:
        result = dict(status="failed", error=f"{type(exc).__name__}: {exc}",
                      sequence=[], offset=LAG, error_free_annealings=0)
    result["worker_fit_seconds"] = time.perf_counter() - started
    Path(output).write_text(json.dumps(result), encoding="utf-8")


def run_worker_with_budget(command, timeout_seconds, output, log_path):
    """A real wall timeout terminates the isolated single-job worker."""
    started = time.perf_counter()
    Path(output).unlink(missing_ok=True)
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return dict(status="timeout", error=f"worker terminated at fixed {timeout_seconds:g}s wall budget",
                        sequence=[], offset=LAG, error_free_annealings=None,
                        worker_wall_seconds=time.perf_counter() - started)
    if not Path(output).is_file():
        return dict(status="failed", error=f"worker exit {process.returncode} without result", sequence=[],
                    offset=LAG, error_free_annealings=None, worker_wall_seconds=time.perf_counter() - started)
    result = json.loads(Path(output).read_text(encoding="utf-8"))
    result["worker_wall_seconds"] = time.perf_counter() - started
    return result


def alignment(true, predicted):
    return dict(ari=float(adjusted_rand_score(true, predicted)),
                aligned_accuracy=float(align_labels(true, predicted)[2]))


def score_segment(dataset, interval, labels_frame, method, result):
    start, end = interval["start"], interval["end"]
    bounds = complete_windows(0, end - start)
    rows = []
    for label_column in DATASETS[dataset]["labels"]:
        truth = labels_frame[label_column].iloc[start:end].astype(str).tolist()
        window_truth, _ = base.window_majority_labels(truth, bounds)
        row = dict(dataset=dataset, label_column=label_column, interval_id=interval["interval_id"],
                   method=method, seed=0, start=start, end=end, supplied_k=2,
                   status=result["status"], error=result.get("error", ""),
                   true_sample_k=len(set(truth)), true_window_k=len(set(window_truth)),
                   true_label_counts=counts(truth), transitions=sum(a != b for a, b in zip(truth, truth[1:])),
                   n_windows=len(bounds), error_free_annealings=result.get("error_free_annealings", ""),
                   fit_seconds=result.get("worker_fit_seconds", result.get("fit_seconds", math.nan)),
                   worker_wall_seconds=result.get("worker_wall_seconds", math.nan),
                   timeout_budget_seconds=120 if method == "Regime-PCMCI K2" else math.nan)
        if result["status"] == "timeout":
            # A terminated worker has no returned optimizer diagnostics.
            row["error_free_annealings"] = None
        if dataset == "AReMActivities":
            recording_ids = labels_frame["sequence_id"].iloc[start:end].to_numpy()
            row["recording_boundaries"] = int(np.sum(recording_ids[1:] != recording_ids[:-1]))
        if result["status"] == "ok":
            sequence = np.asarray(result["sequence"], dtype=int)
            offset = int(result.get("offset", 0))
            if len(sequence) != len(truth):
                raise ValueError("State sequence must be aligned to the full segment")
            pred_windows = [int(base.majority_label(sequence[max(a, offset):b].tolist())[0]) for a, b in bounds]
            row.update({f"sample_{k}": v for k, v in alignment(truth[offset:], sequence[offset:].tolist()).items()})
            row.update({f"window_{k}": v for k, v in alignment(window_truth, pred_windows).items()})
            row.update(inferred_sample_k=len(set(sequence[offset:])), inferred_window_k=len(set(pred_windows)),
                       state_collapse=int(len(set(sequence[offset:])) == 1),
                       scored_samples=len(truth) - offset)
        rows.append(row)
    return rows


def evaluate_segments(dataset, x, labels_frame, plan, signature, aggregate_only=False):
    rows = []
    for interval in [row for row in plan if row["protocol"] == "segment"]:
        segment = x[interval["start"]:interval["end"]]
        bounds = complete_windows(0, len(segment))
        features = forward.window_features(segment, bounds, LAG, config(0))
        for method in SEGMENT_METHODS:
            stem = f"segment_{dataset}_{interval['interval_id']}_{method.replace(' ', '_')}"
            checkpoint = PARTS / f"{stem}.json"
            if checkpoint.is_file():
                cached = json.loads(checkpoint.read_text(encoding="utf-8"))
                if cached["signature"] != signature:
                    raise ValueError(f"Stale checkpoint: {checkpoint}; use a distinct parts directory")
                result = cached["result"]
            else:
                if aggregate_only:
                    raise ValueError(f"Missing checkpoint for reporting-only aggregation: {checkpoint}")
                started = time.perf_counter()
                if method == "Regime-PCMCI K2":
                    raw_output = PARTS / f"{stem}_worker.json"
                    result = run_worker_with_budget(
                        [sys.executable, str(Path(__file__).resolve()), "--worker", dataset,
                         str(interval["interval_id"]), str(raw_output)], 120, raw_output,
                        PARTS / f"{stem}.log")
                else:
                    if method == "STCG Fixed K2":
                        states, info = train_only_states(features, np.arange(len(bounds)), 0)
                    else:
                        sensor = StandardScaler().fit_transform(base.sensor_window_features(segment, bounds))
                        states = KMeans(n_clusters=2, n_init=10, random_state=95000).fit_predict(sensor)
                    result = dict(status="ok", sequence=np.repeat(states, WINDOW).tolist(), offset=0,
                                  fit_seconds=time.perf_counter() - started)
                checkpoint.write_text(json.dumps(dict(signature=signature, result=result)), encoding="utf-8")
            rows.extend(score_segment(dataset, interval, labels_frame, method, result))
            print(f"Segment completed: {dataset} {interval['interval_id'] + 1}/4 {method}: {result['status']}", flush=True)
    return rows


def save_csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / f"revision_real_{name}.csv", index=False, lineterminator="\n")


def summarize_forward(rows):
    frame = pd.DataFrame(rows)
    keys = ["dataset", "label_column", "method"]
    origin = frame.groupby(keys + ["interval_id"], as_index=False).agg(
        balanced_accuracy=("balanced_accuracy", "mean"), macro_f1=("macro_f1", "mean"),
        accuracy=("accuracy", "mean"), seed_balanced_accuracy_std=("balanced_accuracy", "std"),
        true_test_k=("true_test_k", "first"), unseen_class_fraction=("unseen_class_fraction", "first"))
    save_csv("forward_origins", origin.to_dict("records"))
    summaries = []
    for key, group in origin.groupby(keys):
        row = dict(zip(keys, key))
        row.update(origins=len(group), multistate_test_origins=int((group.true_test_k > 1).sum()),
                   unseen_class_fraction_mean=float(group.unseen_class_fraction.mean()))
        for metric in ["balanced_accuracy", "macro_f1", "accuracy"]:
            row.update({f"{metric}_mean": float(group[metric].mean()),
                        f"{metric}_min": float(group[metric].min()), f"{metric}_max": float(group[metric].max())})
        summaries.append(row)
    save_csv("forward_summary", summaries)
    lines = ["# Frozen forward diagnosis validation", "", "Seeds are averaged within each origin. Ranges describe four dependent origins; no significance claim is made.", "", "| Dataset / label | Method | Multi-state test origins | Balanced accuracy mean [range] | Macro F1 | Unseen fraction |", "|---|---|---:|---|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['dataset']} / {row['label_column']} | {row['method']} | {row['multistate_test_origins']}/4 | {row['balanced_accuracy_mean']:.3f} [{row['balanced_accuracy_min']:.3f}, {row['balanced_accuracy_max']:.3f}] | {row['macro_f1_mean']:.3f} | {row['unseen_class_fraction_mean']:.3f} |")
    (OUT / "revision_real_forward_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    latex = [r"\begin{tabular}{llrrr}", r"\toprule", r"Label & Method & Bal. Acc. & Macro F1 & Multi-state tests \\", r"\midrule"]
    for row in summaries:
        label = str(row["label_column"]).replace("_", r"\_")
        latex.append(f"{label} & {row['method']} & {row['balanced_accuracy_mean']:.3f} & {row['macro_f1_mean']:.3f} & {row['multistate_test_origins']}/4 \\\\")
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "revision_real_forward_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def summarize_segments(rows):
    frame = pd.DataFrame(rows)
    lines = ["# Label-independent fixed-segment validation", "", "All eight selected sensor segments are retained, including no-transition segments and resource-limited failures. Hydraulic label columns reuse the same fitted states. ARI on a single true state is not informative evidence of recovery.", "", "| Dataset / label | Segment | Method | Status | True sample/window K | Sample ARI | Window ARI | Inferred K |", "|---|---:|---|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['dataset']} / {row['label_column']} | {row['interval_id']} | {row['method']} | {row['status']} | {row['true_sample_k']}/{row['true_window_k']} | {row.get('sample_ari', math.nan):.3f} | {row.get('window_ari', math.nan):.3f} | {row.get('inferred_sample_k', '')} |")
    summaries = []
    for key, group in frame.groupby(["dataset", "label_column", "method"]):
        successful = group[group.status == "ok"]
        multi = successful[successful.true_window_k > 1]
        summaries.append(dict(dataset=key[0], label_column=key[1], method=key[2], planned=len(group),
                              completed=len(successful), failed=int((group.status == "failed").sum()),
                              timed_out=int((group.status == "timeout").sum()),
                              multistate_window_segments=len(multi),
                              multistate_window_ari_mean=float(multi.window_ari.mean()) if len(multi) else math.nan,
                              all_sample_ari_mean=float(successful.sample_ari.mean()) if len(successful) else math.nan))
    save_csv("segment_summary", summaries)
    (OUT / "revision_real_segment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    global PARTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["plan", "forward", "segments", "all"], default="all")
    parser.add_argument("--worker", nargs=3, metavar=("DATASET", "INTERVAL", "OUTPUT"))
    parser.add_argument("--parts-dir", type=Path, default=PARTS)
    parser.add_argument("--aggregate-from-manifest", action="store_true",
                        help="Rebuild reports from the recorded execution and its cached results; never fit missing jobs.")
    args = parser.parse_args()
    if args.worker:
        segment_worker(args.worker[0], int(args.worker[1]), args.worker[2])
        return 0
    PARTS = args.parts_dir
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    arrays = {dataset: load_sensors(dataset) for dataset in DATASETS}
    plans = {dataset: interval_plan(len(values)) for dataset, values in arrays.items()}
    plan_rows = [dict(dataset=dataset, **row) for dataset, entries in plans.items() for row in entries]
    save_csv("plan", plan_rows)
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in [Path(__file__).resolve(),
              ROOT / "src/stcg/baselines.py", ROOT / "src/stcg/self_supervised.py",
              ROOT / "scripts/run_real_state_label_prediction.py", ROOT / "scripts/run_real_state_label_prediction_forward.py"]}
    for dataset in DATASETS:
        for path in [ROOT / f"data/raw/{dataset}.csv", ROOT / f"data/processed/{dataset}_labels.csv"]:
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    signature = hashlib.sha256(json.dumps(dict(plan=plan_rows, hashes=hashes), sort_keys=True).encode()).hexdigest()
    manifest = dict(signature=signature, plan=plan_rows, hashes=hashes, seeds=SEEDS, segment_seed=0, worker_budget_seconds=120)
    manifest_path = PARTS / "manifest.json"
    if args.aggregate_from_manifest:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if recorded["plan"] != plan_rows:
            raise ValueError("Recorded execution coordinates differ from the current plan")
        for path, digest in recorded["hashes"].items():
            if path != str(Path(__file__).resolve().relative_to(ROOT)) and hashes[path] != digest:
                raise ValueError(f"Recorded data/dependency changed: {path}")
        signature = recorded["signature"]
        (PARTS / "reporting_manifest.json").write_text(json.dumps(
            dict(execution_signature=signature, reporting_source_sha256=hashes[str(Path(__file__).resolve().relative_to(ROOT))]),
            indent=2), encoding="utf-8")
    elif manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8"))["signature"] != signature:
        raise ValueError("Existing execution has different source/data; preserve it and use --parts-dir for a fresh run, or --aggregate-from-manifest to rebuild its reports")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Frozen plan written: {len(plan_rows)} dataset/protocol intervals; signature={signature}", flush=True)
    if args.stage == "plan":
        return 0
    if args.stage in ("forward", "all"):
        rows, predictions = [], []
        for dataset, x in arrays.items():
            checkpoint = PARTS / f"forward_{dataset}.json"
            if checkpoint.is_file():
                cached = json.loads(checkpoint.read_text(encoding="utf-8"))
                if cached["signature"] != signature:
                    raise ValueError(f"Stale checkpoint: {checkpoint}")
                result = cached["result"]
            else:
                if args.aggregate_from_manifest:
                    raise ValueError(f"Missing checkpoint for reporting-only aggregation: {checkpoint}")
                labels = pd.read_csv(ROOT / f"data/processed/{dataset}_labels.csv")
                if len(labels) != len(x):
                    raise ValueError(f"{dataset}: labels and sensor rows do not align")
                result = evaluate_forward(dataset, x, labels, plans[dataset])
                checkpoint.write_text(json.dumps(dict(signature=signature, result=result)), encoding="utf-8")
            rows.extend(result[0]); predictions.extend(result[1])
        save_csv("forward_detail", rows)
        save_csv("forward_predictions", predictions)
        summarize_forward(rows)
    if args.stage in ("segments", "all"):
        rows = []
        for dataset, x in arrays.items():
            labels = pd.read_csv(ROOT / f"data/processed/{dataset}_labels.csv")
            if len(labels) != len(x):
                raise ValueError(f"{dataset}: labels and sensor rows do not align")
            rows.extend(evaluate_segments(dataset, x, labels, plans[dataset], signature,
                                          aggregate_only=args.aggregate_from_manifest))
        save_csv("segment_detail", rows)
        summarize_segments(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
