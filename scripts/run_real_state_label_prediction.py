"""DEPRECATED -- kept for archaeology; use run_real_state_label_prediction_forward.py.

This harness is systematically optimistic: it fits KMeans state centroids on
the full series (including test windows) and randomly splits OVERLAPPING
windows (window 300 / stride 150) into train/test, so test windows share
most of their samples with train windows. The corrected canonical variant
scripts/run_real_state_label_prediction_forward.py fits centroids on train
windows only, splits windows chronologically, and drops cross-boundary
windows; it is what the paper cites. This file remains in the repo only
because docs/reproduction_manifest.md and git history reference it; do not
use it for new numbers.

Original docstring: Evaluate inferred graph states on held-out real window
labels.

This is a downstream diagnosis harness rather than a one-step forecasting
benchmark. It infers STCG graph states without label access, then uses a
stratified subset of window labels to map states to external operating labels
and evaluates that mapping on held-out windows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

# Set before sklearn imports so joblib/loky does not oversubscribe CPU cores
# when KMeans runs on Windows.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from _common import counts_text, interpolate_within_segments, majority_label, parse_columns, write_csv
from stcg import STCGStateConfig, stcg_v1_diagnostics
from stcg.self_supervised import _nonself_flatten, _window_reconstruction_scores

warnings.filterwarnings("ignore", category=ConvergenceWarning)


METHOD_MAJORITY = "Train Majority"
METHOD_SENSOR = "SensorMeanStd Centroid"
METHOD_GRAPH = "GraphFeature Centroid"
METHOD_STCG_STATE = "STCG-v1 State Mapping"
METRICS = ["accuracy", "balanced_accuracy", "macro_f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=ROOT / "data" / "raw" / "HydraulicSystems.csv")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=ROOT / "data" / "processed" / "HydraulicSystems_labels.csv",
    )
    parser.add_argument("--dataset-name", type=str, default="HydraulicSystems")
    parser.add_argument("--label-column", type=str, default="cooler_condition")
    parser.add_argument("--columns", type=str, default="TS1,TS2,TS3,TS4,VS1,CE,CP,SE")
    parser.add_argument("--time-column", type=str, default="time_index")
    parser.add_argument("--missing", choices=["drop", "interpolate"], default="interpolate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--test-fraction", type=float, default=0.50)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--lasso-alpha", type=float, default=0.005)
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--window-stride", type=int, default=150)
    parser.add_argument("--window-aggregate", choices=["max", "mean", "p90"], default="max")
    parser.add_argument("--state-aggregate", choices=["mean", "median", "p75", "max"], default="mean")
    parser.add_argument("--stcg-v1-states", type=int, default=2)
    parser.add_argument("--stcg-v1-min-improvement", type=float, default=0.12)
    parser.add_argument("--stcg-v1-force-states", action="store_true")
    parser.add_argument("--plot-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "results" / "figures")
    return parser.parse_args()


def load_series_and_labels(
    args: argparse.Namespace,
    split_fractions: tuple[float, ...] = (0.5,),
) -> tuple[str, np.ndarray, list[str], list[str]]:
    frame = pd.read_csv(args.input_csv)
    labels_frame = pd.read_csv(args.labels_csv)
    if len(frame) != len(labels_frame):
        raise ValueError(
            f"Input rows ({len(frame)}) and label rows ({len(labels_frame)}) must match"
        )
    if args.label_column not in labels_frame.columns:
        raise ValueError(f"{args.labels_csv} is missing label column: {args.label_column}")

    columns = parse_columns(args.columns)
    if columns:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Requested columns not found: {missing}")
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    else:
        drop_columns = [args.time_column] if args.time_column else []
        numeric = frame.drop(columns=drop_columns, errors="ignore").select_dtypes(include=[np.number])
        columns = [str(column) for column in numeric.columns]

    if numeric.shape[1] < 2:
        raise ValueError("Need at least two numeric columns for graph-state diagnosis")
    if args.missing == "interpolate":
        # Interpolate within chronological segments so fill values never cross
        # a train/test boundary. ``split_fractions`` marks the caller's
        # chronological boundary: the forward variant passes its train
        # fraction (0.80); this deprecated script and the count-selection
        # script split windows 50/50, so the midpoint is the minimal
        # equivalent boundary.
        split_at = [int(round(len(numeric) * fraction)) for fraction in split_fractions]
        numeric = interpolate_within_segments(numeric, split_at).dropna(axis=0)
    else:
        numeric = numeric.dropna(axis=0)

    values = numeric.to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    kept_index = numeric.index.to_numpy()[finite]
    std = values.std(axis=0)
    keep_columns = std > 1e-10
    values = values[:, keep_columns]
    kept_columns = [column for column, keep in zip(columns, keep_columns) if keep]
    if values.shape[1] < 2:
        raise ValueError("Need at least two non-constant numeric columns after cleaning")

    label_values = labels_frame.iloc[kept_index][args.label_column].dropna().astype(str).tolist()
    if len(label_values) != len(values):
        raise ValueError("Label cleaning removed rows inconsistently with the time series")
    name = args.dataset_name or args.input_csv.stem
    return name, values, kept_columns, label_values


def stcg_config(args: argparse.Namespace, seed: int) -> STCGStateConfig:
    return STCGStateConfig(
        n_states=args.stcg_v1_states,
        base_model="lasso",
        lasso_alpha=args.lasso_alpha,
        window_size=args.window_size,
        stride=args.window_stride,
        aggregate=args.window_aggregate,
        state_aggregate=args.state_aggregate,
        auto_states=not args.stcg_v1_force_states,
        min_state_improvement=args.stcg_v1_min_improvement,
        contrast_mix=0.0,
        seed=seed + 70_000,
    )


def window_majority_labels(
    labels: list[str],
    bounds: tuple[tuple[int, int], ...],
) -> tuple[list[str], list[float]]:
    true_labels: list[str] = []
    purities: list[float] = []
    for start, end in bounds:
        if start < 0 or end > len(labels) or end <= start:
            raise ValueError(f"Invalid window [{start}, {end}) for {len(labels)} labels")
        label, purity = majority_label(labels[start:end])
        true_labels.append(label)
        purities.append(purity)
    return true_labels, purities


def stratified_window_split(
    labels: list[str],
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in sorted(set(labels)):
        indices = np.array([idx for idx, value in enumerate(labels) if value == label], dtype=int)
        rng.shuffle(indices)
        if indices.size == 1:
            train_indices.extend(indices.tolist())
            continue
        n_test = int(round(indices.size * test_fraction))
        n_test = min(max(1, n_test), indices.size - 1)
        test_indices.extend(indices[:n_test].tolist())
        train_indices.extend(indices[n_test:].tolist())
    return np.array(sorted(train_indices), dtype=int), np.array(sorted(test_indices), dtype=int)


def fit_state_label_mapping(
    states: np.ndarray,
    true_labels: list[str],
    train_indices: np.ndarray,
) -> tuple[dict[int, str], str]:
    fallback, _ = majority_label([true_labels[idx] for idx in train_indices])
    mapping: dict[int, str] = {}
    for state in sorted(set(int(states[idx]) for idx in train_indices)):
        member_labels = [true_labels[idx] for idx in train_indices if int(states[idx]) == state]
        mapping[state], _ = majority_label(member_labels)
    return mapping, fallback


def predict_from_state_mapping(
    states: np.ndarray,
    mapping: dict[int, str],
    fallback: str,
    indices: np.ndarray,
) -> list[str]:
    return [mapping.get(int(states[idx]), fallback) for idx in indices]


def zscore_by_train(train_features: np.ndarray, test_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True) + 1e-8
    return (train_features - mean) / std, (test_features - mean) / std


def centroid_predict(
    features: np.ndarray,
    true_labels: list[str],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> list[str]:
    train_features, test_features = zscore_by_train(features[train_indices], features[test_indices])
    train_labels = [true_labels[idx] for idx in train_indices]
    labels = sorted(set(train_labels))
    if len(labels) == 1:
        return [labels[0]] * len(test_indices)
    centroids = np.stack(
        [train_features[np.array([label == value for label in train_labels])].mean(axis=0) for value in labels],
        axis=0,
    )
    distances = np.square(test_features[:, None, :] - centroids[None, :, :]).sum(axis=2)
    return [labels[int(idx)] for idx in distances.argmin(axis=1)]


def sensor_window_features(x: np.ndarray, bounds: tuple[tuple[int, int], ...]) -> np.ndarray:
    rows = []
    for start, end in bounds:
        window = x[start:end]
        rows.append(np.concatenate([window.mean(axis=0), window.std(axis=0), window[-1] - window[0]]))
    return np.stack(rows, axis=0)


def graph_window_features(
    x: np.ndarray,
    bounds: tuple[tuple[int, int], ...],
    max_lag: int,
    config: STCGStateConfig,
) -> np.ndarray:
    rows = []
    for start, end in bounds:
        scores = _window_reconstruction_scores(x[start:end], max_lag=max_lag, config=config)
        rows.append(_nonself_flatten(scores))
    return np.stack(rows, axis=0)


def accuracy_score(true: list[str], predicted: list[str]) -> float:
    return float(np.mean([str(a) == str(b) for a, b in zip(true, predicted)]))


def balanced_accuracy(true: list[str], predicted: list[str]) -> float:
    recalls = []
    for label in sorted(set(true)):
        indices = [idx for idx, value in enumerate(true) if value == label]
        recalls.append(float(np.mean([predicted[idx] == label for idx in indices])))
    return float(np.mean(recalls)) if recalls else math.nan


def macro_f1(true: list[str], predicted: list[str]) -> float:
    scores = []
    for label in sorted(set(true) | set(predicted)):
        tp = sum(1 for a, b in zip(true, predicted) if a == label and b == label)
        fp = sum(1 for a, b in zip(true, predicted) if a != label and b == label)
        fn = sum(1 for a, b in zip(true, predicted) if a == label and b != label)
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else float(2 * tp / denom))
    return float(np.mean(scores)) if scores else math.nan


def metric_values(true: list[str], predicted: list[str]) -> dict[str, float]:
    if len(true) != len(predicted):
        raise ValueError("true and predicted labels must have the same length")
    return {
        "accuracy": accuracy_score(true, predicted),
        "balanced_accuracy": balanced_accuracy(true, predicted),
        "macro_f1": macro_f1(true, predicted),
    }


def row_for_method(
    dataset: str,
    seed: int,
    label_column: str,
    method: str,
    true_test: list[str],
    predicted: list[str],
    train_labels: list[str],
    test_labels: list[str],
    diagnostics,
    mapping: dict[int, str] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset,
        "seed": seed,
        "label_column": label_column,
        "method": method,
        "train_windows": len(train_labels),
        "test_windows": len(test_labels),
        "train_label_counts": counts_text(train_labels),
        "test_label_counts": counts_text(test_labels),
        "inferred_state_count": int(np.unique(diagnostics.labels).size),
        "state_improvement": float(diagnostics.state_improvement),
        "temporal_coherence": float(diagnostics.temporal_coherence),
        "states_accepted": bool(diagnostics.states_accepted),
        "state_mapping": json.dumps(mapping or {}, sort_keys=True),
    }
    row.update(metric_values(true_test, predicted))
    return row


def evaluate_one(
    dataset: str,
    x: np.ndarray,
    labels: list[str],
    label_column: str,
    args: argparse.Namespace,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    config = stcg_config(args, seed)
    diagnostics = stcg_v1_diagnostics(x, max_lag=args.max_lag, config=config)
    bounds = tuple(diagnostics.bounds)
    true_labels, purities = window_majority_labels(labels, bounds)
    train_indices, test_indices = stratified_window_split(true_labels, args.test_fraction, seed)
    train_labels = [true_labels[idx] for idx in train_indices]
    test_labels = [true_labels[idx] for idx in test_indices]

    majority, _ = majority_label(train_labels)
    majority_pred = [majority] * len(test_indices)
    sensor_pred = centroid_predict(sensor_window_features(x, bounds), true_labels, train_indices, test_indices)
    graph_features = diagnostics.features
    if graph_features.shape[0] != len(bounds):
        graph_features = graph_window_features(x, bounds, args.max_lag, config)
    graph_pred = centroid_predict(graph_features, true_labels, train_indices, test_indices)
    state_mapping, fallback = fit_state_label_mapping(diagnostics.labels, true_labels, train_indices)
    state_pred = predict_from_state_mapping(diagnostics.labels, state_mapping, fallback, test_indices)

    rows = [
        row_for_method(
            dataset,
            seed,
            label_column,
            METHOD_MAJORITY,
            test_labels,
            majority_pred,
            train_labels,
            test_labels,
            diagnostics,
        ),
        row_for_method(
            dataset,
            seed,
            label_column,
            METHOD_SENSOR,
            test_labels,
            sensor_pred,
            train_labels,
            test_labels,
            diagnostics,
        ),
        row_for_method(
            dataset,
            seed,
            label_column,
            METHOD_GRAPH,
            test_labels,
            graph_pred,
            train_labels,
            test_labels,
            diagnostics,
        ),
        row_for_method(
            dataset,
            seed,
            label_column,
            METHOD_STCG_STATE,
            test_labels,
            state_pred,
            train_labels,
            test_labels,
            diagnostics,
            state_mapping,
        ),
    ]

    predictions_by_method = {
        METHOD_MAJORITY: majority_pred,
        METHOD_SENSOR: sensor_pred,
        METHOD_GRAPH: graph_pred,
        METHOD_STCG_STATE: state_pred,
    }
    detail_rows: list[dict[str, object]] = []
    for method, predicted_labels in predictions_by_method.items():
        for local_idx, window_idx in enumerate(test_indices):
            start, end = bounds[int(window_idx)]
            detail_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "label_column": label_column,
                    "method": method,
                    "window_id": int(window_idx),
                    "start": int(start),
                    "end": int(end),
                    "true_label": test_labels[local_idx],
                    "predicted_label": predicted_labels[local_idx],
                    "match": int(test_labels[local_idx] == predicted_labels[local_idx]),
                    "label_purity": float(purities[int(window_idx)]),
                    "inferred_state": int(diagnostics.labels[int(window_idx)]),
                }
            )
    return rows, detail_rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["label_column"]), str(row["method"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (dataset, label_column, method), items in sorted(groups.items()):
        summary: dict[str, object] = {
            "dataset": dataset,
            "label_column": label_column,
            "method": method,
            "n": len(items),
            "accepted_rate": float(np.mean([bool(item["states_accepted"]) for item in items])),
            "inferred_state_count_distribution": json.dumps(
                dict(sorted(Counter(int(item["inferred_state_count"]) for item in items).items())),
                sort_keys=True,
            ),
        }
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    return summary_rows


def write_markdown(path: Path, summary_rows: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    lines = [
        "# Real State Label Prediction",
        "",
        "This downstream diagnosis uses labels only after unsupervised graph-state inference.",
        "Window labels are split stratified by external label; train windows define label mappings and held-out windows are evaluated.",
        "",
        "## Aggregate",
        "",
        "| Dataset | Label | Method | N | Accuracy | Balanced Acc. | Macro F1 | Accepted Rate | State Counts |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {label_column} | {method} | {n} | {accuracy_mean:.3f} +/- {accuracy_std:.3f} | "
            "{balanced_accuracy_mean:.3f} +/- {balanced_accuracy_std:.3f} | "
            "{macro_f1_mean:.3f} +/- {macro_f1_std:.3f} | {accepted_rate:.3f} | `{counts}` |".format(
                counts=row["inferred_state_count_distribution"],
                **row,
            )
        )

    lines.extend(
        [
            "",
            "## By Seed",
            "",
            "| Dataset | Seed | Label | Method | Train Windows | Test Windows | Accuracy | Balanced Acc. | Macro F1 | Train Counts | Test Counts | State Mapping |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {dataset} | {seed} | {label_column} | {method} | {train_windows} | {test_windows} | "
            "{accuracy:.3f} | {balanced_accuracy:.3f} | {macro_f1:.3f} | {train_label_counts} | "
            "{test_label_counts} | `{state_mapping}` |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `STCG-v1 State Mapping` is the compact unsupervised graph-state signal.",
            "- `GraphFeature Centroid` tests whether windowed graph-score features contain label information with supervised centroids.",
            "- `SensorMeanStd Centroid` is a simple supervised sensor-feature ceiling, not a graph-state method.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, summary_rows: list[dict[str, object]]) -> None:
    methods = [str(row["method"]) for row in summary_rows]
    values = [float(row["balanced_accuracy_mean"]) for row in summary_rows]
    errors = [float(row["balanced_accuracy_std"]) for row in summary_rows]
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    bars = ax.barh(methods, values, xerr=errors, capsize=3, color="#4c78a8")
    ax.set_xlabel("Held-out balanced accuracy")
    ax.set_xlim(0.0, 1.05)
    ax.grid(axis="x", alpha=0.25)
    best = max(values)
    for bar, value in zip(bars, values):
        ax.text(min(value + 0.02, 1.02), bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
        bar.set_alpha(1.0 if value == best else 0.72)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    dataset, x, _columns, labels = load_series_and_labels(args)

    rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        eval_rows, eval_details = evaluate_one(dataset, x, labels, args.label_column, args, seed)
        rows.extend(eval_rows)
        detail_rows.extend(eval_details)

    summary_rows = aggregate(rows)
    prefix = dataset
    detail_path = args.output_dir / f"{prefix}_state_label_prediction_detail.csv"
    summary_path = args.output_dir / f"{prefix}_state_label_prediction_summary.csv"
    markdown_path = args.output_dir / f"{prefix}_state_label_prediction_summary.md"
    figure_path = args.figure_dir / f"{prefix}_state_label_prediction_balanced_accuracy.png"

    write_csv(
        detail_path,
        detail_rows,
        [
            "dataset",
            "seed",
            "label_column",
            "method",
            "window_id",
            "start",
            "end",
            "true_label",
            "predicted_label",
            "match",
            "label_purity",
            "inferred_state",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        [
            "dataset",
            "label_column",
            "method",
            "n",
            "accepted_rate",
            "inferred_state_count_distribution",
            "accuracy_mean",
            "accuracy_std",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
        ],
    )
    write_markdown(markdown_path, summary_rows, rows)
    write_figure(figure_path, summary_rows)

    best = max(summary_rows, key=lambda row: float(row["balanced_accuracy_mean"]))
    print(
        json.dumps(
            {
                "detail_csv": str(detail_path),
                "summary_csv": str(summary_path),
                "summary_md": str(markdown_path),
                "figure": str(figure_path),
                "n_rows": len(rows),
                "n_detail_rows": len(detail_rows),
                "best_balanced_accuracy_method": best["method"],
                "best_balanced_accuracy": best["balanced_accuracy_mean"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
