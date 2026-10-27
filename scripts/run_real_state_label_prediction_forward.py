"""Forward-only real state-label diagnosis with train-fitted graph-state centroids."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
from sklearn.cluster import KMeans

from _common import write_csv
import run_real_state_label_prediction as base
from stcg import STCGStateConfig, sliding_window_bounds
from stcg.self_supervised import (
    _min_cluster_fraction,
    _nonself_flatten,
    _state_improvement,
    _temporal_coherence,
    _window_reconstruction_scores,
)


METHOD_MAJORITY = "Forward Train Majority"
METHOD_SENSOR = "Forward SensorMeanStd Centroid"
METHOD_GRAPH = "Forward GraphFeature Centroid"
METHOD_STCG_STATE = "Forward STCG-v1 State Mapping"
METRICS = ["accuracy", "balanced_accuracy", "macro_f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=ROOT / "data" / "raw" / "HydraulicSystems.csv")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=ROOT / "data" / "processed" / "HydraulicSystems_labels.csv",
    )
    parser.add_argument("--dataset-name", type=str, default="HydraulicSystems_forward")
    parser.add_argument("--label-column", type=str, default="cooler_condition")
    parser.add_argument("--columns", type=str, default="TS1,TS2,TS3,TS4,VS1,CE,CP,SE")
    parser.add_argument("--time-column", type=str, default="time_index")
    parser.add_argument("--missing", choices=["drop", "interpolate"], default="interpolate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--lasso-alpha", type=float, default=0.005)
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--window-stride", type=int, default=150)
    parser.add_argument("--stcg-v1-states", type=int, default=3)
    parser.add_argument("--stcg-v1-min-improvement", type=float, default=0.0)
    parser.add_argument("--stcg-v1-force-states", action="store_true")
    parser.add_argument("--plot-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "results" / "figures")
    return parser.parse_args()


def stcg_config(args: argparse.Namespace, seed: int) -> STCGStateConfig:
    return STCGStateConfig(
        n_states=args.stcg_v1_states,
        base_model="lasso",
        lasso_alpha=args.lasso_alpha,
        window_size=args.window_size,
        stride=args.window_stride,
        auto_states=not args.stcg_v1_force_states,
        min_state_improvement=args.stcg_v1_min_improvement,
        contrast_mix=0.0,
        seed=seed + 95_000,
    )


def window_features(
    x: np.ndarray,
    bounds: tuple[tuple[int, int], ...],
    max_lag: int,
    config: STCGStateConfig,
) -> np.ndarray:
    rows = [
        _nonself_flatten(_window_reconstruction_scores(x[start:end], max_lag=max_lag, config=config))
        for start, end in bounds
    ]
    return np.stack(rows, axis=0)


def chronological_window_split(
    bounds: tuple[tuple[int, int], ...],
    n_rows: int,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split = int(round(n_rows * train_fraction))
    train = [idx for idx, (_start, end) in enumerate(bounds) if end <= split]
    test = [idx for idx, (start, _end) in enumerate(bounds) if start >= split]
    if not train or not test:
        raise ValueError("Chronological split produced an empty train or test window set")
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int), split


def assign_train_fitted_states(
    features: np.ndarray,
    train_indices: np.ndarray,
    config: STCGStateConfig,
) -> tuple[np.ndarray, float, float, bool]:
    train_features = features[train_indices]
    if train_features.shape[0] < 2 or config.n_states <= 1:
        return np.zeros(features.shape[0], dtype=int), 0.0, 0.0, False

    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True) + 1e-8
    scaled_train = (train_features - mean) / std
    scaled_all = (features - mean) / std
    n_states = min(config.n_states, train_features.shape[0])
    model = KMeans(n_clusters=n_states, n_init=10, random_state=config.seed)
    train_labels = model.fit_predict(scaled_train)
    improvement = _state_improvement(scaled_train, float(model.inertia_))
    temporal_coherence = _temporal_coherence(train_labels)

    accepted = True
    if config.auto_states:
        inertia_pass = improvement >= config.min_state_improvement
        temporal_pass = (
            improvement >= config.min_temporal_state_improvement
            and temporal_coherence >= config.min_temporal_coherence
            and _min_cluster_fraction(train_labels) >= config.min_temporal_state_fraction
        )
        accepted = inertia_pass or temporal_pass
    if not accepted:
        return np.zeros(features.shape[0], dtype=int), improvement, temporal_coherence, False
    return model.predict(scaled_all), improvement, temporal_coherence, True


def row_for_method(
    dataset: str,
    seed: int,
    label_column: str,
    method: str,
    true_test: list[str],
    predicted: list[str],
    train_labels: list[str],
    test_labels: list[str],
    state_labels: np.ndarray,
    improvement: float,
    coherence: float,
    accepted: bool,
    mapping: dict[int, str] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset,
        "seed": seed,
        "label_column": label_column,
        "method": method,
        "train_windows": len(train_labels),
        "test_windows": len(test_labels),
        "train_label_counts": base.counts_text(train_labels),
        "test_label_counts": base.counts_text(test_labels),
        "inferred_state_count": int(np.unique(state_labels).size),
        "state_improvement": float(improvement),
        "temporal_coherence": float(coherence),
        "states_accepted": bool(accepted),
        "state_mapping": json.dumps(mapping or {}, sort_keys=True),
    }
    row.update(base.metric_values(true_test, predicted))
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
    bounds = tuple(sliding_window_bounds(x.shape[0], args.max_lag, args.window_size, args.window_stride))
    true_labels, purities = base.window_majority_labels(labels, bounds)
    train_indices, test_indices, split = chronological_window_split(bounds, len(labels), args.train_fraction)
    train_labels = [true_labels[idx] for idx in train_indices]
    test_labels = [true_labels[idx] for idx in test_indices]

    graph_features = window_features(x, bounds, args.max_lag, config)
    state_labels, improvement, coherence, accepted = assign_train_fitted_states(graph_features, train_indices, config)

    majority, _ = base.majority_label(train_labels)
    majority_pred = [majority] * len(test_indices)
    sensor_pred = base.centroid_predict(base.sensor_window_features(x, bounds), true_labels, train_indices, test_indices)
    graph_pred = base.centroid_predict(graph_features, true_labels, train_indices, test_indices)
    state_mapping, fallback = base.fit_state_label_mapping(state_labels, true_labels, train_indices)
    state_pred = base.predict_from_state_mapping(state_labels, state_mapping, fallback, test_indices)

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
            state_labels,
            improvement,
            coherence,
            accepted,
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
            state_labels,
            improvement,
            coherence,
            accepted,
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
            state_labels,
            improvement,
            coherence,
            accepted,
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
            state_labels,
            improvement,
            coherence,
            accepted,
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
                    "split_index": split,
                    "window_id": int(window_idx),
                    "start": int(start),
                    "end": int(end),
                    "true_label": test_labels[local_idx],
                    "predicted_label": predicted_labels[local_idx],
                    "match": int(test_labels[local_idx] == predicted_labels[local_idx]),
                    "label_purity": float(purities[int(window_idx)]),
                    "inferred_state": int(state_labels[int(window_idx)]),
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
        "# Forward Real State Label Prediction",
        "",
        "Graph-state centroids and state-label mappings are fit only on chronological train windows.",
        "Test windows start after the train split and are assigned to the train-fitted centroids.",
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
            "| Seed | Method | Train Windows | Test Windows | Balanced Acc. | Train Counts | Test Counts | State Mapping |",
            "|---:|---|---:|---:|---:|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {seed} | {method} | {train_windows} | {test_windows} | {balanced_accuracy:.3f} | "
            "{train_label_counts} | {test_label_counts} | `{state_mapping}` |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, summary_rows: list[dict[str, object]]) -> None:
    methods = [str(row["method"]) for row in summary_rows]
    values = [float(row["balanced_accuracy_mean"]) for row in summary_rows]
    errors = [float(row["balanced_accuracy_std"]) for row in summary_rows]
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    bars = ax.barh(methods, values, xerr=errors, capsize=3, color="#59a14f")
    ax.set_xlabel("Future-window balanced accuracy")
    ax.set_xlim(0.0, 1.05)
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(min(value + 0.02, 1.02), bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    dataset, x, _columns, labels = base.load_series_and_labels(
        args, split_fractions=(args.train_fraction,)
    )

    rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        eval_rows, eval_details = evaluate_one(dataset, x, labels, args.label_column, args, seed)
        rows.extend(eval_rows)
        detail_rows.extend(eval_details)

    summary_rows = aggregate(rows)
    prefix = dataset
    detail_path = args.output_dir / f"{prefix}_state_label_prediction_forward_detail.csv"
    summary_path = args.output_dir / f"{prefix}_state_label_prediction_forward_summary.csv"
    markdown_path = args.output_dir / f"{prefix}_state_label_prediction_forward_summary.md"
    figure_path = args.figure_dir / f"{prefix}_state_label_prediction_forward_balanced_accuracy.png"

    write_csv(
        detail_path,
        detail_rows,
        [
            "dataset",
            "seed",
            "label_column",
            "method",
            "split_index",
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
                "best_method": best["method"],
                "best_balanced_accuracy": best["balanced_accuracy_mean"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
