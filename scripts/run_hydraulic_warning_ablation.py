"""Ablate Hydraulic transition-warning STCG selector and window settings.

The script evaluates a small grid of label-free STCG warning configurations.
Configuration selection is performed from the chronological validation segment;
test metrics are then reported for the validation-selected rows.
"""

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
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import write_csv
import run_hydraulic_transition_warning as warning


STCG_METHODS = [warning.METHOD_STCG, warning.METHOD_SENSOR_STCG]
CONFIG_FIELDS = [
    "config_id",
    "config_rank",
    "selection_objective",
    "stcg_window_size",
    "stcg_window_stride",
    "state_count_distribution",
    "inferred_state_count_distribution",
    "dynamic_feature_nonzero_variance",
]
EVAL_FIELDS = [
    "dataset",
    "label_column",
    "horizon_cycles",
    "lookback_cycles",
    "split_policy",
    "purge_horizon_cycles",
    "stcg_selection_policy",
    "method",
    "train_rows",
    "validation_rows",
    "test_rows",
    "train_positive_rate",
    "validation_positive_rate",
    "test_positive_rate",
    "train_positives",
    "validation_positives",
    "test_positives",
    "threshold_metric",
    "threshold",
    "validation_threshold_metric",
    *warning.METRICS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=ROOT / "data" / "raw" / "HydraulicSystems.csv")
    parser.add_argument("--labels-csv", type=Path, default=ROOT / "data" / "processed" / "HydraulicSystems_labels.csv")
    parser.add_argument("--dataset-name", type=str, default="HydraulicSystems")
    parser.add_argument("--label-columns", nargs="+", default=["valve_condition", "stable_flag"])
    parser.add_argument("--columns", type=str, default="TS1,TS2,TS3,TS4,VS1,CE,CP,SE")
    parser.add_argument("--cycle-column", type=str, default="cycle")
    parser.add_argument("--horizon-cycles", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--lookback-cycles", nargs="+", type=int, default=[30, 40, 60])
    parser.add_argument("--selection-objectives", nargs="+", choices=["stable", "transition"], default=["stable", "transition"])
    parser.add_argument("--stcg-window-sizes", nargs="+", type=int, default=[16, 24])
    parser.add_argument("--stcg-window-strides", nargs="+", type=int, default=[8])
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--stcg-v1-base", choices=["ridge", "lasso"], default="ridge")
    parser.add_argument("--stcg-v1-lasso-alpha", type=float, default=0.005)
    parser.add_argument("--stcg-state-counts", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--stcg-min-improvement", type=float, default=0.04)
    parser.add_argument("--stcg-min-temporal-coherence", type=float, default=0.55)
    parser.add_argument("--stcg-min-cluster-fraction", type=float, default=0.05)
    parser.add_argument("--stcg-selection-policy", choices=["reject_nonpositive", "legacy"], default="reject_nonpositive")
    parser.add_argument("--stcg-seed", type=int, default=70_000)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--threshold-metric", choices=["f1", "balanced_accuracy"], default="f1")
    parser.add_argument("--selection-metric", choices=["validation_threshold_metric"], default="validation_threshold_metric")
    parser.add_argument("--min-test-positives", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "results" / "figures")
    return parser.parse_args()


def config_id(objective: str, lookback: int, window_size: int, stride: int) -> str:
    return f"{objective}_L{lookback}_W{window_size}_S{stride}"


def format_count_distribution(values: np.ndarray) -> str:
    counts = Counter(int(value) for value in values)
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def annotate_state_diagnostics(
    row: dict[str, object],
    graph_x: np.ndarray,
    stcg_x: np.ndarray,
    args: argparse.Namespace,
) -> None:
    dynamic_width = max(args.stcg_state_counts) * 3 + 6
    state_count_column = graph_x.shape[1] + dynamic_width
    inferred_count_column = state_count_column + 1
    dynamic_features = stcg_x[:, graph_x.shape[1] : graph_x.shape[1] + dynamic_width]
    row["state_count_distribution"] = format_count_distribution(stcg_x[:, state_count_column])
    row["inferred_state_count_distribution"] = format_count_distribution(stcg_x[:, inferred_count_column])
    row["dynamic_feature_nonzero_variance"] = int(np.sum(np.var(dynamic_features, axis=0) > 1e-12))


def build_warning_args(base: argparse.Namespace, objective: str, lookback: int, window_size: int, stride: int) -> argparse.Namespace:
    values = vars(base).copy()
    values["lookback_cycles"] = lookback
    values["stcg_selection_objective"] = objective
    values["stcg_window_size"] = window_size
    values["stcg_window_stride"] = stride
    return argparse.Namespace(**values)


def evaluate_label_horizon(
    dataset: str,
    label_column: str,
    horizon: int,
    cycle_features,
    cycle_labels,
    args: argparse.Namespace,
    feature_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[dict[str, object]]:
    sensor_x, graph_x, stcg_x, y, _cycles = warning.build_warning_design(
        cycle_features,
        cycle_labels[label_column],
        horizon=horizon,
        lookback=args.lookback_cycles,
        max_lag=args.max_lag,
        ridge_alpha=args.ridge_alpha,
        args=args,
        feature_cache=feature_cache,
    )
    train_idx, validation_idx, test_idx = warning.chronological_train_validation_test_split(
        len(y),
        args.train_fraction,
        args.validation_fraction,
        horizon=horizon,
    )
    y_train = y[train_idx]
    y_validation = y[validation_idx]
    y_test = y[test_idx]
    if int(np.sum(y_test)) < args.min_test_positives:
        return []

    majority_score = np.full(y_test.shape[0], float(np.mean(y_train)))
    majority_validation_score = np.full(y_validation.shape[0], float(np.mean(y_train)))
    sensor_validation_score, sensor_score = warning.fit_logistic_scores(
        sensor_x[train_idx],
        y_train,
        sensor_x[validation_idx],
        sensor_x[test_idx],
    )
    graph_validation_score, graph_score = warning.fit_logistic_scores(
        graph_x[train_idx],
        y_train,
        graph_x[validation_idx],
        graph_x[test_idx],
    )
    stcg_validation_score, stcg_score = warning.fit_logistic_scores(
        stcg_x[train_idx],
        y_train,
        stcg_x[validation_idx],
        stcg_x[test_idx],
    )
    combined_x = np.concatenate([sensor_x, graph_x], axis=1)
    combined_validation_score, combined_score = warning.fit_logistic_scores(
        combined_x[train_idx],
        y_train,
        combined_x[validation_idx],
        combined_x[test_idx],
    )
    sensor_stcg_x = np.concatenate([sensor_x, stcg_x], axis=1)
    sensor_stcg_validation_score, sensor_stcg_score = warning.fit_logistic_scores(
        sensor_stcg_x[train_idx],
        y_train,
        sensor_stcg_x[validation_idx],
        sensor_stcg_x[test_idx],
    )

    rows = [
        warning.evaluate_method(dataset, label_column, horizon, args.lookback_cycles, warning.METHOD_MAJORITY, y_train, y_validation, y_test, majority_validation_score, majority_score, args.threshold_metric),
        warning.evaluate_method(dataset, label_column, horizon, args.lookback_cycles, warning.METHOD_SENSOR, y_train, y_validation, y_test, sensor_validation_score, sensor_score, args.threshold_metric),
        warning.evaluate_method(dataset, label_column, horizon, args.lookback_cycles, warning.METHOD_GRAPH, y_train, y_validation, y_test, graph_validation_score, graph_score, args.threshold_metric),
        warning.evaluate_method(dataset, label_column, horizon, args.lookback_cycles, warning.METHOD_SENSOR_GRAPH, y_train, y_validation, y_test, combined_validation_score, combined_score, args.threshold_metric),
        warning.evaluate_method(dataset, label_column, horizon, args.lookback_cycles, warning.METHOD_STCG, y_train, y_validation, y_test, stcg_validation_score, stcg_score, args.threshold_metric),
        warning.evaluate_method(dataset, label_column, horizon, args.lookback_cycles, warning.METHOD_SENSOR_STCG, y_train, y_validation, y_test, sensor_stcg_validation_score, sensor_stcg_score, args.threshold_metric),
    ]
    for row in rows:
        annotate_state_diagnostics(row, graph_x, stcg_x, args)
        row["stcg_selection_policy"] = getattr(args, "stcg_selection_policy", "reject_nonpositive")
    return rows


def evaluate_config(
    base_args: argparse.Namespace,
    cycle_features,
    cycle_labels,
    objective: str,
    lookback: int,
    window_size: int,
    stride: int,
    config_rank: int,
) -> list[dict[str, object]]:
    args = build_warning_args(base_args, objective, lookback, window_size, stride)
    feature_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    for label_column in args.label_columns:
        for horizon in args.horizon_cycles:
            rows.extend(
                evaluate_label_horizon(
                    args.dataset_name,
                    label_column,
                    horizon,
                    cycle_features,
                    cycle_labels,
                    args,
                    feature_cache,
                )
            )
    cid = config_id(objective, lookback, window_size, stride)
    for row in rows:
        row["config_id"] = cid
        row["config_rank"] = config_rank
        row["selection_objective"] = objective
        row["stcg_window_size"] = window_size
        row["stcg_window_stride"] = stride
    return rows


def select_validation_rows(
    rows: list[dict[str, object]],
    methods: list[str],
    metric: str,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["method"] in methods:
            grouped[(str(row["label_column"]), int(row["horizon_cycles"]), str(row["method"]))].append(row)

    selected = []
    for key in sorted(grouped):
        candidates = grouped[key]
        selected.append(
            max(
                candidates,
                key=lambda row: (
                    float(row[metric]),
                    -int(row["config_rank"]),
                ),
            )
        )
    return selected


def write_summary(path: Path, selected_rows: list[dict[str, object]], selection_metric: str) -> None:
    lines = [
        "# Hydraulic Warning STCG Selector Ablation",
        "",
        f"Configurations are selected by `{selection_metric}` on the chronological validation segment.",
        "The table reports held-out test metrics for those validation-selected configurations.",
        "The final horizon targets before each train/validation and validation/test boundary are purged; no target supervision crosses a split boundary.",
        f"STCG selection policy: `{selected_rows[0].get('stcg_selection_policy', 'unknown')}`; all-nonpositive candidate sets use their fitted K=1 candidate by default.",
        "",
        "| Label | Horizon | Method | Objective | Lookback | Window | Validation F1 | AP | F1 | ROC AUC | State K | Active Dyn. |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in selected_rows:
        lines.append(
            "| {label} | {horizon} | {method} | {objective} | {lookback} | {window} | {validation} | {ap} | {f1} | {auc} | `{state_k}` | {active} |".format(
                label=row["label_column"],
                horizon=row["horizon_cycles"],
                method=row["method"],
                objective=row["selection_objective"],
                lookback=row["lookback_cycles"],
                window=row["stcg_window_size"],
                validation=warning.fmt(row["validation_threshold_metric"]),
                ap=warning.fmt(row["average_precision"]),
                f1=warning.fmt(row["f1"]),
                auc=warning.fmt(row["roc_auc"]),
                state_k=row["state_count_distribution"],
                active=row["dynamic_feature_nonzero_variance"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, selected_rows: list[dict[str, object]]) -> None:
    labels = sorted(set(str(row["label_column"]) for row in selected_rows))
    fig, axes = plt.subplots(len(labels), 1, figsize=(8.0, 3.2 * len(labels)), squeeze=False)
    for ax, label in zip(axes[:, 0], labels):
        label_rows = [row for row in selected_rows if str(row["label_column"]) == label]
        horizons = sorted(set(int(row["horizon_cycles"]) for row in label_rows))
        methods = STCG_METHODS
        width = 0.34
        x = np.arange(len(horizons))
        for method_idx, method in enumerate(methods):
            values = []
            for horizon in horizons:
                row = next(item for item in label_rows if int(item["horizon_cycles"]) == horizon and item["method"] == method)
                values.append(float(row["average_precision"]))
            ax.bar(x + (method_idx - 0.5) * width, values, width=width, label=method)
        ax.set_title(label)
        ax.set_xlabel("Warning horizon (cycles)")
        ax.set_ylabel("Average precision")
        ax.set_xticks(x)
        ax.set_xticklabels([str(horizon) for horizon in horizons])
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    cycle_features, cycle_labels, _columns = warning.load_cycle_tables(args)

    rows: list[dict[str, object]] = []
    rank = 0
    for objective in args.selection_objectives:
        for lookback in args.lookback_cycles:
            for window_size in args.stcg_window_sizes:
                for stride in args.stcg_window_strides:
                    rows.extend(evaluate_config(args, cycle_features, cycle_labels, objective, lookback, window_size, stride, rank))
                    print(f"Completed ablation {rank + 1}: {config_id(objective, lookback, window_size, stride)}", flush=True)
                    rank += 1
    if not rows:
        raise ValueError("No ablation rows produced")

    selected_rows = select_validation_rows(rows, STCG_METHODS, args.selection_metric)
    detail_path = args.output_dir / f"{args.dataset_name}_warning_ablation_detail.csv"
    selected_path = args.output_dir / f"{args.dataset_name}_warning_ablation_selected.csv"
    summary_path = args.output_dir / f"{args.dataset_name}_warning_ablation_summary.md"
    figure_path = args.figure_dir / f"{args.dataset_name}_warning_ablation_average_precision.png"
    fieldnames = CONFIG_FIELDS + EVAL_FIELDS
    write_csv(detail_path, rows, fieldnames)
    write_csv(selected_path, selected_rows, fieldnames)
    write_summary(summary_path, selected_rows, args.selection_metric)
    write_figure(figure_path, selected_rows)

    best = max(selected_rows, key=lambda row: (float(row["average_precision"]), float(row["f1"])))
    print(
        json.dumps(
            {
                "detail_csv": str(detail_path),
                "selected_csv": str(selected_path),
                "summary_md": str(summary_path),
                "figure": str(figure_path),
                "n_rows": len(rows),
                "n_selected_rows": len(selected_rows),
                "best_selected_label": best["label_column"],
                "best_selected_horizon_cycles": best["horizon_cycles"],
                "best_selected_method": best["method"],
                "best_selected_average_precision": best["average_precision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
