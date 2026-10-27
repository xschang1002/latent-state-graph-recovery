"""Sweep AReM STCG auto-state thresholds and downstream forecasting."""

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

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _common import (
    align_labels,
    counts_text,
    fmt_na_lenient as fmt,
    interpolate_within_segments,
    majority_label,
    parse_columns,
    state_config,
    write_csv,
)
from stcg import (
    STCGStateConfig,
    chronological_lagged_split,
    dynamic_lasso_var_scores,
    forecast_metrics,
    graph_feature_weights,
    ridge_forecast,
    ridge_forecast_weighted,
    stcg_v1_diagnostics,
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)

METRICS = ["mae", "rmse", "node_rmse_mean", "node_rmse_max"]
STATE_METRICS = [
    "kmeans_improvement",
    "temporal_coherence",
    "adjusted_rand",
    "aligned_accuracy",
    "mean_label_purity",
]
METHOD_ORDER = {
    "Dense Ridge": 0,
    "DynVAR-Lasso Weighted Ridge": 1,
    "STCG-v1 Weighted Ridge": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=ROOT / "data" / "raw" / "AReMActivities.csv")
    parser.add_argument("--labels-csv", type=Path, default=ROOT / "data" / "processed" / "AReMActivities_labels.csv")
    parser.add_argument("--dataset-name", type=str, default="AReMActivities")
    parser.add_argument("--label-column", type=str, default="activity")
    parser.add_argument("--time-column", type=str, default="time_index")
    parser.add_argument("--columns", type=str, default="")
    parser.add_argument("--missing", choices=["drop", "interpolate"], default="interpolate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--lasso-alpha", type=float, default=0.005)
    parser.add_argument("--state-counts", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--window-stride", type=int, default=150)
    parser.add_argument("--window-aggregate", choices=["max", "mean", "p90"], default="max")
    parser.add_argument("--state-aggregate", choices=["mean", "median", "p75", "max"], default="mean")
    parser.add_argument(
        "--min-improvements",
        nargs="+",
        type=float,
        default=[0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20],
    )
    parser.add_argument("--include-forced", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ridge-alphas", nargs="+", type=float, default=[30.0, 100.0])
    parser.add_argument("--graph-min-weights", nargs="+", type=float, default=[0.01])
    parser.add_argument("--graph-self-weight", type=float, default=1.0)
    parser.add_argument("--graph-weight-powers", nargs="+", type=float, default=[4.0])
    parser.add_argument("--meaningful-rmse-delta", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    return parser.parse_args()


def load_series(args: argparse.Namespace) -> tuple[np.ndarray, list[str], list[int]]:
    frame = pd.read_csv(args.input_csv)
    columns = parse_columns(args.columns)
    if columns:
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    else:
        numeric = frame.drop(columns=[args.time_column], errors="ignore").select_dtypes(include=[np.number])
        columns = [str(column) for column in numeric.columns]

    if args.missing == "interpolate":
        # Interpolate within the chronological train/test segments so a train
        # gap never borrows test-period values (and vice versa). The boundary
        # mirrors chronological_lagged_split: int(round(n * train_fraction)).
        split_at = int(round(len(numeric) * args.train_fraction))
        numeric = interpolate_within_segments(numeric, [split_at]).dropna(axis=0)
    else:
        numeric = numeric.dropna(axis=0)

    values = numeric.to_numpy(dtype=float)
    row_positions = numeric.index.to_numpy()
    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    row_positions = row_positions[finite]
    keep = values.std(axis=0) > 1e-10
    values = values[:, keep]
    columns = [column for column, keep_column in zip(columns, keep) if keep_column]
    if values.shape[1] < 2:
        raise ValueError("Need at least two non-constant numeric columns")
    return values, columns, [int(position) for position in row_positions]


def load_labels(path: Path, label_column: str) -> list[str]:
    frame = pd.read_csv(path)
    if label_column not in frame.columns:
        raise ValueError(f"{path} is missing label column: {label_column}")
    return frame[label_column].astype(str).tolist()


def kmeans_improvement(features: np.ndarray, n_states: int, seed: int) -> float:
    if features.shape[0] < 2 or n_states <= 1:
        return 0.0
    n_states = min(n_states, features.shape[0])
    centered = features - features.mean(axis=0, keepdims=True)
    scaled = centered / (centered.std(axis=0, keepdims=True) + 1e-8)
    inertia_one = float(np.square(scaled - scaled.mean(axis=0, keepdims=True)).sum())
    if inertia_one <= 1e-8:
        return 0.0
    model = KMeans(n_clusters=n_states, n_init=10, random_state=seed)
    model.fit(scaled)
    return max(0.0, (inertia_one - float(model.inertia_)) / inertia_one)


def state_metrics(
    args: argparse.Namespace,
    diagnostics,
    train_labels: list[str],
    seed: int,
    state_count: int,
    min_improvement: float,
    auto_states: bool,
) -> dict[str, object]:
    true_labels: list[str] = []
    purities: list[float] = []
    for start, end in diagnostics.bounds:
        label, purity = majority_label(train_labels[start:end])
        true_labels.append(label)
        purities.append(purity)

    inferred_states = [int(state) for state in diagnostics.labels]
    _, mapping, accuracy = align_labels(true_labels, inferred_states)
    improvement = diagnostics.state_improvement
    return {
        "state_count": state_count,
        "auto_states": auto_states,
        "min_state_improvement": min_improvement,
        "threshold_passed": bool(diagnostics.states_accepted),
        "kmeans_improvement": improvement,
        "temporal_coherence": diagnostics.temporal_coherence,
        "n_windows": len(true_labels),
        "true_state_count": len(set(true_labels)),
        "inferred_state_count": len(set(inferred_states)),
        "adjusted_rand": float(adjusted_rand_score(true_labels, inferred_states)),
        "aligned_accuracy": accuracy,
        "mean_label_purity": float(np.mean(purities)),
        "true_label_counts": counts_text(true_labels),
        "inferred_state_counts": counts_text(inferred_states),
        "state_mapping": json.dumps(mapping, sort_keys=True),
    }


def metric_row(
    args: argparse.Namespace,
    seed: int,
    setting: str,
    state_row: dict[str, object],
    method: str,
    ridge_alpha: float,
    prediction: np.ndarray,
    target: np.ndarray,
    graph_min_weight: float | None = None,
    graph_weight_power: float | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": args.dataset_name,
        "seed": seed,
        "setting": setting,
        "method": method,
        "ridge_alpha": ridge_alpha,
        "graph_min_weight": graph_min_weight,
        "graph_weight_power": graph_weight_power,
        "graph_self_weight": args.graph_self_weight if graph_min_weight is not None else None,
        **state_row,
    }
    row.update(forecast_metrics(target, prediction))
    return row


def setting_name(state_count: int, min_improvement: float, auto_states: bool) -> str:
    if not auto_states:
        return f"forced_k{state_count}"
    return f"auto_k{state_count}_min_{min_improvement:.2f}"


def evaluate_setting(
    args: argparse.Namespace,
    seed: int,
    split,
    x_train: np.ndarray,
    train_labels: list[str],
    dyn_scores: np.ndarray,
    state_count: int,
    min_improvement: float,
    auto_states: bool,
) -> list[dict[str, object]]:
    config = state_config(args, seed, n_states=state_count, min_state_improvement=min_improvement, auto_states=auto_states)
    diagnostics = stcg_v1_diagnostics(x_train, max_lag=args.max_lag, config=config)
    states = state_metrics(args, diagnostics, train_labels, seed, state_count, min_improvement, auto_states)
    stcg_scores = diagnostics.scores
    target = split.target_test
    setting = setting_name(state_count, min_improvement, auto_states)

    rows: list[dict[str, object]] = []
    weight_configs = []
    for graph_min_weight in args.graph_min_weights:
        for graph_weight_power in args.graph_weight_powers:
            weight_configs.append(
                (
                    graph_min_weight,
                    graph_weight_power,
                    graph_feature_weights(
                        dyn_scores,
                        min_weight=graph_min_weight,
                        self_weight=args.graph_self_weight,
                        power=graph_weight_power,
                    ),
                    graph_feature_weights(
                        stcg_scores,
                        min_weight=graph_min_weight,
                        self_weight=args.graph_self_weight,
                        power=graph_weight_power,
                    ),
                )
            )

    for ridge_alpha in args.ridge_alphas:
        rows.append(
            metric_row(
                args,
                seed,
                setting,
                states,
                "Dense Ridge",
                ridge_alpha,
                ridge_forecast(split, alpha=ridge_alpha),
                target,
            )
        )
        for graph_min_weight, graph_weight_power, dyn_weights, stcg_weights in weight_configs:
            rows.append(
                metric_row(
                    args,
                    seed,
                    setting,
                    states,
                    "DynVAR-Lasso Weighted Ridge",
                    ridge_alpha,
                    ridge_forecast_weighted(split, alpha=ridge_alpha, feature_weights=dyn_weights),
                    target,
                    graph_min_weight=graph_min_weight,
                    graph_weight_power=graph_weight_power,
                )
            )
            rows.append(
                metric_row(
                    args,
                    seed,
                    setting,
                    states,
                    "STCG-v1 Weighted Ridge",
                    ridge_alpha,
                    ridge_forecast_weighted(split, alpha=ridge_alpha, feature_weights=stcg_weights),
                    target,
                    graph_min_weight=graph_min_weight,
                    graph_weight_power=graph_weight_power,
                )
            )
    return rows


def evaluate(args: argparse.Namespace) -> list[dict[str, object]]:
    values, _, row_positions = load_series(args)
    labels = load_labels(args.labels_csv, args.label_column)
    if row_positions and max(row_positions) >= len(labels):
        raise ValueError("Labels are shorter than the feature matrix")
    aligned_labels = [labels[position] for position in row_positions]

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        split = chronological_lagged_split(values, max_lag=args.max_lag, train_fraction=args.train_fraction)
        x_train = values[: split.train_end]
        train_labels = aligned_labels[: split.train_end]
        dyn_scores = dynamic_lasso_var_scores(
            x_train,
            max_lag=args.max_lag,
            alpha=args.lasso_alpha,
            window_size=args.window_size,
            stride=args.window_stride,
            aggregate=args.window_aggregate,
        )
        for state_count in args.state_counts:
            for min_improvement in args.min_improvements:
                rows.extend(
                    evaluate_setting(
                        args,
                        seed,
                        split,
                        x_train,
                        train_labels,
                        dyn_scores,
                        state_count=state_count,
                        min_improvement=min_improvement,
                        auto_states=True,
                    )
                )
            if args.include_forced:
                rows.extend(
                    evaluate_setting(
                        args,
                        seed,
                        split,
                        x_train,
                        train_labels,
                        dyn_scores,
                        state_count=state_count,
                        min_improvement=0.0,
                        auto_states=False,
                    )
                )
    return rows


def group_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["dataset"],
        row["setting"],
        row["method"],
        row["ridge_alpha"],
        row["graph_min_weight"],
        row["graph_weight_power"],
        row["graph_self_weight"],
    )


def sort_optional(value: object) -> float:
    if value in {None, ""}:
        return math.inf
    number = float(value)
    return number if math.isfinite(number) else math.inf


def aggregate_summary(rows: list[dict[str, object]], meaningful_rmse_delta: float) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    summary: list[dict[str, object]] = []
    for key, items in sorted(
        groups.items(),
        key=lambda item: (
            str(item[0][1]),
            METHOD_ORDER.get(str(item[0][2]), 99),
            sort_optional(item[0][3]),
            sort_optional(item[0][4]),
            sort_optional(item[0][5]),
        ),
    ):
        dataset, setting, method, ridge_alpha, graph_min_weight, graph_weight_power, graph_self_weight = key
        first = items[0]
        row: dict[str, object] = {
            "dataset": dataset,
            "setting": setting,
            "method": method,
            "n": len(items),
            "ridge_alpha": ridge_alpha,
            "graph_min_weight": graph_min_weight,
            "graph_weight_power": graph_weight_power,
            "graph_self_weight": graph_self_weight,
            "state_count": first["state_count"],
            "auto_states": first["auto_states"],
            "min_state_improvement": first["min_state_improvement"],
            "threshold_pass_rate": float(np.mean([bool(item["threshold_passed"]) for item in items])),
            "inferred_state_count_distribution": json.dumps(
                dict(sorted(Counter(int(item["inferred_state_count"]) for item in items).items())),
                sort_keys=True,
            ),
        }
        for metric in [*METRICS, *STATE_METRICS]:
            values = [float(item[metric]) for item in items]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(row)

    best_dense_by_dataset: dict[str, float] = {}
    same_dense: dict[tuple[str, object, float], float] = {}
    same_dynvar: dict[tuple[str, object, float, object, object], float] = {}
    for row in summary:
        dataset = str(row["dataset"])
        setting = row["setting"]
        method = str(row["method"])
        ridge_alpha = float(row["ridge_alpha"])
        rmse = float(row["rmse_mean"])
        if method == "Dense Ridge":
            best_dense_by_dataset[dataset] = min(best_dense_by_dataset.get(dataset, math.inf), rmse)
            same_dense[(dataset, setting, ridge_alpha)] = rmse
        elif method == "DynVAR-Lasso Weighted Ridge":
            same_dynvar[(dataset, setting, ridge_alpha, row["graph_min_weight"], row["graph_weight_power"])] = rmse

    for row in summary:
        dataset = str(row["dataset"])
        setting = row["setting"]
        ridge_alpha = float(row["ridge_alpha"])
        rmse = float(row["rmse_mean"])
        dense_rmse = same_dense.get((dataset, setting, ridge_alpha), math.nan)
        best_dense = best_dense_by_dataset.get(dataset, math.nan)
        dynvar_rmse = same_dynvar.get(
            (dataset, setting, ridge_alpha, row["graph_min_weight"], row["graph_weight_power"]),
            math.nan,
        )
        row["delta_vs_same_alpha_dense_rmse"] = rmse - dense_rmse
        row["delta_vs_best_dense_rmse"] = rmse - best_dense
        row["delta_vs_dynvar_weighted_rmse"] = rmse - dynvar_rmse
        row["meaningfully_beats_same_alpha_dense"] = bool(
            math.isfinite(dense_rmse) and rmse < dense_rmse - meaningful_rmse_delta
        )
        row["meaningfully_beats_best_dense"] = bool(
            math.isfinite(best_dense) and rmse < best_dense - meaningful_rmse_delta
        )
        row["meaningfully_beats_dynvar_weighted"] = bool(
            math.isfinite(dynvar_rmse) and rmse < dynvar_rmse - meaningful_rmse_delta
        )
    return summary


def best_rows(rows: list[dict[str, object]], method: str) -> list[dict[str, object]]:
    return sorted([row for row in rows if str(row["method"]) == method], key=lambda row: float(row["rmse_mean"]))


def state_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: dict[str, dict[str, object]] = {}
    for row in rows:
        key = str(row["setting"])
        current = seen.get(key)
        if current is None or float(row["adjusted_rand_mean"]) > float(current["adjusted_rand_mean"]):
            seen[key] = row
    return sorted(seen.values(), key=lambda row: (int(row["state_count"]), float(row["min_state_improvement"]), str(row["setting"])))


def write_markdown(path: Path, rows: list[dict[str, object]], meaningful_rmse_delta: float) -> None:
    stcg_best = best_rows(rows, "STCG-v1 Weighted Ridge")
    dynvar_best = best_rows(rows, "DynVAR-Lasso Weighted Ridge")
    dense_best = best_rows(rows, "Dense Ridge")
    best_stcg = stcg_best[0]
    dataset = str(best_stcg["dataset"])
    lines = [
        f"# {dataset} State-Threshold Forecasting Sweep",
        "",
        "This sweep tests whether relaxing the STCG-v1 auto-state acceptance threshold converts external mode-label alignment into downstream forecasting gains.",
        f"Improvements smaller than `{meaningful_rmse_delta:.3f}` RMSE are treated as numerical-scale changes.",
        "",
        "## State Threshold Curve",
        "",
        "| Setting | K | Pass Rate | KMeans Improvement | Temporal Coherence | Inferred States | ARI | Aligned Acc. | RMSE (Best STCG) | Delta vs DynVAR |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    best_stcg_by_setting: dict[str, dict[str, object]] = {}
    for row in sorted(stcg_best, key=lambda row: (str(row["setting"]), float(row["rmse_mean"]))):
        best_stcg_by_setting.setdefault(str(row["setting"]), row)
    for row in state_rows(rows):
        setting = str(row["setting"])
        best_for_setting = best_stcg_by_setting.get(setting, row)
        lines.append(
            "| {setting} | {state_count} | {pass_rate} | {imp} | {coherence} | `{states}` | {ari} +/- {ari_std} | {acc} +/- {acc_std} | {rmse} | {dyn_delta} |".format(
                setting=setting,
                state_count=row["state_count"],
                pass_rate=fmt(row["threshold_pass_rate"], 2),
                imp=fmt(row["kmeans_improvement_mean"], 3),
                coherence=fmt(row["temporal_coherence_mean"], 3),
                states=row["inferred_state_count_distribution"],
                ari=fmt(row["adjusted_rand_mean"], 3),
                ari_std=fmt(row["adjusted_rand_std"], 3),
                acc=fmt(row["aligned_accuracy_mean"], 3),
                acc_std=fmt(row["aligned_accuracy_std"], 3),
                rmse=fmt(best_for_setting["rmse_mean"], 6),
                dyn_delta=fmt(best_for_setting["delta_vs_dynvar_weighted_rmse"], 6),
            )
        )

    lines.extend(
        [
            "",
            "## Best Forecasting Rows",
            "",
            "| Method | Setting | RMSE | Alpha | Min Weight | Power | Delta vs Best Dense | Delta vs DynVAR | Meaningful vs DynVAR |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in [dense_best[0], dynvar_best[0], best_stcg]:
        lines.append(
            "| {method} | {setting} | {rmse} | {alpha} | {min_weight} | {power} | {dense_delta} | {dyn_delta} | {beats_dynvar} |".format(
                method=row["method"],
                setting=row["setting"],
                rmse=fmt(row["rmse_mean"], 6),
                alpha=fmt(row["ridge_alpha"], 3),
                min_weight=fmt(row["graph_min_weight"], 2),
                power=fmt(row["graph_weight_power"], 2),
                dense_delta=fmt(row["delta_vs_best_dense_rmse"], 6),
                dyn_delta=fmt(row["delta_vs_dynvar_weighted_rmse"], 6),
                beats_dynvar="yes" if row["meaningfully_beats_dynvar_weighted"] else "no",
            )
        )

    any_stateful_stcg = any(float(row["adjusted_rand_mean"]) > 0.0 and int(json.loads(row["inferred_state_count_distribution"]).get("1", 0)) < int(row["n"]) for row in stcg_best)
    any_meaningful_dynvar_gain = any(bool(row["meaningfully_beats_dynvar_weighted"]) for row in stcg_best)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Any non-collapsed STCG setting tested: {'yes' if any_stateful_stcg else 'no'}.",
            f"- Best STCG setting meaningfully beats DynVAR-Lasso weighted ridge: {'yes' if any_meaningful_dynvar_gain else 'no'}.",
            f"- Best STCG setting: `{best_stcg['setting']}` with RMSE `{fmt(best_stcg['rmse_mean'], 6)}` and ARI `{fmt(best_stcg['adjusted_rand_mean'], 3)}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_rows = evaluate(args)
    summary_rows = aggregate_summary(detail_rows, args.meaningful_rmse_delta)

    prefix = args.dataset_name
    detail_path = args.output_dir / f"{prefix}_state_threshold_forecasting_detail.csv"
    summary_path = args.output_dir / f"{prefix}_state_threshold_forecasting_summary.csv"
    markdown_path = args.output_dir / f"{prefix}_state_threshold_forecasting_summary.md"

    common_fields = [
        "dataset",
        "seed",
        "setting",
        "method",
        "ridge_alpha",
        "graph_min_weight",
        "graph_weight_power",
        "graph_self_weight",
        "state_count",
        "auto_states",
        "min_state_improvement",
        "threshold_passed",
        "kmeans_improvement",
        "temporal_coherence",
        "n_windows",
        "true_state_count",
        "inferred_state_count",
        "adjusted_rand",
        "aligned_accuracy",
        "mean_label_purity",
        "true_label_counts",
        "inferred_state_counts",
        "state_mapping",
        *METRICS,
    ]
    summary_fields = [
        "dataset",
        "setting",
        "method",
        "n",
        "ridge_alpha",
        "graph_min_weight",
        "graph_weight_power",
        "graph_self_weight",
        "state_count",
        "auto_states",
        "min_state_improvement",
        "threshold_pass_rate",
        "inferred_state_count_distribution",
        *[f"{metric}_{suffix}" for metric in [*METRICS, *STATE_METRICS] for suffix in ("mean", "std")],
        "delta_vs_same_alpha_dense_rmse",
        "delta_vs_best_dense_rmse",
        "delta_vs_dynvar_weighted_rmse",
        "meaningfully_beats_same_alpha_dense",
        "meaningfully_beats_best_dense",
        "meaningfully_beats_dynvar_weighted",
    ]
    write_csv(detail_path, detail_rows, common_fields)
    write_csv(summary_path, summary_rows, summary_fields)
    write_markdown(markdown_path, summary_rows, args.meaningful_rmse_delta)
    print(
        json.dumps(
            {
                "detail_csv": str(detail_path),
                "summary_csv": str(summary_path),
                "summary_md": str(markdown_path),
                "n_detail_rows": len(detail_rows),
                "n_summary_rows": len(summary_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
