"""Sweep soft graph-weighted ridge settings on real benchmark CSV datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import warnings
from collections import defaultdict
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

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from _common import fmt_na_lenient as fmt, interpolate_within_segments, parse_columns, write_csv
from stcg import (
    STCGStateConfig,
    chronological_lagged_split,
    dynamic_lasso_var_scores,
    forecast_metrics,
    graph_feature_weights,
    persistence_forecast,
    ridge_forecast,
    ridge_forecast_weighted,
    stcg_v1_diagnostics,
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)


METRICS = ["mae", "rmse", "node_rmse_mean", "node_rmse_max"]
METHOD_ORDER = {
    "Persistence": 0,
    "Dense Ridge": 1,
    "DynVAR-Lasso Weighted Ridge": 2,
    "STCG-v1 Weighted Ridge": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-csv",
        type=Path,
        default=ROOT / "results" / "tables" / "real_data_inventory.csv",
    )
    parser.add_argument("--datasets", nargs="+", default=[])
    parser.add_argument("--statuses", nargs="+", default=["ready", "warning"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--ridge-alphas", nargs="+", type=float, default=[0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    parser.add_argument("--lasso-alpha", type=float, default=0.005)
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--window-stride", type=int, default=150)
    parser.add_argument("--window-aggregate", choices=["max", "mean", "p90"], default="max")
    parser.add_argument("--state-aggregate", choices=["mean", "median", "p75", "max"], default="mean")
    parser.add_argument("--graph-min-weights", nargs="+", type=float, default=[0.01, 0.03, 0.10, 0.30, 0.60])
    parser.add_argument("--graph-self-weight", type=float, default=1.0)
    parser.add_argument("--graph-weight-powers", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--meaningful-rmse-delta", type=float, default=0.001)
    parser.add_argument("--missing", choices=["drop", "interpolate"], default="interpolate")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    return parser.parse_args()


def read_inventory(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Inventory CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def eligible_inventory_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_inventory(args.inventory_csv)
    statuses = set(args.statuses)
    datasets = set(args.datasets)
    selected = []
    for row in rows:
        if row.get("status") not in statuses:
            continue
        if datasets and row.get("dataset") not in datasets:
            continue
        selected.append(row)
    if not selected:
        raise ValueError("No inventory rows matched the requested statuses/datasets")
    return selected


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def load_csv_series(row: dict[str, str], missing: str, train_fraction: float) -> tuple[str, np.ndarray]:
    path = resolve_path(row["file"])
    frame = pd.read_csv(path)
    columns = parse_columns(row.get("selected_columns", ""))
    if not columns:
        time_column = row.get("time_column", "")
        drop_columns = [time_column] if time_column else []
        numeric = frame.drop(columns=drop_columns, errors="ignore").select_dtypes(include=[np.number])
    else:
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")

    if missing == "interpolate":
        # Interpolate within the chronological train/test segments so a train
        # gap never borrows test-period values (and vice versa). The boundary
        # mirrors chronological_lagged_split: int(round(n * train_fraction)).
        split_at = int(round(len(numeric) * train_fraction))
        numeric = interpolate_within_segments(numeric, [split_at]).dropna(axis=0)
    else:
        numeric = numeric.dropna(axis=0)

    values = numeric.to_numpy(dtype=float)
    values = values[np.isfinite(values).all(axis=1)]
    keep = values.std(axis=0) > 1e-10
    values = values[:, keep]
    if values.shape[1] < 2:
        raise ValueError(f"{row['dataset']} has fewer than two non-constant numeric columns")
    return row["dataset"], values


def stcg_config(args: argparse.Namespace, seed: int) -> STCGStateConfig:
    return STCGStateConfig(
        base_model="lasso",
        lasso_alpha=args.lasso_alpha,
        window_size=args.window_size,
        stride=args.window_stride,
        aggregate=args.window_aggregate,
        state_aggregate=args.state_aggregate,
        contrast_mix=0.0,
        seed=seed + 70_000,
    )


def metric_row(
    dataset: str,
    seed: int,
    method: str,
    prediction: np.ndarray,
    target: np.ndarray,
    ridge_alpha: float,
    graph_min_weight: float | None = None,
    graph_weight_power: float | None = None,
    graph_self_weight: float | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "ridge_alpha": ridge_alpha,
        "graph_min_weight": graph_min_weight,
        "graph_weight_power": graph_weight_power,
        "graph_self_weight": graph_self_weight,
    }
    row.update(forecast_metrics(target, prediction))
    return row


def evaluate_dataset_seed(args: argparse.Namespace, dataset: str, x: np.ndarray, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    split = chronological_lagged_split(x, max_lag=args.max_lag, train_fraction=args.train_fraction)
    target = split.target_test
    x_train = x[: split.train_end]

    rows.append(
        metric_row(
            dataset,
            seed,
            "Persistence",
            persistence_forecast(split),
            target,
            ridge_alpha=math.nan,
        )
    )

    dyn_scores = dynamic_lasso_var_scores(
        x_train,
        max_lag=args.max_lag,
        alpha=args.lasso_alpha,
        window_size=args.window_size,
        stride=args.window_stride,
        aggregate=args.window_aggregate,
    )
    stcg_scores = stcg_v1_diagnostics(
        x_train,
        max_lag=args.max_lag,
        config=stcg_config(args, seed),
    ).scores

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
                dataset,
                seed,
                "Dense Ridge",
                ridge_forecast(split, alpha=ridge_alpha),
                target,
                ridge_alpha=ridge_alpha,
            )
        )
        for graph_min_weight, graph_weight_power, dyn_weights, stcg_weights in weight_configs:
            rows.append(
                metric_row(
                    dataset,
                    seed,
                    "DynVAR-Lasso Weighted Ridge",
                    ridge_forecast_weighted(split, alpha=ridge_alpha, feature_weights=dyn_weights),
                    target,
                    ridge_alpha=ridge_alpha,
                    graph_min_weight=graph_min_weight,
                    graph_weight_power=graph_weight_power,
                    graph_self_weight=args.graph_self_weight,
                )
            )
            rows.append(
                metric_row(
                    dataset,
                    seed,
                    "STCG-v1 Weighted Ridge",
                    ridge_forecast_weighted(split, alpha=ridge_alpha, feature_weights=stcg_weights),
                    target,
                    ridge_alpha=ridge_alpha,
                    graph_min_weight=graph_min_weight,
                    graph_weight_power=graph_weight_power,
                    graph_self_weight=args.graph_self_weight,
                )
            )
    return rows


def group_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["dataset"],
        row["method"],
        row["ridge_alpha"],
        row["graph_min_weight"],
        row["graph_weight_power"],
        row["graph_self_weight"],
    )


def sort_value(value: object) -> float:
    if value in {None, ""}:
        return math.inf
    number = float(value)
    return number if math.isfinite(number) else math.inf


def aggregate(rows: list[dict[str, object]], meaningful_rmse_delta: float) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    summary_rows: list[dict[str, object]] = []
    for key, items in sorted(
        groups.items(),
        key=lambda item: (
            str(item[0][0]),
            METHOD_ORDER.get(str(item[0][1]), 99),
            sort_value(item[0][2]),
            sort_value(item[0][3]),
            sort_value(item[0][4]),
        ),
    ):
        dataset, method, ridge_alpha, graph_min_weight, graph_weight_power, graph_self_weight = key
        row: dict[str, object] = {
            "dataset": dataset,
            "method": method,
            "n": len(items),
            "ridge_alpha": ridge_alpha,
            "graph_min_weight": graph_min_weight,
            "graph_weight_power": graph_weight_power,
            "graph_self_weight": graph_self_weight,
        }
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)

    persistence: dict[str, float] = {}
    dense_by_alpha: dict[tuple[str, float], float] = {}
    best_dense: dict[str, float] = {}
    for row in summary_rows:
        dataset = str(row["dataset"])
        method = str(row["method"])
        rmse = float(row["rmse_mean"])
        if method == "Persistence":
            persistence[dataset] = rmse
        elif method == "Dense Ridge":
            ridge_alpha = float(row["ridge_alpha"])
            dense_by_alpha[(dataset, ridge_alpha)] = rmse
            best_dense[dataset] = min(rmse, best_dense.get(dataset, math.inf))

    for row in summary_rows:
        dataset = str(row["dataset"])
        ridge_alpha = row["ridge_alpha"]
        same_dense = dense_by_alpha.get((dataset, float(ridge_alpha)), math.nan) if ridge_alpha is not None else math.nan
        best_dense_rmse = best_dense.get(dataset, math.nan)
        persistence_rmse = persistence.get(dataset, math.nan)
        rmse = float(row["rmse_mean"])
        row["delta_vs_same_alpha_dense_rmse"] = rmse - same_dense
        row["delta_vs_best_dense_rmse"] = rmse - best_dense_rmse
        row["delta_vs_persistence_rmse"] = rmse - persistence_rmse
        row["beats_same_alpha_dense"] = bool(math.isfinite(same_dense) and rmse < same_dense)
        row["beats_best_dense"] = bool(math.isfinite(best_dense_rmse) and rmse < best_dense_rmse)
        row["beats_persistence"] = bool(math.isfinite(persistence_rmse) and rmse < persistence_rmse)
        row["meaningfully_beats_same_alpha_dense"] = bool(
            math.isfinite(same_dense) and rmse < same_dense - meaningful_rmse_delta
        )
        row["meaningfully_beats_best_dense"] = bool(
            math.isfinite(best_dense_rmse) and rmse < best_dense_rmse - meaningful_rmse_delta
        )
        row["meaningfully_beats_persistence"] = bool(
            math.isfinite(persistence_rmse) and rmse < persistence_rmse - meaningful_rmse_delta
        )
    return summary_rows


def best_rows(rows: list[dict[str, object]], method: str) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row["method"]) == method:
            groups[str(row["dataset"])].append(row)
    return [min(items, key=lambda item: float(item["rmse_mean"])) for _, items in sorted(groups.items())]


def write_markdown(path: Path, rows: list[dict[str, object]], meaningful_rmse_delta: float) -> None:
    lines = [
        "# Real Weighted-Prior Sweep Summary",
        "",
        "Lower RMSE is better. Weighted ridge keeps all lagged features and uses graph scores only as target-specific diagonal ridge-prior weights.",
        f"Improvements smaller than `{meaningful_rmse_delta:.3f}` RMSE are treated as numerical-scale rather than meaningful gains.",
        "",
        "## Best Dense Ridge Baseline",
        "",
        "| Dataset | Persistence RMSE | Best Dense RMSE | Dense Alpha |",
        "|---|---:|---:|---:|",
    ]
    persistence = {str(row["dataset"]): row for row in rows if str(row["method"]) == "Persistence"}
    dense_best = {str(row["dataset"]): row for row in best_rows(rows, "Dense Ridge")}
    for dataset in sorted(dense_best):
        dense = dense_best[dataset]
        lines.append(
            "| {dataset} | {persistence} | {dense_rmse} | {alpha} |".format(
                dataset=dataset,
                persistence=fmt(persistence.get(dataset, {}).get("rmse_mean", math.nan), 6),
                dense_rmse=fmt(dense["rmse_mean"], 6),
                alpha=fmt(dense["ridge_alpha"], 3),
            )
        )

    lines.extend(
        [
            "",
            "## Best Weighted Priors",
            "",
            "| Dataset | Method | RMSE | Alpha | Min Weight | Power | Delta vs Same Dense | Delta vs Best Dense | Raw Beats Best | Meaningful Beat |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    weighted_methods = ["DynVAR-Lasso Weighted Ridge", "STCG-v1 Weighted Ridge"]
    best_weighted: list[dict[str, object]] = []
    for method in weighted_methods:
        best_weighted.extend(best_rows(rows, method))
    best_weighted.sort(key=lambda row: (str(row["dataset"]), METHOD_ORDER.get(str(row["method"]), 99)))
    for row in best_weighted:
        lines.append(
            "| {dataset} | {method} | {rmse} | {alpha} | {min_weight} | {power} | {same_delta} | {best_delta} | {raw_beats_best} | {meaningful_beat} |".format(
                dataset=row["dataset"],
                method=row["method"],
                rmse=fmt(row["rmse_mean"], 6),
                alpha=fmt(row["ridge_alpha"], 3),
                min_weight=fmt(row["graph_min_weight"], 2),
                power=fmt(row["graph_weight_power"], 2),
                same_delta=fmt(row["delta_vs_same_alpha_dense_rmse"], 6),
                best_delta=fmt(row["delta_vs_best_dense_rmse"], 6),
                raw_beats_best="yes" if row["beats_best_dense"] else "no",
                meaningful_beat="yes" if row["meaningfully_beats_best_dense"] else "no",
            )
        )

    stcg_best = [row for row in best_weighted if str(row["method"]) == "STCG-v1 Weighted Ridge"]
    any_stcg_raw_beats_best = any(bool(row["beats_best_dense"]) for row in stcg_best)
    any_stcg_meaningfully_beats_best = any(bool(row["meaningfully_beats_best_dense"]) for row in stcg_best)
    any_stcg_meaningfully_beats_same = any(bool(row["meaningfully_beats_same_alpha_dense"]) for row in stcg_best)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Best STCG weighted priors raw-beat best tuned Dense Ridge: {'yes' if any_stcg_raw_beats_best else 'no'}.",
            f"- Best STCG weighted priors meaningfully beat best tuned Dense Ridge: {'yes' if any_stcg_meaningfully_beats_best else 'no'}.",
            f"- Best STCG weighted priors meaningfully beat same-alpha Dense Ridge: {'yes' if any_stcg_meaningfully_beats_same else 'no'}.",
        ]
    )
    if not any_stcg_meaningfully_beats_best:
        lines.append("- Current soft graph priors do not establish a real-data forecasting gain over tuned dense ridge.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows = eligible_inventory_rows(args)

    detail_rows: list[dict[str, object]] = []
    for inventory_row in inventory_rows:
        dataset, x = load_csv_series(inventory_row, missing=args.missing, train_fraction=args.train_fraction)
        for seed in args.seeds:
            detail_rows.extend(evaluate_dataset_seed(args, dataset=dataset, x=x, seed=seed))

    summary_rows = aggregate(detail_rows, meaningful_rmse_delta=args.meaningful_rmse_delta)
    detail_path = args.output_dir / "real_weight_sweep_detail.csv"
    summary_path = args.output_dir / "real_weight_sweep_summary.csv"
    markdown_path = args.output_dir / "real_weight_sweep_summary.md"

    detail_fields = [
        "dataset",
        "seed",
        "method",
        "ridge_alpha",
        "graph_min_weight",
        "graph_weight_power",
        "graph_self_weight",
        *METRICS,
    ]
    summary_fields = [
        "dataset",
        "method",
        "n",
        "ridge_alpha",
        "graph_min_weight",
        "graph_weight_power",
        "graph_self_weight",
        *[f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")],
        "delta_vs_same_alpha_dense_rmse",
        "delta_vs_best_dense_rmse",
        "delta_vs_persistence_rmse",
        "beats_same_alpha_dense",
        "beats_best_dense",
        "beats_persistence",
        "meaningfully_beats_same_alpha_dense",
        "meaningfully_beats_best_dense",
        "meaningfully_beats_persistence",
    ]
    write_csv(detail_path, detail_rows, detail_fields)
    write_csv(summary_path, summary_rows, summary_fields)
    write_markdown(markdown_path, summary_rows, meaningful_rmse_delta=args.meaningful_rmse_delta)

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


if __name__ == "__main__":
    main()
