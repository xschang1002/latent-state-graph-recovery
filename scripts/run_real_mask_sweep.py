"""Sweep hard graph-mask settings on real benchmark CSV datasets."""

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

from _common import fmt_na as fmt, interpolate_within_segments, parse_columns, write_csv
from stcg import (
    STCGStateConfig,
    chronological_lagged_split,
    dynamic_lasso_var_scores,
    forecast_metrics,
    graph_feature_mask,
    persistence_forecast,
    ridge_forecast,
    stcg_v1_diagnostics,
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)


METRICS = ["mae", "rmse", "node_rmse_mean", "node_rmse_max"]
METHOD_ORDER = {
    "Persistence": 0,
    "Dense Ridge": 1,
    "Nonself Dense Ridge": 2,
    "DynVAR-Lasso Graph Ridge": 3,
    "STCG-v1 Graph Ridge": 4,
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--graph-fractions", nargs="+", type=float, default=[0.25, 0.50, 0.75, 1.00])
    parser.add_argument("--lasso-alphas", nargs="+", type=float, default=[0.001, 0.003, 0.005, 0.01])
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[300, 600, 1200])
    parser.add_argument("--window-aggregate", choices=["max", "mean", "p90"], default="max")
    parser.add_argument("--state-aggregate", choices=["mean", "median", "p75", "max"], default="mean")
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


def load_csv_series(row: dict[str, str], missing: str, train_fraction: float) -> tuple[str, np.ndarray, list[str]]:
    path = resolve_path(row["file"])
    frame = pd.read_csv(path)
    columns = parse_columns(row.get("selected_columns", ""))
    if not columns:
        time_column = row.get("time_column", "")
        drop_columns = [time_column] if time_column else []
        numeric = frame.drop(columns=drop_columns, errors="ignore").select_dtypes(include=[np.number])
        columns = [str(column) for column in numeric.columns]
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
    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    std = values.std(axis=0)
    keep = std > 1e-10
    values = values[:, keep]
    kept_columns = [column for column, keep_column in zip(columns, keep) if keep_column]
    if values.shape[1] < 2:
        raise ValueError(f"{row['dataset']} has fewer than two non-constant numeric columns")
    return row["dataset"], values, kept_columns


def nonself_mask(n_nodes: int, max_lag: int) -> np.ndarray:
    mask = np.ones((max_lag * n_nodes, n_nodes), dtype=bool)
    for target in range(n_nodes):
        for lag in range(max_lag):
            mask[lag * n_nodes + target, target] = False
    return mask


def stcg_config(
    lasso_alpha: float,
    window_size: int,
    window_stride: int,
    window_aggregate: str,
    state_aggregate: str,
    seed: int,
) -> STCGStateConfig:
    return STCGStateConfig(
        base_model="lasso",
        lasso_alpha=lasso_alpha,
        window_size=window_size,
        stride=window_stride,
        aggregate=window_aggregate,
        state_aggregate=state_aggregate,
        contrast_mix=0.0,
        seed=seed + 70_000,
    )


def metric_row(
    dataset: str,
    seed: int,
    method: str,
    prediction: np.ndarray,
    target: np.ndarray,
    graph_fraction: float,
    lasso_alpha: float,
    window_size: int,
    window_stride: int,
    feature_budget: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "graph_fraction": graph_fraction,
        "lasso_alpha": lasso_alpha,
        "window_size": window_size,
        "window_stride": window_stride,
        "feature_budget_per_target": feature_budget,
    }
    row.update(forecast_metrics(target, prediction))
    return row


def prediction_rows(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    split,
    predictions: dict[str, np.ndarray],
    graph_fraction: float,
    lasso_alpha: float,
    window_size: int,
    window_stride: int,
    feature_budget: int,
    nonself_budget: int,
) -> list[dict[str, object]]:
    target = split.target_test
    return [
        metric_row(
            dataset,
            seed,
            "Persistence",
            predictions["Persistence"],
            target,
            graph_fraction,
            lasso_alpha,
            window_size,
            window_stride,
            0,
        ),
        metric_row(
            dataset,
            seed,
            "Dense Ridge",
            predictions["Dense Ridge"],
            target,
            graph_fraction,
            lasso_alpha,
            window_size,
            window_stride,
            0,
        ),
        metric_row(
            dataset,
            seed,
            "Nonself Dense Ridge",
            predictions["Nonself Dense Ridge"],
            target,
            graph_fraction,
            lasso_alpha,
            window_size,
            window_stride,
            nonself_budget,
        ),
        metric_row(
            dataset,
            seed,
            "DynVAR-Lasso Graph Ridge",
            predictions["DynVAR-Lasso Graph Ridge"],
            target,
            graph_fraction,
            lasso_alpha,
            window_size,
            window_stride,
            feature_budget,
        ),
        metric_row(
            dataset,
            seed,
            "STCG-v1 Graph Ridge",
            predictions["STCG-v1 Graph Ridge"],
            target,
            graph_fraction,
            lasso_alpha,
            window_size,
            window_stride,
            feature_budget,
        ),
    ]


def evaluate_dataset_seed(args: argparse.Namespace, dataset: str, x: np.ndarray, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    split = chronological_lagged_split(x, max_lag=args.max_lag, train_fraction=args.train_fraction)
    x_train = x[: split.train_end]
    n_nodes = x.shape[1]
    nonself_budget = args.max_lag * (n_nodes - 1)

    fixed_predictions = {
        "Persistence": persistence_forecast(split),
        "Dense Ridge": ridge_forecast(split, alpha=args.ridge_alpha),
        "Nonself Dense Ridge": ridge_forecast(
            split,
            alpha=args.ridge_alpha,
            feature_mask=nonself_mask(n_nodes=n_nodes, max_lag=args.max_lag),
        ),
    }

    for lasso_alpha in args.lasso_alphas:
        for window_size in args.window_sizes:
            window_stride = max(1, window_size // 2)
            dyn_scores = dynamic_lasso_var_scores(
                x_train,
                max_lag=args.max_lag,
                alpha=lasso_alpha,
                window_size=window_size,
                stride=window_stride,
                aggregate=args.window_aggregate,
            )
            stcg_scores = stcg_v1_diagnostics(
                x_train,
                max_lag=args.max_lag,
                config=stcg_config(
                    lasso_alpha=lasso_alpha,
                    window_size=window_size,
                    window_stride=window_stride,
                    window_aggregate=args.window_aggregate,
                    state_aggregate=args.state_aggregate,
                    seed=seed,
                ),
            ).scores
            for graph_fraction in args.graph_fractions:
                feature_budget = int(math.ceil(graph_fraction * nonself_budget))
                predictions = dict(fixed_predictions)
                predictions["DynVAR-Lasso Graph Ridge"] = ridge_forecast(
                    split,
                    alpha=args.ridge_alpha,
                    feature_mask=graph_feature_mask(dyn_scores, fraction=graph_fraction),
                )
                predictions["STCG-v1 Graph Ridge"] = ridge_forecast(
                    split,
                    alpha=args.ridge_alpha,
                    feature_mask=graph_feature_mask(stcg_scores, fraction=graph_fraction),
                )
                rows.extend(
                    prediction_rows(
                        args,
                        dataset=dataset,
                        seed=seed,
                        split=split,
                        predictions=predictions,
                        graph_fraction=graph_fraction,
                        lasso_alpha=lasso_alpha,
                        window_size=window_size,
                        window_stride=window_stride,
                        feature_budget=feature_budget,
                        nonself_budget=nonself_budget,
                    )
                )
    return rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["dataset"],
                row["method"],
                row["graph_fraction"],
                row["lasso_alpha"],
                row["window_size"],
                row["window_stride"],
                row["feature_budget_per_target"],
            )
        ].append(row)

    summary_rows: list[dict[str, object]] = []
    for key, items in sorted(groups.items(), key=lambda item: (item[0][0], METHOD_ORDER.get(str(item[0][1]), 99), *item[0][2:])):
        dataset, method, graph_fraction, lasso_alpha, window_size, window_stride, feature_budget = key
        summary: dict[str, object] = {
            "dataset": dataset,
            "method": method,
            "n": len(items),
            "graph_fraction": graph_fraction,
            "lasso_alpha": lasso_alpha,
            "window_size": window_size,
            "window_stride": window_stride,
            "feature_budget_per_target": feature_budget,
        }
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)

    baseline_lookup: dict[str, dict[str, float]] = {}
    for row in summary_rows:
        dataset = str(row["dataset"])
        method = str(row["method"])
        if method in {"Persistence", "Dense Ridge"}:
            baseline_lookup.setdefault(dataset, {})[method] = float(row["rmse_mean"])

    for row in summary_rows:
        baselines = baseline_lookup.get(str(row["dataset"]), {})
        dense_rmse = baselines.get("Dense Ridge", math.nan)
        persistence_rmse = baselines.get("Persistence", math.nan)
        rmse = float(row["rmse_mean"])
        row["delta_vs_dense_rmse"] = rmse - dense_rmse
        row["delta_vs_persistence_rmse"] = rmse - persistence_rmse
        row["beats_dense"] = bool(rmse < dense_rmse)
        row["beats_persistence"] = bool(rmse < persistence_rmse)
    return summary_rows


def best_rows(rows: list[dict[str, object]], method: str) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row["method"]) == method:
            groups[str(row["dataset"])].append(row)
    return [min(items, key=lambda row: float(row["rmse_mean"])) for _, items in sorted(groups.items())]


def baseline_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["dataset"]), str(row["method"]))
        if row["method"] in {"Persistence", "Dense Ridge", "Nonself Dense Ridge"} and key not in seen:
            selected.append(row)
            seen.add(key)
    return selected


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Real Hard-Mask Sweep Summary",
        "",
        "Lower RMSE is better. `Nonself Dense Ridge` keeps all non-self lagged features and drops self-lags, matching the feature universe available to graph masks at `graph_fraction=1.0`.",
        "",
        "## Baseline Ceiling",
        "",
        "| Dataset | Method | RMSE | Delta vs Dense | Delta vs Persistence |",
        "|---|---|---:|---:|---:|",
    ]
    for row in baseline_rows(rows):
        lines.append(
            "| {dataset} | {method} | {rmse} | {dense_delta} | {pers_delta} |".format(
                dataset=row["dataset"],
                method=row["method"],
                rmse=fmt(row["rmse_mean"]),
                dense_delta=fmt(row["delta_vs_dense_rmse"]),
                pers_delta=fmt(row["delta_vs_persistence_rmse"]),
            )
        )

    lines.extend(
        [
            "",
            "## Best Hard Graph Masks",
            "",
            "| Dataset | Method | RMSE | Graph Fraction | Lasso Alpha | Window | Beats Dense | Beats Persistence |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    graph_methods = ["DynVAR-Lasso Graph Ridge", "STCG-v1 Graph Ridge"]
    for method in graph_methods:
        for row in best_rows(rows, method):
            lines.append(
                "| {dataset} | {method} | {rmse} | {fraction} | {alpha} | {window} | {beats_dense} | {beats_persistence} |".format(
                    dataset=row["dataset"],
                    method=row["method"],
                    rmse=fmt(row["rmse_mean"]),
                    fraction=fmt(row["graph_fraction"], 2),
                    alpha=fmt(row["lasso_alpha"], 3),
                    window=row["window_size"],
                    beats_dense="yes" if row["beats_dense"] else "no",
                    beats_persistence="yes" if row["beats_persistence"] else "no",
                )
            )

    stcg_best = best_rows(rows, "STCG-v1 Graph Ridge")
    any_stcg_beats_dense = any(bool(row["beats_dense"]) for row in stcg_best)
    any_stcg_beats_persistence = any(bool(row["beats_persistence"]) for row in stcg_best)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Best STCG hard masks beat Dense Ridge: {'yes' if any_stcg_beats_dense else 'no'}.",
            f"- Best STCG hard masks beat Persistence: {'yes' if any_stcg_beats_persistence else 'no'}.",
        ]
    )
    if not any_stcg_beats_dense:
        lines.append("- Current hard graph masks are not sufficient for ETT-style dense forecasting.")
    if stcg_best and all(float(row["graph_fraction"]) >= 0.99 for row in stcg_best):
        lines.append("- Best STCG settings use `graph_fraction=1.0`, so sparsification itself is not helping.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows = eligible_inventory_rows(args)

    detail_rows: list[dict[str, object]] = []
    for inventory_row in inventory_rows:
        dataset, x, _ = load_csv_series(inventory_row, missing=args.missing, train_fraction=args.train_fraction)
        for seed in args.seeds:
            detail_rows.extend(evaluate_dataset_seed(args, dataset=dataset, x=x, seed=seed))

    summary_rows = aggregate(detail_rows)
    detail_path = args.output_dir / "real_mask_sweep_detail.csv"
    summary_path = args.output_dir / "real_mask_sweep_summary.csv"
    markdown_path = args.output_dir / "real_mask_sweep_summary.md"

    detail_fields = [
        "dataset",
        "seed",
        "method",
        "graph_fraction",
        "lasso_alpha",
        "window_size",
        "window_stride",
        "feature_budget_per_target",
        *METRICS,
    ]
    summary_fields = [
        "dataset",
        "method",
        "n",
        "graph_fraction",
        "lasso_alpha",
        "window_size",
        "window_stride",
        "feature_budget_per_target",
        *[f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")],
        "delta_vs_dense_rmse",
        "delta_vs_persistence_rmse",
        "beats_dense",
        "beats_persistence",
    ]
    write_csv(detail_path, detail_rows, detail_fields)
    write_csv(summary_path, summary_rows, summary_fields)
    write_markdown(markdown_path, summary_rows)

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
