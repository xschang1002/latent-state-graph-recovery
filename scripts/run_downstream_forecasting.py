"""Evaluate learned graph scores through one-step forecasting."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import interpolate_within_segments, parse_columns, warn_if_results_tables_output
from stcg import (
    STCGStateConfig,
    chronological_lagged_split,
    dynamic_lasso_var_scores,
    forecast_metrics,
    generate_lagged_var,
    graph_feature_mask,
    graph_feature_weights,
    persistence_forecast,
    random_scores,
    ridge_forecast,
    ridge_forecast_weighted,
    stcg_v1_diagnostics,
)


METRICS = ["mae", "rmse", "node_rmse_mean", "node_rmse_max"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--columns", type=str, default="")
    parser.add_argument("--time-column", type=str, default="")
    parser.add_argument("--missing", choices=["drop", "interpolate"], default="interpolate")
    parser.add_argument("--dataset-name", type=str, default="")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--edge-prob", type=float, default=0.18)
    parser.add_argument("--noise-scale", type=float, default=0.10)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--lasso-alpha", type=float, default=0.005)
    parser.add_argument("--graph-fraction", type=float, default=0.25)
    parser.add_argument("--graph-min-weight", type=float, default=0.10)
    parser.add_argument("--graph-self-weight", type=float, default=1.0)
    parser.add_argument("--graph-weight-power", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--window-stride", type=int, default=150)
    parser.add_argument("--window-aggregate", choices=["max", "mean", "p90"], default="max")
    parser.add_argument("--state-aggregate", choices=["mean", "median", "p75", "max"], default="mean")
    parser.add_argument("--stcg-v1-states", type=int, default=2)
    parser.add_argument("--stcg-v1-min-improvement", type=float, default=0.12)
    parser.add_argument("--stcg-v1-force-states", action="store_true")
    parser.add_argument("--diagnostic-top-k", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "results" / "figures")
    return parser.parse_args()


def load_csv_series(args: argparse.Namespace) -> tuple[str, np.ndarray, list[str]]:
    if args.input_csv is None:
        raise ValueError("input_csv is required for CSV loading")
    frame = pd.read_csv(args.input_csv)
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
        raise ValueError("Need at least two numeric columns for graph forecasting")
    if args.missing == "interpolate":
        # Interpolate within the chronological train/test segments so a train
        # gap never borrows test-period values (and vice versa). The boundary
        # mirrors chronological_lagged_split: int(round(n * train_fraction)).
        split_at = int(round(len(numeric) * args.train_fraction))
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
        raise ValueError("Need at least two non-constant numeric columns after cleaning")
    name = args.dataset_name or args.input_csv.stem
    return name, values, kept_columns


def synthetic_series(args: argparse.Namespace, seed: int) -> tuple[str, np.ndarray, list[str]]:
    dataset = generate_lagged_var(
        n_nodes=args.n_nodes,
        timesteps=args.timesteps,
        max_lag=args.max_lag,
        edge_prob=args.edge_prob,
        noise_scale=args.noise_scale,
        seed=seed,
        switching=True,
    )
    columns = [f"x{idx}" for idx in range(args.n_nodes)]
    return "synthetic_switching", dataset.x, columns


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


def top_edge_rows(
    dataset_name: str,
    seed: int,
    scores: np.ndarray,
    columns: list[str],
    top_k: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    max_lag, n_sources, n_targets = scores.shape
    candidates: list[tuple[float, int, int, int]] = []
    for lag_idx in range(max_lag):
        for source in range(n_sources):
            for target in range(n_targets):
                if source == target:
                    continue
                candidates.append((float(scores[lag_idx, source, target]), lag_idx + 1, source, target))
    candidates.sort(reverse=True, key=lambda item: item[0])
    for rank, (score, lag, source, target) in enumerate(candidates[:top_k], start=1):
        rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "rank": rank,
                "lag": lag,
                "source": columns[source],
                "target": columns[target],
                "source_index": source,
                "target_index": target,
                "score": score,
            }
        )
    return rows


def state_rows(dataset_name: str, seed: int, diagnostics) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for window_id, ((start, end), state) in enumerate(zip(diagnostics.bounds, diagnostics.labels)):
        rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "window_id": window_id,
                "start": start,
                "end": end,
                "center": (start + end) / 2,
                "inferred_state": int(state),
            }
        )
    return rows


def evaluate_one(
    args: argparse.Namespace,
    dataset_name: str,
    x: np.ndarray,
    columns: list[str],
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    split = chronological_lagged_split(x, max_lag=args.max_lag, train_fraction=args.train_fraction)
    x_train = x[: split.train_end]

    stcg_diagnostics = stcg_v1_diagnostics(x_train, max_lag=args.max_lag, config=stcg_config(args, seed))
    stcg_scores = stcg_diagnostics.scores
    dyn_scores = dynamic_lasso_var_scores(
        x_train,
        max_lag=args.max_lag,
        alpha=args.lasso_alpha,
        window_size=args.window_size,
        stride=args.window_stride,
        aggregate=args.window_aggregate,
    )
    rand_scores = random_scores(n_nodes=x.shape[1], max_lag=args.max_lag, seed=seed + 90_000)
    dyn_weights = graph_feature_weights(
        dyn_scores,
        min_weight=args.graph_min_weight,
        self_weight=args.graph_self_weight,
        power=args.graph_weight_power,
    )
    stcg_weights = graph_feature_weights(
        stcg_scores,
        min_weight=args.graph_min_weight,
        self_weight=args.graph_self_weight,
        power=args.graph_weight_power,
    )

    predictions = {
        "Persistence": persistence_forecast(split),
        "Dense Ridge": ridge_forecast(split, alpha=args.ridge_alpha),
        "RandomGraph Ridge": ridge_forecast(
            split,
            alpha=args.ridge_alpha,
            feature_mask=graph_feature_mask(rand_scores, fraction=args.graph_fraction),
        ),
        "DynVAR-Lasso Graph Ridge": ridge_forecast(
            split,
            alpha=args.ridge_alpha,
            feature_mask=graph_feature_mask(dyn_scores, fraction=args.graph_fraction),
        ),
        "DynVAR-Lasso Weighted Ridge": ridge_forecast_weighted(
            split,
            alpha=args.ridge_alpha,
            feature_weights=dyn_weights,
        ),
        "STCG-v1 Graph Ridge": ridge_forecast(
            split,
            alpha=args.ridge_alpha,
            feature_mask=graph_feature_mask(stcg_scores, fraction=args.graph_fraction),
        ),
        "STCG-v1 Weighted Ridge": ridge_forecast_weighted(
            split,
            alpha=args.ridge_alpha,
            feature_weights=stcg_weights,
        ),
    }

    rows: list[dict[str, object]] = []
    feature_budget = int(math.ceil(args.graph_fraction * args.max_lag * (x.shape[1] - 1)))
    for method, prediction in predictions.items():
        row: dict[str, object] = {
            "dataset": dataset_name,
            "seed": seed,
            "method": method,
            "timesteps": x.shape[0],
            "n_nodes": x.shape[1],
            "train_end": split.train_end,
            "test_rows": split.target_test.shape[0],
            "max_lag": args.max_lag,
            "graph_fraction": args.graph_fraction,
            "graph_min_weight": args.graph_min_weight,
            "graph_self_weight": args.graph_self_weight,
            "graph_weight_power": args.graph_weight_power,
            "feature_budget_per_target": (
                args.max_lag * x.shape[1]
                if "Weighted Ridge" in method
                else 0
                if method in {"Persistence", "Dense Ridge"}
                else feature_budget
            ),
        }
        row.update(forecast_metrics(split.target_test, prediction))
        rows.append(row)
    return (
        rows,
        top_edge_rows(dataset_name, seed, stcg_scores, columns, args.diagnostic_top_k),
        state_rows(dataset_name, seed, stcg_diagnostics),
    )


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["method"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (dataset, method), items in sorted(groups.items()):
        summary: dict[str, object] = {"dataset": dataset, "method": method, "n": len(items)}
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        summary_rows.append(summary)
    return summary_rows


def write_detail(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "dataset",
        "seed",
        "method",
        "timesteps",
        "n_nodes",
        "train_end",
        "test_rows",
        "max_lag",
        "graph_fraction",
        "graph_min_weight",
        "graph_self_weight",
        "graph_weight_power",
        "feature_budget_per_target",
        *METRICS,
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["dataset", "method", "n"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Downstream Forecasting Summary",
        "",
        "| Dataset | Method | RMSE | MAE | Mean Node RMSE | Max Node RMSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {method} | {rmse_mean:.3f} +/- {rmse_std:.3f} | "
            "{mae_mean:.3f} +/- {mae_std:.3f} | "
            "{node_rmse_mean_mean:.3f} +/- {node_rmse_mean_std:.3f} | "
            "{node_rmse_max_mean:.3f} +/- {node_rmse_max_std:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, object]]) -> None:
    methods = [str(row["method"]) for row in rows]
    means = [float(row["rmse_mean"]) for row in rows]
    stds = [float(row["rmse_std"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars = ax.barh(methods, means, xerr=stds, capsize=3, color="#4c78a8")
    ax.set_xlabel("Standardized RMSE (lower is better)")
    ax.grid(axis="x", alpha=0.25)
    best = min(means)
    for bar, mean in zip(bars, means):
        ax.text(mean + max(means) * 0.015, bar.get_y() + bar.get_height() / 2, f"{mean:.3f}", va="center")
        bar.set_alpha(1.0 if mean == best else 0.72)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_edge_diagnostics(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "dataset",
        "seed",
        "rank",
        "lag",
        "source",
        "target",
        "source_index",
        "target_index",
        "score",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_state_diagnostics(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["dataset", "seed", "window_id", "start", "end", "center", "inferred_state"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_graph_markdown(
    path: Path,
    edge_rows: list[dict[str, object]],
    state_diag_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Downstream Graph Diagnostics",
        "",
        "## Top STCG-v1 Edges",
        "",
        "| Dataset | Seed | Rank | Lag | Source | Target | Score |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    edge_seed = min((int(item["seed"]) for item in edge_rows), default=None)
    for row in edge_rows:
        if int(row["seed"]) != edge_seed:
            continue
        lines.append(
            "| {dataset} | {seed} | {rank} | {lag} | {source} | {target} | {score:.4f} |".format(**row)
        )
    if edge_seed is None:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Inferred State Windows",
            "",
            "| Dataset | Seed | Window | Start | End | State |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    state_seed = min((int(item["seed"]) for item in state_diag_rows), default=None)
    for row in state_diag_rows:
        if int(row["seed"]) != state_seed:
            continue
        lines.append(
            "| {dataset} | {seed} | {window_id} | {start} | {end} | {inferred_state} |".format(**row)
        )
    if state_seed is None:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    warn_if_results_tables_output(
        "run_downstream_forecasting.py", args.output_dir, len(args.seeds), canonical_seeds=5
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    state_diag_rows: list[dict[str, object]] = []
    if args.input_csv is not None:
        dataset_name, x, columns = load_csv_series(args)
        for seed in args.seeds:
            eval_rows, eval_edges, eval_states = evaluate_one(args, dataset_name, x, columns, seed)
            rows.extend(eval_rows)
            edge_rows.extend(eval_edges)
            state_diag_rows.extend(eval_states)
    else:
        for seed in args.seeds:
            dataset_name, x, columns = synthetic_series(args, seed)
            eval_rows, eval_edges, eval_states = evaluate_one(args, dataset_name, x, columns, seed)
            rows.extend(eval_rows)
            edge_rows.extend(eval_edges)
            state_diag_rows.extend(eval_states)

    summary_rows = aggregate(rows)
    detail_path = args.output_dir / "downstream_forecasting_detail.csv"
    summary_path = args.output_dir / "downstream_forecasting_summary.csv"
    markdown_path = args.output_dir / "downstream_forecasting_summary.md"
    edge_path = args.output_dir / "downstream_graph_top_edges.csv"
    state_path = args.output_dir / "downstream_graph_states.csv"
    graph_markdown_path = args.output_dir / "downstream_graph_diagnostics.md"
    figure_path = args.figure_dir / "downstream_forecasting_rmse.png"

    write_detail(detail_path, rows)
    write_summary(summary_path, summary_rows)
    write_markdown(markdown_path, summary_rows)
    write_edge_diagnostics(edge_path, edge_rows)
    write_state_diagnostics(state_path, state_diag_rows)
    write_graph_markdown(graph_markdown_path, edge_rows, state_diag_rows)
    write_figure(figure_path, summary_rows)

    best = min(summary_rows, key=lambda row: float(row["rmse_mean"]))
    print(
        json.dumps(
            {
                "detail_csv": str(detail_path),
                "summary_csv": str(summary_path),
                "summary_md": str(markdown_path),
                "graph_edges_csv": str(edge_path),
                "graph_states_csv": str(state_path),
                "graph_diagnostics_md": str(graph_markdown_path),
                "figure": str(figure_path),
                "n_rows": len(rows),
                "n_summary_rows": len(summary_rows),
                "best_rmse_method": best["method"],
                "best_rmse": best["rmse_mean"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
