"""Summarize executed real benchmark runs into tracked result tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _common import fmt_na as fmt, write_csv  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


METHOD_ORDER = [
    "Persistence",
    "Dense Ridge",
    "RandomGraph Ridge",
    "DynVAR-Lasso Graph Ridge",
    "DynVAR-Lasso Weighted Ridge",
    "STCG-v1 Graph Ridge",
    "STCG-v1 Weighted Ridge",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=ROOT / "experiments" / "logs" / "real_benchmarks",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def benchmark_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def method_rank(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


def collect_forecasting(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_dir in benchmark_dirs(root):
        summary_path = dataset_dir / "tables" / "downstream_forecasting_summary.csv"
        if not summary_path.exists():
            continue
        for row in read_csv(summary_path):
            row_out: dict[str, object] = dict(row)
            for key in [
                "n",
                "mae_mean",
                "mae_std",
                "rmse_mean",
                "rmse_std",
                "node_rmse_mean_mean",
                "node_rmse_mean_std",
                "node_rmse_max_mean",
                "node_rmse_max_std",
            ]:
                row_out[key] = float(row[key]) if key != "n" else int(row[key])
            rows.append(row_out)
    rows.sort(key=lambda row: (str(row["dataset"]), method_rank(str(row["method"]))))
    return rows


def collect_top_edges(root: Path, top_k: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_dir in benchmark_dirs(root):
        edge_path = dataset_dir / "tables" / "downstream_graph_top_edges.csv"
        if not edge_path.exists():
            continue
        edge_rows = read_csv(edge_path)
        first_seed = min((int(row["seed"]) for row in edge_rows), default=None)
        for row in edge_rows:
            if first_seed is None or int(row["seed"]) != first_seed or int(row["rank"]) > top_k:
                continue
            rows.append(
                {
                    "dataset": row["dataset"],
                    "seed": int(row["seed"]),
                    "rank": int(row["rank"]),
                    "lag": int(row["lag"]),
                    "source": row["source"],
                    "target": row["target"],
                    "score": float(row["score"]),
                }
            )
    rows.sort(key=lambda row: (str(row["dataset"]), int(row["rank"])))
    return rows


def collect_state_summary(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_dir in benchmark_dirs(root):
        state_path = dataset_dir / "tables" / "downstream_graph_states.csv"
        if not state_path.exists():
            continue
        groups: dict[tuple[str, int], list[int]] = defaultdict(list)
        for row in read_csv(state_path):
            groups[(row["dataset"], int(row["seed"]))].append(int(row["inferred_state"]))
        for (dataset, seed), states in sorted(groups.items()):
            counts = Counter(states)
            n_windows = len(states)
            dominant = max(counts.values()) if counts else 0
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "n_windows": n_windows,
                    "n_states": len(counts),
                    "dominant_state_fraction": dominant / n_windows if n_windows else math.nan,
                    "state_counts": ";".join(f"{state}:{count}" for state, count in sorted(counts.items())),
                }
            )
    return rows


def write_forecasting_md(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Real Benchmark Forecasting Summary",
        "",
        "| Dataset | Method | N | RMSE | MAE | Mean Node RMSE | Max Node RMSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {method} | {n} | {rmse} +/- {rmse_std} | {mae} +/- {mae_std} | "
            "{node_mean} +/- {node_mean_std} | {node_max} +/- {node_max_std} |".format(
                dataset=row["dataset"],
                method=row["method"],
                n=row["n"],
                rmse=fmt(row["rmse_mean"]),
                rmse_std=fmt(row["rmse_std"]),
                mae=fmt(row["mae_mean"]),
                mae_std=fmt(row["mae_std"]),
                node_mean=fmt(row["node_rmse_mean_mean"]),
                node_mean_std=fmt(row["node_rmse_mean_std"]),
                node_max=fmt(row["node_rmse_max_mean"]),
                node_max_std=fmt(row["node_rmse_max_std"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_forecasting_tex(path: Path, rows: list[dict[str, object]]) -> None:
    by_dataset: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_dataset[str(row["dataset"])][str(row["method"])] = row
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Dataset & Persistence & Dense & DynVAR Mask & DynVAR Weight & STCG Mask & STCG Weight \\\\",
        "\\midrule",
    ]
    for dataset, methods in sorted(by_dataset.items()):
        lines.append(
            "{dataset} & {persistence} & {dense} & {dynvar_mask} & {dynvar_weight} & {stcg_mask} & {stcg_weight} \\\\".format(
                dataset=dataset,
                persistence=fmt(methods.get("Persistence", {}).get("rmse_mean", math.nan), 3),
                dense=fmt(methods.get("Dense Ridge", {}).get("rmse_mean", math.nan), 3),
                dynvar_mask=fmt(methods.get("DynVAR-Lasso Graph Ridge", {}).get("rmse_mean", math.nan), 3),
                dynvar_weight=fmt(
                    methods.get("DynVAR-Lasso Weighted Ridge", {}).get("rmse_mean", math.nan), 3
                ),
                stcg_mask=fmt(methods.get("STCG-v1 Graph Ridge", {}).get("rmse_mean", math.nan), 3),
                stcg_weight=fmt(methods.get("STCG-v1 Weighted Ridge", {}).get("rmse_mean", math.nan), 3),
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_edges_md(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Real Benchmark Top STCG-v1 Edges",
        "",
        "| Dataset | Rank | Lag | Source | Target | Score |",
        "|---|---:|---:|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {rank} | {lag} | {source} | {target} | {score} |".format(
                dataset=row["dataset"],
                rank=row["rank"],
                lag=row["lag"],
                source=row["source"],
                target=row["target"],
                score=fmt(row["score"], 4),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_state_md(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Real Benchmark State Summary",
        "",
        "| Dataset | Seed | Windows | States | Dominant Fraction | Counts |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {seed} | {n_windows} | {n_states} | {dominant} | {counts} |".format(
                dataset=row["dataset"],
                seed=row["seed"],
                n_windows=row["n_windows"],
                n_states=row["n_states"],
                dominant=fmt(row["dominant_state_fraction"], 3),
                counts=row["state_counts"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    forecasting = collect_forecasting(args.benchmark_root)
    edges = collect_top_edges(args.benchmark_root, args.top_k)
    state_summary = collect_state_summary(args.benchmark_root)

    forecasting_csv = args.output_dir / "real_benchmark_forecasting_summary.csv"
    forecasting_md = args.output_dir / "real_benchmark_forecasting_summary.md"
    forecasting_tex = args.output_dir / "real_benchmark_forecasting_table.tex"
    edges_csv = args.output_dir / "real_benchmark_top_edges.csv"
    edges_md = args.output_dir / "real_benchmark_top_edges.md"
    states_csv = args.output_dir / "real_benchmark_state_summary.csv"
    states_md = args.output_dir / "real_benchmark_state_summary.md"

    write_csv(
        forecasting_csv,
        forecasting,
        [
            "dataset",
            "method",
            "n",
            "mae_mean",
            "mae_std",
            "rmse_mean",
            "rmse_std",
            "node_rmse_mean_mean",
            "node_rmse_mean_std",
            "node_rmse_max_mean",
            "node_rmse_max_std",
        ],
    )
    write_forecasting_md(forecasting_md, forecasting)
    write_forecasting_tex(forecasting_tex, forecasting)
    write_csv(edges_csv, edges, ["dataset", "seed", "rank", "lag", "source", "target", "score"])
    write_edges_md(edges_md, edges)
    write_csv(
        states_csv,
        state_summary,
        ["dataset", "seed", "n_windows", "n_states", "dominant_state_fraction", "state_counts"],
    )
    write_state_md(states_md, state_summary)

    print(
        json.dumps(
            {
                "forecasting_csv": str(forecasting_csv),
                "forecasting_md": str(forecasting_md),
                "forecasting_tex": str(forecasting_tex),
                "top_edges_csv": str(edges_csv),
                "top_edges_md": str(edges_md),
                "state_summary_csv": str(states_csv),
                "state_summary_md": str(states_md),
                "n_forecasting_rows": len(forecasting),
                "n_edge_rows": len(edges),
                "n_state_rows": len(state_summary),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
