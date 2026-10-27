"""Run STCG-v1 robustness sweeps on switching synthetic graphs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import replace
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

import _common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stcg import (
    STCGStateConfig,
    dynamic_lasso_var_scores,
    generate_lagged_var,
    lasso_var_scores,
    recovery_summary,
    stcg_v1_scores,
)


METRICS = ["edge_auc", "pair_auc", "precision_at_k", "shd", "lag_accuracy"]
METHODS = ["VAR-Lasso", "DynVAR-Lasso", "STCG-v1"]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--default-timesteps", type=int, default=1000)
    parser.add_argument("--default-noise", type=float, default=0.10)
    parser.add_argument("--default-edge-prob", type=float, default=0.18)
    parser.add_argument("--noise-values", type=parse_float_list, default=parse_float_list("0.00,0.05,0.10,0.20"))
    parser.add_argument("--timesteps-values", type=parse_int_list, default=parse_int_list("300,600,1000,1500"))
    parser.add_argument("--edge-prob-values", type=parse_float_list, default=parse_float_list("0.10,0.18,0.30,0.45"))
    parser.add_argument("--lasso-alpha", type=float, default=0.005)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--window-aggregate", choices=["max", "mean", "p90"], default="max")
    parser.add_argument("--state-aggregate", choices=["mean", "median", "p75", "max"], default="mean")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "results" / "figures")
    return parser.parse_args()


def adaptive_window(timesteps: int, requested: int | None) -> int:
    if requested is not None:
        return requested
    return min(300, max(120, timesteps // 2))


def adaptive_stride(window_size: int, requested: int | None) -> int:
    if requested is not None:
        return requested
    return max(1, window_size // 2)


def settings(args: argparse.Namespace) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for value in args.noise_values:
        items.append(
            {
                "axis": "obs_noise",
                "value": value,
                "timesteps": args.default_timesteps,
                "noise_scale": args.default_noise,
                "observation_noise": value,
                "edge_prob": args.default_edge_prob,
            }
        )
    for value in args.timesteps_values:
        items.append(
            {
                "axis": "timesteps",
                "value": value,
                "timesteps": value,
                "noise_scale": args.default_noise,
                "observation_noise": 0.0,
                "edge_prob": args.default_edge_prob,
            }
        )
    for value in args.edge_prob_values:
        items.append(
            {
                "axis": "edge_prob",
                "value": value,
                "timesteps": args.default_timesteps,
                "noise_scale": args.default_noise,
                "observation_noise": 0.0,
                "edge_prob": value,
            }
        )
    return items


def method_scores(dataset, args: argparse.Namespace, timesteps: int, seed: int) -> dict[str, object]:
    window_size = adaptive_window(timesteps, args.window_size)
    stride = adaptive_stride(window_size, args.window_stride)
    state_config = STCGStateConfig(
        base_model="lasso",
        lasso_alpha=args.lasso_alpha,
        window_size=window_size,
        stride=stride,
        aggregate=args.window_aggregate,
        state_aggregate=args.state_aggregate,
        contrast_mix=0.0,
        seed=seed + 50_000,
    )
    return {
        "VAR-Lasso": lasso_var_scores(dataset.x, max_lag=args.max_lag, alpha=args.lasso_alpha),
        "DynVAR-Lasso": dynamic_lasso_var_scores(
            dataset.x,
            max_lag=args.max_lag,
            alpha=args.lasso_alpha,
            window_size=window_size,
            stride=stride,
            aggregate=args.window_aggregate,
        ),
        "STCG-v1": stcg_v1_scores(dataset.x, max_lag=args.max_lag, config=state_config),
    }


def row_for(setting: dict[str, object], seed: int, method: str, summary: dict[str, float]) -> dict[str, object]:
    row: dict[str, object] = {
        "axis": setting["axis"],
        "value": setting["value"],
        "seed": seed,
        "method": method,
        "timesteps": setting["timesteps"],
        "noise_scale": setting["noise_scale"],
        "edge_prob": setting["edge_prob"],
        "observation_noise": setting["observation_noise"],
    }
    row.update(summary)
    return row


def write_detail(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "axis",
        "value",
        "seed",
        "method",
        "timesteps",
        "noise_scale",
        "observation_noise",
        "edge_prob",
        *METRICS,
        "true_edges",
        "edge_density",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["axis"]), str(row["value"]), str(row["method"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (axis, value, method), items in sorted(groups.items(), key=lambda item: (item[0][0], float(item[0][1]), item[0][2])):
        summary: dict[str, object] = {"axis": axis, "value": value, "method": method, "n": len(items)}
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            mean = sum(values) / len(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = (
                math.sqrt(sum((value_i - mean) ** 2 for value_i in values) / (len(values) - 1))
                if len(values) > 1
                else 0.0
            )
        summary_rows.append(summary)
    return summary_rows


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["axis", "value", "method", "n"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_mean_std(row: dict[str, object], metric: str, digits: int = 3, missing: str = "n/a") -> str:
    mean = float(row[f"{metric}_mean"])
    std = float(row[f"{metric}_std"])
    if not math.isfinite(mean) or not math.isfinite(std):
        return missing
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = ["# STCG-v1 Robustness Summary", ""]
    for axis in ["obs_noise", "timesteps", "edge_prob"]:
        lines.extend(
            [
                f"## {axis}",
                "",
                "| Value | Method | N | Edge AUC | Pair AUC | Precision@K | SHD | Lag Acc |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        axis_rows = [row for row in rows if row["axis"] == axis]
        for row in axis_rows:
            lines.append(
                "| {value} | {method} | {n} | {edge_auc} | {pair_auc} | {precision_at_k} | {shd} | {lag_accuracy} |".format(
                    value=row["value"],
                    method=row["method"],
                    n=row["n"],
                    edge_auc=format_mean_std(row, "edge_auc"),
                    pair_auc=format_mean_std(row, "pair_auc"),
                    precision_at_k=format_mean_std(row, "precision_at_k"),
                    shd=format_mean_std(row, "shd", digits=1),
                    lag_accuracy=format_mean_std(row, "lag_accuracy"),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def format_latex_mean_std(row: dict[str, object], metric: str, digits: int = 3) -> str:
    value = format_mean_std(row, metric, digits=digits, missing="--")
    return value.replace("+/-", "$\\pm$")


def write_latex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "Axis & Value & Method & Edge AUC & P@K & SHD \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            "{axis} & {value} & {method} & {edge_auc} & {precision_at_k} & {shd} \\\\".format(
                axis=row["axis"],
                value=row["value"],
                method=row["method"],
                edge_auc=format_latex_mean_std(row, "edge_auc"),
                precision_at_k=format_latex_mean_std(row, "precision_at_k"),
                shd=format_latex_mean_std(row, "shd", digits=1),
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    axis_names = ["obs_noise", "timesteps", "edge_prob"]
    labels = {"obs_noise": "Observation noise", "timesteps": "Timesteps", "edge_prob": "Edge probability"}
    colors = {"VAR-Lasso": "#8c564b", "DynVAR-Lasso": "#1f77b4", "STCG-v1": "#2ca02c"}

    for ax, axis_name in zip(axes, axis_names):
        axis_rows = [row for row in rows if row["axis"] == axis_name]
        values = sorted({float(row["value"]) for row in axis_rows})
        for method in METHODS:
            method_rows = {
                float(row["value"]): row
                for row in axis_rows
                if row["method"] == method
            }
            means = [float(method_rows[value]["edge_auc_mean"]) for value in values]
            stds = [float(method_rows[value]["edge_auc_std"]) for value in values]
            ax.errorbar(values, means, yerr=stds, marker="o", linewidth=2, capsize=3, label=method, color=colors[method])
        ax.set_title(labels[axis_name])
        ax.set_xlabel(labels[axis_name])
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Lagged Edge AUC")
    axes[-1].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def win_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    wins = {method: 0 for method in METHODS}
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["axis"]), str(row["value"]))].append(row)
    for items in grouped.values():
        best = max(items, key=lambda item: float(item["edge_auc_mean"]))
        wins[str(best["method"])] += 1
    return wins


def add_observation_noise(dataset, ratio: float, seed: int):
    if ratio <= 0:
        return dataset
    rng = np.random.default_rng(seed)
    scale = float(np.std(dataset.x))
    noisy = dataset.x + rng.normal(0.0, ratio * scale, size=dataset.x.shape)
    return replace(dataset, x=noisy)


def main() -> None:
    args = parse_args()
    _common.warn_if_results_tables_output(
        "run_stcg_robustness.py", args.output_dir, len(args.seeds), canonical_seeds=20
    )
    rows: list[dict[str, object]] = []
    for setting in settings(args):
        timesteps = int(setting["timesteps"])
        for seed in args.seeds:
            dataset = generate_lagged_var(
                n_nodes=args.n_nodes,
                timesteps=timesteps,
                max_lag=args.max_lag,
                edge_prob=float(setting["edge_prob"]),
                noise_scale=float(setting["noise_scale"]),
                seed=seed,
                switching=True,
            )
            dataset = add_observation_noise(
                dataset,
                ratio=float(setting["observation_noise"]),
                seed=seed + 60_000,
            )
            for method, scores in method_scores(dataset, args, timesteps, seed).items():
                rows.append(row_for(setting, seed, method, recovery_summary(scores, dataset.edge_truth)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = aggregate(rows)

    detail_path = args.output_dir / "stcg_v1_robustness_detail.csv"
    summary_path = args.output_dir / "stcg_v1_robustness_summary.csv"
    markdown_path = args.output_dir / "stcg_v1_robustness_summary.md"
    latex_path = args.output_dir / "stcg_v1_robustness_table.tex"
    figure_path = args.figure_dir / "stcg_v1_robustness_edge_auc.png"

    write_detail(detail_path, rows)
    write_summary(summary_path, summary_rows)
    write_markdown(markdown_path, summary_rows)
    write_latex(latex_path, summary_rows)
    write_figure(figure_path, summary_rows)

    print(
        json.dumps(
            {
                "detail_csv": str(detail_path),
                "summary_csv": str(summary_path),
                "summary_md": str(markdown_path),
                "table_tex": str(latex_path),
                "figure": str(figure_path),
                "n_rows": len(rows),
                "n_summary_rows": len(summary_rows),
                "edge_auc_wins": win_counts(summary_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
