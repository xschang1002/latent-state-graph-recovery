"""Run focused STCG-v1 ablations on synthetic recovery."""

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

import _common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stcg import (
    STCGStateConfig,
    dynamic_lasso_var_scores,
    generate_lagged_var,
    lasso_var_scores,
    recovery_summary,
    stcg_v1_scores,
)


METRICS = ["edge_auc", "pair_auc", "precision_at_k", "shd", "lag_accuracy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["var", "nonlinear_var", "switching_var"], default="switching_var")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--edge-prob", type=float, default=0.18)
    parser.add_argument("--noise-scale", type=float, default=0.10)
    parser.add_argument("--lasso-alpha", type=float, default=0.005)
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--window-stride", type=int, default=150)
    parser.add_argument("--window-aggregate", choices=["max", "mean", "p90"], default="max")
    parser.add_argument("--state-aggregate", choices=["mean", "median", "p75", "max"], default="mean")
    parser.add_argument("--contrast-mix", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "results" / "figures")
    return parser.parse_args()


def ablation_scores(dataset, args: argparse.Namespace, seed: int) -> dict[str, object]:
    base = {
        "VAR-Lasso (global)": lasso_var_scores(
            dataset.x,
            max_lag=args.max_lag,
            alpha=args.lasso_alpha,
        ),
        "DynVAR-Lasso (window max)": dynamic_lasso_var_scores(
            dataset.x,
            max_lag=args.max_lag,
            alpha=args.lasso_alpha,
            window_size=args.window_size,
            stride=args.window_stride,
            aggregate=args.window_aggregate,
        ),
    }

    state_base = {
        "base_model": "lasso",
        "lasso_alpha": args.lasso_alpha,
        "window_size": args.window_size,
        "stride": args.window_stride,
        "aggregate": args.window_aggregate,
        "state_aggregate": args.state_aggregate,
        "seed": seed + 40_000,
    }
    base["STCG-v1/no-state"] = stcg_v1_scores(
        dataset.x,
        args.max_lag,
        STCGStateConfig(n_states=1, contrast_mix=0.0, **state_base),
    )
    base["STCG-v1/state-only"] = stcg_v1_scores(
        dataset.x,
        args.max_lag,
        STCGStateConfig(n_states=2, contrast_mix=0.0, **state_base),
    )
    base["STCG-v1/forced-states+contrast"] = stcg_v1_scores(
        dataset.x,
        args.max_lag,
        STCGStateConfig(n_states=2, auto_states=False, contrast_mix=args.contrast_mix, **state_base),
    )
    base["STCG-v1/state+contrast"] = stcg_v1_scores(
        dataset.x,
        args.max_lag,
        STCGStateConfig(n_states=2, contrast_mix=args.contrast_mix, **state_base),
    )
    return base


def row_for(kind: str, seed: int, method: str, summary: dict[str, float]) -> dict[str, object]:
    row: dict[str, object] = {"generator": kind, "seed": seed, "method": method}
    row.update(summary)
    return row


def write_detail_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["generator", "seed", "method", *METRICS, "true_edges", "edge_density"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["method"])].append(row)

    summary_rows: list[dict[str, object]] = []
    for method, items in groups.items():
        summary: dict[str, object] = {"method": method, "n": len(items)}
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            mean = sum(values) / len(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = (
                math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
                if len(values) > 1
                else 0.0
            )
        summary_rows.append(summary)
    return summary_rows


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["method", "n"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], kind: str) -> None:
    lines = [
        f"# STCG-v1 Ablation Summary ({kind})",
        "",
        "| Method | N | Edge AUC | Pair AUC | Precision@K | SHD | Lag Acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {n} | {edge_auc_mean:.3f} +/- {edge_auc_std:.3f} | "
            "{pair_auc_mean:.3f} +/- {pair_auc_std:.3f} | "
            "{precision_at_k_mean:.3f} +/- {precision_at_k_std:.3f} | "
            "{shd_mean:.1f} +/- {shd_std:.1f} | "
            "{lag_accuracy_mean:.3f} +/- {lag_accuracy_std:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Edge AUC & Pair AUC & P@K & SHD & Lag Acc \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            "{method} & {edge_auc_mean:.3f} $\\pm$ {edge_auc_std:.3f} & "
            "{pair_auc_mean:.3f} $\\pm$ {pair_auc_std:.3f} & "
            "{precision_at_k_mean:.3f} $\\pm$ {precision_at_k_std:.3f} & "
            "{shd_mean:.1f} $\\pm$ {shd_std:.1f} & "
            "{lag_accuracy_mean:.3f} $\\pm$ {lag_accuracy_std:.3f} \\\\".format(**row)
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, object]]) -> None:
    methods = [str(row["method"]) for row in rows]
    means = [float(row["edge_auc_mean"]) for row in rows]
    stds = [float(row["edge_auc_std"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars = ax.barh(methods, means, xerr=stds, capsize=3)
    ax.set_xlabel("Lagged Edge AUC")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="x", alpha=0.25)
    best = max(means)
    for bar, mean in zip(bars, means):
        ax.text(min(mean + 0.015, 0.97), bar.get_y() + bar.get_height() / 2, f"{mean:.3f}", va="center")
        if mean == best:
            bar.set_alpha(1.0)
        else:
            bar.set_alpha(0.72)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _common.warn_if_results_tables_output(
        "run_stcg_ablation.py", args.output_dir, len(args.seeds), canonical_seeds=20
    )
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        dataset = generate_lagged_var(
            n_nodes=args.n_nodes,
            timesteps=args.timesteps,
            max_lag=args.max_lag,
            edge_prob=args.edge_prob,
            noise_scale=args.noise_scale,
            seed=seed,
            nonlinear=args.kind == "nonlinear_var",
            switching=args.kind == "switching_var",
        )
        for method, scores in ablation_scores(dataset, args, seed).items():
            rows.append(row_for(args.kind, seed, method, recovery_summary(scores, dataset.edge_truth)))

    summary_rows = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "stcg_v1_ablation_detail.csv"
    summary_path = args.output_dir / "stcg_v1_ablation_summary.csv"
    markdown_path = args.output_dir / "stcg_v1_ablation_summary.md"
    latex_path = args.output_dir / "stcg_v1_ablation_table.tex"
    figure_path = args.figure_dir / "stcg_v1_ablation_edge_auc.png"

    write_detail_csv(detail_path, rows)
    write_summary_csv(summary_path, summary_rows)
    write_markdown(markdown_path, summary_rows, args.kind)
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
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
