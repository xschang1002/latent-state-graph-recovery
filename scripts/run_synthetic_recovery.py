"""Run the synthetic structure recovery experiment loop."""

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
    STCGConfig,
    STCGStateConfig,
    dynamic_granger_f_scores,
    dynamic_lasso_var_scores,
    dynamic_ridge_var_scores,
    generate_lagged_var,
    granger_f_scores,
    lagged_correlation_scores,
    lasso_var_scores,
    random_scores,
    recovery_summary,
    ridge_var_scores,
    stcg_v0_scores,
    stcg_v1_scores,
)


DEFAULT_KINDS = ["var", "nonlinear_var", "switching_var"]
METRICS = ["edge_auc", "pair_auc", "precision_at_k", "shd", "lag_accuracy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinds", nargs="+", choices=DEFAULT_KINDS, default=DEFAULT_KINDS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--edge-prob", type=float, default=0.18)
    parser.add_argument("--noise-scale", type=float, default=0.10)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--lasso-alpha", type=float, default=0.005)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--window-aggregate", choices=["max", "mean", "p90"], default="max")
    parser.add_argument("--skip-stcg", action="store_true")
    parser.add_argument("--stcg-steps", type=int, default=80)
    parser.add_argument("--stcg-lr", type=float, default=0.03)
    parser.add_argument("--stcg-sparsity-weight", type=float, default=0.015)
    parser.add_argument("--stcg-l1-weight", type=float, default=0.002)
    parser.add_argument("--stcg-entropy-weight", type=float, default=0.001)
    parser.add_argument("--stcg-contrast-weight", type=float, default=0.0)
    parser.add_argument("--stcg-consistency-weight", type=float, default=0.01)
    parser.add_argument("--stcg-score-prior-mix", type=float, default=0.0)
    parser.add_argument("--stcg-v1-states", type=int, default=2)
    parser.add_argument("--stcg-v1-base", choices=["ridge", "lasso"], default="lasso")
    parser.add_argument("--stcg-v1-state-aggregate", choices=["mean", "median", "p75", "max"], default="mean")
    parser.add_argument("--stcg-v1-min-improvement", type=float, default=0.12)
    parser.add_argument("--stcg-v1-contrast-mix", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "results" / "figures")
    return parser.parse_args()


def score_methods(
    dataset,
    max_lag: int,
    seed: int,
    ridge_alpha: float,
    lasso_alpha: float,
    window_size: int | None,
    window_stride: int | None,
    window_aggregate: str,
    stcg_config: STCGConfig | None,
    stcg_state_config: STCGStateConfig | None,
):
    n_nodes = dataset.x.shape[1]
    methods = {
        "Random": random_scores(n_nodes, max_lag, seed=seed),
        "LagCorr": lagged_correlation_scores(dataset.x, max_lag),
        "VAR-Ridge": ridge_var_scores(dataset.x, max_lag, alpha=ridge_alpha),
        "VAR-Lasso": lasso_var_scores(dataset.x, max_lag, alpha=lasso_alpha),
        "Granger-F": granger_f_scores(dataset.x, max_lag),
        "DynVAR-Ridge": dynamic_ridge_var_scores(
            dataset.x,
            max_lag,
            alpha=ridge_alpha,
            window_size=window_size,
            stride=window_stride,
            aggregate=window_aggregate,
        ),
        "DynVAR-Lasso": dynamic_lasso_var_scores(
            dataset.x,
            max_lag,
            alpha=lasso_alpha,
            window_size=window_size,
            stride=window_stride,
            aggregate=window_aggregate,
        ),
        "DynGranger-F": dynamic_granger_f_scores(
            dataset.x,
            max_lag,
            window_size=window_size,
            stride=window_stride,
            aggregate=window_aggregate,
        ),
    }
    if stcg_config is not None:
        methods["STCG-v0"] = stcg_v0_scores(dataset.x, max_lag, config=stcg_config)
    if stcg_state_config is not None:
        methods["STCG-v1"] = stcg_v1_scores(dataset.x, max_lag, config=stcg_state_config)
    return methods


def row_for(kind: str, seed: int, method: str, summary: dict[str, float]) -> dict[str, object]:
    row: dict[str, object] = {"generator": kind, "seed": seed, "method": method}
    row.update(summary)
    return row


def format_float(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    _common.write_csv(path, rows, ["generator", "seed", "method", *METRICS, "true_edges", "edge_density"])


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["generator"]), str(row["method"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (generator, method), items in sorted(groups.items()):
        summary: dict[str, object] = {
            "generator": generator,
            "method": method,
            "n": len(items),
        }
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            summary[f"{metric}_mean"] = sum(values) / len(values)
            if len(values) > 1:
                mean = float(summary[f"{metric}_mean"])
                summary[f"{metric}_std"] = math.sqrt(
                    sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                )
            else:
                summary[f"{metric}_std"] = 0.0
        summary_rows.append(summary)
    return summary_rows


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["generator", "method", "n"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Synthetic Recovery Summary",
        "",
        "| Generator | Method | N | Edge AUC | Pair AUC | Precision@K | SHD | Lag Acc |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {generator} | {method} | {n} | {edge_auc_mean:.3f} +/- {edge_auc_std:.3f} | "
            "{pair_auc_mean:.3f} +/- {pair_auc_std:.3f} | "
            "{precision_at_k_mean:.3f} +/- {precision_at_k_std:.3f} | "
            "{shd_mean:.1f} +/- {shd_std:.1f} | "
            "{lag_accuracy_mean:.3f} +/- {lag_accuracy_std:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Generator & Method & Edge AUC & Pair AUC & P@K & SHD & Lag Acc \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            "{generator} & {method} & {edge_auc_mean:.3f} $\\pm$ {edge_auc_std:.3f} & "
            "{pair_auc_mean:.3f} $\\pm$ {pair_auc_std:.3f} & "
            "{precision_at_k_mean:.3f} $\\pm$ {precision_at_k_std:.3f} & "
            "{shd_mean:.1f} $\\pm$ {shd_std:.1f} & "
            "{lag_accuracy_mean:.3f} $\\pm$ {lag_accuracy_std:.3f} \\\\".format(**row)
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_edge_auc_figure(path: Path, rows: list[dict[str, object]]) -> None:
    generators = list(dict.fromkeys(str(row["generator"]) for row in rows))
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    lookup = {(row["generator"], row["method"]): row for row in rows}

    width = min(0.12, 0.78 / max(1, len(methods)))
    x_positions = list(range(len(generators)))
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    for method_idx, method in enumerate(methods):
        offsets = [x + (method_idx - (len(methods) - 1) / 2) * width for x in x_positions]
        means = [float(lookup[(generator, method)]["edge_auc_mean"]) for generator in generators]
        stds = [float(lookup[(generator, method)]["edge_auc_std"]) for generator in generators]
        ax.bar(offsets, means, width=width, yerr=stds, capsize=3, label=method)

    ax.set_ylabel("Lagged Edge AUC")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(generators)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.33))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _common.warn_if_results_tables_output(
        "run_synthetic_recovery.py", args.output_dir, len(args.seeds), canonical_seeds=20
    )
    rows: list[dict[str, object]] = []

    for kind in args.kinds:
        for seed in args.seeds:
            stcg_config = None
            stcg_state_config = None
            if not args.skip_stcg:
                stcg_config = STCGConfig(
                    steps=args.stcg_steps,
                    learning_rate=args.stcg_lr,
                    ridge_alpha=args.ridge_alpha,
                    sparsity_weight=args.stcg_sparsity_weight,
                    l1_weight=args.stcg_l1_weight,
                    entropy_weight=args.stcg_entropy_weight,
                    contrast_weight=args.stcg_contrast_weight,
                    consistency_weight=args.stcg_consistency_weight,
                    score_prior_mix=args.stcg_score_prior_mix,
                    window_size=args.window_size,
                    stride=args.window_stride,
                    aggregate=args.window_aggregate,
                    seed=seed + 20_000,
                )
                stcg_state_config = STCGStateConfig(
                    n_states=args.stcg_v1_states,
                    base_model=args.stcg_v1_base,
                    ridge_alpha=args.ridge_alpha,
                    lasso_alpha=args.lasso_alpha,
                    window_size=args.window_size,
                    stride=args.window_stride,
                    aggregate=args.window_aggregate,
                    state_aggregate=args.stcg_v1_state_aggregate,
                    min_state_improvement=args.stcg_v1_min_improvement,
                    contrast_mix=args.stcg_v1_contrast_mix,
                    seed=seed + 30_000,
                )
            dataset = generate_lagged_var(
                n_nodes=args.n_nodes,
                timesteps=args.timesteps,
                max_lag=args.max_lag,
                edge_prob=args.edge_prob,
                noise_scale=args.noise_scale,
                seed=seed,
                nonlinear=kind == "nonlinear_var",
                switching=kind == "switching_var",
            )
            for method, scores in score_methods(
                dataset,
                max_lag=args.max_lag,
                seed=seed + 10_000,
                ridge_alpha=args.ridge_alpha,
                lasso_alpha=args.lasso_alpha,
                window_size=args.window_size,
                window_stride=args.window_stride,
                window_aggregate=args.window_aggregate,
                stcg_config=stcg_config,
                stcg_state_config=stcg_state_config,
            ).items():
                rows.append(row_for(kind, seed, method, recovery_summary(scores, dataset.edge_truth)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "synthetic_recovery_detail.csv"
    summary_path = args.output_dir / "synthetic_recovery_summary.csv"
    markdown_path = args.output_dir / "synthetic_recovery_summary.md"
    latex_path = args.output_dir / "synthetic_recovery_table.tex"
    figure_path = args.figure_dir / "synthetic_recovery_edge_auc.png"

    summary_rows = aggregate(rows)
    write_csv(detail_path, rows)
    write_summary_csv(summary_path, summary_rows)
    write_markdown(markdown_path, summary_rows)
    write_latex(latex_path, summary_rows)
    write_edge_auc_figure(figure_path, summary_rows)

    payload = {
        "detail_csv": str(detail_path),
        "summary_csv": str(summary_path),
        "summary_md": str(markdown_path),
        "table_tex": str(latex_path),
        "figure": str(figure_path),
        "n_rows": len(rows),
        "n_summary_rows": len(summary_rows),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
