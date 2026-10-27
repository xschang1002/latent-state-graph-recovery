"""Compare inferred STCG graph states with an oracle regime-state upper bound."""

from __future__ import annotations

import argparse
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

from _common import fmt_dash as fmt, state_config, write_csv
from stcg import (
    STCGStateConfig,
    dynamic_lasso_var_scores,
    generate_lagged_var,
    recovery_summary,
    stcg_v1_diagnostics,
)
from stcg.self_supervised import _aggregate_window_scores, _state_aggregate


METRICS = ["edge_auc", "pair_auc", "precision_at_k", "shd", "lag_accuracy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "results" / "figures")
    return parser.parse_args()


def majority_regime(regimes: np.ndarray, start: int, end: int) -> tuple[int, float]:
    values = regimes[start:end].astype(int)
    if values.size == 0:
        raise ValueError("Cannot compute majority regime for an empty window")
    counts = np.bincount(values)
    label = int(np.argmax(counts))
    return label, float(counts[label] / values.size)


def oracle_state_scores(
    window_scores: np.ndarray,
    bounds: tuple[tuple[int, int], ...],
    regimes: np.ndarray,
    state_aggregate: str,
    aggregate: str,
) -> tuple[np.ndarray, float, int]:
    labels: list[int] = []
    purities: list[float] = []
    for start, end in bounds:
        label, purity = majority_regime(regimes, start, end)
        labels.append(label)
        purities.append(purity)

    label_array = np.asarray(labels, dtype=int)
    state_scores = []
    for label in sorted(np.unique(label_array)):
        members = window_scores[label_array == label]
        state_scores.append(_state_aggregate(members, state_aggregate))
    return _aggregate_window_scores(state_scores, aggregate), float(np.mean(purities)), int(np.unique(label_array).size)


def row_for(seed: int, method: str, summary: dict[str, float], extra: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {"generator": "switching_var", "seed": seed, "method": method}
    row.update(summary)
    row.update(extra)
    return row


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["method"])].append(row)
    summary_rows: list[dict[str, object]] = []
    for method, items in sorted(groups.items()):
        out: dict[str, object] = {"method": method, "n": len(items)}
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            mean = float(np.mean(values))
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        for key in ["inferred_state_count", "oracle_state_count", "oracle_window_purity"]:
            values = [float(item[key]) for item in items if key in item and not math.isnan(float(item[key]))]
            out[f"{key}_mean"] = float(np.mean(values)) if values else math.nan
        summary_rows.append(out)
    return summary_rows


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Synthetic Oracle-State Upper Bound",
        "",
        "Oracle-state aggregation uses true synthetic regime labels only for analysis.",
        "It is an upper-bound diagnostic, not a supervised variant of STCG-v1.",
        "",
        "| Method | N | Edge AUC | P@K | SHD | Lag Acc. | Inferred K | Oracle K | Oracle purity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {n} | {edge_auc_mean:.3f} +/- {edge_auc_std:.3f} | "
            "{precision_at_k_mean:.3f} +/- {precision_at_k_std:.3f} | "
            "{shd_mean:.1f} +/- {shd_std:.1f} | {lag_accuracy_mean:.3f} +/- {lag_accuracy_std:.3f} | "
            "{inferred_k} | {oracle_k} | {oracle_purity} |".format(
                inferred_k=fmt(row["inferred_state_count_mean"], 2),
                oracle_k=fmt(row["oracle_state_count_mean"], 2),
                oracle_purity=fmt(row["oracle_window_purity_mean"], 3),
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Edge AUC & P@K & SHD & Lag Acc. \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            "{method} & {edge_auc_mean:.3f} & {precision_at_k_mean:.3f} & "
            "{shd_mean:.1f} & {lag_accuracy_mean:.3f} \\\\".format(**row)
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, object]]) -> None:
    methods = [str(row["method"]) for row in rows]
    means = [float(row["edge_auc_mean"]) for row in rows]
    stds = [float(row["edge_auc_std"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    bars = ax.barh(methods, means, xerr=stds, capsize=3, color="#4c78a8")
    ax.set_xlabel("Lagged Edge AUC")
    ax.set_xlim(0.75, min(1.0, max(means) + 0.08))
    ax.grid(axis="x", alpha=0.25)
    for bar, mean in zip(bars, means):
        ax.text(min(mean + 0.004, 0.995), bar.get_y() + bar.get_height() / 2, f"{mean:.3f}", va="center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        dataset = generate_lagged_var(
            n_nodes=args.n_nodes,
            timesteps=args.timesteps,
            max_lag=args.max_lag,
            edge_prob=args.edge_prob,
            noise_scale=args.noise_scale,
            seed=seed,
            switching=True,
        )
        inferred = stcg_v1_diagnostics(dataset.x, args.max_lag, state_config(args, seed, n_states=2, seed_offset=80_000))
        no_state = stcg_v1_diagnostics(dataset.x, args.max_lag, state_config(args, seed, n_states=1, seed_offset=80_000))
        oracle_scores, oracle_purity, oracle_k = oracle_state_scores(
            inferred.window_scores,
            inferred.bounds,
            dataset.regimes,
            args.state_aggregate,
            args.window_aggregate,
        )
        methods = {
            "DynVAR-Lasso": dynamic_lasso_var_scores(
                dataset.x,
                max_lag=args.max_lag,
                alpha=args.lasso_alpha,
                window_size=args.window_size,
                stride=args.window_stride,
                aggregate=args.window_aggregate,
            ),
            "STCG-v1/no-state": no_state.scores,
            "STCG-v1/inferred-state": inferred.scores,
            "Oracle-state upper bound": oracle_scores,
        }
        for method, scores in methods.items():
            rows.append(
                row_for(
                    seed,
                    method,
                    recovery_summary(scores, dataset.edge_truth),
                    {
                        "inferred_state_count": int(np.unique(inferred.labels).size)
                        if method == "STCG-v1/inferred-state"
                        else math.nan,
                        "oracle_state_count": oracle_k if method == "Oracle-state upper bound" else math.nan,
                        "oracle_window_purity": oracle_purity if method == "Oracle-state upper bound" else math.nan,
                    },
                )
            )

    summary_rows = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "synthetic_state_oracle_detail.csv"
    summary_path = args.output_dir / "synthetic_state_oracle_summary.csv"
    markdown_path = args.output_dir / "synthetic_state_oracle_summary.md"
    latex_path = args.output_dir / "synthetic_state_oracle_table.tex"
    figure_path = args.figure_dir / "synthetic_state_oracle_edge_auc.png"

    write_csv(
        detail_path,
        rows,
        [
            "generator",
            "seed",
            "method",
            *METRICS,
            "true_edges",
            "edge_density",
            "inferred_state_count",
            "oracle_state_count",
            "oracle_window_purity",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        [
            "method",
            "n",
            *[f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")],
            "inferred_state_count_mean",
            "oracle_state_count_mean",
            "oracle_window_purity_mean",
        ],
    )
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
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
