"""Sweep switching-regime dwell time to probe when graph-state aggregation helps."""

from __future__ import annotations

import argparse
import json
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
from stcg import STCGStateConfig, dynamic_lasso_var_scores, generate_lagged_var, recovery_summary, stcg_v1_diagnostics
from stcg.self_supervised import _aggregate_window_scores, _state_aggregate


METRICS = ["edge_auc", "pair_auc", "precision_at_k", "shd", "lag_accuracy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--switch-periods", nargs="+", type=int, default=[150, 300, 600])
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1200)
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
    counts = np.bincount(values)
    label = int(np.argmax(counts))
    return label, float(counts[label] / values.size)


def oracle_scores(diagnostics, regimes: np.ndarray, state_aggregate: str, aggregate: str) -> tuple[np.ndarray, float]:
    labels = []
    purities = []
    for start, end in diagnostics.bounds:
        label, purity = majority_regime(regimes, start, end)
        labels.append(label)
        purities.append(purity)
    label_array = np.asarray(labels, dtype=int)
    state_scores = [
        _state_aggregate(diagnostics.window_scores[label_array == label], state_aggregate)
        for label in sorted(np.unique(label_array))
    ]
    return _aggregate_window_scores(state_scores, aggregate), float(np.mean(purities))


def row_for(period: int, seed: int, method: str, summary: dict[str, float], extra: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {"switch_period": period, "seed": seed, "method": method}
    row.update(summary)
    row.update(extra)
    return row


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["switch_period"]), str(row["method"]))].append(row)
    summary_rows = []
    for (period, method), items in sorted(groups.items()):
        out: dict[str, object] = {"switch_period": period, "method": method, "n": len(items)}
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            out[f"{metric}_mean"] = float(np.mean(values))
            out[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        for key in ["state_count", "state_improvement", "temporal_coherence", "oracle_window_purity"]:
            values = [float(item[key]) for item in items if not np.isnan(float(item[key]))]
            out[f"{key}_mean"] = float(np.mean(values)) if values else np.nan
        summary_rows.append(out)
    return summary_rows


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Synthetic Switch-Difficulty Sweep",
        "",
        "Smaller switch periods create more frequent graph changes and more mixed sliding windows.",
        "",
        "| Switch period | Method | N | Edge AUC | P@K | SHD | State K | State improvement | Temporal coherence | Oracle purity |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {switch_period} | {method} | {n} | {edge_auc_mean:.3f} +/- {edge_auc_std:.3f} | "
            "{precision_at_k_mean:.3f} +/- {precision_at_k_std:.3f} | {shd_mean:.1f} +/- {shd_std:.1f} | "
            "{state_k} | {state_improvement} | {temporal_coherence} | {oracle_purity} |".format(
                state_k=fmt(row["state_count_mean"], 2),
                state_improvement=fmt(row["state_improvement_mean"], 3),
                temporal_coherence=fmt(row["temporal_coherence_mean"], 3),
                oracle_purity=fmt(row["oracle_window_purity_mean"], 3),
                **row,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "\\begin{tabular}{rlrrr}",
        "\\toprule",
        "Period & Method & Edge AUC & P@K & SHD \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            "{switch_period} & {method} & {edge_auc_mean:.3f} & {precision_at_k_mean:.3f} & "
            "{shd_mean:.1f} \\\\".format(**row)
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, rows: list[dict[str, object]]) -> None:
    periods = sorted(set(int(row["switch_period"]) for row in rows))
    methods = ["DynVAR-Lasso", "STCG-v1", "Oracle-state upper bound"]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    lows: list[float] = []
    highs: list[float] = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        values = [next(float(row["edge_auc_mean"]) for row in method_rows if int(row["switch_period"]) == period) for period in periods]
        errors = [next(float(row["edge_auc_std"]) for row in method_rows if int(row["switch_period"]) == period) for period in periods]
        ax.errorbar(periods, values, yerr=errors, marker="o", capsize=3, label=method)
        lows.extend(v - e for v, e in zip(values, errors))
        highs.extend(v + e for v, e in zip(values, errors))
    ax.set_xlabel("Regime switch period (timesteps)")
    ax.set_ylabel("Lagged Edge AUC")
    if lows:
        # Adaptive y-limits: fit the data (means +/- std) with small padding
        # instead of a hardcoded band.
        y_low, y_high = min(lows), max(highs)
        pad = max(0.01, 0.05 * (y_high - y_low))
        ax.set_ylim(y_low - pad, y_high + pad)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for period in args.switch_periods:
        for seed in args.seeds:
            dataset = generate_lagged_var(
                n_nodes=args.n_nodes,
                timesteps=args.timesteps,
                max_lag=args.max_lag,
                edge_prob=args.edge_prob,
                noise_scale=args.noise_scale,
                seed=seed,
                switching=True,
                switch_period=period,
            )
            diagnostics = stcg_v1_diagnostics(dataset.x, args.max_lag, state_config(args, seed, seed_offset=90_000))
            oracle, purity = oracle_scores(
                diagnostics,
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
                "STCG-v1": diagnostics.scores,
                "Oracle-state upper bound": oracle,
            }
            for method, scores in methods.items():
                rows.append(
                    row_for(
                        period,
                        seed,
                        method,
                        recovery_summary(scores, dataset.edge_truth),
                        {
                            "state_count": int(np.unique(diagnostics.labels).size) if method == "STCG-v1" else np.nan,
                            "state_improvement": float(diagnostics.state_improvement) if method == "STCG-v1" else np.nan,
                            "temporal_coherence": float(diagnostics.temporal_coherence) if method == "STCG-v1" else np.nan,
                            "oracle_window_purity": purity if method == "Oracle-state upper bound" else np.nan,
                        },
                    )
                )

    summary_rows = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "synthetic_switch_difficulty_detail.csv"
    summary_path = args.output_dir / "synthetic_switch_difficulty_summary.csv"
    markdown_path = args.output_dir / "synthetic_switch_difficulty_summary.md"
    latex_path = args.output_dir / "synthetic_switch_difficulty_table.tex"
    figure_path = args.figure_dir / "synthetic_switch_difficulty_edge_auc.png"
    write_csv(
        detail_path,
        rows,
        [
            "switch_period",
            "seed",
            "method",
            *METRICS,
            "true_edges",
            "edge_density",
            "state_count",
            "state_improvement",
            "temporal_coherence",
            "oracle_window_purity",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        [
            "switch_period",
            "method",
            "n",
            *[f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")],
            "state_count_mean",
            "state_improvement_mean",
            "temporal_coherence_mean",
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
