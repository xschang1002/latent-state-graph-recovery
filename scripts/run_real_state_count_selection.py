"""Select real graph-state count without using external labels.

The selector sweeps candidate state counts, scores each candidate only from
graph-state diagnostics, and then uses held-out labels only for audit metrics.
"""

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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from _common import fmt, state_config, write_csv
from _state_selection import SELECTION_POLICIES, select_state_candidate_index
from run_real_state_label_prediction import (
    fit_state_label_mapping,
    load_series_and_labels,
    metric_values,
    predict_from_state_mapping,
    stratified_window_split,
    window_majority_labels,
)
from stcg import STCGStateConfig, stcg_v1_diagnostics
from stcg.self_supervised import _min_cluster_fraction

warnings.filterwarnings("ignore", category=ConvergenceWarning)


METRICS = ["accuracy", "balanced_accuracy", "macro_f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=ROOT / "data" / "raw" / "HydraulicSystems.csv")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=ROOT / "data" / "processed" / "HydraulicSystems_labels.csv",
    )
    parser.add_argument("--dataset-name", type=str, default="HydraulicSystems")
    parser.add_argument("--label-column", type=str, default="cooler_condition")
    parser.add_argument("--columns", type=str, default="TS1,TS2,TS3,TS4,VS1,CE,CP,SE")
    parser.add_argument("--time-column", type=str, default="time_index")
    parser.add_argument("--missing", choices=["drop", "interpolate"], default="interpolate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--state-counts", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--selection-policy", choices=SELECTION_POLICIES, default="reject_nonpositive",
                        help="Reject to fitted K=1 if all scores are nonpositive; legacy reproduces historical all-zero ties")
    parser.add_argument("--test-fraction", type=float, default=0.50)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--lasso-alpha", type=float, default=0.005)
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--window-stride", type=int, default=150)
    parser.add_argument("--window-aggregate", choices=["max", "mean", "p90"], default="max")
    parser.add_argument("--state-aggregate", choices=["mean", "median", "p75", "max"], default="mean")
    parser.add_argument("--min-improvement", type=float, default=0.04)
    parser.add_argument("--min-temporal-coherence", type=float, default=0.55)
    parser.add_argument("--min-cluster-fraction", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "results" / "figures")
    args = parser.parse_args()
    if args.selection_policy == "reject_nonpositive" and 1 not in args.state_counts:
        parser.error("--state-counts must include 1 with --selection-policy reject_nonpositive")
    return args


def state_count_selection_score(
    improvement: float,
    temporal_coherence: float,
    min_cluster_fraction: float,
    inferred_state_count: int,
    min_improvement: float = 0.04,
    min_temporal_coherence: float = 0.55,
    min_fraction: float = 0.05,
) -> tuple[float, float]:
    """Return label-free selection score and balance term.

    The score favors candidates that improve over one cluster, are temporally
    persistent, and do not split off tiny residual clusters. The balance term is
    one when every cluster is at least as large as a perfectly balanced cluster's
    minimum share, and decreases for over-split candidates. A logarithmic
    parsimony term discourages gratuitous state splitting.
    """

    if inferred_state_count <= 1:
        return 0.0, 0.0
    if improvement < min_improvement or temporal_coherence < min_temporal_coherence:
        return 0.0, 0.0
    if min_cluster_fraction < min_fraction:
        return 0.0, 0.0
    balance = min(1.0, float(min_cluster_fraction * inferred_state_count))
    parsimony = math.log2(float(inferred_state_count) + 1.0)
    score = float(improvement * temporal_coherence * balance / parsimony)
    return score, balance


def evaluate_candidate(
    dataset: str,
    label_column: str,
    x: np.ndarray,
    labels: list[str],
    args: argparse.Namespace,
    seed: int,
    state_count: int,
) -> dict[str, object]:
    diagnostics = stcg_v1_diagnostics(
        x,
        max_lag=args.max_lag,
        config=state_config(args, seed, n_states=state_count, auto_states=False),
    )
    true_labels, purities = window_majority_labels(labels, tuple(diagnostics.bounds))
    train_indices, test_indices = stratified_window_split(true_labels, args.test_fraction, seed)
    mapping, fallback = fit_state_label_mapping(diagnostics.labels, true_labels, train_indices)
    predicted = predict_from_state_mapping(diagnostics.labels, mapping, fallback, test_indices)
    true_test = [true_labels[idx] for idx in test_indices]
    metrics = metric_values(true_test, predicted)

    inferred_count = int(np.unique(diagnostics.labels).size)
    min_fraction = _min_cluster_fraction(diagnostics.labels)
    selection_score, balance = state_count_selection_score(
        float(diagnostics.state_improvement),
        float(diagnostics.temporal_coherence),
        min_fraction,
        inferred_count,
        min_improvement=args.min_improvement,
        min_temporal_coherence=args.min_temporal_coherence,
        min_fraction=args.min_cluster_fraction,
    )

    row: dict[str, object] = {
        "dataset": dataset,
        "seed": seed,
        "label_column": label_column,
        "candidate_state_count": state_count,
        "inferred_state_count": inferred_count,
        "n_windows": len(true_labels),
        "train_windows": len(train_indices),
        "test_windows": len(test_indices),
        "state_improvement": float(diagnostics.state_improvement),
        "temporal_coherence": float(diagnostics.temporal_coherence),
        "min_cluster_fraction": min_fraction,
        "balance_score": balance,
        "selection_score": selection_score,
        "label_purity_mean": float(np.mean(purities)),
        "state_counts": json.dumps(
            dict(sorted(Counter(int(value) for value in diagnostics.labels).items())),
            sort_keys=True,
        ),
        "state_mapping": json.dumps(mapping, sort_keys=True),
        "selected": False,
    }
    row.update(metrics)
    return row


def mark_selected(rows: list[dict[str, object]], policy: str = "reject_nonpositive") -> None:
    by_seed: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        row["selected"] = False
        by_seed[int(row["seed"])].append(row)
    for seed_rows in by_seed.values():
        best_index = select_state_candidate_index(
            [(
                float(row["selection_score"]),
                float(row["temporal_coherence"]),
                int(row["candidate_state_count"]),
            ) for row in seed_rows],
            policy=policy,
        )
        seed_rows[best_index]["selected"] = True


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.array(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def aggregate_by_candidate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[int(row["candidate_state_count"])].append(row)

    summary_rows: list[dict[str, object]] = []
    for state_count, items in sorted(groups.items()):
        summary: dict[str, object] = {
            "candidate_state_count": state_count,
            "n": len(items),
            "selected_count": sum(1 for item in items if bool(item["selected"])),
            "inferred_state_count_distribution": json.dumps(
                dict(sorted(Counter(int(item["inferred_state_count"]) for item in items).items())),
                sort_keys=True,
            ),
        }
        for key in [
            "selection_score",
            "state_improvement",
            "temporal_coherence",
            "min_cluster_fraction",
            "balance_score",
            *METRICS,
        ]:
            mean, std = mean_std([float(item[key]) for item in items])
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = std
        summary_rows.append(summary)
    return summary_rows


def aggregate_selected(rows: list[dict[str, object]]) -> dict[str, object]:
    selected = [row for row in rows if bool(row["selected"])]
    if not selected:
        raise ValueError("No selected state-count rows found")
    summary: dict[str, object] = {
        "n": len(selected),
        "selected_state_count_distribution": json.dumps(
            dict(sorted(Counter(int(item["candidate_state_count"]) for item in selected).items())),
            sort_keys=True,
        ),
    }
    for key in [
        "selection_score",
        "state_improvement",
        "temporal_coherence",
        "min_cluster_fraction",
        "balance_score",
        *METRICS,
    ]:
        mean, std = mean_std([float(item[key]) for item in selected])
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
    return summary


def write_markdown(
    path: Path,
    dataset: str,
    label_column: str,
    candidate_summary: list[dict[str, object]],
    selected_summary: dict[str, object],
    selection_policy: str = "reject_nonpositive",
) -> None:
    lines = [
        "# Real State Count Selection",
        "",
        "State counts are selected without external labels using:",
        "",
        "`selection_score = state_improvement * temporal_coherence * balance_score / log2(K + 1)`",
        "",
        "External labels are used only for the audit metrics shown below.",
        f"Selection policy: `{selection_policy}`. The default rejects to the fitted K=1 candidate when no score is positive; `legacy` reproduces the historical tie-break.",
        "",
        "## Selected",
        "",
        "| Dataset | Label | Selected K Distribution | Balanced Acc. | Macro F1 | Score |",
        "|---|---|---|---:|---:|---:|",
        "| {dataset} | {label} | `{dist}` | {bal} +/- {bal_std} | {f1} +/- {f1_std} | {score} +/- {score_std} |".format(
            dataset=dataset,
            label=label_column,
            dist=selected_summary["selected_state_count_distribution"],
            bal=fmt(selected_summary["balanced_accuracy_mean"]),
            bal_std=fmt(selected_summary["balanced_accuracy_std"]),
            f1=fmt(selected_summary["macro_f1_mean"]),
            f1_std=fmt(selected_summary["macro_f1_std"]),
            score=fmt(selected_summary["selection_score_mean"]),
            score_std=fmt(selected_summary["selection_score_std"]),
        ),
        "",
        "## Candidate Sweep",
        "",
        "| K | Selected | Score | Improvement | Temporal Coherence | Min Fraction | Balance | Balanced Acc. | Macro F1 | Inferred Counts |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in candidate_summary:
        lines.append(
            "| {k} | {selected} | {score} +/- {score_std} | {imp} | {coh} | {minfrac} | {balance} | {bal} +/- {bal_std} | {f1} +/- {f1_std} | `{counts}` |".format(
                k=row["candidate_state_count"],
                selected=row["selected_count"],
                score=fmt(row["selection_score_mean"]),
                score_std=fmt(row["selection_score_std"]),
                imp=fmt(row["state_improvement_mean"]),
                coh=fmt(row["temporal_coherence_mean"]),
                minfrac=fmt(row["min_cluster_fraction_mean"]),
                balance=fmt(row["balance_score_mean"]),
                bal=fmt(row["balanced_accuracy_mean"]),
                bal_std=fmt(row["balanced_accuracy_std"]),
                f1=fmt(row["macro_f1_mean"]),
                f1_std=fmt(row["macro_f1_std"]),
                counts=row["inferred_state_count_distribution"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, candidate_summary: list[dict[str, object]]) -> None:
    state_counts = [int(row["candidate_state_count"]) for row in candidate_summary]
    scores = [float(row["selection_score_mean"]) for row in candidate_summary]
    balanced = [float(row["balanced_accuracy_mean"]) for row in candidate_summary]

    fig, ax1 = plt.subplots(figsize=(7.2, 3.8))
    ax1.plot(state_counts, scores, marker="o", color="#4c78a8", label="Selection score")
    ax1.set_xlabel("Candidate state count K")
    ax1.set_ylabel("Label-free selection score")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(state_counts, balanced, marker="s", color="#f58518", label="Audit balanced accuracy")
    ax2.set_ylabel("Held-out balanced accuracy")
    ax2.set_ylim(0.0, 1.05)
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    dataset, x, _columns, labels = load_series_and_labels(args)

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for state_count in args.state_counts:
            rows.append(
                evaluate_candidate(
                    dataset=dataset,
                    label_column=args.label_column,
                    x=x,
                    labels=labels,
                    args=args,
                    seed=seed,
                    state_count=state_count,
                )
            )
    mark_selected(rows, policy=args.selection_policy)
    candidate_summary = aggregate_by_candidate(rows)
    selected_summary = aggregate_selected(rows)

    prefix = dataset
    detail_path = args.output_dir / f"{prefix}_state_count_selection_detail.csv"
    summary_path = args.output_dir / f"{prefix}_state_count_selection_summary.csv"
    selected_path = args.output_dir / f"{prefix}_state_count_selection_selected.csv"
    markdown_path = args.output_dir / f"{prefix}_state_count_selection_summary.md"
    figure_path = args.figure_dir / f"{prefix}_state_count_selection.png"

    write_csv(
        detail_path,
        rows,
        [
            "dataset",
            "seed",
            "label_column",
            "candidate_state_count",
            "inferred_state_count",
            "n_windows",
            "train_windows",
            "test_windows",
            "state_improvement",
            "temporal_coherence",
            "min_cluster_fraction",
            "balance_score",
            "selection_score",
            "label_purity_mean",
            "state_counts",
            "state_mapping",
            "selected",
            *METRICS,
        ],
    )
    write_csv(
        summary_path,
        candidate_summary,
        [
            "candidate_state_count",
            "n",
            "selected_count",
            "inferred_state_count_distribution",
            "selection_score_mean",
            "selection_score_std",
            "state_improvement_mean",
            "state_improvement_std",
            "temporal_coherence_mean",
            "temporal_coherence_std",
            "min_cluster_fraction_mean",
            "min_cluster_fraction_std",
            "balance_score_mean",
            "balance_score_std",
            "accuracy_mean",
            "accuracy_std",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
        ],
    )
    write_csv(
        selected_path,
        [selected_summary],
        [
            "n",
            "selected_state_count_distribution",
            "selection_score_mean",
            "selection_score_std",
            "state_improvement_mean",
            "state_improvement_std",
            "temporal_coherence_mean",
            "temporal_coherence_std",
            "min_cluster_fraction_mean",
            "min_cluster_fraction_std",
            "balance_score_mean",
            "balance_score_std",
            "accuracy_mean",
            "accuracy_std",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
        ],
    )
    write_markdown(markdown_path, dataset, args.label_column, candidate_summary, selected_summary, args.selection_policy)
    write_figure(figure_path, candidate_summary)

    print(
        json.dumps(
            {
                "detail_csv": str(detail_path),
                "summary_csv": str(summary_path),
                "selected_csv": str(selected_path),
                "summary_md": str(markdown_path),
                "figure": str(figure_path),
                "selected_state_count_distribution": selected_summary[
                    "selected_state_count_distribution"
                ],
                "selected_balanced_accuracy": selected_summary["balanced_accuracy_mean"],
                "selection_policy": args.selection_policy,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
