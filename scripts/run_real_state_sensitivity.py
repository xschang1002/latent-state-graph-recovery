"""Run light STCG-v1 state-sensitivity checks against external real labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

# Set before sklearn imports so joblib/loky does not oversubscribe CPU cores
# when KMeans runs on Windows.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _common import (
    align_labels,
    counts_text,
    fmt,
    interpolate_within_segments,
    majority_label,
    parse_columns,
    write_csv,
)
from stcg import STCGStateConfig, chronological_lagged_split, stcg_v1_diagnostics


SENSOR_DEFAULT = ""
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=ROOT / "data" / "raw" / "AReMActivities.csv")
    parser.add_argument("--labels-csv", type=Path, default=ROOT / "data" / "processed" / "AReMActivities_labels.csv")
    parser.add_argument("--dataset-name", type=str, default="AReMActivities")
    parser.add_argument("--label-column", type=str, default="activity")
    parser.add_argument("--time-column", type=str, default="time_index")
    parser.add_argument("--columns", type=str, default=SENSOR_DEFAULT)
    parser.add_argument("--missing", choices=["drop", "interpolate"], default="interpolate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--lasso-alpha", type=float, default=0.005)
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[120, 300])
    parser.add_argument("--state-counts", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    return parser.parse_args()


def load_series(args: argparse.Namespace) -> tuple[np.ndarray, list[str], list[int]]:
    frame = pd.read_csv(args.input_csv)
    columns = parse_columns(args.columns)
    if columns:
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    else:
        numeric = frame.drop(columns=[args.time_column], errors="ignore").select_dtypes(include=[np.number])
        columns = [str(column) for column in numeric.columns]
    if args.missing == "interpolate":
        # Interpolate within the chronological train/test segments so a train
        # gap never borrows test-period values (and vice versa). The boundary
        # mirrors chronological_lagged_split: int(round(n * train_fraction)).
        split_at = int(round(len(numeric) * args.train_fraction))
        numeric = interpolate_within_segments(numeric, [split_at]).dropna(axis=0)
    else:
        numeric = numeric.dropna(axis=0)
    values = numeric.to_numpy(dtype=float)
    row_positions = numeric.index.to_numpy()
    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    row_positions = row_positions[finite]
    if values.shape[1] < 2:
        raise ValueError("Need at least two numeric columns")
    return values, columns, [int(position) for position in row_positions]


def load_labels(path: Path, label_column: str) -> list[str]:
    frame = pd.read_csv(path)
    if label_column not in frame.columns:
        raise ValueError(f"{path} is missing label column: {label_column}")
    return frame[label_column].astype(str).tolist()


def setting_rows(
    args: argparse.Namespace,
    x_train: np.ndarray,
    label_values: list[str],
    seed: int,
    setting_name: str,
    window_size: int,
    n_states: int,
    auto_states: bool,
    min_state_improvement: float,
) -> dict[str, object]:
    config = STCGStateConfig(
        base_model="lasso",
        lasso_alpha=args.lasso_alpha,
        window_size=window_size,
        stride=max(1, window_size // 2),
        n_states=n_states,
        auto_states=auto_states,
        min_state_improvement=min_state_improvement,
        contrast_mix=0.0,
        seed=seed + 110_000,
    )
    diagnostics = stcg_v1_diagnostics(x_train, args.max_lag, config)
    true_labels: list[str] = []
    purities: list[float] = []
    for start, end in diagnostics.bounds:
        label, purity = majority_label(label_values[start:end])
        true_labels.append(label)
        purities.append(purity)

    inferred_states = [int(state) for state in diagnostics.labels]
    _, mapping, accuracy = align_labels(true_labels, inferred_states)
    ari = float(adjusted_rand_score(true_labels, inferred_states))
    return {
        "dataset": args.dataset_name,
        "label_column": args.label_column,
        "setting": setting_name,
        "seed": seed,
        "window_size": window_size,
        "window_stride": max(1, window_size // 2),
        "requested_states": n_states,
        "auto_states": auto_states,
        "min_state_improvement": min_state_improvement,
        "n_windows": len(true_labels),
        "true_state_count": len(set(true_labels)),
        "inferred_state_count": len(set(inferred_states)),
        "adjusted_rand": ari,
        "aligned_accuracy": accuracy,
        "mean_label_purity": float(np.mean(purities)),
        "true_label_counts": counts_text(true_labels),
        "inferred_state_counts": counts_text(inferred_states),
        "state_mapping": json.dumps(mapping, sort_keys=True),
    }


def run_sensitivity(args: argparse.Namespace) -> list[dict[str, object]]:
    values, _, row_positions = load_series(args)
    labels = load_labels(args.labels_csv, args.label_column)
    if row_positions and max(row_positions) >= len(labels):
        raise ValueError("Labels are shorter than the feature matrix")
    aligned_labels = [labels[position] for position in row_positions]
    split = chronological_lagged_split(values, max_lag=args.max_lag, train_fraction=args.train_fraction)
    x_train = values[: split.train_end]
    train_labels = aligned_labels[: split.train_end]

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        rows.append(
            setting_rows(
                args,
                x_train,
                train_labels,
                seed,
                "default_auto_k2_w300",
                300,
                2,
                True,
                0.12,
            )
        )
        for window_size in args.window_sizes:
            for n_states in args.state_counts:
                rows.append(
                    setting_rows(
                        args,
                        x_train,
                        train_labels,
                        seed,
                        f"loose_auto_k{n_states}_w{window_size}",
                        window_size,
                        n_states,
                        True,
                        0.0,
                    )
                )
                rows.append(
                    setting_rows(
                        args,
                        x_train,
                        train_labels,
                        seed,
                        f"forced_k{n_states}_w{window_size}",
                        window_size,
                        n_states,
                        False,
                        0.0,
                    )
                )
    return rows


SUMMARY_FIELDNAMES = [
    "dataset",
    "label_column",
    "setting",
    "seed",
    "window_size",
    "window_stride",
    "requested_states",
    "auto_states",
    "min_state_improvement",
    "n_windows",
    "true_state_count",
    "inferred_state_count",
    "adjusted_rand",
    "aligned_accuracy",
    "mean_label_purity",
    "true_label_counts",
    "inferred_state_counts",
    "state_mapping",
]


def grouped_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["setting"]), []).append(row)
    summaries: list[dict[str, object]] = []
    for setting, items in sorted(groups.items()):
        ari = np.array([float(item["adjusted_rand"]) for item in items], dtype=float)
        acc = np.array([float(item["aligned_accuracy"]) for item in items], dtype=float)
        state_counts = Counter(int(item["inferred_state_count"]) for item in items)
        first = items[0]
        summaries.append(
            {
                "setting": setting,
                "window_size": first["window_size"],
                "requested_states": first["requested_states"],
                "auto_states": first["auto_states"],
                "min_state_improvement": first["min_state_improvement"],
                "ari_mean": float(ari.mean()),
                "ari_std": float(ari.std(ddof=0)),
                "aligned_accuracy_mean": float(acc.mean()),
                "aligned_accuracy_std": float(acc.std(ddof=0)),
                "inferred_state_count_distribution": dict(sorted(state_counts.items())),
            }
        )
    return sorted(summaries, key=lambda row: (-float(row["ari_mean"]), -float(row["aligned_accuracy_mean"]), str(row["setting"])))


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    summaries = grouped_summary(rows)
    best = summaries[0]
    lines = [
        "# Real State Sensitivity",
        "",
        "This diagnostic tests whether real-data single-state collapse is caused by the default auto-state acceptance threshold.",
        "",
        "## Best Setting",
        "",
        "| Setting | ARI | Aligned Acc. | Inferred States |",
        "|---|---:|---:|---|",
        (
            f"| {best['setting']} | {fmt(best['ari_mean'])} +/- {fmt(best['ari_std'])} | "
            f"{fmt(best['aligned_accuracy_mean'])} +/- {fmt(best['aligned_accuracy_std'])} | "
            f"`{best['inferred_state_count_distribution']}` |"
        ),
        "",
        "## All Settings",
        "",
        "| Setting | Window | Requested K | Auto | Min Improvement | ARI | Aligned Acc. | Inferred States |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {setting} | {window_size} | {requested_states} | {auto_states} | {min_imp} | "
            "{ari} +/- {ari_std} | {acc} +/- {acc_std} | `{counts}` |".format(
                setting=row["setting"],
                window_size=row["window_size"],
                requested_states=row["requested_states"],
                auto_states=row["auto_states"],
                min_imp=fmt(row["min_state_improvement"], 2),
                ari=fmt(row["ari_mean"]),
                ari_std=fmt(row["ari_std"]),
                acc=fmt(row["aligned_accuracy_mean"]),
                acc_std=fmt(row["aligned_accuracy_std"]),
                counts=row["inferred_state_count_distribution"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = run_sensitivity(args)
    prefix = args.dataset_name
    summary_csv = args.output_dir / f"{prefix}_state_sensitivity_summary.csv"
    summary_md = args.output_dir / f"{prefix}_state_sensitivity_summary.md"
    write_csv(summary_csv, rows, SUMMARY_FIELDNAMES)
    write_markdown(summary_md, rows)
    print(
        json.dumps(
            {
                "summary_csv": str(summary_csv),
                "summary_md": str(summary_md),
                "n_rows": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
