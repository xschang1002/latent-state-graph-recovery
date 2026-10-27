"""Compare real-data inferred graph states against external window labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _common import align_labels, counts_text, fmt, majority_label, write_csv  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--states-csv",
        type=Path,
        default=ROOT
        / "experiments"
        / "logs"
        / "real_benchmarks"
        / "OccupancySensors"
        / "tables"
        / "downstream_graph_states.csv",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=ROOT / "data" / "processed" / "OccupancySensors_labels.csv",
    )
    parser.add_argument("--dataset-name", type=str, default="OccupancySensors")
    parser.add_argument("--label-column", type=str, default="Occupancy")
    parser.add_argument("--plot-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "results" / "figures")
    return parser.parse_args()


def compare_state_labels(
    states_csv: Path,
    labels_csv: Path,
    dataset_name: str,
    label_column: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    states = pd.read_csv(states_csv)
    labels = pd.read_csv(labels_csv)
    required_state_columns = {"dataset", "seed", "window_id", "start", "end", "center", "inferred_state"}
    missing_state_columns = sorted(required_state_columns - set(states.columns))
    if missing_state_columns:
        raise ValueError(f"{states_csv} is missing columns: {missing_state_columns}")
    if label_column not in labels.columns:
        raise ValueError(f"{labels_csv} is missing label column: {label_column}")

    if dataset_name:
        states = states[states["dataset"].astype(str) == dataset_name]
    if states.empty:
        raise ValueError(f"No state rows found for dataset: {dataset_name}")

    label_values = labels[label_column].dropna().astype(str).tolist()
    if not label_values:
        raise ValueError(f"No usable labels found in {labels_csv}:{label_column}")

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for seed, seed_rows in states.groupby("seed", sort=True):
        seed_rows = seed_rows.sort_values("window_id")
        true_labels: list[str] = []
        purities: list[float] = []
        inferred_states: list[int] = []
        windows: list[tuple[int, int]] = []
        for _, row in seed_rows.iterrows():
            start = int(row["start"])
            end = int(row["end"])
            if start < 0 or end <= start:
                raise ValueError(f"Invalid window [{start}, {end}) in {states_csv}")
            if end > len(label_values):
                raise ValueError(
                    f"Window [{start}, {end}) exceeds label length {len(label_values)} in {labels_csv}"
                )
            true_label, purity = majority_label(label_values[start:end])
            true_labels.append(true_label)
            purities.append(purity)
            inferred_states.append(int(row["inferred_state"]))
            windows.append((start, end))

        aligned, mapping, accuracy = align_labels(true_labels, inferred_states)
        ari = float(adjusted_rand_score(true_labels, inferred_states))
        for row, (start, end), true_label, purity, inferred_state, aligned_label in zip(
            seed_rows.to_dict("records"), windows, true_labels, purities, inferred_states, aligned
        ):
            detail_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": int(seed),
                    "window_id": int(row["window_id"]),
                    "start": start,
                    "end": end,
                    "center": float(row["center"]),
                    "true_label": true_label,
                    "label_purity": purity,
                    "inferred_state": inferred_state,
                    "aligned_state_label": aligned_label,
                    "match": int(str(true_label) == str(aligned_label)),
                }
            )

        summary_rows.append(
            {
                "dataset": dataset_name,
                "seed": int(seed),
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
        )
    return summary_rows, detail_rows


def mean_std(rows: list[dict[str, object]], key: str) -> tuple[float, float]:
    values = np.array([float(row[key]) for row in rows], dtype=float)
    return float(values.mean()), float(values.std(ddof=0))


def write_markdown(path: Path, summary_rows: list[dict[str, object]], detail_rows: list[dict[str, object]]) -> None:
    ari_mean, ari_std = mean_std(summary_rows, "adjusted_rand")
    acc_mean, acc_std = mean_std(summary_rows, "aligned_accuracy")
    purity_mean, purity_std = mean_std(summary_rows, "mean_label_purity")
    inferred_counts = Counter(int(row["inferred_state_count"]) for row in summary_rows)
    true_counts = Counter(int(row["true_state_count"]) for row in summary_rows)

    lines = [
        "# Real State Label Alignment",
        "",
        "This post-hoc diagnostic compares external window-majority labels with graph states inferred without label access.",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Adjusted Rand Index | {ari_mean:.3f} +/- {ari_std:.3f} |",
        f"| Aligned state accuracy | {acc_mean:.3f} +/- {acc_std:.3f} |",
        f"| Mean label purity | {purity_mean:.3f} +/- {purity_std:.3f} |",
        f"| True state count distribution | `{dict(sorted(true_counts.items()))}` |",
        f"| Inferred state count distribution | `{dict(sorted(inferred_counts.items()))}` |",
        "",
        "## By Seed",
        "",
        "| Dataset | Seed | Windows | True States | Inferred States | ARI | Aligned Acc. | Label Purity | True Counts | Inferred Counts | Mapping |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {seed} | {n_windows} | {true_state_count} | {inferred_state_count} | "
            "{ari} | {acc} | {purity} | {true_counts} | {inferred_counts} | `{mapping}` |".format(
                dataset=row["dataset"],
                seed=row["seed"],
                n_windows=row["n_windows"],
                true_state_count=row["true_state_count"],
                inferred_state_count=row["inferred_state_count"],
                ari=fmt(row["adjusted_rand"]),
                acc=fmt(row["aligned_accuracy"]),
                purity=fmt(row["mean_label_purity"]),
                true_counts=row["true_label_counts"],
                inferred_counts=row["inferred_state_counts"],
                mapping=row["state_mapping"],
            )
        )

    mismatch_count = sum(1 for row in detail_rows if int(row["match"]) == 0)
    noncollapsed_seeds = sum(1 for row in summary_rows if int(row["inferred_state_count"]) > 1)
    if noncollapsed_seeds:
        decision = (
            f"- Non-collapsed inferred states: {noncollapsed_seeds} of {len(summary_rows)} seeds; "
            "use ARI and aligned accuracy above to judge whether they match the external labels."
        )
    else:
        decision = "- A single inferred state with ARI near zero is evidence against real graph-state separation under the current settings."
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Window-label mismatches after alignment: {mismatch_count} of {len(detail_rows)}.",
            decision,
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_seed_alignment(path: Path, detail_rows: list[dict[str, object]], plot_seed: int) -> None:
    rows = [row for row in detail_rows if int(row["seed"]) == plot_seed]
    if not rows:
        return
    rows = sorted(rows, key=lambda row: int(row["window_id"]))
    labels = sorted(set(str(row["true_label"]) for row in rows))
    label_index = {label: idx for idx, label in enumerate(labels)}
    true_codes = np.array([label_index[str(row["true_label"])] for row in rows], dtype=float)
    aligned_codes = np.array([label_index[str(row["aligned_state_label"])] for row in rows], dtype=float)
    inferred_states = np.array([int(row["inferred_state"]) for row in rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 2.6), constrained_layout=True)
    panels = [
        ("External majority label", true_codes),
        ("Aligned inferred label", aligned_codes),
        ("Raw inferred state", inferred_states),
    ]
    for ax, (title, values) in zip(axes, panels):
        ax.imshow(values[np.newaxis, :], aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_yticks([])
        ax.set_ylabel(title, rotation=0, ha="right", va="center", labelpad=85)
    axes[-1].set_xlabel("Window index")
    fig.suptitle(f"Real state-label alignment (seed {plot_seed})")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary_rows, detail_rows = compare_state_labels(
        args.states_csv,
        args.labels_csv,
        args.dataset_name,
        args.label_column,
    )

    prefix = args.dataset_name or args.labels_csv.stem
    detail_path = args.output_dir / f"{prefix}_state_label_alignment_detail.csv"
    summary_path = args.output_dir / f"{prefix}_state_label_alignment_summary.csv"
    markdown_path = args.output_dir / f"{prefix}_state_label_alignment_summary.md"
    figure_path = args.figure_dir / f"{prefix}_state_label_alignment_seed{args.plot_seed}.png"

    write_csv(
        summary_path,
        summary_rows,
        [
            "dataset",
            "seed",
            "n_windows",
            "true_state_count",
            "inferred_state_count",
            "adjusted_rand",
            "aligned_accuracy",
            "mean_label_purity",
            "true_label_counts",
            "inferred_state_counts",
            "state_mapping",
        ],
    )
    write_csv(
        detail_path,
        detail_rows,
        [
            "dataset",
            "seed",
            "window_id",
            "start",
            "end",
            "center",
            "true_label",
            "label_purity",
            "inferred_state",
            "aligned_state_label",
            "match",
        ],
    )
    write_markdown(markdown_path, summary_rows, detail_rows)
    plot_seed_alignment(figure_path, detail_rows, args.plot_seed)

    print(
        json.dumps(
            {
                "detail_csv": str(detail_path),
                "summary_csv": str(summary_path),
                "summary_md": str(markdown_path),
                "figure": str(figure_path),
                "n_detail_rows": len(detail_rows),
                "n_summary_rows": len(summary_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
