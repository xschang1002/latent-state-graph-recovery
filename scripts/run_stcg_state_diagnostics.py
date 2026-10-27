"""Run STCG-v1 latent state diagnostics on switching synthetic graphs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
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
from matplotlib.patches import Patch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from stcg import STCGStateConfig, generate_lagged_var, recovery_summary, stcg_v1_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--plot-seed", type=int, default=0)
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


def config_for(args: argparse.Namespace, seed: int) -> STCGStateConfig:
    return STCGStateConfig(
        base_model="lasso",
        lasso_alpha=args.lasso_alpha,
        window_size=args.window_size,
        stride=args.window_stride,
        aggregate=args.window_aggregate,
        state_aggregate=args.state_aggregate,
        contrast_mix=0.0,
        seed=seed + 40_000,
    )


def majority_label(labels: np.ndarray) -> tuple[int, float]:
    counts = np.bincount(labels.astype(int))
    majority = int(np.argmax(counts))
    purity = float(counts[majority] / labels.size)
    return majority, purity


def align_labels(true_labels: np.ndarray, predicted_labels: np.ndarray) -> tuple[np.ndarray, dict[int, int], float]:
    true_values = sorted(int(value) for value in np.unique(true_labels))
    pred_values = sorted(int(value) for value in np.unique(predicted_labels))
    matrix = np.zeros((len(true_values), len(pred_values)), dtype=int)
    true_index = {value: idx for idx, value in enumerate(true_values)}
    pred_index = {value: idx for idx, value in enumerate(pred_values)}

    for true_value, pred_value in zip(true_labels, predicted_labels):
        matrix[true_index[int(true_value)], pred_index[int(pred_value)]] += 1

    row_ind, col_ind = linear_sum_assignment(-matrix)
    mapping = {pred_values[col]: true_values[row] for row, col in zip(row_ind, col_ind)}
    fallback = true_values[0] if true_values else 0
    aligned = np.array([mapping.get(int(label), fallback) for label in predicted_labels], dtype=int)
    accuracy = float((aligned == true_labels).mean()) if true_labels.size else math.nan
    return aligned, mapping, accuracy


def generate_dataset(args: argparse.Namespace, seed: int):
    return generate_lagged_var(
        n_nodes=args.n_nodes,
        timesteps=args.timesteps,
        max_lag=args.max_lag,
        edge_prob=args.edge_prob,
        noise_scale=args.noise_scale,
        seed=seed,
        switching=True,
    )


def run_one(args: argparse.Namespace, seed: int) -> tuple[dict[str, object], list[dict[str, object]], object, np.ndarray]:
    dataset = generate_dataset(args, seed)
    diagnostics = stcg_v1_diagnostics(dataset.x, args.max_lag, config_for(args, seed))
    true_window = []
    purities = []

    for start, end in diagnostics.bounds:
        label, purity = majority_label(dataset.regimes[start:end])
        true_window.append(label)
        purities.append(purity)

    true_window_arr = np.array(true_window, dtype=int)
    aligned, mapping, accuracy = align_labels(true_window_arr, diagnostics.labels)
    ari = float(adjusted_rand_score(true_window_arr, diagnostics.labels))
    recovery = recovery_summary(diagnostics.scores, dataset.edge_truth)

    detail_rows: list[dict[str, object]] = []
    for window_id, ((start, end), true_label, purity, inferred, aligned_label) in enumerate(
        zip(diagnostics.bounds, true_window_arr, purities, diagnostics.labels, aligned)
    ):
        detail_rows.append(
            {
                "seed": seed,
                "window_id": window_id,
                "start": start,
                "end": end,
                "center": (start + end) / 2,
                "true_regime": int(true_label),
                "regime_purity": purity,
                "inferred_state": int(inferred),
                "aligned_state": int(aligned_label),
                "match": int(true_label == aligned_label),
            }
        )

    summary_row: dict[str, object] = {
        "seed": seed,
        "n_windows": len(diagnostics.bounds),
        "state_count": int(np.unique(diagnostics.labels).size),
        "adjusted_rand": ari,
        "aligned_accuracy": accuracy,
        "mean_regime_purity": float(np.mean(purities)),
        "state_mapping": json.dumps(mapping, sort_keys=True),
    }
    summary_row.update(recovery)
    return summary_row, detail_rows, diagnostics, true_window_arr


def write_detail(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "seed",
        "window_id",
        "start",
        "end",
        "center",
        "true_regime",
        "regime_purity",
        "inferred_state",
        "aligned_state",
        "match",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "seed",
        "n_windows",
        "state_count",
        "adjusted_rand",
        "aligned_accuracy",
        "mean_regime_purity",
        "edge_auc",
        "pair_auc",
        "precision_at_k",
        "shd",
        "lag_accuracy",
        "true_edges",
        "edge_density",
        "state_mapping",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(rows: list[dict[str, object]], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in rows]
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    ari_mean, ari_std = mean_std(rows, "adjusted_rand")
    acc_mean, acc_std = mean_std(rows, "aligned_accuracy")
    edge_mean, edge_std = mean_std(rows, "edge_auc")
    pk_mean, pk_std = mean_std(rows, "precision_at_k")
    shd_mean, shd_std = mean_std(rows, "shd")
    state_counts = Counter(int(row["state_count"]) for row in rows)

    lines = [
        "# STCG-v1 State Diagnostics Summary",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Adjusted Rand Index | {ari_mean:.3f} +/- {ari_std:.3f} |",
        f"| Aligned state accuracy | {acc_mean:.3f} +/- {acc_std:.3f} |",
        f"| Edge AUC | {edge_mean:.3f} +/- {edge_std:.3f} |",
        f"| Precision@K | {pk_mean:.3f} +/- {pk_std:.3f} |",
        f"| SHD | {shd_mean:.1f} +/- {shd_std:.1f} |",
        f"| State count distribution | {dict(sorted(state_counts.items()))} |",
        "",
        "## Per Seed",
        "",
        "| Seed | States | ARI | Aligned Acc. | Edge AUC | Precision@K | SHD | Mapping |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {state_count} | {adjusted_rand:.3f} | {aligned_accuracy:.3f} | "
            "{edge_auc:.3f} | {precision_at_k:.3f} | {shd:.1f} | `{state_mapping}` |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def contiguous_segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    if labels.size == 0:
        return []
    changes = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = [0, *changes.tolist()]
    ends = [*changes.tolist(), labels.size]
    return [(start, end, int(labels[start])) for start, end in zip(starts, ends)]


def write_figure(
    path: Path,
    dataset,
    diagnostics,
    true_window: np.ndarray,
    aligned: np.ndarray,
    title_seed: int,
) -> None:
    state_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b"]
    regime_colors = ["#f2f2f2", "#d8e8ff"]
    fig = plt.figure(figsize=(10.5, 7.8))
    grid = fig.add_gridspec(3, 2, height_ratios=[2.35, 0.95, 1.55], width_ratios=[1.0, 1.0])
    timeline_ax = fig.add_subplot(grid[0, :])
    tile_ax = fig.add_subplot(grid[1, :])
    confusion_ax = fig.add_subplot(grid[2, 0])
    summary_ax = fig.add_subplot(grid[2, 1])

    ax = timeline_ax
    for start, end, regime in contiguous_segments(dataset.regimes.astype(int)):
        ax.axvspan(start, end, color=regime_colors[regime % len(regime_colors)], alpha=0.9)
    for window_id, ((start, end), state) in enumerate(zip(diagnostics.bounds, diagnostics.labels)):
        color = state_colors[int(state) % len(state_colors)]
        ax.broken_barh([(start, end - start)], (window_id - 0.35, 0.7), facecolors=color, edgecolors="black", linewidth=0.7)
        ax.text((start + end) / 2, window_id, str(int(state)), ha="center", va="center", color="white", fontsize=9)
    ax.set_xlim(0, dataset.x.shape[0])
    ax.set_ylim(-0.7, len(diagnostics.bounds) - 0.3)
    ax.set_ylabel("Window")
    ax.set_xlabel("Time index")
    ax.set_title(f"STCG-v1 inferred graph states over switching regimes (seed {title_seed})")
    ax.set_yticks(range(len(diagnostics.bounds)))
    ax.grid(axis="x", alpha=0.18)
    legend_items = [
        Patch(facecolor=regime_colors[0], edgecolor="none", label="True regime 0"),
        Patch(facecolor=regime_colors[1], edgecolor="none", label="True regime 1"),
    ]
    for state in sorted(int(value) for value in np.unique(diagnostics.labels)):
        legend_items.append(Patch(facecolor=state_colors[state % len(state_colors)], edgecolor="black", label=f"Inferred state {state}"))
    ax.legend(handles=legend_items, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))

    ax = tile_ax
    matrix = np.vstack([true_window, aligned])
    cmap = matplotlib.colors.ListedColormap(["#f2f2f2", "#d8e8ff"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks([0, 1], labels=["True majority", "Aligned inferred"])
    ax.set_xticks(range(len(diagnostics.bounds)), labels=[str(idx) for idx in range(len(diagnostics.bounds))])
    ax.set_xlabel("Window index")
    for y in range(matrix.shape[0]):
        for x_pos in range(matrix.shape[1]):
            ax.text(x_pos, y, str(int(matrix[y, x_pos])), ha="center", va="center", fontsize=9)

    ax = confusion_ax
    confusion = np.zeros((2, 2), dtype=int)
    for true_label, aligned_label in zip(true_window, aligned):
        if int(true_label) < 2 and int(aligned_label) < 2:
            confusion[int(true_label), int(aligned_label)] += 1
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks([0, 1], labels=["State 0", "State 1"])
    ax.set_yticks([0, 1], labels=["Regime 0", "Regime 1"])
    ax.set_xlabel("Aligned inferred state")
    ax.set_ylabel("True majority regime")
    ax.set_title("Window-level alignment confusion")
    for y in range(confusion.shape[0]):
        for x_pos in range(confusion.shape[1]):
            value = confusion[y, x_pos]
            color = "white" if value > confusion.max() / 2 else "black"
            ax.text(x_pos, y, str(value), ha="center", va="center", color=color, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    accuracy = float((aligned == true_window).mean()) if true_window.size else math.nan
    ari = float(adjusted_rand_score(true_window, diagnostics.labels)) if true_window.size else math.nan
    boundary_windows = int(sum(row_purity < 1.0 for row_purity in [
        majority_label(dataset.regimes[start:end])[1] for start, end in diagnostics.bounds
    ]))
    summary_ax.axis("off")
    summary_lines = [
        "Alignment summary",
        f"Adjusted Rand Index: {ari:.3f}",
        f"Aligned accuracy: {accuracy:.3f}",
        f"Inferred states: {np.unique(diagnostics.labels).size}",
        f"Windows crossing switch: {boundary_windows}",
        "",
        "State labels are assigned without",
        "access to synthetic regime labels.",
    ]
    summary_ax.text(0.02, 0.92, "\n".join(summary_lines), va="top", ha="left", fontsize=11)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _common.warn_if_results_tables_output(
        "run_stcg_state_diagnostics.py", args.output_dir, len(args.seeds), canonical_seeds=5
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    plot_payload = None

    for seed in args.seeds:
        summary, detail, diagnostics, true_window = run_one(args, seed)
        summary_rows.append(summary)
        detail_rows.extend(detail)
        if seed == args.plot_seed:
            aligned, _, _ = align_labels(true_window, diagnostics.labels)
            plot_payload = (generate_dataset(args, seed), diagnostics, true_window, aligned)

    detail_path = args.output_dir / "stcg_v1_state_diagnostics_detail.csv"
    summary_path = args.output_dir / "stcg_v1_state_diagnostics_summary.csv"
    markdown_path = args.output_dir / "stcg_v1_state_diagnostics_summary.md"
    figure_path = args.figure_dir / f"stcg_v1_state_diagnostics_seed{args.plot_seed}.png"

    write_detail(detail_path, detail_rows)
    write_summary(summary_path, summary_rows)
    write_markdown(markdown_path, summary_rows)
    if plot_payload is not None:
        write_figure(figure_path, *plot_payload, title_seed=args.plot_seed)

    print(
        json.dumps(
            {
                "detail_csv": str(detail_path),
                "summary_csv": str(summary_path),
                "summary_md": str(markdown_path),
                "figure": str(figure_path) if plot_payload is not None else None,
                "n_detail_rows": len(detail_rows),
                "n_summary_rows": len(summary_rows),
                "mean_adjusted_rand": mean_std(summary_rows, "adjusted_rand")[0],
                "mean_aligned_accuracy": mean_std(summary_rows, "aligned_accuracy")[0],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
