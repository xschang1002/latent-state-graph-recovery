"""Run knob-isolation ablations for STCG-v1 on switching-regime synthetic data.

Each heuristic knob flagged in the paper's knob-map table (tab:knob-map) gets a
cheap one-factor-at-a-time empirical isolation on the switching_var generator
with the main-suite data parameters (n=8, T=1000, max_lag=3, seeds 0-19):

- gates: auto-accept gate thresholds (inertia improvement, temporal coherence,
  minimum state share), varied one at a time around defaults.
- aggregators: within-state aggregation (mean/median/p75/max) and cross-state
  aggregation (max/mean/p90).
- features: state-feature construction (non-self edge scores / all edges /
  non-self edges plus per-window mean/std stats).
- windows: window size h in {150, 300, 450} with stride h/2 on periodic
  switching (switch_period=300, matching run_synthetic_switch_difficulty.py).

The default config row is reconciled against the tracked main table
(results/tables/synthetic_recovery_detail.csv, STCG-v1 switching_var rows) so
the default path is provably unchanged. Detail CSVs are upserted per
(config, seed), so interrupted runs resume by re-invoking with the same flags.
"""

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
import numpy as np
from sklearn.metrics import adjusted_rand_score

from stcg import STCGStateConfig, generate_lagged_var, recovery_summary, stcg_v1_diagnostics


METRICS = ["edge_auc", "pair_auc", "precision_at_k", "shd", "lag_accuracy"]
GATE_FIELDS = ["states_accepted", "state_k", "state_improvement", "temporal_coherence", "state_ari"]
GROUPS = ["gates", "aggregators", "features", "windows"]
MAIN_DETAIL_CSV = ROOT / "results" / "tables" / "synthetic_recovery_detail.csv"

# One-factor-at-a-time configs. The "default" row must equal the main-suite
# STCG-v1 config (see run_synthetic_recovery.py): window 300/stride 150,
# aggregate=max, state_aggregate=mean, min_state_improvement=0.12,
# min_temporal_coherence=0.55, min_temporal_state_fraction=0.10,
# state_features=nonself, contrast_mix=0.0, seed offset 30_000.
CONFIGS: dict[str, list[tuple[str, str, str, dict[str, object]]]] = {
    "gates": [
        ("default", "default", "", {}),
        ("inertia=0.06", "inertia improvement", "0.06", {"min_state_improvement": 0.06}),
        ("inertia=0.18", "inertia improvement", "0.18", {"min_state_improvement": 0.18}),
        ("coherence=0.45", "temporal coherence", "0.45", {"min_temporal_coherence": 0.45}),
        ("coherence=0.65", "temporal coherence", "0.65", {"min_temporal_coherence": 0.65}),
        ("share=0.05", "min state share", "0.05", {"min_temporal_state_fraction": 0.05}),
        ("share=0.15", "min state share", "0.15", {"min_temporal_state_fraction": 0.15}),
    ],
    "aggregators": [
        ("default", "default", "", {}),
        ("state=median", "within-state agg", "median", {"state_aggregate": "median"}),
        ("state=p75", "within-state agg", "p75", {"state_aggregate": "p75"}),
        ("state=max", "within-state agg", "max", {"state_aggregate": "max"}),
        ("cross=mean", "cross-state agg", "mean", {"aggregate": "mean"}),
        ("cross=p90", "cross-state agg", "p90", {"aggregate": "p90"}),
    ],
    "features": [
        ("default", "default", "", {}),
        ("features=all", "state features", "all edges", {"state_features": "all"}),
        ("features=+winstats", "state features", "edges+window stats", {"state_features": "nonself_stats"}),
    ],
    "windows": [
        ("h=150", "window size", "150", {"window_size": 150, "stride": 75}),
        ("h=300", "window size", "300", {}),
        ("h=450", "window size", "450", {"window_size": 450, "stride": 225}),
    ],
}

GROUP_TITLES = {
    "gates": "Auto-accept gate thresholds",
    "aggregators": "Within-/cross-state aggregators",
    "features": "State-feature construction",
    "windows": "Window size (stride = h/2, switch period 300)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=[*GROUPS, "all"], default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--edge-prob", type=float, default=0.18)
    parser.add_argument("--noise-scale", type=float, default=0.10)
    parser.add_argument("--lasso-alpha", type=float, default=0.005)
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--window-stride", type=int, default=150)
    parser.add_argument("--switch-period", type=int, default=300,
                        help="Periodic switch period for the windows group; groups 1-3 use the main-suite midpoint switch.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--skip-reconcile", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace, seed: int, overrides: dict[str, object]) -> STCGStateConfig:
    base: dict[str, object] = {
        "n_states": 2,
        "base_model": "lasso",
        "lasso_alpha": args.lasso_alpha,
        "window_size": args.window_size,
        "stride": args.window_stride,
        "aggregate": "max",
        "state_aggregate": "mean",
        "contrast_mix": 0.0,
        "seed": seed + 30_000,
    }
    base.update(overrides)
    return STCGStateConfig(**base)


def window_regimes(regimes: np.ndarray, bounds: tuple[tuple[int, int], ...]) -> np.ndarray:
    labels = []
    for start, end in bounds:
        counts = np.bincount(regimes[start:end].astype(int))
        labels.append(int(np.argmax(counts)))
    return np.asarray(labels, dtype=int)


def load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def upsert_rows(path: Path, rows: list[dict[str, object]]) -> None:
    existing = load_rows(path)
    keys = {(str(row["config"]), int(row["seed"])) for row in rows}
    kept = [
        row for row in existing
        if (str(row["config"]), int(row["seed"])) not in keys
    ]
    fieldnames = ["group", "config", "knob", "value", "seed", *METRICS, *GATE_FIELDS, "true_edges", "edge_density"]
    _common.write_csv(path, [*kept, *rows], fieldnames)


def reconcile_default(args: argparse.Namespace, seed: int, edge_auc_value: float) -> None:
    if args.skip_reconcile or not MAIN_DETAIL_CSV.exists():
        return
    if not hasattr(reconcile_default, "_expected"):
        expected: dict[int, float] = {}
        with MAIN_DETAIL_CSV.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["generator"] == "switching_var" and row["method"] == "STCG-v1":
                    expected[int(row["seed"])] = float(row["edge_auc"])
        reconcile_default._expected = expected  # type: ignore[attr-defined]
    expected = reconcile_default._expected  # type: ignore[attr-defined]
    if seed not in expected:
        return
    if abs(edge_auc_value - expected[seed]) > 1e-12:
        raise AssertionError(
            f"Default config edge_auc {edge_auc_value!r} != main table {expected[seed]!r} (seed {seed})"
        )


def run_group(args: argparse.Namespace, group: str) -> Path:
    rows: list[dict[str, object]] = []
    for config_name, knob, value, overrides in CONFIGS[group]:
        for seed in args.seeds:
            periodic = group == "windows"
            dataset = generate_lagged_var(
                n_nodes=args.n_nodes,
                timesteps=args.timesteps,
                max_lag=args.max_lag,
                edge_prob=args.edge_prob,
                noise_scale=args.noise_scale,
                seed=seed,
                switching=True,
                switch_period=args.switch_period if periodic else None,
            )
            config = build_config(args, seed, overrides)
            diagnostics = stcg_v1_diagnostics(dataset.x, args.max_lag, config=config)
            summary = recovery_summary(diagnostics.scores, dataset.edge_truth)
            truth_window = window_regimes(dataset.regimes, diagnostics.bounds)
            state_ari = float(adjusted_rand_score(truth_window, diagnostics.labels))
            if group != "windows" and config_name == "default":
                reconcile_default(args, seed, float(summary["edge_auc"]))
            row: dict[str, object] = {
                "group": group,
                "config": config_name,
                "knob": knob,
                "value": value,
                "seed": seed,
                "states_accepted": int(diagnostics.states_accepted),
                "state_k": int(np.unique(diagnostics.labels).size),
                "state_improvement": float(diagnostics.state_improvement),
                "temporal_coherence": float(diagnostics.temporal_coherence),
                "state_ari": state_ari,
            }
            row.update(summary)
            rows.append(row)
        # Checkpoint after each config so partial progress survives interrupts.
        detail_path = args.output_dir / f"stcg_knob_ablation_{group}_detail.csv"
        upsert_rows(detail_path, rows)
        print(f"[{group}] finished {config_name}: {len(args.seeds)} seed rows", flush=True)

    detail_path = args.output_dir / f"stcg_knob_ablation_{group}_detail.csv"
    return detail_path


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        key = str(row["config"])
        if key not in groups:
            order.append(key)
        groups[key].append(row)

    summary_rows: list[dict[str, object]] = []
    for config in order:
        items = groups[config]
        out: dict[str, object] = {
            "config": config,
            "knob": str(items[0]["knob"]),
            "value": str(items[0]["value"]),
            "n": len(items),
        }
        for metric in [*METRICS, "state_improvement", "temporal_coherence", "state_ari"]:
            values = [float(item[metric]) for item in items]
            mean = sum(values) / len(values)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = (
                math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
                if len(values) > 1
                else 0.0
            )
        out["accept_rate"] = sum(int(item["states_accepted"]) for item in items) / len(items)
        out["state_k2_rate"] = sum(1 for item in items if int(item["state_k"]) == 2) / len(items)
        summary_rows.append(out)
    return summary_rows


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["config", "knob", "value", "n", "accept_rate", "state_k2_rate"]
    for metric in [*METRICS, "state_improvement", "temporal_coherence", "state_ari"]:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    _common.write_csv(path, rows, fieldnames)


def write_markdown(path: Path, group: str, rows: list[dict[str, object]]) -> None:
    if group == "windows":
        generator_note = (
            "Generator: switching_var with periodic switch period 300 "
            "(n=8, T=1000, max_lag=3), 20 seeds."
        )
    else:
        generator_note = (
            "Generator: switching_var, midpoint switch (n=8, T=1000, max_lag=3), 20 seeds. "
            "The default row reconciles with synthetic_recovery_detail.csv STCG-v1."
        )
    lines = [
        f"# STCG-v1 Knob Ablation: {GROUP_TITLES[group]}",
        "",
        generator_note,
        "",
        "| Config | Knob | N | Edge AUC | Pair AUC | P@K | SHD | Lag Acc | Accept rate | State ARI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {config} | {knob} | {n} | {edge_auc_mean:.3f} +/- {edge_auc_std:.3f} | "
            "{pair_auc_mean:.3f} +/- {pair_auc_std:.3f} | "
            "{precision_at_k_mean:.3f} +/- {precision_at_k_std:.3f} | "
            "{shd_mean:.1f} +/- {shd_std:.1f} | "
            "{lag_accuracy_mean:.3f} +/- {lag_accuracy_std:.3f} | "
            "{accept_rate:.2f} | {state_ari_mean:.3f} +/- {state_ari_std:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_combined_markdown(path: Path, per_group: dict[str, list[dict[str, object]]]) -> None:
    lines = [
        "# STCG-v1 Knob Ablation Summary (switching_var, 20 seeds)",
        "",
        "One-factor-at-a-time isolation of the heuristic knobs mapped in the paper's",
        "knob table (tab:knob-map). Edge AUC is mean +/- std over seeds; Accept rate is",
        "the fraction of seeds where the auto-accept gate fires (K=2 states chosen);",
        "State ARI compares inferred window states with the ground-truth regime.",
        "",
        "| Group | Config | Edge AUC | Accept rate | State ARI |",
        "|---|---|---:|---:|---:|",
    ]
    for group in GROUPS:
        for row in per_group.get(group, []):
            lines.append(
                "| {group} | {config} | {edge_auc_mean:.3f} +/- {edge_auc_std:.3f} | "
                "{accept_rate:.2f} | {state_ari_mean:.3f} +/- {state_ari_std:.3f} |".format(
                    group=group, **row
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(path: Path, per_group: dict[str, list[dict[str, object]]]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15.6, 4.2))
    for ax, group in zip(axes, GROUPS):
        rows = per_group.get(group, [])
        if not rows:
            ax.set_visible(False)
            continue
        configs = [str(row["config"]) for row in rows]
        means = [float(row["edge_auc_mean"]) for row in rows]
        stds = [float(row["edge_auc_std"]) for row in rows]
        aris = [float(row["state_ari_mean"]) for row in rows]
        colors = ["#7a7a7a" if config == "default" else "#1f77b4" for config in configs]
        if group == "windows":
            colors = ["#7a7a7a" if config == "h=300" else "#1f77b4" for config in configs]
        x = np.arange(len(configs))
        ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.85)
        ax.plot(x, aris, "d", color="#d62728", markersize=5, label="State ARI")
        default_mean = next(
            (m for c, m in zip(configs, means) if c in {"default", "h=300"}), None
        )
        if default_mean is not None:
            ax.axhline(default_mean, color="#7a7a7a", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(GROUP_TITLES[group], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Lagged Edge AUC")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("STCG-v1 knob ablations on switching_var (20 seeds); grey = default config", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _common.warn_if_results_tables_output(
        "run_stcg_knob_ablation.py", args.output_dir, len(args.seeds), canonical_seeds=20
    )
    selected = GROUPS if args.group == "all" else [args.group]
    detail_paths: dict[str, Path] = {}
    for group in selected:
        detail_paths[group] = run_group(args, group)

    per_group: dict[str, list[dict[str, object]]] = {}
    payload: dict[str, object] = {"details": {g: str(p) for g, p in detail_paths.items()}}
    for group, detail_path in detail_paths.items():
        rows = load_rows(detail_path)
        summary_rows = aggregate(rows)
        summary_csv = args.output_dir / f"stcg_knob_ablation_{group}_summary.csv"
        summary_md = args.output_dir / f"stcg_knob_ablation_{group}_summary.md"
        write_summary_csv(summary_csv, summary_rows)
        write_markdown(summary_md, group, summary_rows)
        per_group[group] = summary_rows
        payload[f"{group}_summary_csv"] = str(summary_csv)
        payload[f"{group}_summary_md"] = str(summary_md)

    combined_md = args.output_dir / "stcg_knob_ablation_summary.md"
    # A targeted rerun (for example --group features) must retain the other
    # completed groups in the combined report and four-panel figure.
    for group in GROUPS:
        if group not in per_group:
            previous_detail = args.output_dir / f"stcg_knob_ablation_{group}_detail.csv"
            if previous_detail.exists():
                per_group[group] = aggregate(load_rows(previous_detail))
    write_combined_markdown(combined_md, per_group)
    figure_path = args.figure_dir / "stcg_knob_ablation_edge_auc.png"
    write_figure(figure_path, per_group)
    payload["combined_summary_md"] = str(combined_md)
    payload["figure"] = str(figure_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
