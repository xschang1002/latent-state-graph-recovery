"""Evaluate variance-reduced STCG-v1 with K-means score ensembling.

STCG-v1e averages the aggregated score tensors produced by independent
K-means restarts. The primary ``n_init=10`` partition and the acceptance gate
are unchanged, so ``--ensemble-size 1`` reproduces STCG-v1 exactly. The script
covers the three-generator main panel and the frozen specialist stress grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
from scipy import stats

from stcg import STCGStateConfig, generate_lagged_var, stcg_v1_diagnostics
from stcg.ensemble import ensemble_scores
from stcg.metrics import recovery_summary

METRICS = ["edge_auc", "pair_auc", "precision_at_k", "shd", "lag_accuracy"]
DEFAULT_KINDS = ["var", "nonlinear_var", "switching_var"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["main", "stress"], default="main")
    parser.add_argument("--kinds", nargs="+", choices=DEFAULT_KINDS, default=DEFAULT_KINDS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--edge-prob", type=float, default=0.18)
    parser.add_argument("--noise-scale", type=float, default=0.10)
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--window-stride", type=int, default=150)
    parser.add_argument("--ensemble-size", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    return parser.parse_args()


def state_config(args: argparse.Namespace, seed: int) -> STCGStateConfig:
    return STCGStateConfig(
        n_states=2,
        base_model="lasso",
        lasso_alpha=0.005,
        window_size=args.window_size,
        stride=args.window_stride,
        aggregate="max",
        state_aggregate="mean",
        auto_states=True,
        contrast_mix=0.0,
        seed=seed,
    )


def metrics(scores: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    summary = recovery_summary(scores, truth)
    return {key: float(summary[key]) for key in METRICS}


def main_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kind in args.kinds:
        for seed in args.seeds:
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
            base_config = state_config(args, seed)
            diagnostics = stcg_v1_diagnostics(dataset.x, args.max_lag, base_config)
            for label, size in (("STCG-v1", 1), ("STCG-v1e", args.ensemble_size)):
                scores = ensemble_scores(diagnostics, base_config, size, seed) if size > 1 else diagnostics.scores
                row: dict[str, object] = {
                    "phase": "main",
                    "generator": kind,
                    "seed": seed,
                    "method": label,
                    "ensemble_size": size,
                    "inferred_state_count": int(np.unique(diagnostics.labels).size),
                    "state_improvement": float(diagnostics.state_improvement),
                    "temporal_coherence": float(diagnostics.temporal_coherence),
                    "states_accepted": bool(diagnostics.states_accepted),
                }
                row.update(metrics(scores, dataset.edge_truth))
                rows.append(row)
    return rows


def stress_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    detail_path = ROOT / "results" / "tables" / "revision_specialist_detail.csv"
    detail = pd.read_csv(detail_path)
    detail = detail[(detail.phase == "stress") & (detail.method == "STCG-v1")]
    rows: list[dict[str, object]] = []
    for setting_id, group in detail.groupby("setting_id"):
        config = json.loads(group.config.iloc[0])
        for seed in sorted(int(value) for value in group.seed.unique()):
            dataset = generate_lagged_var(
                n_nodes=config["n_nodes"],
                timesteps=config["timesteps"],
                max_lag=config["max_lag"],
                edge_prob=config["edge_prob"],
                noise_scale=config["noise_scale"],
                seed=seed,
                switching=True,
                switch_period=config["switch_period"],
            )
            ratio = float(config["observation_noise"])
            if ratio > 0:
                rng = np.random.default_rng(seed + 60000)
                dataset = replace(
                    dataset,
                    x=dataset.x
                    + rng.normal(0.0, ratio * float(np.std(dataset.x)), size=dataset.x.shape),
                )
            base_config = STCGStateConfig(
                n_states=2,
                base_model="lasso",
                lasso_alpha=0.005,
                window_size=config["window_size"],
                stride=config["window_stride"],
                aggregate="max",
                state_aggregate="mean",
                auto_states=True,
                contrast_mix=0.0,
                seed=seed + int(config["state_seed_offset"]),
            )
            diagnostics = stcg_v1_diagnostics(dataset.x, config["max_lag"], base_config)
            for label, size in (("STCG-v1", 1), ("STCG-v1e", args.ensemble_size)):
                scores = (
                    diagnostics.scores
                    if size == 1
                    else ensemble_scores(
                        diagnostics, base_config, size, seed + int(config["state_seed_offset"])
                    )
                )
                row = {
                    "phase": "stress",
                    "generator": "switching_var",
                    "setting_id": setting_id,
                    "seed": seed,
                    "method": label,
                    "ensemble_size": size,
                    "inferred_state_count": int(np.unique(diagnostics.labels).size),
                    "state_improvement": float(diagnostics.state_improvement),
                    "temporal_coherence": float(diagnostics.temporal_coherence),
                    "states_accepted": bool(diagnostics.states_accepted),
                    "switch_period": config["switch_period"],
                    "timesteps": config["timesteps"],
                    "observation_noise": ratio,
                    "edge_prob": config["edge_prob"],
                }
                row.update(metrics(scores, dataset.edge_truth))
                rows.append(row)
    return rows


def summarize(rows: list[dict[str, object]], keys: list[str]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(keys + ["method"], dropna=False)
    out: list[dict[str, object]] = []
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(keys + ["method"], group_key))
        row["n"] = int(len(group))
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        out.append(row)
    return out


def paired_summary(
    rows: list[dict[str, object]], unit_keys: list[str], group_keys: list[str]
) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    base = frame[frame.method == "STCG-v1"].set_index(unit_keys)
    variant = frame[frame.method == "STCG-v1e"].set_index(unit_keys)
    joined = base.join(variant, lsuffix="_base", rsuffix="_ens", how="inner")
    out: list[dict[str, object]] = []
    for group_key, group in joined.groupby(level=list(range(len(group_keys)))):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_keys, group_key))
        row["n"] = int(len(group))
        for metric in METRICS:
            diff = group[f"{metric}_ens"] - group[f"{metric}_base"]
            if metric == "shd":
                diff = -diff
            row[f"{metric}_improvement_mean"] = float(diff.mean())
            row[f"{metric}_improvement_std"] = (
                float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
            )
            row[f"{metric}_wins"] = int((diff > 0).sum())
            row[f"{metric}_p_t"] = float(stats.ttest_rel(
                group[f"{metric}_ens"], group[f"{metric}_base"]
            ).pvalue)
            try:
                row[f"{metric}_p_w"] = float(stats.wilcoxon(diff).pvalue)
            except ValueError:
                row[f"{metric}_p_w"] = float("nan")
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.ensemble_size < 1:
        raise ValueError("--ensemble-size must be positive")
    rows = stress_rows(args) if args.mode == "stress" else main_rows(args)
    if args.mode == "stress":
        frame = pd.DataFrame(rows)
        recomputed = frame[frame.method == "STCG-v1"].set_index(["setting_id", "seed"])["edge_auc"]
        frozen = pd.read_csv(ROOT / "results" / "tables" / "revision_specialist_detail.csv")
        frozen = frozen[(frozen.phase == "stress") & (frozen.method == "STCG-v1")]
        frozen = frozen.set_index(["setting_id", "seed"])["edge_auc"]
        joined = recomputed.to_frame("new").join(frozen.rename("frozen"), how="inner")
        delta = float((joined["new"] - joined["frozen"]).abs().max())
        if len(joined) != len(recomputed) or delta > 1e-9:
            raise SystemExit(
                f"STCG-v1 rows do not reproduce the frozen detail table: "
                f"matched={len(joined)}/{len(recomputed)} max_abs_delta={delta:.3e}"
            )
        print(f"STCG-v1 reproduces the frozen detail table (max delta {delta:.1e})")
    keys = ["generator"] if args.mode == "main" else ["setting_id"]
    unit_keys = keys + ["seed"]
    prefix = "stcg_ensemble" if args.mode == "main" else "stcg_ensemble_stress"
    output = args.output_dir
    write_csv(output / f"{prefix}_detail.csv", rows)
    write_csv(output / f"{prefix}_summary.csv", summarize(rows, keys))
    write_csv(output / f"{prefix}_paired.csv", paired_summary(rows, unit_keys, keys))
    print(f"wrote {len(rows)} rows to {output / (prefix + '_detail.csv')}")
    frame = pd.DataFrame(rows)
    for method, group in frame.groupby("method"):
        print(f"{method}: Edge AUC {group.edge_auc.mean():.4f} (n={len(group)})")


if __name__ == "__main__":
    main()
