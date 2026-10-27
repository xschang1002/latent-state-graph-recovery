"""Sensitivity of STCG-v1e to the number of K-means partitions.

Runs the three main generators over the twenty canonical seeds for a small grid
of ensemble sizes.  Window scoring, the supplied state count, and the gate are
unchanged; only ``B`` varies.  The primary paper setting is ``B=20``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
from scipy import stats

from stcg import STCGStateConfig, generate_lagged_var, stcg_v1_diagnostics
from stcg.ensemble import ensemble_scores
from stcg.metrics import recovery_summary

KINDS = ["var", "nonlinear_var", "switching_var"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument("--members", nargs="+", type=int, default=[5, 10, 20, 40])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for kind in KINDS:
        for seed in args.seeds:
            dataset = generate_lagged_var(
                n_nodes=8,
                timesteps=1000,
                max_lag=3,
                edge_prob=0.18,
                noise_scale=0.10,
                seed=seed,
                nonlinear=kind == "nonlinear_var",
                switching=kind == "switching_var",
            )
            config = STCGStateConfig(
                n_states=2,
                base_model="lasso",
                lasso_alpha=0.005,
                window_size=300,
                stride=150,
                aggregate="max",
                state_aggregate="mean",
                auto_states=True,
                contrast_mix=0.0,
                seed=seed,
            )
            diagnostics = stcg_v1_diagnostics(dataset.x, 3, config)
            base = float(recovery_summary(diagnostics.scores, dataset.edge_truth)["edge_auc"])
            rows.append(
                {"generator": kind, "seed": seed, "members": 1, "edge_auc": base}
            )
            for members in args.members:
                scores = ensemble_scores(diagnostics, config, members, seed)
                rows.append(
                    {
                        "generator": kind,
                        "seed": seed,
                        "members": members,
                        "edge_auc": float(
                            recovery_summary(scores, dataset.edge_truth)["edge_auc"]
                        ),
                    }
                )

    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "stcg_ensemble_b_sensitivity.csv", index=False)
    summary: list[dict[str, object]] = []
    for (kind, members), group in frame.groupby(["generator", "members"]):
        row = {
            "generator": kind,
            "members": int(members),
            "n": int(len(group)),
            "edge_auc_mean": float(group["edge_auc"].mean()),
            "edge_auc_std": float(group["edge_auc"].std(ddof=1)),
        }
        if members > 1:
            base = frame[(frame.generator == kind) & (frame.members == 1)].set_index("seed")
            variant = group.set_index("seed")
            diff = variant["edge_auc"] - base["edge_auc"]
            row["delta_vs_base"] = float(diff.mean())
            row["wins_vs_base"] = int((diff > 0).sum())
            row["p_t"] = float(stats.ttest_rel(variant["edge_auc"], base["edge_auc"]).pvalue)
        summary.append(row)
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(
        args.output_dir / "stcg_ensemble_b_sensitivity_summary.csv", index=False
    )
    print(summary_frame.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
