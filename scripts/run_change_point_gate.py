"""Evaluate the change-point calibrated gate on short-window regimes.

Development split uses the seeds that exposed the score-gate failure
(main 0--19, gate stress 100--119).  Validation uses fresh seeds 200--219 for
the same four groups.  The rule is frozen before validation: accept the
two-state partition when the VAR change-point bootstrap p-value is below 0.05
(99 residual-bootstrap draws, scan step 10, minimum segment fraction 0.2).
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

from stcg.change_point_gate import change_point_pvalue
from run_revision_gate_stress import CONDITIONS, generate_condition
from stcg import STCGStateConfig, generate_lagged_var, stcg_v1_diagnostics
from stcg.metrics import recovery_summary
from stcg.self_supervised import _aggregate_window_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["development", "validation"], required=True)
    parser.add_argument("--draws", type=int, default=99)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    return parser.parse_args()


def config_for(window: int, stride: int, seed: int, auto_states: bool) -> STCGStateConfig:
    return STCGStateConfig(
        n_states=2,
        window_size=window,
        stride=stride,
        aggregate="max",
        state_aggregate="mean",
        auto_states=auto_states,
        seed=seed,
    )


def evaluate(x, truth, window, stride, seed, draws, alpha):
    forced = stcg_v1_diagnostics(x, 3, config_for(window, stride, seed, auto_states=False))
    default = stcg_v1_diagnostics(x, 3, config_for(window, stride, seed, auto_states=True))
    statistic, p_value = change_point_pvalue(x, 3, seed, draws=draws)
    accept = bool(p_value < alpha)
    one_state = _aggregate_window_scores(forced.window_scores, "max")
    new_scores = forced.scores if accept else one_state
    return {
        "change_point_stat": statistic,
        "change_point_p": p_value,
        "change_point_accept": accept,
        "current_accept": bool(default.states_accepted),
        "n_windows": len(forced.window_scores),
        "improvement": float(forced.state_improvement),
        "coherence": float(forced.temporal_coherence),
        "edge_auc_new": float(recovery_summary(new_scores, truth)["edge_auc"]),
        "edge_auc_current": float(recovery_summary(default.scores, truth)["edge_auc"]),
        "edge_auc_one_state": float(recovery_summary(one_state, truth)["edge_auc"]),
    }


def main() -> None:
    args = parse_args()
    if args.split == "development":
        main_seeds = list(range(20))
        condition_seeds = list(range(100, 120))
    else:
        main_seeds = list(range(200, 220))
        condition_seeds = list(range(200, 220))

    rows = []
    for switching in (False, True):
        for seed in main_seeds:
            dataset = generate_lagged_var(
                n_nodes=8, timesteps=1000, max_lag=3, seed=seed, switching=switching
            )
            row = {
                "group": "main_switching" if switching else "main_stationary",
                "seed": seed,
                "truth_state_count": 2 if switching else 1,
            }
            row.update(evaluate(dataset.x, dataset.edge_truth, 300, 150, seed, args.draws, args.alpha))
            rows.append(row)

    for condition in CONDITIONS:
        if condition.name not in ("stationary_few", "short_switch"):
            continue
        for seed in condition_seeds:
            dataset = generate_condition(condition, seed)
            row = {
                "group": condition.name,
                "seed": seed,
                "truth_state_count": 1 if condition.name == "stationary_few" else 2,
            }
            row.update(
                evaluate(
                    dataset.x,
                    dataset.edge_truth,
                    condition.window,
                    condition.stride,
                    30000 + seed,
                    args.draws,
                    args.alpha,
                )
            )
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame["false_split"] = (frame.truth_state_count == 1) & frame.change_point_accept
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail = args.output_dir / f"change_point_gate_{args.split}.csv"
    frame.to_csv(detail, index=False)
    summary = frame.groupby("group").agg(
        n=("seed", "size"),
        accept_new=("change_point_accept", "mean"),
        accept_current=("current_accept", "mean"),
        false_split=("false_split", "mean"),
        auc_new=("edge_auc_new", "mean"),
        auc_current=("edge_auc_current", "mean"),
        auc_one=("edge_auc_one_state", "mean"),
        p_median=("change_point_p", "median"),
    )
    summary_path = args.output_dir / f"change_point_gate_{args.split}_summary.csv"
    summary.to_csv(summary_path)
    print(summary.round(3).to_string())
    print(f"\nwrote {detail} and {summary_path}")


if __name__ == "__main__":
    main()
