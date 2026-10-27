"""Validate the change-point gate on real data with known segment boundaries.

HydraulicSystems: 193 operating-condition changes (cooler, valve, pump,
accumulator, stability) and a long stable run for negative controls.
AReMActivities: seven activity blocks with six internal boundaries.

For every segment we compute the change-point bootstrap p-value and the
current score gate's accept decision.  A true boundary should be accepted; a
stable segment should be rejected.
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

from stcg import STCGStateConfig, stcg_v1_diagnostics
from stcg.change_point_gate import change_point_pvalue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["hydraulic", "arem"], required=True)
    parser.add_argument("--half-window", type=int, default=600)
    parser.add_argument("--draws", type=int, default=99)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0, help="0 uses all boundaries")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    return parser.parse_args()


def load_dataset(name: str):
    if name == "hydraulic":
        raw = pd.read_csv(ROOT / "data/raw/HydraulicSystems.csv")
        labels = pd.read_csv(ROOT / "data/processed/HydraulicSystems_labels.csv")
        signals = ["TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"]
        change_columns = [
            "cooler_condition",
            "valve_condition",
            "internal_pump_leakage",
            "hydraulic_accumulator",
            "stable_flag",
        ]
        x = raw[signals].to_numpy(dtype=float)
        changed = (labels[change_columns].diff().fillna(0) != 0).any(axis=1).to_numpy()
        # Longest constant-label run for negative controls.
        runs = []
        start = 0
        for index in range(1, len(labels) + 1):
            if index == len(labels) or changed[index]:
                runs.append((start, index))
                start = index
        longest = max(runs, key=lambda item: item[1] - item[0])
        return x, changed, longest
    raw = pd.read_csv(ROOT / "data/raw/AReMActivities.csv")
    labels = pd.read_csv(ROOT / "data/processed/AReMActivities_labels.csv")
    signals = [
        "avg_rss12", "var_rss12", "avg_rss13", "var_rss13", "avg_rss23", "var_rss23"
    ]
    x = raw[signals].to_numpy(dtype=float)
    activity = labels["activity"].to_numpy()
    changed = np.r_[False, activity[1:] != activity[:-1]]
    runs = []
    start = 0
    for index in range(1, len(activity) + 1):
        if index == len(activity) or changed[index]:
            runs.append((start, index))
            start = index
    longest = max(runs, key=lambda item: item[1] - item[0])
    return x, changed, longest


def main() -> None:
    args = parse_args()
    x, changed, longest = load_dataset(args.dataset)
    half = args.half_window
    boundaries = [
        int(index)
        for index in np.flatnonzero(changed)
        if index >= half and index + half <= len(x)
    ]
    if args.limit:
        step = max(1, len(boundaries) // args.limit)
        boundaries = boundaries[::step][: args.limit]

    negatives = []
    start, end = longest
    for offset in range(start, end - 2 * half + 1, 2 * half):
        negatives.append(offset + half)
    negatives = negatives[: max(10, len(boundaries) // 4)]

    rows = []
    for kind, indices in (("boundary", boundaries), ("stable", negatives)):
        for center in indices:
            segment = x[center - half:center + half]
            statistic, p_value = change_point_pvalue(
                segment, 3, seed=center, draws=args.draws, step=10
            )
            config = STCGStateConfig(
                n_states=2,
                window_size=300,
                stride=150,
                aggregate="max",
                state_aggregate="mean",
                auto_states=True,
                seed=0,
            )
            diagnostics = stcg_v1_diagnostics(segment, 3, config)
            rows.append(
                {
                    "dataset": args.dataset,
                    "kind": kind,
                    "center": center,
                    "n_rows": len(segment),
                    "change_point_stat": statistic,
                    "change_point_p": p_value,
                    "change_point_accept": bool(p_value < args.alpha),
                    "score_gate_accept": bool(diagnostics.states_accepted),
                    "score_gate_states": int(np.unique(diagnostics.labels).size),
                    "improvement": float(diagnostics.state_improvement),
                    "coherence": float(diagnostics.temporal_coherence),
                }
            )
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail = args.output_dir / f"real_change_point_gate_{args.dataset}.csv"
    frame.to_csv(detail, index=False)
    summary = frame.groupby("kind").agg(
        n=("center", "size"),
        change_point_accept=("change_point_accept", "mean"),
        score_gate_accept=("score_gate_accept", "mean"),
        p_median=("change_point_p", "median"),
        p_q25=("change_point_p", lambda values: float(np.quantile(values, 0.25))),
    )
    summary_path = args.output_dir / f"real_change_point_gate_{args.dataset}_summary.csv"
    summary.to_csv(summary_path)
    print(summary.round(3).to_string())
    print(f"\nwrote {detail} and {summary_path}")


if __name__ == "__main__":
    main()
