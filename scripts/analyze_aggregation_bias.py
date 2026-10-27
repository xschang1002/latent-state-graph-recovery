"""Toy analysis of within-/cross-state aggregation.

Two states, independent noise, sparse true edges.  ``shared`` means the edge
is present in both states (a stationary graph); ``state_specific`` means it is
present in one state only.  Mean aggregation is compared with max aggregation,
and the null-edge inflation of the max is reported.
"""

from __future__ import annotations

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
from sklearn.metrics import roc_auc_score


def main() -> None:
    rng = np.random.default_rng(0)
    edges, fraction, repetitions, signal, sigma = 2000, 0.05, 200, 1.5, 1.0
    rows = []
    for mode in ("shared", "state_specific"):
        mean_auc, max_auc = [], []
        for _ in range(repetitions):
            truth = (rng.random(edges) < fraction).astype(int)
            noise_a = rng.normal(0, sigma, edges)
            noise_b = rng.normal(0, sigma, edges)
            state_a = signal * truth + noise_a
            state_b = (
                signal * truth + noise_b if mode == "shared" else noise_b
            )
            mean_auc.append(roc_auc_score(truth, (state_a + state_b) / 2))
            max_auc.append(roc_auc_score(truth, np.maximum(state_a, state_b)))
        rows.append(
            {
                "mode": mode,
                "mean_stat": float(np.mean(mean_auc)),
                "max_stat": float(np.mean(max_auc)),
                "difference": float(np.mean(max_auc) - np.mean(mean_auc)),
            }
        )
    null_a = rng.normal(0, 1, 200_000)
    null_b = rng.normal(0, 1, 200_000)
    rows.append(
        {
            "mode": "null_edge",
            "mean_stat": float(np.mean((null_a + null_b) / 2)),
            "max_stat": float(np.mean(np.maximum(null_a, null_b))),
            "difference": float(
                np.mean(np.maximum(null_a, null_b)) - np.mean((null_a + null_b) / 2)
            ),
        }
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results/tables/aggregation_bias.csv", index=False)
    print(frame.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
