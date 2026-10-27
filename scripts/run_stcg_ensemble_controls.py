"""Controls for the STCG-v1e score ensemble.

Three alternatives are compared with STCG-v1 and STCG-v1e on the three main
generators:

* ``bag_global``: a state-free bagged mean over twenty bootstrap resamples of
  the global window-mean score tensor.  It uses no state partition at all.
* ``random_partition``: the ensemble aggregation applied to twenty random
  balanced partitions of the windows.
* ``permuted_kmeans``: the ensemble aggregation applied to K-means partitions
  of column-permuted (hence cross-feature-destroyed) window features.

If the gain of STCG-v1e were generic score smoothing, the state-free control
would reproduce it.  If it required score-structured partitions, random and
permuted partitions would not.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

from stcg import STCGStateConfig, generate_lagged_var, stcg_v1_diagnostics
from stcg.ensemble import ensemble_scores
from stcg.metrics import recovery_summary
from stcg.self_supervised import _aggregate_window_scores, _state_aggregate

KINDS = ["var", "nonlinear_var", "switching_var"]
METHODS = ["STCG-v1", "STCG-v1e", "bag_global", "random_partition", "permuted_kmeans"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument("--members", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    return parser.parse_args()


def _aggregate(diagnostics, labels: np.ndarray, config: STCGStateConfig) -> np.ndarray:
    state_scores = [
        _state_aggregate(diagnostics.window_scores[labels == state], config.state_aggregate)
        for state in sorted(np.unique(labels))
    ]
    return _aggregate_window_scores(state_scores, config.aggregate)


def control_scores(
    diagnostics, config: STCGStateConfig, seed: int, members: int
) -> dict[str, np.ndarray]:
    features = diagnostics.features
    scaled = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    n_windows = scaled.shape[0]
    rng = np.random.default_rng(seed)
    bagged, random_parts, permuted = [], [], []
    for member in range(members):
        index = rng.integers(0, n_windows, n_windows)
        bagged.append(
            _aggregate_window_scores(
                [diagnostics.window_scores[i] for i in index], "mean"
            )
        )
        order = rng.permutation(n_windows)
        labels = np.zeros(n_windows, dtype=int)
        labels[order[: n_windows // 2]] = 1
        random_parts.append(_aggregate(diagnostics, labels, config))
        shuffled = np.empty_like(scaled)
        for column in range(scaled.shape[1]):
            shuffled[:, column] = rng.permutation(scaled[:, column])
        labels = KMeans(
            n_clusters=min(config.n_states, n_windows),
            n_init=1,
            random_state=seed + 31 * member,
        ).fit_predict(shuffled)
        if np.unique(labels).size <= 1:
            permuted.append(diagnostics.scores)
        else:
            permuted.append(_aggregate(diagnostics, labels, config))
    return {
        "bag_global": np.mean(np.stack(bagged, axis=0), axis=0),
        "random_partition": np.mean(np.stack(random_parts, axis=0), axis=0),
        "permuted_kmeans": np.mean(np.stack(permuted, axis=0), axis=0),
    }


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
            scores = {
                "STCG-v1": diagnostics.scores,
                "STCG-v1e": ensemble_scores(diagnostics, config, args.members, seed),
            }
            scores.update(control_scores(diagnostics, config, seed, args.members))
            for method, tensor in scores.items():
                rows.append(
                    {
                        "generator": kind,
                        "seed": seed,
                        "method": method,
                        "edge_auc": float(
                            recovery_summary(tensor, dataset.edge_truth)["edge_auc"]
                        ),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "stcg_ensemble_controls_detail.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import pandas as pd

    frame = pd.DataFrame(rows)
    summary: list[dict[str, object]] = []
    for kind, group in frame.groupby("generator"):
        pivot = group.pivot(index="seed", columns="method", values="edge_auc")
        for method in METHODS:
            row = {
                "generator": kind,
                "method": method,
                "n": int(pivot[method].notna().sum()),
                "edge_auc_mean": float(pivot[method].mean()),
                "edge_auc_std": float(pivot[method].std(ddof=1)),
            }
            if method != "STCG-v1":
                diff = pivot[method] - pivot["STCG-v1"]
                row["delta_vs_v1"] = float(diff.mean())
                row["wins_vs_v1"] = int((diff > 0).sum())
                row["p_t"] = float(stats.ttest_rel(pivot[method], pivot["STCG-v1"]).pvalue)
            summary.append(row)
    summary_path = args.output_dir / "stcg_ensemble_controls_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"wrote {detail_path} and {summary_path}")
    print(pd.DataFrame(summary).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
