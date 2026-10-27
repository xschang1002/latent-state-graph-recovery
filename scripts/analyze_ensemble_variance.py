"""Analyze the variance structure of the STCG-v1e ensemble.

For each member partition, flatten the aggregated score tensor and compute the
mean pairwise correlation across members.  Under a simple equicorrelated model,
Var(mean) = sigma^2 (1 + (B-1) rho) / B, so the effective number of independent
members is B / (1 + (B-1) rho).  The script relates rho to the observed Edge AUC
gain on stationary and switching panels.
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
from scipy import stats
from sklearn.cluster import KMeans

from stcg import STCGStateConfig, generate_lagged_var, stcg_v1_diagnostics
from stcg.metrics import recovery_summary
from stcg.self_supervised import _aggregate_window_scores, _state_aggregate


def member_scores(diagnostics, config, seed, members=20):
    features = diagnostics.features
    scaled = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    out = [diagnostics.scores]
    for member in range(1, members):
        labels = KMeans(
            n_clusters=min(config.n_states, scaled.shape[0]),
            n_init=1,
            random_state=seed + 1009 * member,
        ).fit_predict(scaled)
        if np.unique(labels).size <= 1:
            continue
        state_scores = [
            _state_aggregate(diagnostics.window_scores[labels == state], config.state_aggregate)
            for state in sorted(np.unique(labels))
        ]
        out.append(_aggregate_window_scores(state_scores, config.aggregate))
    return np.stack(out, axis=0)


def main() -> None:
    rows = []
    for kind in ("var", "switching_var"):
        for seed in range(20):
            dataset = generate_lagged_var(
                n_nodes=8,
                timesteps=1000,
                max_lag=3,
                seed=seed,
                switching=kind == "switching_var",
            )
            config = STCGStateConfig(
                n_states=2,
                window_size=300,
                stride=150,
                aggregate="max",
                state_aggregate="mean",
                auto_states=True,
                seed=seed,
            )
            diagnostics = stcg_v1_diagnostics(dataset.x, 3, config)
            members = member_scores(diagnostics, config, seed)
            flat = members.reshape(len(members), -1)
            correlations = np.corrcoef(flat)
            upper = correlations[np.triu_indices(len(members), 1)]
            rho = float(np.mean(upper))
            effective = len(members) / (1 + (len(members) - 1) * rho)
            ensemble = flat.mean(axis=0).reshape(diagnostics.scores.shape)
            rows.append(
                {
                    "generator": kind,
                    "seed": seed,
                    "n_members": len(members),
                    "mean_member_correlation": rho,
                    "effective_members": effective,
                    "edge_auc_base": float(
                        recovery_summary(diagnostics.scores, dataset.edge_truth)["edge_auc"]
                    ),
                    "edge_auc_ensemble": float(
                        recovery_summary(ensemble, dataset.edge_truth)["edge_auc"]
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame["gain"] = frame.edge_auc_ensemble - frame.edge_auc_base
    frame.to_csv(ROOT / "results/tables/stcg_ensemble_variance.csv", index=False)
    for kind, group in frame.groupby("generator"):
        r, p = stats.pearsonr(group.mean_member_correlation, group.gain)
        print(
            f"{kind}: rho={group.mean_member_correlation.mean():.3f} "
            f"B_eff={group.effective_members.mean():.1f} "
            f"gain={group.gain.mean():+.4f} corr(rho,gain)={r:+.2f} (p={p:.3f})"
        )


if __name__ == "__main__":
    main()
