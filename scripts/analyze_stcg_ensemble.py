"""Analyze why the STCG score ensemble helps: partition instability vs gain.

For every frozen stress fit, refit ten K-means restarts on the same window
features, measure mean pairwise ARI (instability), and relate it to the
paired Edge AUC gain of STCG-v1e over STCG-v1. Reads the saved ensemble
detail table; regenerates only the frozen stress inputs.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from stcg import STCGStateConfig, generate_lagged_var, stcg_v1_diagnostics


def instability(features: np.ndarray, seed: int, restarts: int = 10) -> float:
    """Partition instability measured with seeds disjoint from the ensemble.

    STCG-v1e members use ``seed + 1009 * member``; the offset below keeps the
    diagnostic restarts independent so that the correlation with the gain is
    not inflated by shared random draws.
    """

    centered = features - features.mean(axis=0, keepdims=True)
    scaled = centered / (centered.std(axis=0, keepdims=True) + 1e-8)
    labels = [
        KMeans(
            n_clusters=2, n_init=1, random_state=seed + 500_000 + 1009 * member
        ).fit_predict(scaled)
        for member in range(restarts)
    ]
    scores = [
        adjusted_rand_score(labels[i], labels[j])
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    ]
    return 1.0 - float(np.mean(scores))


def main() -> None:
    detail = pd.read_csv(ROOT / "results" / "tables" / "revision_specialist_detail.csv")
    detail = detail[(detail.phase == "stress") & (detail.method == "STCG-v1")]
    ensemble = pd.read_csv(ROOT / "results" / "tables" / "stcg_ensemble_stress_detail.csv")
    pivot = ensemble.pivot_table(
        index=["setting_id", "seed"], columns="method", values="edge_auc"
    ).reset_index()
    pivot["gain"] = pivot["STCG-v1e"] - pivot["STCG-v1"]

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
            state_config = STCGStateConfig(
                n_states=2,
                base_model="lasso",
                lasso_alpha=0.005,
                window_size=config["window_size"],
                stride=config["window_stride"],
                aggregate="max",
                seed=seed + int(config["state_seed_offset"]),
            )
            diagnostics = stcg_v1_diagnostics(dataset.x, config["max_lag"], state_config)
            rows.append(
                {
                    "setting_id": setting_id,
                    "seed": seed,
                    "instability": instability(
                        diagnostics.features, seed + int(config["state_seed_offset"])
                    ),
                }
            )

    frame = pd.DataFrame(rows).merge(
        pivot[["setting_id", "seed", "STCG-v1", "STCG-v1e", "gain"]],
        on=["setting_id", "seed"],
        how="inner",
    )
    frame = frame.rename(columns={"STCG-v1": "stcg_v1_edge_auc", "STCG-v1e": "stcg_v1e_edge_auc"})
    r, p = stats.pearsonr(frame["instability"], frame["gain"])
    rho, p_rho = stats.spearmanr(frame["instability"], frame["gain"])
    r_base, p_base = stats.pearsonr(frame["instability"], frame["stcg_v1_edge_auc"])
    design = pd.get_dummies(frame["setting_id"], drop_first=True).astype(float).values
    design = np.column_stack([np.ones(len(frame)), design])

    def residual(values: np.ndarray) -> np.ndarray:
        beta, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
        return values - design @ beta

    r_within, p_within = stats.pearsonr(
        residual(frame["instability"].to_numpy()), residual(frame["gain"].to_numpy())
    )
    bins = pd.qcut(frame["instability"], 5, labels=False, duplicates="drop")
    binned = (
        frame.assign(bin=bins)
        .groupby("bin")
        .agg(
            n=("gain", "size"),
            instability=("instability", "mean"),
            gain=("gain", "mean"),
            gain_se=("gain", lambda values: values.std(ddof=1) / np.sqrt(len(values))),
        )
        .reset_index()
    )
    frame.to_csv(ROOT / "results" / "tables" / "stcg_ensemble_instability.csv", index=False)
    binned.to_csv(
        ROOT / "results" / "tables" / "stcg_ensemble_instability_summary.csv", index=False
    )
    stats_path = ROOT / "results" / "tables" / "stcg_ensemble_instability_stats.csv"
    pd.DataFrame(
        [
            {"statistic": "pearson_r", "value": float(r)},
            {"statistic": "pearson_p", "value": float(p)},
            {"statistic": "spearman_rho", "value": float(rho)},
            {"statistic": "spearman_p", "value": float(p_rho)},
            {"statistic": "baseline_r", "value": float(r_base)},
            {"statistic": "baseline_p", "value": float(p_base)},
            {"statistic": "within_condition_r", "value": float(r_within)},
            {"statistic": "within_condition_p", "value": float(p_within)},
            {"statistic": "stable_quintile_gain", "value": float(binned["gain"].iloc[0])},
            {"statistic": "unstable_quintile_gain", "value": float(binned["gain"].iloc[-1])},
        ]
    ).to_csv(stats_path, index=False)

    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.axhline(0.0, color="0.6", linewidth=0.8)
    axis.scatter(frame["instability"], frame["gain"], s=12, alpha=0.35, color="#4C72B0")
    axis.errorbar(
        binned["instability"],
        binned["gain"],
        yerr=1.96 * binned["gain_se"],
        color="#C44E52",
        marker="o",
        linewidth=1.6,
        capsize=3,
        label="quintile mean $\\pm$ 95% CI",
    )
    axis.set_xlabel("partition instability (1 - mean pairwise ARI)")
    axis.set_ylabel("Edge AUC gain of STCG-v1e")
    axis.set_title(f"r = {r:.2f} (p = {p:.1e}); within-condition r = {r_within:.2f}")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(ROOT / "results" / "figures" / "stcg_ensemble_instability.png", dpi=220)
    print(
        f"n={len(frame)} pearson r={r:.3f} p={p:.1e} spearman rho={rho:.3f} p={p_rho:.1e}"
    )
    print(f"instability vs baseline accuracy r={r_base:.3f} p={p_base:.3f}")
    print(f"within-condition r={r_within:.3f} p={p_within:.3f}")
    print(binned.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
