"""Denoising intervention: does averaging independent noise replicas help?

Freezes one real coefficient tensor and state schedule, re-simulates only the
innovation noise ``B`` times, averages the per-window Lasso scores across the
``B`` replicas (``bar S_w = (1/B) sum_b S_w^(b)``), then runs the unchanged
K-means clustering and computes the state discrimination ``J``, the edge
direction score ``R_edge`` and the changing-edge temporal AUROC
``T-AUC_delta``. If the noise-limited reading of Section ``sec:window-average``
is right, these quantities should improve toward the oracle as ``B`` grows.
This is a mechanism diagnostic, not a recommended deployment (real deployments
do not observe multiple independent replicas of the same mechanism).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stcg.baselines import lasso_var_scores, sliding_window_bounds  # noqa: E402
from stcg.evaluation import (  # noqa: E402
    GraphTruth,
    match_states,
    nonself_mask,
    temporal_delta_auc,
    window_regime_labels,
)
from stcg.mechanism import MechanismConfig, generate_mechanism  # noqa: E402

OUT = ROOT / "results" / "revision2"
CONFIG = ROOT / "configs" / "aggregation_controls.yaml"


def simulate_fixed(coefficients: np.ndarray, regimes: np.ndarray, config: MechanismConfig, noise_seed: int) -> np.ndarray:
    rng = np.random.default_rng(noise_seed)
    total = config.timesteps + config.burn_in
    n_nodes = config.n_nodes
    x = rng.normal(0.0, config.noise_scale, size=(total, n_nodes))
    for t in range(config.max_lag, total):
        state = int(regimes[t])
        value = np.zeros(n_nodes)
        for lag in range(1, config.max_lag + 1):
            value += x[t - lag] @ coefficients[state, lag - 1]
        x[t] = value + rng.normal(0.0, config.noise_scale, size=n_nodes)
    return x[config.burn_in :].copy()


def _nonself_flatten(scores: np.ndarray) -> np.ndarray:
    _, n_nodes, _ = scores.shape
    mask = np.ones_like(scores, dtype=bool)
    for node in range(n_nodes):
        mask[:, node, node] = False
    return scores[mask]


def window_features(window_scores: np.ndarray) -> np.ndarray:
    return np.stack([_nonself_flatten(s) for s in window_scores], axis=0)


def cluster_window_scores(window_scores: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """K-means on cross-window standardized non-self absolute scores."""
    features = window_features(window_scores)
    centred = features - features.mean(axis=0, keepdims=True)
    scaled = centred / (centred.std(axis=0, keepdims=True) + 1e-8)
    model = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(scaled)
    labels = model.labels_
    state_scores = np.stack(
        [window_scores[labels == s].mean(axis=0) for s in np.unique(labels)], axis=0
    )
    return state_scores, labels


def state_discrimination(pred: np.ndarray, truth: np.ndarray, mapping: dict[int, int]) -> float:
    aligned = np.array([mapping.get(int(p), int(p)) for p in pred], dtype=int)
    tpr = float(np.mean(aligned[truth == 1] == 1)) if (truth == 1).any() else 0.0
    fpr = float(np.mean(aligned[truth == 0] == 1)) if (truth == 0).any() else 0.0
    return tpr - fpr


def edge_direction_score(state_scores: np.ndarray, state_support: np.ndarray, mapping: dict[int, int]) -> float:
    n_states = state_scores.shape[0]
    aligned = np.zeros_like(state_scores)
    for pred_state in range(n_states):
        truth_state = mapping.get(pred_state, pred_state)
        if 0 <= truth_state < n_states:
            aligned[truth_state] = state_scores[pred_state]
    lags, nodes, _ = state_scores.shape[1:]
    mask = nonself_mask((lags, nodes, nodes))
    delta = (state_support[0] != state_support[1]) & mask
    edges = np.flatnonzero(delta.reshape(-1))
    d = (state_support[1].astype(int) - state_support[0].astype(int)).reshape(-1)[edges]
    g_flat = aligned.reshape(n_states, -1)[:, edges]
    delta_e = g_flat[1] - g_flat[0]
    return float(np.mean(np.sign(d * delta_e)))


def run_B(frozen, regimes_full, config: MechanismConfig, B: int, max_lag: int, base_seed: int) -> dict:
    bounds = sliding_window_bounds(config.timesteps, config.max_lag, 300, 150)
    replicas = []
    for b in range(B):
        x = simulate_fixed(frozen.coefficients, regimes_full, config, base_seed + b)
        ws = np.stack([lasso_var_scores(x[s:e], max_lag, 0.005) for s, e in bounds], axis=0)
        replicas.append(ws)
    avg = np.mean(replicas, axis=0)

    state_scores, labels = cluster_window_scores(avg, seed=config.seed)
    z_w = window_regime_labels(bounds, frozen.regimes)
    matched = match_states(labels, z_w, n_truth_states=frozen.state_support.shape[0])
    mapping = {int(k): int(v) for k, v in matched["mapping"].items()}
    J = state_discrimination(labels, z_w, mapping)
    R = edge_direction_score(state_scores, frozen.state_support, mapping)
    delta = temporal_delta_auc(state_scores, labels, bounds, frozen.regimes, frozen.state_support)
    return {
        "B": B,
        "J": J,
        "R_edge": R,
        "J_times_R": J * R,
        "t_auc_delta": float(delta["temporal_delta_auroc"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--B-list", nargs="*", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=5000)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    setting = config["settings"]["switching_core"]
    base_cfg = MechanismConfig(**setting, seed=0)
    frozen = generate_mechanism(base_cfg)
    from stcg.mechanism import _regime_schedule

    regimes_full, _ = _regime_schedule(base_cfg, base_cfg.timesteps + base_cfg.burn_in)
    max_lag = int(setting["max_lag"])
    B_list = args.B_list if args.B_list else [1, 2, 4, 8, 16, 32]

    rows = [run_B(frozen, regimes_full, base_cfg, B, max_lag, args.base_seed) for B in B_list]
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "denoising_intervention.csv", index=False)
    print(frame.round(4).to_string(index=False))

    print("\nReading: as B grows (noise variance /B), J, R_edge and T-AUC_delta should")
    print("rise toward the oracle (J=1, T-AUC_delta -> oracle 0.814) if noise is the limiter.")
    print(f"oracle reference: T-AUC_delta ~ 0.814 (J=1), raw window 0.716, K-means 0.649")


if __name__ == "__main__":
    main()
