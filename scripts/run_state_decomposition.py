"""State decomposition: J x R_edge identity and the pooled same/cross split.

For a two-state hard-assignment predictor that emits one fixed score map per
state, ``G_{z_hat, e}``, the changing-edge temporal AUROC admits the exact
identity (reviewer-derived, verified here to float precision)

    T-AUC_delta = 1/2 + 1/2 * J * R_edge,

where ``J = Pr(z_hat=1 | z=1) - Pr(z_hat=1 | z=0)`` is the state-discrimination
(true-positive minus false-positive rate under a fixed label matching) and
``R_edge = mean_e sign(d_e * delta_e)`` with ``d_e = A_{1,e} - A_{0,e}`` the true
direction and ``delta_e = G_{1,e} - G_{0,e}`` the predicted direction.  Swapping
the predicted state numbering flips both ``J`` and ``R_edge``, leaving the
product invariant.

The pooled temporal AUROC additionally splits exactly into a same-edge and a
cross-edge term,

    AUC_pooled = w * AUC_same + (1 - w) * AUC_cross,   w = sum_e n_e^+ n_e^- / ((sum n_e^+) (sum n_e^-)),

which turns "the pooled metric hides time-localisation" from an interpretation
into a checkable contribution table.

Reported per seed (not just as pooled means), because E[J R_edge] != E[J] E[R_edge].
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stcg import STCGStateConfig, stcg_v1_diagnostics  # noqa: E402
from stcg.evaluation import (  # noqa: E402
    GraphTruth,
    match_states,
    nonself_mask,
    temporal_delta_auc,
    temporal_graph_metrics,
    window_regime_labels,
)
from stcg.mechanism import MechanismConfig, generate_mechanism  # noqa: E402

OUT = ROOT / "results" / "revision2"
CONFIG = ROOT / "configs" / "aggregation_controls.yaml"


def _edge_directions(state_support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (d_e, edge_indices) for changing edges, d_e = A[1,e] - A[0,e] in {-1, +1}."""
    lags, nodes, _ = state_support.shape[1:]
    mask = nonself_mask((lags, nodes, nodes))
    delta = state_support[0] != state_support[1]
    delta &= mask
    edges = np.flatnonzero(delta.reshape(-1))
    d = (state_support[1].astype(int) - state_support[0].astype(int)).reshape(-1)[edges]
    return d.astype(int), edges


def state_discrimination(pred: np.ndarray, truth: np.ndarray, mapping: dict[int, int]) -> float:
    """J = Pr(z_hat_aligned=1 | z=1) - Pr(z_hat_aligned=1 | z=0)."""
    aligned = np.array([mapping.get(int(p), int(p)) for p in pred], dtype=int)
    mask1 = truth == 1
    mask0 = truth == 0
    tpr = float(np.mean(aligned[mask1] == 1)) if mask1.any() else 0.0
    fpr = float(np.mean(aligned[mask0] == 1)) if mask0.any() else 0.0
    return tpr - fpr


def edge_direction_score(state_scores: np.ndarray, d: np.ndarray, edges: np.ndarray, mapping: dict[int, int]) -> float:
    """R_edge = mean sign(d_e * delta_e), with state rows reordered to truth indexing."""
    n_states = state_scores.shape[0]
    # Align predicted state rows to truth state indexing: G_aligned[truth] = state_scores[pred].
    aligned = np.zeros_like(state_scores)
    for pred_state in range(n_states):
        truth_state = mapping.get(pred_state, pred_state)
        if 0 <= truth_state < n_states:
            aligned[truth_state] = state_scores[pred_state]
    lags, nodes, _ = state_scores.shape[1:]
    g_flat = aligned.reshape(n_states, -1)[:, edges]
    delta_e = g_flat[1] - g_flat[0]
    return float(np.mean(np.sign(d * delta_e)))


def pooled_same_cross_split(scores_by_window: np.ndarray, active: np.ndarray) -> dict[str, float]:
    """Split the pooled window x edge AUROC into same-edge and cross-edge terms.

    ``scores_by_window`` is (n_windows, n_edges); ``active`` is the same shape.
    """
    n_windows, n_edges = scores_by_window.shape
    n_pos = active.sum(axis=0)
    n_neg = n_windows - n_pos
    same_pairs = float((n_pos * n_neg).sum())
    total_pairs = float(n_pos.sum()) * float(n_neg.sum())
    weight = same_pairs / total_pairs if total_pairs > 0 else 0.0

    # Same-edge AUROC: for each edge, the time AUROC of scores against active.
    same_correct = 0.0
    same_total = 0.0
    for e in range(n_edges):
        if n_pos[e] == 0 or n_neg[e] == 0:
            continue
        y = active[:, e]
        s = scores_by_window[:, e]
        if len(np.unique(y)) < 2:
            continue
        auc = roc_auc_score(y, s)
        same_correct += auc * n_pos[e] * n_neg[e]
        same_total += n_pos[e] * n_neg[e]
    auc_same = same_correct / same_total if same_total > 0 else float("nan")

    # Pooled AUROC over all window x edge pairs.
    y_flat = active.reshape(-1)
    s_flat = scores_by_window.reshape(-1)
    auc_pooled = roc_auc_score(y_flat, s_flat)

    # Cross-edge term from the exact split.
    auc_cross = (auc_pooled - weight * auc_same) / (1.0 - weight) if weight < 1.0 else float("nan")
    return {
        "weight_same": weight,
        "auc_pooled": float(auc_pooled),
        "auc_same": float(auc_same),
        "auc_cross": float(auc_cross),
    }


def active_matrix(state_support: np.ndarray, bounds, regimes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lags, nodes, _ = state_support.shape[1:]
    mask = nonself_mask((lags, nodes, nodes))
    edges = np.flatnonzero(mask.reshape(-1))
    active = np.zeros((len(bounds), len(edges)), dtype=float)
    for w, (start, end) in enumerate(bounds):
        centre = (start + end - 1) // 2
        active[w] = state_support[int(regimes[centre])].reshape(-1)[edges]
    return active, edges


def run_seed(setting: dict, seed: int, config: dict) -> dict:
    dataset = generate_mechanism(MechanismConfig(**setting, seed=seed))
    truth = GraphTruth(
        support=dataset.union_support,
        state_support=dataset.state_support,
        regimes=dataset.regimes,
    )
    state_config = STCGStateConfig(
        n_states=int(config["n_states"]),
        base_model="lasso",
        lasso_alpha=0.005,
        window_size=int(config["window"]["size"]),
        stride=int(config["window"]["stride"]),
        aggregate="max",
        state_aggregate="mean",
        seed=seed,
    )
    diagnostics = stcg_v1_diagnostics(dataset.x, int(setting["max_lag"]), state_config)
    window_scores = diagnostics.window_scores
    bounds = tuple(tuple(b) for b in diagnostics.bounds)
    fitted_labels = np.asarray(diagnostics.labels)
    z_w = window_regime_labels(bounds, dataset.regimes)

    # Align predicted labels to truth by overlap.
    matched = match_states(fitted_labels, z_w, n_truth_states=dataset.state_support.shape[0])
    mapping = {int(k): int(v) for k, v in matched["mapping"].items()}

    J = state_discrimination(fitted_labels, z_w, mapping)
    d, edges = _edge_directions(dataset.state_support)
    R = edge_direction_score(diagnostics.state_scores, d, edges, mapping)

    delta = temporal_delta_auc(
        diagnostics.state_scores, fitted_labels, bounds, truth.regimes, truth.state_support
    )
    t_auc_delta = float(delta["temporal_delta_auroc"])
    identity_lhs = 0.5 + 0.5 * J * R

    # Pooled same/cross split for K-means.
    active, act_edges = active_matrix(dataset.state_support, bounds, dataset.regimes)
    # Build per-window predicted scores for K-means: G[z_hat_w, e].
    n_states = diagnostics.state_scores.shape[0]
    index = {int(label): pos for pos, label in enumerate(np.unique(fitted_labels))}
    pred_by_window = np.zeros((len(bounds), len(act_edges)), dtype=float)
    for w in range(len(bounds)):
        row = index.get(int(fitted_labels[w]), 0)
        pred_by_window[w] = diagnostics.state_scores[row].reshape(-1)[act_edges]
    split = pooled_same_cross_split(pred_by_window, active)

    return {
        "seed": seed,
        "J": J,
        "R_edge": R,
        "J_times_R": J * R,
        "t_auc_delta": t_auc_delta,
        "identity_predicted": identity_lhs,
        "identity_gap": abs(t_auc_delta - identity_lhs),
        **split,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds else list(config["seeds"])
    setting = config["settings"]["switching_core"]

    rows = [run_seed(setting, seed, config) for seed in seeds]
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "state_decomposition.csv", index=False)

    print(frame.round(4).to_string(index=False))
    print("\nsummary:")
    print(f"  J           mean {frame.J.mean():.4f}  min {frame.J.min():.4f}  max {frame.J.max():.4f}")
    print(f"  R_edge      mean {frame.R_edge.mean():.4f}  min {frame.R_edge.min():.4f}  max {frame.R_edge.max():.4f}")
    print(f"  J*R_edge    mean {frame.J_times_R.mean():.4f}")
    print(f"  E[J]E[R]    {frame.J.mean()*frame.R_edge.mean():.4f}  (differs from E[J*R] in general)")
    print(f"  T-AUC_delta mean {frame.t_auc_delta.mean():.4f}")
    print(f"  max |identity gap| = {frame.identity_gap.max():.2e}")
    print(f"  pooled mean {frame.auc_pooled.mean():.4f} = w*{frame.auc_same.mean():.4f} + (1-w)*{frame.auc_cross.mean():.4f}, w={frame.weight_same.mean():.4f}")
    print(f"  auc_same mean {frame.auc_same.mean():.4f} (weighted per-edge time AUC)")
    print(f"  auc_cross mean {frame.auc_cross.mean():.4f} (cross-edge ranking)")


if __name__ == "__main__":
    main()
