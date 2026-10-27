"""Small metric helpers for synthetic structure recovery sanity checks."""

from __future__ import annotations

import numpy as np


def lagged_correlation_scores(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Score directed lagged pairs with absolute lagged Pearson correlation.

    Returns an array with shape `(max_lag, n_nodes, n_nodes)` where entry
    `[lag - 1, source, target]` scores `source(t-lag) -> target(t)`.
    """

    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (timesteps, n_nodes), got {x.shape}")

    timesteps, n_nodes = x.shape
    if timesteps <= max_lag + 1:
        raise ValueError("Need more timesteps than max_lag + 1")

    scores = np.zeros((max_lag, n_nodes, n_nodes), dtype=float)
    for lag in range(1, max_lag + 1):
        src = x[:-lag]
        tgt = x[lag:]
        src = (src - src.mean(axis=0, keepdims=True)) / (src.std(axis=0, keepdims=True) + 1e-8)
        tgt = (tgt - tgt.mean(axis=0, keepdims=True)) / (tgt.std(axis=0, keepdims=True) + 1e-8)
        scores[lag - 1] = np.abs(src.T @ tgt) / src.shape[0]

    for node in range(n_nodes):
        scores[:, node, node] = 0.0
    return scores


def _flatten_without_self(values: np.ndarray) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError(f"Expected shape (lags, nodes, nodes), got {values.shape}")
    _, n_nodes, _ = values.shape
    mask = np.ones_like(values, dtype=bool)
    for node in range(n_nodes):
        mask[:, node, node] = False
    return values[mask]


def _nonself_mask(values: np.ndarray) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError(f"Expected shape (lags, nodes, nodes), got {values.shape}")
    _, n_nodes, _ = values.shape
    mask = np.ones_like(values, dtype=bool)
    for node in range(n_nodes):
        mask[:, node, node] = False
    return mask


def edge_auc(scores: np.ndarray, truth: np.ndarray) -> float:
    """Compute ROC AUC for lagged edge recovery without external dependencies."""

    y_score = _flatten_without_self(scores)
    y_true = _flatten_without_self(truth).astype(bool)
    positives = int(y_true.sum())
    negatives = int((~y_true).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)

    unique_scores, inverse, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    if len(unique_scores) < len(y_score):
        rank_sums = np.zeros(len(unique_scores), dtype=float)
        np.add.at(rank_sums, inverse, ranks)
        ranks = rank_sums[inverse] / counts[inverse]

    positive_rank_sum = ranks[y_true].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def pair_edge_auc(scores: np.ndarray, truth: np.ndarray) -> float:
    """Compute ROC AUC after collapsing lags by max score per directed pair."""

    if scores.shape != truth.shape:
        raise ValueError(f"Shape mismatch: scores {scores.shape}, truth {truth.shape}")
    return edge_auc(scores.max(axis=0, keepdims=True), truth.max(axis=0, keepdims=True))


def top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Return a binary lagged-edge mask selecting top-k non-self scores."""

    if k <= 0:
        return np.zeros_like(scores, dtype=bool)
    mask = _nonself_mask(scores)
    flat_scores = scores[mask]
    k = min(k, flat_scores.size)
    top_positions = np.argpartition(-flat_scores, kth=k - 1)[:k]
    selected_flat = np.zeros(flat_scores.size, dtype=bool)
    selected_flat[top_positions] = True
    selected = np.zeros_like(scores, dtype=bool)
    selected[mask] = selected_flat
    return selected


def precision_at_k(scores: np.ndarray, truth: np.ndarray, k: int | None = None) -> float:
    """Compute precision among the top-k lagged edge scores."""

    y_score = _flatten_without_self(scores)
    y_true = _flatten_without_self(truth).astype(bool)
    if k is None:
        k = max(1, int(y_true.sum()))
    k = min(k, len(y_score))
    top = np.argpartition(-y_score, kth=k - 1)[:k]
    return float(y_true[top].mean())


def structural_hamming_distance(scores: np.ndarray, truth: np.ndarray, k: int | None = None) -> int:
    """Compute lagged SHD after top-k thresholding scores."""

    if scores.shape != truth.shape:
        raise ValueError(f"Shape mismatch: scores {scores.shape}, truth {truth.shape}")
    y_true = truth.astype(bool)
    if k is None:
        k = int(y_true[_nonself_mask(y_true)].sum())
    selected = top_k_mask(scores, k)
    mask = _nonself_mask(scores)
    return int(np.logical_xor(selected[mask], y_true[mask]).sum())


def lag_recovery_accuracy(scores: np.ndarray, truth: np.ndarray) -> float:
    """Return how often the best-scored lag matches a true lag for true pairs."""

    if scores.shape != truth.shape:
        raise ValueError(f"Shape mismatch: scores {scores.shape}, truth {truth.shape}")

    _, n_nodes, _ = truth.shape
    correct = 0
    total = 0
    for source in range(n_nodes):
        for target in range(n_nodes):
            if source == target:
                continue
            true_lags = np.flatnonzero(truth[:, source, target])
            if true_lags.size == 0:
                continue
            total += 1
            predicted_lag = int(np.argmax(scores[:, source, target]))
            correct += int(predicted_lag in true_lags)
    if total == 0:
        return float("nan")
    return float(correct / total)


def recovery_summary(scores: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Compute the default synthetic recovery metrics for a score tensor."""

    true_k = int(_flatten_without_self(truth).astype(bool).sum())
    n_possible = int(_flatten_without_self(truth).size)
    return {
        "edge_auc": edge_auc(scores, truth),
        "pair_auc": pair_edge_auc(scores, truth),
        "precision_at_k": precision_at_k(scores, truth, true_k),
        "shd": float(structural_hamming_distance(scores, truth, true_k)),
        "lag_accuracy": lag_recovery_accuracy(scores, truth),
        "true_edges": float(true_k),
        "edge_density": float(true_k / n_possible),
    }
