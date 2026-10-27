"""Downstream forecasting utilities for graph-score evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baselines import make_lagged_design


@dataclass(frozen=True)
class ForecastSplit:
    """Chronological one-step forecasting design matrices."""

    design_train: np.ndarray
    target_train: np.ndarray
    design_test: np.ndarray
    target_test: np.ndarray
    raw_design_test: np.ndarray
    train_end: int
    target_mean: np.ndarray
    target_std: np.ndarray
    n_nodes: int
    max_lag: int


def chronological_lagged_split(
    x: np.ndarray,
    max_lag: int,
    train_fraction: float = 0.70,
) -> ForecastSplit:
    """Build leakage-free train/test matrices for one-step forecasting."""

    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (timesteps, n_nodes), got {x.shape}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    timesteps, n_nodes = x.shape
    train_end = int(round(timesteps * train_fraction))
    if train_end <= max_lag + 1:
        raise ValueError("Training split is too short for the requested max_lag")
    if train_end >= timesteps:
        raise ValueError("Training split leaves no test samples")

    x_mean = x[:train_end].mean(axis=0, keepdims=True)
    x_std = x[:train_end].std(axis=0, keepdims=True) + 1e-8
    x_scaled = (x - x_mean) / x_std

    raw_design, raw_target = make_lagged_design(x_scaled, max_lag=max_lag)
    target_index = np.arange(max_lag, timesteps)
    train_mask = target_index < train_end
    test_mask = ~train_mask
    if not np.any(test_mask):
        raise ValueError("No test rows remain after lagged split")

    design_mean = raw_design[train_mask].mean(axis=0, keepdims=True)
    design_std = raw_design[train_mask].std(axis=0, keepdims=True) + 1e-8
    target_mean = raw_target[train_mask].mean(axis=0, keepdims=True)
    target_std = raw_target[train_mask].std(axis=0, keepdims=True) + 1e-8

    design = (raw_design - design_mean) / design_std
    target = (raw_target - target_mean) / target_std

    return ForecastSplit(
        design_train=design[train_mask],
        target_train=target[train_mask],
        design_test=design[test_mask],
        target_test=target[test_mask],
        raw_design_test=raw_design[test_mask],
        train_end=train_end,
        target_mean=target_mean.reshape(-1),
        target_std=target_std.reshape(-1),
        n_nodes=n_nodes,
        max_lag=max_lag,
    )


def graph_feature_mask(
    scores: np.ndarray,
    fraction: float = 0.20,
    topk_per_target: int | None = None,
) -> np.ndarray:
    """Convert lagged edge scores into a feature mask for target-wise regressions."""

    if scores.ndim != 3:
        raise ValueError(f"Expected scores with shape (max_lag, source, target), got {scores.shape}")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")

    max_lag, n_sources, n_targets = scores.shape
    if n_sources != n_targets:
        raise ValueError("scores must have matching source and target dimensions")

    nonself_per_target = max_lag * (n_sources - 1)
    budget = topk_per_target if topk_per_target is not None else int(np.ceil(fraction * nonself_per_target))
    budget = max(1, min(int(budget), nonself_per_target))

    mask = np.zeros((max_lag * n_sources, n_targets), dtype=bool)
    for target in range(n_targets):
        values = scores[:, :, target].reshape(-1).astype(float)
        eligible = np.ones_like(values, dtype=bool)
        for lag in range(max_lag):
            eligible[lag * n_sources + target] = False
        ranked_values = values.copy()
        ranked_values[~eligible] = -np.inf
        selected = np.argpartition(ranked_values, ranked_values.size - budget)[-budget:]
        selected = selected[eligible[selected]]
        mask[selected, target] = True
    return mask


def graph_feature_weights(
    scores: np.ndarray,
    min_weight: float = 0.10,
    self_weight: float = 1.0,
    power: float = 1.0,
) -> np.ndarray:
    """Convert lagged edge scores into positive feature weights for ridge penalties.

    The returned matrix keeps every feature available. Larger graph scores receive
    larger weights, which translates to weaker ridge shrinkage in
    :func:`ridge_forecast_weighted`.
    """

    if scores.ndim != 3:
        raise ValueError(f"Expected scores with shape (max_lag, source, target), got {scores.shape}")
    if not 0.0 < min_weight <= 1.0:
        raise ValueError("min_weight must be in (0, 1]")
    if self_weight <= 0.0:
        raise ValueError("self_weight must be positive")
    if power <= 0.0:
        raise ValueError("power must be positive")

    max_lag, n_sources, n_targets = scores.shape
    if n_sources != n_targets:
        raise ValueError("scores must have matching source and target dimensions")

    clean_scores = np.nan_to_num(scores.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    clean_scores = np.maximum(clean_scores, 0.0)

    weights = np.full((max_lag * n_sources, n_targets), min_weight, dtype=float)
    for target in range(n_targets):
        values = clean_scores[:, :, target].reshape(-1)
        eligible = np.ones_like(values, dtype=bool)
        for lag in range(max_lag):
            eligible[lag * n_sources + target] = False

        eligible_values = values[eligible]
        max_score = float(np.max(eligible_values)) if eligible_values.size else 0.0
        if max_score > 0.0:
            normalized = np.zeros_like(values, dtype=float)
            normalized[eligible] = np.clip(eligible_values / max_score, 0.0, 1.0)
            weights[:, target] = min_weight + (1.0 - min_weight) * (normalized**power)
        for lag in range(max_lag):
            weights[lag * n_sources + target, target] = self_weight
    return weights


def ridge_forecast(
    split: ForecastSplit,
    alpha: float = 1.0,
    feature_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fit ridge regressions and return standardized test predictions."""

    n_features = split.design_train.shape[1]
    if feature_mask is not None and feature_mask.shape != (n_features, split.n_nodes):
        raise ValueError(
            f"feature_mask must have shape {(n_features, split.n_nodes)}, got {feature_mask.shape}"
        )

    predictions = np.zeros_like(split.target_test)
    for target in range(split.n_nodes):
        if feature_mask is None:
            selected = np.ones(n_features, dtype=bool)
        else:
            selected = feature_mask[:, target]
            if not np.any(selected):
                selected = np.ones(n_features, dtype=bool)
        train_x = split.design_train[:, selected]
        test_x = split.design_test[:, selected]
        train_y = split.target_train[:, target]
        gram = train_x.T @ train_x
        penalty = alpha * np.eye(gram.shape[0])
        coeff = np.linalg.solve(gram + penalty, train_x.T @ train_y)
        predictions[:, target] = test_x @ coeff
    return predictions


def ridge_forecast_weighted(
    split: ForecastSplit,
    alpha: float = 1.0,
    feature_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Fit target-wise ridge regressions with graph-weighted diagonal penalties."""

    if feature_weights is None:
        return ridge_forecast(split, alpha=alpha)

    n_features = split.design_train.shape[1]
    if feature_weights.shape != (n_features, split.n_nodes):
        raise ValueError(
            f"feature_weights must have shape {(n_features, split.n_nodes)}, got {feature_weights.shape}"
        )
    if not np.all(np.isfinite(feature_weights)):
        raise ValueError("feature_weights must be finite")
    if not np.all(feature_weights > 0.0):
        raise ValueError("feature_weights must be positive")

    predictions = np.zeros_like(split.target_test)
    for target in range(split.n_nodes):
        train_x = split.design_train
        test_x = split.design_test
        train_y = split.target_train[:, target]
        gram = train_x.T @ train_x
        penalty = np.diag(alpha / feature_weights[:, target])
        coeff = np.linalg.solve(gram + penalty, train_x.T @ train_y)
        predictions[:, target] = test_x @ coeff
    return predictions


def persistence_forecast(split: ForecastSplit) -> np.ndarray:
    """Use the most recent observed value as the one-step prediction."""

    lag_one = split.raw_design_test[:, : split.n_nodes]
    return (lag_one - split.target_mean) / split.target_std


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return simple standardized forecasting errors."""

    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    node_rmse = np.sqrt(np.mean(residual**2, axis=0))
    return {
        "mae": mae,
        "rmse": rmse,
        "node_rmse_mean": float(np.mean(node_rmse)),
        "node_rmse_max": float(np.max(node_rmse)),
    }
