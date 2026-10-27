"""Synthetic temporal systems with known directed lagged structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticDataset:
    """Container for synthetic time series and lagged graph ground truth."""

    x: np.ndarray
    coefficients: np.ndarray
    edge_truth: np.ndarray
    regimes: np.ndarray


def _sample_coefficients(
    rng: np.random.Generator,
    n_nodes: int,
    max_lag: int,
    edge_prob: float,
) -> np.ndarray:
    coeffs = np.zeros((max_lag, n_nodes, n_nodes), dtype=float)
    for lag in range(max_lag):
        mask = rng.random((n_nodes, n_nodes)) < edge_prob
        np.fill_diagonal(mask, False)
        signs = rng.choice([-1.0, 1.0], size=(n_nodes, n_nodes))
        magnitudes = rng.uniform(0.08, 0.35, size=(n_nodes, n_nodes))
        coeffs[lag] = mask * signs * magnitudes / (lag + 1)

    if not np.any(coeffs):
        source = int(rng.integers(0, n_nodes))
        target = (source + 1) % n_nodes
        coeffs[0, source, target] = 0.25

    incoming = np.abs(coeffs).sum(axis=(0, 1), keepdims=True)
    scale = np.maximum(1.0, incoming / 0.85)
    return coeffs / scale


def generate_lagged_var(
    n_nodes: int = 8,
    timesteps: int = 1000,
    max_lag: int = 3,
    edge_prob: float = 0.18,
    noise_scale: float = 0.10,
    seed: int = 0,
    nonlinear: bool = False,
    switching: bool = False,
    switch_period: int | None = None,
    burn_in: int = 100,
) -> SyntheticDataset:
    """Generate a stable lagged VAR-style system.

    `coefficients[lag - 1, source, target]` represents the directed effect
    `source(t-lag) -> target(t)`.
    """

    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    if timesteps <= max_lag + 1:
        raise ValueError("timesteps must be greater than max_lag + 1")

    rng = np.random.default_rng(seed)
    total_steps = timesteps + burn_in
    if switch_period is not None and switch_period <= max_lag + 1:
        raise ValueError("switch_period must be greater than max_lag + 1")

    coeffs_a = _sample_coefficients(rng, n_nodes, max_lag, edge_prob)
    coeffs_b = _sample_coefficients(rng, n_nodes, max_lag, edge_prob) if switching else coeffs_a

    x = rng.normal(0.0, noise_scale, size=(total_steps, n_nodes))
    regimes = np.zeros(total_steps, dtype=int)
    switch_point = total_steps // 2

    for t in range(max_lag, total_steps):
        if switching and switch_period is not None:
            observed_index = max(0, t - burn_in)
            regime = (observed_index // switch_period) % 2
        else:
            regime = int(switching and t >= switch_point)
        active = coeffs_b if switching and regime == 1 else coeffs_a
        regimes[t] = int(regime)
        value = np.zeros(n_nodes, dtype=float)
        for lag in range(1, max_lag + 1):
            source = x[t - lag]
            if nonlinear:
                source = np.tanh(2.0 * source) + 0.5 * np.sin(4.0 * source)
            value += source @ active[lag - 1]
        x[t] = value + rng.normal(0.0, noise_scale, size=n_nodes)

    coeffs = np.stack([coeffs_a, coeffs_b], axis=0) if switching else coeffs_a[None, ...]
    edge_truth = np.any(np.abs(coeffs) > 0, axis=0)
    return SyntheticDataset(
        x=x[burn_in:],
        coefficients=coeffs,
        edge_truth=edge_truth.astype(int),
        regimes=regimes[burn_in:],
    )
