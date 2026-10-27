"""Small statistical helpers used by the confirmatory analysis."""

from __future__ import annotations

import numpy as np
from scipy import stats as _stats

__all__ = ["holm", "bootstrap_ci", "paired_summary"]


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, monotone and capped at one."""

    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        return []
    order = np.argsort(values)
    adjusted = np.empty(values.size, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (values.size - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def bootstrap_ci(
    values: np.ndarray, draws: int = 10000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values``."""

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.choice(values, size=values.size, replace=True).mean() for _ in range(draws)]
    )
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def paired_summary(
    treatment: np.ndarray, control: np.ndarray, draws: int = 10000, seed: int = 0
) -> dict[str, float]:
    """Paired difference summary with a cluster bootstrap interval."""

    treatment = np.asarray(treatment, dtype=float)
    control = np.asarray(control, dtype=float)
    if treatment.shape != control.shape:
        raise ValueError("treatment and control must have the same shape")
    difference = treatment - control
    low, high = bootstrap_ci(difference, draws=draws, seed=seed)
    test = _stats.ttest_1samp(difference, 0.0)
    return {
        "n": int(difference.size),
        "mean_difference": float(difference.mean()),
        "ci_low": low,
        "ci_high": high,
        "t_stat": float(test.statistic),
        "p_value": float(test.pvalue),
    }
