"""Change-point calibrated acceptance gate for STCG.

The score-space gate decides whether to keep a two-state partition using the
same window scores that produced the partition.  At short window counts this
is not identifiable: a stationary series can produce a K-means improvement as
large as a real switch.  This module tests the structural assumption directly
on the raw series with a VAR coefficient change-point scan, calibrated by a
parametric residual bootstrap.

The scan statistic is the maximum Chow-type F over candidate splits.  The null
simulates stationary series from a global VAR fit with bootstrapped residuals;
the p-value is the fraction of null scans at least as large as the observed
scan.  The gate accepts the two-state partition when ``p < alpha``.

References:
    Chow, G. C. (1960). Tests of equality between sets of coefficients in two
    linear regressions. Econometrica, 28(3), 591--605.
    Bai, J., & Perron, P. (1998). Estimating and testing linear models with
    multiple structural changes. Econometrica, 66(1), 47--78.
"""

from __future__ import annotations

import numpy as np


def _design(x: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    timesteps, nodes = x.shape
    rows = timesteps - lag
    design = np.ones((rows, 1 + nodes * lag))
    target = x[lag:]
    for offset in range(1, lag + 1):
        start = 1 + (offset - 1) * nodes
        design[:, start:start + nodes] = x[lag - offset:timesteps - offset]
    return design, target


def _fit(design: np.ndarray, target: np.ndarray):
    beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ beta
    return float((residual ** 2).sum()), beta, residual


def _scan_statistic(x: np.ndarray, lag: int, step: int, min_fraction: float) -> float:
    design, target = _design(x, lag)
    rows = len(design)
    parameters = design.shape[1]
    nodes = target.shape[1]
    total_rss, _, _ = _fit(design, target)
    low = int(rows * min_fraction)
    high = rows - low
    best = 0.0
    for split in range(low, high, step):
        left_rss, _, _ = _fit(design[:split], target[:split])
        right_rss, _, _ = _fit(design[split:], target[split:])
        value = ((total_rss - (left_rss + right_rss)) / (parameters * nodes)) / (
            (left_rss + right_rss) / (rows - 2 * parameters)
        )
        if value > best:
            best = value
    return float(best)


def _simulate(beta: np.ndarray, residual: np.ndarray, timesteps: int, lag: int, rng, burn: int = 100):
    nodes = residual.shape[1]
    series = np.zeros((timesteps + burn, nodes))
    for t in range(lag, timesteps + burn):
        row = np.empty(1 + nodes * lag)
        row[0] = 1.0
        for offset in range(1, lag + 1):
            row[1 + (offset - 1) * nodes:1 + offset * nodes] = series[t - offset]
        series[t] = row @ beta + residual[rng.integers(0, len(residual))]
    return series[burn:]


def change_point_pvalue(
    x: np.ndarray,
    lag: int,
    seed: int,
    draws: int = 99,
    step: int | None = None,
    min_fraction: float = 0.2,
) -> tuple[float, float]:
    """Return the observed scan statistic and its parametric-bootstrap p-value."""

    if step is None:
        step = 10 if len(x) <= 2000 else 50
    design, target = _design(x, lag)
    _, beta, residual = _fit(design, target)
    observed = _scan_statistic(x, lag, step, min_fraction)
    rng = np.random.default_rng(seed + 424242)
    null = np.array(
        [
            _scan_statistic(
                _simulate(beta, residual, len(x), lag, rng), lag, step, min_fraction
            )
            for _ in range(draws)
        ]
    )
    p_value = (1 + int((null >= observed).sum())) / (draws + 1)
    return observed, float(p_value)
