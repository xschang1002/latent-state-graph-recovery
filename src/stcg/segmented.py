"""PELT + segmented VAR-Lasso baseline.

PELT (Killick et al., 2012) is applied to the sequence of window-level Lasso
scores, reduced to a few principal components, and each detected segment is
re-estimated with Lasso on its own time range.  The union score is the maximum
over segments.

This is the low-cost classical alternative to STCG: it reuses the same window
pass but replaces K-means with optimal partitioning of the score sequence.  The
adapter is deliberately restricted to the union estimand: PELT returns
segments, not repeated states, so two segments that share a graph are still two
segments and cannot be merged without an extra matching step.  Regime-specific
scores are therefore reported per segment, not per latent state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SegmentationResult",
    "pelt_on_features",
    "segmented_lasso_scores",
]


@dataclass(frozen=True)
class SegmentationResult:
    labels: np.ndarray
    breakpoints: list[int]
    penalty: float
    n_pcs: int


def _standardise(features: np.ndarray) -> np.ndarray:
    centred = features - features.mean(axis=0, keepdims=True)
    return centred / (centred.std(axis=0, keepdims=True) + 1e-8)


def pelt_on_features(
    features: np.ndarray,
    penalty_factor: float = 1.0,
    n_pcs: int = 5,
    min_size: int = 1,
    n_bkps: int | None = None,
) -> SegmentationResult:
    """Segment a window-feature sequence with PELT.

    ``n_bkps`` gives the oracle-segmentation upper bound (the number of true
    segments); ``penalty_factor`` scales the BIC penalty
    ``d_eff * log(W)`` used when the number of change points is unknown.
    """

    import ruptures as rpt

    features = np.asarray(features, dtype=float)
    n_windows = features.shape[0]
    if n_windows < 3:
        return SegmentationResult(np.zeros(n_windows, dtype=int), [], 0.0, 0)

    standardised = _standardise(features)
    n_pcs = int(min(n_pcs, standardised.shape[0] - 1, standardised.shape[1]))
    if n_pcs >= 1:
        u, s, vt = np.linalg.svd(standardised - standardised.mean(axis=0), full_matrices=False)
        projected = (standardised - standardised.mean(axis=0)) @ vt[:n_pcs].T
    else:
        projected = standardised
        n_pcs = standardised.shape[1]

    if n_bkps is not None:
        # Exact k-segmentation upper bound. Pelt.predict only accepts a
        # penalty in ruptures >= 1.1, so the oracle-k variant uses Dynp. Dynp
        # requires at least two samples per segment, so the requested number of
        # change points is capped by what the window grid can represent.
        exact_min = max(2, min_size)
        feasible = max(0, n_windows // exact_min - 1)
        k = int(min(int(n_bkps), feasible))
        if k >= 1:
            detector = rpt.Dynp(model="l2", min_size=exact_min, jump=1).fit(projected)
            breakpoints = detector.predict(n_bkps=k)
            penalty = float("nan")
        else:
            detector = rpt.Pelt(model="l2", min_size=exact_min, jump=1).fit(projected)
            penalty = float(penalty_factor * n_pcs * np.log(max(n_windows, 2)))
            breakpoints = detector.predict(pen=penalty)
    else:
        detector = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(projected)
        penalty = float(penalty_factor * n_pcs * np.log(max(n_windows, 2)))
        breakpoints = detector.predict(pen=penalty)

    labels = np.zeros(n_windows, dtype=int)
    previous = 0
    for segment, end in enumerate(breakpoints):
        labels[previous:end] = segment
        previous = end
    return SegmentationResult(labels, list(breakpoints), penalty, n_pcs)


def segmented_lasso_scores(
    x: np.ndarray,
    bounds: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    labels: np.ndarray,
    max_lag: int,
    alpha: float = 0.005,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Re-fit Lasso on each segmented time range and return per-segment scores."""

    from .baselines import lasso_var_scores

    labels = np.asarray(labels).astype(int)
    if labels.size != len(bounds):
        raise ValueError("labels and bounds disagree on the number of windows")

    n_windows = labels.size
    if n_windows == 0:
        raise ValueError("Need at least one labelled window")
    centers = np.array([(start + end - 1) // 2 for start, end in bounds], dtype=int)
    total = x.shape[0]

    ranges: list[list[int]] = []
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, n_windows]
    for first, stop in zip(cuts[:-1], cuts[1:]):
        first, last = int(first), int(stop - 1)
        # Boundaries at the midpoint between adjacent window centres, so the
        # segments partition the series instead of overlapping.
        start = 0 if first == 0 else int((centers[first - 1] + centers[first]) // 2)
        end = total if last == n_windows - 1 else int((centers[last] + centers[last + 1]) // 2)
        end = max(end, start + 1)
        ranges.append([start, end])

    # Merge segments that are too short to fit a VAR.
    minimum = max(2 * max_lag + 5, max_lag + 3)
    merged: list[list[int]] = []
    for start, end in ranges:
        if merged and end - start < minimum:
            merged[-1][1] = end
        elif merged and merged[-1][1] - merged[-1][0] < minimum:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    if len(merged) > 1 and merged[-1][1] - merged[-1][0] < minimum:
        merged[-2][1] = merged[-1][1]
        merged.pop()

    scores = [
        lasso_var_scores(x[start:end], max_lag=max_lag, alpha=alpha)
        for start, end in merged
    ]
    return np.stack(scores, axis=0), [tuple(pair) for pair in merged]


def constrained_segments(
    features: np.ndarray, n_segments: int = 2, min_size: int = 2
) -> SegmentationResult:
    """Best contiguous segmentation with exactly ``n_segments`` segments.

    Unlike PELT with a penalty, the number of segments is fixed, which makes the
    comparison with a supplied two-state K-means partition like-for-like. The
    result is still a set of contiguous segments, not recurring regimes.
    """

    import ruptures as rpt

    features = np.asarray(features, dtype=float)
    n_windows = features.shape[0]
    if n_windows < 2 * n_segments:
        return SegmentationResult(np.zeros(n_windows, dtype=int), [], float("nan"), features.shape[1])

    standardised = _standardise(features)
    n_pcs = int(min(5, n_windows - 1, standardised.shape[1]))
    centred = standardised - standardised.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ vt[:n_pcs].T

    detector = rpt.Dynp(model="l2", min_size=min_size, jump=1).fit(projected)
    breakpoints = detector.predict(n_bkps=n_segments - 1)
    labels = np.zeros(n_windows, dtype=int)
    previous = 0
    for segment, end in enumerate(breakpoints):
        labels[previous:end] = segment
        previous = end
    return SegmentationResult(labels, list(breakpoints), float("nan"), n_pcs)


def _var_design(x: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    timesteps, n_nodes = x.shape
    rows = timesteps - max_lag
    design = np.ones((rows, 1 + n_nodes * max_lag))
    target = x[max_lag:]
    for lag in range(1, max_lag + 1):
        design[:, 1 + (lag - 1) * n_nodes : 1 + lag * n_nodes] = x[max_lag - lag : timesteps - lag]
    return design, target


def _rss(design: np.ndarray, target: np.ndarray) -> float:
    beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ beta
    return float((residual**2).sum())


def var_change_point(
    x: np.ndarray, max_lag: int, min_fraction: float = 0.2, step: int = 5
) -> dict[str, float]:
    """Chow-type scan for a single change in the VAR coefficient matrix.

    Returns the best split index in observed time and the scan statistic. This
    is the raw-coefficient change detector used as a segmentation comparator; it
    assumes piecewise-stationary coefficients, which is exactly the synthetic
    mechanism.
    """

    design, target = _var_design(x, max_lag)
    n_rows, n_params = design.shape
    n_targets = target.shape[1]
    rss_full = _rss(design, target)
    low = int(n_rows * min_fraction)
    high = n_rows - low
    best = {"split": float("nan"), "f_stat": -np.inf, "delta_rss": float("nan")}
    for split in range(low, high, step):
        rss_left = _rss(design[:split], target[:split])
        rss_right = _rss(design[split:], target[split:])
        denominator = (rss_left + rss_right) / max(1, n_rows - 2 * n_params)
        if denominator <= 0:
            continue
        f_stat = ((rss_full - (rss_left + rss_right)) / (n_params * n_targets)) / denominator
        if f_stat > best["f_stat"]:
            best = {
                "split": float(split + max_lag),
                "f_stat": float(f_stat),
                "delta_rss": float(rss_full - (rss_left + rss_right)),
            }
    return best
