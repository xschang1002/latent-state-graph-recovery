"""Refit controls for the aggregation/segmentation factorial.

The factorial separates two things that the PELT+refit baseline changed at
once:

- **partitioning**: K-means on window scores versus PELT segmentation;
- **re-estimation**: averaging existing window scores versus refitting Lasso on
  the time ranges implied by the partition.

Helpers here build both variants from one detector output, so the only
difference between cells of the factorial is the factor under test.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "group_time_ranges",
    "aggregate_within_groups",
    "refit_group_scores",
]


def group_time_ranges(
    bounds: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    labels: np.ndarray,
    total: int,
    min_length: int = 3,
) -> dict[int, list[tuple[int, int]]]:
    """Map window labels to contiguous time ranges per group.

    Boundaries fall at the midpoint between adjacent window centres, so the
    ranges partition the series. A group that appears in several runs returns
    one range per run; ranges shorter than ``min_length`` are dropped.
    """

    labels = np.asarray(labels).astype(int)
    if labels.size != len(bounds):
        raise ValueError("labels and bounds disagree on the number of windows")
    centres = np.array([(start + end - 1) // 2 for start, end in bounds], dtype=int)

    ranges: dict[int, list[tuple[int, int]]] = {}
    for group in np.unique(labels):
        members = np.flatnonzero(labels == group)
        runs = np.split(members, np.flatnonzero(np.diff(members) != 1) + 1)
        for run in runs:
            if run.size == 0:
                continue
            first, last = int(run[0]), int(run[-1])
            start = 0 if first == 0 else int((centres[first - 1] + centres[first]) // 2)
            end = total if last == labels.size - 1 else int((centres[last] + centres[last + 1]) // 2)
            end = max(end, start + 1)
            if end - start >= min_length:
                ranges.setdefault(int(group), []).append((start, end))
    return ranges


def aggregate_within_groups(
    window_scores: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Average window scores inside each group, without refitting."""

    labels = np.asarray(labels).astype(int)
    if window_scores.shape[0] != labels.size:
        raise ValueError("window_scores and labels disagree on the number of windows")
    groups = np.unique(labels)
    return np.stack([window_scores[labels == group].mean(axis=0) for group in groups], axis=0)


def refit_group_scores(
    x: np.ndarray,
    bounds: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    labels: np.ndarray,
    max_lag: int,
    alpha: float = 0.005,
) -> np.ndarray:
    """Refit Lasso on each group's contiguous runs and average within groups.

    Each run is estimated on its own data, so no artificial lag links are
    created across discontinuities; the per-group score is the mean over its
    runs. Groups whose runs are all too short are estimated on the full series,
    which is the least favourable fallback but keeps the row defined.
    """

    from .baselines import lasso_var_scores

    minimum = max(2 * max_lag + 5, max_lag + 3)
    ranges = group_time_ranges(bounds, labels, x.shape[0], min_length=minimum)
    groups = np.unique(np.asarray(labels).astype(int))
    scores = []
    for group in groups:
        runs = ranges.get(int(group), [])
        if runs:
            run_scores = [
                lasso_var_scores(x[start:end], max_lag=max_lag, alpha=alpha)
                for start, end in runs
            ]
            scores.append(np.mean(run_scores, axis=0))
        else:
            scores.append(lasso_var_scores(x, max_lag=max_lag, alpha=alpha))
    return np.stack(scores, axis=0)
