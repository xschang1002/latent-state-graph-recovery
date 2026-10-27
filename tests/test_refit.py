"""Tests for the partition/refit helpers used by the 2x2 factorial."""

from __future__ import annotations

import numpy as np

from stcg.refit import aggregate_within_groups, group_time_ranges, refit_group_scores


def test_group_ranges_partition_the_series():
    bounds = ((0, 100), (50, 150), (100, 200), (150, 250), (200, 300))
    labels = np.array([0, 0, 1, 1, 1])
    ranges = group_time_ranges(bounds, labels, total=300, min_length=10)
    flat = sorted(pair for pairs in ranges.values() for pair in pairs)
    assert flat[0][0] == 0
    assert flat[-1][1] == 300
    for (_, end), (start, _) in zip(flat[:-1], flat[1:]):
        assert end <= start


def test_group_ranges_split_discontiguous_runs():
    bounds = ((0, 100), (50, 150), (100, 200), (150, 250))
    labels = np.array([0, 1, 0, 1])
    ranges = group_time_ranges(bounds, labels, total=250, min_length=10)
    assert len(ranges[0]) == 2
    assert len(ranges[1]) == 2


def test_aggregate_within_groups_matches_manual_mean():
    scores = np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    labels = np.array([0, 1])
    aggregated = aggregate_within_groups(scores, labels)
    assert aggregated.shape == (2, 2, 2, 2)
    assert np.allclose(aggregated, scores)


def test_refit_returns_one_score_per_group():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 4))
    bounds = tuple((start, start + 150) for start in range(0, 450, 75))
    labels = np.array([0, 0, 1, 1, 1, 1])
    scores = refit_group_scores(x, bounds, labels, max_lag=2)
    assert scores.shape[0] == 2
    assert scores.shape[1:] == (2, 4, 4)
