"""Tests for the PELT-based segmented baseline."""

from __future__ import annotations

import numpy as np

from stcg.segmented import pelt_on_features, segmented_lasso_scores


def test_pelt_finds_a_clear_level_shift():
    rng = np.random.default_rng(0)
    features = np.vstack(
        [rng.normal(0.0, 0.2, size=(8, 4)), rng.normal(3.0, 0.2, size=(8, 4))]
    )
    result = pelt_on_features(features, penalty_factor=0.5, n_pcs=2)
    assert any(abs(breakpoint - 8) <= 1 for breakpoint in result.breakpoints)
    assert result.labels[0] == 0 and result.labels[-1] == 1


def test_oracle_breakpoints_are_capped_by_grid_size():
    features = np.random.default_rng(1).normal(size=(7, 5))
    result = pelt_on_features(features, n_bkps=10)
    assert len(result.breakpoints) <= 3


def test_segmented_scores_partition_the_series():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(600, 4))
    bounds = tuple((start, start + 150) for start in range(0, 450, 75))
    labels = np.array([0, 0, 1, 1, 1, 1])
    scores, ranges = segmented_lasso_scores(x, bounds, labels, max_lag=2)
    assert len(ranges) == 2
    assert ranges == [(0, 186), (186, 600)]
    assert scores.shape[0] == len(ranges)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 600
    for (_, end), (start, _) in zip(ranges[:-1], ranges[1:]):
        assert end == start


def test_short_segments_are_merged():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 3))
    bounds = tuple((start, start + 10) for start in range(0, 12, 2))
    labels = np.array([0, 1, 2, 2, 2, 2])
    scores, ranges = segmented_lasso_scores(x, bounds, labels, max_lag=3)
    assert len(ranges) < 3
    assert scores.shape[0] == len(ranges)


def test_segmented_scores_ignore_label_numbers_and_keep_recurring_runs():
    x = np.random.default_rng(11).normal(size=(600, 3))
    bounds = tuple((start, start + 100) for start in range(0, 600, 100))
    labels = np.array([5, 5, -2, -2, 5, 5])
    scores, ranges = segmented_lasso_scores(x, bounds, labels, max_lag=2)
    assert ranges == [(0, 199), (199, 399), (399, 600)]
    assert len(scores) == 3


def test_chronological_segments_match_the_synthetic_refit_adapter():
    from stcg.refit import refit_group_scores
    x = np.random.default_rng(12).normal(size=(600, 3))
    bounds = tuple((start, start + 100) for start in range(0, 600, 100))
    labels = np.array([0, 0, 1, 1, 2, 2])
    scores, ranges = segmented_lasso_scores(x, bounds, labels, max_lag=2)
    assert ranges == [(0, 199), (199, 399), (399, 600)]
    np.testing.assert_array_equal(scores, refit_group_scores(x, bounds, labels, max_lag=2))


def test_constrained_segments_returns_requested_count():
    rng = np.random.default_rng(5)
    features = np.vstack(
        [rng.normal(0.0, 0.2, size=(6, 4)), rng.normal(2.0, 0.2, size=(6, 4))]
    )
    from stcg.segmented import constrained_segments

    result = constrained_segments(features, n_segments=2, min_size=2)
    assert np.unique(result.labels).size == 2
    assert abs(result.breakpoints[0] - 6) <= 1


def test_var_change_point_finds_the_switch():
    from stcg.mechanism import MechanismConfig, generate_mechanism
    from stcg.segmented import var_change_point

    dataset = generate_mechanism(MechanismConfig(seed=0, timesteps=800, switch_period=400))
    change = var_change_point(dataset.x, 3, min_fraction=0.2, step=10)
    # The schedule runs over the burn-in as well, so the first switch appears at
    # switch_period - burn_in in observed time.
    expected = 400 - 100
    assert abs(change["split"] - expected) <= 40
