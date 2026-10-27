"""Direct coverage for the scoring adapters shared by every experiment."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from stcg.analysis import (
    aggregate_state_scores,
    oracle_label_prediction,
    prediction_from_diagnostics,
    truth_from_dataset,
    union_scores_from_states,
)


def test_aggregate_state_scores_supports_mean_median_max():
    window_scores = np.array([[[1.0]], [[3.0]], [[10.0]], [[14.0]]])
    labels = np.array([0, 0, 1, 1])
    assert aggregate_state_scores(window_scores, labels, "mean").ravel().tolist() == [2.0, 12.0]
    assert aggregate_state_scores(window_scores, labels, "median").ravel().tolist() == [2.0, 12.0]
    assert aggregate_state_scores(window_scores, labels, "max").ravel().tolist() == [3.0, 14.0]


def test_aggregate_state_scores_rejects_mismatch_and_unknown_mode():
    with pytest.raises(ValueError, match="one label per window"):
        aggregate_state_scores(np.zeros((3, 1, 1)), np.array([0, 1]), "mean")
    with pytest.raises(ValueError, match="Unknown aggregate mode"):
        aggregate_state_scores(np.zeros((2, 1, 1)), np.array([0, 1]), "sum")


def test_union_scores_from_states_modes():
    state_scores = np.array([[[1.0]], [[3.0]]])
    assert union_scores_from_states(state_scores, "max").item() == 3.0
    assert union_scores_from_states(state_scores, "mean").item() == 2.0
    with pytest.raises(ValueError, match="Unknown union mode"):
        union_scores_from_states(state_scores, "median")


def test_truth_from_dataset_and_oracle_label_prediction():
    dataset = SimpleNamespace(
        union_support=np.zeros((1, 2, 2)),
        state_support=np.zeros((2, 1, 2, 2)),
        regimes=np.array([0, 0, 1, 1]),
    )
    truth = truth_from_dataset(dataset)
    assert truth.support.shape == (1, 2, 2)
    assert truth.state_support.shape == (2, 1, 2, 2)

    window_scores = np.arange(2 * 1 * 2 * 2, dtype=float).reshape(2, 1, 2, 2)
    prediction = oracle_label_prediction(window_scores, ((0, 2), (2, 4)), dataset.regimes)
    assert prediction.window_assignments.tolist() == [0, 1]
    assert prediction.state_scores.shape == (2, 1, 2, 2)
    assert prediction.union_scores.shape == (1, 2, 2)


def test_prediction_from_diagnostics_preserves_shapes_and_bounds():
    diagnostics = SimpleNamespace(
        scores=np.zeros((1, 2, 2)),
        window_scores=np.zeros((3, 1, 2, 2)),
        bounds=((0, 2), (2, 4), (4, 6)),
        state_scores=np.zeros((2, 1, 2, 2)),
        labels=np.array([0, 1, 1]),
    )
    prediction = prediction_from_diagnostics(diagnostics)
    assert prediction.union_scores.shape == (1, 2, 2)
    assert prediction.window_scores.shape == (3, 1, 2, 2)
    assert prediction.window_bounds == ((0, 2), (2, 4), (4, 6))
    assert prediction.window_assignments.tolist() == [0, 1, 1]
