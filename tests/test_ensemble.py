"""Regression tests for the STCG score-space ensemble."""
from __future__ import annotations

import numpy as np

from stcg import STCGStateConfig, generate_lagged_var, stcg_v1_diagnostics
from stcg.ensemble import ensemble_scores


def _fixture():
    dataset = generate_lagged_var(n_nodes=4, timesteps=400, max_lag=2, seed=21, switching=True)
    config = STCGStateConfig(window_size=80, stride=40, seed=14)
    diagnostics = stcg_v1_diagnostics(dataset.x, max_lag=2, config=config)
    return dataset, config, diagnostics


def test_ensemble_size_one_matches_base_scores():
    _, config, diagnostics = _fixture()
    scores = ensemble_scores(diagnostics, config, 1)
    np.testing.assert_allclose(scores, diagnostics.scores)


def test_ensemble_scores_are_deterministic():
    _, config, diagnostics = _fixture()
    first = ensemble_scores(diagnostics, config, 8, seed=config.seed)
    second = ensemble_scores(diagnostics, config, 8, seed=config.seed)
    np.testing.assert_allclose(first, second)


def test_ensemble_changes_scores_but_preserves_shape():
    _, config, diagnostics = _fixture()
    scores = ensemble_scores(diagnostics, config, 8)
    assert scores.shape == diagnostics.scores.shape
    assert not np.allclose(scores, diagnostics.scores)


def test_ensemble_respects_one_state_fallback():
    dataset = generate_lagged_var(
        n_nodes=8, timesteps=1200, max_lag=3, seed=0, switching=False, noise_scale=0.3
    )
    config = STCGStateConfig(window_size=150, stride=75, seed=0)
    diagnostics = stcg_v1_diagnostics(dataset.x, max_lag=3, config=config)
    assert not diagnostics.states_accepted
    scores = ensemble_scores(diagnostics, config, 20, seed=0)
    np.testing.assert_allclose(scores, diagnostics.scores)


def test_ensemble_rejects_nonpositive_size():
    _, config, diagnostics = _fixture()
    try:
        ensemble_scores(diagnostics, config, 0)
    except ValueError as exc:
        assert "size" in str(exc)
    else:
        raise AssertionError("Expected nonpositive size to raise ValueError")
