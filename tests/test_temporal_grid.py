"""Regression checks for paired temporal-grid estimands and gate separation."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_temporal_grid as grid


def test_soft_emission_mixes_maps_instead_of_hardening_the_window(monkeypatch):
    weights = np.array([[0.8, 0.2], [0.6, 0.4], [0.2, 0.8]])
    class Mixture:
        converged_, n_iter_ = True, 2
        def __init__(self, **kwargs):
            assert kwargs == dict(n_components=2, covariance_type="diag", n_init=3, random_state=9)
        def fit(self, features):
            return self
        def predict_proba(self, features):
            return weights
    monkeypatch.setattr(grid, "GaussianMixture", Mixture)
    windows = np.array([1.0, 3.0, 9.0]).reshape(3, 1, 1, 1)
    predictions, info = grid.gmm_window_predictions(windows, np.arange(6).reshape(3, 2), 9)
    maps = weights.T @ windows[:, 0, 0, 0] / weights.sum(axis=0)
    np.testing.assert_allclose(predictions["gmm_soft"][:, 0, 0, 0], weights @ maps)
    np.testing.assert_allclose(predictions["gmm_weighted_argmax"][:, 0, 0, 0], maps[[0, 0, 1]])
    assert predictions["gmm_soft"][0].item() != predictions["gmm_soft"][1].item()
    assert info["gmm_entropy_normalized"] > 0


def test_explicit_continuous_predictions_recover_per_edge_temporal_ranking():
    support = np.zeros((2, 1, 2, 2)); support[0, 0, 0, 1] = 1; support[1, 0, 1, 0] = 1
    regimes = np.array([0, 0, 1, 1])
    scores = np.stack([0.9 * support[0] + 0.1 * support[1], 0.7 * support[0] + 0.3 * support[1],
                       0.4 * support[0] + 0.6 * support[1], 0.2 * support[0] + 0.8 * support[1]])
    bounds = [(i, i + 1) for i in range(4)]
    dataset = SimpleNamespace(regimes=regimes, state_support=support)
    assert grid.temporal_metrics(scores, bounds, dataset)["temporal_delta_auroc"] == 1
    constant = np.repeat(scores.mean(axis=0)[None], 4, axis=0)
    assert grid.temporal_metrics(constant, bounds, dataset)["temporal_delta_auroc"] == 0.5
    no_change = SimpleNamespace(regimes=regimes, state_support=np.repeat(support[0][None], 2, axis=0))
    assert np.isnan(grid.temporal_metrics(scores, bounds, no_change)["temporal_delta_auroc"])


def test_summary_keeps_native_execution_and_common_gate_subgroups_separate():
    rows = []
    for method in ("kmeans_hard", "gmm_soft"):
        for seed in range(6):
            accepted = seed < 2
            candidate = 0.1
            delta = candidate if accepted or method == "gmm_soft" else -0.2
            rows.append(dict(grid_entry=0, config_id="x", axis="core", level="core", method=method,
                             seed=seed, hard_gate_accepted=accepted, aggregation_executed=accepted if method == "kmeans_hard" else True,
                             evaluable=True, h_over_D=1, n_windows=19, timesteps=3000, switch_period=300,
                             window_size=300, stride=150, temporal_delta_auroc=0.7 + delta,
                             raw_temporal_delta_auroc=0.7, delta_temporal=delta,
                             candidate_temporal_delta_auroc=0.8, candidate_delta_temporal=candidate,
                             pooled_temporal_auroc=0.8))
    result = grid.summarise(pd.DataFrame(rows)).set_index(["method", "subset"])
    assert result.loc[("kmeans_hard", "all"), "delta_temporal_mean"] == pytest.approx(-0.1)
    assert result.loc[("kmeans_hard", "all"), "candidate_delta_temporal_mean"] == pytest.approx(0.1)
    assert result.loc[("kmeans_hard", "aggregation_executed"), "n_seeds"] == 2
    assert result.loc[("gmm_soft", "aggregation_executed"), "n_seeds"] == 6
    assert result.loc[("gmm_soft", "hard_gate_accepted"), "n_seeds"] == 2
    assert np.isnan(result.loc[("kmeans_hard", "aggregation_executed"), "delta_temporal_ci95_low"])


def test_bootstrap_uses_seed_differences_and_omits_small_or_undefined_intervals():
    assert grid.mean_interval(np.full(20, 0.1)) == pytest.approx((0.1, 0.1, 0.1))
    assert all(np.isnan(value) for value in grid.mean_interval(np.full(20, np.nan)))
    mean, low, high = grid.mean_interval(np.array([0.1, 0.2, np.nan]))
    assert mean == pytest.approx(0.15) and np.isnan(low) and np.isnan(high)


def test_same_seed_methods_share_raw_and_gate_but_keep_ungated_candidate():
    import yaml
    config = yaml.safe_load((grid.ROOT / "configs/mechanism_grid.yaml").read_text())
    setting = {**config["core"], "timesteps": 3000}
    rows = grid.fit_seed(setting, 0, config)
    assert {row["method"] for row in rows} == set(grid.METHODS)
    assert len({row["raw_temporal_delta_auroc"] for row in rows}) == 1
    assert len({row["hard_gate_accepted"] for row in rows}) == 1
    hard = next(row for row in rows if row["method"] == "kmeans_hard")
    if not hard["hard_gate_accepted"]:
        assert hard["temporal_delta_auroc"] == 0.5 and not hard["aggregation_executed"]
    assert all(row["aggregation_executed"] for row in rows if row["method"].startswith("gmm"))


def test_full_grid_retains_pairing_denominators_and_historical_references():
    sys.path.insert(0, str(grid.ROOT / "tools"))
    from verify_temporal_grid import verify
    assert verify()["distinct_fits"] == 620
