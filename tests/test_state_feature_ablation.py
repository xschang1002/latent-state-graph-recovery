"""Self-lag ablations must vary fitted information, not append zero columns."""

from dataclasses import replace

import numpy as np
import pytest

from stcg import STCGStateConfig, stcg_v1_diagnostics
from stcg.baselines import lasso_var_scores, ridge_var_coefficients


def autoregressive_series() -> np.ndarray:
    rng = np.random.default_rng(218)
    x = rng.normal(scale=0.3, size=(360, 3))
    for t in range(1, len(x)):
        x[t] += np.array([0.82, 0.55, 0.30]) * x[t - 1]
        x[t, 1] += 0.25 * x[t - 1, 0]
    return x


@pytest.mark.parametrize("fitter", [lasso_var_scores, ridge_var_coefficients])
def test_retaining_self_lags_preserves_every_nonself_coefficient(fitter):
    x = autoregressive_series()
    default = fitter(x, max_lag=2)
    full = fitter(x, max_lag=2, include_self=True)
    diagonal = np.diagonal(full, axis1=1, axis2=2)
    assert np.max(np.abs(diagonal)) > 0.5
    assert np.all(np.diagonal(default, axis1=1, axis2=2) == 0)
    indices = np.arange(x.shape[1])
    full[:, indices, indices] = 0
    np.testing.assert_array_equal(default, full)


@pytest.mark.parametrize("base_model", ["lasso", "ridge"])
@pytest.mark.parametrize("path", ["accepted", "rejected", "one_window"])
def test_all_features_contain_actual_self_lags_but_graphs_exclude_them(base_model, path):
    x = autoregressive_series()
    config = STCGStateConfig(
        base_model=base_model,
        window_size=len(x) if path == "one_window" else 120,
        stride=60,
        auto_states=path == "rejected",
        min_state_improvement=1.1,
        min_temporal_state_improvement=1.1,
        seed=19,
    )
    default = stcg_v1_diagnostics(x, 2, config)
    ablation = stcg_v1_diagnostics(x, 2, replace(config, state_features="all"))
    n_windows, n_lags, n_nodes, _ = ablation.window_scores.shape
    full_features = ablation.features.reshape(n_windows, n_lags, n_nodes, n_nodes)
    self_features = np.diagonal(full_features, axis1=2, axis2=3)
    assert np.max(self_features) > 0.5
    if path != "one_window":
        assert np.max(self_features.std(axis=0)) > 0.01
    mask = ~np.eye(n_nodes, dtype=bool)
    np.testing.assert_array_equal(full_features[:, :, mask].reshape(n_windows, -1), default.features)
    np.testing.assert_array_equal(ablation.window_scores, default.window_scores)
    for scores in (ablation.scores, ablation.window_scores, ablation.state_scores):
        assert np.all(np.diagonal(scores, axis1=-2, axis2=-1) == 0)
    if path == "accepted":
        assert ablation.states_accepted
    else:
        assert not ablation.states_accepted
        np.testing.assert_array_equal(ablation.scores, default.scores)

