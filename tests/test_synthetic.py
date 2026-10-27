import numpy as np

from stcg import (
    STCGConfig,
    STCGStateConfig,
    chronological_lagged_split,
    dynamic_granger_f_scores,
    dynamic_ridge_var_scores,
    edge_auc,
    forecast_metrics,
    generate_lagged_var,
    granger_f_scores,
    graph_feature_mask,
    graph_feature_weights,
    lag_recovery_accuracy,
    lagged_correlation_scores,
    pcmci_parcorr_scores,
    precision_at_k,
    persistence_forecast,
    recovery_summary,
    ridge_forecast,
    ridge_forecast_weighted,
    ridge_var_scores,
    sliding_window_bounds,
    stcg_v0_scores,
    stcg_v1_diagnostics,
    stcg_v1_scores,
    structural_hamming_distance,
)
from stcg.self_supervised import _infer_state_labels, _window_state_features


def test_lagged_var_shapes_and_no_self_edges():
    dataset = generate_lagged_var(n_nodes=5, timesteps=120, max_lag=3, seed=1)

    assert dataset.x.shape == (120, 5)
    assert dataset.edge_truth.shape == (3, 5, 5)
    assert dataset.coefficients.shape == (1, 3, 5, 5)
    assert np.all(np.diagonal(dataset.edge_truth, axis1=1, axis2=2) == 0)
    assert dataset.edge_truth.sum() > 0


def test_switching_var_has_two_regimes():
    dataset = generate_lagged_var(
        n_nodes=5,
        timesteps=120,
        max_lag=2,
        seed=2,
        switching=True,
    )

    assert dataset.coefficients.shape == (2, 2, 5, 5)
    assert set(np.unique(dataset.regimes)) == {0, 1}


def test_switching_var_supports_periodic_regime_changes():
    dataset = generate_lagged_var(
        n_nodes=4,
        timesteps=220,
        max_lag=2,
        seed=11,
        switching=True,
        switch_period=40,
    )

    changes = np.flatnonzero(np.diff(dataset.regimes) != 0)
    assert len(changes) >= 4
    assert set(np.unique(dataset.regimes)) == {0, 1}


def test_lagged_correlation_metrics_are_bounded():
    dataset = generate_lagged_var(n_nodes=5, timesteps=180, max_lag=3, seed=3)
    scores = lagged_correlation_scores(dataset.x, max_lag=3)

    auc = edge_auc(scores, dataset.edge_truth)
    precision = precision_at_k(scores, dataset.edge_truth)

    assert scores.shape == dataset.edge_truth.shape
    assert 0.0 <= auc <= 1.0
    assert 0.0 <= precision <= 1.0


def test_ridge_var_scores_match_truth_shape():
    dataset = generate_lagged_var(n_nodes=5, timesteps=180, max_lag=3, seed=4)
    scores = ridge_var_scores(dataset.x, max_lag=3)

    assert scores.shape == dataset.edge_truth.shape
    assert np.all(np.diagonal(scores, axis1=1, axis2=2) == 0)


def test_granger_f_scores_match_truth_shape():
    dataset = generate_lagged_var(n_nodes=5, timesteps=180, max_lag=3, seed=5)
    scores = granger_f_scores(dataset.x, max_lag=3)

    assert scores.shape == dataset.edge_truth.shape
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
    assert np.all(np.diagonal(scores, axis1=1, axis2=2) == 0)


def test_pcmci_parcorr_scores_match_truth_shape():
    dataset = generate_lagged_var(n_nodes=4, timesteps=160, max_lag=2, seed=16)
    scores = pcmci_parcorr_scores(dataset.x, max_lag=2, pc_alpha=0.05)

    assert scores.shape == dataset.edge_truth.shape
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
    assert np.all(np.diagonal(scores, axis1=1, axis2=2) == 0)


def test_dynamic_ridge_var_scores_match_truth_shape():
    dataset = generate_lagged_var(n_nodes=5, timesteps=220, max_lag=3, seed=6, switching=True)
    scores = dynamic_ridge_var_scores(dataset.x, max_lag=3, window_size=80, stride=40)

    assert scores.shape == dataset.edge_truth.shape
    assert np.all(np.diagonal(scores, axis1=1, axis2=2) == 0)


def test_dynamic_granger_f_scores_match_truth_shape():
    dataset = generate_lagged_var(n_nodes=5, timesteps=220, max_lag=3, seed=6, switching=True)
    scores = dynamic_granger_f_scores(dataset.x, max_lag=3, window_size=80, stride=40)

    assert scores.shape == dataset.edge_truth.shape
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
    assert np.all(np.diagonal(scores, axis1=1, axis2=2) == 0)


def test_stcg_v0_scores_match_truth_shape():
    dataset = generate_lagged_var(n_nodes=4, timesteps=120, max_lag=2, seed=7, switching=True)
    config = STCGConfig(steps=5, window_size=60, stride=40, seed=11)
    scores = stcg_v0_scores(dataset.x, max_lag=2, config=config)

    assert scores.shape == dataset.edge_truth.shape
    assert np.all(np.diagonal(scores, axis1=1, axis2=2) == 0)


def test_stcg_v1_scores_match_truth_shape():
    dataset = generate_lagged_var(n_nodes=4, timesteps=160, max_lag=2, seed=8, switching=True)
    config = STCGStateConfig(window_size=70, stride=45, seed=12)
    scores = stcg_v1_scores(dataset.x, max_lag=2, config=config)

    assert scores.shape == dataset.edge_truth.shape
    assert np.all(np.diagonal(scores, axis1=1, axis2=2) == 0)


def test_stcg_v1_diagnostics_expose_window_labels():
    dataset = generate_lagged_var(n_nodes=4, timesteps=180, max_lag=2, seed=9, switching=True)
    config = STCGStateConfig(window_size=80, stride=50, seed=13)
    diagnostics = stcg_v1_diagnostics(dataset.x, max_lag=2, config=config)

    assert diagnostics.scores.shape == dataset.edge_truth.shape
    assert diagnostics.window_scores.shape[0] == len(diagnostics.bounds)
    assert diagnostics.features.shape[0] == len(diagnostics.bounds)
    assert diagnostics.labels.shape == (len(diagnostics.bounds),)
    assert diagnostics.state_scores.ndim == 4


def test_auto_state_accepts_temporally_coherent_partition():
    features = np.vstack(
        [
            np.full((20, 3), -1.0),
            np.full((20, 3), 1.0),
        ]
    )
    config = STCGStateConfig(
        n_states=2,
        auto_states=True,
        min_state_improvement=1.10,
        min_temporal_state_improvement=0.01,
        min_temporal_coherence=0.55,
        seed=31,
    )

    labels, improvement, temporal_coherence, accepted = _infer_state_labels(features, config)

    assert accepted is True
    assert improvement >= config.min_temporal_state_improvement
    assert temporal_coherence > 0.94
    assert set(labels) == {0, 1}


def test_auto_state_rejects_temporally_fragmented_partition():
    features = np.array([[float(idx % 2), 0.0] for idx in range(40)])
    config = STCGStateConfig(
        n_states=2,
        auto_states=True,
        min_state_improvement=1.10,
        min_temporal_state_improvement=0.01,
        min_temporal_coherence=0.55,
        seed=32,
    )

    labels, improvement, temporal_coherence, accepted = _infer_state_labels(features, config)

    assert improvement >= config.min_temporal_state_improvement
    assert temporal_coherence == 0.0
    assert accepted is False
    assert set(labels) == {0}


def test_temporal_fallback_rejects_tiny_state():
    features = np.vstack(
        [
            np.full((38, 2), -1.0),
            np.full((2, 2), 1.0),
        ]
    )
    config = STCGStateConfig(
        n_states=2,
        auto_states=True,
        min_state_improvement=1.10,
        min_temporal_state_improvement=0.01,
        min_temporal_coherence=0.55,
        min_temporal_state_fraction=0.10,
        seed=33,
    )

    labels, improvement, temporal_coherence, accepted = _infer_state_labels(features, config)

    assert improvement >= config.min_temporal_state_improvement
    assert temporal_coherence > 0.55
    assert accepted is False
    assert set(labels) == {0}


def test_sliding_window_bounds_include_final_window():
    bounds = sliding_window_bounds(timesteps=220, max_lag=3, window_size=80, stride=60)

    assert bounds[0] == (0, 80)
    assert bounds[-1] == (140, 220)


def test_recovery_summary_contains_main_metrics():
    dataset = generate_lagged_var(n_nodes=5, timesteps=180, max_lag=3, seed=5)
    scores = lagged_correlation_scores(dataset.x, max_lag=3)
    summary = recovery_summary(scores, dataset.edge_truth)

    assert set(summary) >= {
        "edge_auc",
        "pair_auc",
        "precision_at_k",
        "shd",
        "lag_accuracy",
        "true_edges",
        "edge_density",
    }
    assert structural_hamming_distance(scores, dataset.edge_truth) >= 0
    assert 0.0 <= lag_recovery_accuracy(scores, dataset.edge_truth) <= 1.0


def test_forecasting_split_and_ridge_metrics_are_valid():
    dataset = generate_lagged_var(n_nodes=5, timesteps=220, max_lag=3, seed=10)
    split = chronological_lagged_split(dataset.x, max_lag=3, train_fraction=0.7)
    prediction = ridge_forecast(split, alpha=1.0)
    persistence = persistence_forecast(split)
    metrics = forecast_metrics(split.target_test, prediction)

    assert split.design_train.shape[0] > 0
    assert split.design_test.shape[0] == split.target_test.shape[0]
    assert prediction.shape == split.target_test.shape
    assert persistence.shape == split.target_test.shape
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0


def test_graph_feature_mask_selects_budget_per_target():
    scores = np.ones((3, 4, 4), dtype=float)
    for node in range(4):
        scores[:, node, node] = 0.0
    mask = graph_feature_mask(scores, fraction=0.25)

    assert mask.shape == (12, 4)
    assert np.all(mask.sum(axis=0) == 3)
    for target in range(4):
        self_rows = [lag * 4 + target for lag in range(3)]
        assert not np.any(mask[self_rows, target])


def test_graph_feature_weights_keep_dense_shape_and_self_weights():
    scores = np.zeros((2, 3, 3), dtype=float)
    scores[0, 1, 0] = 2.0
    scores[1, 2, 0] = 1.0

    weights = graph_feature_weights(scores, min_weight=0.2, self_weight=1.0)

    assert weights.shape == (6, 3)
    assert weights[1, 0] == 1.0
    assert np.isclose(weights[5, 0], 0.6)
    for target in range(3):
        self_rows = [lag * 3 + target for lag in range(2)]
        assert np.all(weights[self_rows, target] == 1.0)
    assert np.all(weights > 0.0)


def test_weighted_ridge_matches_dense_ridge_with_unit_weights():
    dataset = generate_lagged_var(n_nodes=4, timesteps=180, max_lag=2, seed=14)
    split = chronological_lagged_split(dataset.x, max_lag=2, train_fraction=0.7)

    dense = ridge_forecast(split, alpha=1.0)
    weighted = ridge_forecast_weighted(
        split,
        alpha=1.0,
        feature_weights=np.ones((split.design_train.shape[1], split.n_nodes)),
    )

    np.testing.assert_allclose(weighted, dense)


def test_weighted_ridge_rejects_bad_feature_weights():
    dataset = generate_lagged_var(n_nodes=4, timesteps=180, max_lag=2, seed=15)
    split = chronological_lagged_split(dataset.x, max_lag=2, train_fraction=0.7)

    try:
        ridge_forecast_weighted(split, feature_weights=np.ones((2, 2)))
    except ValueError as exc:
        assert "feature_weights must have shape" in str(exc)
    else:
        raise AssertionError("Expected invalid shape to raise ValueError")

    weights = np.ones((split.design_train.shape[1], split.n_nodes))
    weights[0, 0] = 0.0
    try:
        ridge_forecast_weighted(split, feature_weights=weights)
    except ValueError as exc:
        assert "feature_weights must be positive" in str(exc)
    else:
        raise AssertionError("Expected zero weight to raise ValueError")


def test_stcg_v1_state_features_default_matches_preflag_scores():
    # Regression fixture captured before the state_features flag existed:
    # the default path must stay byte-identical (single-state fallback case).
    dataset = generate_lagged_var(n_nodes=4, timesteps=180, max_lag=2, seed=9, switching=True)
    config = STCGStateConfig(window_size=80, stride=50, seed=13)
    scores = stcg_v1_scores(dataset.x, max_lag=2, config=config)

    assert config.state_features == "nonself"
    assert float(scores.sum()) == 3.9841258724222235
    np.testing.assert_allclose(
        scores.ravel()[:6],
        [0.0, 0.16094627823319757, 0.12592563766259365, 0.06511419461829128, 0.13447723722816488, 0.0],
    )


def test_stcg_v1_state_features_default_matches_preflag_accepted_states():
    # Same regression guard for the multi-state path (gate accepts K=2).
    dataset = generate_lagged_var(n_nodes=4, timesteps=400, max_lag=2, seed=21, switching=True)
    config = STCGStateConfig(window_size=80, stride=40, seed=14)
    diagnostics = stcg_v1_diagnostics(dataset.x, max_lag=2, config=config)

    assert diagnostics.states_accepted is True
    assert float(diagnostics.scores.sum()) == 3.043050779939663
    np.testing.assert_allclose(
        diagnostics.scores.ravel()[:6],
        [0.0, 0.10128886762625278, 0.13483801857370734, 0.14510194061739498, 0.06403343540097302, 0.0],
    )


def test_window_state_features_modes():
    dataset = generate_lagged_var(n_nodes=4, timesteps=160, max_lag=2, seed=16, switching=True)
    bounds = tuple(sliding_window_bounds(160, 2, 70, 45))
    rng_scores = [np.abs(np.random.default_rng(idx).random((2, 4, 4))) for idx, _ in enumerate(bounds)]

    nonself = _window_state_features(dataset.x, bounds, rng_scores, "nonself")
    all_edges = _window_state_features(dataset.x, bounds, rng_scores, "all")
    with_stats = _window_state_features(dataset.x, bounds, rng_scores, "nonself_stats")

    n_windows = len(bounds)
    assert nonself.shape == (n_windows, 2 * (4 * 4 - 4))
    assert all_edges.shape == (n_windows, 2 * 4 * 4)
    assert with_stats.shape == (n_windows, 2 * (4 * 4 - 4) + 2 * 4)
    np.testing.assert_allclose(nonself, _window_state_features(dataset.x, bounds, rng_scores, "nonself"))

    try:
        _window_state_features(dataset.x, bounds, rng_scores, "bogus")
    except ValueError as exc:
        assert "Unknown state_features mode" in str(exc)
    else:
        raise AssertionError("Expected invalid state_features mode to raise ValueError")
