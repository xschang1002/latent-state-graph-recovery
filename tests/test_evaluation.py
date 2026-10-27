"""Tests for the three-layer estimand evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from stcg.evaluation import (
    GraphPrediction,
    GraphTruth,
    evaluate_prediction,
    match_states,
    per_regime_metrics,
    temporal_state_metrics,
    union_metrics,
    window_purity,
)


def _tensors():
    """Two states over three nodes and two lags.

    State 0: 0 -> 1 at lag 1, 1 -> 2 at lag 2.
    State 1: 0 -> 1 at lag 1 (shared), 2 -> 0 at lag 1 (state specific).
    """

    n_lags, n_nodes = 2, 3
    state0 = np.zeros((n_lags, n_nodes, n_nodes))
    state1 = np.zeros((n_lags, n_nodes, n_nodes))
    state0[0, 0, 1] = 1.0
    state0[1, 1, 2] = 1.0
    state1[0, 0, 1] = 1.0
    state1[0, 2, 0] = 1.0
    state_support = np.stack([state0, state1], axis=0)
    union = state_support.max(axis=0)
    return state_support, union


def test_union_perfect_recovery():
    state_support, union = _tensors()
    metrics = union_metrics(union.astype(float), union)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["precision_at_k"] == pytest.approx(1.0)
    assert metrics["shd"] == 0


def test_union_metrics_report_nan_for_empty_support():
    _, union = _tensors()
    empty = np.zeros_like(union)
    metrics = union_metrics(empty.astype(float), empty)
    assert np.isnan(metrics["auroc"])
    assert np.isnan(metrics["auprc"])
    assert metrics["true_edges"] == 0
    assert metrics["shd"] == 0


def test_explicit_zero_edge_budget_selects_no_edges():
    _, union = _tensors()
    result = union_metrics(union.astype(float), union, k=0)
    assert result["shd"] == union.sum()


def test_regime_macro_f1_includes_completely_wrong_graphs():
    support, _ = _tensors()
    # All non-self scores rank non-edges above edges in the second graph.
    wrong = 1.0 - support[1]
    result = per_regime_metrics(
        np.stack([support[0], wrong]), support, np.array([0, 1]), np.array([0, 1])
    )
    assert result["regime_f1_macro"] == pytest.approx(0.5)
    assert result["state_edge_f1_micro"] == pytest.approx(0.5)
    all_wrong = per_regime_metrics(1.0 - support, support, np.array([0, 1]), np.array([0, 1]))
    assert all_wrong["regime_f1_macro"] == 0.0
    assert all_wrong["state_edge_f1_micro"] == 0.0


@pytest.mark.parametrize("state_ids", [(3, 5), (-5, 99), (5, 3)])
def test_all_matched_metrics_are_invariant_to_arbitrary_state_ids(state_ids):
    support, _ = _tensors()
    plain_labels = np.array([0, 0, 1, 1])
    labels = np.array([state_ids[i] for i in plain_labels])
    scores = support if state_ids[0] < state_ids[1] else support[::-1]
    plain = per_regime_metrics(support, support, plain_labels, plain_labels)
    renamed = per_regime_metrics(scores, support, labels, plain_labels)
    for key in plain:
        assert renamed[key] == pytest.approx(plain[key], nan_ok=True)
    bounds = [(0, 10), (10, 20), (20, 30), (30, 40)]
    regimes = np.repeat(plain_labels, 10)
    for mode in ("center", "coverage"):
        result = temporal_state_metrics(labels, bounds, regimes, mode=mode)
        assert result["state_macro_f1"] == 1.0
        assert result["state_ari"] == 1.0


def test_state_macro_f1_penalizes_unmatched_states_without_label_collisions():
    bounds = [(0, 10), (10, 20), (20, 30), (30, 40)]
    regimes = np.repeat([0, 0, 1, 1], 10)
    plain = temporal_state_metrics(np.array([0, 2, 1, 1]), bounds, regimes)
    renamed = temporal_state_metrics(np.array([-5, 99, 8, 8]), bounds, regimes)
    assert renamed["state_macro_f1"] == pytest.approx(plain["state_macro_f1"])
    assert renamed["state_macro_f1"] < 1.0


def test_union_perfect_but_per_regime_wrong_is_detected():
    """A method can recover the union while getting every regime graph wrong."""

    state_support, union = _tensors()
    window_assignments = np.array([0, 0, 1, 1])
    true_labels = np.array([0, 0, 1, 1])
    state_scores = np.stack([union, union], axis=0).astype(float)

    union_result = union_metrics(union.astype(float), union)
    regime_result = per_regime_metrics(
        state_scores, state_support, window_assignments, true_labels
    )

    assert union_result["auroc"] == pytest.approx(1.0)
    assert regime_result["regime_auprc_macro"] < 0.9
    assert regime_result["regime_f1_macro"] < 1.0


def test_perfect_per_regime_recovery():
    state_support, union = _tensors()
    scores = state_support.astype(float)
    result = per_regime_metrics(
        scores, state_support, np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])
    )
    assert result["regime_auprc_macro"] == pytest.approx(1.0)
    assert result["regime_f1_macro"] == pytest.approx(1.0)
    assert result["regime_shd_mean"] == pytest.approx(0.0)
    assert result["unmatched_predicted_fraction"] == pytest.approx(0.0)


def test_swapped_state_numbering_does_not_change_matched_metrics():
    state_support, _ = _tensors()
    scores = state_support.astype(float)
    true_labels = np.array([0, 0, 1, 1])

    plain = per_regime_metrics(scores, state_support, true_labels, true_labels)
    swapped_labels = np.array([1, 1, 0, 0])
    swapped_scores = np.stack([scores[1], scores[0]], axis=0)
    swapped = per_regime_metrics(swapped_scores, state_support, swapped_labels, true_labels)

    assert swapped["regime_auprc_macro"] == pytest.approx(plain["regime_auprc_macro"])
    assert swapped["regime_f1_macro"] == pytest.approx(plain["regime_f1_macro"])
    assert swapped["regime_shd_mean"] == pytest.approx(plain["regime_shd_mean"])


def test_missing_state_is_reported():
    state_support, union = _tensors()
    scores = union[None, ...].astype(float)
    result = per_regime_metrics(scores, state_support, np.zeros(4, int), np.array([0, 0, 1, 1]))
    assert result["missed_truth_states"] == 1.0
    assert match_states(np.zeros(4, int), np.array([0, 0, 1, 1]))["missed_truth"] == [1]


def test_spurious_state_is_penalised():
    state_support, _ = _tensors()
    scores = np.stack([state_support[0], state_support[1], state_support[0]], axis=0).astype(float)
    labels = np.array([0, 0, 1, 1, 2, 2])
    truth = np.array([0, 1, 0, 1, 0, 1])
    result = per_regime_metrics(scores, state_support, labels, truth)
    assert 0.0 < result["unmatched_predicted_fraction"] < 1.0


def test_window_purity_flags_mixed_windows():
    regimes = np.array([0] * 10 + [1] * 10)
    bounds = [(0, 10), (5, 15), (10, 20)]
    purity, majority = window_purity(bounds, regimes)
    assert purity[0] == pytest.approx(1.0)
    assert purity[1] == pytest.approx(0.5)
    assert purity[2] == pytest.approx(1.0)
    assert majority.tolist() == [0, 0, 1]


def test_temporal_state_metrics_are_invariant_to_label_permutation():
    regimes = np.array([0] * 20 + [1] * 20)
    bounds = [(0, 10), (10, 20), (20, 30), (30, 40)]
    plain = temporal_state_metrics(np.array([0, 0, 1, 1]), bounds, regimes)
    swapped = temporal_state_metrics(np.array([5, 5, 3, 3]), bounds, regimes)
    assert plain["state_ari"] == pytest.approx(1.0)
    assert swapped["state_ari"] == pytest.approx(1.0)
    assert plain["boundary_f1"] == swapped["boundary_f1"] == pytest.approx(1.0)
    assert plain["k_hat_minus_k"] == 0.0


def test_temporal_metrics_penalise_extra_states_and_boundary_error():
    regimes = np.array([0] * 10 + [1] * 10)
    bounds = [(0, 5), (5, 10), (10, 15), (15, 20)]
    result = temporal_state_metrics(np.array([0, 2, 1, 3]), bounds, regimes)
    assert result["k_hat_minus_k"] == 2.0
    assert result["boundary_f1"] < 1.0
    assert result["state_ari"] < 1.0


def test_evaluate_prediction_scores_each_layer():
    state_support, union = _tensors()
    regimes = np.array([0] * 10 + [1] * 10)
    bounds = ((0, 10), (10, 20))
    prediction = GraphPrediction(
        union_scores=union.astype(float),
        window_scores=np.stack([state_support[0], state_support[1]], axis=0).astype(float),
        window_bounds=bounds,
        state_scores=state_support.astype(float),
        window_assignments=np.array([0, 1]),
    )
    truth = GraphTruth(support=union, state_support=state_support, regimes=regimes)
    metrics = evaluate_prediction(prediction, truth)
    for key in (
        "union_auroc",
        "regime_auprc_macro",
        "state_ari",
        "windowed_edge_auroc",
    ):
        assert key in metrics
    assert metrics["union_auroc"] == pytest.approx(1.0)
    assert metrics["state_ari"] == pytest.approx(1.0)
    assert metrics["windowed_edge_auroc"] == pytest.approx(1.0)


def test_prevalence_and_lift_are_reported():
    state_support, union = _tensors()
    metrics = union_metrics(union.astype(float), union)
    assert metrics["prevalence"] == pytest.approx(union.sum() / 12)
    assert metrics["auprc_lift"] == pytest.approx(
        metrics["auprc"] / metrics["prevalence"]
    )
    regime = per_regime_metrics(
        state_support.astype(float), state_support, np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])
    )
    assert 0.0 < regime["regime_prevalence"] < 1.0
    assert regime["regime_auprc_lift"] == pytest.approx(
        regime["regime_auprc_macro"] / regime["regime_prevalence"]
    )


def test_normalized_ap_is_zero_at_prevalence_and_one_at_perfect():
    state_support, union = _tensors()
    perfect = union_metrics(union.astype(float), union)
    assert perfect["auprc_normalized"] == pytest.approx(1.0)
    constant = union_metrics(np.ones_like(union, dtype=float), union)
    assert constant["auprc_normalized"] == pytest.approx(0.0, abs=0.05)


def test_temporal_graph_metric_scores_time_varying_graph():
    from stcg.evaluation import temporal_graph_metrics

    state_support, _ = _tensors()
    regimes = np.array([0] * 10 + [1] * 10)
    bounds = [(0, 10), (10, 20)]
    perfect = temporal_graph_metrics(
        state_support.astype(float), np.array([0, 1]), bounds, regimes, state_support
    )
    assert perfect["temporal_graph_auroc"] == pytest.approx(1.0)

    swapped = temporal_graph_metrics(
        np.stack([state_support[1], state_support[0]]).astype(float),
        np.array([1, 0]),
        bounds,
        regimes,
        state_support,
    )
    assert swapped["temporal_graph_auroc"] == pytest.approx(1.0)

    wrong_time = temporal_graph_metrics(
        state_support.astype(float), np.array([1, 0]), bounds, regimes, state_support
    )
    assert wrong_time["temporal_graph_auroc"] < perfect["temporal_graph_auroc"]


def test_static_temporal_graph_metrics_scores_constant_prediction():
    from stcg.evaluation import static_temporal_graph_metrics

    state_support, union = _tensors()
    regimes = np.array([0] * 10 + [1] * 10)
    bounds = [(0, 10), (10, 20)]
    # A static method predicts the union graph at every window. The union graph
    # is state 0's graph at window 1 (perfect) but not state 1's graph.
    result = static_temporal_graph_metrics(
        union.astype(float), bounds, regimes, state_support.astype(float)
    )
    assert result["temporal_graph_auroc"] < 1.0
    assert np.isfinite(result["temporal_graph_auroc"])


def test_temporal_delta_auc_isolates_changing_edges():
    from stcg.evaluation import temporal_delta_auc

    state_support, _ = _tensors()
    regimes = np.array([0] * 10 + [1] * 10)
    bounds = [(0, 10), (10, 20)]
    # Perfect assignment and perfect state scores -> 1.0 on changing edges.
    perfect = temporal_delta_auc(
        state_support.astype(float),
        np.array([0, 1]),
        bounds,
        regimes,
        state_support.astype(float),
    )
    assert perfect["temporal_delta_auroc"] == pytest.approx(1.0)
    assert perfect["temporal_delta_edges"] > 0

    # A constant predictor scores 0.5 on every changing edge.
    constant_scores = np.stack([state_support[0], state_support[0]]).astype(float)
    constant = temporal_delta_auc(
        constant_scores, np.array([0, 1]), bounds, regimes, state_support.astype(float)
    )
    assert constant["temporal_delta_auroc"] == pytest.approx(0.5)
