"""Numerical-correctness tests for scoring conventions and leakage control.

These tests pin down the repository's ``(lag, source, target)`` convention for
lagged edge scores, the agreement of ``edge_auc`` with sklearn's ROC AUC
implementation, and the leakage-free behaviour of the chronological
forecasting split. They intentionally exercise small, fast code paths only.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from stcg import (
    chronological_lagged_split,
    edge_auc,
    granger_f_scores,
    lagged_correlation_scores,
    pcmci_parcorr_scores,
    ridge_var_scores,
)


def _nonself_flatten(values: np.ndarray) -> np.ndarray:
    """Flatten a (lags, nodes, nodes) tensor without self edges, independently."""

    _, n_nodes, _ = values.shape
    mask = ~np.eye(n_nodes, dtype=bool)[None, :, :]
    return values[np.broadcast_to(mask, values.shape)]


def test_edge_auc_matches_sklearn_across_seeds():
    for seed in range(6):
        rng = np.random.default_rng(seed)
        max_lag, n_nodes = 3, 4
        scores = rng.normal(size=(max_lag, n_nodes, n_nodes))
        for node in range(n_nodes):
            scores[:, node, node] = 0.0
        truth = np.zeros((max_lag, n_nodes, n_nodes), dtype=int)
        truth[rng.random((max_lag, n_nodes, n_nodes)) < 0.25] = 1
        for node in range(n_nodes):
            truth[:, node, node] = 0
        # Guarantee both classes are present among non-self entries.
        truth[0, 0, 1] = 1
        truth[0, 1, 0] = 0

        expected = roc_auc_score(_nonself_flatten(truth), _nonself_flatten(scores))
        actual = edge_auc(scores, truth)
        assert actual == pytest.approx(expected, abs=1e-12), (
            f"seed {seed}: edge_auc={actual!r} != sklearn roc_auc_score={expected!r}"
        )


def test_edge_auc_all_ties_is_one_half():
    max_lag, n_nodes = 2, 3
    scores = np.full((max_lag, n_nodes, n_nodes), 0.5)
    truth = np.zeros((max_lag, n_nodes, n_nodes), dtype=int)
    truth[0, 0, 1] = 1
    truth[1, 2, 0] = 1

    expected = roc_auc_score(_nonself_flatten(truth), _nonself_flatten(scores))
    actual = edge_auc(scores, truth)

    assert expected == pytest.approx(0.5)
    assert actual == pytest.approx(0.5, abs=1e-12)


def _single_edge_series(timesteps: int = 1000, seed: int = 7) -> np.ndarray:
    """Four nodes with one directed edge: node 2 at t-2 drives node 0 at t.

    Node 0 keeps substantial own noise so that the lag-1 feature of node 0 and
    the lag-3 feature of node 2 are only moderately collinear; an
    (almost) deterministic link would make the ridge design nearly singular
    and produce large spurious coefficients on collinear columns.
    """

    rng = np.random.default_rng(seed)
    driver = rng.normal(0.0, 1.0, size=timesteps)
    x = rng.normal(0.0, 1.0, size=(timesteps, 4))
    x[:, 2] = driver
    x[2:, 0] = 0.6 * driver[:-2] + 0.8 * x[2:, 0]
    return x


@pytest.mark.parametrize(
    "scorer",
    [
        pytest.param(ridge_var_scores, id="ridge_var_scores"),
        pytest.param(granger_f_scores, id="granger_f_scores"),
        pytest.param(lagged_correlation_scores, id="lagged_correlation_scores"),
    ],
)
def test_lag_direction_convention_places_known_edge(scorer):
    x = _single_edge_series()
    scores = scorer(x, max_lag=3)

    assert scores.shape == (3, 4, 4)
    # Convention: scores[lag - 1, source, target] scores source(t-lag) -> target(t).
    # The single planted edge node2(t-2) -> node0(t) must be the global maximum
    # at (lag_idx=1, source=2, target=0).
    flat = np.unravel_index(np.argmax(scores), scores.shape)
    assert flat == (1, 2, 0), f"argmax at {flat}, expected (1, 2, 0) for node2(t-2)->node0(t)"
    # And it must stand out clearly above every other entry.
    other = scores.copy()
    other[1, 2, 0] = -np.inf
    assert scores[1, 2, 0] > 2.0 * other.max()


def test_chronological_split_ignores_corrupted_test_segment():
    rng = np.random.default_rng(11)
    timesteps, n_nodes, max_lag = 200, 3, 2
    x = rng.normal(0.0, 1.0, size=(timesteps, n_nodes))
    clean = chronological_lagged_split(x, max_lag=max_lag, train_fraction=0.7)

    x_corrupted = x.copy()
    x_corrupted[clean.train_end :] = 1e6  # extreme corruption, test segment only
    corrupted = chronological_lagged_split(x_corrupted, max_lag=max_lag, train_fraction=0.7)

    # Train-side statistics and features must be bit-for-bit unchanged.
    np.testing.assert_array_equal(corrupted.design_train, clean.design_train)
    np.testing.assert_array_equal(corrupted.target_train, clean.target_train)
    np.testing.assert_array_equal(corrupted.target_mean, clean.target_mean)
    np.testing.assert_array_equal(corrupted.target_std, clean.target_std)
    assert corrupted.train_end == clean.train_end
    # Sanity: the corruption really did reach the test design matrices.
    assert not np.array_equal(corrupted.raw_design_test, clean.raw_design_test)


class _FakePCMCI:
    """Stand-in for tigramite's PCMCI returning a fixed p/val matrix pair.

    Matrices follow the tigramite convention ``(source, target, tau)`` with
    ``tau`` ranging from 0 to ``max_lag`` inclusive.
    """

    def __init__(self, val_matrix: np.ndarray, p_matrix: np.ndarray):
        self._val_matrix = val_matrix
        self._p_matrix = p_matrix

    def run_pcmci(self, tau_min, tau_max, pc_alpha):
        return {"val_matrix": self._val_matrix, "p_matrix": self._p_matrix}


def _install_fake_pcmci(monkeypatch, val_matrix: np.ndarray, p_matrix: np.ndarray) -> None:
    import tigramite.data_processing as pp
    import tigramite.independence_tests.parcorr as parcorr_module
    import tigramite.pcmci as pcmci_module

    class _FakeDataFrame:
        def __init__(self, data):
            self.data = data

    class _FakeParCorr:
        def __init__(self, **kwargs):
            pass

    fake = _FakePCMCI(val_matrix, p_matrix)

    def _fake_pcmci_ctor(dataframe, cond_ind_test, verbosity=0):
        return fake

    monkeypatch.setattr(pp, "DataFrame", _FakeDataFrame)
    monkeypatch.setattr(parcorr_module, "ParCorr", _FakeParCorr)
    monkeypatch.setattr(pcmci_module, "PCMCI", _fake_pcmci_ctor)


def _fake_pcmci_matrices(max_lag: int = 3, n_nodes: int = 4):
    # tigramite layout: (source, target, tau), tau = 0..max_lag.
    val = np.full((n_nodes, n_nodes, max_lag + 1), 0.01)
    p = np.full((n_nodes, n_nodes, max_lag + 1), 0.9)
    # Known link: source=2 -> target=0 at tau=2 with a strong negative statistic.
    val[2, 0, 2] = -0.8
    p[2, 0, 2] = 1e-6
    # A weaker link the other way at tau=1, to check direction is not flipped.
    val[0, 2, 1] = 0.2
    p[0, 2, 1] = 0.05
    return val, p


def test_pcmci_wrapper_maps_source_target_tau_to_lag_source_target(monkeypatch):
    max_lag, n_nodes = 3, 4
    val, p = _fake_pcmci_matrices(max_lag, n_nodes)
    _install_fake_pcmci(monkeypatch, val, p)

    x = np.random.default_rng(0).normal(size=(60, n_nodes))
    scores = pcmci_parcorr_scores(x, max_lag=max_lag, score="statistic")

    assert scores.shape == (max_lag, n_nodes, n_nodes)
    # (source=2, target=0, tau=2) must land at (lag_idx=1, source=2, target=0)
    # with its absolute value, and remain the global maximum.
    assert scores[1, 2, 0] == pytest.approx(0.8, abs=1e-12)
    flat = np.unravel_index(np.argmax(scores), scores.shape)
    assert flat == (1, 2, 0)
    # The reverse-direction tau=1 link must appear at (lag_idx=0, source=0, target=2),
    # not mirrored.
    assert scores[0, 0, 2] == pytest.approx(0.2, abs=1e-12)
    assert scores[0, 2, 0] == pytest.approx(0.01, abs=1e-12)
    # tau=0 (contemporaneous) entries must be dropped entirely.
    assert not np.any(np.isclose(scores, np.abs(val[:, :, 0]).max())) or abs(val[:, :, 0].max()) <= 0.01
    # Self edges are zeroed regardless of the matrix contents.
    for node in range(n_nodes):
        assert np.all(scores[:, node, node] == 0.0)


def test_pcmci_wrapper_p_value_score_uses_negative_log10(monkeypatch):
    max_lag, n_nodes = 3, 4
    val, p = _fake_pcmci_matrices(max_lag, n_nodes)
    _install_fake_pcmci(monkeypatch, val, p)

    x = np.random.default_rng(0).normal(size=(60, n_nodes))
    scores = pcmci_parcorr_scores(x, max_lag=max_lag, score="p_value")

    # -log10(1e-6) = 6 at the planted strong link; -log10(0.05) ~= 1.301 at the weak one.
    assert scores[1, 2, 0] == pytest.approx(6.0, abs=1e-9)
    assert scores[0, 0, 2] == pytest.approx(-np.log10(0.05), abs=1e-9)
    flat = np.unravel_index(np.argmax(scores), scores.shape)
    assert flat == (1, 2, 0)
