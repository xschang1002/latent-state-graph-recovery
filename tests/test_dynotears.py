"""Tests for the DYNOTEARS baseline port (``stcg.dynotears``).

A tiny synthetic series (T=300, n=4) with one planted lag-1 edge checks the
score-tensor shape, self-edge zeroing, the (lag, source, target) direction
convention, and acyclicity of the fitted contemporaneous matrix W. The fits
take milliseconds, so the real optimizer runs unpatched.
"""

import numpy as np

from stcg import dynotears_scores
from stcg.dynotears import acyclicity_h


def _planted_edge_series(timesteps: int = 300, n_nodes: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(timesteps, n_nodes))
    for t in range(1, timesteps):
        x[t, 2] += 0.9 * x[t - 1, 1]  # source 1 -> target 2 at lag 1
    return x


def test_dynotears_scores_shape_and_finite():
    x = _planted_edge_series()
    scores = dynotears_scores(x, max_lag=3)

    assert scores.shape == (3, 4, 4)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)


def test_dynotears_self_edges_zeroed():
    x = _planted_edge_series()
    scores = dynotears_scores(x, max_lag=3)

    assert np.all(np.diagonal(scores, axis1=1, axis2=2) == 0)


def test_dynotears_direction_convention():
    x = _planted_edge_series()
    scores = dynotears_scores(x, max_lag=3)

    # The planted source 1 -> target 2 edge at lag 1 is the top-scoring edge.
    assert np.unravel_index(scores.argmax(), scores.shape) == (0, 1, 2)
    # ... and dominates the reverse direction.
    assert scores[0, 1, 2] > scores[0, 2, 1]


def test_dynotears_w_is_acyclic():
    x = _planted_edge_series()
    _, info = dynotears_scores(x, max_lag=3, return_info=True)

    w_mat = info["w_matrix"]
    assert w_mat.shape == (4, 4)
    assert np.all(np.diag(w_mat) == 0)
    assert abs(acyclicity_h(w_mat)) < 1e-6
    assert info["converged"]
    assert info["h_value"] <= 1e-6
