"""Tests for the cMLP neural-Granger baseline (Tank et al. 2021).

The fits run at a tiny budget (hidden=4, T=200) so the suite stays fast
while still checking the score-tensor conventions, determinism, and
direction recovery on a planted lag-2 edge.
"""

import numpy as np
import pytest

from stcg import cmlp_scores

TINY = dict(hidden=4, penalty=0.15, lr=0.05, epochs=150, momentum=0.9)


def _planted_lag2_series(timesteps: int = 200, seed: int = 0) -> np.ndarray:
    """Three-node system whose only cross edge is node 0 -> node 1 at lag 2."""

    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.05, size=(timesteps, 3))
    x[:, 0] = rng.normal(scale=1.0, size=timesteps)  # pure exogenous driver
    for t in range(2, timesteps):
        x[t, 1] = 0.9 * x[t - 2, 0] + rng.normal(scale=0.1)  # lag-2 edge 0 -> 1
        x[t, 2] = 0.5 * x[t - 1, 2] + rng.normal(scale=0.1)  # own AR only
    return x


def test_cmlp_scores_shape_self_edges_and_finiteness():
    x = np.random.default_rng(0).normal(size=(200, 4))
    scores = cmlp_scores(x, max_lag=3, **TINY)

    assert scores.shape == (3, 4, 4)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)
    assert np.all(np.diagonal(scores, axis1=1, axis2=2) == 0)


def test_cmlp_scores_determinism():
    x = np.random.default_rng(1).normal(size=(200, 3))
    first = cmlp_scores(x, max_lag=2, seed=11, **TINY)
    second = cmlp_scores(x, max_lag=2, seed=11, **TINY)
    assert np.array_equal(first, second)

    # A different base seed should give a different fit.
    third = cmlp_scores(x, max_lag=2, seed=12, **TINY)
    assert not np.array_equal(first, third)


def test_cmlp_scores_recovers_planted_lag2_edge_direction():
    scores = cmlp_scores(_planted_lag2_series(), max_lag=3, seed=7, **TINY)

    # The single strongest non-self score is the true edge (lag 2, 0 -> 1).
    assert np.unravel_index(np.argmax(scores), scores.shape) == (1, 0, 1)
    # Direction: the reverse edge (1 -> 0 at lag 2) stays near zero.
    assert scores[1, 0, 1] > scores[1, 1, 0] + 0.3
    # Wrong-lag copies of the true pair score below the true lag.
    assert scores[1, 0, 1] > scores[0, 0, 1]
    assert scores[1, 0, 1] > scores[2, 0, 1]


def test_cmlp_scores_return_info_records_budget():
    scores, info = cmlp_scores(
        _planted_lag2_series(), max_lag=3, seed=3, return_info=True, **TINY
    )

    assert scores.shape == (3, 3, 3)
    assert len(info["final_losses"]) == 3
    assert all(loss >= 0.0 for loss in info["final_losses"])
    assert info["hidden"] == TINY["hidden"]
    assert info["penalty"] == TINY["penalty"]
    assert info["lr"] == TINY["lr"]
    assert info["epochs"] == TINY["epochs"]
    assert info["momentum"] == TINY["momentum"]
    assert info["optimizer"] == "sgd"
    assert info["activation"] == "tanh"
    assert info["n_samples"] == 200 - 3


def test_cmlp_scores_rejects_invalid_inputs():
    x = np.zeros((200, 3))
    with pytest.raises(ValueError, match="timesteps"):
        cmlp_scores(np.zeros((4, 3)), max_lag=3, **TINY)
    with pytest.raises(ValueError, match="activation"):
        cmlp_scores(x, max_lag=3, activation="gelu", **{k: v for k, v in TINY.items() if k != "hidden"})
    with pytest.raises(ValueError, match="penalty"):
        cmlp_scores(x, max_lag=3, penalty=-0.1, **{k: v for k, v in TINY.items() if k != "penalty"})
    with pytest.raises(ValueError, match="optimizer"):
        cmlp_scores(x, max_lag=3, optimizer="rmsprop", **TINY)
