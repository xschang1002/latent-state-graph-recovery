"""Tests for the change-point calibrated gate."""
from __future__ import annotations

from stcg import generate_lagged_var
from stcg.change_point_gate import change_point_pvalue


def test_change_point_pvalue_detects_switch():
    dataset = generate_lagged_var(
        n_nodes=8, timesteps=800, max_lag=2, seed=0, switching=True
    )
    statistic, p_value = change_point_pvalue(dataset.x, 2, seed=0, draws=19, step=20)
    assert statistic > 0
    assert 0.0 <= p_value <= 1.0
    assert p_value < 0.1


def test_change_point_pvalue_does_not_flag_stationary():
    dataset = generate_lagged_var(
        n_nodes=8, timesteps=800, max_lag=2, seed=0, switching=False
    )
    statistic, p_value = change_point_pvalue(dataset.x, 2, seed=0, draws=19, step=20)
    assert statistic > 0
    assert p_value > 0.2


def test_change_point_pvalue_is_deterministic():
    dataset = generate_lagged_var(
        n_nodes=8, timesteps=800, max_lag=2, seed=0, switching=True
    )
    first = change_point_pvalue(dataset.x, 2, seed=0, draws=19, step=20)
    second = change_point_pvalue(dataset.x, 2, seed=0, draws=19, step=20)
    assert first == second
