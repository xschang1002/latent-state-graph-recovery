"""Tests for the confirmatory statistics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from stcg.stats import bootstrap_ci, holm, paired_summary


def test_holm_matches_worked_example():
    adjusted = holm([0.01, 0.02, 0.03])
    assert adjusted == pytest.approx([0.03, 0.04, 0.04])


def test_holm_is_monotone_and_capped():
    adjusted = holm([0.9, 0.001, 0.5])
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted[2] <= 1.0
    assert adjusted[1] < adjusted[2]


def test_holm_handles_empty_input():
    assert holm([]) == []


def test_bootstrap_interval_contains_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(1.0, 0.5, size=200)
    low, high = bootstrap_ci(values, draws=2000)
    assert low < values.mean() < high
    assert high - low < 0.5


def test_paired_summary_detects_shift():
    rng = np.random.default_rng(1)
    control = rng.normal(0.0, 0.5, size=80)
    treatment = control + 0.2 + rng.normal(0.0, 0.05, size=80)
    result = paired_summary(treatment, control, draws=1000)
    assert result["mean_difference"] == pytest.approx(0.2, abs=0.03)
    assert result["p_value"] < 1e-6
    assert result["ci_low"] > 0.0


def test_paired_summary_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        paired_summary(np.zeros(3), np.zeros(4))
