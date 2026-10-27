"""Tests for the controlled mechanism generator."""

from __future__ import annotations

import numpy as np
import pytest

from stcg.mechanism import (
    CONTROL_KINDS,
    MechanismConfig,
    generate_mechanism,
    misassign_labels,
    spectral_radius,
)


def test_states_are_stable():
    for seed in range(5):
        data = generate_mechanism(MechanismConfig(seed=seed, overlap=0.3))
        assert np.all(data.spectral_radius <= 0.9001)


def test_spectral_radius_matches_companion_matrix():
    coefficients = np.zeros((2, 2, 2))
    coefficients[0, 0, 1] = 0.5
    coefficients[1, 1, 0] = 0.25
    companion = np.array(
        [
            [0.0, 0.0, 0.0, 0.25],
            [0.5, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    expected = float(np.max(np.abs(np.linalg.eigvals(companion))))
    assert spectral_radius(coefficients) == pytest.approx(expected)


def test_overlap_controls_realised_shared_fraction():
    high = generate_mechanism(MechanismConfig(seed=0, overlap=1.0, control="same_support"))
    low = generate_mechanism(MechanismConfig(seed=0, overlap=0.0))
    assert high.overlap_realised == pytest.approx(1.0)
    assert low.overlap_realised == pytest.approx(0.0, abs=1e-12)


def test_same_support_control_keeps_support_and_changes_coefficients():
    data = generate_mechanism(MechanismConfig(seed=1, control="same_support"))
    assert np.array_equal(data.state_support[0], data.state_support[1])
    assert not np.allclose(data.coefficients[0], data.coefficients[1])


def test_sign_flip_control_preserves_absolute_coefficients():
    data = generate_mechanism(MechanismConfig(seed=2, control="sign_flip"))
    assert np.allclose(np.abs(data.coefficients[0]), np.abs(data.coefficients[1]))
    assert np.allclose(data.coefficients[1], -data.coefficients[0])


def test_lag_shift_control_preserves_pair_support_but_changes_lag():
    data = generate_mechanism(MechanismConfig(seed=3, control="lag_shift"))
    pair_state0 = data.state_support[0].max(axis=0)
    pair_state1 = data.state_support[1].max(axis=0)
    assert np.array_equal(pair_state0, pair_state1)
    assert not np.array_equal(data.state_support[0], data.state_support[1])


def test_nuisance_controls_keep_coefficients_identical():
    for control in ("variance_only", "mean_only", "trend"):
        data = generate_mechanism(MechanismConfig(seed=4, control=control))
        assert np.allclose(data.coefficients[0], data.coefficients[1]), control
        assert np.array_equal(data.state_support[0], data.state_support[1]), control


def test_stationary_control_has_one_state():
    data = generate_mechanism(MechanismConfig(seed=5, control="stationary"))
    assert data.coefficients.shape[0] == 1
    assert np.unique(data.regimes).tolist() == [0]
    assert data.n_switches == 0


def test_state_proportions_change_durations():
    balanced = generate_mechanism(
        MechanismConfig(seed=6, timesteps=1200, switch_period=300, state_proportions=(0.5, 0.5))
    )
    rare = generate_mechanism(
        MechanismConfig(seed=6, timesteps=1200, switch_period=300, state_proportions=(0.95, 0.05))
    )
    assert rare.durations[1] < balanced.durations[1]
    assert rare.durations[1] / sum(rare.durations.values()) < 0.15


def test_misassign_labels_respects_rate():
    labels = np.array([0, 1] * 500)
    for eta in (0.0, 0.2, 0.5):
        corrupted = misassign_labels(labels, eta, seed=0)
        rate = float(np.mean(corrupted != labels))
        assert rate == pytest.approx(eta, abs=0.05)
        if eta == 0.0:
            assert np.array_equal(corrupted, labels)


def test_control_kinds_are_all_generatable():
    for control in CONTROL_KINDS:
        data = generate_mechanism(MechanismConfig(seed=7, timesteps=400, control=control))
        assert data.x.shape == (400, 8)
        assert data.union_support.shape == (3, 8, 8)


def test_observation_noise_degrades_without_changing_truth():
    clean = generate_mechanism(MechanismConfig(seed=11))
    noisy = generate_mechanism(MechanismConfig(seed=11, observation_noise=0.5))
    assert np.array_equal(clean.state_support, noisy.state_support)
    assert np.allclose(clean.coefficients, noisy.coefficients)
    assert not np.allclose(clean.x, noisy.x)
    assert noisy.x.std() > clean.x.std()


def test_innovation_scale_is_not_an_effective_axis():
    """Scaling innovation noise rescales the series but not its standardised form."""

    small = generate_mechanism(MechanismConfig(seed=12, noise_scale=0.05))
    large = generate_mechanism(MechanismConfig(seed=12, noise_scale=0.20))
    z_small = (small.x - small.x.mean(0)) / small.x.std(0)
    z_large = (large.x - large.x.mean(0)) / large.x.std(0)
    assert np.allclose(z_small, z_large, atol=1e-6)


def test_structure_counts_separate_dynamics_from_support():
    from stcg.mechanism import structure_counts

    cases = {
        "same_support": (2, 1),
        "sign_flip": (2, 1),
        "lag_shift": (2, 2),
        "variance_only": (2, 1),
        "mean_only": (2, 1),
        "trend": (2, 1),
        "stationary": (1, 1),
    }
    for control, expected in cases.items():
        data = generate_mechanism(MechanismConfig(seed=8, timesteps=400, control=control))
        counts = structure_counts(data)
        assert (counts["k_dyn"], counts["k_graph"]) == expected, control
