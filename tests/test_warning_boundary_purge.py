"""Regression checks for future-label and future-sensor isolation in warnings."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("horizon", [1, 3, 5, 10])
def test_future_labels_cannot_change_earlier_split_targets(horizon):
    warning = load_script("run_hydraulic_transition_warning")
    lookback = 12
    labels = ["A"] * 150
    y = warning.transition_targets(labels, horizon)[lookback:]
    train, validation, test = warning.chronological_train_validation_test_split(
        len(y), 0.7, 0.2, horizon=horizon
    )
    # These are cycle positions, rather than positions within the design rows.
    validation_start = lookback + validation[0]
    test_start = lookback + test[0]
    changed_validation = labels.copy()
    changed_validation[validation_start:] = ["B"] * (len(labels) - validation_start)
    changed_test = labels.copy()
    changed_test[test_start:] = ["B"] * (len(labels) - test_start)
    y_changed_validation = warning.transition_targets(changed_validation, horizon)[lookback:]
    y_changed_test = warning.transition_targets(changed_test, horizon)[lookback:]
    np.testing.assert_array_equal(y[train], y_changed_validation[train])
    np.testing.assert_array_equal(y[train], y_changed_test[train])
    np.testing.assert_array_equal(y[validation], y_changed_test[validation])
    assert lookback + train[-1] + horizon < validation_start
    assert lookback + validation[-1] + horizon < test_start
    assert validation[0] - train[-1] - 1 == horizon
    assert test[0] - validation[-1] - 1 == horizon
    # Prove the perturbation would alter supervised targets without the purge.
    assert y_changed_validation[validation[0] - 1] != y[validation[0] - 1]
    assert y_changed_test[test[0] - 1] != y[test[0] - 1]


@pytest.mark.parametrize("runner", ["run_hydraulic_transition_warning", "run_hydraulic_warning_ablation"])
def test_warning_runners_fit_and_calibrate_only_on_purged_targets(monkeypatch, runner):
    module = load_script(runner)
    warning = module.warning if hasattr(module, "warning") else module
    labels = ["A"] * 100
    horizon = 3
    lookback = 5
    n_rows = len(labels) - horizon - lookback
    x = np.arange(n_rows, dtype=float).reshape(-1, 1)
    captured = []

    def design(*args, **kwargs):
        y = warning.transition_targets(labels, horizon)[lookback:]
        return x, x, x, y, list(range(lookback, lookback + n_rows))

    def fit(x_train, y_train, x_validation, x_test):
        captured.append((x_train.copy(), y_train.copy(), x_validation.copy(), x_test.copy()))
        return np.full(len(x_validation), 0.5), np.full(len(x_test), 0.5)

    monkeypatch.setattr(warning, "build_warning_design", design)
    monkeypatch.setattr(warning, "fit_logistic_scores", fit)
    monkeypatch.setattr(warning, "evaluate_method", lambda *args: {})
    if hasattr(module, "annotate_state_diagnostics"):
        monkeypatch.setattr(module, "annotate_state_diagnostics", lambda *args: None)
    args = SimpleNamespace(
        lookback_cycles=lookback, max_lag=1, ridge_alpha=1.0, train_fraction=0.7,
        validation_fraction=0.2, min_test_positives=0, threshold_metric="f1",
    )
    module.evaluate_label_horizon("test", "state", horizon, None, {"state": None}, args, {})
    train, validation, test = warning.chronological_train_validation_test_split(
        n_rows, 0.7, 0.2, horizon=horizon
    )
    assert len(captured) == 5
    for x_train, y_train, x_validation, x_test in captured:
        np.testing.assert_array_equal(x_train[:, 0], x[train, 0])
        np.testing.assert_array_equal(x_validation[:, 0], x[validation, 0])
        np.testing.assert_array_equal(x_test[:, 0], x[test, 0])
        assert len(y_train) == len(train)


def test_purge_rejects_too_short_validation_segment():
    warning = load_script("run_hydraulic_transition_warning")
    with pytest.raises(ValueError, match="more rows than the target horizon"):
        warning.chronological_train_validation_test_split(30, 0.7, 0.1, horizon=3)


def test_missing_sensor_imputation_never_uses_future_values(tmp_path):
    warning = load_script("run_hydraulic_transition_warning")
    sensors_path = tmp_path / "sensors.csv"
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame({"cycle": [0, 0, 1, 1, 2, 2], "state": ["A"] * 6}).to_csv(labels_path, index=False)
    args = SimpleNamespace(
        input_csv=sensors_path, labels_csv=labels_path,
        columns="sensor", cycle_column="cycle", label_columns=["state"],
    )
    sensor_values = [1.0, 2.0, np.nan, np.nan, 9.0, 10.0]
    pd.DataFrame({"sensor": sensor_values}).to_csv(sensors_path, index=False)
    before, _, _ = warning.load_cycle_tables(args)
    sensor_values[4:] = [900.0, 1000.0]
    pd.DataFrame({"sensor": sensor_values}).to_csv(sensors_path, index=False)
    after, _, _ = warning.load_cycle_tables(args)
    pd.testing.assert_frame_equal(before.iloc[:2], after.iloc[:2])
    assert before.loc[1, "mean_sensor"] == 2.0
    assert before.loc[1, "std_sensor"] == 0.0


def test_complete_sensor_data_preserve_original_aggregation(tmp_path):
    warning = load_script("run_hydraulic_transition_warning")
    sensors_path = tmp_path / "sensors.csv"
    labels_path = tmp_path / "labels.csv"
    sensors = pd.DataFrame({"sensor": [1.0, 3.0, 5.0, 7.0]})
    sensors.to_csv(sensors_path, index=False)
    pd.DataFrame({"cycle": [0, 0, 1, 1], "state": ["A"] * 4}).to_csv(labels_path, index=False)
    args = SimpleNamespace(
        input_csv=sensors_path, labels_csv=labels_path,
        columns="sensor", cycle_column="cycle", label_columns=["state"],
    )
    actual, _, _ = warning.load_cycle_tables(args)
    np.testing.assert_allclose(actual["mean_sensor"], [2.0, 6.0])
    np.testing.assert_allclose(actual["std_sensor"], [np.sqrt(2.0), np.sqrt(2.0)])
    sensors.loc[0, "sensor"] = np.nan
    sensors.to_csv(sensors_path, index=False)
    with pytest.raises(ValueError, match="causal forward fill"):
        warning.load_cycle_tables(args)


@pytest.mark.parametrize("objective", ["stable", "transition"])
def test_warning_rejects_zero_score_tie_to_actual_one_state_features(monkeypatch, objective):
    warning = load_script("run_hydraulic_transition_warning")
    args = SimpleNamespace(
        stcg_state_counts=[1, 2], stcg_selection_objective=objective,
        stcg_min_improvement=0.04, stcg_min_temporal_coherence=0.55,
        stcg_min_cluster_fraction=0.05, stcg_seed=70000,
        stcg_selection_policy="reject_nonpositive",
    )
    monkeypatch.setattr(warning, "stcg_config", lambda args, state_count, seed: state_count)

    def diagnostic(window, max_lag, config):
        return SimpleNamespace(
            labels=np.arange(8) % config, scores=np.full((1, 2, 2), float(config)),
            state_improvement=0.01, temporal_coherence=0.1 if config == 1 else 0.9,
        )

    monkeypatch.setattr(warning, "stcg_v1_diagnostics", diagnostic)
    features = warning.stcg_selected_lookback_features(np.ones((20, 2)), 15, 10, 1, args)
    assert features[-7] == 1  # Actual fitted candidate K, including its graph features.
    assert features[0] == 1
    args.stcg_selection_policy = "legacy"
    legacy = warning.stcg_selected_lookback_features(np.ones((20, 2)), 15, 10, 1, args)
    assert legacy[-7] == 2
    assert legacy[0] == 2
    args.stcg_selection_policy = "reject_nonpositive"
    args.stcg_state_counts = [2]
    with pytest.raises(ValueError, match="include a fitted K=1"):
        warning.stcg_selected_lookback_features(np.ones((20, 2)), 15, 10, 1, args)


def test_warning_ablation_preserves_selection_policy_when_building_configs():
    ablation = load_script("run_hydraulic_warning_ablation")
    args = ablation.build_warning_args(SimpleNamespace(stcg_selection_policy="legacy"), "stable", 40, 24, 8)
    assert args.stcg_selection_policy == "legacy"
