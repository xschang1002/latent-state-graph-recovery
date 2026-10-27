"""Schedule, pairing, retention and inference regressions; no study fits run."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_schedule_robustness as study


@pytest.mark.parametrize("duration,actual,centres", [(300, 12, 11), (600, 6, 5)])
def test_fixed_schedule_keeps_original_full_origin_and_burn_in(duration, actual, centres):
    full, definition = study.make_schedule(3700, duration, "fixed_phase", np.random.default_rng(1))
    np.testing.assert_array_equal(full, np.arange(3700) // duration % 2)
    config = study.load_config()
    diagnostic, windows = study.describe_schedule(full, study._bounds(config), 100)
    assert len(windows) == 23
    assert diagnostic["n_actual_switches"] == actual
    assert diagnostic["n_centre_switches"] == centres
    assert diagnostic["centre_state0_windows"] + diagnostic["centre_state1_windows"] == 23
    expected = [int((row["centre"] + 100) // duration % 2) for row in windows]
    assert diagnostic["centre_label_sequence"] == expected
    assert definition["phase_full"] == 0
    assert diagnostic["first_observed_segment_left_truncated"]


def test_random_phase_draws_over_full_two_state_cycle():
    class Phase:
        def integers(self, low, high):
            assert (low, high) == (0, 600)
            return 599
    full, definition = study.make_schedule(1500, 300, "random_phase", Phase())
    np.testing.assert_array_equal(full, (np.arange(1500) + 599) // 300 % 2)
    assert full[0] == 1 and definition["phase_full"] == 599
    assert definition["segments_full"][0]["left_truncated_at_origin"]


def test_irregular_durations_are_independent_inclusive_draws_and_last_segment_is_censored():
    class Durations:
        def __init__(self):
            self.values = iter([6, 10, 7, 9])
        def integers(self, low, high):
            assert (low, high) == (6, 11)
            return next(self.values)
    full, definition = study.make_schedule(30, 8, "irregular", Durations())
    assert [row["nominal_duration"] for row in definition["segments_full"]] == [6, 10, 7, 9]
    assert [row["state"] for row in definition["segments_full"]] == [0, 1, 0, 1]
    assert definition["segments_full"][-1]["stop_full"] == 30
    assert definition["segments_full"][-1]["right_truncated_at_end"]
    np.testing.assert_array_equal(full, np.repeat([0, 1, 0, 1], [6, 10, 7, 7]))


def test_streams_are_reproducible_domain_separated_and_independent_of_cell_order():
    expected = study.domain_seed(60000, "schedule", 600, "irregular")
    study.domain_seed(60000, "schedule", 300, "random_phase")
    assert study.domain_seed(60000, "schedule", 600, "irregular") == expected
    seeds = {study.domain_seed(seed, domain, *parts) for seed in (60000, 60001)
             for domain, parts in (("base_inputs", ()), ("clustering", ()),
                                   ("schedule", (300, "random_phase")), ("schedule", (600, "random_phase")))}
    assert len(seeds) == 8
    assert 0 <= study.domain_seed(60000, "clustering", bits=32) < 2**32
    assert np.random.default_rng(np.random.SeedSequence([expected, 2])).integers(0, 10) >= 0


def test_purity_ties_and_center_convention_match_public_evaluation():
    from stcg.evaluation import window_purity, window_regime_labels
    full = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1])
    bounds = np.array([[0, 4], [2, 6], [4, 8]])
    diagnostic, windows = study.describe_schedule(full, bounds, 1)
    purity, majority = window_purity(bounds, full[1:])
    assert diagnostic["centre_label_sequence"] == window_regime_labels(bounds, full[1:]).tolist()
    np.testing.assert_allclose([row["purity"] for row in windows], purity)
    assert [row["majority_label"] for row in windows] == majority.tolist()
    with pytest.raises(ValueError, match="inside"):
        study.describe_schedule(full, np.array([[0, 99]]), 1)


class FakeBackend:
    """Numeric fixtures only: six calls are orchestration, never estimators."""
    def __init__(self, fail_simulation=False, mutate=False):
        self.draws = 0
        self.calls = []
        self.fail_simulation = fail_simulation
        self.mutate = mutate
    def draw_base_inputs(self, config, seed):
        self.draws += 1
        shape = (2, config.max_lag, config.n_nodes, config.n_nodes)
        support = np.zeros(shape, dtype=np.int8)
        support[0, 0, 0, 1] = 1; support[1, 0, 1, 0] = 1
        self.inputs = {"coefficients": support * 0.1, "support": support,
                       "initial_values": np.zeros((config.max_lag, config.n_nodes)),
                       "innovations": np.zeros((config.timesteps + config.burn_in, config.n_nodes)),
                       "spectral_radius": np.array([0.1, 0.1])}
        return self.inputs
    def simulate_schedule(self, inputs, full, config):
        self.calls.append((id(inputs), study.array_fingerprints(inputs), full.copy()))
        if self.fail_simulation:
            raise ValueError("fixture nonfinite simulation")
        if self.mutate:
            inputs["innovations"][0, 0] += 1
        return SimpleNamespace(x=np.zeros((config.timesteps, config.n_nodes)),
                               regimes=full[config.burn_in:], state_support=inputs["support"],
                               coefficients=inputs["coefficients"], spectral_radius=inputs["spectral_radius"], config=config)
    def fit_aggregation(self, dataset, **kwargs):
        assert kwargs["exact_random"] is False and kwargs["scorer"] == "lasso"
        bounds = np.array([(start, start + 300) for start in range(0, len(dataset.x) - 299, 150)])
        accepted = dataset.config.switch_period == 600
        scores = np.zeros((len(bounds), *dataset.state_support.shape[1:]))
        rows = [{"method": name, "temporal_delta_auroc": 0.6 if name == "raw_window" else
                  (0.55 if name == "forced_hard" or (name == "native_hard" and accepted) else 0.5),
                 "J": 0.2, "C_edge": 0.5, "J_times_C": 0.1, "temporal_delta_edges": 2,
                 "temporal_delta_edges_total": 2, "hard_gate_accepted": accepted} for name in study.METHODS]
        return rows, {"window_scores": scores, "bounds": bounds,
                      "true_window_labels": dataset.regimes[(bounds[:, 0] + bounds[:, 1] - 1) // 2]}, {"hard_gate_accepted": accepted}


def test_one_shared_base_draw_six_cells_and_numeric_retention(tmp_path):
    config = study.load_config()
    backend = FakeBackend()
    records = study.run_repeat(config, 60000, tmp_path, backend)
    assert backend.draws == 1 and len(backend.calls) == len(records) == 6
    assert len({call[0] for call in backend.calls}) == 1
    assert all(call[1] == backend.calls[0][1] for call in backend.calls)
    assert len({record["identity"]["cluster_seed"] for record in records}) == 1
    assert len({record["identity"]["schedule_seed"] for record in records}) == 6
    for record in records:
        identity = record["identity"]
        npz = tmp_path / "parts" / f"repeat_60000_{identity['cell']}.npz"
        with np.load(npz, allow_pickle=False) as saved:
            assert saved["window_scores"].shape[0] == 23
            assert saved["regimes_full"].shape == (3700,)
            assert saved["observed_series"].shape == (3600, 8)
            np.testing.assert_array_equal(saved["coefficients"], backend.inputs["coefficients"])
    resumed = study.run_repeat(config, 60000, tmp_path, backend)
    assert backend.draws == 1 and len(backend.calls) == 6
    assert resumed == records
    summary = study.summarise(records, config)
    gates = summary["schedule_robustness_gate_summary.csv"]
    assert gates.loc[gates.duration_mean == 300, "gate_acceptance"].eq(0).all()
    assert gates.loc[gates.duration_mean == 600, "gate_acceptance"].eq(1).all()


def test_failures_retain_planned_rows_truth_and_exogenous_inputs(tmp_path):
    records = study.run_repeat(study.load_config(), 60000, tmp_path, FakeBackend(fail_simulation=True))
    assert len(records) == 6
    assert all(record["diagnostic"]["status"] == "failed" for record in records)
    assert all(record["diagnostic"]["simulation_completed"] is False for record in records)
    assert all(len(record["methods"]) == 5 for record in records)
    assert all(row["temporal_delta_auroc"] is None for record in records for row in record["methods"])
    assert (tmp_path / "inputs/repeat_60000.npz").is_file()
    with np.load(next((tmp_path / "parts").glob("*.npz")), allow_pickle=False) as saved:
        assert "regimes_full" in saved and "state_support" in saved and "observed_series" not in saved


def test_mutating_shared_exogenous_input_is_a_hard_error(tmp_path):
    with pytest.raises(RuntimeError, match="mutated paired"):
        study.run_repeat(study.load_config(), 60000, tmp_path, FakeBackend(mutate=True))


def test_protocol_default_and_freeze_do_not_run_fits(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(study, "OUT", tmp_path / "unused")
    monkeypatch.setattr(study, "_backend", lambda: pytest.fail("protocol description must not fit"))
    monkeypatch.setattr(sys, "argv", ["run_schedule_robustness.py"])
    study.main()
    proposed = json.loads(capsys.readouterr().out)
    assert proposed["expected_fits"] == 300
    assert proposed["repeats"] == list(range(60000, 60050))
    assert not (tmp_path / "unused").exists()
    with pytest.raises(RuntimeError, match="No frozen"):
        study.ensure_protocol(tmp_path / "frozen", proposed)
    study.ensure_protocol(tmp_path / "frozen", proposed, freeze=True)
    changed = deepcopy(proposed); changed["configuration"]["mechanism"]["timesteps"] = 1200
    with pytest.raises(RuntimeError, match="differs"):
        study.ensure_protocol(tmp_path / "frozen", changed)


def _statistic_records():
    records = []
    for seed in range(60000, 60006):
        for duration in (300, 600):
            for kind in study.KINDS:
                delta = (-0.08 + 0.001 * (seed - 60000)) if duration == 300 else (0.02 + 0.002 * (seed - 60000))
                forced = 0.6 + delta
                base = {"seed": seed, "duration_mean": duration, "schedule": kind, "cell": f"D{duration}_{kind}"}
                metrics = [{**base, "method": method, "temporal_delta_auroc": 0.6 if method == "raw_window" else forced,
                            "fit_status": "ok"} for method in study.METHODS]
                records.append({"methods": metrics, "windows": [], "diagnostic": {**base, "hard_gate_accepted": False,
                                  "J": 0.2, "C_edge": 0.5, "J_times_C": 0.1}})
    return records


def test_gamma_uses_within_repeat_differences_and_secondary_families_stay_separate():
    config = study.load_config(); records = _statistic_records()
    tables = study.summarise(records, config)
    tests = tables["schedule_robustness_secondary_tests.csv"]
    assert len(tests) == 9
    assert tests.groupby("family").size().to_dict() == {"B_duration_gamma": 3, "B_forced_minus_raw": 6}
    gamma = tests[tests.family == "B_duration_gamma"]
    assert gamma.n_pairs.eq(6).all()
    np.testing.assert_allclose(gamma.mean_difference, 0.1025)
    for family, group in tests.groupby("family"):
        np.testing.assert_allclose(group.p_holm, study.holm_fixed(group.p_raw.tolist()))
    missing = next(record for record in records if record["diagnostic"]["seed"] == 60000 and
                   record["diagnostic"]["duration_mean"] == 600 and record["diagnostic"]["schedule"] == "fixed_phase")
    for row in missing["methods"]:
        row["temporal_delta_auroc"] = None; row["fit_status"] = "failed"
    tests = study.summarise(records, config)["schedule_robustness_secondary_tests.csv"]
    row = tests[(tests.family == "B_duration_gamma") & (tests.schedule == "fixed_phase")].iloc[0]
    assert row.n_pairs == 5 and row.mean_difference == pytest.approx(0.103)


def test_holm_retains_missing_test_slot_and_small_samples_have_no_ci():
    adjusted = study.holm_fixed([0.01, np.nan, 0.02, 0.5, 0.6, 0.9])
    assert adjusted[0] == pytest.approx(0.06) and np.isnan(adjusted[1])
    result = study.mean_ci(np.array([0.1, 0.2, np.nan]), study.load_config())
    assert result["n_pairs"] == 2 and np.isnan(result["ci95_low"])
    assert study.mean_ci(np.zeros(5), study.load_config())["p_raw"] == 1


def test_aggregate_refuses_silent_deletion_of_planned_cells(tmp_path):
    with pytest.raises(RuntimeError, match="Missing planned case"):
        study.aggregate(tmp_path, study.load_config())


def test_configuration_rejects_reusing_old_seeds_or_adding_other_scorers(tmp_path):
    config = study.load_config()
    config["seed_start"] = 0
    path = tmp_path / "config.yaml"; path.write_text(study.yaml.safe_dump(config))
    with pytest.raises(ValueError, match="60000"):
        study.load_config(path)
    config = study.load_config(); config["model"]["scorer"] = "ols"
    path.write_text(study.yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Lasso"):
        study.load_config(path)


def test_thread_count_is_frozen_and_execution_uses_threadpool_limit(tmp_path, monkeypatch):
    config = study.load_config()
    assert config["threads"] == 1
    config["threads"] = 2
    path = tmp_path / "config.yaml"; path.write_text(study.yaml.safe_dump(config))
    with pytest.raises(ValueError, match="one thread"):
        study.load_config(path)
    entered = []
    class Limit:
        def __init__(self, limits):
            assert limits == 1
        def __enter__(self):
            entered.append(True)
        def __exit__(self, *args):
            entered.append(False)
    monkeypatch.setattr(study, "threadpool_limits", Limit)
    monkeypatch.setattr(study, "ensure_protocol", lambda *args, **kwargs: None)
    monkeypatch.setattr(study, "run_repeat", lambda *args, **kwargs: pytest.fail("aggregate must not fit"))
    monkeypatch.setattr(study, "aggregate", lambda *args, **kwargs:
                        {"schedule_robustness_secondary_tests.csv": study.pd.DataFrame()})
    monkeypatch.setattr(sys, "argv", ["run_schedule_robustness.py", "--aggregate", "--output-dir", str(tmp_path)])
    study.main()
    assert entered == [True, False]
