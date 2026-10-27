"""Frozen paired-study regression tests; fixtures are not planned new repeats."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aggregation_study_common as common
import run_duration_confirmation as confirm
from stcg import STCGStateConfig, stcg_v1_diagnostics
from stcg.baselines import _zscore, make_lagged_design
from stcg.evaluation import temporal_delta_auc, temporal_graph_metrics
from stcg.mechanism import MechanismConfig


def fixture_config():
    return MechanismConfig(n_nodes=3, max_lag=1, timesteps=120, burn_in=5,
                           switch_period=30, edge_prob=0.5, overlap=0.5, seed=0)


def fixture_dataset(duration=30):
    config = replace(fixture_config(), switch_period=duration)
    inputs = common.draw_base_inputs(config, 0)
    schedule = common.fixed_schedule(config.timesteps + config.burn_in, duration)
    return common.simulate_schedule(inputs, schedule, config)


def test_common_exogenous_inputs_are_shared_and_repeats_are_independent():
    config = fixture_config()
    first = common.draw_base_inputs(config, 0)
    replay = common.draw_base_inputs(replace(config, switch_period=60), 0)
    other = common.draw_base_inputs(config, 1)
    for key in first:
        np.testing.assert_array_equal(first[key], replay[key])
    assert not np.array_equal(first["innovations"], other["innovations"])
    assert not np.array_equal(first["coefficients"], other["coefficients"])
    for duration in (30, 60):
        schedule = common.fixed_schedule(config.timesteps + config.burn_in, duration)
        dataset = common.simulate_schedule(first, schedule, config)
        np.testing.assert_array_equal(dataset.coefficients, first["coefficients"])
        np.testing.assert_array_equal(dataset.regimes, schedule[config.burn_in:])


def test_explicit_common_innovations_drive_each_different_schedule():
    config = fixture_config(); inputs = common.draw_base_inputs(config, 0)
    # With zero coefficient maps, either schedule must expose identical noise.
    inputs["coefficients"] = np.zeros_like(inputs["coefficients"])
    one = common.simulate_schedule(inputs, common.fixed_schedule(125, 30), config)
    two = common.simulate_schedule(inputs, common.fixed_schedule(125, 60), config)
    np.testing.assert_array_equal(one.x, inputs["innovations"][5:])
    np.testing.assert_array_equal(one.x, two.x)
    assert not np.array_equal(one.regimes, two.regimes)


def test_full_time_phase_and_burn_offset_reproduce_original_schedule_convention():
    schedule = common.fixed_schedule(1300, 300)
    assert schedule[:300].sum() == 0 and schedule[300:600].sum() == 300
    observed = schedule[100:]
    np.testing.assert_array_equal(np.flatnonzero(np.diff(observed) != 0), [199, 499, 799, 1099])
    assert common.fixed_schedule(10, 3, phase=6)[0] == 0


def test_lasso_native_and_forced_match_existing_estimator_on_the_same_fixture():
    dataset = fixture_dataset()
    config = STCGStateConfig(window_size=30, stride=15, seed=0)
    with threadpool_limits(limits=1):
        old = stcg_v1_diagnostics(dataset.x, 1, config)
        rows, arrays, info = common.fit_aggregation(dataset, "lasso", 0, window_size=30, stride=15)
    np.testing.assert_array_equal(arrays["window_scores"], old.window_scores)
    lookup = {int(k): i for i, k in enumerate(np.unique(old.labels))}
    emitted = old.state_scores[[lookup[int(v)] for v in old.labels]]
    np.testing.assert_array_equal(arrays["emission_native_hard"], emitted)
    assert info["hard_gate_accepted"] == old.states_accepted
    assert info["hard_gate_improvement"] == pytest.approx(old.state_improvement, abs=1e-14)
    assert len(rows) == 6
    assert max(r["identity_gap"] for r in rows if np.isfinite(r["identity_gap"])) < 1e-14


def test_vectorised_temporal_targets_preserve_ties_and_match_public_metrics():
    dataset = fixture_dataset()
    bounds = [(0, 30), (15, 45), (30, 60), (45, 75), (60, 90), (75, 105), (90, 120)]
    values = np.random.default_rng(2).integers(0, 3, size=(7, 1, 3, 3)).astype(float)
    actual = common.evaluate_predictions(values, bounds, dataset)
    expected = temporal_delta_auc(values, np.arange(7), bounds, dataset.regimes, dataset.state_support)
    pooled = temporal_graph_metrics(values, np.arange(7), bounds, dataset.regimes, dataset.state_support)
    assert actual["temporal_delta_auroc"] == pytest.approx(expected["temporal_delta_auroc"], abs=1e-15)
    assert actual["temporal_delta_edges"] == expected["temporal_delta_edges"]
    assert actual["pooled_temporal_auroc"] == pytest.approx(pooled["temporal_graph_auroc"], abs=1e-15)


def test_undefined_temporal_target_is_not_imputed_to_half():
    dataset = fixture_dataset()
    dataset = replace(dataset, regimes=np.zeros_like(dataset.regimes))
    bounds = [(0, 30), (15, 45)]
    values = np.ones((2, 1, 3, 3))
    result = common.evaluate_predictions(values, bounds, dataset)
    assert np.isnan(result["temporal_delta_auroc"])
    assert result["temporal_delta_edges"] == 0 and result["undefined_reason"] == "no_observed_flip"


def test_ols_is_strict_full_rank_and_preserves_lasso_preprocessing():
    x = np.random.default_rng(3).normal(size=(40, 3))
    score, diagnostic = common._window_scores(x, 2, [(0, 40)], "ols", 0.005)
    design, target = make_lagged_design(_zscore(x), 2)
    coeff = np.linalg.lstsq(_zscore(design), _zscore(target), rcond=None)[0]
    expected = np.abs(coeff.reshape(2, 3, 3)); index = np.arange(3); expected[:, index, index] = 0
    np.testing.assert_array_equal(score[0], expected)
    assert diagnostic[0]["rank"] == diagnostic[0]["n_features"] == 6
    with pytest.raises(common.ScorerFailure, match="rank deficient") as failure:
        common._window_scores(np.ones((40, 3)), 2, [(0, 40)], "ols", 0.005)
    assert failure.value.diagnostic["windows"][0]["rank"] == 0


def test_exact_random_scores_each_partition_before_averaging():
    dataset = fixture_dataset()
    with threadpool_limits(limits=1):
        rows, arrays, _ = common.fit_aggregation(dataset, "lasso", 0, window_size=30, stride=15)
    random = next(r for r in rows if r["method"] == "matched_random")
    assert random["temporal_delta_auroc"] == pytest.approx(arrays["matched_random_temporal_delta_auroc"].mean())
    assert random["n_partitions"] in (7, 21, 35)
    candidates = arrays["matched_random_partitions"]
    assert len({tuple(row) for row in candidates}) == len(candidates)
    assert all(min(np.bincount(row)) == min(np.bincount(arrays["candidate_labels"])) for row in candidates)


def test_protocol_freezing_does_not_run_simulations_and_refuses_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(confirm, "draw_base_inputs", lambda *a, **k: pytest.fail("Freezing must not generate data"))
    path = tmp_path / "frozen.json"
    protocol = confirm.freeze_protocol(path)
    confirm.validate_protocol(protocol)
    assert protocol["repeat_seeds"] == list(range(50000, 50100))
    assert protocol["intervals"]["seed"] == 20261024
    assert len(protocol["primary_family"]["contrasts"]) == 4
    with pytest.raises(FileExistsError):
        confirm.freeze_protocol(path)
    changed = deepcopy(protocol); changed["repeat_seeds"].append(50100)
    with pytest.raises(ValueError, match="differ"):
        confirm.validate_protocol(changed)
    changed = deepcopy(protocol); changed["source_sha256"]["src/stcg/mechanism.py"] = "changed"
    with pytest.raises(ValueError, match="differ"):
        confirm.validate_protocol(changed)


def test_retained_json_keeps_sub_decimal_precision_and_explicit_missing_values():
    tiny = 2.220446049250313e-16
    values = {"identity_gap": tiny, "missing": np.nan, "rank": np.int64(24)}
    saved = json.loads(json.dumps(confirm.json_safe(values), allow_nan=False))
    assert saved["identity_gap"] == tiny and saved["missing"] is None and saved["rank"] == 24


def contrast_fixture():
    rows = []
    for seed in range(10):
        for scorer in ("lasso", "ols"):
            for duration in (300, 600):
                for method in confirm.METHODS:
                    raw = 0.6 + seed * 0.002
                    effect = (-0.07 + seed * 0.003) if duration == 300 else (0.01 + seed * 0.002)
                    value = raw if method == "raw_window" else raw + effect
                    if method == "matched_random": value = raw - 0.1
                    rows.append({"repeat_seed": seed, "D": duration, "scorer": scorer, "method": method,
                                 "temporal_delta_auroc": value, "pooled_temporal_auroc": 0.7,
                                 "union_auroc": 0.7, "J": np.nan, "C_edge": np.nan, "J_times_C": np.nan,
                                 "hard_gate_accepted": True, "aggregation_executed": method != "raw_window",
                                 "evaluable": True, "status": "success"})
    return pd.DataFrame(rows)


def test_four_primary_tests_and_gamma_use_paired_repeats_not_methods_or_partitions():
    frame = contrast_fixture(); protocol = confirm.default_protocol()
    summary, tests = confirm.summarise(frame, protocol)
    primary = tests[tests.primary].set_index("contrast")
    assert len(primary) == 4 and (primary.n_paired == 10).all()
    gamma = primary.loc["native_hard_duration_interaction"]
    expected = (0.01 + np.arange(10) * 0.002) - (-0.07 + np.arange(10) * 0.003)
    assert gamma["mean"] == pytest.approx(expected.mean())
    assert gamma["p_raw"] == pytest.approx(confirm.test_zero(expected))
    assert tests.loc[~tests.primary, "p_holm"].isna().all()
    assert primary.p_holm.ge(primary.p_raw).all()
    assert summary.n_paired.max() == 10


def test_missing_pair_does_not_create_a_replacement_or_shrink_the_holm_family():
    frame = contrast_fixture(); protocol = confirm.default_protocol()
    frame.loc[(frame.repeat_seed == 0) & (frame.D == 600) & (frame.method == "native_hard"), "temporal_delta_auroc"] = np.nan
    _, tests = confirm.summarise(frame, protocol)
    selected = tests[(tests.scorer == "lasso") & tests.contrast.str.contains("interaction")]
    assert selected[selected.contrast.str.startswith("native")].n_paired.iloc[0] == 9
    assert tests.primary.sum() == 4
    corrected = confirm.holm([0.01, np.nan, 0.04, 0.02])
    np.testing.assert_allclose(corrected[[0, 2, 3]], [0.04, 0.08, 0.06])
    assert np.isnan(corrected[1])


def test_failed_fits_retain_inputs_rank_reason_and_all_planned_rows(tmp_path, monkeypatch):
    protocol = confirm.default_protocol(); protocol["base_config"] = dict(
        protocol["base_config"], n_nodes=2, max_lag=1, timesteps=30, burn_in=5)
    def fail(*args, **kwargs):
        raise common.ScorerFailure("fixture failure", {"rank": 0, "n_features": 2})
    monkeypatch.setattr(confirm, "fit_aggregation", fail)
    rows = confirm.run_repeat(protocol, 0, tmp_path, "fixture-protocol")
    assert len(rows) == 24 and all(r["status"] == "failed" for r in rows)
    assert (tmp_path / "repeat_0_inputs.npz").is_file()
    info = json.loads((tmp_path / "repeat_0_D300_ols.json").read_text())
    assert info["diagnostic"]["rank"] == 0
    assert len(confirm.read_repeat(tmp_path, 0, "fixture-protocol")) == 24
    (tmp_path / "repeat_0_D300_ols.json").write_text("corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        confirm.read_repeat(tmp_path, 0, "fixture-protocol")


def test_toy_bundle_readonly_replay_never_refits_or_changes_retained_files(tmp_path, monkeypatch):
    # A tiny seed-0 unit fixture exercises delivery plumbing, not the new study.
    protocol = confirm.default_protocol()
    protocol["repeat_seeds"] = [0]
    protocol["n_planned_repeats"] = 1
    protocol["base_config"] = dict(protocol["base_config"], n_nodes=3, max_lag=1, timesteps=120, burn_in=5)
    protocol["score_pipeline"] = dict(protocol["score_pipeline"], window_size=30, stride=15)
    monkeypatch.setattr(confirm, "default_protocol", lambda: protocol)
    frozen = tmp_path / "fixture_protocol.json"; frozen.write_bytes(confirm.canonical_bytes(protocol))
    output = tmp_path / "outputs"
    confirm.run_study(frozen, output)
    monkeypatch.setattr(common, "_window_scores", lambda *a, **k: pytest.fail("Cache replay must not fit scorers"))
    sys.path.insert(0, str(ROOT / "tools"))
    from verify_duration_confirmation import verify
    before = {p.relative_to(output).as_posix(): confirm.file_sha(p) for p in output.rglob("*") if p.is_file()}
    result = verify(output)
    after = {p.relative_to(output).as_posix(): confirm.file_sha(p) for p in output.rglob("*") if p.is_file()}
    assert result["scorer_refits"] == 0 and result["replayed_successful_fits"] == 4
    assert before == after
    with pytest.raises(ValueError, match="refuse to overwrite"):
        confirm.run_study(frozen, output)
