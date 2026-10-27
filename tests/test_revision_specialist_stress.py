"""Experiment-integrity checks for the independent specialist extension."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("specialist_stress", ROOT / "scripts/run_revision_specialist_stress.py")
stress = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(stress)


def test_complete_original_axes_and_only_exact_baseline_deduplication():
    grid = stress.make_grid()
    assert len(grid) == 15
    assert len(stress.unique_settings(grid)) == 13
    observed = {axis: [r["value"] for r in grid if r["axis"] == axis]
                for axis in ("obs_noise", "timesteps", "edge_prob", "switch_period")}
    assert observed == {
        "obs_noise": [0, .05, .10, .20], "timesteps": [300, 600, 1000, 1500],
        "edge_prob": [.10, .18, .30, .45], "switch_period": [150, 300, 600],
    }
    aliases = [r for r in grid if r["setting_id"] == stress.base_setting(grid)]
    assert {(r["axis"], r["value"]) for r in aliases} == {
        ("obs_noise", 0.0), ("timesteps", 1000), ("edge_prob", .18),
    }
    shortest = next(r["config"] for r in grid if r["axis"] == "timesteps" and r["value"] == 300)
    assert (shortest["window_size"], shortest["window_stride"]) == (150, 75)


def test_new_seed_population_budgets_and_separate_timing_phase():
    fixed = stress.fixed_spec()
    frozen = {"specification": fixed, "protocol_hash": stress.digest(fixed)}
    jobs = stress.make_jobs(frozen, "stress")
    assert len(jobs) == 1300
    assert set(job["seed"] for job in jobs) == set(range(100, 120))
    assert not set(job["seed"] for job in jobs).intersection(range(20))
    assert len(stress.make_jobs(frozen, "timing")) == 100
    assert len(stress.make_jobs(frozen, "stress", calibrate=True)) == 5
    assert fixed["rpcmci"]["num_iterations"] == 20
    assert fixed["rpcmci"]["max_anneal"] == 10
    assert fixed["cmlp"]["epochs"] == 300
    assert [stress.job_key(j) for j in jobs] == [stress.job_key(j) for j in stress.make_jobs(frozen, "stress")]
    timing = stress.make_jobs(frozen, "timing")[0]
    assert stress.job_key(timing).startswith("timing_")


def test_input_generation_is_independent_of_dispatch_order():
    configs = stress.unique_settings(stress.make_grid())
    config = next(iter(configs.values()))
    first = stress.generate_dataset(config, 91731)
    _other = stress.generate_dataset(config, 91732)
    again = stress.generate_dataset(config, 91731)
    np.testing.assert_array_equal(first.x, again.x)
    assert stress.array_hash(first.x) == stress.array_hash(again.x)
    assert stress.array_hash(first.x) != stress.array_hash(_other.x)


def test_cache_refuses_a_different_method_input_or_protocol(tmp_path):
    config = stress.make_grid()[0]["config"]
    job = {"phase": "stress", "setting_id": "same", "seed": 100, "method": "STCG-v1",
           "protocol_hash": "frozen", "config": config}
    path = tmp_path / "fit.json"
    stress.atomic_json(path, dict(job, status="failed", error="Retained optimizer failure"))
    assert stress.valid_cached(path, job)["status"] == "failed"
    for key, value in (("protocol_hash", "changed"), ("seed", 101), ("method", "Regime-PCMCI")):
        with pytest.raises(ValueError, match="disagrees"):
            stress.valid_cached(path, dict(job, **{key: value}))
    stored = json.loads(path.read_text())
    assert stored["error"] == "Retained optimizer failure"


def test_paired_metrics_match_scipy_and_shd_direction():
    a = [.72, .75, .69, .8, .81]
    b = [.7, .72, .7, .73, .76]
    out = stress.paired_statistics(a, b, "edge_auc")
    assert out["paired_t_two_sided_p"] == pytest.approx(stats.ttest_rel(a, b).pvalue)
    assert out["wilcoxon_two_sided_p"] == pytest.approx(stats.wilcoxon(np.array(a) - np.array(b)).pvalue)
    assert out["mean_improvement"] == pytest.approx(.032)
    reverse = stress.paired_statistics(a, b, "shd")
    assert reverse["mean_improvement"] == pytest.approx(-out["mean_improvement"])
    assert reverse["wins"] == out["losses"]
    identical = stress.paired_statistics(a, a, "edge_auc")
    assert identical["paired_t_two_sided_p"] == identical["wilcoxon_two_sided_p"] == 1
    with pytest.raises(ValueError):
        stress.paired_statistics([.5, np.nan], [.4, .4], "edge_auc")


def test_holm_and_cost_ratio_are_correct_at_known_values():
    assert stress.holm_adjust([.01, .04, .03]) == pytest.approx([.03, .06, .06])
    # Frozen-size correction cannot become len(observed)=1 in an interim table.
    assert stress.holm_adjust([.01] + [1] * 12)[0] == pytest.approx(.13)
    ratio = stress.paired_cost_ratio([1, 2, 4], [8, 16, 32])
    assert ratio["geometric_mean_opponent_over_stcg"] == pytest.approx(8)
    assert ratio["ratio_ci_low"] == pytest.approx(8)
    assert ratio["ratio_ci_high"] == pytest.approx(8)


def test_timing_cannot_run_with_concurrent_workers(tmp_path):
    with pytest.raises(ValueError, match="Dedicated timing"):
        stress.run_controller(tmp_path, "timing", 4, False, True)
    with pytest.raises(ValueError, match="quiet machine"):
        stress.run_controller(tmp_path, "timing", 1, False, False)


def _analysis_module():
    spec = importlib.util.spec_from_file_location("resolution_analysis", ROOT / "tools/summarize_revision_specialist.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_zero_cpu_does_not_hide_available_wall_time_ratio():
    analysis = _analysis_module()
    cpu = analysis.resolution_aware_cost_ratio([.01, .02, .03], [0, .01, 0])
    wall = analysis.resolution_aware_cost_ratio([.01, .02, .03], [.001, .002, .003])
    assert cpu["ratio_available"] is False
    assert cpu["zero_clock_opponent_count"] == 2
    assert cpu["geometric_mean_opponent_over_stcg"] is None
    assert wall["ratio_available"] is True
    assert wall["geometric_mean_opponent_over_stcg"] == pytest.approx(.1)


def test_resolution_amendment_still_rejects_corrupted_times():
    analysis = _analysis_module()
    for invalid in ([0, -1], [1, float("nan")], [1, float("inf")]):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            analysis.resolution_aware_cost_ratio([1, 2], invalid)


def _write_fake_completed_cell(tmp_path):
    frozen = stress.freeze(tmp_path)
    base = stress.base_setting(frozen["specification"]["display_cells"])
    jobs = [job for job in stress.make_jobs(frozen, "stress")
            if job["setting_id"] == base and job["seed"] in (100, 101, 102)]
    for job in jobs:
        method_index = stress.METHODS.index(job["method"])
        row = dict(job, status="ok", input_sha256=f"shared-input-{job['seed']}",
                   fit_seconds=.02 * (method_index + 1),
                   fit_cpu_seconds=0.0 if job["method"] == "VAR-Lasso" else .01 * (method_index + 1))
        for metric in stress.METRICS:
            row[metric] = .5 + .01 * method_index + .001 * (job["seed"] - 100) * (method_index + 1)
        if job["method"] == "Regime-PCMCI" and job["seed"] == 101:
            row["status"] = "failed"
            row["error"] = "Fake solver failure, retained in coverage"
            for metric in stress.METRICS:
                row[metric] = None
        stress.atomic_json(stress.result_path(tmp_path, job), row)
    return jobs


def test_full_analysis_retains_failures_common_seeds_and_zero_clock(tmp_path):
    import csv
    _write_fake_completed_cell(tmp_path)
    result = _analysis_module().summarize(tmp_path)
    assert result["stored_fits"] == 15
    assert result["paired_rows"] == 20  # Four opponents, five metrics, one unique setting.
    with (tmp_path / "revision_specialist_paired.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert {row["common_seeds"] for row in rows} == {"[100, 102]"}
    assert {row["n"] for row in rows} == {"2"}
    with (tmp_path / "revision_specialist_timing.csv").open(newline="", encoding="utf-8") as file:
        timings = list(csv.DictReader(file))
    cpu = next(r for r in timings if r["opponent"] == "VAR-Lasso" and r["metric"] == "fit_cpu_seconds")
    wall = next(r for r in timings if r["opponent"] == "VAR-Lasso" and r["metric"] == "fit_seconds")
    assert cpu["ratio_available"] == "False"
    assert cpu["zero_clock_opponent_count"] == "2"
    assert wall["ratio_available"] == "True"
    assert float(wall["geometric_mean_opponent_over_stcg"]) == pytest.approx(3)


def test_analysis_refuses_mismatched_actual_inputs(tmp_path):
    jobs = _write_fake_completed_cell(tmp_path)
    job = next(job for job in jobs if job["method"] == "cMLP" and job["seed"] == 102)
    path = stress.result_path(tmp_path, job)
    row = json.loads(path.read_text())
    row["input_sha256"] = "wrong-input"
    stress.atomic_json(path, row)
    with pytest.raises(ValueError, match="identical input arrays"):
        _analysis_module().summarize(tmp_path)


def test_undefined_pair_auc_is_not_a_failed_fit_or_lost_edge_observation(tmp_path):
    import csv
    jobs = _write_fake_completed_cell(tmp_path)
    for job in jobs:
        if job["seed"] == 102:
            path = stress.result_path(tmp_path, job)
            row = json.loads(path.read_text())
            row["pair_auc"] = float("nan")
            row["undefined_metrics"] = ["pair_auc"]
            stress.atomic_json(path, row)
            saved = json.loads(path.read_text())
            assert saved["status"] == "ok"
            assert saved["pair_auc"] is None
    _analysis_module().summarize(tmp_path)
    with (tmp_path / "revision_specialist_paired.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    edge = next(r for r in rows if r["metric"] == "edge_auc" and r["opponent"] == "Regime-PCMCI")
    pair = next(r for r in rows if r["metric"] == "pair_auc" and r["opponent"] == "Regime-PCMCI")
    assert edge["n"] == "2"
    assert pair["n"] == "1"
    assert pair["undefined_metric_seeds"] == "[102]"
    assert pair["successful_fit_common_n"] == "2"


def test_storage_whitelist_cannot_authorize_estimator_changes(tmp_path, monkeypatch):
    import copy
    frozen = stress.freeze(tmp_path)
    old = copy.deepcopy(frozen)
    name = "scripts/run_revision_specialist_stress.py"
    old["specification"]["source_hashes"][name] = "old-recording-source"
    stress.atomic_json(stress.protocol_path(tmp_path), old)
    amendment = {"old_runner_sha256": "old-recording-source",
                 "new_runner_sha256": frozen["specification"]["source_hashes"][name],
                 "frozen_function_ast_hashes": stress.function_ast_hashes()}
    stress.atomic_json(tmp_path / "revision_specialist_storage_amendment.json", amendment)
    assert stress.load_protocol(tmp_path)["specification"]["source_hashes"][name] == "old-recording-source"
    original = stress.fixed_spec
    def changed_budget():
        spec = original()
        spec["rpcmci"]["max_anneal"] = 5
        return spec
    monkeypatch.setattr(stress, "fixed_spec", changed_budget)
    with pytest.raises(ValueError, match="budget"):
        stress.load_protocol(tmp_path)


def test_scheduler_refuses_overlap_and_unrecorded_previous_workers(tmp_path, monkeypatch):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "tools/run_revision_specialist_schedule.py"
    spec = importlib.util.spec_from_file_location("specialist_schedule_test", path)
    schedule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schedule)
    stress.atomic_json(tmp_path / "revision_specialist_schedule_drain.json", {
        "old_controller_pid": 123, "running_children_to_drain": [{"pid": 456}],
        "old_status": {"active": [{"pid": 456, "job": "stress_fake-setting_seed100_regimepcmci"}]},
    })
    monkeypatch.setattr(schedule, "process_active", lambda pid: pid == 456)
    with pytest.raises(RuntimeError, match="Wait for previous"):
        schedule.verify_drain(tmp_path)
    monkeypatch.setattr(schedule, "process_active", lambda pid: False)
    with pytest.raises(RuntimeError, match="no retained completion"):
        schedule.verify_drain(tmp_path)
    job = {"phase": "stress", "setting_id": "fake-setting", "seed": 100,
           "method": "Regime-PCMCI", "worker_pid": 456, "status": "failed"}
    stress.atomic_json(tmp_path / "revision_specialist_parts" / (stress.job_key(job) + ".json"), job)
    assert schedule.verify_drain(tmp_path)["preserved_worker_completions"] == [stress.job_key(job)]
    for workers in (0, 17):
        with pytest.raises(ValueError, match="1--16"):
            schedule.run(tmp_path, workers)
