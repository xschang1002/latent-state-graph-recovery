"""Regression checks for the separate PELT adapter and fresh-fit evidence."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_segmentation_validation as study


def test_refit_emission_uses_final_disjoint_ranges_after_merge():
    maps = np.array([1., 3.])[:, None, None, None]
    bounds = np.array([[0, 4], [2, 6], [4, 8], [6, 10]])
    emitted, labels = study.range_emission(maps, [(0, 5), (5, 10)], bounds, 10)
    np.testing.assert_array_equal(labels, [0, 0, 1, 1])
    np.testing.assert_array_equal(emitted[:, 0, 0, 0], [1, 1, 3, 3])
    # An exact-cut centre belongs to the right [start,stop) range.
    _, labels = study.range_emission(maps, [(0, 3), (3, 10)], bounds, 10)
    np.testing.assert_array_equal(labels, [0, 1, 1, 1])


@pytest.mark.parametrize("ranges", [[(0, 4), (5, 10)], [(0, 6), (5, 10)], [(1, 5), (5, 10)], [(0, 5), (5, 9)]])
def test_refit_emission_rejects_gaps_overlaps_or_lost_endpoints(ranges):
    with pytest.raises(ValueError, match="partition"):
        study.range_emission(np.zeros((2, 1, 2, 2)), ranges, [[0, 4], [6, 10]], 10)


def test_adapter_fixed_nonoracle_parameters_and_nonself_features(monkeypatch):
    raw = np.arange(6 * 1 * 2 * 2, dtype=float).reshape(6, 1, 2, 2)
    bounds = np.array([(i * 10, i * 10 + 20) for i in range(6)])
    labels = np.array([0, 0, 1, 1, 2, 2]); calls = {}
    def detector(features, **kwargs):
        calls.update(kwargs)
        np.testing.assert_array_equal(features, raw[:, np.array([[[False, True], [True, False]]])])
        return SimpleNamespace(labels=labels, breakpoints=[2, 4, 6], penalty=3., n_pcs=5)
    def refit(x, supplied_bounds, supplied_labels, **kwargs):
        np.testing.assert_array_equal(supplied_labels, labels)
        calls["refit"] = kwargs
        # Simulates an existing short-range merge: final ranges need not have one map per PELT label.
        return np.stack([np.ones((1, 2, 2)), np.ones((1, 2, 2)) * 2]), [(0, 35), (35, 70)]
    monkeypatch.setattr(study, "pelt_on_features", detector); monkeypatch.setattr(study, "segmented_lasso_scores", refit)
    saved, diagnostic = study.fit_pelt(np.zeros((70, 2)), raw, bounds)
    assert calls == {"penalty_factor": 1., "n_pcs": 5, "min_size": 2, "n_bkps": None,
                     "refit": {"max_lag": 3, "alpha": .005}}
    assert diagnostic["n_detected_segments"] == 3 and diagnostic["n_refit_segments"] == 2
    assert saved["emission_pelt_var_adapter"].shape == raw.shape


def test_actual_pelt_constant_features_retains_one_segment():
    raw = np.zeros((8, 3, 3, 3)); x = np.random.default_rng(41).normal(size=(180, 3))
    bounds = np.array([(i * 20, i * 20 + 40) for i in range(8)])
    fitted, diag = study.fit_pelt(x, raw, bounds)
    assert diag["n_detected_segments"] == diag["n_refit_segments"] == 1
    np.testing.assert_array_equal(fitted["segment_ranges"], [[0, 180]])
    assert np.all(fitted["emission_pelt_var_adapter"] == fitted["emission_pelt_var_adapter"][0])


def test_array_comparisons_separate_exact_tolerance_and_failure():
    old = {"exact": np.array([1.]), "tol": np.array([1.]), "different": np.array([1.]),
           "dtype": np.array([1], dtype=np.int64), "nan": np.array([np.nan])}
    new = {"exact": np.array([1.]), "tol": np.array([1. + 1e-13]), "different": np.array([1.1]),
           "dtype": np.array([1], dtype=np.int32), "nan": np.array([np.nan])}
    got = {r["array"]: r for r in study.comparison_rows(new, old, {}, "fit")}
    assert got["exact"]["classification"] == got["nan"]["classification"] == "exact"
    assert got["tol"]["classification"] == "tolerance" and not got["tol"]["exact_equal"]
    assert got["different"]["classification"] == got["dtype"]["classification"] == "different"


def test_summary_is_paired_by_seed_and_retains_failed_denominator():
    rows = []
    for seed, value in [(1, .9), (2, .1), (3, .5)]:
        for method, delta in [("raw_window", 0), ("forced_hard", -.1), ("pelt_var_adapter", .2)]:
            row = {"duration": 300, "repeat_seed": seed, "method": method,
                   **{m: value + delta for m in study.METRICS}}
            if seed == 2 and method == "pelt_var_adapter":
                row.update({m: np.nan for m in study.METRICS})
            rows.append(row)
    detail = pd.DataFrame(rows).sample(frac=1, random_state=5)
    absolute, contrast = study.summaries(detail)
    row = contrast[(contrast.method == "pelt_var_adapter") & (contrast.reference == "raw_window")].iloc[0]
    assert row.n_planned == 3 and row.n_paired == 2 and row.n_missing_pair == 1
    assert row["mean"] == pytest.approx(.2)
    row = absolute[absolute.method == "pelt_var_adapter"].iloc[0]
    assert row.n_finite == 2 and row.n_missing == 1
    pd.testing.assert_frame_equal(contrast, study.summaries(detail)[1])


def test_freeze_only_writes_new_output_execution_and_rejects_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "protocol", lambda root: {"configuration": study.config()})
    monkeypatch.setattr(study, "fit_pelt", lambda *a: pytest.fail("fit during freeze"))
    monkeypatch.setattr(study.common, "fit_aggregation", lambda *a, **k: pytest.fail("fresh fit during freeze"))
    output = tmp_path / "new"; study.freeze(tmp_path, output)
    assert set(study.tree_hashes(output)) == {"protocol.json", "execution/freeze_record.json"}
    assert not (tmp_path / "experiments").exists()
    with pytest.raises(ValueError, match="empty"):
        study.freeze(tmp_path, output)


def test_adapter_failure_retains_shared_references_and_diagnostic(monkeypatch, tmp_path):
    output = tmp_path / "out"; (output / "parts").mkdir(parents=True)
    source = tmp_path / "old.npz"; source.write_bytes(b"reference")
    raw = np.zeros((3, 3, 2, 2))
    values = {"window_scores": raw, "bounds": np.array([[0, 20], [10, 30], [20, 40]]),
              "regimes": np.zeros(40, int), "state_support": np.zeros((2, 3, 2, 2), bool),
              "emission_raw_window": raw, "emission_forced_hard": raw, "observed_series": np.zeros((40, 2))}
    monkeypatch.setattr(study, "retained_case", lambda *a: (values, "old.npz"))
    def fail(*args):
        raise RuntimeError("planned toy failure")
    monkeypatch.setattr(study, "fit_pelt", fail)
    rec = study.run_adapter_case(tmp_path, output, 70000, 300)
    assert rec["status"] == "failed" and "planned toy failure" in rec["error"]
    saved = study.arrays(output / "parts/repeat_70000_D300.npz")
    assert "emission_pelt_var_adapter" not in saved
    np.testing.assert_array_equal(saved["window_scores"], raw)


def test_adapter_convergence_warning_fails_without_dropping_warned_arrays(monkeypatch, tmp_path):
    output = tmp_path / "out"; (output / "parts").mkdir(parents=True)
    source = tmp_path / "old.npz"; source.write_bytes(b"reference")
    raw = np.zeros((3, 3, 2, 2))
    values = {"window_scores": raw, "bounds": np.array([[0, 20], [10, 30], [20, 40]]),
              "regimes": np.zeros(40, int), "state_support": np.zeros((2, 3, 2, 2), bool),
              "emission_raw_window": raw, "emission_forced_hard": raw, "observed_series": np.zeros((40, 2))}
    monkeypatch.setattr(study, "retained_case", lambda *a: (values, "old.npz"))
    def warned(*args):
        warnings.warn("toy refit convergence", study.ConvergenceWarning)
        return {"emission_pelt_var_adapter": raw, "segment_maps": raw[:1]}, {"n_refit_segments": 1}
    monkeypatch.setattr(study, "fit_pelt", warned)
    rec = study.run_adapter_case(tmp_path, output, 70000, 300)
    assert rec["status"] == "failed" and "ConvergenceWarning" in rec["error"]
    assert rec["warnings"] == ["toy refit convergence"]
    saved = study.arrays(output / "parts/repeat_70000_D300.npz")
    assert "emission_pelt_var_adapter" not in saved
    np.testing.assert_array_equal(saved["failed_emission_pelt_var_adapter"], raw)


def test_protocol_no_oracle_or_optimised_penalty_and_fixed_bounded_fresh_seeds():
    config = study.config()
    assert config["pelt"] == {"penalty_factor": 1., "n_pcs": 5, "min_size": 2, "jump": 1, "model": "l2", "n_bkps": None}
    assert config["fresh"]["A_seeds"] == [50000, 50001]
    assert config["fresh"]["C_seeds"] == [70000, 70001]
    assert config["fresh"]["fresh_cells"] == 8
    assert config["statistics"]["tests"].startswith("none")


def test_shared_boundary_api_reports_standard_ap_and_method_scale_control():
    bounds = np.array([(i * 10, i * 10 + 10) for i in range(6)])
    regimes = np.zeros(60, int); regimes[15:35] = 1
    support = np.zeros((2, 1, 2, 2), bool); support[1, 0, 0, 1] = True
    raw = np.zeros((6, 1, 2, 2)); raw[:, 0, 0, 1] = [0, 1, 0, .5, .25, .75]
    raw[:, 0, 1, 0] = [.125, .25, .125, .25, .5, 0]
    constant = np.repeat(raw.mean(axis=0)[None], 6, axis=0)
    rows = {r["method"]: r for r in study.evaluate({"raw_window": raw, "scaled": .25 * raw, "constant": constant},
                                                  raw, bounds, regimes, support)}
    assert set(study.METRICS) <= set(rows["raw_window"])
    assert rows["constant"]["boundary_positive_change_ap"] == 0
    assert rows["constant"]["boundary_standard_ap"] == pytest.approx(.4)
    assert rows["constant"]["boundary_slot_f1_method_q95"] == 0
    assert rows["raw_window"]["boundary_slot_f1_method_q95"] == rows["scaled"]["boundary_slot_f1_method_q95"]
    assert rows["raw_window"]["stable_false_change_rate_method_q95"] == rows["scaled"]["stable_false_change_rate_method_q95"]


def test_rebuild_verifies_integrity_before_any_write(monkeypatch, tmp_path):
    def fail(*a):
        raise ValueError("tampered input")
    monkeypatch.setattr(study, "verify", fail)
    monkeypatch.setattr(study, "write_derived", lambda *a: pytest.fail("write before integrity"))
    with pytest.raises(ValueError, match="tampered"):
        study.rebuild(tmp_path, tmp_path / "unused")


def test_fresh_reproduction_generates_inputs_then_series_before_scorer(monkeypatch, tmp_path):
    old = tmp_path / study.BASE / "duration_confirmation"
    old.mkdir(parents=True); output = tmp_path / "new"; (output / "fresh").mkdir(parents=True)
    cfg = dict(n_nodes=2, timesteps=40, max_lag=3, edge_prob=.18, switch_period=300,
               overlap=.5, effect_scale=1., noise_scale=.1, observation_noise=0, burn_in=10)
    (old / "protocol.json").write_text(json.dumps({"base_config": cfg}))
    calls = []
    inputs = {"coefficients": np.zeros((2, 3, 2, 2)), "support": np.zeros((2, 3, 2, 2), int),
              "initial_values": np.ones((3, 2)), "innovations": np.full((50, 2), 7.)}
    def draw(c, seed):
        calls.append("new_rng_inputs"); assert seed == 50000
        return inputs
    def simulate(supplied, schedule, c):
        calls.append("new_simulation"); assert supplied is inputs
        return SimpleNamespace(x=np.full((40, 2), 13.), regimes=np.zeros(40, int))
    def fit(dataset, scorer, seed, **kwargs):
        calls.append("new_fit")
        assert (output / "fresh/A_50000_D300_inputs.npz").is_file()
        assert (output / "fresh/A_50000_D300_series.npz").is_file()
        np.testing.assert_array_equal(dataset.x, np.full((40, 2), 13.))
        assert kwargs["exact_random"] is True and scorer == "lasso" and seed == 50000
        return [], {"window_scores": np.zeros((2, 3, 2, 2))}, {}
    monkeypatch.setattr(study.common, "draw_base_inputs", draw)
    monkeypatch.setattr(study.common, "simulate_schedule", simulate)
    monkeypatch.setattr(study.common, "fit_aggregation", fit)
    rec = study.run_fresh_case(tmp_path, output, "A", 50000, 300)
    assert rec["status"] == "ok" and rec["new_generation"] and rec["new_fitting"]
    assert calls == ["new_rng_inputs", "new_simulation", "new_fit"]


def test_verify_replays_saved_arrays_without_any_fit_or_simulation(monkeypatch, tmp_path):
    output = tmp_path / "new"; (output / "parts").mkdir(parents=True); (output / "fresh").mkdir()
    study.write_json(output / "protocol.json", {})
    monkeypatch.setattr(study, "check_protocol", lambda *a: {})
    values = {"window_scores": np.zeros((3, 3, 2, 2)), "bounds": np.array([[0, 20], [10, 30], [20, 40]]),
              "regimes": np.zeros(40, int), "state_support": np.zeros((2, 3, 2, 2), bool),
              "emission_raw_window": np.zeros((3, 3, 2, 2)), "emission_forced_hard": np.zeros((3, 3, 2, 2)),
              "observed_series": np.zeros((40, 2))}
    reference = tmp_path / "reference.npz"; np.savez_compressed(reference, **values)
    monkeypatch.setattr(study, "retained_case", lambda *a: (values, "reference.npz"))
    for seed in range(70000, 70030):
        for duration in (300, 600):
            path = output / "parts" / f"repeat_{seed}_D{duration}"
            np.savez_compressed(path.with_suffix(".npz"), **{k: v for k, v in values.items() if k != "observed_series"})
            study.write_json(path.with_suffix(".json"), {"repeat_seed": seed, "duration": duration, "status": "failed",
                "source": "reference.npz", "source_sha256": study.sha(reference), "arrays_sha256": study.sha(path.with_suffix(".npz"))})
    for st, seeds in (("A", [50000, 50001]), ("C", [70000, 70001])):
        for seed in seeds:
            for duration in (300, 600):
                study.write_json(output / "fresh" / f"{st}_{seed}_D{duration}.json", {"study": st, "repeat_seed": seed, "duration": duration})
    expected = {"detail.csv": pd.DataFrame({"value": [.25]}), "diagnostics.json": {"retained_adapter_cases": 60}}
    expected["detail.csv"].to_csv(output / "detail.csv", index=False)
    study.write_json(output / "diagnostics.json", expected["diagnostics.json"])
    monkeypatch.setattr(study, "derived", lambda *a: expected)
    def forbidden(*a, **k):
        pytest.fail("read-only verifier called fitting/simulation")
    monkeypatch.setattr(study, "fit_pelt", forbidden)
    monkeypatch.setattr(study.common, "fit_aggregation", forbidden)
    monkeypatch.setattr(study.common, "simulate_schedule", forbidden)
    monkeypatch.setattr(study.symmetric, "simulate_full", forbidden)
    study.seal(output); before = study.tree_hashes(output)
    result = study.verify(tmp_path, output)
    assert result["status"] == "passed" and result["new_fits"] == result["new_simulations"] == 0
    assert study.tree_hashes(output) == before
