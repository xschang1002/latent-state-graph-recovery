"""Meaningful regression checks for the separate Study C generator and replay."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_symmetric_sensitivity as study


def small_config():
    config = deepcopy(study.load_config())
    config.update(seed_start=9, seed_count=1, durations=[30, 60])
    config["mechanism"].update(n_nodes=3, timesteps=120, max_lag=2, burn_in=10)
    config["windows"] = [{"size": 40, "stride": 20}, {"size": 40, "stride": 40}]
    config["statistics"].update(bootstrap_draws=100, minimum_ci_n=1)
    return config


def test_exact_symmetric_support_rule_and_empty_graph_policy():
    rng = np.random.default_rng(274)
    mask = ~np.eye(8, dtype=bool)
    pairs = np.concatenate([study.sample_supports(rng, 8, 3, .18, .5)[:, :, mask].reshape(2, -1)
                            for _ in range(1500)], axis=1)
    np.testing.assert_allclose(pairs.mean(axis=1), [.18, .18], atol=.004, rtol=0)
    assert abs(np.mean(pairs[0] & pairs[1]) - .09) < .003
    assert abs(np.mean(pairs[0] & ~pairs[1]) - np.mean(~pairs[0] & pairs[1])) < .003
    assert .18 * .5 + (1 - .18) * (.18 * .5 / (1 - .18)) == pytest.approx(.18)

    class NoEdges:
        def random(self, shape):
            return np.ones(shape)

    empty = study.sample_supports(NoEdges(), 3, 2, .18, .5)
    assert not empty.any()  # No asymmetric fallback or conditional redraw.
    for state in study.sample_supports(rng, 8, 3, .18, .5):
        assert not state[:, np.arange(8), np.arange(8)].any()


def test_lag_aware_common_scaling_caps_radii_and_preserves_shared_values():
    raw = np.zeros((2, 3, 2, 2))
    raw[0, :, 0, 1] = [1.8, .8, .2]
    raw[0, :, 1, 0] = [1.8, .8, .2]
    raw[1] = raw[0]
    raw[1, 2, 1, 0] = .4
    scaled = study.lag_aware_scale_pair(raw)
    assert scaled["rho_initial"].min() > .9
    assert scaled["rho_final"].max() <= .9
    np.testing.assert_allclose(scaled["rho_final"], scaled["common_scale"] * scaled["rho_initial"], atol=1e-12)
    np.testing.assert_array_equal(scaled["coefficients"], raw * scaled["lag_scale"][None, :, None, None])
    shared_equal = raw[0] == raw[1]
    np.testing.assert_array_equal(scaled["coefficients"][0][shared_equal], scaled["coefficients"][1][shared_equal])
    assert study.lag_aware_scale_pair(np.zeros_like(raw))["common_scale"] == 1


def test_input_streams_and_true_shared_coefficients_are_reproducible():
    config = study.load_config()
    first = study.draw_inputs(config, 61); replay = study.draw_inputs(config, 61)
    assert study.fingerprints(first) == study.fingerprints(replay)
    assert study.fingerprints(first) != study.fingerprints(study.draw_inputs(config, 62))
    support = first["support"]
    shared = support[0] & support[1]
    assert shared.any()
    for key in ("coefficients_raw", "coefficients"):
        np.testing.assert_array_equal(first[key][0][shared], first[key][1][shared])
        assert not first[key][:, :, np.arange(8), np.arange(8)].any()
    p = study.protocol(config)
    assert p["expected_fits"] == 300 and len(p["repeats"]) == 30
    assert [len(study.bounds(config, w)) for w in config["windows"]] == [35, 23, 17, 45, 12]


def test_freeze_cache_resume_and_verification_recalculate_without_refitting(tmp_path, monkeypatch):
    config = small_config(); output = tmp_path / "study"
    proposed = study.protocol(config)
    with pytest.raises(RuntimeError, match="frozen"):
        study.ensure_protocol(output, proposed)
    freeze = study.freeze(output, proposed, tmp_path / "clock_logs")
    assert freeze["before_freeze_utc"] <= freeze["after_freeze_utc"] and freeze["new_fits"] == 0
    study.ensure_protocol(output, proposed)
    with pytest.raises(RuntimeError, match="empty"):
        study.freeze(output, proposed, tmp_path / "clock_logs")
    with study.threadpool_limits(limits=1):
        records = study.run_repeat(config, 9, output)
    assert len(records) == 4 and all(r["diagnostic"]["status"] == "ok" for r in records)
    study.aggregate(output, config)
    before = study.directory_hashes(output)

    def no_fitting(*args, **kwargs):
        raise AssertionError("Cache replay must not refit")

    monkeypatch.setattr(study.common, "fit_aggregation", no_fitting)
    monkeypatch.setattr(study, "simulate_full", no_fitting)
    study.run_repeat(config, 9, output)
    report = study.verify(output)
    assert report["planned_fits"] == 4 and report["metric_rows_recomputed"] == 20
    assert report["new_fits"] == 0 and report["new_simulations"] == 0
    assert study.directory_hashes(output) == before
    assert np.isfinite(study.load_arrays(next((output / "series").glob("*.npz")))["full_series"]).all()
    altered = deepcopy(proposed); altered["configuration"]["model"]["lasso_alpha"] = .1
    with pytest.raises(RuntimeError, match="identical"):
        study.ensure_protocol(output, altered)
    part = next((output / "parts").glob("*.npz"))
    with part.open("ab") as stream:
        stream.write(b"changed retained bytes")
    with pytest.raises(AssertionError):
        study.verify(output)


def test_fitting_failure_is_retained_for_every_planned_cell(tmp_path):
    config = small_config(); output = tmp_path / "failures"
    study.freeze(output, study.protocol(config), tmp_path / "clock_logs")

    def fail(*args, **kwargs):
        raise study.common.ScorerFailure("fixture numerical failure", {"reason": "deliberate test"})

    records = study.run_repeat(config, 9, output, fitter=fail)
    assert len(records) == 4
    assert all(r["diagnostic"]["status"] == "failed" for r in records)
    assert all(row["temporal_delta_auroc"] is None for r in records for row in r["methods"])
    study.aggregate(output, config)
    report = study.verify(output)
    assert report["failed_fits"] == 4 and report["metric_rows_recomputed"] == 0
