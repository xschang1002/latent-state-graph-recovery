"""Regression fixtures for post-review boundary-slot, abstention and false changes."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_temporal_localisation as loc


def fixture():
    support = np.zeros((2, 1, 3, 3), dtype=int)
    support[:, 0, 0, 1] = 1  # persistent
    support[0, 0, 0, 2] = 1
    support[1, 0, 1, 0] = 1
    regimes = np.r_[np.zeros(6, int), np.ones(6, int)]
    bounds = np.array([[start, start + 2] for start in range(0, 12, 2)])
    centres = (bounds[:, 0] + bounds[:, 1] - 1) // 2
    perfect = support[regimes[centres]].astype(float)
    raw = np.repeat(np.arange(6)[:, None, None, None] * .01, 9, axis=1).reshape(6, 1, 3, 3)
    return raw, bounds, regimes, support, perfect


def test_positive_change_ap_uses_whole_tie_groups_and_zero_abstention():
    assert loc.positive_change_ap([1, 0, 1, 0], [2, 2, 1, 0]) == pytest.approx(7 / 12)
    assert loc.positive_change_ap([1, 0, 1, 0], [0, 1, 0, 1]) == 0
    assert loc.positive_change_ap([1, 0], [0, 0]) == 0
    assert np.isnan(loc.positive_change_ap([1, 1], [1, 2]))
    assert np.isnan(loc.positive_change_ap([0, 0], [1, 2]))


def test_perfect_slots_and_constant_misses_have_explicit_penalties():
    raw, bounds, regimes, support, perfect = fixture()
    rows, arrays, grid = loc.evaluate_localisation({"perfect": perfect, "constant": np.zeros_like(raw)}, raw, bounds, regimes, support)
    good, constant = rows
    assert good["boundary_positive_change_ap"] == good["boundary_slot_f1"] == 1
    assert constant["boundary_positive_change_ap"] == constant["boundary_slot_f1"] == 0
    assert constant["boundary_fn"] == 2
    assert constant["stable_false_change_rate"] == 0
    assert good["boundary_ap_edges_used"] == 2
    assert arrays["truth_event_counts"][2].sum() == 2
    assert grid["covered_start"] == 0 and grid["covered_stop_inclusive"] == 10


def test_raw_threshold_is_shared_unlabelled_and_detection_is_strict():
    raw, bounds, regimes, support, perfect = fixture()
    # Integer differences make the equality-at-threshold case exact.
    raw = np.broadcast_to(np.arange(6)[:, None, None, None], raw.shape).astype(float).copy()
    rows, _, _ = loc.evaluate_localisation({"raw": raw, "large": raw * 100}, raw, bounds, regimes, support)
    assert rows[0]["raw_change_threshold"] == rows[1]["raw_change_threshold"] == 1
    assert rows[0]["boundary_tp"] == rows[0]["stable_false_events"] == 0
    assert rows[1]["stable_false_change_rate"] == 1
    altered_support = support.copy(); altered_support[:, 0, 0, 1] = 0
    again, _, _ = loc.evaluate_localisation({"raw": raw}, raw, bounds, regimes[::-1], altered_support)
    assert again[0]["raw_change_threshold"] == 1


def test_stable_changes_include_persistent_and_absent_edges_without_becoming_support_labels():
    raw, bounds, regimes, support, perfect = fixture()
    raw[:] = 0
    prediction = perfect.copy()
    prediction[:, 0, 0, 1] = [0, 1, 0, 0, 0, 0]
    prediction[:, 0, 2, 0] = [0, 0, 0, 1, 0, 0]
    rows, _, _ = loc.evaluate_localisation({"detector": prediction}, raw, bounds, regimes, support)
    row = rows[0]
    assert row["threshold_zero"]
    assert row["persistent_edges"] == 1 and row["absent_edges"] == 3
    assert row["persistent_false_events"] == row["absent_false_events"] == 2
    assert row["stable_false_change_rate"] == pytest.approx(4 / 20)
    assert row["stable_any_change_edge_fraction"] == .5


def test_grid_reads_actual_events_even_when_two_changes_have_equal_endpoint_states():
    raw, bounds, regimes, support, perfect = fixture()
    regimes[:] = 0; regimes[2] = 1
    grid = loc.boundary_grid([[0, 1], [4, 5], [8, 9]], regimes, support)
    assert grid["truth_event_counts"][0].max() == 2
    assert grid["support_events_collapsed"] == 2
    assert grid["actual_state_changes_covered"] == 2


def test_no_negative_boundary_slots_are_missing_for_ap_but_f1_is_counted():
    raw, bounds, regimes, support, perfect = fixture()
    regimes = np.repeat([0, 1, 0, 1, 0, 1], 2)
    perfect = support[regimes[(bounds[:, 0] + bounds[:, 1] - 1) // 2]].astype(float)
    rows, _, _ = loc.evaluate_localisation({"oracle": perfect}, raw, bounds, regimes, support)
    assert np.isnan(rows[0]["boundary_positive_change_ap"])
    assert rows[0]["changing_edges_all_slots_boundary"] == 2
    assert rows[0]["boundary_ap_edges_used"] == 0
    assert rows[0]["boundary_slot_f1"] == 1


def test_no_covered_transition_is_missing_and_outside_centre_domain_is_not_scored():
    raw, bounds, regimes, support, perfect = fixture()
    regimes[:] = 0; regimes[-1] = 1  # actual switch after final centre
    rows, _, grid = loc.evaluate_localisation({"raw": raw}, raw, bounds, regimes, support)
    assert grid["actual_state_changes_total"] == 1 and grid["actual_state_changes_covered"] == 0
    assert rows[0]["changing_edges_no_covered_boundary"] == 2
    assert np.isnan(rows[0]["boundary_positive_change_ap"])
    assert np.isnan(rows[0]["boundary_slot_f1"])


def test_positive_affine_rescaling_preserves_change_ap_without_using_a_true_budget():
    raw, bounds, regimes, support, perfect = fixture()
    offset = np.arange(9).reshape(1, 1, 3, 3)
    transformed = .5 * perfect + offset
    rows, _, _ = loc.evaluate_localisation({"original": perfect, "affine": transformed}, raw, bounds, regimes, support)
    assert rows[0]["boundary_positive_change_ap"] == rows[1]["boundary_positive_change_ap"] == 1


def test_nonfinite_scores_preserve_method_missingness_and_coverage():
    raw, bounds, regimes, support, perfect = fixture()
    bad = perfect.copy(); bad[0, 0, 0, 2] = np.nan
    rows, _, _ = loc.evaluate_localisation({"bad": bad, "perfect": perfect}, raw, bounds, regimes, support)
    assert rows[0]["status"] == "nonfinite_scores"
    assert rows[0]["boundary_ap_edges_eligible"] == 2 and rows[0]["boundary_ap_edges_used"] == 0
    assert np.isnan(rows[0]["stable_false_change_rate"])
    assert rows[1]["boundary_positive_change_ap"] == 1


def test_summary_resamples_independent_paired_repeats_and_keeps_cells_separate():
    rows = []
    for cell in ["h200_s100", "h300_s150"]:
        for seed in range(3):
            for method, value in [("raw_window", .2), ("forced_hard", .45)]:
                rows.append({"study": "fixture", "scorer": "lasso", "duration": 300, "schedule": "fixed", "cell": cell,
                             "repeat_seed": seed, "method": method, **{metric: value for metric in loc.METRICS}})
    summary, paired = loc.summarise(pd.DataFrame(rows))
    assert len(summary) == len(paired) == 2 * 2 * len(loc.METRICS)
    assert (summary.n_planned == 3).all()
    differences = paired[paired.method == "forced_hard"]
    np.testing.assert_allclose(differences[["mean", "ci95_low", "ci95_high"]], .25)
    with pytest.raises(ValueError, match="Duplicate"):
        loc.summarise(pd.DataFrame(rows + [rows[0]]))


def test_freeze_run_and_readonly_verification_reject_resealed_numeric_corruption(tmp_path, monkeypatch):
    raw, bounds, regimes, support, perfect = fixture()
    method_rows, edge_arrays, grid = loc.evaluate_localisation({"raw_window": raw, "forced_hard": perfect}, raw, bounds, regimes, support)
    locked = {"frozen_input_sha256": {}, "frozen_source_sha256": {}, "analysis_source_sha256": {"fixture": "fixed"}}
    monkeypatch.setattr(loc, "protocol", lambda root: locked)
    monkeypatch.setattr(loc, "frozen_hashes", lambda root: {})
    monkeypatch.setattr(loc, "frozen_sources", lambda root: {})
    def cases(root):
        yield {"study": "fixture", "scorer": "lasso", "repeat_seed": 7, "duration": 300, "schedule": "fixed", "cell": "one", "status": "ok", "error": ""}, method_rows, edge_arrays, grid
    monkeypatch.setattr(loc, "retained_cases", cases)
    output, stage = tmp_path / "output", tmp_path / "stage"
    loc.freeze(tmp_path, output, stage)
    loc.run(tmp_path, output, stage)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}
    assert loc.verify(tmp_path, output)["method_rows_recomputed"] == 2
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="overwrite"):
        loc.run(tmp_path, output, stage)
    data = pd.read_csv(output / "detail.csv")
    data.loc[0, "boundary_positive_change_ap"] = .9
    data.to_csv(output / "detail.csv", index=False)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["sha256"]["detail.csv"] = loc.sha(output / "detail.csv")
    (output / "manifest.json").write_bytes(loc.json_bytes(manifest))
    with pytest.raises(AssertionError):
        loc.verify(tmp_path, output)
