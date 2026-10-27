"""Fail-closed sensitivity-gate checks using small independent fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "scripts")]
import verify_evaluation_sensitivity as gate
import analyze_boundary_sensitivity as analysis


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def small_details(monkeypatch):
    case = ("A", "lasso", 300, "fixed_phase", "D300_fixed_phase", 50000)
    monkeypatch.setattr(gate, "expected_cases", lambda: {case})
    meta = dict(zip(gate.CASE, case))
    ap, threshold, residual = [], [], []
    for method in sorted(gate.METHODS):
        ap.append({**meta, "method": method, "status": "ok", "boundary_ap_edges_used": 1,
            "changing_edges_all_slots_boundary": 0, "boundary_positive_change_ap": .4,
            "boundary_standard_ap": .5, "boundary_zero_group_increment": .1})
        for q in gate.QUANTILES:
            for mode in ("raw_shared", "method_self"):
                threshold.append({**meta, "method": method, "q": q, "threshold_mode": mode, "status": "ok",
                    **{m: .1 for m in gate.THRESHOLD_METRICS}, "boundary_f1_edges_used": 1,
                    "stable_false_events": 2, "stable_event_opportunities": 20,
                    "persistent_false_events": 1, "persistent_event_opportunities": 10,
                    "absent_false_events": 1, "absent_event_opportunities": 10,
                    "threshold": .1, "raw_threshold": .1, "method_threshold": .1, "threshold_zero": False})
    for weight in gate.LAMBDAS:
        residual.append({**meta, "method": f"residual_{weight:g}", "lambda": weight, "status": "ok",
            "n_evaluable_edges": 1, "hard_state_auroc": .6, "residual_state_auroc": .6 if weight == 0 else .65,
            "within_auc_delta": 0 if weight == 0 else .1, "across_auc_delta": 0 if weight == 0 else -.05,
            "total_auc_delta": 0 if weight == 0 else .05,
            "positive_negative_pairs": 10, "within_positive_negative_pairs": 5, "across_positive_negative_pairs": 5})
    return pd.DataFrame(ap), pd.DataFrame(threshold), pd.DataFrame(residual)


def test_missing_required_block_fails_closed_and_writes_nothing(tmp_path):
    original = sys.dont_write_bytecode
    with pytest.raises(gate.VerificationError, match="Required file missing"):
        gate.verify(tmp_path, replay=False)
    assert list(tmp_path.iterdir()) == []
    assert sys.dont_write_bytecode == original


@pytest.mark.parametrize("name", ["../outside.csv", "/outside.csv", "x/../outside.csv",
    chr(67) + ":/outside.csv", chr(67) + ":" + chr(92) + "outside.csv", "nested//file.csv"])
def test_cross_platform_unsafe_dependency_names_are_rejected(tmp_path, name):
    with pytest.raises(gate.VerificationError, match="Unsafe"):
        gate.safe_path(tmp_path, name)


def test_portable_windows_relative_paths_have_one_canonical_dependency(tmp_path):
    folder = tmp_path / "nested"; folder.mkdir(); path = folder / "file.csv"; path.write_text("value\n1\n")
    assert gate.inside(tmp_path, "nested" + chr(92) + "file.csv") == path
    with pytest.raises(gate.VerificationError, match="Duplicate normalized"):
        gate.check_hashes(tmp_path, {"nested/file.csv": gate.sha(path), "nested" + chr(92) + "file.csv": gate.sha(path)}, "fixture")


def test_extra_output_and_hash_damage_are_rejected(tmp_path):
    output = tmp_path / gate.BASE / "fixture"; output.mkdir(parents=True)
    write_json(output / "protocol.json", {})
    data = output / "data.csv"; data.write_text("score\n.5\n")
    seal = {"protocol_sha256": gate.sha(output / "protocol.json"),
        "sha256": {"data.csv": gate.sha(data), "protocol.json": gate.sha(output / "protocol.json")}}
    write_json(output / "manifest.json", seal)
    gate.check_manifest(tmp_path, "fixture")
    (output / "extra.csv").write_text("unlisted")
    with pytest.raises(gate.VerificationError, match="inventory drift"):
        gate.check_manifest(tmp_path, "fixture")
    (output / "extra.csv").unlink(); data.write_text("score\n.7\n")
    with pytest.raises(gate.VerificationError, match="Checksum"):
        gate.check_manifest(tmp_path, "fixture")


@pytest.mark.parametrize("damage", ["missing_method", "duplicate", "quantile", "mode", "lambda"])
def test_resealed_coverage_cannot_drop_methods_or_settings(monkeypatch, damage):
    ap, threshold, residual = small_details(monkeypatch)
    assert gate.check_boundary_coverage(ap, threshold, residual)["cases"] == 1
    if damage == "missing_method": ap = ap.iloc[:-1]
    if damage == "duplicate": ap = pd.concat([ap, ap.iloc[:1]], ignore_index=True)
    if damage == "quantile": threshold = threshold[threshold.q != .975]
    if damage == "mode": threshold = threshold[threshold.threshold_mode != "method_self"]
    if damage == "lambda": residual = residual[residual["lambda"] != 1e-8]
    with pytest.raises(gate.VerificationError, match="coverage"):
        gate.check_boundary_coverage(ap, threshold, residual)


def test_ap_missing_target_cannot_be_filled_with_zero(monkeypatch):
    ap, threshold, residual = small_details(monkeypatch)
    ap.loc[0, "boundary_standard_ap"] = np.nan
    with pytest.raises(gate.VerificationError, match="missing/all-positive"):
        gate.check_boundary_coverage(ap, threshold, residual)


@pytest.mark.parametrize("damage", ["AP_identity", "false_denominator", "pair_partition", "infinite"])
def test_numerical_accounting_cannot_be_resigned_away(monkeypatch, damage):
    ap, threshold, residual = small_details(monkeypatch)
    if damage == "AP_identity": ap.loc[0, "boundary_standard_ap"] = .7
    if damage == "false_denominator": threshold.loc[0, "stable_event_opportunities"] = 0
    if damage == "pair_partition": residual.loc[1, "across_auc_delta"] = .2
    if damage == "infinite": threshold.loc[0, "boundary_slot_f1"] = np.inf
    with pytest.raises(gate.VerificationError):
        gate.check_boundary_coverage(ap, threshold, residual)


@pytest.mark.parametrize("damage", ["q", "lambda", "mode", "metrics"])
def test_fixed_sensitivity_configuration_cannot_be_shortened(damage):
    protocol = analysis.protocol("0" * 64)
    seal = {"new_fits": 0, "new_simulations": 0}
    gate.check_boundary_settings(protocol, seal)
    if damage == "q": protocol["quantiles"].pop()
    if damage == "lambda": protocol["residual_lambdas"].pop(1)
    if damage == "mode": protocol["thresholds"]["modes"].pop()
    if damage == "metrics": protocol["statistics"]["bootstrap_metrics"]["AP"].pop()
    with pytest.raises(gate.VerificationError, match="configuration"):
        gate.check_boundary_settings(protocol, seal)


def test_summary_requires_every_setting_and_exact_denominators(monkeypatch):
    ap, _, _ = small_details(monkeypatch)
    summary, paired = analysis.summarise_block(ap, gate.AP_METRICS)
    gate.check_summary(ap, summary, gate.GROUP, gate.AP_METRICS)
    gate.check_summary(ap, paired, gate.GROUP, gate.AP_METRICS, paired=True)
    with pytest.raises(gate.VerificationError, match="coverage"):
        gate.check_summary(ap, summary.iloc[:-1], gate.GROUP, gate.AP_METRICS)
    damaged = summary.copy(); damaged.loc[0, "n_finite"] = 0
    with pytest.raises(gate.VerificationError, match="denominator"):
        gate.check_summary(ap, damaged, gate.GROUP, gate.AP_METRICS)
    damaged = paired.copy(); damaged.loc[0, "mean"] = .1
    with pytest.raises(gate.VerificationError, match="summary mean"):
        gate.check_summary(ap, damaged, gate.GROUP, gate.AP_METRICS, paired=True)


def test_reserved_lambda_column_and_empty_metric_denominators_are_verified(monkeypatch):
    _, _, residual = small_details(monkeypatch)
    summary, paired = analysis.summarise_block(residual, gate.PAIR_METRICS, ["lambda"], reference="residual_0")
    gate.check_summary(residual, summary, gate.GROUP, gate.PAIR_METRICS, ["lambda"])
    gate.check_summary(residual, paired, gate.GROUP, gate.PAIR_METRICS, ["lambda"], paired=True,
                       references=("residual_0",), reference_extra=(0.,))
    residual.loc[:, "total_auc_delta"] = np.nan
    summary, _ = analysis.summarise_block(residual, gate.PAIR_METRICS, ["lambda"], reference="residual_0")
    gate.check_summary(residual, summary, gate.GROUP, gate.PAIR_METRICS, ["lambda"])
    summary.loc[summary.metric == "total_auc_delta", "mean"] = 0
    with pytest.raises(gate.VerificationError, match="summary mean"):
        gate.check_summary(residual, summary, gate.GROUP, gate.PAIR_METRICS, ["lambda"])


def compatibility_fixture(tmp_path, monkeypatch):
    ap, threshold, _ = small_details(monkeypatch)
    old = ap[[*gate.CASE, "method", "boundary_positive_change_ap", "boundary_ap_edges_used"]].copy()
    for metric in gate.THRESHOLD_METRICS: old[metric] = .1
    old["boundary_f1_edges_used"] = 1
    paths = []
    for block, rows in (("temporal_localisation", old), ("symmetric_localisation", old.iloc[:0])):
        output = tmp_path / gate.BASE / block; output.mkdir(parents=True)
        path = output / "detail.csv"; rows.to_csv(path, index=False); paths.append(path)
    return ap, threshold, paths


@pytest.mark.parametrize("damage", ["old_value", "old_row", "new_value", "q95_value"])
def test_old_new_compatibility_rejects_changed_values_or_omitted_rows(tmp_path, monkeypatch, damage):
    ap, threshold, paths = compatibility_fixture(tmp_path, monkeypatch)
    assert gate.check_compatibility(tmp_path, ap, threshold)[0]["method_rows_compared"] == 15
    if damage in ("old_value", "old_row"):
        old = pd.read_csv(paths[0])
        if damage == "old_row": old = old.iloc[:-1]
        else: old.loc[0, "boundary_positive_change_ap"] = .2
        old.to_csv(paths[0], index=False)
    if damage == "new_value": ap.loc[0, "boundary_positive_change_ap"] = .2
    if damage == "q95_value": threshold.loc[(threshold.q == .95) & (threshold.threshold_mode == "raw_shared"), "boundary_slot_f1"] = .2
    with pytest.raises(gate.VerificationError):
        gate.check_compatibility(tmp_path, ap, threshold)


def test_replay_fit_is_blocked_before_body_and_restores_profile(tmp_path):
    called = []
    def fit():
        called.append(True)
    def verify(**kwargs):
        fit()
    original = sys.getprofile()
    with pytest.raises(gate.VerificationError, match="attempted fitting"):
        gate.guarded_replay(SimpleNamespace(verify=verify), tmp_path, tmp_path)
    assert called == [] and sys.getprofile() is original


@pytest.mark.parametrize("key", ["new_fits", "new_simulations"])
def test_verifier_cannot_report_new_fitting_or_simulation(tmp_path, key):
    report = {"status": "passed", "read_only": True, "new_fits": 0, "new_simulations": 0, key: 1}
    with pytest.raises(gate.VerificationError, match="reports new"):
        gate.guarded_replay(SimpleNamespace(verify=lambda **kw: report), tmp_path, tmp_path)


def test_provenance_missing_or_duplicate_record_fails_before_private_lookup(tmp_path):
    output = tmp_path / gate.BASE / "evaluation_provenance"; output.mkdir(parents=True)
    manifest = {"records": [], "external_time_attestation": False, "private_originals_required": False, "scope": "internal"}
    write_json(output / "export_manifest.json", manifest)
    with pytest.raises(gate.VerificationError, match="record count"):
        gate.check_provenance(tmp_path)


def provenance_fixture(root):
    output = root / gate.BASE / "evaluation_provenance"; output.mkdir(parents=True)
    boundary = root / gate.BASE / "boundary_sensitivity"; boundary.mkdir()
    write_json(boundary / "protocol.json", {"source_sha256": {"retained.py": "1" * 64}})
    write_json(boundary / "input_manifest.json", {"retained": True})
    write_json(boundary / "manifest.json", {"retained": True})
    actual = {name: gate.sha(boundary / name) for name in gate.inventory(boundary)}
    fields = {"new_fits": 0, "new_simulations": 0}
    values = {
        "boundary_freeze.json": {**fields, "before_freeze_utc": "2026-10-25T00:00:00+00:00",
            "after_freeze_before_computation_utc": "2026-10-25T00:00:01+00:00",
            "protocol_sha256": actual["protocol.json"], "input_manifest_sha256": actual["input_manifest.json"],
            "source_sha256": {"retained.py": "1" * 64}},
        "boundary_run.json": {**fields, "started_utc": "2026-10-25T00:00:02+00:00",
            "completed_utc": "2026-10-25T00:00:03+00:00", "manifest_sha256": actual["manifest.json"]},
        "boundary_rebuild.json": {**fields, "status": "passed", "started_utc": "2026-10-25T00:00:04+00:00",
            "completed_utc": "2026-10-25T00:00:05+00:00", "before_sha256": actual, "after_sha256": actual,
            "byte_identical_files": len(actual)},
        "boundary_preservation.json": {**fields, "before_after_inputs_equal": True,
            "frozen_input_manifest_sha256": actual["input_manifest.json"], "after_input_snapshot_sha256": actual["input_manifest.json"]},
        "segmentation_isolated_environment.json": {"environment": "recorded separately"},
    }
    records = []
    for name, value in values.items():
        write_json(output / name, value)
        records.append({"exported_file": name, "exported_sha256": gate.sha(output / name), "original_relative_path": "unavailable_private/" + name,
                        "original_sha256": "2" * 64, "normalization": "canonical JSON with declared redaction"})
    manifest = {"records": records, "external_time_attestation": False, "private_originals_required": False, "scope": "internal only"}
    write_json(output / "export_manifest.json", manifest)
    return output, manifest


def test_portable_provenance_links_internal_clock_without_private_logs(tmp_path):
    output, manifest = provenance_fixture(tmp_path)
    report, closure = gate.check_provenance(tmp_path)
    assert report["exported_records"] == 5 and report["private_originals_required"] is False
    assert not (tmp_path / "unavailable_private").exists()
    assert output / "export_manifest.json" in closure


@pytest.mark.parametrize("damage", ["missing", "corrupt", "duplicate", "origin_escape", "clock", "byte_invariance"])
def test_export_inventory_hashes_and_resigned_clock_evidence_are_checked(tmp_path, damage):
    output, manifest = provenance_fixture(tmp_path)
    record = manifest["records"][0]
    if damage == "missing": (output / record["exported_file"]).unlink()
    if damage == "corrupt": (output / record["exported_file"]).write_text("{}")
    if damage == "duplicate": manifest["records"][1] = deepcopy(record)
    if damage == "origin_escape": record["original_relative_path"] = "../private.json"
    if damage == "clock":
        value = json.loads((output / "boundary_run.json").read_text()); value["started_utc"] = "2026-10-24T00:00:00+00:00"
        write_json(output / "boundary_run.json", value)
        next(x for x in manifest["records"] if x["exported_file"] == "boundary_run.json")["exported_sha256"] = gate.sha(output / "boundary_run.json")
    if damage == "byte_invariance":
        value = json.loads((output / "boundary_rebuild.json").read_text()); value["before_sha256"]["protocol.json"] = "0" * 64
        write_json(output / "boundary_rebuild.json", value)
        next(x for x in manifest["records"] if x["exported_file"] == "boundary_rebuild.json")["exported_sha256"] = gate.sha(output / "boundary_rebuild.json")
    write_json(output / "export_manifest.json", manifest)
    with pytest.raises(gate.VerificationError):
        gate.check_provenance(tmp_path)

