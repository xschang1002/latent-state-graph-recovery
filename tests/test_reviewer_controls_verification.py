"""Negative cases for the post-review gate, without copying production outputs."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "scripts")]
import verify_reviewer_controls as gate
import analyze_score_preserving_controls as controls
import analyze_temporal_localisation as local


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_detail():
    values = []
    for spec in controls.method_spec():
        name = spec["method"]
        temporal = .60 if name in {"forced_hard", "residual_hard_l000"} else .75
        pooled = .80 if name in {"forced_hard", "residual_hard_l000"} else .70
        values.append({"study": "A", "scorer": "lasso", "repeat_seed": 1, "duration": 300, "cell": "fixture",
            "method": name, "status": "ok", "gate_accepted": False,
            "temporal_delta_auroc": temporal, "pooled_temporal_auroc": pooled})
    return pd.DataFrame(values)


def test_gate_fails_closed_when_required_modules_are_missing(tmp_path):
    original = sys.dont_write_bytecode
    with pytest.raises(gate.VerificationError, match="Required file missing"):
        gate.verify(tmp_path, replay=False)
    assert list(tmp_path.iterdir()) == []
    assert sys.dont_write_bytecode == original


def test_sealed_payload_tampering_and_path_escape_are_rejected(tmp_path):
    output = tmp_path / "derived"; output.mkdir()
    data = output / "value.csv"; data.write_text("score\n.7\n", encoding="utf-8")
    write_json(output / "manifest.json", {"sha256": {"value.csv": gate.sha(data)}})
    gate.check_manifest(tmp_path, "derived")
    data.write_text("score\n.8\n", encoding="utf-8")
    with pytest.raises(gate.VerificationError, match="Checksum mismatch"):
        gate.check_manifest(tmp_path, "derived")
    for name in ("../escape.csv", "Z" + chr(58) + chr(92) + "outside.csv"):
        with pytest.raises(gate.VerificationError, match="Unsafe"):
            gate.check_hashes(tmp_path, {name: "0" * 64}, "untrusted inventory")


def test_extra_unsealed_file_is_not_silently_accepted(tmp_path):
    output = tmp_path / "derived"; output.mkdir()
    data = output / "value.csv"; data.write_text("score\n.7\n", encoding="utf-8")
    write_json(output / "manifest.json", {"sha256": {"value.csv": gate.sha(data)}})
    (output / "unlisted.txt").write_text("unsealed", encoding="utf-8")
    with pytest.raises(gate.VerificationError, match="inventory drift"):
        gate.check_manifest(tmp_path, "derived")


@pytest.mark.parametrize("error", ["fallback", "positive_shrink"])
def test_wrong_formula_implementation_is_rejected(error):
    gate.check_control_contract(controls)

    def broken(*args, **kwargs):
        emitted = controls.build_control_emissions(*args, **kwargs)
        if error == "fallback":
            emitted["raw_fallback"] = emitted["forced_hard"]
        else:
            emitted["shrink_l050"] = emitted["forced_hard"]
        return emitted

    with pytest.raises((gate.VerificationError, AssertionError)):
        gate.check_control_contract(SimpleNamespace(build_control_emissions=broken))


@pytest.mark.parametrize("method,metric", [("raw_fallback", "pooled_temporal_auroc"),
                                         ("shrink_l025", "temporal_delta_auroc")])
def test_resigned_but_wrong_record_rules_are_rejected(method, metric):
    data = fixture_detail(); gate.check_control_rows(data)
    data.loc[data.method == method, metric] = .13
    with pytest.raises(AssertionError):
        gate.check_control_rows(data)


def test_missing_method_and_duplicate_case_cannot_hide_in_coverage(monkeypatch):
    data = fixture_detail()
    monkeypatch.setattr(gate, "expected_cases", lambda symmetric=False: {("A", "lasso", 1, 300, "fixture")})
    assert gate.check_coverage(data, "controls", False)["method_rows"] == 13
    with pytest.raises(gate.VerificationError, match="lost or gained"):
        gate.check_coverage(data.iloc[:-1], "controls", False)
    with pytest.raises(gate.VerificationError, match="duplicate"):
        gate.check_coverage(pd.concat([data, data.iloc[:1]]), "controls", False)
    data.loc[0, "repeat_seed"] = 2
    with pytest.raises(gate.VerificationError, match="coverage differs"):
        gate.check_coverage(data, "controls", False)


def test_infinite_metrics_cannot_be_counted_as_evaluable(monkeypatch):
    data = fixture_detail()
    monkeypatch.setattr(gate, "expected_cases", lambda symmetric=False: {("A", "lasso", 1, 300, "fixture")})
    data.loc[0, "temporal_delta_auroc"] = np.inf
    with pytest.raises(gate.VerificationError, match="finite numbers"):
        gate.check_coverage(data, "controls", False)


def test_undefined_metric_stays_missing_in_summary_and_denominators():
    detail = fixture_detail().iloc[:1].copy()
    detail["temporal_delta_auroc"] = np.nan; detail["status"] = "undefined"
    summary = pd.DataFrame([{"study": "A", "scorer": "lasso", "duration": 300, "cell": "fixture",
        "method": detail.method.iloc[0], "metric": "temporal_delta_auroc", "n_planned": 1,
        "n_valid": 0, "n_missing": 1, "mean": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}])
    gate.check_summary(detail, summary)
    bad = summary.copy(); bad["mean"] = 0
    with pytest.raises(AssertionError): gate.check_summary(detail, bad)
    bad = summary.copy(); bad["n_valid"] = 1
    with pytest.raises(gate.VerificationError, match="denominator"):
        gate.check_summary(detail, bad)


def test_resigned_summary_cannot_omit_a_metric_or_repeat_a_row():
    detail = fixture_detail()
    summary = pd.DataFrame([{"study": "A", "scorer": "lasso", "duration": 300,
        "cell": "fixture", "method": method, "metric": metric}
        for method in gate.CONTROL_METHODS
        for metric in ("temporal_delta_auroc", "pooled_temporal_auroc")])
    gate.check_summary_coverage(detail, summary, "controls")
    with pytest.raises(gate.VerificationError, match="coverage differs"):
        gate.check_summary_coverage(detail, summary.iloc[:-1], "controls")
    with pytest.raises(gate.VerificationError, match="Duplicate summary"):
        gate.check_summary_coverage(detail, pd.concat([summary, summary.iloc[:1]]), "controls")


def test_zero_strength_and_all_positive_boundary_targets_keep_their_definition():
    gate.check_local_contract(local)
    with pytest.raises(gate.VerificationError, match="abstain"):
        gate.check_local_contract(SimpleNamespace(positive_change_ap=lambda y, s: .5))


def provenance_fixture(root):
    output = root / gate.BASE / "reviewer_provenance"; output.mkdir(parents=True)
    records = []
    for index, name in enumerate(sorted(gate.PROVENANCE_EXPORTS)):
        code = name.endswith(".py.txt")
        target = output / name
        target.write_text("VALUE = 1\n" if code else '{"retained": true}\n', encoding="utf-8", newline="\n")
        records.append({"exported_file": name, "exported_sha256": gate.sha(target),
            "original_relative_path": f"private_logs/record_{index}" + (".py" if code else ".json"),
            "original_sha256": gate.sha(target),
            "normalization": gate.CODE_NORMALIZATION if code else gate.JSON_NORMALIZATION})
    manifest = {"records": records, "external_time_attestation": False,
        "scope": "internal retained QA records", "no_hidden_change": "numeric outputs retained",
        "reviewer_material": "private source text excluded", "C_original_records": "retained in C seal"}
    write_json(output / "export_manifest.json", manifest)
    return output, manifest


def test_portable_provenance_does_not_require_private_originals(tmp_path):
    output, manifest = provenance_fixture(tmp_path)
    report, closure = gate.check_provenance(tmp_path)
    assert report["records"] == 17 and len(closure) == 18
    assert report["local_originals_checked"] == 0
    item = manifest["records"][0]; original = tmp_path / item["original_relative_path"]
    original.parent.mkdir(); original.write_bytes((output / item["exported_file"]).read_bytes())
    assert gate.check_provenance(tmp_path)[0]["local_originals_checked"] == 1
    original.write_text("changed", encoding="utf-8")
    with pytest.raises(gate.VerificationError, match="original checksum"):
        gate.check_provenance(tmp_path)


@pytest.mark.parametrize("damage", ["corrupt", "missing", "path_escape", "original_escape"])
def test_provenance_damage_and_unsafe_paths_fail_closed(tmp_path, damage):
    output, manifest = provenance_fixture(tmp_path)
    item = manifest["records"][0]; target = output / item["exported_file"]
    if damage == "corrupt":
        target.write_text("{}", encoding="utf-8")
    elif damage == "missing":
        target.unlink()
    else:
        item["exported_file" if damage == "path_escape" else "original_relative_path"] = "../escape.json"
        write_json(output / "export_manifest.json", manifest)
    with pytest.raises(gate.VerificationError):
        gate.check_provenance(tmp_path)


def test_historical_code_cannot_be_normalized_as_json(tmp_path):
    output, manifest = provenance_fixture(tmp_path)
    item = next(row for row in manifest["records"] if row["exported_file"].endswith(".py.txt"))
    item["normalization"] = gate.JSON_NORMALIZATION
    write_json(output / "export_manifest.json", manifest)
    with pytest.raises(gate.VerificationError, match="original bytes"):
        gate.check_provenance(tmp_path)
