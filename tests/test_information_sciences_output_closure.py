"""Negative closure checks for post hoc diagnostics and distributable provenance."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import verify_information_sciences as gate


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def audit_manifest_fixture(root):
    output = root / "results/information_sciences/generator_audit"
    output.mkdir(parents=True)
    builder = root / "tools/audit_generator_outputs.py"
    builder.parent.mkdir()
    builder.write_text("fixture builder\n", encoding="utf-8")
    source = root / "retained_source.py"
    source.write_text("fixture frozen source\n", encoding="utf-8")
    sources = {source.name: gate.sha(source)}
    for name in gate.GENERATOR_AUDIT_OUTPUTS:
        value = {source.name: gate.sha(source)} if "before_sha256" in name or "after_sha256" in name else {}
        write_json(output / name, value)
    manifest = {"audit_builder_sha256": gate.sha(builder), "frozen_source_sha256": sources,
                "sha256": {name: gate.sha(output / name) for name in gate.GENERATOR_AUDIT_OUTPUTS}}
    write_json(output / "manifest.json", manifest)
    return output, manifest


def test_generator_audit_cannot_drop_diagnostic_rows_or_files_from_its_inventory(tmp_path):
    output, manifest = audit_manifest_fixture(tmp_path)
    del manifest["sha256"]["observed_trajectories.csv"]
    write_json(output / "manifest.json", manifest)
    with pytest.raises(gate.VerificationError, match="all nine"):
        gate.verify_generator_audit(tmp_path)


def test_generator_diagnostic_payload_corruption_is_rejected_before_recomputation(tmp_path):
    output, _ = audit_manifest_fixture(tmp_path)
    (output / "study_summary.csv").write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(gate.VerificationError, match="Generator audit output checksum mismatch"):
        gate.verify_generator_audit(tmp_path)


def test_generator_before_after_snapshots_must_match_even_if_the_manifest_is_resealed(tmp_path, monkeypatch):
    from types import SimpleNamespace
    output, manifest = audit_manifest_fixture(tmp_path)
    source_hashes = manifest["frozen_source_sha256"]
    write_json(output / "frozen_after_sha256.json", {"different.csv": "wrong"})
    manifest["sha256"]["frozen_after_sha256.json"] = gate.sha(output / "frozen_after_sha256.json")
    write_json(output / "manifest.json", manifest)
    module = SimpleNamespace(frozen_hashes=lambda root: source_hashes,
                             frozen_source_hashes=lambda root: source_hashes)
    monkeypatch.setattr(gate.importlib, "import_module", lambda name: module)
    with pytest.raises(gate.VerificationError, match="before/after snapshots differ"):
        gate.verify_generator_audit(tmp_path)


def provenance_fixture(root):
    prefix = "results/information_sciences/provenance/"
    directory = root / prefix
    directory.mkdir(parents=True)
    exports = []
    for name in sorted(gate.PROVENANCE_EXPORTS):
        path = directory / name
        path.write_text("fixture exported record\n", encoding="utf-8")
        exports.append({"exported_relative_path": prefix + name, "exported_sha256": gate.sha(path),
                        "bytes": path.stat().st_size, "original_relative_path": "private_logs/" + name,
                        "original_sha256": gate.sha(path)})
    chronology = directory / "chronology.json"
    write_json(chronology, {"events": []})
    manifest = {"exports": exports, "derived_chronology": {"path": prefix + "chronology.json", "sha256": gate.sha(chronology)}}
    write_json(directory / "export_manifest.json", manifest)
    return directory, manifest


def test_public_provenance_verification_needs_no_private_original_logs_and_writes_nothing(tmp_path):
    directory, _ = provenance_fixture(tmp_path)
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in directory.iterdir()}
    _, counts = gate.verify_provenance_exports(tmp_path)
    assert counts == {"exported_records": 8, "optional_private_originals_checked": 0}
    assert before == {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in directory.iterdir()}


def test_missing_or_corrupted_export_cannot_pass_a_resealed_export_manifest(tmp_path):
    directory, manifest = provenance_fixture(tmp_path)
    removed = manifest["exports"].pop()
    write_json(directory / "export_manifest.json", manifest)
    with pytest.raises(gate.VerificationError, match="all eight"):
        gate.verify_provenance_exports(tmp_path)
    manifest["exports"].append(removed)
    write_json(directory / "export_manifest.json", manifest)
    (tmp_path / removed["exported_relative_path"]).write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(gate.VerificationError, match="Provenance export checksum mismatch"):
        gate.verify_provenance_exports(tmp_path)


def test_chronology_uses_retained_clock_fields_and_cannot_omit_one(tmp_path):
    name = "results/information_sciences/provenance/A_freeze_stage.json"
    write_json(tmp_path / name, {"before_freeze_utc": "2026-10-23T20:03:25+00:00"})
    event = {"source": name, "field": "before_freeze_utc", "utc": "2026-10-23T20:03:25+00:00",
             "asia_shanghai": "2026-10-24T04:03:25+08:00", "local_date": "2026-10-24"}
    chronology = {"timezone": "Asia/Shanghai", "events": [event]}
    assert gate.verify_chronology(tmp_path, chronology, {name}) == 1
    chronology["events"] = []
    with pytest.raises(gate.VerificationError, match="omits a retained"):
        gate.verify_chronology(tmp_path, chronology, {name})
    chronology["events"] = [{**event, "asia_shanghai": "2026-10-24T04:04:25+08:00"}]
    with pytest.raises(gate.VerificationError, match="local conversion"):
        gate.verify_chronology(tmp_path, chronology, {name})


def test_record_templates_expand_all_paired_cases_and_reject_unknown_placeholders():
    assert len(gate._template_paths("records/repeat_{seed}_D{300|600}_{lasso|ols}.npz", [1, 2])) == 8
    with pytest.raises(gate.VerificationError, match="Unknown"):
        gate._template_paths("records/repeat_{unknown}.npz", [1])
    with pytest.raises(gate.VerificationError, match="Malformed"):
        gate._template_paths("records/repeat_{unknown.npz", [1])
