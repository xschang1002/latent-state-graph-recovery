"""A documented fresh analysis must preserve historical provenance clocks."""
from pathlib import Path
import json
import shlex
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tools")]
import analyze_temporal_localisation as ab
import analyze_symmetric_localisation as c
import analyze_boundary_sensitivity as sensitivity
import run_symmetric_sensitivity as generator
import verify_reviewer_controls as gate


def commands():
    lines = (ROOT / "docs/information_sciences_artifact_readme.md").read_text(encoding="utf-8").splitlines()
    return [shlex.split(line) for line in lines if line.startswith("python -B scripts/analyze_")
            and "localisation.py " in line and "results/reproduction_fresh/direct_" in line]


def option(command, name):
    return command[command.index(name) + 1]


def provenance(root, clock_name):
    folder = root / gate.BASE / "reviewer_provenance"
    folder.mkdir(parents=True)
    records = []
    historical = root / "experiments/logs/reviewer_controls_2026-10-25" / clock_name
    historical.parent.mkdir(parents=True)
    historical.write_text('{"historical_clock": true}\n', encoding="utf-8")
    for index, name in enumerate(sorted(gate.PROVENANCE_EXPORTS)):
        path = folder / name
        code = name.endswith(".py.txt")
        path.write_text("VALUE = 1\n" if code else '{"exported": true}\n', encoding="utf-8")
        original = historical.relative_to(root).as_posix() if name == clock_name else (
            f"private_origins/{index}" + (".py" if code else ".json"))
        records.append({"exported_file": name, "exported_sha256": gate.sha(path),
                        "original_relative_path": original,
                        "original_sha256": gate.sha(historical) if name == clock_name else gate.sha(path),
                        "normalization": gate.CODE_NORMALIZATION if code else gate.JSON_NORMALIZATION})
    manifest = {"records": records, "external_time_attestation": False,
                "scope": "fixture clocks", "no_hidden_change": "fixture values retained",
                "reviewer_material": "not present", "C_original_records": "fixture seal"}
    (folder / "export_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return historical


def test_fresh_commands_isolate_both_results_and_execution_records():
    lines = commands()
    assert len(lines) == 6
    for script in ("scripts/analyze_temporal_localisation.py", "scripts/analyze_symmetric_localisation.py"):
        selected = [line for line in lines if line[2] == script]
        assert {line[3] for line in selected} == {"freeze", "run", "verify"}
        assert len({option(line, "--output-dir") for line in selected}) == 1
        assert len({option(line, "--stage-dir") for line in selected}) == 1
        for line in selected:
            assert option(line, "--stage-dir").startswith("experiments/logs/reproduction_fresh/")
            assert "reviewer_controls_2026-10-25" not in option(line, "--stage-dir")
    assert len({option(line, "--stage-dir") for line in lines}) == 2


@pytest.mark.parametrize("module,clock_name", [(ab, "temporal_localisation_freeze.json"),
                                                (c, "symmetric_localisation_freeze.json")])
def test_documented_freeze_then_provenance_check_preserves_original(tmp_path, monkeypatch, module, clock_name):
    old = provenance(tmp_path, clock_name)
    before = old.read_bytes()
    assert gate.check_provenance(tmp_path)[0]["local_originals_checked"] == 1
    script = f"scripts/{Path(module.__file__).name}"
    command = next(line for line in commands() if line[2] == script and line[3] == "freeze")
    output = tmp_path / option(command, "--output-dir")
    stage = tmp_path / option(command, "--stage-dir")
    fixture_protocol = {"frozen_input_sha256": {}, "frozen_source_sha256": {}, "frozen_C_input_sha256": {},
                        "definition_protocol_sha256": "1" * 64, "methods": []}
    monkeypatch.setattr(module, "protocol", lambda root: fixture_protocol)
    monkeypatch.setattr(sys, "argv", [script, "freeze", "--root", str(tmp_path),
                                     "--output-dir", str(output), "--stage-dir", str(stage)])
    module.main()
    assert (output / "protocol.json").is_file()
    assert (stage / clock_name).is_file()
    assert old.read_bytes() == before
    assert gate.check_provenance(tmp_path)[0]["local_originals_checked"] == 1


def test_historical_stage_collision_is_still_rejected(tmp_path, monkeypatch):
    old = provenance(tmp_path, "temporal_localisation_freeze.json")
    monkeypatch.setattr(ab, "protocol", lambda root: {"frozen_input_sha256": {}, "frozen_source_sha256": {}})
    ab.freeze(tmp_path, tmp_path / "fresh_analysis", old.parent)
    with pytest.raises(gate.VerificationError, match="original checksum mismatch"):
        gate.check_provenance(tmp_path)


@pytest.mark.parametrize("script,flag", [("analyze_boundary_sensitivity.py", "--log-dir"),
                                         ("run_symmetric_sensitivity.py", "--log-dir"),
                                         ("run_segmentation_validation.py", "--stage-dir")])
def test_other_fresh_routes_also_isolate_execution_records(script, flag):
    lines=(ROOT/"docs/information_sciences_artifact_readme.md").read_text(encoding="utf-8").splitlines()
    selected=[shlex.split(line) for line in lines if line.startswith(f"python -B scripts/{script} ")
              and "results/reproduction_fresh/" in line]
    assert len(selected)==3
    for mode in ("--freeze","--run","--verify"):
        line=next(line for line in selected if mode in line)
        assert option(line,flag).startswith("experiments/logs/reproduction_fresh/")
    assert len({option(line,flag) for line in selected})==1


@pytest.mark.parametrize("block", ["boundary", "generator_C"])
def test_fresh_freeze_uses_documented_log_and_keeps_old_clock(tmp_path,monkeypatch,block):
    script="analyze_boundary_sensitivity.py" if block=="boundary" else "run_symmetric_sensitivity.py"
    clock="freeze.json" if block=="boundary" else "symmetric_sensitivity_freeze.json"
    lines=(ROOT/"docs/information_sciences_artifact_readme.md").read_text(encoding="utf-8").splitlines()
    line=next(shlex.split(line) for line in lines if line.startswith(f"python -B scripts/{script} --freeze ")
              and "results/reproduction_fresh/" in line)
    output=tmp_path/option(line,"--output-dir"); log=tmp_path/option(line,"--log-dir")
    old=tmp_path/"historical_execution"/clock;old.parent.mkdir();old.write_text("historical",encoding="utf-8")
    if block=="boundary":
        monkeypatch.setattr(sensitivity,"input_snapshot",lambda root:{"fixture":True})
        monkeypatch.setattr(sensitivity,"protocol",lambda digest,root:{"source_sha256":{},"fixture":True})
        sensitivity.freeze(tmp_path,output,log)
    else:
        generator.freeze(output,{"source_sha256":{},"fixture":True},log)
    assert (log/clock).is_file()
    assert old.read_text(encoding="utf-8")=="historical"
