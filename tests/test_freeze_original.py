"""Protect historical evidence from the default reproduction command."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import freeze_original as freeze


@pytest.mark.parametrize("arguments", [[], ["--check"]])
@pytest.mark.parametrize("status,expected", [("match", 0), ("drift", 1), ("no_detail", 1)])
def test_default_check_never_writes_or_replaces_historical_evidence(tmp_path, monkeypatch, arguments, status, expected):
    historical = tmp_path / "configs/original_frozen.yaml"
    historical.parent.mkdir()
    historical.write_bytes(b"historical baseline\nprovenance_note: retain this\n")
    reproduction = tmp_path / "results/original_reproduction.csv"
    reproduction.parent.mkdir()
    reproduction.write_bytes(b"original results\n")
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    monkeypatch.setattr(freeze, "reproduction_table", lambda: pd.DataFrame({"status": [status]}))
    monkeypatch.setattr(freeze, "git_state", lambda: pytest.fail("Read-only check must work without Git"))
    assert freeze.main(arguments) == expected
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_snapshot_refuses_existing_directory_before_computation(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "reproduction_table", lambda: pytest.fail("Must refuse before computing"))
    with pytest.raises(SystemExit) as error:
        freeze.main(["--snapshot-dir", str(tmp_path)])
    assert error.value.code == 2


def test_explicit_snapshot_is_created_separately_and_cannot_be_overwritten(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    tables = root / "results/tables"
    tables.mkdir(parents=True)
    historical = root / "configs/original_frozen.yaml"
    historical.write_text("retained provenance\n", encoding="utf-8")
    monkeypatch.setattr(freeze, "ROOT", root)
    monkeypatch.setattr(freeze, "TABLES", tables)
    monkeypatch.setattr(freeze, "reproduction_table", lambda: pd.DataFrame({"status": ["match"]}))
    destination = tmp_path / "new-snapshot"
    assert freeze.main(["--snapshot-dir", str(destination)]) == 0
    assert (destination / "original_frozen.yaml").is_file()
    assert (destination / "original_reproduction.csv").is_file()
    assert historical.read_text(encoding="utf-8") == "retained provenance\n"
    with pytest.raises(SystemExit):
        freeze.main(["--snapshot-dir", str(destination)])
