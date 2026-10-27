"""Negative and read-only checks for the manuscript/artifact consistency gate."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import verify_information_sciences as gate


def test_hash_verification_is_readonly_and_rejects_corruption(tmp_path):
    payload = tmp_path / "retained.csv"
    payload.write_text("score\n0.5\n", encoding="utf-8")
    entries = {payload.name: gate.sha(payload)}
    before = (payload.read_bytes(), payload.stat().st_mtime_ns, set(tmp_path.iterdir()))
    assert gate.check_hashes(entries, tmp_path, "fixture") == 1
    assert before == (payload.read_bytes(), payload.stat().st_mtime_ns, set(tmp_path.iterdir()))
    payload.write_text("score\n0.6\n", encoding="utf-8")
    with pytest.raises(gate.VerificationError, match="checksum mismatch"):
        gate.check_hashes(entries, tmp_path, "fixture")


@pytest.mark.parametrize("name", ["../outside.csv", "nested/../../outside.csv", chr(67) + ":/outside.csv"])
def test_manifest_paths_cannot_escape_or_reenter_the_declared_root(tmp_path, name):
    with pytest.raises(gate.VerificationError, match="Invalid|escapes"):
        gate.inside(tmp_path, name)


def test_tiny_p_values_cannot_be_accepted_as_zero_and_auxiliary_null_is_preserved():
    expected = pd.DataFrame([{"contrast": "primary", "mean": 0.03, "p_raw": 4e-47, "p_holm": 1.6e-46},
                             {"contrast": "auxiliary", "mean": 0.02, "p_raw": 1e-4, "p_holm": np.nan}])
    records = [{"contrast": "primary", "mean": 0.03, "p_raw": 4e-47, "p_holm": 1.6e-46},
               {"contrast": "auxiliary", "mean": 0.02, "p_raw": 1e-4, "p_holm": None}]
    assert gate.compare_record_lists(records[::-1], expected, ("contrast",), "fixture") == 2
    wrong = deepcopy(records); wrong[0]["p_holm"] = 0
    with pytest.raises(gate.VerificationError, match="numeric/text drift"):
        gate.compare_record_lists(wrong, expected, ("contrast",), "fixture")
    wrong = deepcopy(records); wrong[1]["p_holm"] = 0
    with pytest.raises(gate.VerificationError, match="numeric/text drift"):
        gate.compare_record_lists(wrong, expected, ("contrast",), "fixture")


def test_duplicate_or_missing_interpretation_records_are_rejected():
    expected = pd.DataFrame([{"id": "a", "mean": -0.1}, {"id": "b", "mean": 0.1}])
    with pytest.raises(gate.VerificationError, match="record count"):
        gate.compare_record_lists([{"id": "a", "mean": -0.1}], expected, ("id",), "fixture")
    with pytest.raises(gate.VerificationError, match="duplicate"):
        gate.compare_record_lists([{"id": "a", "mean": -0.1}] * 2, expected, ("id",), "fixture")


def test_exported_bytes_are_checked_by_default_and_only_explicitly_skipped_after_rebuild(tmp_path):
    payload = tmp_path / "figure.pdf"; payload.write_bytes(b"original exported figure bytes")
    manifest = tmp_path / "PACKAGE_MANIFEST.json"
    manifest.write_text(json.dumps({"sha256": {payload.name: gate.sha(payload)}}), encoding="utf-8")
    assert gate.verify_bundle(tmp_path) == 1
    payload.write_bytes(b"legitimately rebuilt figure bytes")
    with pytest.raises(gate.VerificationError, match="Exported bundle checksum mismatch"):
        gate.verify_bundle(tmp_path)
    assert gate.verify_bundle(tmp_path, skip_bundle=True) == 0
    # The exception is scoped to the exported manifest; source/output manifests
    # still use the ordinary checksum checker without a skip parameter.
    with pytest.raises(gate.VerificationError, match="Asset output checksum mismatch"):
        gate.check_hashes({payload.name: "outdated asset checksum"}, tmp_path, "Asset output")


def test_supplement_manifest_cannot_silently_drop_a_source_csv_from_its_closure(tmp_path):
    output = tmp_path / "results/information_sciences"; output.mkdir(parents=True)
    frozen = tmp_path / "retained.csv"; frozen.write_text("repeat,score\n0,0.5\n", encoding="utf-8")
    for name in gate.SUPPLEMENT_OUTPUTS:
        (output / name).write_text("retained presentation fixture\n", encoding="utf-8")
    manifest = {"frozen_source_sha256": {frozen.name: gate.sha(frozen)},
                "sha256": {name: gate.sha(output / name) for name in gate.SUPPLEMENT_OUTPUTS}}
    path = output / "supplement_assets_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert gate.verify_supplement_manifest(tmp_path)["supplement_outputs"] == 8
    del manifest["sha256"]["supp_ols_absolute_source.csv"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(gate.VerificationError, match="three tables and five source CSVs"):
        gate.verify_supplement_manifest(tmp_path)


def manuscript_fixture(tmp_path):
    assets = tmp_path / "results/information_sciences"
    assets.mkdir(parents=True)
    paper = tmp_path / "paper"; paper.mkdir()
    abstract = assets / "abstract.tex"; abstract.write_text(" ".join(["Evidence"] * 182), encoding="utf-8")
    for table in gate.MAIN_TABLES:
        (assets / f"{table}.tex").write_text(r"\begin{tabular}{r}1\end{tabular}", encoding="utf-8")
    author_command = chr(92) + "author" + chr(123)
    parts = [r"\documentclass{article}", author_command + gate.AUTHOR_NAME + r"\\" + gate.AUTHOR_AFFILIATION +
             r"\\\texttt{" + gate.AUTHOR_EMAIL + "}}", r"\begin{document}",
             r"\begin{abstract}\input{results/information_sciences/abstract.tex}\end{abstract}",
             r"\section{Introduction}\begin{itemize}\item Evaluation\item Conditions\item Diagnostics\end{itemize}",
             r"\section{Results}"]
    parts.extend(r"\begin{figure}\includegraphics{results/information_sciences/" + name + r".pdf}\end{figure}" for name in sorted(gate.MAIN_FIGURES))
    parts.extend(r"\begin{table}\input{results/information_sciences/" + name + r".tex}\end{table}" for name in sorted(gate.MAIN_TABLES))
    parts.append(r"\end{document}")
    main = paper / "main.tex"; main.write_text("\n".join(parts), encoding="utf-8")
    return main, abstract


def test_abstract_count_resolves_the_actual_input_and_does_not_write(tmp_path):
    main, abstract = manuscript_fixture(tmp_path)
    before = {p.relative_to(tmp_path): (gate.sha(p), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    result = gate.verify_manuscript(tmp_path)
    assert result["abstract_words"] == 182 and result["contributions"] == 3
    assert result["main_figures"] == 6 and result["main_tables"] == 5
    after = {p.relative_to(tmp_path): (gate.sha(p), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
    abstract.write_text("Too short", encoding="utf-8")
    with pytest.raises(gate.VerificationError, match="Abstract has 2 words"):
        gate.verify_manuscript(tmp_path)


def test_real_author_and_exact_main_asset_inventory_are_enforced(tmp_path):
    main, _ = manuscript_fixture(tmp_path)
    original = main.read_text(encoding="utf-8")
    main.write_text(original.replace(gate.AUTHOR_NAME, "Anonymous Authors"), encoding="utf-8")
    with pytest.raises(gate.VerificationError, match="author"):
        gate.verify_manuscript(tmp_path)
    main.write_text(original.replace("fig_fallback.pdf", "unverified_extra.pdf"), encoding="utf-8")
    with pytest.raises(gate.VerificationError, match="figure references"):
        gate.verify_manuscript(tmp_path)
    main.write_text(original.replace(r"\item Diagnostics", r"\item Diagnostics\item Extra"), encoding="utf-8")
    with pytest.raises(gate.VerificationError, match="three substantive"):
        gate.verify_manuscript(tmp_path)


def test_recursive_input_paths_are_checked_and_cycles_do_not_hang(tmp_path):
    first = tmp_path / "first.tex"; second = tmp_path / "second.tex"
    first.write_text(r"\input{second}", encoding="utf-8")
    second.write_text(r"\input{first}", encoding="utf-8")
    with pytest.raises(gate.VerificationError, match="Cyclic"):
        gate.expand_inputs(first.read_text(), tmp_path, (first.resolve(),))
    with pytest.raises(gate.VerificationError, match="Invalid"):
        gate.expand_inputs(r"\input{../outside}", tmp_path)
