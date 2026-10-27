"""Regression coverage for recursive LaTeX checks and source artifact closure."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))
from validate_paper_static import syntax_errors, validate  # noqa: E402
from build_overleaf_package import build_package  # noqa: E402
from build_submission_package import (collect_entries, should_include, specialist_snapshot,
                                     snapshot_entry, scan_identity, scan_text, zip_package)  # noqa: E402
from summarize_all_recovery_methods import latex_text, write_summary_latex, DEFAULT_METRICS  # noqa: E402


def test_recursive_table_underscore_is_rejected_and_escaped_text_passes(tmp_path):
    (tmp_path / "tables").mkdir()
    (tmp_path / "main.tex").write_text(r"\input{tables/outer_table}", encoding="utf-8")
    (tmp_path / "tables/outer_table.tex").write_text(r"\input{tables/inner_table}", encoding="utf-8")
    inner = tmp_path / "tables/inner_table.tex"
    inner.write_text("\\begin{tabular}{ll}\nnonlinear_var & switching_var \\\\\n\\end{tabular}\n", encoding="utf-8")
    report = validate(tmp_path, Path("main.tex"))
    assert report["tex_file_count"] == 3
    assert len(report["errors"]) == 2
    assert all("tables/inner_table.tex:2" in error and "unescaped '_'" in error for error in report["errors"])
    inner.write_text(inner.read_text().replace("_", r"\_"), encoding="utf-8")
    assert validate(tmp_path, Path("main.tex"))["errors"] == []


def test_math_literal_arguments_comments_and_verbatim_are_not_false_positives():
    text = r"""
Text\_field and \{escaped\} \% \$.
$x_i^2$ and $$y_t$$ and \(z_i\) and \[a_j^2\].
\begin{align*} a_i &= b_j^2 \\ c_i &= 0 \end{align*}
\ensuremath{x_i^2} and \textbf{clean text}.
\textbf{bold $x_i$} and $x_i + \text{inside $y_i$ text}$.
\label{key_with_under} \ref{key_with_under} \cite{key_with_under}
\includegraphics[width=\linewidth]{figure_name.png}
\href{https://example.org/a_b}{display\_name}
\url{https://example.org/a_b} \verb|verbatim_#^%|
\begin{CCSXML}<concept_id>123</concept_id>\end{CCSXML}
% ignored_under ^ { \input{missing}
"""
    assert syntax_errors(text, "example.tex") == []
    assert syntax_errors(r"\href{https://example.org/a_b}{bad_name}", "x.tex")
    assert syntax_errors(r"$x_i + \text{bad_name}$", "x.tex")


def test_syntax_checks_unbalanced_math_braces_and_environments():
    assert syntax_errors(r"$x_i", "x.tex")
    assert syntax_errors(r"\begin{tabular}{ll}x\end{table}", "x.tex")
    assert syntax_errors(r"text}", "x.tex")


def test_generator_escapes_csv_fields_without_changing_metric_values(tmp_path):
    row = {"generator": "switching_var", "method": "method_1 & 2%", "fit_seconds_mean": 1.5}
    for metric in DEFAULT_METRICS:
        row[f"{metric}_mean"], row[f"{metric}_std"] = 0.75, 0.125
    output = tmp_path / "table.tex"
    write_summary_latex(output, [row])
    text = output.read_text()
    assert r"switching\_var & method\_1 \& 2\%" in text
    assert r"0.750 $\pm$ 0.125" in text
    assert syntax_errors(text, "table.tex") == []
    assert latex_text("a$b#c{d}e^f~g\\h") == r"a\$b\#c\{d\}e\textasciicircum{}f\textasciitilde{}g\textbackslash{}h"


def test_submission_includes_new_untracked_tools_and_lockfile(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    required = ["docs/anonymous_artifact_readme.md", "requirements.txt", "requirements-revision.txt", "pyproject.toml",
                "tools/build_revision_tables.py", "tools/verify_paper_numbers.py", "tools/validate_paper_static.py",
                "tools/build_overleaf_package.py", "docs/reproduction_manifest.md", "paper/main.tex", "paper/references.bib",
                "scripts/build_submission_package.py"]
    for name in required + ["tools/new_untracked_tool.py", "tools/__pycache__/bad.pyc", "data/raw/private.csv"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture", encoding="utf-8")
    names = {entry.archive_name for entry in collect_entries(False, tmp_path)}
    assert "requirements-revision.txt" in names
    assert "tools/new_untracked_tool.py" in names
    assert "tools/verify_paper_numbers.py" in names
    assert "scripts/build_submission_package.py" in names
    assert "tools/__pycache__/bad.pyc" not in names
    assert "data/raw/private.csv" not in names
    assert should_include("tools/build_revision_tables.py")
    assert not should_include("results/tables/private.npz")
    assert not should_include("results/figures/model.pt")


def test_overleaf_package_closure_and_fresh_extraction(tmp_path):
    (tmp_path / "paper").mkdir()
    (tmp_path / "tables").mkdir()
    (tmp_path / "paper/main.tex").write_text(r"\input{tables/a_table}\bibliography{paper/references}", encoding="utf-8")
    (tmp_path / "paper/references.bib").write_text("", encoding="utf-8")
    (tmp_path / "tables/a_table.tex").write_text(r"safe\_text", encoding="utf-8")
    output = tmp_path / "package.zip"
    report = build_package(tmp_path, output)
    assert report["file_count"] == 3 and report["pdf_compiled"] is False
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {"main.tex", "paper/references.bib", "tables/a_table.tex"}
        archive.extractall(tmp_path / "extracted")
    assert validate(tmp_path / "extracted", Path("main.tex"))["errors"] == []
    assert build_package(tmp_path, output)["sha256"] == report["sha256"]


def _specialist_export_fixture(root):
    required = ["docs/anonymous_artifact_readme.md", "requirements.txt", "requirements-revision.txt", "pyproject.toml",
                "tools/build_revision_tables.py", "tools/verify_paper_numbers.py", "tools/validate_paper_static.py",
                "tools/build_overleaf_package.py", "docs/reproduction_manifest.md", "paper/main.tex", "paper/references.bib",
                "scripts/build_submission_package.py", "scripts/run_revision_specialist_stress.py",
                "tools/summarize_revision_specialist.py", "docs/revision_specialist_protocol_2026-09-13.md"]
    for name in required:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture\n", encoding="utf-8")
    runner = "scripts/run_revision_specialist_stress.py"
    doc = "docs/revision_specialist_protocol_2026-09-13.md"
    spec = {"seeds": [100], "methods": ["STCG-v1"],
            "source_hashes": {runner: hashlib.sha256((root / runner).read_bytes()).hexdigest()},
            "protocol_document_sha256": hashlib.sha256((root / doc).read_bytes()).hexdigest(),
            "display_cells": [{"setting_id": "abcdef123456", "axis": "obs_noise", "value": 0, "config": {"n": 8}}]}
    protocol_hash = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    table = root / "results/tables"
    (table / "revision_specialist_parts").mkdir(parents=True)
    (table / "revision_specialist_protocol.json").write_text(json.dumps({"specification": spec, "protocol_hash": protocol_hash}))
    record = {"phase": "stress", "setting_id": "abcdef123456", "config": {"n": 8}, "seed": 100,
              "method": "STCG-v1", "protocol_hash": protocol_hash, "status": "ok", "edge_auc": 0.75}
    part = table / "revision_specialist_parts/stress_abcdef123456_seed100_stcgv1.json"
    part.write_text(json.dumps(record))
    return table, part, record


def test_ignored_specialist_evidence_is_allowlisted_snapshotted_and_exportable(tmp_path):
    import pytest
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    table, part, record = _specialist_export_fixture(tmp_path)
    (tmp_path / ".gitignore").write_text("*.json\n*.log\n")
    (table / "private.json").write_text('{"private": 1}')
    (part.parent / "private.json").write_text('{"private": 1}')
    (part.parent / (part.stem + ".tmp")).write_text("unfinished")
    real_dir = tmp_path / "experiments/logs/revision_real_validation"
    real_dir.mkdir(parents=True)
    (real_dir / "executed_manifest.json").write_text('{"signature": "fixture"}')
    (real_dir / "reporting_manifest.json").write_text('{"execution_signature": "fixture"}')
    (real_dir / "forward_private.json").write_text('{"raw": [1, 2]}')
    entries = collect_entries(False, tmp_path)
    names = {entry.archive_name for entry in entries}
    assert "results/tables/revision_specialist_protocol.json" in names
    assert "experiments/logs/revision_real_validation/executed_manifest.json" in names
    assert "experiments/logs/revision_real_validation/reporting_manifest.json" in names
    assert part.relative_to(tmp_path).as_posix() in names
    assert not any("private" in name or name.endswith(".tmp") for name in names)
    report = specialist_snapshot(entries)
    assert report["included_job_records"] == 1 and report["missing_job_records"] == 1
    with pytest.raises(ValueError, match="incomplete"):
        specialist_snapshot(entries, require_complete=True)
    # Mutating the live checkpoint after collection must not change ZIP bytes.
    original = part.read_bytes()
    part.write_text("concurrent write")
    output = tmp_path / "artifact.zip"
    zip_package(entries, output, {"specialist_checkpoint": report})
    with zipfile.ZipFile(output) as archive:
        assert archive.read(part.relative_to(tmp_path).as_posix()) == original
        archive.extractall(tmp_path / "extracted")
    extracted = collect_entries(False, tmp_path / "extracted")
    assert specialist_snapshot(extracted) == report
    assert scan_identity(extracted) == []
    # A retained failure counts as a finished record, not a successful result.
    record.update(phase="timing", status="failed", edge_auc=None)
    timing = tmp_path / "extracted/results/tables/revision_specialist_parts/timing_abcdef123456_seed100_stcgv1.json"
    timing.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="summaries are missing/stale"):
        specialist_snapshot(collect_entries(False, tmp_path / "extracted"), True)
    provenance = tmp_path / "extracted/results/tables/revision_specialist_analysis_provenance.json"
    provenance.write_text(json.dumps({"summary": {"stored_fits": 2}}))
    assert specialist_snapshot(collect_entries(False, tmp_path / "extracted"), True)["complete"] is True


def test_specialist_export_preserves_original_evidence_and_rejects_changed_sources(tmp_path):
    import pytest
    table, part, record = _specialist_export_fixture(tmp_path)
    # Construct synthetic paths so the artifact's identity scan does not mistake
    # the regression fixture itself for an actual developer-machine reference.
    drive = "Z:"
    record.update(status="failed", error='File "' + drive + '\\fixture\\stcg-revision\\scripts\\runner.py"',
                  traceback='File "' + drive + '\\runtime\\Lib\\json\\encoder.py"')
    part.write_text(json.dumps(record))
    original = part.read_bytes()
    entry = snapshot_entry(part, part.relative_to(tmp_path).as_posix())
    safe = json.loads(entry.read_bytes())
    assert part.read_bytes() == original
    assert safe["edge_auc"] == record["edge_auc"] and safe["config"] == record["config"]
    assert "<artifact>/scripts/runner.py" in safe["error"]
    assert "<python-runtime>/Lib/json/encoder.py" in safe["traceback"]
    assert entry.original_sha256 == hashlib.sha256(original).hexdigest()
    assert scan_identity([entry]) == []
    assert scan_text(drive + r'\\escaped\\private', "x.json")
    assert scan_text(drive + '/' + 'Users' + '/fixture/home', "x.json")
    (tmp_path / "scripts/run_revision_specialist_stress.py").write_text("changed")
    with pytest.raises(ValueError, match="source dependency mismatch"):
        specialist_snapshot(collect_entries(False, tmp_path))


def test_specialist_export_rejects_mismatched_record_and_protocol(tmp_path):
    import pytest
    table, part, record = _specialist_export_fixture(tmp_path)
    record["seed"] = 101
    part.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity/configuration mismatch"):
        specialist_snapshot(collect_entries(False, tmp_path))
    protocol = table / "revision_specialist_protocol.json"
    frozen = json.loads(protocol.read_text())
    frozen["specification"]["seeds"] = [101]
    protocol.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        specialist_snapshot(collect_entries(False, tmp_path))


def test_specialist_history_hashes_survive_anonymous_round_trip(tmp_path):
    import pytest
    table, part, record = _specialist_export_fixture(tmp_path)
    history = table / "revision_specialist_parts_history"
    history.mkdir()
    previous = history / part.name
    previous.write_bytes(part.read_bytes())
    log = previous.with_suffix(".log")
    log.write_text('File "' + 'Z:' + '\\fixture\\stcg-revision\\scripts\\runner.py"')
    spec = json.loads((table / "revision_specialist_protocol.json").read_text())
    runner_hash = spec["specification"]["source_hashes"]["scripts/run_revision_specialist_stress.py"]
    amendment = {"protocol_hash": spec["protocol_hash"], "old_runner_sha256": runner_hash,
                 "new_runner_sha256": runner_hash, "serialization_retries": [{
                     "job": part.stem, "original_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                     "original_record": previous.relative_to(table).as_posix(),
                     "original_record_sha256": hashlib.sha256(previous.read_bytes()).hexdigest()}]}
    (table / "revision_specialist_storage_amendment.json").write_text(json.dumps(amendment))
    entries = collect_entries(False, tmp_path)
    specialist_snapshot(entries)
    redactions = {entry.archive_name: {"original_sha256": entry.original_sha256,
                   "exported_sha256": hashlib.sha256(entry.read_bytes()).hexdigest()}
                  for entry in entries if entry.original_sha256}
    assert len(redactions) == 1
    package = tmp_path / "artifact.zip"
    zip_package(entries, package, {"diagnostic_redactions": redactions})
    with zipfile.ZipFile(package) as archive:
        archive.extractall(tmp_path / "extracted")
    extracted = collect_entries(False, tmp_path / "extracted")
    specialist_snapshot(extracted)
    exported_log = next(entry for entry in extracted if entry.archive_name.endswith(".log"))
    assert exported_log.original_sha256 == hashlib.sha256(log.read_bytes()).hexdigest()
    log.write_text("tampered history")
    with pytest.raises(ValueError, match="history hash mismatch"):
        specialist_snapshot(collect_entries(False, tmp_path))


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _manuscript_title() -> str:
    source = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    match = re.search(r"\\title\{(.+?)\}\s*\n\s*\\author", source, re.S)
    assert match, "main.tex must declare \\title{...} before \\author"
    return _normalise_whitespace(match.group(1).replace("\\\\", " "))


def test_artifact_readme_states_current_manuscript_title():
    """The packaged README is written from this file, so a stale title ships silently."""
    title = _manuscript_title()
    readme = _normalise_whitespace(
        (ROOT / "docs" / "information_sciences_artifact_readme.md").read_text(encoding="utf-8")
    )
    assert title in readme, f"artifact README does not state the current title: {title}"


def test_identity_scan_covers_orcid_github_tex_author_and_pdf_metadata(tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text(
        "orcid 0000-0002-1825-" + "0097\nhttps://github.com/" + "example-user\n", encoding="utf-8"
    )
    tex = tmp_path / "paper.tex"
    # Split so the fixture itself does not trip the scanner when tests/ is packaged.
    tex.write_text("\\auth" + "or{" + "Jane Doe}\n", encoding="utf-8")
    named_pdf = tmp_path / "named.pdf"
    named_pdf.write_bytes(b"%PDF-1.4\n/Author (Jane Doe)\n/Title (x)\n%%EOF\n")
    anonymous_pdf = tmp_path / "anonymous.pdf"
    anonymous_pdf.write_bytes(b"%PDF-1.4\n/Author (Anonymous Authors)\n%%EOF\n")

    entries = [
        snapshot_entry(text, "notes.txt"),
        snapshot_entry(tex, "paper.tex"),
        snapshot_entry(named_pdf, "named.pdf"),
        snapshot_entry(anonymous_pdf, "anonymous.pdf"),
    ]
    findings = scan_identity(entries)
    patterns = {finding["pattern"] for finding in findings}
    assert {"orcid", "github_profile", "tex_non_anonymous_author", "pdf_author"} <= patterns
    assert all(finding["file"] != "anonymous.pdf" for finding in findings)


def test_temporal_delta_table_matches_committed_bytes():
    """The core temporal table must come from the CSV, not a renamed scratch file."""
    from build_revision2_tables import table_temporal_delta

    committed = (ROOT / "results" / "revision2" / "tab_temporal_delta.tex").read_text(
        encoding="utf-8"
    )
    assert table_temporal_delta() + "\n" == committed


def test_ess_table_matches_current_experiment_csv():
    from build_revision2_tables import table_ess

    committed = (ROOT / "results/revision2/tab_ess.tex").read_text(encoding="utf-8")
    assert table_ess() + "\n" == committed


def test_ess_diagnostic_table_matches_chain_csv():
    from build_revision2_tables import table_ess_diagnostics
    assert table_ess_diagnostics() + "\n" == (ROOT / "results/revision2/tab_ess_diagnostics.tex").read_text()


def test_temporal_grid_tables_match_paired_source_data():
    from build_revision2_tables import table_temporal_grid, table_temporal_gmm_numerics
    for name, actual in (("tab_temporal_grid.tex", table_temporal_grid()),
                         ("tab_temporal_grid_accepted.tex", table_temporal_grid(True)),
                         ("tab_temporal_gmm_numerics.tex", table_temporal_gmm_numerics())):
        assert actual + "\n" == (ROOT / "results/revision2" / name).read_text()


def test_grouping_diagnostics_and_coverage_tables_match_retained_outputs():
    from build_revision2_tables import table_grouping_factors, table_matched_grouping, table_temporal_coverage
    for name, actual in (("tab_grouping_factors.tex", table_grouping_factors()),
                         ("tab_matched_grouping.tex", table_matched_grouping()),
                         ("tab_temporal_coverage.tex", table_temporal_coverage())):
        assert actual + "\n" == (ROOT / "results/revision2" / name).read_text()
    from build_submission_package import collect_entries
    entries = {entry.archive_name for entry in collect_entries(False)}
    assert {"scripts/reproduce_paper.py", "results/revision2/grouping_replay_outputs.npz"} <= entries


def test_manuscript_has_compact_abstract_three_contributions_and_reproduction_entry():
    source = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    abstract = source.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
    abstract_input = re.fullmatch(r"\s*\\input\{([^}]+)\}\s*", abstract)
    if abstract_input:
        abstract = (ROOT / abstract_input.group(1)).read_text(encoding="utf-8")
    prose = re.sub(r"\$[^$]*\$", " VALUE ", abstract)
    assert 180 <= len(prose.split()) <= 195
    assert "independent paired" in abstract
    contributions = source.split(r"\paragraph{Contributions.}", 1)[1].split(r"\end{itemize}", 1)[0]
    assert contributions.count(r"\item") == 3
    assert "K-means reaches it" not in source and "K-means reaches that" not in source
    reproduction = (ROOT / "docs/information_sciences_artifact_readme.md").read_text(encoding="utf-8")
    assert "reproduce_paper.py --verify" in reproduction
    assert "information_sciences_reproducibility.zip" in reproduction


def test_temporal_grid_figure_uses_paired_gains_and_real_gate_frequencies(monkeypatch):
    import numpy as np
    import pandas as pd
    from matplotlib.figure import Figure
    from build_revision2_tables import figure_temporal_grid, temporal_grid_scans
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *args, **kwargs: figures.append(self))
    figure_temporal_grid()
    assert len(figures) == 2 and figures[0] is figures[1]
    figure = figures[0]
    frame = pd.read_csv(ROOT / "results/revision2/temporal_grid_summary.csv")
    methods = [("kmeans_hard", "delta_temporal"), ("kmeans_hard", "candidate_delta_temporal"),
               ("gmm_soft", "delta_temporal"), ("oracle_label", "delta_temporal")]
    for column, (_, _, scan) in enumerate(temporal_grid_scans(frame)):
        entries = scan[scan.subset == "all"].grid_entry.drop_duplicates()
        for row_index, subset in enumerate(("all", "hard_gate_accepted")):
            expected_methods = methods if row_index == 0 else [methods[i] for i in (0, 2, 3)]
            axis = figure.axes[row_index * 2 + column]
            for line, (method, metric) in zip(axis.lines, expected_methods):
                source = scan[(scan.subset == subset) & (scan.method == method)].set_index("grid_entry").loc[entries]
                np.testing.assert_allclose(line.get_ydata(), source[metric + "_mean"])
                assert min(line.get_ydata()) >= axis.get_ylim()[0]
                assert max(line.get_ydata()) <= axis.get_ylim()[1]
        gate = figure.axes[4 + column]
        source = scan[(scan.subset == "all") & (scan.method == "raw_window")].set_index("grid_entry").loc[entries]
        np.testing.assert_allclose([patch.get_height() for patch in gate.patches], source.hard_gate_acceptance)


def test_temporal_vector_figure_is_included_without_allowing_arbitrary_pdfs():
    from build_submission_package import should_include, collect_entries, scan_identity
    assert should_include("results/figures/rev2_temporal_grid.pdf")
    assert not should_include("results/figures/private_notes.pdf")
    entries = [entry for entry in collect_entries(False) if entry.archive_name == "results/figures/rev2_temporal_grid.pdf"]
    assert len(entries) == 1 and scan_identity(entries) == []


def test_ess_diagnostic_figure_uses_complete_reference_traces_without_acf_clipping(monkeypatch):
    import numpy as np
    import pandas as pd
    from matplotlib.figure import Figure
    from build_revision2_tables import figure_ess_diagnostics
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *args, **kwargs: figures.append(self))
    figure_ess_diagnostics()
    data = pd.read_csv(ROOT / "results/tables/ess_null_traces.csv")
    for column, edge in enumerate(("all_edges", "strong_only")):
        expected = data[(data.binarisation == edge) & (data.null_seed == 0) & (data.burn_in_per_edge == 100)]
        trace_axis = figures[0].axes[column]
        np.testing.assert_allclose(trace_axis.lines[0].get_ydata(), expected.null_auroc)
        np.testing.assert_array_equal(trace_axis.lines[0].get_xdata(), np.arange(1, 5001))
        acf_axis = figures[0].axes[column + 2]
        low, high = acf_axis.get_ylim()
        assert all(np.min(line.get_ydata()) >= low and np.max(line.get_ydata()) <= high
                   for line in acf_axis.lines[:2])


def test_ess_stability_figure_plots_each_seeds_primary_family(monkeypatch):
    import numpy as np
    import pandas as pd
    from matplotlib.figure import Figure
    from build_revision2_tables import figure_ess

    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *args, **kwargs: figures.append(self))
    figure_ess()
    assert len(figures[0].axes) == 2
    axis = figures[0].axes[1]
    assert axis.get_xscale() == "log"
    data = pd.read_csv(ROOT / "results/tables/ess_alignment_seed_sensitivity.csv")
    labels = list(pd.read_csv(ROOT / "results/tables/ess_alignment.csv").method.drop_duplicates())
    cells = [(method, edge) for method in labels for edge in ("all_edges", "strong_only")]
    assert len(axis.collections) == 5
    for seed, collection in enumerate(axis.collections):
        primary = data[(data.imputation == "ffill") & (data.null_seed == seed)].set_index(["method", "binarisation"])
        expected = [primary.loc[cell, "p_holm"] for cell in cells]
        np.testing.assert_allclose(np.asarray(collection.get_offsets()[:, 0], dtype=float), expected)


def test_temporal_delta_table_matches_committed_bytes():
    """The core temporal table must come from the CSV, not a renamed scratch file."""
    from build_revision2_tables import table_temporal_delta

    committed = (ROOT / "results" / "revision2" / "tab_temporal_delta.tex").read_text(
        encoding="utf-8"
    )
    assert table_temporal_delta() + "\n" == committed
