"""Subprocess smoke test for scripts/summarize_all_recovery_methods.py.

Runs the consolidation against the tracked detail CSVs but redirects all
outputs into repo-local scratch space so the tracked tables/figures are not
touched. Mirrors the conventions of tests/test_cli_smoke.py.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_TMP = ROOT / "experiments" / "logs" / "test_summarize_all_recovery"
SCRIPT_TIMEOUT = 180


def test_summarize_all_recovery_methods_smoke():
    tables_out = TEST_TMP / "tables"
    figures_out = TEST_TMP / "figures"
    tables_out.mkdir(parents=True, exist_ok=True)
    figures_out.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_all_recovery_methods.py"),
            "--output-dir",
            str(tables_out),
            "--figures-dir",
            str(figures_out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    summary = json.loads(result.stdout)
    assert summary["problems"] == []

    summary_csv = tables_out / "synthetic_all_methods_summary.csv"
    summary_md = tables_out / "synthetic_all_methods_summary.md"
    summary_tex = tables_out / "synthetic_all_methods_table.tex"
    matrix_csv = tables_out / "synthetic_switching_paired_matrix.csv"
    matrix_md = tables_out / "synthetic_switching_paired_matrix.md"
    figure = figures_out / "synthetic_all_methods_edge_auc.png"
    for artifact in (summary_csv, summary_md, summary_tex, matrix_csv, matrix_md, figure):
        assert artifact.is_file(), f"missing {artifact}"

    rows = list(csv.DictReader(summary_csv.open()))
    assert len(rows) == 48  # 3 generators x 16 methods
    lookup = {(row["generator"], row["method"]): row for row in rows}
    stcg = lookup[("switching_var", "STCG-v1")]
    assert int(stcg["n"]) == 20
    assert abs(float(stcg["edge_auc_mean"]) - 0.8124606203901402) < 1e-12
    # Regime-PCMCI nonlinear_var is the known n=19 exception (seed 18 failed).
    assert int(lookup[("nonlinear_var", "Regime-PCMCI")]["n"]) == 19

    matrix_rows = list(csv.DictReader(matrix_csv.open()))
    assert len(matrix_rows) == 75  # 15 opponents x 5 metrics
    assert {row["generator"] for row in matrix_rows} == {"switching_var"}
    assert {row["method_a"] for row in matrix_rows} == {"STCG-v1"}

    tex_text = summary_tex.read_text(encoding="utf-8")
    assert "\\textbf{" in tex_text  # bold-best per generator column block
    md_text = summary_md.read_text(encoding="utf-8")
    assert "N=19" in md_text  # rpcmci footnote
