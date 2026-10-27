"""Display-only regressions: semantic labels may change, scientific values may not."""
from pathlib import Path
import sys

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_narrative_assets as display


def test_current_table_rewrites_preserve_every_printed_number():
    outputs, checks = display.table_outputs(ROOT)
    assert len(outputs) == len(checks) == 35
    for name, proof in checks.items():
        if name != "tab_protocol.tex":
            if "authorized_interval_overrides" not in proof:
                assert proof["printed_numeric_tokens_unchanged"]
            else:
                assert name in {"tab_fallback_full.tex", "tab_external_contrasts.tex", "tab_residual_pairs.tex"}
                assert proof["all_other_printed_numeric_tokens_unchanged"]
                restored = outputs[name].decode()
                for change in proof["authorized_interval_overrides"]:
                    lines = restored.splitlines()
                    index, = [i for i, line in enumerate(lines) if line.startswith(change["row_prefix"])]
                    cells = lines[index].split(" & ")
                    column = change["cell_column_zero_based"]
                    assert cells[column].startswith(change["new"])
                    cells[column] = cells[column].replace(change["new"], change["old"], 1)
                    lines[index] = " & ".join(cells); restored = "\n".join(lines) + "\n"
                original = (ROOT / proof["source"]).read_text(encoding="utf-8")
                assert display.numeric_tokens(restored) == display.numeric_tokens(original)
    assert not any("aux_validation_scenarios" in x for x in outputs)
    assert b"STCG-v1" not in b"".join(outputs.values())
    assert b"T-AUROC" not in b"".join(outputs.values())
    publication = b"".join(outputs.values()).lower()
    for stale in (b"post-review", b"post review", b"stcg", b"frozen"):
        assert stale not in publication


def test_metric_labels_distinguish_actual_estimands_and_scenario_mean():
    pooled = (ROOT / "results/revision2/tab_aggregation.tex").read_text(encoding="utf-8")
    within = (ROOT / "results/revision2/tab_temporal_delta.tex").read_text(encoding="utf-8")
    factorial = (ROOT / "results/revision2/tab_factorial.tex").read_text(encoding="utf-8")
    assert "Pooled AUROC" in display.transform_table("tab_aggregation.tex", pooled)
    assert "Within-edge AUROC" in display.transform_table("tab_temporal_delta.tex", within)
    changed = display.transform_table("tab_factorial.tex", factorial)
    assert "Mean across scenarios" in changed and " & Pooled " not in changed
    assert display.numeric_tokens(factorial) == display.numeric_tokens(changed)
    assert display.transform_table("display.tex", "state AUROC") == "Within-edge temporal AUROC\n"


def test_only_reviewed_unmatched_historical_runtime_row_is_removed():
    source = (ROOT / "results/revision2/tab_cost.tex").read_text(encoding="utf-8")
    changed = display.transform_table("tab_cost.tex", source)
    assert "166.9" not in changed and "Regime-PCMCI" not in changed
    assert changed.count("measured") == 4
    assert display.numeric_tokens(changed) == display.numeric_tokens(display.cost_kept(source))
    with pytest.raises(ValueError, match="reviewed form"):
        display.cost_kept(source.replace("166.9", "167.0"))


def test_formal_design_is_from_three_protocols_and_drops_earlier_exploration():
    text, record = display.protocol_table(ROOT)
    assert "Earlier" not in text and "C: 30" in text
    assert record["A"]["repeats"] == 100 and record["B"]["repeats"] == 50
    assert record["C"]["seeds"] == list(range(70000, 70030))
    assert record["C"]["M"] == [35, 23, 17, 45, 12]
    assert "Holm4" in text and "Holm6" in text and "Holm3" in text
    assert "Separate robustness study" in text


def test_figure_policy_names_depend_on_the_original_comparison():
    assert display.text_label("D = 300: hard − raw", "fig_confirmation") == "D = 300: gated hard − raw"
    assert display.text_label("Hard minus raw on the same slots", "fig_boundary_sensitivity").startswith("Forced hard")
    assert display.text_label("Native", "fig_fallback") == "Gated hard"
    assert display.text_label("State 0", "fig_target_example") == "State 0"
    assert display.text_label("Temporal graph AUROC", "rev2_eta") == "Pooled AUROC"
    assert display.text_label("one frozen primary family", "fig_confirmation") == "one prespecified primary family"


def test_duplicate_intervals_reuse_original_study_and_leave_fallback_boundary_components():
    records = display.canonical_intervals(ROOT)
    record = records["B/D300_fixed_phase/forced_hard/temporal_delta_auroc"]
    assert display.interval_text(record["canonical"]) == r"$-0.025\;[-0.054,0.004]$"
    assert display.interval_text(record["duplicate"]) == r"$-0.025\;[-0.053,0.004]$"
    assert record["paired_vector_evidence"]["exact"] and record["paired_vector_evidence"]["n_finite"] == 50
    outputs, proof = display.table_outputs(ROOT)
    fallback = outputs["tab_fallback_full.tex"].decode()
    original = display.transform_table("tab_fallback_full.tex", (ROOT / "results/information_sciences/reviewer_assets/tab_fallback_full.tex").read_text())
    for before, after in zip([x for x in original.splitlines() if " & " in x], [x for x in fallback.splitlines() if " & " in x]):
        assert before.split(" & ")[4] == after.split(" & ")[4]  # genuine raw-fallback controls
    external = outputs["tab_external_contrasts.tex"].decode()
    old_external = display.transform_table("tab_external_contrasts.tex", (ROOT / "results/information_sciences/evaluation_assets/tab_external_contrasts.tex").read_text())
    for before, after in zip(old_external.splitlines(), external.splitlines()):
        if " & Within-edge AUROC & " not in before: assert before == after
    residual = outputs["tab_residual_pairs.tex"].decode()
    assert r"C, 300 & 1 & 0.661 & 0.707" in residual and r"$+0.047\;[0.006,0.084]$" in residual
    assert r"$-0.037\;[-0.067,-0.002]$" in residual
    changes = proof["tab_residual_pairs.tex"]["authorized_interval_overrides"]
    assert all(x["reverse_comparison"] and x["cell_column_zero_based"] == 6 for x in changes)


def test_canonical_intervals_reject_different_repeat_population_or_values():
    import pandas as pd
    a = pd.Series([.1, .2, .3], index=[1, 2, 3])
    assert display.paired_vector_evidence(a, a.copy(), range(1, 4))["exact"]
    with pytest.raises(ValueError, match="population/repeat"):
        display.paired_vector_evidence(a, pd.Series(a.values, index=[2, 3, 4]), range(1, 4))
    with pytest.raises(ValueError, match="paired-value"):
        display.paired_vector_evidence(a, a + .001, range(1, 4))
    with pytest.raises(ValueError, match="finite-population"):
        display.paired_vector_evidence(a, pd.Series([.1, np.nan, .3], index=a.index), range(1, 4))
    original = display.transform_table("tab_fallback_full.tex", (ROOT / "results/information_sciences/reviewer_assets/tab_fallback_full.tex").read_text())
    with pytest.raises(ValueError, match="original value mismatch"):
        display.canonical_table_intervals("tab_fallback_full.tex", original.replace("-0.053,0.004", "-0.153,0.004"), display.canonical_intervals(ROOT))


def test_temporal_dp_display_is_bound_to_unrounded_summary_not_rounded_prose():
    text = (ROOT / "results/revision2/tab_temporal_delta.tex").read_text()
    record = display.validate_temporal_dp_display(ROOT, text)
    assert record["unrounded_mean"] == 0.5624986297566504 and record["three_decimal_display"] == "0.562"
    with pytest.raises(ValueError, match="unrounded"):
        display.validate_temporal_dp_display(ROOT, text.replace("Temporal DP & 0.765 & 0.562", "Temporal DP & 0.765 & 0.563"))


def test_numeric_artist_signature_detects_data_changes_but_not_labels():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(); line, = ax.plot([0, 1, 2], [0.5, 0.7, 0.6])
    ax.set(xlim=(-1, 3), ylim=(0, 1))
    original = display.figure_signature(fig)
    ax.set_xlabel("Within-edge AUROC"); ax.set_title("Publication label")
    assert display.figure_signature(fig) == original
    line.set_ydata([0.5, 0.8, 0.6])
    assert display.figure_signature(fig) != original
    plt.close(fig)


def test_plot_draw_cannot_restore_old_method_tick_labels():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(); ax.scatter([0, 1], [.6, .7])
    ax.set_xticks([0, 1], ["STCG-v1", "PELT + refit"])
    before = display.figure_signature(fig)
    changes = display.rewrite_figure_text(fig, "rev2_ess")
    fig.canvas.draw()
    assert ax.get_xticklabels()[0].get_text() == "Gated hard"
    assert display.figure_signature(fig) == before and changes
    plt.close(fig)


def test_pdf_date_exception_does_not_hide_scientific_byte_changes(tmp_path):
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    a.write_bytes(b"PDF /CreationDate (D:20261027010000) DATA 0.707")
    b.write_bytes(b"PDF /CreationDate (D:20261027020000) DATA 0.707")
    assert display.sha(a) != display.sha(b)
    assert display.creation_date_normalized_sha(a) == display.creation_date_normalized_sha(b)
    b.write_bytes(b"PDF /CreationDate (D:20261027020000) DATA 0.708")
    assert display.creation_date_normalized_sha(a) != display.creation_date_normalized_sha(b)


def test_rebuild_retains_the_original_full_pdf_sha_under_date_exception(tmp_path):
    path = tmp_path / display.PDF_DATE_SOURCE; path.parent.mkdir(parents=True)
    path.write_bytes(b"PDF /CreationDate (D:20261027010000) DATA 0.707")
    original = display.sha(path)
    source, exception = display.source_records(tmp_path, {display.PDF_DATE_SOURCE: original}, {})
    previous = {"source_sha256": source, "creation_date_only_source": exception}
    path.write_bytes(b"PDF /CreationDate (D:20261027020000) DATA 0.707")
    revised, revised_exception = display.source_records(tmp_path, {display.PDF_DATE_SOURCE: display.sha(path)}, previous)
    assert revised == source and revised_exception == exception
    path.write_bytes(b"PDF /CreationDate (D:20261027020000) DATA 0.708")
    with pytest.raises(ValueError, match="beyond CreationDate"):
        display.source_records(tmp_path, {display.PDF_DATE_SOURCE: display.sha(path)}, previous)
