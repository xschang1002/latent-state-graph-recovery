"""Publication-only labels/layout; immutable original artifacts remain auditable.

Only the current manuscript assets are copied/re-rendered. No scorer, generator,
experiment runner or old asset builder entry point is executed. --verify builds
tables and figure data signatures in memory and never writes display artifacts.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import importlib
import importlib.metadata
import io
import json
from pathlib import Path
import re
import sys
import platform
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/information_sciences/narrative_assets"
BASE = "results/information_sciences/"
TABLE_GROUPS = {
    "results/revision2/": "controls aggregation temporal_delta metrics_layers temporal_grid grouping_factors factorial segmentation final cost ess known_k penalty_scan matched_grouping temporal_grid_accepted temporal_gmm_numerics temporal_coverage ess_diagnostics".split(),
    BASE: "protocol main_results factors supp_a_contrasts supp_b_gate_factors supp_diagnostics_ols".split(),
    BASE + "generator_audit/": ["generator_diagnostics"],
    BASE + "reviewer_assets/": "score_controls score_controls_aux fallback_full localisation_full stable_changes symmetric_full".split(),
    BASE + "evaluation_assets/": "external_framework ap_sensitivity residual_pairs external_contrasts".split(),
}
TABLES = tuple(prefix + "tab_" + name + ".tex" for prefix, names in TABLE_GROUPS.items() for name in names)
OLD_FIGURES = {"rev2_" + name: "figure_" + name for name in
               ("mechanism", "controls", "aggregation", "eta", "temporal_grid", "segmentation", "ess", "ess_diagnostics")}
FIGURE_SOURCES = {name: "results/figures/" + name + (".pdf" if name == "rev2_temporal_grid" else ".png") for name in OLD_FIGURES}
FIGURE_SOURCES.update({"fig_" + n: BASE + "fig_" + n + ".pdf" for n in ("target_example", "confirmation", "schedule")})
FIGURE_SOURCES.update({"fig_" + n: BASE + "reviewer_assets/fig_" + n + ".pdf" for n in ("fallback", "symmetric")})
FIGURE_SOURCES.update({"fig_" + n: BASE + "evaluation_assets/fig_" + n + ".pdf" for n in
                       ("boundary_sensitivity", "threshold_curves", "residual_pairs")})
BUILDERS = ("tools/build_revision2_tables.py", "tools/build_information_sciences_assets.py",
            "tools/build_reviewer_controls_assets.py", "tools/build_evaluation_sensitivity_assets.py")
PROTOCOLS = (BASE + "duration_confirmation/protocol.json", BASE + "schedule_robustness/schedule_robustness_protocol.json",
             BASE + "symmetric_sensitivity/symmetric_sensitivity_protocol.json")
GLOSSARY = {"Gated hard": "native_hard / STCG-v1: gate-returned hard maps; rejection emits the constant window-maximum map",
            "Forced hard": "forced_hard: candidate hard maps regardless of gate decision",
            "Raw fallback": "same gate and accepted candidate as native_hard; rejection emits raw window scores",
            "Pooled AUROC": "AUROC pooled over nonself edge/window entries, formerly temporal_graph_auroc / T-AUROC",
            "Within-edge AUROC": "mean active/inactive-window AUROC on evaluable changing nonself edges, formerly temporal_delta_auroc / T-AUC_delta / State AUROC",
            "Assignment ARI": "state-assignment adjusted Rand index; not within-edge AUROC"}
PDF_DATE_SOURCE = "results/figures/rev2_temporal_grid.pdf"
CANONICAL_CI_SOURCES = tuple(BASE + p for p in (
    "schedule_robustness/schedule_robustness_paired.csv",
    "schedule_robustness/schedule_robustness_secondary_tests.csv", "supp_b_gate_source.csv",
    "score_preserving_controls/detail.csv", "score_preserving_controls/contrasts.csv",
    "symmetric_sensitivity/symmetric_sensitivity_detail.csv",
    "symmetric_sensitivity/symmetric_sensitivity_summary.csv",
    "segmentation_validation/detail.csv", "segmentation_validation/contrasts.csv",
    "boundary_sensitivity/residual_detail.csv", "boundary_sensitivity/residual_summary.csv"))
TEMPORAL_DP_SOURCE = "results/revision2/temporal_delta_summary.csv"
SCHEDULES = ("fixed_phase", "random_phase", "irregular")
SCHEDULE_LABELS = {"fixed_phase": "Fixed", "random_phase": "Random", "irregular": "Irregular"}
DERIVED_TABLES = ("tab_study_a_combined.tex", "tab_ap_sensitivity_ap.tex", "tab_ap_sensitivity_f1.tex")
STUDY_A_DETAIL = BASE + "duration_confirmation/detail.csv"


def one(frame, **where):
    selected = frame
    for key, value in where.items(): selected = selected[selected[key] == value]
    if len(selected) != 1: raise ValueError("Canonical display needs exactly one record: " + str(where))
    return selected.iloc[0]


def paired_vector_evidence(canonical, duplicate, seeds):
    """No CI is transferable unless all paired repeat identities and values match."""
    canonical, duplicate = canonical.sort_index(), duplicate.sort_index()
    if (canonical.index.has_duplicates or duplicate.index.has_duplicates
            or canonical.index.tolist() != list(seeds) or not canonical.index.equals(duplicate.index)):
        raise ValueError("Canonical CI population/repeat identity mismatch")
    a, b = canonical.to_numpy(float), duplicate.to_numpy(float)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)) or not np.allclose(a, b, atol=1e-14, rtol=0, equal_nan=True):
        raise ValueError("Canonical CI paired-value or finite-population mismatch")
    usable = np.isfinite(a)
    return {"n_planned": len(a), "n_finite": int(usable.sum()), "repeat_seeds": list(seeds),
            "exact": bool(np.array_equal(a, b, equal_nan=True)),
            "max_abs_difference": float(np.max(np.abs(a[usable] - b[usable]))) if usable.any() else 0.,
            "absolute_tolerance": 1e-14, "relative_tolerance": 0.}


def ci_record(row, mean="mean"):
    return {"mean": float(row[mean]), "ci95_low": float(row.ci95_low), "ci95_high": float(row.ci95_high)}


def interval_text(record):
    return "$" + f"{record['mean']:+.3f}" + r"\;[" + f"{record['ci95_low']:.3f},{record['ci95_high']:.3f}" + "]$"


def canonical_intervals(root):
    """Reuse original-study intervals, without drawing a new bootstrap sample.

    Only duplicated B native/forced AUROC and central C forced AUROC are covered.
    Later fallback, AP/F1, subgroup and pair-component intervals keep their own
    fixed analysis streams and inferential identities.
    """
    data = {Path(p).name: pd.read_csv(root / p, float_precision="round_trip") for p in CANONICAL_CI_SOURCES}
    # Two different studies use detail/contrasts names; address them explicitly.
    get = lambda p: pd.read_csv(root / BASE / p, float_precision="round_trip")
    bp, bt, bg = (data[n] for n in ("schedule_robustness_paired.csv", "schedule_robustness_secondary_tests.csv", "supp_b_gate_source.csv"))
    control, contrasts = get("score_preserving_controls/detail.csv"), get("score_preserving_controls/contrasts.csv")
    cd, cs = data["symmetric_sensitivity_detail.csv"], data["symmetric_sensitivity_summary.csv"]
    segment, segment_contrasts = get("segmentation_validation/detail.csv"), get("segmentation_validation/contrasts.csv")
    residual, residual_summary = data["residual_detail.csv"], data["residual_summary.csv"]
    records = {}
    for duration in (300, 600):
        for schedule in SCHEDULES:
            cell = f"D{duration}_{schedule}"
            original = bp[(bp.duration_mean == duration) & (bp.schedule == schedule)].set_index("seed")
            for method, prefix in (("native_hard", "native"), ("forced_hard", "forced")):
                duplicate = control[(control.study == "B") & (control.scorer == "lasso") &
                                    (control.cell == cell) & (control.method.isin([method, "raw_window"]))]
                wide = duplicate.pivot(index="repeat_seed", columns="method", values="temporal_delta_auroc")
                evidence = paired_vector_evidence(original[method] - original.raw_window,
                                                  wide[method] - wide.raw_window, range(60000, 60050))
                old = one(contrasts, study="B", scorer="lasso", cell=cell, method=method,
                          reference="raw_window", metric="temporal_delta_auroc")
                if method == "forced_hard":
                    source = one(bt, family="B_forced_minus_raw", duration_mean=duration, schedule=schedule)
                    canonical = ci_record(source, "mean_difference")
                    source_path = CANONICAL_CI_SOURCES[1]
                else:
                    source = one(bg, duration_mean=duration, schedule=schedule)
                    canonical = {"mean": float(source.native_minus_raw), "ci95_low": float(source.native_ci95_low),
                                 "ci95_high": float(source.native_ci95_high)}
                    source_path = BASE + "supp_b_gate_source.csv"
                if not np.isclose(canonical["mean"], old["mean"], atol=1e-14, rtol=0):
                    raise ValueError("Canonical B mean mismatch")
                records[f"B/{cell}/{method}/temporal_delta_auroc"] = {
                    "study": "B", "scorer": "lasso", "cell": cell, "duration": duration, "schedule": schedule,
                    "method": method, "reference": "raw_window", "metric": "temporal_delta_auroc",
                    "canonical_source": source_path, "canonical": canonical, "duplicate_source": BASE + "score_preserving_controls/contrasts.csv",
                    "duplicate": ci_record(old), "paired_vector_evidence": evidence,
                    "canonical_bootstrap": "10000 paired-repeat draws; seed20261024, original Study B stream",
                    "duplicate_bootstrap": "10000 paired-repeat draws; seed20261025, later controls stream",
                    "inferential_change": "none; no p value, family or point estimate is reassigned"}
        cell = f"D{duration}_h300_s150"
        original = cd[(cd.cell == cell) & (cd.method.isin(["forced_hard", "raw_window"]))]
        reference = segment[(segment.duration == duration) & (segment.method.isin(["forced_hard", "raw_window"]))]
        for metric in ("temporal_delta_auroc",):
            ow = original.pivot(index="seed", columns="method", values=metric)
            dw = reference.pivot(index="repeat_seed", columns="method", values=metric)
            vector = ow.forced_hard - ow.raw_window
            evidence = paired_vector_evidence(vector, dw.forced_hard - dw.raw_window, range(70000, 70030))
            canonical = ci_record(one(cs, cell=cell, method="forced_hard", metric=metric, comparison="minus_raw"))
            old = ci_record(one(segment_contrasts, duration=duration, method="forced_hard", reference="raw_window", metric=metric))
            if not np.isclose(canonical["mean"], old["mean"], atol=1e-14, rtol=0): raise ValueError("Canonical C mean mismatch")
            record = {"study": "C", "scorer": "lasso", "cell": cell, "duration": duration,
                      "method": "forced_hard", "reference": "raw_window", "metric": metric,
                      "canonical_source": BASE + "symmetric_sensitivity/symmetric_sensitivity_summary.csv", "canonical": canonical,
                      "duplicate_source": BASE + "segmentation_validation/contrasts.csv", "duplicate": old,
                      "paired_vector_evidence": evidence,
                      "canonical_bootstrap": "10000 paired-repeat draws; reset seed20261025, original Study C stream",
                      "duplicate_bootstrap": "10000 paired-repeat draws; identity-addressed seed20261025, adapter analysis stream",
                      "inferential_change": "none; no p value, family or point estimate is reassigned"}
            if metric == "temporal_delta_auroc":
                reverse = residual[(residual.study == "C") & (residual.scorer == "lasso") & (residual.cell == cell) & (residual["lambda"] == 1.)].set_index("repeat_seed")
                record["reverse_vector_evidence"] = paired_vector_evidence(-vector, reverse.total_auc_delta, range(70000, 70030))
                record["reverse_duplicate_source"] = BASE + "boundary_sensitivity/residual_summary.csv"
                record["reverse_duplicate"] = ci_record(one(residual_summary, study="C", scorer="lasso", cell=cell,
                    method="residual_1", metric="total_auc_delta", **{"lambda": 1.}))
                record["reverse_canonical"] = {"mean": -canonical["mean"], "ci95_low": -canonical["ci95_high"], "ci95_high": -canonical["ci95_low"]}
            records[f"C/{cell}/forced_hard/{metric}"] = record
    return records


def validate_temporal_dp_display(root, text):
    source = pd.read_csv(root / TEMPORAL_DP_SOURCE, float_precision="round_trip")
    value = float(one(source, method="temporal_dp").temporal_delta_auroc)
    rows = [line for line in text.splitlines() if line.startswith("Temporal DP & ")]
    if len(rows) != 1 or rows[0].split(" & ")[-1].replace(r"\\", "").strip() != f"{value:.3f}":
        raise ValueError("Temporal DP publication display differs from unrounded corresponding-study summary")
    return {"source": TEMPORAL_DP_SOURCE, "unrounded_mean": value, "three_decimal_display": f"{value:.3f}", "n": 20}


def canonical_table_intervals(name, text, records):
    """Replace only the authorized interval cells; reject any unexpected table."""
    changes = []
    def replace_cell(prefix, column, record, reverse=False):
        nonlocal text
        old = interval_text(record["reverse_duplicate" if reverse else "duplicate"])
        new = interval_text(record["reverse_canonical" if reverse else "canonical"])
        if old.split(r"\;[")[0] != new.split(r"\;[")[0]:
            raise ValueError("Canonical CI cannot change a displayed point estimate")
        lines = text.splitlines(); matches = [i for i, line in enumerate(lines) if line.startswith(prefix)]
        if len(matches) != 1: raise ValueError("Canonical interval display row mismatch: " + prefix)
        index = matches[0]; cells = lines[index].split(" & ")
        actual = cells[column].strip()
        tail = r" \\" if actual.endswith(r"\\") else ""
        if tail: actual = actual[:-2].rstrip()
        if actual != old: raise ValueError("Canonical interval display original value mismatch: " + prefix)
        cells[column] = new + tail; lines[index] = " & ".join(cells); text = "\n".join(lines) + "\n"
        changes.append({"comparison": {k: record[k] for k in ("study", "scorer", "cell", "method", "reference", "metric")},
                        "reverse_comparison": reverse, "old": old, "new": new, "changed": old != new,
                        "canonical_source": record["canonical_source"], "cell_column_zero_based": column, "row_prefix": prefix})
    if name == "tab_fallback_full.tex":
        for record in records.values():
            if record["study"] == "B":
                prefix = f"{record['duration']} & {SCHEDULE_LABELS[record['schedule']]} & "
                replace_cell(prefix, 3 if record["method"] == "native_hard" else 5, record)
    elif name == "tab_external_contrasts.tex":
        for record in records.values():
            if record["study"] == "C":
                target = "Within-edge AUROC" if record["metric"] == "temporal_delta_auroc" else "Pooled AUROC"
                replace_cell(f"{record['duration']} & {target} & ", 2, record)
    elif name == "tab_residual_pairs.tex":
        for record in records.values():
            if record["study"] == "C" and record["metric"] == "temporal_delta_auroc":
                replace_cell(f"C, {record['duration']} & 1 & ", 6, record, reverse=True)
    return text, changes


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def creation_date_normalized_sha(path):
    data, count = re.subn(rb"/CreationDate\s*\([^)]*\)", b"/CreationDate ()", Path(path).read_bytes())
    if count != 1:
        raise ValueError("The sole reviewed PDF CreationDate exception changed form")
    return hashlib.sha256(data).hexdigest()


def source_records(root, actual, previous):
    """Retain the original full SHA, with one explicit metadata-only exception."""
    sources = dict(actual)
    normalized = creation_date_normalized_sha(root / PDF_DATE_SOURCE)
    prior = previous.get("creation_date_only_source", {})
    if prior:
        if normalized != prior["normalized_sha256"]:
            raise ValueError("Original temporal-grid PDF changed beyond CreationDate")
        sources[PDF_DATE_SOURCE] = previous["source_sha256"][PDF_DATE_SOURCE]
    return sources, {"path": PDF_DATE_SOURCE, "normalized_sha256": normalized,
                     "rule": "Only /CreationDate(...); every other PDF byte must remain identical; original full source SHA retained"}


def numeric_tokens(text):
    """Include every printed number; remove only comments and the old code ID."""
    text = re.sub(r"(?m)^\s*%.*$", "", text).replace("STCG-v1e", "").replace("STCG-v1", "")
    return re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)


def cost_kept(text):
    lines = text.splitlines(); selected = [i for i, line in enumerate(lines) if "Regime-PCMCI" in line]
    if len(selected) != 1 or "166.9" not in lines[selected[0]] or "historical, different grid" not in lines[selected[0]]:
        raise ValueError("Historical runtime exception no longer has its reviewed form")
    index = selected[0]; remove = {index}
    if index and lines[index - 1] == r"\midrule":
        remove.add(index - 1)
    return "\n".join(line for i, line in enumerate(lines) if i not in remove) + "\n"


def transform_table(source, text):
    name = Path(source).name
    if name == "tab_cost.tex":
        text = cost_kept(text)
    text = text.replace("STCG-v1", "Gated hard")
    text = re.sub(r"\\stcg\b(?:\{\})?", "Gated hard", text)
    text = text.replace("Native hard", "Gated hard").replace("native=forced", "gated=forced")
    text = text.replace("Hard (gated=forced)", "Gated = forced hard")
    text = re.sub(r"\bNative\b", "Gated", text)
    text = text.replace("Temporal graph AUROC", "Pooled AUROC").replace("pooled temporal AUROC", "Pooled AUROC")
    text = text.replace("T-AUROC", "Pooled AUROC").replace(r"$\mathrm{T\!-\!AUC}_{\Delta}$", "Within-edge AUROC")
    text = text.replace("Changing-edge", "Within-edge AUROC").replace("changing-edge temporal AUROC", "within-edge AUROC")
    text = text.replace("State ARI", "Assignment ARI")
    text = text.replace("temporal state AUROC", "Within-edge temporal AUROC").replace("state AUROC", "Within-edge temporal AUROC").replace("State AUROC", "Within-edge temporal AUROC")
    text = text.replace("K-means hard", "Gated hard")
    if name == "tab_factorial.tex":
        text = text.replace(" & Pooled ", " & Mean across scenarios ")
        text = text.replace("K-means &", "Returned hard groups &").replace("PELT &", "PELT (earlier grid) &")
    if name in {"tab_segmentation.tex", "tab_cost.tex", "tab_final.tex"}:
        text = text.replace("PELT BIC", "PELT (earlier grid)").replace("PELT+refit", "PELT refit (earlier)")
    if name in {"tab_score_controls.tex", "tab_score_controls_aux.tex", "tab_external_framework.tex", "tab_residual_pairs.tex"}:
        text = re.sub(r"(?m)^Hard &", "Forced hard &", text).replace(" & Hard &", " & Forced hard &")
    if name in {"tab_score_controls.tex", "tab_score_controls_aux.tex"}:
        text = text.replace("State,", "Within-edge,").replace("Pooled,", "Pooled AUROC,")
    if name == "tab_main_results.tex":
        text = text.replace("Method & Pooled &", "Method & Pooled AUROC &").replace(" & Pooled &", " & Pooled AUROC &")
    if name == "tab_external_framework.tex":
        text = text.replace(" & Pooled & State &", " & Pooled AUROC & Within-edge &")
    if name == "tab_external_contrasts.tex":
        text = text.replace(" & Pooled &", " & Pooled AUROC &").replace(" & State &", " & Within-edge AUROC &")
    if name in {"tab_temporal_grid.tex", "tab_temporal_grid_accepted.tex"}:
        text = text.replace("Hard $\\Delta$", "Gated $\\Delta$")
    if name == "tab_matched_grouping.tex":
        text = text.replace("Hard $-$", "Gated hard $-$")
    text = text.replace("Hard--raw", "Forced--raw").replace("Hard minus raw", "Forced hard minus raw")
    text = text.replace("Raw-fallback", "Raw fallback")
    if name == "tab_stable_changes.tex":
        text = text.replace(" & hard & fallback", " & forced & raw fallback")
    if name == "tab_penalty_scan.tex":
        text = text.replace("Min segment", "Min windows/segment")
    return text.rstrip() + "\n"


def data_cells(text):
    """Return intact printed cells; no float parsing or rounding is involved."""
    return [line.rstrip().removesuffix(r"\\").rstrip().split(" & ")
            for line in text.splitlines() if " & " in line and line.rstrip().endswith(r"\\")]


def compact_table_layout(name, text):
    """Wrap labels/intervals without changing their numeric token sequence."""
    before = numeric_tokens(text)
    if name == "tab_external_framework.tex":
        text = text.replace("Pooled AUROC & Within-edge &", r"\shortstack{Pooled\\AUROC} & \shortstack{Within-edge\\AUROC} &")
    elif name == "tab_residual_pairs.tex":
        text = text.replace("Forced hard & Residual & Within $\\Delta$ & Cross $\\Delta$ & Total $\\Delta$",
            r"\shortstack{Forced\\hard} & Residual & \shortstack{Within-group\\$\Delta$} & \shortstack{Cross-group\\$\Delta$} & \shortstack{Total\\$\Delta$}")
        text = re.sub(r"\$([-+]?[0-9.]+)\\;\[([^\]]+)\]\$",
            lambda m: r"\shortstack[r]{$" + m[1] + r"$\\$[" + m[2] + r"]$}", text)
    elif name == "tab_controls.tex":
        for old, new in (("structure change", r"\shortstack[l]{structure\\change}"),
                         ("mechanism change", r"\shortstack[l]{mechanism\\change}"),
                         ("nuisance change", r"\shortstack[l]{nuisance\\change}"),
                         ("Switching (reference)", r"\shortstack[l]{Switching\\(reference)}"),
                         ("Lag AUROC", r"\shortstack{Lag\\AUROC}"),
                         ("Pair AUROC", r"\shortstack{Pair\\AUROC}"),
                         ("Regime AUPRC", r"\shortstack{Regime\\AUPRC}"),
                         ("Pooled AUROC", r"\shortstack{Pooled\\AUROC}"),
                         ("Assignment ARI", r"\shortstack{Assignment\\ARI}")):
            text = text.replace(old, new)
    if numeric_tokens(text) != before:
        raise AssertionError("Table wrapping changed printed numeric tokens: " + name)
    return text


def combined_study_a_table(root):
    references = BASE + "tab_main_results.tex"
    controls = BASE + "reviewer_assets/tab_score_controls.tex"
    ref = {row[0]: row[1:] for row in data_cells((root / references).read_text(encoding="utf-8"))[2:]}
    ctl = {row[0]: row[1:] for row in data_cells((root / controls).read_text(encoding="utf-8"))[1:]}
    expected_ref = {"Raw windows", "Static mean", "Native hard", "Forced hard", "Matched random", "Oracle labels"}
    expected_ctl = {"Raw", "Hard", "Shrink 0.25", "Shrink 0.5", "Shrink 0.75", "Residual 0.25", "Residual 0.5", "Residual 0.75"}
    if set(ref) != expected_ref or set(ctl) != expected_ctl:
        raise ValueError("Combined Study A input row inventory differs")
    if (ref["Native hard"] != ref["Forced hard"] or ref["Raw windows"] != ctl["Raw"]
            or ref["Forced hard"] != ctl["Hard"]):
        raise ValueError("Combined Study A de-duplication requires identical reference cells")
    detail = pd.read_csv(root / STUDY_A_DETAIL, float_precision="round_trip")
    main = detail[detail.scorer == "lasso"]
    if set(main.D) != {300, 600}:
        raise ValueError("Combined Study A duration population differs")
    for duration in (300, 600):
        selected = main[(main.D == duration) & (main.method == "native_hard")]
        if len(selected) != 100 or selected.repeat_seed.nunique() != 100 or not selected.hard_gate_accepted.all():
            raise ValueError("Combined hard row is valid only when all Study A gates accept")
        paired = main[(main.D == duration) & main.method.isin(["native_hard", "forced_hard"])].pivot(
            index="repeat_seed", columns="method", values=["pooled_temporal_auroc", "temporal_delta_auroc"])
        for metric in ("pooled_temporal_auroc", "temporal_delta_auroc"):
            if not np.array_equal(paired[metric].native_hard, paired[metric].forced_hard, equal_nan=True):
                raise ValueError("Combined hard row requires identical unrounded Study A metrics")
    rows = [r"\begin{tabular}{lrrrr}", r"\toprule",
        r" & \multicolumn{2}{c}{$D=300$} & \multicolumn{2}{c}{$D=600$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Emission & Pooled & Within-edge & Pooled & Within-edge \\", r"\midrule",
        r"\multicolumn{5}{l}{\textit{(a) Reference emissions}} \\"]
    evidence = []
    for source_name, label in (("Raw windows", "Raw windows"), ("Static mean", "Static mean"),
            ("Native hard", "Hard (gated = forced)"), ("Matched random", "Size-matched random"),
            ("Oracle labels", "Oracle labels")):
        rows.append(" & ".join([label, *ref[source_name]]) + r" \\")
        evidence.append({"source": references, "source_row": source_name, "display_row": label, "cells": ref[source_name]})
    rows.extend([r"\midrule", r"\multicolumn{5}{l}{\textit{(b) Additional score-emission controls}} \\"])
    for family in ("Shrink", "Residual"):
        for weight in ("0.25", "0.5", "0.75"):
            source_name = family + " " + weight
            label = ("Shrinkage" if family == "Shrink" else family) + r", $\lambda=" + weight + "$"
            rows.append(" & ".join([label, *ctl[source_name]]) + r" \\")
            evidence.append({"source": controls, "source_row": source_name, "display_row": label, "cells": ctl[source_name]})
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(rows) + "\n", {"derived_layout_only": True, "numeric_cells_unchanged": True,
        "sources": {p: sha(root / p) for p in (references, controls, STUDY_A_DETAIL)}, "rows": evidence,
        "merged_reference_rows": {"Native hard": "Forced hard", "Raw": "Raw windows", "Hard": "Forced hard"},
        "gate_acceptance": {"300": "100/100", "600": "100/100"}, "primary_family_unchanged": True}


def split_ap_sensitivity(root):
    source = BASE + "evaluation_assets/tab_ap_sensitivity.tex"
    original = data_cells((root / source).read_text(encoding="utf-8"))
    rows = original[1:]
    if len(rows) != 18 or any(len(row) != 7 for row in rows) or len({row[0] for row in rows}) != 18:
        raise ValueError("AP/F1 split must preserve eighteen distinct seven-cell source rows")
    outputs, proof = {}, {}
    for suffix, columns, header in (("ap", (0, 1, 2, 3, 4),
            r"Condition & $n_{\rm AP}$ & \shortstack{Raw\\AP$_{\rm std}$} & $\Delta$AP$_+$ & $\Delta$AP$_{\rm std}$ \\") ,
            ("f1", (0, 5, 6), r"Condition & $\Delta$F1$_{\rm raw}$ & $\Delta$F1$_{\rm self}$ \\")):
        projected = [[row[c] for c in columns] for row in rows]
        text = "\n".join([r"\begin{tabular}{l" + "r" * (len(columns) - 1) + "}", r"\toprule", header,
            r"\midrule", *[" & ".join(row) + r" \\" for row in projected], r"\bottomrule", r"\end{tabular}"]) + "\n"
        name = "tab_ap_sensitivity_" + suffix + ".tex"
        outputs[name] = text.encode()
        proof[name] = {"derived_layout_only": True, "numeric_cells_unchanged": True,
            "sources": {source: sha(root / source)}, "source_columns_zero_based": list(columns),
            "rows": projected, "source_row_count": len(rows), "missing_cells_preserved": True}
    return outputs, proof


def protocol_table(root):
    a, b, c = [read(root / n) for n in PROTOCOLS]
    bc, cc = b["configuration"], c["configuration"]
    if (a["repeat_seeds"] != list(range(50000, 50100)) or b["repeats"] != list(range(60000, 60050))
            or c["repeats"] != list(range(70000, 70030)) or a["durations"] != [300, 600]
            or bc["durations"] != [300, 600] or cc["durations"] != [300, 600]):
        raise ValueError("Formal A/B/C protocol identities differ")
    windows = cc["windows"]; counts = [1 + (cc["mechanism"]["timesteps"] - w["size"]) // w["stride"] for w in windows]
    if a["base_config"]["timesteps"] != 1200 or bc["mechanism"]["timesteps"] != 3600 or counts != [35, 23, 17, 45, 12]:
        raise ValueError("Formal sample/window counts differ")
    if (a["score_pipeline"]["window_size"], a["score_pipeline"]["stride"]) != (300, 150) or bc["window"] != {"size": 300, "stride": 150}:
        raise ValueError("Formal A/B window specification differs")
    if (a["primary_family"]["hard"] != "native_hard" or len(a["primary_family"]["contrasts"]) != 4
            or "Holm6" not in b["inference"]["B_forced_family"] or "Holm3" not in b["inference"]["B_duration_difference_family"]):
        raise ValueError("Formal inferential identities differ")
    def row(*cells):
        # End each paragraph inside its group; the tabular row separator keeps
        # its ordinary meaning without array.sty or an arraybackslash macro.
        return " & ".join(value if index == 3 else r"{\raggedright " + value + r"\par}"
                          for index, value in enumerate(cells)) + r" \\"
    rows = [r"\begingroup", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{@{}p{0.055\linewidth}p{0.045\linewidth}p{0.24\linewidth}rp{0.19\linewidth}p{0.27\linewidth}@{}}",
        r"\toprule", row("Study", "$n$", "Schedule / generator", "$T$", "Windows: $h/s$; $M$", "Statistical role"), r"\midrule",
        row("A", "100", "Asymmetric support; fixed phase; $D=300,600$", "1200", "$300/150$; $7$", "Primary confirmation: gated Lasso Holm4; forced and OLS auxiliary"),
        row("B", "50", r"Asymmetric support; fixed/random/irregular; $\bar D=300,600$", "3600", "$300/150$; $23$", "Secondary robustness: forced--raw Holm6; duration Holm3"),
        row("C", "30", "Symmetric generator; fixed phase; $D=300,600$", "3600", "Five specified pairs; $M=12$--$45$; see S4.2", r"Separate robustness study; pointwise 95\% intervals; no confirmatory family"),
        r"\bottomrule", r"\end{tabular}", r"\endgroup"]
    evidence = {"A": {"repeats": len(a["repeat_seeds"]), "seeds": a["repeat_seeds"], "T": a["base_config"]["timesteps"], "protocol": PROTOCOLS[0]},
                "B": {"repeats": len(b["repeats"]), "seeds": b["repeats"], "T": bc["mechanism"]["timesteps"], "protocol": PROTOCOLS[1]},
                "C": {"repeats": len(c["repeats"]), "seeds": c["repeats"], "T": cc["mechanism"]["timesteps"], "windows": windows, "M": counts, "protocol": PROTOCOLS[2]}}
    return "\n".join(rows) + "\n", evidence


def table_outputs(root):
    outputs, proof = {}, {}
    canonical = canonical_intervals(root)
    for source in TABLES:
        original = (root / source).read_text(encoding="utf-8"); name = Path(source).name
        if name == "tab_protocol.tex":
            display, evidence = protocol_table(root)
            check = {"exception": "Replace earlier exploratory row with the unchanged formal A/B/C designs; no result estimate changed", "protocol_evidence": evidence}
        else:
            display = transform_table(source, original)
            compared = cost_kept(original) if name == "tab_cost.tex" else original
            if numeric_tokens(compared) != numeric_tokens(display):
                raise AssertionError("Publication label rewrite changed printed numbers: " + source)
            check = {"printed_numeric_tokens_unchanged": True, "numeric_token_count": len(numeric_tokens(compared))}
            if name == "tab_cost.tex":
                check["exception"] = "Move only historical Regime-PCMCI166.9 row out of publication display; original table retained"
            display, overrides = canonical_table_intervals(name, display, canonical)
            if overrides:
                check["printed_numeric_tokens_unchanged"] = numeric_tokens(compared) == numeric_tokens(display)
                check["authorized_interval_overrides"] = overrides
                check["all_other_printed_numeric_tokens_unchanged"] = True
            if name == "tab_temporal_delta.tex":
                check["temporal_dp_unrounded_binding"] = validate_temporal_dp_display(root, display)
            display = compact_table_layout(name, display)
        outputs[name] = display.encode(); proof[name] = {"source": source, "source_sha256": sha(root / source), **check}
    combined, combined_proof = combined_study_a_table(root)
    outputs[DERIVED_TABLES[0]] = combined.encode(); proof[DERIVED_TABLES[0]] = combined_proof
    split, split_proof = split_ap_sensitivity(root)
    outputs.update(split); proof.update(split_proof)
    return outputs, proof


def text_label(text, figure):
    text = text.replace("STCG-v1", "Gated hard").replace("K-means hard", "Gated hard")
    text = text.replace("Temporal graph AUROC", "Pooled AUROC")
    text = text.replace("changing-edge temporal AUROC", "within-edge AUROC").replace("Changing\nedge", "Within-\nedge")
    text = text.replace("temporal state-discrimination AUROC", "within-edge AUROC").replace("state AUROC", "within-edge AUROC")
    text = text.replace("Temporal AUROC difference", "Within-edge AUROC difference")
    text = text.replace("frozen primary family", "prespecified primary family")
    if figure == "fig_confirmation":
        text = text.replace("hard −", "gated hard −")
    elif figure == "fig_fallback":
        text = {"Native": "Gated hard", "Hard": "Forced hard", "Raw fallback": "Raw fallback"}.get(text, text)
    elif figure in {"fig_threshold_curves", "fig_boundary_sensitivity", "fig_residual_pairs"}:
        text = text.replace("Hard minus raw", "Forced hard minus raw")
        if text == "Hard":
            text = "Forced hard"
    elif figure == "rev2_temporal_grid":
        text = text.replace("Hard, native gate", "Gated hard").replace("Hard, forced", "Forced hard")
    return text


def array_signature(value):
    a = np.asarray(value)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}


def figure_signature(fig):
    result = []
    for ax in fig.axes:
        result.append({"xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()), "xscale": ax.get_xscale(), "yscale": ax.get_yscale(),
            "lines": [array_signature(line.get_xydata()) for line in ax.lines],
            "collections": [{"offsets": array_signature(c.get_offsets()), "paths": [array_signature(p.vertices) for p in c.get_paths()],
                             "array": array_signature(c.get_array()) if c.get_array() is not None else None} for c in ax.collections],
            "patches": [array_signature(p.get_path().vertices) for p in ax.patches],
            "images": [array_signature(im.get_array()) for im in ax.images]})
    return hashlib.sha256(json_bytes(result)).hexdigest()


def rewrite_figure_text(fig, name):
    """Update tick formatters as well as Text; a later draw must not restore IDs."""
    from matplotlib.text import Text
    from matplotlib.ticker import FixedFormatter
    changes = []
    for ax in fig.axes:
        for axis, labels in ((ax.xaxis, ax.get_xticklabels()), (ax.yaxis, ax.get_yticklabels())):
            original = [item.get_text() for item in labels]
            revised = [text_label(value, name) for value in original]
            if revised != original:
                axis.set_major_formatter(FixedFormatter(revised))
                for item, old_text, new_text in zip(labels, original, revised):
                    item.set_text(new_text)
                    if old_text != new_text: changes.append({"before": old_text, "after": new_text})
    for item in fig.findobj(Text):
        old_text = item.get_text(); new_text = text_label(old_text, name)
        if new_text != old_text:
            item.set_text(new_text); changes.append({"before": old_text, "after": new_text})
    return changes


def layout_figure(fig, name):
    """Keep all coordinates/limits fixed while moving the B-policy legend."""
    if name != "fig_fallback":
        return {}
    if len(fig.axes) != 1:
        raise ValueError("Study B policy figure axis inventory changed")
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    labels = [text_label(label, name) for label in labels]
    if labels != ["Gated hard", "Forced hard", "Raw fallback"]:
        raise ValueError("Study B policy legend inventory changed")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.18), frameon=False, ncols=3)
    return {"legend_location": "outside below axis", "legend_labels": labels,
            "axes_limits_and_data_unchanged": True}


def figure_outputs(root, output, write):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    sys.path[:0] = [str(root / "tools"), str(root / "scripts")]
    old = importlib.import_module("build_revision2_tables")
    main = importlib.import_module("build_information_sciences_assets")
    reviewer = importlib.import_module("build_reviewer_controls_assets")
    evaluation = importlib.import_module("build_evaluation_sensitivity_assets")
    original_save, original_csv = Figure.savefig, pd.DataFrame.to_csv
    sources = {}; proofs = {}; csvs = {}; original_read = pd.read_csv
    canonical = canonical_intervals(root)
    def tracked_read(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).is_file():
            p = Path(path).resolve()
            if p.is_relative_to(root): sources[p.relative_to(root).as_posix()] = sha(p)
        return original_read(path, *args, **kwargs)
    def to_csv(frame, path=None, *args, **kwargs):
        if path is not None:
            if not Path(path).resolve().is_relative_to(output):
                raise ValueError("Original builder attempted a CSV write outside the display destination")
            value = original_csv(frame, None, *args, **kwargs)
            csvs[Path(path).name] = value.encode()
            if write: Path(path).write_bytes(csvs[Path(path).name])
            return None
        return original_csv(frame, None, *args, **kwargs)
    def save(fig, target, *args, **kwargs):
        name = Path(target).stem
        if name not in FIGURE_SOURCES:
            raise ValueError("Unexpected publication figure: " + name)
        if getattr(fig, "_narrative_display_name", None) != name:
            before = figure_signature(fig); changes = rewrite_figure_text(fig, name)
            layout = layout_figure(fig, name)
            after = figure_signature(fig)
            if before != after: raise AssertionError("Figure label rewrite changed data or axes")
            proofs[name] = {"source": FIGURE_SOURCES[name], "source_sha256": sha(root / FIGURE_SOURCES[name]),
                "data_signature_before": before, "data_signature_after": after, "data_and_axes_unchanged_by_text_rewrite": True,
                "text_changes": changes, "layout_changes": layout, "size_inches": list(fig.get_size_inches())}
            fig._narrative_display_name = name
        if write:
            suffix = Path(target).suffix.lower(); k = dict(kwargs)
            if suffix == ".pdf": k["metadata"] = {"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None, "Author": None}
            original_save(fig, output / (name + suffix), *args, **k)
            if suffix == ".png" and name in OLD_FIGURES:
                original_save(fig, output / (name + ".pdf"), metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None, "Author": None})
    def export(fig, destination, name, description, provenance):
        fig.savefig(destination / (name + ".pdf")); fig.savefig(destination / (name + ".png"), dpi=600)
        plt.close(fig)
        return {"name": name}
    with ExitStack() as stack:
        stack.enter_context(patch.object(Figure, "savefig", save)); stack.enter_context(patch.object(pd.DataFrame, "to_csv", to_csv))
        stack.enter_context(patch.object(pd, "read_csv", tracked_read)); stack.enter_context(patch.object(main, "export", export))
        for module in (old, main): stack.enter_context(patch.object(module, "ROOT", root))
        stack.enter_context(patch.object(main, "A", root / BASE / "duration_confirmation")); stack.enter_context(patch.object(main, "B", root / BASE / "schedule_robustness"))
        with plt.rc_context(main.STYLE):
            data = main.inputs(); sources.update({p.relative_to(root).as_posix(): sha(p) for p in data["paths"].values()})
            main.target_example(output); main.confirmation(data, output); main.schedule_figure(main.b_plot_data(data), output)
        with plt.rc_context({"font.size": 9, "axes.titlesize": 10, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"}):
            data = reviewer.load(root)
            # Copies only: keep the original post-hoc controls CSV and all raw-
            # fallback intervals intact while reusing original B duplicate CIs.
            key = "score_preserving_controls/contrasts.csv"
            data[key] = data[key].copy()
            for record in canonical.values():
                if record["study"] != "B": continue
                frame = data[key]
                mask = ((frame.study == "B") & (frame.scorer == "lasso") & (frame.cell == record["cell"]) &
                        (frame.method == record["method"]) & (frame.reference == "raw_window") &
                        (frame.metric == "temporal_delta_auroc"))
                if int(mask.sum()) != 1: raise ValueError("Canonical B figure population mismatch")
                # Point estimates remain the original values (equivalent above).
                for field in ("ci95_low", "ci95_high"): frame.loc[mask, field] = record["canonical"][field]
            for name, factory in (("fig_fallback", reviewer.fallback_figure), ("fig_symmetric", reviewer.symmetric_figure)):
                fig = factory(data); fig.savefig(output / (name + ".pdf")); fig.savefig(output / (name + ".png"), dpi=600); plt.close(fig)
            proofs["fig_fallback"]["authorized_interval_overrides"] = [record for record in canonical.values() if record["study"] == "B"]
            data = evaluation.load(root)
            for name, factory in (("fig_boundary_sensitivity", evaluation.boundary_figure), ("fig_threshold_curves", evaluation.threshold_figure), ("fig_residual_pairs", evaluation.residual_figure)):
                fig = factory(data); fig.savefig(output / (name + ".pdf")); fig.savefig(output / (name + ".png"), dpi=600); plt.close(fig)
        with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"}):
            for name, function in OLD_FIGURES.items(): getattr(old, function)()
    if set(proofs) != set(FIGURE_SOURCES):
        raise ValueError("Publication figure inventory incomplete")
    return proofs, sources, csvs


def build(root=ROOT, output=None):
    root = Path(root).resolve(); output = (root / BASE / "narrative_assets").resolve() if output is None else Path(output).resolve()
    if output != (root / BASE / "narrative_assets").resolve():
        raise ValueError("Narrative output must use its dedicated new directory")
    output.mkdir(parents=True, exist_ok=True)
    previous = read(output / "manifest.json") if (output / "manifest.json").is_file() else {}
    sources = {n: sha(root / n) for n in (*TABLES, *FIGURE_SOURCES.values(), *BUILDERS, *PROTOCOLS, *CANONICAL_CI_SOURCES, TEMPORAL_DP_SOURCE, STUDY_A_DETAIL,
        "configs/aggregation_controls.yaml", "scripts/run_final_confirmation.py", "scripts/diagnose_ess_null.py",
        "requirements-revision.txt", "tools/build_narrative_assets.py", "tests/test_narrative_assets.py")}
    tables, table_proofs = table_outputs(root)
    for name, value in tables.items(): (output / name).write_bytes(value)
    figure_proofs, figure_sources, csvs = figure_outputs(root, output, True); sources.update(figure_sources)
    for name, expected in sources.items():
        if sha(root / name) != expected: raise ValueError("Original source changed during display generation: " + name)
    sources, date_exception = source_records(root, sources, previous)
    figure_proofs["rev2_temporal_grid"]["source_sha256"] = sources[PDF_DATE_SOURCE]
    mapping = {source: BASE + "narrative_assets/" + Path(source).name for source in TABLES}
    mapping.update({source: BASE + "narrative_assets/" + Path(source).name for source in FIGURE_SOURCES.values()})
    (output / "source_to_display.json").write_bytes(json_bytes(mapping))
    manifest = {"role": "publication-only display copies; no new scientific fits, metrics, tests or selected results",
        "source_sha256": sources, "creation_date_only_source": date_exception,
        "table_checks": table_proofs, "figure_checks": figure_proofs, "method_glossary": GLOSSARY,
        "canonical_interval_display": canonical_intervals(root),
        "transformations": ["metric and policy labels; all original printed estimates retained except explicitly audited repeated CI endpoint displays", "same paired B/C AUROC comparisons reuse original corresponding-study intervals; later raw-fallback, boundary and component intervals retain separate fixed streams", "protocol table replaces earlier exploratory row with formal A/B/C designs; seed ranges remain in supplement/protocols", "Study A reference/control table merges only identical A reference rows; AP/F1 split projects all eighteen source rows without numeric reformatting", "table headers/interval cells wrapped; B-policy legend moved outside below axes; every figure data/axis signature preserved", "historical unmatched runtime row retained in source but omitted from publication display", "no new scenario table, bootstrap draws or numerical analysis"],
        "source_to_display": mapping,
        "derived_table_sources": {name: list(table_proofs[name]["sources"]) for name in DERIVED_TABLES},
        "table_count": len(table_proofs), "figure_count": len(FIGURE_SOURCES),
        "new_fits": 0, "new_simulations": 0, "no_original_assets_modified": True,
        "environment": {"python": platform.python_version(), "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "matplotlib")}},
        "original_sample_distinction": "tab_factorial: early seeds0-19; tab_final: separate seeds3000-3049, retaining the original within-seed four-scenario comparison family",
        "PELT_scope": {"earlier_tables": "seven-window grid; penalty_factor1/n_pcs5/min_size1 unless explicitly scanned or oracle segment count", "Study_C_adapter": "23 windows; penalty_factor1/n_pcs5/min_size2/jump1; independent nonoverlapping VAR refits"},
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"}}
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    return {"tables": len(table_proofs), "figures": len(figure_proofs), "new_fits": 0}


def verify(root=ROOT, output=None):
    root = Path(root).resolve(); output = (root / BASE / "narrative_assets").resolve() if output is None else Path(output).resolve()
    manifest = read(output / "manifest.json")
    before = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    if set(before) != set(manifest["output_sha256"]) | {"manifest.json"}: raise ValueError("Narrative inventory drift")
    source_date_difference = []
    for name, expected in manifest["source_sha256"].items():
        if sha(root / name) != expected:
            exception = manifest["creation_date_only_source"]
            if name != PDF_DATE_SOURCE or exception["path"] != name or creation_date_normalized_sha(root / name) != exception["normalized_sha256"]:
                raise ValueError("Narrative original-source checksum mismatch: " + name)
            source_date_difference.append(name)
    for name, expected in manifest["output_sha256"].items():
        if before[name] != expected: raise ValueError("Narrative display checksum mismatch: " + name)
    tables, proof = table_outputs(root)
    if canonical_intervals(root) != manifest["canonical_interval_display"]: raise ValueError("Canonical interval evidence differs")
    if proof != manifest["table_checks"]: raise ValueError("Narrative table numeric checks differ")
    for name, value in tables.items():
        if (output / name).read_bytes() != value: raise ValueError("Narrative table/source reconstruction differs: " + name)
    figures, sources, csvs = figure_outputs(root, output, False)
    figures["rev2_temporal_grid"]["source_sha256"] = manifest["source_sha256"][PDF_DATE_SOURCE]
    if figures != manifest["figure_checks"]: raise ValueError("Narrative figure data/label signature replay differs")
    for name, value in csvs.items():
        if (output / name).read_bytes() != value: raise ValueError("Narrative figure-source CSV differs")
    if before != {p.name: sha(p) for p in output.iterdir() if p.is_file()}: raise AssertionError("Read-only narrative verification wrote files")
    return {"status": "passed", "read_only": True, "tables": len(proof), "figures": len(figures),
            "source_files": len(manifest["source_sha256"]), "source_creation_date_only_differences": source_date_difference,
            "numeric_labels_preserved_except_authorized_ci_duplicates": True,
            "canonical_interval_records": len(manifest["canonical_interval_display"]), "new_fits": 0, "new_simulations": 0}


verify_assets = verify

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT); parser.add_argument("--verify", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.root / BASE / "narrative_assets") if args.verify else build(args.root, args.root / BASE / "narrative_assets"), indent=2))
