"""Presentation-only detailed tables for the frozen Information Sciences studies.

Read retained CSV/fit records, emit three supplement tables and their source
CSVs outside the frozen study directories, and confirm every frozen byte stayed
unchanged. No simulation, estimator, selected sample or adjusted family changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/information_sciences"
A = OUT / "duration_confirmation"
B = OUT / "schedule_robustness"
SCHEDULES = ("fixed_phase", "random_phase", "irregular")
SCHEDULE_LABELS = {"fixed_phase": "Fixed", "random_phase": "Random", "irregular": "Irregular"}
METHOD_LABELS = {"raw_window": "Raw", "native_hard": "Native", "forced_hard": "Forced",
                 "matched_random": "Matched", "oracle_label": "Oracle", "static_mean": "Static mean"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_hashes() -> dict:
    return {path.relative_to(ROOT).as_posix(): digest(path)
            for directory in (A, B) for path in sorted(directory.rglob("*")) if path.is_file()}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def number(value: float, digits: int = 3, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def interval(mean: float, low: float, high: float) -> str:
    return "$" + number(mean, signed=True) + r"\;[" + number(low) + "," + number(high) + "]$"


def pvalue(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value == 0 or value >= 0.001:
        return f"${value:.3g}$"
    exponent = int(np.floor(np.log10(value)))
    return f"${value / 10.0**exponent:.2f}" + r"\times10^{" + str(exponent) + "}$"


def pointwise_ci(values, settings: dict) -> tuple[float, float, float, int]:
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan, 0
    mean = float(values.mean())
    if len(values) < settings["minimum_ci_n"]:
        return mean, np.nan, np.nan, len(values)
    rng = np.random.default_rng(settings["bootstrap_seed"])
    means = values[rng.integers(len(values), size=(settings["bootstrap_draws"], len(values)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return mean, float(low), float(high), len(values)


def a_contrasts() -> tuple[str, pd.DataFrame]:
    frame = read_csv(A / "contrasts.csv")
    assert len(frame) == 16 and int(frame.primary.sum()) == 4
    assert frame.loc[~frame.primary, "p_holm"].isna().all()
    lines = [r"\begin{tabular}{lllrrrr}", r"\toprule",
             r"Scorer & Hard policy & Contrast & $n$ & Mean [95\% interval] & $p_{\rm raw}$ & $p_{\rm Holm4}$ \\", r"\midrule"]
    for scorer in ("lasso", "ols"):
        for policy in ("native_hard", "forced_hard"):
            contrasts = ((f"{policy}_minus_raw_D300", r"$\Delta_{300}$"),
                         (f"{policy}_minus_raw_D600", r"$\Delta_{600}$"),
                         (f"{policy}_duration_interaction", r"$\Gamma$"),
                         (f"{policy}_minus_matched_random_D600", r"Hard--matched, $600$"))
            for name, label in contrasts:
                row = frame[(frame.scorer == scorer) & (frame.contrast == name)].iloc[0]
                hard = "Native" if policy == "native_hard" else "Forced"
                if row.primary: hard += r"$^{P}$"
                lines.append(f"{scorer.upper() if scorer == 'ols' else 'Lasso'} & {hard} & {label} & "
                             f"{int(row.n_paired)} & {interval(row['mean'], row.ci95_low, row.ci95_high)} & "
                             f"{pvalue(row.p_raw)} & {pvalue(row.p_holm)} " + r"\\")
            lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    return "\n".join(lines + [r"\end{tabular}"]), frame


def b_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    gate = read_csv(B / "schedule_robustness_gate_summary.csv")
    paired = read_csv(B / "schedule_robustness_paired.csv")
    diagnostic = read_csv(B / "schedule_robustness_diagnostics.csv")
    tests = read_csv(B / "schedule_robustness_secondary_tests.csv")
    settings = json.loads((B / "schedule_robustness_protocol.json").read_text())["configuration"]["statistics"]
    gates, factors = [], []
    for duration in (300, 600):
        for kind in SCHEDULES:
            cell = paired[(paired.duration_mean == duration) & (paired.schedule == kind)].sort_values("seed")
            meta = diagnostic[(diagnostic.duration_mean == duration) & (diagnostic.schedule == kind)].sort_values("seed")
            original = gate[(gate.duration_mean == duration) & (gate.schedule == kind)].iloc[0]
            forced_test = tests[(tests.family == "B_forced_minus_raw") & (tests.duration_mean == duration) & (tests.schedule == kind)].iloc[0]
            accepted_seeds = meta.loc[meta.hard_gate_accepted == True, "seed"]
            accepted = cell[cell.seed.isin(accepted_seeds)]
            native = pointwise_ci(cell.native_minus_raw, settings)
            forced = pointwise_ci(cell.forced_minus_raw, settings)
            accepted_mean = float(accepted.forced_minus_raw.mean())
            np.testing.assert_allclose([forced[0], forced[1], forced[2]],
                                       [forced_test.mean_difference, forced_test.ci95_low, forced_test.ci95_high], atol=1e-14, rtol=0)
            np.testing.assert_allclose([native[0], forced[0], accepted_mean],
                                       [original.all_native_minus_raw_mean, original.all_forced_minus_raw_mean,
                                        original.accepted_forced_minus_raw_mean], atol=1e-14, rtol=0, equal_nan=True)
            assert len(accepted) == original.n_gate_accepted == original.n_accepted_evaluable
            gates.append({"duration_mean": duration, "schedule": kind, "n_planned": int(original.n_planned),
                          "n_gate_observed": int(original.n_gate_observed), "n_accepted": len(accepted),
                          "native_minus_raw": native[0], "native_ci95_low": native[1], "native_ci95_high": native[2],
                          "forced_minus_raw": forced[0], "forced_ci95_low": forced[1], "forced_ci95_high": forced[2],
                          "accepted_n_evaluable": len(accepted), "accepted_forced_minus_raw": accepted_mean})
            factor = {"duration_mean": duration, "schedule": kind, "n_evaluable": int(np.isfinite(cell.forced_minus_raw).sum()),
                      "J_mean": float(meta.J.mean()), "C_edge_mean": float(meta.C_edge.mean()),
                      "J_times_C_mean": float(meta.J_times_C.mean()), "raw_temporal_auroc": float(cell.raw_window.mean()),
                      "forced_temporal_auroc": float(cell.forced_hard.mean())}
            np.testing.assert_allclose([factor["J_mean"], factor["C_edge_mean"], factor["J_times_C_mean"]],
                                       [original.J_mean, original.C_edge_mean, original.J_times_C_mean], atol=1e-14, rtol=0)
            np.testing.assert_allclose(factor["forced_temporal_auroc"], 0.5 + 0.5 * factor["J_times_C_mean"], atol=1e-14, rtol=0)
            factors.append(factor)
    return pd.DataFrame(gates), pd.DataFrame(factors)


def b_table(gates: pd.DataFrame, factors: pd.DataFrame) -> str:
    lines = [r"\begin{minipage}{\linewidth}", r"\centering\small",
             r"(a) Gate coverage and paired temporal gains\par\medskip",
             r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"$\bar D$ & Schedule & Gate & Native--raw [95\%] & Forced--raw [95\%] & $n_{\rm acc}$ & Accepted mean \\", r"\midrule"]
    for row in gates.itertuples(index=False):
        accepted = "--" if not np.isfinite(row.accepted_forced_minus_raw) else "$" + number(row.accepted_forced_minus_raw, signed=True) + "$"
        lines.append(f"{row.duration_mean} & {SCHEDULE_LABELS[row.schedule]} & {row.n_accepted}/{row.n_gate_observed} & "
                     f"{interval(row.native_minus_raw, row.native_ci95_low, row.native_ci95_high)} & "
                     f"{interval(row.forced_minus_raw, row.forced_ci95_low, row.forced_ci95_high)} & "
                     f"{row.accepted_n_evaluable} & {accepted} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\par\bigskip", r"(b) Factors and absolute temporal AUROC on all fifty repeats\par\medskip",
                  r"\begin{tabular}{llrrrrr}", r"\toprule",
                  r"$\bar D$ & Schedule & $\overline J$ & $\overline C_{\rm edge}$ & $\overline{JC_{\rm edge}}$ & Raw & Forced \\", r"\midrule"])
    for row in factors.itertuples(index=False):
        lines.append(f"{row.duration_mean} & {SCHEDULE_LABELS[row.schedule]} & {number(row.J_mean)} & {number(row.C_edge_mean)} & "
                     f"{number(row.J_times_C_mean)} & {number(row.raw_temporal_auroc)} & {number(row.forced_temporal_auroc)} " + r"\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{minipage}"])


def numerical_sources() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    protocol_a = json.loads((A / "protocol.json").read_text())
    protocol_b = json.loads((B / "schedule_robustness_protocol.json").read_text())["configuration"]
    fits, all_ols = [], []
    for scorer in ("lasso", "ols"):
        for duration in (300, 600):
            subset = []
            for seed in protocol_a["repeat_seeds"]:
                part = json.loads((A / "records" / f"repeat_{seed}_D{duration}_{scorer}.json").read_text())
                info = part["diagnostic"]
                numerics = info.get("numerical", [])
                if scorer == "ols": all_ols.extend(numerics)
                subset.append((part["status"] == "success", numerics, info.get("clustering_warnings", [])))
            fits.append(fit_row("A", scorer, duration, subset))
    for duration in (300, 600):
        subset = []
        for seed in range(protocol_b["seed_start"], protocol_b["seed_start"] + protocol_b["seed_count"]):
            for kind in SCHEDULES:
                part = json.loads((B / "parts" / f"repeat_{seed}_D{duration}_{kind}.json").read_text())
                info = part["diagnostic"]
                subset.append((info["status"] == "ok", info.get("numerical", []), info.get("clustering_warnings", [])))
        fits.append(fit_row("B", "lasso", duration, subset))
    ols_summary = read_csv(A / "summary.csv")
    rows = []
    for duration in (300, 600):
        row = {"D": duration, "n": 100}
        for method in METHOD_LABELS:
            values = ols_summary[(ols_summary.scorer == "ols") & (ols_summary.D == duration) & (ols_summary.method == method)].iloc[0]
            assert values.temporal_delta_auroc_n == row["n"]
            row[method] = values.temporal_delta_auroc_mean
        assert row["native_hard"] == row["forced_hard"]
        rows.append(row)
    detail = {"ols_window_matrices": len(all_ols), "ols_row_counts": sorted({row["n_rows"] for row in all_ols}),
              "ols_feature_counts": sorted({row["n_features"] for row in all_ols}),
              "ols_ranks": sorted({row["rank"] for row in all_ols}),
              "ols_minimum_singular_value": min(row["singular_min"] for row in all_ols),
              "ols_maximum_condition_number": max(row["singular_max"] / row["singular_min"] for row in all_ols)}
    assert detail["ols_window_matrices"] == 1400 and detail["ols_ranks"] == [24]
    return pd.DataFrame(fits), pd.DataFrame(rows), detail


def fit_row(study: str, scorer: str, duration: int, subset: list) -> dict:
    numerical = [row for _, numbers, _ in subset for row in numbers]
    ranked = [row for row in numerical if "rank" in row]
    return {"study": study, "scorer": scorer, "D": duration, "n_fits": len(subset),
            "n_failed": sum(not success for success, _, _ in subset), "n_window_matrices": len(numerical),
            "rank_min": min((row["rank"] for row in ranked), default=np.nan),
            "rank_max": max((row["rank"] for row in ranked), default=np.nan),
            "rank_deficient_matrices": sum(row["rank"] < row["n_features"] for row in ranked),
            "window_warning_count": sum(len(row.get("warnings", [])) for row in numerical),
            "cluster_warning_count": sum(len(warnings) for _, _, warnings in subset)}


def numerical_table(fits: pd.DataFrame, ols: pd.DataFrame, detail: dict) -> str:
    lines = [r"\begin{minipage}{\linewidth}", r"\centering\small",
             r"(a) Auxiliary OLS absolute changing-edge temporal AUROC\par\medskip",
             r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"$D$ & Raw & Hard (native=forced) & Matched random & Oracle & Static mean \\", r"\midrule"]
    for row in ols.itertuples(index=False):
        lines.append(f"{row.D} & {number(row.raw_window)} & {number(row.native_hard)} & {number(row.matched_random)} & "
                     f"{number(row.oracle_label)} & {number(row.static_mean)} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\par\bigskip", r"(b) Complete fit and window-design coverage\par\medskip",
                  r"\begin{tabular}{lllrrrrr}", r"\toprule",
                  r"Study & Scorer & $D$ & Fits & Failed & Window matrices & Rank range & Warnings \\", r"\midrule"])
    for row in fits.itertuples(index=False):
        rank = "--" if not np.isfinite(row.rank_min) else f"{int(row.rank_min)}--{int(row.rank_max)}"
        lines.append(f"{row.study} & {'OLS' if row.scorer == 'ols' else 'Lasso'} & {row.D} & {row.n_fits} & {row.n_failed} & "
                     f"{row.n_window_matrices} & {rank} & {row.window_warning_count + row.cluster_warning_count} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\par\medskip\raggedright\small",
                  r"All OLS matrices are $297\times24$ with rank $24$. The minimum recorded singular value is $" +
                  number(detail["ols_minimum_singular_value"], 3) + r"$; the largest recorded condition number is $" +
                  number(detail["ols_maximum_condition_number"], 3) + r"$. Lasso ranks were not a fitting criterion and are not reported.",
                  r"\end{minipage}"])
    return "\n".join(lines)


def main() -> None:
    before = frozen_hashes()
    assert before, "Frozen studies must be supplied"
    a_tex, contrasts = a_contrasts()
    gates, factors = b_sources()
    fits, ols, diagnostic = numerical_sources()
    outputs = {"tab_supp_a_contrasts.tex": a_tex, "tab_supp_b_gate_factors.tex": b_table(gates, factors),
               "tab_supp_diagnostics_ols.tex": numerical_table(fits, ols, diagnostic)}
    for name, content in outputs.items():
        (OUT / name).write_text(content + "\n", encoding="utf-8", newline="\n")
    csvs = {"supp_a_contrasts_source.csv": contrasts, "supp_b_gate_source.csv": gates,
            "supp_b_factors_source.csv": factors, "supp_fit_diagnostics_source.csv": fits,
            "supp_ols_absolute_source.csv": ols}
    for name, frame in csvs.items():
        frame.to_csv(OUT / name, index=False, lineterminator="\n")
    assert before == frozen_hashes(), "Presentation rebuilding changed frozen studies"
    manifest = {"role": "presentation only; no new fits or inferential families", "frozen_files_unchanged": len(before),
                "frozen_source_sha256": before, "numerical_diagnostics": diagnostic,
                "primary_A_tests": 4, "auxiliary_A_rows_with_no_Holm_assignment": 12,
                "secondary_B_family_sizes": [6, 3], "accepted_only_role": "conditional descriptive means, no tests or intervals",
                "sha256": {name: digest(OUT / name) for name in (*outputs, *csvs)}}
    (OUT / "supplement_assets_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote 3 supplementary tables and 5 source CSVs; {len(before)} frozen files unchanged")
    print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    main()
