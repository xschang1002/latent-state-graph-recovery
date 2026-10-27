"""Build every table and figure used by the revised manuscript.

All numbers come from CSV files produced by the experiment scripts; this file
contains no hard-coded results. Run it after the experiment scripts and before
compiling the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision2"
FIG = ROOT / "results" / "figures"

SETTING_LABEL = {
    "switching_core": "Switching",
    "state_specific": "State-specific",
    "shared_graph": "Shared graph",
    "stationary": "Stationary",
}


def humanise(name: str) -> str:
    return {
        "stcg_v1": "STCG-v1",
        "window_max": "Window max",
        "window_mean": "Window mean",
        "pelt_var_lasso": "PELT + refit",
        "pelt_refit": "PELT + refit",
        "pelt_no_refit": "PELT + window average",
        "global_var_lasso": "Global VAR-Lasso",
        "random_balanced": "Random balanced",
        "kmeans_hard": "K-means hard",
        "kmeans_refit": "K-means + refit",
        "kmeans_soft": "GMM weighted + argmax",
        "temporal_dp": "Temporal DP",
        "oracle_label": "Oracle label",
        "oracle_refit": "Oracle refit",
        "bagged_mean": "Bagged mean",
        "kmeans_eta": "K-means labels",
        "oracle_eta": "Oracle labels",
        "same_support": "Same support",
        "variance_only": "Variance only",
        "mean_only": "Mean only",
        "trend": "Trend",
        "sign_flip": "Sign flip",
        "lag_shift": "Lag shift",
        "stationary": "Stationary",
        "core": "Switching (reference)",
        "lag_corr": "Lagged correlation",
        "regime_pcmci": "Regime-PCMCI",
        "kmeans": "K-means (2 groups)",
        "constrained_2seg": "Constrained 2 segments",
        "constrained_2seg_refit": "Constrained 2 segments + refit",
        "constrained_refit": "Constrained 2 segments + refit",
        "raw_var_split": "Raw VAR split",
        "pelt_bic": "PELT BIC",
        "oracle_2seg": "First-true-changepoint",
    }.get(name, name)


def fmt(value: float, digits: int = 3, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def table_confirmatory() -> str:
    frame = pd.read_csv(ROOT / "results" / "confirmatory" / "confirmatory_tests_confirmation.csv")
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Outcome & Comparison & $\Delta$ & 95\% CI & $p_{\mathrm{Holm}}$ & Sig. \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        metric = "AUROC" if row.metric == "union_auroc" else "AUPRC"
        comparison = row.comparison.replace("stcg_v1_minus_", "STCG $-$ ").replace(
            "window_max", "window max"
        ).replace("pelt", "PELT+refit")
        p_text = "$<$0.0001" if row.p_holm < 1e-4 else f"{row.p_holm:.4f}"
        lines.append(
            f"{metric} & {comparison} & {row.mean_difference:+.4f} & "
            f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] & {p_text} & "
            f"{'yes' if row.significant_holm else 'no'} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_final() -> str:
    frame = pd.read_csv(ROOT / "results" / "confirmatory" / "final_confirmation_tests.csv")
    label = {
        "union_auroc": "Union AUROC",
        "union_auprc": "Union AUPRC",
        "temporal_graph_auroc": "Temporal graph AUROC",
    }
    competitor = {
        "stcg_v1_minus_window_max": "$-$ window max",
        "stcg_v1_minus_pelt_refit": "$-$ PELT+refit",
        "stcg_v1_minus_constrained_refit": "$-$ constrained refit",
        "stcg_v1_minus_global_var_lasso": "$-$ global VAR-Lasso",
    }
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Outcome & Comparison & $\Delta$ & 95\% CI & $p_{\mathrm{Holm}}$ & Sig. \\",
        r"\midrule",
    ]
    primary = frame[frame.family == "primary"]
    descriptive = frame[frame.family == "descriptive"]
    for row in primary.itertuples(index=False):
        p_text = "$<$0.0001" if row.p_holm < 1e-4 else f"{row.p_holm:.4f}"
        lines.append(
            f"{label.get(row.metric, row.metric)} & {competitor.get(row.comparison, row.comparison)} & "
            f"{row.mean_difference:+.4f} & [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] & "
            f"{p_text} & {'yes' if row.significant_holm else 'no'} \\\\"
        )
    if not descriptive.empty:
        lines.append(r"\midrule")
        lines.append(
            r"\multicolumn{6}{l}{\emph{Descriptive reference, not in the Holm family}} \\"
        )
        for row in descriptive.itertuples(index=False):
            lines.append(
                f"{label.get(row.metric, row.metric)} & "
                f"{competitor.get(row.comparison, row.comparison)} & "
                f"{row.mean_difference:+.4f} & [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] & "
                f"-- & -- \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_factorial() -> str:
    frame = pd.read_csv(ROOT / "results" / "baselines" / "refit_factorial.csv")
    order = [
        ("global_var_lasso", "None", "No"),
        ("kmeans_no_refit", "K-means", "No"),
        ("pelt_no_refit", "PELT", "No"),
        ("kmeans_refit", "K-means", "Yes"),
        ("pelt_refit", "PELT", "Yes"),
    ]
    settings = ["switching_core", "state_specific", "shared_graph", "stationary"]
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Partition & Refit & Switching & State-specific & Shared & Stationary & Pooled \\",
        r"\midrule",
    ]
    for method, partition, refit in order:
        subset = frame[frame.method == method]
        cells = [fmt(subset[subset.setting == setting]["union_auroc"].mean()) for setting in settings]
        pooled = fmt(subset["union_auroc"].mean())
        lines.append(f"{partition} & {refit} & " + " & ".join(cells) + f" & {pooled} \\\\")
        if method == "global_var_lasso":
            lines.append(r"\midrule")
    oracle = frame[frame.method == "oracle_refit"]["union_auroc"].mean()
    lines.append(f"Oracle & Yes & \\multicolumn{{4}}{{c}}{{}} & {fmt(oracle)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_segmentation() -> str:
    frame = pd.read_csv(ROOT / "results" / "baselines" / "segmentation_comparison_summary.csv")
    methods = [
        "kmeans",
        "constrained_2seg",
        "raw_var_split",
        "pelt_bic",
        "oracle_2seg",
    ]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Switching} & \multicolumn{2}{c}{State-specific} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Partition & ARI & U-AUROC & ARI & U-AUROC \\",
        r"\midrule",
    ]
    for method in methods:
        cells = []
        for setting in ("switching_core", "state_specific"):
            row = frame[(frame.setting == setting) & (frame.method == method)]
            if row.empty:
                cells.extend(["--", "--"])
                continue
            cells.extend([
                f"{float(row.partition_ari.iloc[0]):+.3f}",
                f"{float(row.union_auroc.iloc[0]):.3f}",
            ])
        lines.append(f"{humanise(method)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_penalty_scan() -> str:
    frame = pd.read_csv(ROOT / "results" / "baselines" / "pelt_penalty_scan.csv")
    # The appendix caption says "switching setting", so the table reports the
    # switching_core numbers, not the pooled mean over the four settings.
    frame = frame[frame.setting == "switching_core"]
    summary = (
        frame.groupby(["penalty_factor", "min_size"])
        .agg(segments=("n_segments", "mean"), union_auroc=("union_auroc", "mean"))
        .reset_index()
    )
    lines = [
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"Penalty factor & Min segment & Mean segments & Union AUROC \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.penalty_factor:.2f} & {row.min_size} & {row.segments:.1f} & {row.union_auroc:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_metrics_layers() -> str:
    frame = pd.read_csv(ROOT / "results" / "aggregation" / "aggregation_controls.csv")
    frame = frame[(frame.setting == "switching_core") & (frame.method == "kmeans_hard")]
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Layer & Prevalence & AUROC & AUPRC & AUPRC / prev. & Normalized AP \\",
        r"\midrule",
    ]
    for label, prefix in (("Union", "union"), ("Regime-specific", "regime")):
        prevalence = float(frame[f"{prefix}_prevalence"].mean())
        auroc = float(frame[f"{prefix}_auroc" if prefix == "union" else "regime_auroc_macro"].mean())
        auprc = float(frame[f"{prefix}_auprc" if prefix == "union" else "regime_auprc_macro"].mean())
        normalized = (auprc - prevalence) / (1.0 - prevalence)
        lines.append(
            f"{label} & {prevalence:.3f} & {auroc:.3f} & {auprc:.3f} & "
            f"{auprc / prevalence:.2f}$\\times$ & {normalized:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_controls() -> str:
    summary = pd.read_csv(ROOT / "results" / "mechanism" / "control_taxonomy_summary.csv")
    order = [
        ("structure_change", "core"),
        ("structure_change", "lag_shift"),
        ("mechanism_change", "same_support"),
        ("mechanism_change", "sign_flip"),
        ("nuisance_change", "variance_only"),
        ("nuisance_change", "mean_only"),
        ("nuisance_change", "trend"),
        ("no_change", "stationary"),
    ]
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Category & Condition & $K_{\mathrm{dyn}}$ & $K_{\mathrm{graph}}$ & Lag AUROC & Pair AUROC & Regime AUPRC & T-AUROC & State ARI \\",
        r"\midrule",
    ]
    previous = None
    for category, control in order:
        row = summary[(summary.category == category) & (summary.control == control)]
        if row.empty:
            continue
        row = row.iloc[0]
        if previous is not None and category != previous:
            lines.append(r"\midrule")
        previous = category
        lines.append(
            f"{category.replace('_', ' ')} & {humanise(control)} & "
            f"{row.k_dyn:.0f} & {row.k_graph:.0f} & {row.union_auroc_lag:.3f} & "
            f"{row.union_auroc_pair:.3f} & {row.regime_auprc:.3f} & "
            f"{row.temporal_graph_auroc:.3f} & {row.state_ari:+.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_known_k() -> str:
    summary = pd.read_csv(ROOT / "results" / "mechanism" / "known_k_check_summary.csv")
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Condition & $K_{\mathrm{true}}$ & $K_{\mathrm{cand}}$ & $\hat K$ & Accepted & Union AUROC & State ARI \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        label = {
            "track1_known_k": "Track I",
            "gate_stress_wrong_candidate": "Gate stress",
            "under_specified_candidate": "Under-specified",
        }.get(row.protocol, row.protocol)
        lines.append(
            f"{SETTING_LABEL.get(row.setting, row.setting)}, {label} & {row.k_true} & "
            f"{row.k_candidate} & {row.k_hat:.1f} & {row.accepted:.2f} & "
            f"{row.union_auroc:.3f} & {row.state_ari:+.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_aggregation() -> str:
    frame = pd.read_csv(ROOT / "results" / "aggregation" / "aggregation_controls_summary.csv")
    frame = frame[(frame.method != "kmeans_eta") & (frame.method != "oracle_eta")]
    methods = [
        "window_mean",
        "window_max",
        "bagged_mean",
        "random_balanced",
        "kmeans_hard",
        "kmeans_soft",
        "temporal_dp",
        "oracle_label",
        "oracle_refit",
    ]
    lookup = frame.set_index(["setting", "method"])
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Switching} & \multicolumn{3}{c}{Stationary} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Method & U-AUROC & R-AUROC & T-AUROC & U-AUROC & R-AUROC & T-AUROC \\",
        r"\midrule",
    ]
    for method in methods:
        cells = []
        for setting in ("switching_core", "stationary"):
            if (setting, method) in lookup.index:
                row = lookup.loc[(setting, method)]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                cells.extend([
                    fmt(row["union_auroc"]),
                    fmt(row["regime_auroc"]),
                    fmt(row.get("temporal_graph", np.nan)),
                ])
            else:
                cells.extend(["--", "--", "--"])
        lines.append(f"{humanise(method)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_ess() -> str:
    frame = pd.read_csv(ROOT / "results" / "tables" / "ess_alignment.csv")
    sensitivity = pd.read_csv(ROOT / "results/tables/ess_alignment_seed_sensitivity.csv")
    main = frame[frame.imputation == "ffill"]
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Method & Expert edges & AUROC & Imputation range & Raw $p$ & Holm $p$ & Holm range \\",
        r"\midrule",
    ]
    for row in main.itertuples(index=False):
        subset = frame[
            (frame.method == row.method) & (frame.binarisation == row.binarisation)
        ]
        low, high = float(subset.auroc.min()), float(subset.auroc.max())
        seed_values = sensitivity[
            (sensitivity.imputation == "ffill") & (sensitivity.method == row.method)
            & (sensitivity.binarisation == row.binarisation)
        ].p_holm
        lines.append(
            f"{humanise(row.method)} & {row.binarisation.replace('_', ' ')} & "
            f"{row.auroc:.3f} & [{low:.3f}, {high:.3f}] & {row.p_raw:.4f} & {row.p_holm:.4f} & "
            f"[{seed_values.min():.4f}, {seed_values.max():.4f}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_temporal_delta() -> str:
    frame = pd.read_csv(ROOT / "results" / "revision2" / "temporal_delta_summary.csv")
    grid = pd.read_csv(ROOT / "results/revision2/temporal_grid_summary.csv")
    core_soft = grid[(grid.axis == "core") & (grid.subset == "all") &
                     grid.method.isin(["gmm_soft", "gmm_weighted_argmax"])]
    frame = pd.concat([frame, core_soft.rename(columns={"temporal_delta_auroc_mean": "temporal_delta_auroc",
                                                       "pooled_temporal_auroc_mean": "pooled_temporal_auroc"})], ignore_index=True)
    order = [
        ("window_mean", "Window mean"),
        ("window_max", "Window max"),
        ("bagged_mean", "Bagged mean"),
        ("raw_window", "Raw window (no grouping)"),
        ("random_balanced", "Random balanced"),
        ("kmeans_hard", "K-means hard"),
        ("gmm_weighted_argmax", "GMM weighted + argmax"),
        ("gmm_soft", "GMM continuous soft"),
        ("temporal_dp", "Temporal DP"),
        ("oracle_label", "Oracle label"),
    ]
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & pooled temporal AUROC & $\mathrm{T\!-\!AUC}_{\Delta}$ \\",
        r"\midrule",
    ]
    for method, label in order:
        row = frame[frame.method == method].iloc[0]
        lines.append(
            f"{label} & {float(row.pooled_temporal_auroc):.3f} & "
            f"{float(row.temporal_delta_auroc):.3f} \\\\"
        )
    ceiling = float(frame.static_ceiling.mean())
    lines += [
        r"\midrule",
        f"Static ceiling $\\mathrm{{AUC}}^*_{{\\mathrm{{static}}}}$ & "
        f"\\multicolumn{{2}}{{c}}{{{ceiling:.3f}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def table_cost() -> str:
    detail = pd.read_csv(ROOT / "results" / "confirmatory" / "confirmatory_detail.csv")
    seconds = detail.groupby("method")["seconds"].mean()
    lines = [
        r"\begin{tabular}{lrl}",
        r"\toprule",
        r"Method & Seconds per fit & Source \\",
        r"\midrule",
    ]
    for method, label in [
        ("window_max", "Window max"),
        ("stcg_v1", "STCG-v1"),
        ("pelt_var_lasso", "PELT + refit"),
        ("pelt_no_refit", "PELT + window average"),
    ]:
        lines.append(f"{label} & {float(seconds[method]):.3f} & measured \\\\")
    lines.append(r"\midrule")
    lines.append(r"Regime-PCMCI & 166.9 & historical, different grid \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def figure_aggregation() -> None:
    frame = pd.read_csv(ROOT / "results" / "aggregation" / "aggregation_controls_summary.csv")
    frame = frame[(frame.method != "kmeans_eta") & (frame.method != "oracle_eta")]
    methods = [
        "window_mean",
        "window_max",
        "bagged_mean",
        "random_balanced",
        "kmeans_hard",
        "kmeans_soft",
        "temporal_dp",
        "oracle_label",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), sharey=True)
    width = 0.27
    x = np.arange(len(methods))
    for axis, setting in zip(axes, ("switching_core", "stationary")):
        subset = frame[frame.setting == setting].set_index("method")
        union = [subset.loc[m, "union_auroc"] if m in subset.index else np.nan for m in methods]
        regime = [subset.loc[m, "regime_auroc"] if m in subset.index else np.nan for m in methods]
        temporal = [subset.loc[m, "temporal_graph"] if m in subset.index else np.nan for m in methods]
        axis.bar(x - width, union, width, color="#1f4e79", label="Union AUROC")
        axis.bar(x, regime, width, color="#c0504d", label="Regime AUROC")
        axis.bar(x + width, temporal, width, color="#2e7d32", label="Temporal graph AUROC")
        axis.set_xticks(x)
        axis.set_xticklabels([humanise(m) for m in methods], rotation=40, ha="right", fontsize=7.5)
        axis.set_ylim(0.5, 1.0)
        axis.set_title(SETTING_LABEL[setting])
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("AUROC")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    figure.subplots_adjust(top=0.80)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(FIG / "rev2_aggregation.png", dpi=300)
    plt.close(figure)


def figure_eta() -> None:
    frame = pd.read_csv(ROOT / "results" / "aggregation" / "aggregation_controls.csv")
    frame = frame[frame.method.isin(["kmeans_eta", "oracle_eta"])]
    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.0))
    styles = {
        "kmeans_eta": ("-o", "#1f4e79", "K-means labels"),
        "oracle_eta": ("--s", "#c0504d", "Oracle labels"),
    }
    for method, (style, colour, label) in styles.items():
        subset = frame[(frame.method == method) & (frame.setting == "switching_core")]
        grouped = subset.groupby("eta").mean(numeric_only=True)
        for axis, column, title in zip(
            axes,
            ("union_auroc", "regime_auprc_macro", "temporal_graph_auroc"),
            ("Union AUROC", "Regime AUPRC", "Temporal graph AUROC"),
        ):
            axis.plot(grouped.index, grouped[column], style, markersize=4, color=colour, label=label)
            axis.set_xlabel(r"Label error $\eta$")
            axis.set_title(title)
            axis.grid(alpha=0.25, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(FIG / "rev2_eta.png", dpi=300)
    plt.close(figure)


def figure_controls() -> None:
    summary = pd.read_csv(ROOT / "results" / "mechanism" / "control_taxonomy_summary.csv")
    order = ["core", "lag_shift", "same_support", "sign_flip", "variance_only", "mean_only", "trend", "stationary"]
    summary = summary.set_index("control").loc[order].reset_index()
    x = np.arange(len(summary))
    figure, axis = plt.subplots(figsize=(9.4, 3.2))
    axis.bar(x - 0.3, summary.union_auroc_lag, 0.3, color="#1f4e79", label="Lag-level AUROC")
    axis.bar(x, summary.union_auroc_pair, 0.3, color="#7f9dc0", label="Pair-level AUROC")
    axis.bar(x + 0.3, summary.temporal_graph_auroc, 0.3, color="#2e7d32", label="Temporal graph AUROC")
    axis.set_xticks(x)
    axis.set_xticklabels([humanise(c) for c in summary.control], rotation=25, ha="right", fontsize=8)
    axis.axhline(0, color="black", linewidth=0.6)
    for boundary in (1.5, 3.5, 6.5):
        axis.axvline(boundary, color="0.7", linewidth=0.7, linestyle=":")
    axis.set_ylabel("AUROC")
    axis.set_ylim(0.5, 1.0)
    axis.legend(frameon=False, fontsize=8, loc="lower left")
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(FIG / "rev2_controls.png", dpi=300)
    plt.close(figure)


def figure_segmentation() -> None:
    frame = pd.read_csv(ROOT / "results" / "baselines" / "segmentation_comparison_summary.csv")
    methods = ["kmeans", "constrained_2seg", "raw_var_split", "pelt_bic", "oracle_2seg"]
    figure, axis = plt.subplots(figsize=(6.6, 3.2))
    width = 0.2
    x = np.arange(len(methods))
    for offset, (setting, colour, label) in zip(
        (-1.5, -0.5, 0.5, 1.5),
        [
            ("switching_core", "#1f4e79", "Switching ARI"),
            ("state_specific", "#c0504d", "State-specific ARI"),
            ("switching_core", "#7f9dc0", "Switching U-AUROC"),
            ("state_specific", "#e39a97", "State-specific U-AUROC"),
        ],
    ):
        values = []
        for method in methods:
            row = frame[(frame.setting == setting) & (frame.method == method)]
            values.append(
                float(row.partition_ari.iloc[0]) if "ARI" in label and not row.empty
                else (float(row.union_auroc.iloc[0]) if not row.empty else np.nan)
            )
        axis.bar(x + offset * width, values, width, color=colour, label=label)
    axis.axhline(0, color="black", linewidth=0.6)
    axis.set_xticks(x)
    axis.set_xticklabels([humanise(m) for m in methods], rotation=20, ha="right", fontsize=8)
    axis.set_ylabel("Score")
    axis.set_ylim(-0.15, 1.0)
    axis.legend(frameon=False, fontsize=7, ncol=2)
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(FIG / "rev2_segmentation.png", dpi=300)
    plt.close(figure)


def figure_factorial() -> None:
    frame = pd.read_csv(ROOT / "results" / "baselines" / "refit_factorial.csv")
    cells = [
        ("kmeans_no_refit", "K-means\nno refit"),
        ("kmeans_refit", "K-means\nrefit"),
        ("pelt_no_refit", "PELT\nno refit"),
        ("pelt_refit", "PELT\nrefit"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
    for axis, setting in zip(axes, ("switching_core", "stationary")):
        subset = frame[frame.setting == setting]
        values = [subset[subset.method == m]["union_auroc"].mean() for m, _ in cells]
        axis.bar(range(len(cells)), values,
                 color=["#1f4e79", "#1f4e79", "#c0504d", "#c0504d"], alpha=0.85)
        global_value = subset[subset.method == "global_var_lasso"]["union_auroc"].mean()
        oracle_value = subset[subset.method == "oracle_refit"]["union_auroc"].mean()
        axis.axhline(global_value, color="black", linestyle="--", linewidth=0.9,
                     label="Global VAR-Lasso")
        axis.axhline(oracle_value, color="#2e7d32", linestyle=":", linewidth=1.0,
                     label="Oracle refit")
        axis.set_xticks(range(len(cells)))
        axis.set_xticklabels([label for _, label in cells], fontsize=7.5)
        axis.set_title(SETTING_LABEL[setting])
        axis.set_ylim(0.7, 1.0)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Union AUROC")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    figure.tight_layout()
    figure.savefig(FIG / "rev2_factorial.png", dpi=300)
    plt.close(figure)


def figure_ess() -> None:
    frame = pd.read_csv(ROOT / "results" / "tables" / "ess_alignment.csv")
    sensitivity = pd.read_csv(ROOT / "results/tables/ess_alignment_seed_sensitivity.csv")
    main = frame[(frame.binarisation == "all_edges") & (frame.imputation == "ffill")].sort_values("auroc")
    figure, (axis, seed_axis) = plt.subplots(1, 2, figsize=(8.4, 4.0),
                                          gridspec_kw={"width_ratios": [1.0, 1.35]})
    axis.set_title("(a) All-edge alignment", fontsize=11)
    x = np.arange(len(main))
    axis.errorbar(
        x,
        main.auroc,
        yerr=[
            main.auroc - [float(frame[(frame.method == m) & (frame.binarisation == "all_edges")].auroc.min()) for m in main.method],
            [float(frame[(frame.method == m) & (frame.binarisation == "all_edges")].auroc.max()) for m in main.method] - main.auroc,
        ],
        fmt="o",
        color="#1f4e79",
        markersize=5,
        label="Observed; imputation range",
        ecolor="0.3",
        capsize=3,
        linewidth=0.8,
    )
    axis.axhline(0.5, color="black", linewidth=0.8, linestyle="--", label="Chance")
    axis.scatter(x, main.null_auroc_mean, color="#c0504d", marker="D", s=24,
                 label="Null mean (seed 0)", zorder=3)
    axis.set_xticks(x)
    axis.set_xticklabels([humanise(m) for m in main.method], rotation=25, ha="right", fontsize=9.5)
    axis.set_ylim(0.4, 0.75)
    axis.set_ylabel("AUROC vs expert graph")
    axis.legend(frameon=False, fontsize=9)
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)
    # Seeds repeat the Monte Carlo calculation, not the physical experiment.
    # Display each adjusted p-value; ranges over seeds are not confidence intervals.
    primary = sensitivity[sensitivity.imputation == "ffill"]
    cells = [(method, edge_set) for method in frame.method.drop_duplicates()
             for edge_set in ("all_edges", "strong_only")]
    seeds = sorted(primary.null_seed.unique())
    colours = ["#1f4e79", "#c0504d", "#38761d", "#674ea7", "#7f6000"]
    markers = ["o", "s", "^", "D", "x"]
    positions = np.arange(len(cells), dtype=float)
    for seed_index, seed in enumerate(seeds):
        selected = primary[primary.null_seed == seed].set_index(["method", "binarisation"])
        values = [float(selected.loc[cell, "p_holm"]) for cell in cells]
        offset = (seed_index - (len(seeds) - 1) / 2) * 0.11
        seed_axis.scatter(values, positions + offset, color=colours[seed_index % len(colours)],
                          marker=markers[seed_index % len(markers)], s=22, label=f"Seed {seed}", zorder=3)
    seed_axis.axvline(0.05, color="black", linestyle="--", linewidth=0.9, label="0.05")
    seed_axis.set_xscale("log")
    seed_axis.set_xlim(min(0.01, float(primary.p_holm.min()) / 2), 1.0)
    seed_axis.set_yticks(positions)
    seed_axis.set_yticklabels([f"{humanise(method)} / {'all' if edge == 'all_edges' else 'strong'}"
                              for method, edge in cells], fontsize=9.5)
    seed_axis.invert_yaxis()
    seed_axis.set_xlabel("Holm-adjusted p (log scale)")
    seed_axis.set_title("(b) Forward-fill null-seed stability", fontsize=11)
    seed_axis.grid(axis="x", alpha=0.25, linewidth=0.6)
    seed_axis.legend(frameon=False, fontsize=9, ncol=3, loc="upper center",
                     bbox_to_anchor=(0.5, -0.19))
    figure.tight_layout()
    figure.savefig(FIG / "rev2_ess.png", dpi=300)
    plt.close(figure)


def figure_mechanism() -> None:
    frame = pd.read_csv(ROOT / "results" / "mechanism" / "mechanism_grid.csv")
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))

    rows = []
    for period in (150, 300, 600, 900):
        subset = frame[(frame.axis == "switch_period") & (frame.level.astype(str) == str(period))]
        if subset.empty:
            continue
        rows.append((period, subset.union_auroc.mean(), subset.regime_auprc_macro.mean(),
                     subset.window_purity_mean.mean()))
    if rows:
        periods, union, regime, purity = zip(*rows)
        axes[0].plot(periods, union, "-o", color="#1f4e79", markersize=4, label="Union AUROC")
        axes[0].plot(periods, regime, "-s", color="#c0504d", markersize=4, label="Regime AUPRC")
        axes[0].plot(periods, purity, "--^", color="#7f7f7f", markersize=4, label="Window purity")
        axes[0].set_xlabel(r"State duration $D$")
        axes[0].set_title("State duration")
        axes[0].legend(frameon=False, fontsize=7.5)

    subset = frame[frame.axis == "overlap"].copy()
    subset["level_float"] = subset.level.astype(float)
    grouped = subset.groupby("level_float")[["union_auroc", "regime_auprc_macro"]].mean().sort_index()
    axes[1].plot(grouped.index, grouped.union_auroc, "-o", color="#1f4e79", markersize=4)
    axes[1].plot(grouped.index, grouped.regime_auprc_macro, "-s", color="#c0504d", markersize=4)
    axes[1].set_xlabel(r"Configured retention probability $o$")
    axes[1].set_title("Support retention")
    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_ylim(0.4, 1.0)
    figure.tight_layout()
    figure.savefig(FIG / "rev2_mechanism.png", dpi=300)
    plt.close(figure)


def temporal_grid_scans(frame: pd.DataFrame):
    """Existing one-factor scans; core appears once within each scan."""
    for name, axis, key in (("Duration", "switch_period", "h_over_D"),
                            ("Length", "timesteps", "n_windows")):
        scan = frame[frame.axis.isin(["core", axis])]
        yield name, key, scan.sort_values(key)


def table_grouping_factors() -> str:
    frame = pd.read_csv(ROOT / "results/revision2/grouping_state_factors_summary.csv")
    lines = [r"\begin{tabular}{rrlrrrrr}", r"\toprule",
             r"$D$ & $M$ & Hard output & $\mathbb E[J]$ & $\mathbb E[C_{\rm edge}]$ & $\mathbb E[JC_{\rm edge}]$ & Hard & Raw \\", r"\midrule"]
    for row in frame.itertuples(index=False):
        lines.append(f"{row.D} & {row.M} & {row.execution.capitalize()} & {row.J:.3f} & {row.C_edge:.3f} & "
                     f"{row.J_times_C:.3f} & {row.hard_temporal_auroc:.3f} & {row.raw_temporal_auroc:.3f} " + r"\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


def table_matched_grouping() -> str:
    frame = pd.read_csv(ROOT / "results/revision2/matched_random_contrasts.csv")
    labels = {"hard_minus_raw": "Hard $-$ raw", "hard_minus_matched_random": "Hard $-$ matched random",
              "matched_random_minus_raw": "Matched random $-$ raw"}
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Comparison & Mean difference & Pointwise 95\% CI & Seeds \\", r"\midrule"]
    for row in frame.itertuples(index=False):
        lines.append(f"{labels[row.contrast]} & {row.mean:+.3f} & [{row.ci95_low:+.3f}, {row.ci95_high:+.3f}] & {row.n_seeds} " + r"\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


def table_temporal_coverage() -> str:
    frame = pd.read_csv(ROOT / "results/revision2/temporal_grid_coverage.csv")
    axis_labels = {"overlap": "Overlap", "switch_period": "Duration", "observation_noise": "Obs. noise",
                   "edge_prob": "Edge probability", "timesteps": "Length", "effect_scale": "Effect scale",
                   "max_lag": "Maximum lag", "state_proportions": "State shares", "control": "Control"}
    lines = [r"\begin{tabular}{rlrrrr}", r"\toprule",
             r"ID & Configuration & $M$ & Valid/20 & No change & No flip \\", r"\midrule"]
    seen = {}
    for row in frame.itertuples(index=False):
        level = str(row.level).replace("_", " ")
        if row.axis == "core":
            description = "Reference"
        elif "+" in row.axis:
            axes = row.axis.split("+"); levels = level.split("|")
            description = " / ".join(f"{axis_labels[axis]} {value}" for axis, value in zip(axes, levels))
        elif row.axis == "state_proportions":
            description = "State shares " + level.strip("[]").replace(", ", "/")
        else:
            description = f"{axis_labels[row.axis]} {level}"
        if row.config_id in seen:
            description += f" (=C{seen[row.config_id]:02d})"
        else:
            seen[row.config_id] = row.grid_entry
        lines.append(f"C{row.grid_entry:02d} & {description} & {row.M} & {row.n_evaluable} & "
                     f"{row.n_no_support_change} & {row.n_no_observed_flip} " + r"\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


def table_temporal_grid(accepted: bool = False) -> str:
    frame = pd.read_csv(ROOT / "results/revision2/temporal_grid_summary.csv")
    subset = "hard_gate_accepted" if accepted else "all"
    frame = frame[frame.subset == subset]
    if accepted:
        lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
                 r"Scan/level & $n$ & Raw & Hard $\Delta$ [95\% CI] & GMM soft $\Delta$ [95\% CI] & Oracle $\Delta$ [95\% CI] \\"]
    else:
        lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
                 r"Scan/level & $M$ & Gate & Raw & Hard $\Delta$ & Forced hard $\Delta$ & GMM soft $\Delta$ & Oracle $\Delta$ \\"]
    lines.append(r"\midrule")
    for name, key, scan in temporal_grid_scans(frame):
        span = 6 if accepted else 8
        description = r"Duration scan ($M=7$)" if name == "Duration" else r"Length scan ($h/D=1$)"
        lines.append(rf"\multicolumn{{{span}}}{{l}}{{{description}}} " + r"\\")
        for entry in scan.grid_entry.drop_duplicates():
            rows = scan[scan.grid_entry == entry].set_index("method")
            raw = rows.loc["raw_window"]
            ratio = "1/3" if np.isclose(raw.h_over_D, 1 / 3) else "1/2" if np.isclose(raw.h_over_D, 0.5) else f"{raw.h_over_D:.3g}"
            level = f"$h/D={ratio}$" if name == "Duration" else f"$M={raw.n_windows:.0f}$"
            cells = [level, str(int(raw.n_evaluable)) if accepted else str(int(raw.n_windows))]
            if not accepted:
                cells.append(f"{raw.n_hard_gate_accepted:.0f}/{raw.n_total:.0f}")
            cells.append(fmt(raw.raw_temporal_delta_auroc_mean))
            for method in ("kmeans_hard", "gmm_soft", "oracle_label"):
                row = rows.loc[method]
                value = fmt(row.delta_temporal_mean, signed=True)
                if accepted:
                    low, high = row.delta_temporal_ci95_low, row.delta_temporal_ci95_high
                    value += f" [{fmt(low, signed=True)}, {fmt(high, signed=True)}]" if np.isfinite(low) else r" [--]"
                cells.append(value)
                if method == "kmeans_hard" and not accepted:
                    cells.append(fmt(row.candidate_delta_temporal_mean, signed=True))
            lines.append(" & ".join(cells) + r" \\")
        if name == "Duration":
            lines.append(r"\midrule")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


def figure_temporal_grid() -> None:
    frame = pd.read_csv(ROOT / "results/revision2/temporal_grid_summary.csv")
    detail = pd.read_csv(ROOT / "results/revision2/temporal_grid_detail.csv")
    style = [("kmeans_hard", "Hard, native gate", "#9C3F32", "o", "delta_temporal"),
             ("kmeans_hard", "Hard, forced", "#555555", "s", "candidate_delta_temporal"),
             ("gmm_soft", "GMM continuous soft", "#005A83", "^", "delta_temporal"),
             ("oracle_label", "Oracle labels", "#634580", "D", "delta_temporal")]
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.facecolor": "white"}):
        figure, axes = plt.subplots(3, 2, figsize=(6.4, 6.7), sharex="col",
                                    gridspec_kw={"height_ratios": [1, 1, 0.65]})
        lower, upper = 0.0, 0.0
        for column, (name, key, scan) in enumerate(temporal_grid_scans(frame)):
            all_rows = scan[scan.subset == "all"]
            entries = all_rows.grid_entry.drop_duplicates().tolist()
            xs = np.arange(len(entries))
            raw_rows = all_rows[all_rows.method == "raw_window"].set_index("grid_entry").loc[entries]
            labels = ([f"{value:.3g}" for value in raw_rows.h_over_D] if name == "Duration"
                      else [str(int(value)) for value in raw_rows.n_windows])
            for row_index, subset in enumerate(("all", "hard_gate_accepted")):
                axis = axes[row_index, column]
                selected = scan[scan.subset == subset]
                current_styles = style if row_index == 0 else [s for s in style if s[1] != "Hard, forced"]
                for index, (method, label, color, marker, metric) in enumerate(current_styles):
                    data = selected[selected.method == method].set_index("grid_entry").loc[entries]
                    means = data[metric + "_mean"].to_numpy()
                    lows = data[metric + "_ci95_low"].to_numpy()
                    highs = data[metric + "_ci95_high"].to_numpy()
                    offsets = xs + (index - (len(current_styles) - 1) / 2) * 0.09
                    axis.plot(offsets, means, marker=marker, markersize=4, linewidth=1,
                              linestyle="--" if label == "Hard, forced" else "-", color=color, label=label)
                    for x, mean, low, high in zip(offsets, means, lows, highs):
                        if np.isfinite(low):
                            axis.vlines(x, low, high, color=color, linewidth=0.8)
                            lower, upper = min(lower, low), max(upper, high)
                        elif np.isfinite(mean):
                            axis.annotate("*", (x, mean), xytext=(3, 3), textcoords="offset points", fontsize=8)
                        if np.isfinite(mean):
                            lower, upper = min(lower, mean), max(upper, mean)
                if row_index == 1:
                    for x, entry in zip(xs, entries):
                        seeds = detail[(detail.grid_entry == entry) & (detail.method == "kmeans_hard")
                                       & detail.hard_gate_accepted]
                        if 0 < len(seeds) < 5:
                            axis.scatter(np.full(len(seeds), x - 0.09), seeds.delta_temporal,
                                         color=style[0][2], marker="x", s=16, zorder=5)
                        axis.text(x, 0.02, f"n={len(seeds)}", transform=axis.get_xaxis_transform(),
                                  ha="center", va="bottom", fontsize=8)
                axis.axhline(0, color="black", linestyle=":", linewidth=0.8)
                axis.set_ylabel(r"$\Delta_{\rm temporal}$")
                axis.grid(alpha=0.18, linewidth=0.5)
                axis.text(0.02, 0.98, f"({chr(97 + 2 * row_index + column)}) " +
                          ("All seeds" if row_index == 0 else "Hard-gate accepted"),
                          transform=axis.transAxes, va="top", fontsize=8)
            axis = axes[2, column]
            gate = raw_rows.hard_gate_acceptance.to_numpy()
            axis.bar(xs, gate, width=0.5, color="#B7BCC1", edgecolor="#444444", linewidth=0.8)
            for x, rate, row in zip(xs, gate, raw_rows.itertuples()):
                axis.text(x, rate + 0.04, f"{int(row.n_hard_gate_accepted)}/{int(row.n_total)}", ha="center", fontsize=8)
            axis.set(ylim=(0, 1.3), yticks=[0, 0.5, 1], ylabel="Hard gate accepted",
                     xticks=xs, xticklabels=labels)
            axis.set_xlabel(r"$h/D$ ($M=7$)" if name == "Duration" else r"$M$ (varying $T$; $h=D=300$)")
            axis.text(0.02, 0.98, f"({chr(101 + column)}) Gate frequency", transform=axis.transAxes, va="top", fontsize=8)
        margin = max(0.04, (upper - lower) * 0.13)
        for axis in axes[:2].flat:
            axis.set_ylim(lower - margin, upper + margin)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
        figure.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.0)
        figure.savefig(FIG / "rev2_temporal_grid.png", dpi=600)
        figure.savefig(FIG / "rev2_temporal_grid.pdf",
                       metadata={"Creator": None, "Producer": None, "Author": None})
        plt.close(figure)


def table_temporal_gmm_numerics() -> str:
    frame = pd.read_csv(ROOT / "results/revision2/temporal_grid_gmm_numerics.csv")
    detail = pd.read_csv(ROOT / "results/revision2/temporal_grid_detail.csv")
    focus = detail[(detail.axis.isin(["core", "switch_period", "timesteps"])) & (detail.method == "raw_window")]
    cells = focus[["grid_entry", "h_over_D", "n_windows", "axis", "level"]].drop_duplicates()
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Cell & $h/D$ & $M$ & GMM argmax & GMM continuous & Continuous, round 12 \\", r"\midrule"]
    for cell in cells.itertuples(index=False):
        selected = frame[frame.grid_entry == cell.grid_entry]
        soft = selected[selected.method == "gmm_soft"]
        argmax = selected[selected.method == "gmm_weighted_argmax"]
        name = "Reference" if cell.axis == "core" else (f"$D={cell.level}$" if cell.axis == "switch_period" else f"$T={cell.level}$")
        lines.append(f"{name} & {cell.h_over_D:.3g} & {cell.n_windows} & "
                     f"{argmax.native_temporal_delta_auroc.mean():.6f} & "
                     f"{soft.native_temporal_delta_auroc.mean():.6f} & "
                     f"{soft.rounded_temporal_delta_auroc.mean():.6f} " + r"\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


def table_ess_diagnostics() -> str:
    frame = pd.read_csv(ROOT / "results/tables/ess_null_diagnostics.csv")
    lines = [r"\begin{tabular}{llrrrr@{\hspace{1.2em}}rrr}", r"\toprule",
             r"Edges & Seed & Burn/$|E|$ & 2-edge (\%) & 3-cycle (\%) & $b$ & $\rho_1$ & ESS & MCSE $\times10^3$ \\",
             r"\midrule"]
    for edge, title in (("all_edges", "All"), ("strong_only", "Strong")):
        for row in frame[frame.binarisation == edge].itertuples(index=False):
            lines.append(f"{title} & {row.null_seed} & {row.burn_in_per_edge} & "
                         f"{100 * row.sampling_two_edge_acceptance:.2f} & "
                         f"{100 * row.sampling_three_cycle_acceptance:.3f} & "
                         f"{row.null_exceedances} & ${row.tail_acf1:+.3f}$ & "
                         f"{row.tail_ess:.0f} & {1000 * row.tail_mcse:.3f} " + r"\\")
        if edge == "all_edges":
            lines.append(r"\midrule")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


def figure_ess_diagnostics() -> None:
    # Show the full saved seed-0 trace, not a selected interval or subsample.
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from diagnose_ess_null import autocorrelation

    traces = pd.read_csv(ROOT / "results/tables/ess_null_traces.csv")
    summary = pd.read_csv(ROOT / "results/tables/ess_null_diagnostics.csv")
    figure, axes = plt.subplots(2, 2, figsize=(6.4, 3.8))
    for column, (edge, title) in enumerate((("all_edges", "All expert edges"),
                                          ("strong_only", "Strong expert edges"))):
        trace = traces[(traces.binarisation == edge) & (traces.null_seed == 0)
                       & (traces.burn_in_per_edge == 100)]
        observed = summary[(summary.binarisation == edge) & (summary.null_seed == 0)
                           & (summary.burn_in_per_edge == 100)].observed_auroc.iloc[0]
        axis = axes[0, column]
        axis.plot(trace.draw, trace.null_auroc, color="#8095a6", alpha=0.5,
                  linewidth=0.35, label="Retained null AUROC")
        axis.plot(trace.draw, trace.null_auroc.rolling(100).mean(), color="#164b73",
                  linewidth=1.0, label="100-draw running mean")
        axis.axhline(observed, color="#a13936", linestyle="--", linewidth=1,
                     label="Observed AUROC")
        events = trace[trace.tail_event == 1]
        axis.scatter(events.draw, events.null_auroc, color="#a13936", s=7, zorder=3)
        axis.set(title=title, xlabel="Retained draw (seed 0)", ylabel="AUROC")
        axis.set_title(title, fontsize=10)
        axis.legend(fontsize=7, loc="lower left", framealpha=0.9)
        axis = axes[1, column]
        for values, label, color in ((trace.null_auroc.to_numpy(), "Null AUROC", "#164b73"),
                                     (trace.tail_event.to_numpy(), "Tail indicator", "#a13936")):
            acf = autocorrelation(values)
            axis.plot(np.arange(1, 41), acf[1:41], linewidth=1, label=label, color=color)
        axis.axhline(0, color="gray", linewidth=0.6)
        axis.set(xlabel="Lag (retained draws)", ylabel="Autocorrelation")
        axis.legend(fontsize=7)
    # Keep a common symmetric scale that includes every plotted lag.
    limit = max(0.05, 1.15 * max(abs(line.get_ydata()).max()
                for axis in axes[1] for line in axis.lines[:-1]))
    for axis in axes[1]:
        axis.set_ylim(-limit, limit)
    for axis in axes.flat:
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.tick_params(labelsize=8)
    figure.tight_layout()
    figure.savefig(FIG / "rev2_ess_diagnostics.png", dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    tables = {
        "tab_confirmatory.tex": table_confirmatory(),
        "tab_final.tex": table_final(),
        "tab_factorial.tex": table_factorial(),
        "tab_segmentation.tex": table_segmentation(),
        "tab_penalty_scan.tex": table_penalty_scan(),
        "tab_metrics_layers.tex": table_metrics_layers(),
        "tab_controls.tex": table_controls(),
        "tab_known_k.tex": table_known_k(),
        "tab_aggregation.tex": table_aggregation(),
        "tab_ess.tex": table_ess(),
        "tab_ess_diagnostics.tex": table_ess_diagnostics(),
        "tab_temporal_delta.tex": table_temporal_delta(),
        "tab_temporal_grid.tex": table_temporal_grid(),
        "tab_temporal_grid_accepted.tex": table_temporal_grid(accepted=True),
        "tab_temporal_gmm_numerics.tex": table_temporal_gmm_numerics(),
        "tab_grouping_factors.tex": table_grouping_factors(),
        "tab_matched_grouping.tex": table_matched_grouping(),
        "tab_temporal_coverage.tex": table_temporal_coverage(),
        "tab_cost.tex": table_cost(),
    }
    for name, content in tables.items():
        (out / name).write_text(content + "\n", encoding="utf-8", newline="\n")

    figure_aggregation()
    figure_eta()
    figure_controls()
    figure_segmentation()
    figure_factorial()
    figure_ess()
    figure_ess_diagnostics()
    figure_mechanism()
    figure_temporal_grid()
    print(f"wrote {len(tables)} tables and 9 figures to {out} / {FIG}")


if __name__ == "__main__":
    main()
