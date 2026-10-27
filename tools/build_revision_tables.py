"""Build the compact paper-facing tex tables for the revision from tracked CSVs.

Generated outputs (all under results/tables/):

- synthetic_switching_paired_edge_auc_table.tex
    One row per opponent: paired STCG-v1 vs opponent Edge AUC on switching_var
    (20 shared seeds), from synthetic_switching_paired_matrix.csv.
- real_regime_discovery_comparison_table.tex
    Compact head-to-head regime-recovery table on labeled real transition
    segments, from real_regime_discovery_comparison.csv. Failure annotations
    follow the "Failed fits" / "Not-run plan combos" sections of
    real_regime_discovery_comparison.md.
- stcg_knob_ablation_table.tex
    One row per knob group with the Edge AUC range and the finding, from the
    four stcg_knob_ablation_<group>_summary.csv files.

- paper_evidence_summary_table.tex
    Interpretation summary distinguishing supplied counts, descriptive diagnosis,
    post-pilot comparisons and horizon-purged warnings.

Run from the repo root:

    .venv/Scripts/python.exe tools/build_revision_tables.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def fmt_p(p):
    if p < 1e-4:
        return "$<$1e-4"
    return f"{p:.4f}".rstrip("0").rstrip(".") if p >= 0.1 else f"{p:.4f}"


def build_paired_edge_auc():
    df = pd.read_csv(TABLES / "synthetic_switching_paired_matrix.csv")
    df = df[df["metric"] == "edge_auc"].copy()
    df = df.sort_values("mean_b", ascending=False)
    rows = ["\\begin{tabular}{lrrrrl}",
            "\\toprule",
            "Opponent & Mean & Improv. & 95\\% CI & $p_t$ & $p_W$ \\\\",
            "\\midrule"]
    for _, r in df.iterrows():
        name = {"Regime-PCMCI": "Regime-PCMCI", "PCMCI+": "PCMCI$^+$", "MSVAR": "MSAR-X (legacy)"}.get(r["method_b"], r["method_b"])
        ci = f"[{r['bootstrap_ci_low']:.4f}, {r['bootstrap_ci_high']:.4f}]"
        rows.append(
            f"{name} & {r['mean_b']:.3f} & {r['mean_improvement']:+.4f} & {ci} "
            f"& {fmt_p(r['paired_t_greater_p'])} & {fmt_p(r['wilcoxon_greater_p'])} \\\\"
        )
    rows += ["\\bottomrule", "\\end{tabular}"]
    out = TABLES / "synthetic_switching_paired_edge_auc_table.tex"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    return out


# Short display names and failure annotations per config, following
# real_regime_discovery_comparison.md ("Failed fits" / "Not-run plan combos").
CONFIG_LABEL = {
    "HydraulicSystems_cooler_transition": "Hydraulic cooler",
    "HydraulicSystems_cooler_transition_K2present": "Hydraulic cooler, $K{=}2$ present",
    "HydraulicSystems_valve_transition": "Hydraulic valve",
    "HydraulicSystems_valve_transition_K3present": "Hydraulic valve, $K{=}3$ present",
    "AReMActivities_bending_transition": "AReM bending",
    "AReMActivities_concat4": "AReM concat-4",
}
CONFIG_ORDER = list(CONFIG_LABEL)
FAILED = {  # (config, method) -> cell text
    ("HydraulicSystems_cooler_transition", "MSVAR"): "budget$^{\\dagger}$",
    ("HydraulicSystems_valve_transition", "Regime-PCMCI"): "no conv.$^{\\ddagger}$",
    ("HydraulicSystems_valve_transition_K3present", "Regime-PCMCI"): "no conv.$^{\\ddagger}$",
}


def build_real_regime():
    df = pd.read_csv(TABLES / "real_regime_discovery_comparison.csv")
    methods = ["STCG-v1", "Regime-PCMCI", "MSVAR"]
    rows = ["\\begin{tabular}{lccc}",
            "\\toprule",
            "Segment ($K$ given; present) & STCG-v1 & Regime-PCMCI & MSAR-X \\\\",
            "\\midrule"]
    for cfg in CONFIG_ORDER:
        sub = df[df["config"] == cfg]
        k_given = int(sub["oracle_k"].iloc[0])
        k_present = int(sub["k_present"].iloc[0])
        vals = {}
        cells = []
        for m in methods:
            s = sub[sub["method"] == m]
            if len(s):
                r = s.iloc[0]
                vals[m] = float(r["window_ari_mean"])
                cells.append(f"{r['window_ari_mean']:.3f} ({r['window_aligned_accuracy_mean']:.3f})")
            else:
                cells.append(FAILED.get((cfg, m), "---"))
        # bold the best numeric ARI in the row
        if vals:
            best = max(vals, key=vals.get)
            if vals[best] > 0:
                i = methods.index(best)
                s = sub[sub["method"] == best].iloc[0]
                cells[i] = f"\\textbf{{{s['window_ari_mean']:.3f}}} ({s['window_aligned_accuracy_mean']:.3f})"
        label = CONFIG_LABEL[cfg]
        rows.append(f"{label} ({k_given}; {k_present}) & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    out = TABLES / "real_regime_discovery_comparison_table.tex"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    return out


def build_knob_ablation():
    gates = pd.read_csv(TABLES / "stcg_knob_ablation_gates_summary.csv")
    aggs = pd.read_csv(TABLES / "stcg_knob_ablation_aggregators_summary.csv")
    feats = pd.read_csv(TABLES / "stcg_knob_ablation_features_summary.csv")
    wins = pd.read_csv(TABLES / "stcg_knob_ablation_windows_summary.csv")
    min_improvement = pd.read_csv(TABLES / "stcg_knob_ablation_gates_detail.csv")["state_improvement"].min()

    def rng(df):
        return df["edge_auc_mean"].min(), df["edge_auc_mean"].max()

    g_lo, g_hi = rng(gates)
    a_lo, a_hi = rng(aggs)
    f_lo, f_hi = rng(feats)
    w_lo, w_hi = rng(wins)
    a_def = aggs.loc[aggs["config"] == "default", "edge_auc_mean"].iloc[0]
    f_all = feats.loc[feats["config"] == "features=all"].iloc[0]
    f_def = feats.loc[feats["config"] == "default"].iloc[0]
    rows = [
        "\\begin{tabular}{p{0.16\\linewidth}p{0.30\\linewidth}rp{0.34\\linewidth}}",
        "\\toprule",
        "Knob group & What varies & Edge AUC & Finding \\\\",
        "\\midrule",
        f"Auto-gate thresholds & inertia $0.06$--$0.18$, coherence $0.45$--$0.65$, share $0.05$--$0.15$ "
        f"& {g_lo:.3f}--{g_hi:.3f} & Never binding: weakest per-seed improvement "
        f"$\\approx {min_improvement:.3f}$ still clears $0.18$; $K{{=}}2$ on 20/20 seeds in all 7 configs \\\\",
        f"Aggregators & within-state mean/median/p75/max; cross-state max/mean/p90 "
        f"& {a_lo:.3f}--{a_hi:.3f} & Best tested mean (cross p90, ${a_hi:.3f}$) "
        f"slightly above default (${a_def:.3f}$); no optimality claim \\\\",
        f"State features & non-self vs all scores; $+$ window stats "
        f"& {f_lo:.3f}--{f_hi:.3f} & Actual self-lag features: ARI {f_all['state_ari_mean']:.3f} "
        f"vs default {f_def['state_ari_mean']:.3f}; window stats leave recovery unchanged \\\\",
        f"Window size $h$ & $150/300/450$, stride $h/2$, switch period 300 "
        f"& {w_lo:.3f}--{w_hi:.3f} & $h{{=}}300$ best (0.781) but state ARI collapses at "
        f"$h \\geq 300$ (0.035/0.050 vs 0.333 at $h{{=}}150$): windows straddle switches \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    out = TABLES / "stcg_knob_ablation_table.tex"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    return out


def build_evidence_summary():
    """Keep the compact interpretation table aligned with the audited protocols."""
    rows = [
        "\\begin{tabular}{lll}", "\\toprule", "Probe & Evidence & Reading \\\\", "\\midrule",
        "Switching synthetic & Close STCG/Regime-PCMCI estimates; no equivalence claim & Supplied-count recovery \\\\",
        "Stationary/nonlinear & Global VAR/Granger/PCMCI/cMLP stronger & Boundary condition \\\\",
        "Real regime recovery & Annotated $K$ given to every method; concat-4 favors specialist & Descriptive alignment \\\\",
        "Real forecasting & Dense ridge or persistence strongest; AReM gain unconfirmed & Diagnostic only \\\\",
        "Fixed-origin diagnosis & Compact states trail sensor or graph centroids & No established utility advantage \\\\",
        "Fixed real intervals & Weak alignment; all specialist fits hit fixed budget & Negative boundary evidence \\\\",
        "Transition warning & Purged boundaries and safe null selection; selected horizons help & Bounded pilot \\\\",
        "Gate stress & Fallback matters; stationary and rare-state failures remain & Empirical heuristic \\\\",
        "\\bottomrule", "\\end{tabular}",
    ]
    out = TABLES / "paper_evidence_summary_table.tex"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    return out


if __name__ == "__main__":
    for path in (build_paired_edge_auc(), build_real_regime(), build_knob_ablation(), build_evidence_summary()):
        print(f"wrote {path.relative_to(ROOT)}")
