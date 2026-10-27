"""Verify every number quoted in the revised manuscript against the CSV files.

Each check names the literal string that must appear in ``paper/main.tex`` and
recomputes the value from the experiment output. A check passes only when both
the recomputation matches the expected value and the literal is present, so a
stale table or a typo in the text both fail.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
if (ROOT / "paper/supplement.tex").exists():
    PAPER += "\n" + (ROOT / "paper/supplement.tex").read_text(encoding="utf-8")
RESULTS = ROOT / "results"

checked = 0
failures: list[str] = []


def check(name: str, literal: str, value: float, expected: float, tol: float = 5e-4,
          artifact: str | None = None) -> None:
    global checked
    checked += 1
    present = literal in (PAPER if artifact is None else (ROOT / artifact).read_text(encoding="utf-8"))
    finite = isinstance(value, (int, float)) and not math.isnan(float(value))
    matches = finite and abs(float(value) - expected) <= tol
    if not (present and matches):
        failures.append(
            f"{name}: literal={literal!r} present={present} value={value:.6f} expected={expected:.6f}"
        )


def load(*parts: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS.joinpath(*parts))


# ---------------- aggregation ----------------
agg = load("aggregation", "aggregation_controls_summary.csv")
base = agg[(agg.method != "kmeans_eta") & (agg.method != "oracle_eta")]
lookup = base.set_index(["setting", "method"])

for setting, method, column, literal, expected in [
    ("switching_core", "window_mean", "union_auroc", "$0.804$", 0.804),
    ("switching_core", "window_max", "union_auroc", "$0.804$", 0.804),
    ("switching_core", "bagged_mean", "union_auroc", "$0.800$", 0.800),
    ("switching_core", "random_balanced", "union_auroc", "$0.798$", 0.798),
    ("switching_core", "kmeans_hard", "union_auroc", "$0.801$", 0.801),
    ("switching_core", "kmeans_soft", "union_auroc", "$0.789$", 0.789),
    ("switching_core", "temporal_dp", "union_auroc", "$0.795$", 0.795),
    ("switching_core", "oracle_label", "union_auroc", "$0.814$", 0.814),
    ("switching_core", "oracle_refit", "union_auroc", "$0.844$", 0.844),
    ("switching_core", "kmeans_hard", "regime_auroc", "$0.786$", 0.786),
    ("switching_core", "random_balanced", "regime_auroc", "$0.783$", 0.783),
    ("switching_core", "oracle_label", "regime_auroc", "$0.805$", 0.805),
    ("switching_core", "kmeans_soft", "regime_auroc", "$0.777$", 0.777),
    ("switching_core", "kmeans_hard", "temporal_graph", "$0.776$", 0.776),
    ("switching_core", "random_balanced", "temporal_graph", "$0.773$", 0.773),
    ("switching_core", "temporal_dp", "temporal_graph", "$0.765$", 0.765),
    ("switching_core", "oracle_label", "temporal_graph", "$0.806$", 0.806),
    ("stationary", "window_max", "union_auroc", "$0.893$", 0.893),
    ("stationary", "kmeans_hard", "union_auroc", "$0.908$", 0.908),
]:
    check(
        f"aggregation {setting}/{method}/{column}",
        literal.strip("$") if method == "oracle_refit" and column == "union_auroc" else literal,
        float(lookup.loc[(setting, method), column]),
        expected,
        artifact="results/revision2/tab_aggregation.tex" if method == "oracle_refit" and column == "union_auroc" else None,
    )

switching = base[base.setting == "switching_core"]
non_oracle = switching[~switching.method.str.startswith("oracle")]
method_means = non_oracle.groupby("method").union_auroc.mean()
check("aggregation spread", "$0.015$", float(method_means.max() - method_means.min()), 0.015)

for method, literal, expected in [
    ("window_mean", "$0.795$", 0.795),
    ("window_max", "$0.791$", 0.791),
    ("bagged_mean", "$0.792$", 0.792),
]:
    check(
        f"aggregation static temporal {method}",
        literal,
        float(lookup.loc[("switching_core", method), "temporal_graph"]),
        expected,
    )

detail = load("aggregation", "aggregation_controls.csv")
sw_detail = detail[(detail.setting == "switching_core") & (detail.method == "kmeans_hard")]
union_prev = float(sw_detail.union_prevalence.mean())
regime_prev = float(sw_detail.regime_prevalence.mean())
union_auprc = float(sw_detail.union_auprc.mean())
regime_auprc = float(sw_detail.regime_auprc_macro.mean())
check("union prevalence", "$0.249$", union_prev, 0.249)
check("regime prevalence", "$0.174$", regime_prev, 0.174)
check("union lift", "$2.79\\times$", union_auprc / union_prev, 2.79, 1e-2)
check("regime lift", "$3.38\\times$", regime_auprc / regime_prev, 3.38, 1e-2)
check("normalized AP union", "$0.593$", (union_auprc - union_prev) / (1 - union_prev), 0.593)
check("normalized AP regime", "$0.500$", (regime_auprc - regime_prev) / (1 - regime_prev), 0.500)

eta = load("aggregation", "aggregation_controls.csv")
kmeans_eta = eta[(eta.method == "kmeans_eta") & (eta.setting == "switching_core")]
oracle_eta = eta[(eta.method == "oracle_eta") & (eta.setting == "switching_core")]
for frame, source, column, eta_value, literal, expected in [
    (kmeans_eta, "kmeans", "union_auroc", 0.0, "$0.801$", 0.801),
    (kmeans_eta, "kmeans", "union_auroc", 0.5, "$0.798$", 0.798),
    (kmeans_eta, "kmeans", "regime_auprc_macro", 0.0, "$0.587$", 0.587),
    (kmeans_eta, "kmeans", "regime_auprc_macro", 0.5, "$0.568$", 0.568),
    (kmeans_eta, "kmeans", "state_ari", 0.5, "$-0.041$", -0.041),
    (kmeans_eta, "kmeans", "temporal_graph_auroc", 0.0, "$0.776$", 0.776),
    (kmeans_eta, "kmeans", "temporal_graph_auroc", 0.5, "$0.775$", 0.775),
    (oracle_eta, "oracle", "union_auroc", 0.0, "$0.814$", 0.814),
    (oracle_eta, "oracle", "union_auroc", 0.5, "$0.794$", 0.794),
    (oracle_eta, "oracle", "regime_auprc_macro", 0.0, "$0.639$", 0.639),
    (oracle_eta, "oracle", "regime_auprc_macro", 0.5, "$0.556$", 0.556),
    (oracle_eta, "oracle", "temporal_graph_auroc", 0.0, "$0.806$", 0.806),
    (oracle_eta, "oracle", "temporal_graph_auroc", 0.5, "$0.775$", 0.775),
]:
    value = float(frame[np.isclose(frame.eta, eta_value)][column].mean())
    check(f"eta {source}/{column}@{eta_value}", literal, value, expected)

# ---------------- mechanism grid ----------------
grid = load("mechanism", "mechanism_grid.csv")
core = grid[grid.axis == "core"]
check("core union", "$0.801$", float(core.union_auroc.mean()), 0.801)
check("core regime", "$0.587$", float(core.regime_auprc_macro.mean()), 0.587)


def axis_value(axis: str, level: str, column: str) -> float:
    subset = grid[(grid.axis == axis) & (grid.level.astype(str) == str(level))]
    return float(subset[column].mean())


for axis, level, column, literal, expected in [
    ("switch_period", "150", "union_auroc", "$0.756$", 0.756),
    ("switch_period", "150", "regime_auprc_macro", "$0.465$", 0.465),
    ("switch_period", "150", "window_purity_mean", "$0.500$", 0.500),
    ("switch_period", "900", "window_purity_mean", "$0.929$", 0.929),
    ("switch_period", "900", "union_auroc", "$0.859$", 0.859),
    ("switch_period", "900", "regime_auprc_macro", "$0.718$", 0.718),
    ("overlap", "0.0", "union_auroc", "$0.768$", 0.768),
    ("overlap", "1.0", "union_auroc", "$0.862$", 0.862),
    ("overlap", "0.0", "regime_auprc_macro", "$0.530$", 0.530),
    ("overlap", "1.0", "regime_auprc_macro", "$0.655$", 0.655),
    ("observation_noise", "0.4", "union_auroc", "0.777", 0.777),
    ("observation_noise", "0.4", "regime_auprc_macro", "0.535", 0.535),
    ("edge_prob", "0.1", "union_auroc", "$0.786$", 0.786),
    ("edge_prob", "0.3", "union_auroc", "$0.803$", 0.803),
    ("edge_prob", "0.1", "regime_auprc_macro", "$0.499$", 0.499),
    ("edge_prob", "0.3", "regime_auprc_macro", "$0.666$", 0.666),
]:
    check(f"grid {axis}={level}/{column}", literal, axis_value(axis, level, column), expected)

accepted = axis_value("timesteps", "3000", "states_accepted")
check("T=3000 rejection", "$18$ of $20$", 20.0 - accepted * 20.0, 18.0, 1e-6)

# ---------------- control taxonomy ----------------
taxonomy = load("mechanism", "control_taxonomy_summary.csv").set_index("control")
for control, column, literal, expected in [
    ("lag_shift", "union_auroc_lag", "$0.784$", 0.784),
    ("lag_shift", "union_auroc_pair", "$0.875$", 0.875),
    ("lag_shift", "state_ari", "$0.407$", 0.407),
    ("sign_flip", "regime_auprc", "$0.462$", 0.462),
    ("sign_flip", "state_ari", "$0.153$", 0.153),
    ("same_support", "state_ari", "$-0.041$", -0.041),
    ("mean_only", "union_auroc_lag", "$0.640$", 0.640),
    ("variance_only", "union_auroc_lag", "$0.881$", 0.881),
    ("trend", "union_auroc_lag", "$0.672$", 0.672),
]:
    check(f"taxonomy {control}/{column}", literal, float(taxonomy.loc[control, column]), expected)
for control, expected_dyn, expected_graph in [
    ("same_support", 2, 1),
    ("sign_flip", 2, 1),
    ("variance_only", 2, 1),
    ("trend", 2, 1),
    ("core", 2, 2),
]:
    row = taxonomy.loc[control]
    check(f"counts {control} k_dyn", f"{expected_dyn}", float(row.k_dyn), float(expected_dyn), 1e-9)
    check(f"counts {control} k_graph", f"{expected_graph}", float(row.k_graph), float(expected_graph), 1e-9)

# ---------------- known-K ----------------
known = load("mechanism", "known_k_check.csv")
stationary_one = known[(known.setting == "stationary") & (known.k_candidate == 1)]
stationary_two = known[(known.setting == "stationary") & (known.k_candidate == 2)]
check("knownK stationary ARI", "$1.000$", float(stationary_one.state_ari.mean()), 1.0)
check("knownK stationary accepted", "$1.00$", float(stationary_one.accepted.mean()), 0.0, 1e-9)
check("knownK stationary union", "$0.893$", float(stationary_one.union_auroc.mean()), 0.893)
check("knownK stress ARI", "ARI $0.000$", float(stationary_two.state_ari.mean()), 0.0, 1e-9)
check("knownK stress union", "$0.908$", float(stationary_two.union_auroc.mean()), 0.908)

# ---------------- factorial ----------------
factorial = load("baselines", "refit_factorial.csv")
pooled = factorial.groupby("method").union_auroc.mean()
for method, literal, expected in [
    ("global_var_lasso", "$0.832$", 0.832),
    ("kmeans_no_refit", "$0.831$", 0.831),
    ("kmeans_refit", "$0.813$", 0.813),
    ("pelt_no_refit", "$0.829$", 0.829),
    ("pelt_refit", "$0.793$", 0.793),
]:
    check(f"factorial {method}", literal, float(pooled[method]), expected)
for setting, method, literal, expected in [
    ("stationary", "global_var_lasso", "$0.954$", 0.954),
    ("stationary", "kmeans_no_refit", "$0.908$", 0.908),
    ("state_specific", "global_var_lasso", "$0.801$", 0.801),
    ("state_specific", "kmeans_no_refit", "$0.768$", 0.768),
    ("shared_graph", "global_var_lasso", "$0.768$", 0.768),
    ("shared_graph", "kmeans_no_refit", "$0.847$", 0.847),
]:
    value = float(
        factorial[(factorial.setting == setting) & (factorial.method == method)].union_auroc.mean()
    )
    check(f"factorial {setting}/{method}", literal, value, expected)

pelt_fits = factorial[factorial.method.str.contains("pelt")]
check("PELT one segment per window", "all $160$ fits", float((pelt_fits.n_segments == 7).sum()), 160.0, 1e-9)

# ---------------- segmentation ----------------
seg = load("baselines", "segmentation_comparison_summary.csv").set_index(["setting", "method"])
for setting, method, column, literal, expected in [
    ("switching_core", "kmeans", "partition_ari", "$0.292$", 0.292),
    ("switching_core", "constrained_2seg", "partition_ari", "$-0.061$", -0.061),
    ("switching_core", "raw_var_split", "partition_ari", "$0.045$", 0.045),
    ("switching_core", "oracle_2seg", "partition_ari", "$0.103$", 0.103),
    ("switching_core", "kmeans", "union_auroc", "$0.801$", 0.801),
    ("switching_core", "constrained_2seg", "union_auroc", "$0.795$", 0.795),
    ("switching_core", "oracle_2seg", "union_auroc", "$0.773$", 0.773),
    ("switching_core", "pelt_bic", "union_auroc", "$0.804$", 0.804),
]:
    check(f"segmentation {setting}/{method}/{column}", literal, float(seg.loc[(setting, method), column]), expected)
check("segmentation spread", "$0.031$", 0.804 - 0.773, 0.031)

# ---------------- final hold-out ----------------
final = load("confirmatory", "final_confirmation_tests.csv").set_index(["metric", "comparison"])
for metric, comparison, column, literal, expected in [
    ("union_auroc", "stcg_v1_minus_window_max", "mean_difference", "$+0.0109$", 0.0109),
    ("union_auroc", "stcg_v1_minus_pelt_refit", "mean_difference", "$+0.0382$", 0.0382),
    ("union_auroc", "stcg_v1_minus_constrained_refit", "mean_difference", "$+0.0234$", 0.0234),
    ("union_auprc", "stcg_v1_minus_window_max", "mean_difference", "$+0.0229$", 0.0229),
    ("union_auprc", "stcg_v1_minus_pelt_refit", "mean_difference", "$+0.0830$", 0.0830),
    ("union_auprc", "stcg_v1_minus_constrained_refit", "mean_difference", "$+0.0398$", 0.0398),
    ("temporal_graph_auroc", "stcg_v1_minus_pelt_refit", "mean_difference", "$+0.102$", 0.1015),
    ("temporal_graph_auroc", "stcg_v1_minus_constrained_refit", "mean_difference", "$+0.042$", 0.0422),
    ("temporal_graph_auroc", "stcg_v1_minus_window_max", "mean_difference", "$+0.0021$", 0.0021),
    ("temporal_graph_auroc", "stcg_v1_minus_global_var_lasso", "mean_difference", "$-0.0001$", -0.0001),
    ("union_auroc", "stcg_v1_minus_global_var_lasso", "mean_difference", "$+0.0006$", 0.0006),
    ("union_auroc", "stcg_v1_minus_global_var_lasso", "ci95_low", "$[-0.0061,0.0071]$", -0.0061),
    ("union_auprc", "stcg_v1_minus_global_var_lasso", "mean_difference", "$0.0206$", -0.0206),
]:
    check(f"final {metric}/{comparison}/{column}", literal, float(final.loc[(metric, comparison), column]), expected, 1e-3)

check(
    "final temporal window max p_holm",
    "Holm $p=0.42$",
    float(final.loc[("temporal_graph_auroc", "stcg_v1_minus_window_max"), "p_holm"]),
    0.42,
    1e-2,
)

# ---------------- theory ----------------
bias = load("theory", "theory_max_bias.csv").set_index("mode")
check("toy shared mean", "$0.933$", float(bias.loc["shared", "mean_auc"]), 0.933)
check("toy shared max", "$0.901$", float(bias.loc["shared", "max_auc"]), 0.901)
check("toy specific mean", "$0.774$", float(bias.loc["state_specific", "mean_auc"]), 0.774)
check("toy specific max", "$0.803$", float(bias.loc["state_specific", "max_auc"]), 0.803)
mse = load("theory", "theory_mse_comparison.csv")
check("theory grouped error", "$0.0015$", float(mse.group_formula_error.max()), 0.0015, 5e-5)
check("theory pooled error", "$0.0003$", float(mse.pool_formula_error.max()), 0.0003, 5e-5)

# ---------------- stability ----------------
stability = load("mechanism", "generator_stability_summary.csv").iloc[0]
check("stability max", "$0.536$", float(stability.max_abs_max), 0.536, 1e-3)
check("stability ratio", "$4.97$", float(stability.max_over_std_max), 4.97, 1e-2)
check("stability variance", "$1.10$", float(stability.variance_ratio_p95), 1.10, 1e-2)

# ---------------- ESS ----------------
ess = load("tables", "ess_alignment.csv")
ess_main = ess[ess.imputation == "ffill"].set_index(["method", "binarisation"])
for method, binarisation, column, literal, expected in [
    ("stcg_v1", "all_edges", "auroc", "$0.571$", 0.571),
    ("stcg_v1", "strong_only", "auroc", "$0.587$", 0.587),
    ("pelt_var_lasso", "all_edges", "auroc", "$0.565$", 0.565),
    ("pelt_var_lasso", "strong_only", "auroc", "$0.555$", 0.555),
    ("lag_corr", "all_edges", "auroc", "$0.554$", 0.554),
    ("lag_corr", "strong_only", "auroc", "$0.677$", 0.677),
    ("pelt_var_lasso", "strong_only", "p_raw", "$p=0.1278$", 0.1278),
    ("pelt_var_lasso", "strong_only", "p_holm", "$0.4047$", 0.4047),
    ("lag_corr", "strong_only", "p_raw", "$p=0.0032$", 0.0032),
    ("lag_corr", "strong_only", "p_holm", "$0.0256$", 0.0256),
    ("stcg_v1", "all_edges", "p_raw", "$p=0.0156$", 0.0156),
    ("stcg_v1", "all_edges", "p_holm", "$0.1092$", 0.1092),
]:
    check(f"ess {method}/{binarisation}/{column}", literal, float(ess_main.loc[(method, binarisation), column]),
          expected, 5e-4 if column == "auroc" else 5e-5)
stcg_range = ess[(ess.method == "stcg_v1") & (ess.binarisation == "all_edges")]
stcg_ranges = ess[ess.method == "stcg_v1"].groupby("binarisation").auroc.agg(lambda s: s.max() - s.min())
check("ess imputation range", "$0.069$", float(stcg_ranges.max()), 0.069, 1e-2)
ess_seeds = load("tables", "ess_alignment_seed_sensitivity.csv")
check("ess null draws", "$5{,}000$", float(ess_seeds.null_draws.min()), 5000)
check("ess null seed count", "seeds $0$--$4$", float(ess_seeds.null_seed.nunique()), 5)
lag_strong = ess_seeds[(ess_seeds.method == "lag_corr") & (ess_seeds.binarisation == "strong_only")]
lag_primary = lag_strong[lag_strong.imputation == "ffill"]
check("ess primary minimum Holm", "$[0.0160,0.0368]$", float(lag_primary.p_holm.min()), 0.0160, 5e-5)
check("ess primary maximum Holm", "$[0.0160,0.0368]$", float(lag_primary.p_holm.max()), 0.0368, 5e-5)
check("ess primary seed stability", "all $5/5$", float((lag_primary.p_holm < 0.05).sum()), 5)
linear_strong = lag_strong[lag_strong.imputation == "linear"]
mean_strong = lag_strong[lag_strong.imputation == "mean"]
check("ess combined reference", "$0.0448$", float(linear_strong[linear_strong.null_seed == 0].p_holm_all_imputations.iloc[0]), 0.0448, 5e-5)
check("ess combined minimum", "$[0.0192,0.0512]$", float(linear_strong.p_holm_all_imputations.min()), 0.0192, 5e-5)
check("ess combined maximum", "$[0.0192,0.0512]$", float(linear_strong.p_holm_all_imputations.max()), 0.0512, 5e-5)
check("ess combined linear stability", "$4/5$", float((linear_strong.p_holm_all_imputations < 0.05).sum()), 4)
check("ess combined mean stability", "$1/5$", float((mean_strong.p_holm_all_imputations < 0.05).sum()), 1)

# ---------------- cost ----------------
detail = load("confirmatory", "confirmatory_detail.csv")
seconds = detail.groupby("method").seconds.mean()
check("cost stcg", "$0.051$~s", float(seconds["stcg_v1"]), 0.051)
check("cost pelt refit", "$0.063$~s", float(seconds["pelt_var_lasso"]), 0.063)

# ---------------- real-data boundary check ----------------
real_h = pd.read_csv(RESULTS / "tables" / "real_change_point_gate_hydraulic_summary.csv").set_index("kind")
check("hydraulic boundary", "$34.2\\%$", float(real_h.loc["boundary", "change_point_accept"]) * 100, 34.2, 0.05)
check("hydraulic stable", "$30.0\\%$", float(real_h.loc["stable", "change_point_accept"]) * 100, 30.0, 0.05)

# ---------------- changing-edge temporal AUROC and static ceiling ----------------
td = load("revision2", "temporal_delta_summary.csv").set_index("method")
for method, literal, expected in [
    ("kmeans_hard", "$0.649$", 0.649),
    ("random_balanced", "$0.570$", 0.570),
    ("oracle_label", "$0.814$", 0.814),
]:
    check(f"temporal-delta {method}", literal, float(td.loc[method, "temporal_delta_auroc"]), expected, 1e-3)
dp_value = float(td.loc["temporal_dp", "temporal_delta_auroc"])
dp_display = f"{dp_value:.3f}"
dp_pooled_display = f"{float(td.loc['temporal_dp', 'pooled_temporal_auroc']):.3f}"
check("temporal-delta temporal_dp displayed rounding", f"Temporal DP & {dp_pooled_display} & {dp_display}",
      dp_value, float(dp_display), 5e-4, artifact="results/revision2/tab_temporal_delta.tex")
check("temporal-delta edges", "$27.3$", float(td.loc["kmeans_hard", "temporal_delta_edges"]), 27.3, 1e-2)
check("static ceiling", "$0.979$", float(td.loc["kmeans_hard", "static_ceiling"]), 0.979, 1e-3)
check("temporal-delta raw window", "0.716", float(td.loc["raw_window", "temporal_delta_auroc"]), 0.716, 1e-3)

td_detail = load("revision2", "temporal_delta_detail.csv")
td_pivot = td_detail.pivot(index="seed", columns="method", values="temporal_delta_auroc")
from scipy import stats as _stats  # noqa: E402
kmeans_vs_random = _stats.ttest_1samp(td_pivot["kmeans_hard"] - td_pivot["random_balanced"], 0.0)
check("temporal-delta kmeans-vs-random p", "p{=}0.016", float(kmeans_vs_random.pvalue), 0.016, 5e-3)
kmeans_vs_raw = _stats.ttest_1samp(td_pivot["kmeans_hard"] - td_pivot["raw_window"], 0.0)
check("temporal-delta kmeans-vs-raw p", "p{=}0.03", float(kmeans_vs_raw.pvalue), 0.03, 5e-3)

# ---------------- state decomposition (J x R_edge, pooled same/cross split) ----------------
sd = load("revision2", "state_decomposition.csv")
check("state J", "0.533", float(sd.J.mean()), 0.533, 1e-3, artifact="results/revision2/tab_grouping_factors.tex")
check("state R_edge", "0.414", float(sd.R_edge.mean()), 0.414, 1e-3, artifact="results/revision2/tab_grouping_factors.tex")
check("core mean product", r"$\mathbb E[JC_{\mathrm{edge}}]=0.298$", float(sd.J_times_R.mean()), 0.298, 5e-4)
check("pooled omega", "0.0017", float(sd.weight_same.mean()), 0.0017, 2e-4)
check("identity gap", "10^{-16}", float(sd.identity_gap.max()), 0.0, 1e-12)
check("conditional diagnostic", "0.707", 0.5 + 0.5 * float(sd.R_edge.mean()), 0.707, 1e-3)

# ---------------- ESS reference-chain diagnostics ----------------
diagnostics = pd.read_csv(RESULTS / "tables/ess_null_diagnostics.csv")
reference_tail = diagnostics[(diagnostics.binarisation == "strong_only") &
                             (diagnostics.null_seed == 0) & (diagnostics.burn_in_per_edge == 100)].iloc[0]
long_tail = diagnostics[(diagnostics.binarisation == "strong_only") &
                        (diagnostics.null_seed == 0) & (diagnostics.burn_in_per_edge == 1000)].iloc[0]
check("ESS reference tail count", "$b=15$", float(reference_tail.null_exceedances), 15)
check("ESS reference tail ESS", "$5{,}000$", float(reference_tail.tail_ess), 5000)
check("ESS reference tail MCSE", "$0.00077$", float(reference_tail.tail_mcse), 0.00077, 5e-6)
check("ESS reference tail batch MCSE", "$0.00070$", float(reference_tail.tail_batch_mcse), 0.00070, 5e-6)
check("ESS longer burn raw p", "$p=0.0022$", float(long_tail.p_raw), 0.0022, 5e-5)

# ---------------- gate bipartition enumeration ----------------
gate = load("revision2", "gate_enumeration.csv")
check("gate optimal", "$0.221$", float(gate.delta_optimal.mean()), 0.221, 1e-3)
check("gate code", "$0.219$", float(gate.delta_code.mean()), 0.219, 1e-3)
check("gate gap", "$0.014$", float(gate.gap_optimal_minus_code.max()), 0.014, 1e-3)
check("gate all accepted", "all $20/20$", float(gate.code_accepted.sum()), 20.0)

# ---------------- window mixing contrast ----------------
wc = load("revision2", "window_contrast.csv").set_index("switch_period")
check("window theory signal D=300", "$0.038$", float(wc.loc[300, "theory_signal"]), 0.038, 1e-3)
check("window theory signal D=150", "$0.001$", float(wc.loc[150, "theory_signal"]), 0.001, 1e-3)
check("window noise", "$0.060$", float(wc.loc[300, "est_noise"]), 0.060, 1e-3)
check("window effective rows", "$297$", float(wc.loc[300, "effective_rows_per_window"]), 297.0)
check("window min frac", "$0.16$", float(wc.loc[300, "min_state_frac_min"]), 0.16, 1e-2)
check("window max frac", "$0.34$", float(wc.loc[300, "min_state_frac_max"]), 0.34, 1e-2)

# ---------------- denoising intervention ----------------
dn = load("revision2", "denoising_intervention.csv").set_index("B")
check("denoise B=1", "$0.792$", float(dn.loc[1, "t_auc_delta"]), 0.792, 1e-3)
check("denoise B=16", "to $1.0$", float(dn.loc[16, "t_auc_delta"]), 1.0, 1e-3)

# ---------------- untouched-block replication (seeds 3000--3049) ----------------
td3000 = load("revision2", "temporal_delta_detail_3000.csv")
td3000_pivot = td3000.pivot(index="seed", columns="method", values="temporal_delta_auroc")
raw_minus_kmeans_3000 = td3000_pivot["raw_window"] - td3000_pivot["kmeans_hard"]
check("replication raw-minus-kmeans", "0.053", float(raw_minus_kmeans_3000.mean()), 0.053, 2e-3)

# ---------------- exploratory temporal-grid extension ----------------
grid = load("revision2", "temporal_grid_summary.csv")
all_grid = grid[grid.subset == "all"]
for axis, level, method, field, literal, expected in [
    ("switch_period", "600", "kmeans_hard", "delta_temporal_mean", "$+0.031$", 0.031),
    ("switch_period", "600", "raw_window", "temporal_delta_auroc_mean", "$0.767$", 0.767),
    ("switch_period", "600", "kmeans_hard", "temporal_delta_auroc_mean", "$0.798$", 0.798),
    ("switch_period", "600", "kmeans_hard", "delta_temporal_ci95_low", "$[0.005,0.057]$", 0.005),
    ("switch_period", "600", "kmeans_hard", "delta_temporal_ci95_high", "$[0.005,0.057]$", 0.057),
    ("switch_period", "900", "kmeans_hard", "delta_temporal_mean", "$+0.017$", 0.017),
    ("switch_period", "150", "raw_window", "temporal_delta_auroc_mean", "$0.517$", 0.517),
    ("switch_period", "150", "kmeans_hard", "delta_temporal_mean", "$-0.012$", -0.012),
    ("switch_period", "150", "oracle_label", "delta_temporal_mean", "$+0.029$", 0.029),
    ("timesteps", "600", "kmeans_hard", "delta_temporal_mean", "$-0.017$", -0.017),
    ("timesteps", "3000", "kmeans_hard", "delta_temporal_mean", "$-0.202$", -0.202),
    ("timesteps", "3000", "kmeans_hard", "candidate_delta_temporal_mean", "$-0.042$", -0.042),
    ("timesteps", "3000", "kmeans_hard", "candidate_delta_temporal_ci95_low", "$[-0.089,0.006]$", -0.089),
    ("timesteps", "3000", "kmeans_hard", "candidate_delta_temporal_ci95_high", "$[-0.089,0.006]$", 0.006),
    ("core", "core", "gmm_soft", "temporal_delta_auroc_mean", "$0.608$", 0.608),
    ("core", "core", "gmm_soft", "delta_temporal_mean", "$-0.109$", -0.109),
]:
    row = all_grid[(all_grid.axis == axis) & (all_grid.level == level) & (all_grid.method == method)].iloc[0]
    check(f"temporal grid {axis}/{level}/{method}/{field}", literal, float(row[field]), expected)
accepted_grid = grid[(grid.axis == "timesteps") & (grid.level == "3000") &
                     (grid.subset == "hard_gate_accepted") & (grid.method == "kmeans_hard")].iloc[0]
check("temporal accepted M19 mean", "$-0.046$", float(accepted_grid.delta_temporal_mean), -0.046)
check("temporal accepted M19 raw", "$0.742$", float(accepted_grid.raw_temporal_delta_auroc_mean), 0.742)
check("temporal accepted M19 n", "$2/20$", int(accepted_grid.n_evaluable), 2)
hard19 = all_grid[(all_grid.axis == "timesteps") & (all_grid.level == "3000") & (all_grid.method == "kmeans_hard")].iloc[0]
check("temporal fallback contribution", "$-0.160$", float(hard19.delta_temporal_mean - hard19.candidate_delta_temporal_mean), -0.160)
detail_grid = load("revision2", "temporal_grid_detail.csv")
check("temporal distinct fits", "$620$", len(detail_grid[["config_id", "seed"]].drop_duplicates()), 620)
check("temporal no target contexts", "$240$", int((~detail_grid[detail_grid.method == "raw_window"].evaluable).sum()), 240)
check("temporal GMM maximum entropy", r"$2.15\times10^{-7}$", float(detail_grid.gmm_entropy_normalized.max()), 2.15e-7, 5e-10)
numerics = load("revision2", "temporal_grid_gmm_numerics.csv")
soft19 = numerics[(numerics.axis == "timesteps") & (numerics.level == "3000") & (numerics.method == "gmm_soft")]
check("temporal GMM M19 native", "$0.619735$", float(soft19.native_temporal_delta_auroc.mean()), 0.619735, 5e-7)
check("temporal GMM M19 rounded", "$0.619162$", float(soft19.rounded_temporal_delta_auroc.mean()), 0.619162, 5e-7)
check("temporal GMM max rounding change", "$0.01434$", float((numerics.rounded_temporal_delta_auroc - numerics.native_temporal_delta_auroc).abs().max()), 0.01434, 5e-6)

# ---------------- selected-cell matched control and three-condition factors ----------------
matched = load("revision2", "matched_random_contrasts.csv").set_index("contrast")
matched_seeds = load("revision2", "matched_random_seed_summary.csv")
check("matched random mean", "$0.577$", float(matched_seeds.matched_random_temporal_auroc.mean()), 0.577)
for metric, column, literal, expected in [
    ("hard_minus_matched_random", "mean", "$+0.222$", 0.222),
    ("hard_minus_matched_random", "ci95_low", "$[0.193,0.250]$", 0.193),
    ("hard_minus_matched_random", "ci95_high", "$[0.193,0.250]$", 0.250),
    ("matched_random_minus_raw", "mean", "$-0.191$", -0.191),
    ("matched_random_minus_raw", "ci95_low", "$[-0.204,-0.179]$", -0.204),
    ("matched_random_minus_raw", "ci95_high", "$[-0.204,-0.179]$", -0.179),
]:
    check(f"matched {metric}/{column}", literal, float(matched.loc[metric, column]), expected)
check("matched partition count", "$616$", int(matched_seeds.n_partitions.sum()), 616)
factors = load("revision2", "grouping_state_factors_summary.csv").set_index("condition")
for condition, metric, expected in [("duration600", "J", 0.867), ("duration600", "C_edge", 0.686),
                                    ("duration600", "J_times_C", 0.596), ("long300", "J", 0.537),
                                    ("long300", "C_edge", 0.581), ("long300", "J_times_C", 0.359),
                                    ("long300", "hard_temporal_auroc", 0.679), ("long300", "raw_temporal_auroc", 0.722)]:
    check(f"factor table {condition}/{metric}", f"{expected:.3f}", float(factors.loc[condition, metric]), expected,
          artifact="results/revision2/tab_grouping_factors.tex")

# The connection to gain is an algebraic restatement on the same seed set.
factor_detail = load("revision2", "grouping_state_factors_detail.csv")
seed_sets = [set(group.seed) for _, group in factor_detail.groupby("condition")]
assert all(values == set(range(20)) for values in seed_sets)
for row in factors.itertuples():
    gain = row.hard_temporal_auroc - row.raw_temporal_auroc
    connected = 0.5 * (row.J_times_C - (2 * row.raw_temporal_auroc - 1))
    check(f"factor gain identity {row.Index}", r"\label{eq:temporal-gain-factor}",
          float(gain - connected), 0.0, tol=1e-14)
hard_change = float(factors.loc["duration600", "hard_temporal_auroc"] - factors.loc["core300", "hard_temporal_auroc"])
raw_change = float(factors.loc["duration600", "raw_temporal_auroc"] - factors.loc["core300", "raw_temporal_auroc"])
check("duration descriptive hard change", "$0.149$", hard_change, 0.149)
check("duration descriptive raw change", "$0.051$", raw_change, 0.051)
check("duration descriptive gain change", "$0.098$", hard_change - raw_change, 0.098)

print(f"checked {checked} numbers")
if failures:
    print(f"{len(failures)} FAILURES")
    for failure in failures:
        print(" -", failure)
    sys.exit(1)
print("ALL NUMBERS MATCH")
