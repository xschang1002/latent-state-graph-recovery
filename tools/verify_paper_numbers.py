"""Verify that key numbers quoted in the archived pre-revision manuscript match the tracked CSVs.

Each check pairs (a) a literal string that must appear in the archived manuscript with
(b) a recomputation of the same value from the tracked results/tables/*.csv
files. A check FAILs if the string is absent from the tex or the recomputed
value formats differently. The tex is always fixed to match the data, never
the reverse.

Run from the repo root:

    .venv/Scripts/python.exe tools/verify_paper_numbers.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEX = (ROOT / "paper" / "legacy" / "main_pre_revision.tex").read_text(encoding="utf-8")
T = ROOT / "results" / "tables"

failures = []
checked = 0


def check(desc, tex_snippet, recomputed, expected):
    """tex_snippet: literal that must appear in main.tex;
    recomputed: value computed from CSVs; expected: same value as string."""
    global checked
    checked += 1
    ok_tex = tex_snippet in TEX
    ok_val = str(recomputed) == str(expected)
    if not (ok_tex and ok_val):
        failures.append(
            f"{desc}: tex_present={ok_tex} recomputed={recomputed!r} expected={expected!r}"
            + ("" if ok_tex else f" [missing snippet: {tex_snippet[:60]}...]")
        )


def f3(x):
    return f"{x:.3f}"


def load(name):
    return pd.read_csv(T / name)


# ---------------- synthetic all-methods ----------------
am = load("synthetic_all_methods_summary.csv")


def am_val(gen, method, col):
    return float(am[(am.generator == gen) & (am.method == method)][col].iloc[0])


check("switch STCG-v1 Edge AUC", "$0.812$ versus $0.816$",
      f3(am_val("switching_var", "STCG-v1", "edge_auc_mean")), "0.812")
check("switch RPCMCI Edge AUC", "$0.812$ versus $0.816$",
      f3(am_val("switching_var", "Regime-PCMCI", "edge_auc_mean")), "0.816")
check("var VAR-Lasso", "VAR-Lasso ($0.924$)", f3(am_val("var", "VAR-Lasso", "edge_auc_mean")), "0.924")
check("var PCMCI+", "PCMCI$^+$ ($0.921$)", f3(am_val("var", "PCMCI+", "edge_auc_mean")), "0.921")
check("var PCMCI-ParCorr", "PCMCI-ParCorr ($0.920$)", f3(am_val("var", "PCMCI-ParCorr", "edge_auc_mean")), "0.920")
check("var cMLP", "cMLP ($0.918$)", f3(am_val("var", "cMLP", "edge_auc_mean")), "0.918")
check("var STCG", "exceed STCG-v1 ($0.876$)", f3(am_val("var", "STCG-v1", "edge_auc_mean")), "0.876")
check("var RPCMCI", "$0.809$ on stationary", f3(am_val("var", "Regime-PCMCI", "edge_auc_mean")), "0.809")
check("nl PCMCI-ParCorr", "PCMCI-ParCorr ($0.984$)", f3(am_val("nonlinear_var", "PCMCI-ParCorr", "edge_auc_mean")), "0.984")
check("nl Granger-F", "Granger-F ($0.981$)", f3(am_val("nonlinear_var", "Granger-F", "edge_auc_mean")), "0.981")
check("nl cMLP", "cMLP ($0.980$)", f3(am_val("nonlinear_var", "cMLP", "edge_auc_mean")), "0.980")
check("nl STCG", "exceed STCG-v1 ($0.969$)", f3(am_val("nonlinear_var", "STCG-v1", "edge_auc_mean")), "0.969")
check("nl RPCMCI", "$0.767$ on nonlinear", f3(am_val("nonlinear_var", "Regime-PCMCI", "edge_auc_mean")), "0.767")
check("nl RPCMCI n=19", "$n{=}19$", int(am[(am.generator == "nonlinear_var") & (am.method == "Regime-PCMCI")]["n"].iloc[0]), 19)

# fit times
rpc = [am_val(g, "Regime-PCMCI", "fit_seconds_mean") for g in ("var", "nonlinear_var", "switching_var")]
msv = [am_val(g, "MSVAR", "fit_seconds_mean") for g in ("var", "nonlinear_var", "switching_var")]
check("RPCMCI fit range", "$164.8$--$204.7$", f"{min(rpc):.1f}--{max(rpc):.1f}", "164.8--204.7")
check("MSVAR fit range", "$133.0$--$170.7$", f"{min(msv):.1f}--{max(msv):.1f}", "133.0--170.7")
check("cMLP fit", "cMLP ($0.6$ s)", f"{am_val('switching_var', 'cMLP', 'fit_seconds_mean'):.1f}", "0.6")

# ---------------- paired tests (switching matrix) ----------------
pm = load("synthetic_switching_paired_matrix.csv")


def pm_val(opp, metric, col):
    return float(pm[(pm.metric == metric) & (pm.method_b == opp)][col].iloc[0])


check("RPCMCI edge improvement", "$-0.0031$", f"{pm_val('Regime-PCMCI','edge_auc','mean_improvement'):.4f}", "-0.0031")
check("RPCMCI edge CI", "[-0.0165, 0.0110]",
      f"[{pm_val('Regime-PCMCI','edge_auc','bootstrap_ci_low'):.4f}, {pm_val('Regime-PCMCI','edge_auc','bootstrap_ci_high'):.4f}]",
      "[-0.0165, 0.0110]")
check("RPCMCI edge p_t", "$p=0.666$", f"{pm_val('Regime-PCMCI','edge_auc','paired_t_greater_p'):.3f}", "0.666")
check("RPCMCI edge p_W", "$p=0.715$", f"{pm_val('Regime-PCMCI','edge_auc','wilcoxon_greater_p'):.3f}", "0.715")
for metric, p in [("pair_auc", "0.305"), ("precision_at_k", "0.149"), ("shd", "0.180"), ("lag_accuracy", "0.792")]:
    check(f"RPCMCI {metric} p", f"$p={p}$", f"{pm_val('Regime-PCMCI',metric,'paired_t_greater_p'):.3f}", p)
check("DynVAR edge improvement", "$+0.0143$", f"{pm_val('DynVAR-Lasso','edge_auc','mean_improvement'):+.4f}", "+0.0143")
check("PCMCI edge improvement", "$+0.0334$", f"{pm_val('PCMCI-ParCorr','edge_auc','mean_improvement'):+.4f}", "+0.0334")
# Exclude the misidentified historical MSAR-X control from validated references.
others = pm[(pm.metric == "edge_auc") & (~pm.method_b.isin(["Regime-PCMCI", "MSVAR"]))]
maxp = max(others["paired_t_greater_p"].max(), others["wilcoxon_greater_p"].max())
check("13 reference opponents significant", "$p \\leq 0.0014$", len(others), 13)
assert maxp <= 0.0014, f"max opponent p {maxp}"

# ---------------- oracle ----------------
orc = load("synthetic_state_oracle_summary.csv")


def orc_val(method, col):
    return float(orc[orc.method == method][col].iloc[0])


check("oracle edge", "oracle $0.813$", f3(orc_val("Oracle-state upper bound", "edge_auc_mean")), "0.813")
check("oracle pk", "oracle $0.672$", f3(orc_val("Oracle-state upper bound", "precision_at_k_mean")), "0.672")
check("nostate edge", "no-state $0.798$", f3(orc_val("STCG-v1/no-state", "edge_auc_mean")), "0.798")
check("nostate pk", "no-state $0.646$", f3(orc_val("STCG-v1/no-state", "precision_at_k_mean")), "0.646")

# ---------------- state diagnostics / robustness / difficulty ----------------
diag = load("stcg_v1_state_diagnostics_summary.csv")
check("diag ARI", "$0.595 \\pm 0.370$", f"{diag['adjusted_rand'].mean():.3f}", "0.595")
check("diag ARI std", "$0.595 \\pm 0.370$", f"{diag['adjusted_rand'].std(ddof=1):.3f}", "0.370")
check("diag acc", "$0.900 \\pm 0.091$", f"{diag['aligned_accuracy'].mean():.3f}", "0.900")
check("diag acc std", "$0.900 \\pm 0.091$", f"{diag['aligned_accuracy'].std(ddof=1):.3f}", "0.091")

rob = load("stcg_v1_robustness_summary.csv")
best = 0
for (sweep, val), grp in rob.groupby(["axis", "value"]):
    top = grp.loc[grp["edge_auc_mean"].idxmax(), "method"]
    best += (top == "STCG-v1")
check("robustness 10/12", "10 of 12 sweep settings", f"{best} of {rob.groupby(['axis','value']).ngroups}", "10 of 12")

sd = load("synthetic_switch_difficulty_summary.csv")


def sd_val(period, method, col):
    return float(sd[(sd.switch_period == period) & (sd.method == method)][col].iloc[0])


check("difficulty 150 tie", "($0.733$)", f3(sd_val(150, "STCG-v1", "edge_auc_mean")), "0.733")
check("difficulty 150 purity", "$0.500$", f3(sd_val(150, "Oracle-state upper bound", "oracle_window_purity_mean")), "0.500")
check("difficulty 600 stcg", "$0.847$", f3(sd_val(600, "STCG-v1", "edge_auc_mean")), "0.847")
check("difficulty 600 oracle", "$0.849$", f3(sd_val(600, "Oracle-state upper bound", "edge_auc_mean")), "0.849")
check("difficulty 600 dynvar", "$0.835$", f3(sd_val(600, "DynVAR-Lasso", "edge_auc_mean")), "0.835")
check("difficulty 600 purity", "$0.929$", f3(sd_val(600, "Oracle-state upper bound", "oracle_window_purity_mean")), "0.929")

# RPCMCI synthetic regime ARI 0.695
rbase = load("synthetic_rpcmci_baseline_summary.csv")
reg_ari = float(rbase[(rbase.generator == "switching_var") & (rbase.method == "Regime-PCMCI")]["regime_ari_mean"].iloc[0])
check("RPCMCI regime ARI", "regime ARI ($0.695$", f3(reg_ari), "0.695")

# ---------------- real regime comparison ----------------
rr = load("real_regime_discovery_comparison.csv")
rrd = load("real_regime_discovery_comparison_detail.csv")


def rr_val(cfg, method, col):
    return float(rr[(rr.config == cfg) & (rr.method == method)][col].iloc[0])


check("cooler STCG ARI", "ARI $0.822$", f3(rr_val("HydraulicSystems_cooler_transition_K2present", "STCG-v1", "window_ari_mean")), "0.822")
check("cooler STCG acc", "$0.956$", f3(rr_val("HydraulicSystems_cooler_transition_K2present", "STCG-v1", "window_aligned_accuracy_mean")), "0.956")
check("concat4 RPCMCI", "$0.571$ against STCG-v1's $0.094$",
      f3(rr_val("AReMActivities_concat4", "Regime-PCMCI", "window_ari_mean")), "0.571")
check("concat4 STCG", "$0.571$ against STCG-v1's $0.094$",
      f3(rr_val("AReMActivities_concat4", "STCG-v1", "window_ari_mean")), "0.094")
msvar_bend = rr_val("AReMActivities_bending_transition", "MSVAR", "fit_seconds_mean")
check("MSVAR bending ~260s", "about $260$ s per fit", round(msvar_bend / 10) * 10, 260)
rpc_fit = rrd[rrd.method == "Regime-PCMCI"]["fit_seconds"]
stcg_fit = rrd[rrd.method == "STCG-v1"]["fit_seconds"]
# tex claims per-fit ranges over the detail rows (incl. failed attempts)
check("RPCMCI real fit range", "$46$--$98$ s for Regime-PCMCI",
      f"{rpc_fit.min():.0f}--{rpc_fit.max():.0f}", "46--98")
check("STCG real fit range", "$0.01$--$0.08$ s for STCG-v1",
      f"{stcg_fit.min():.2f}--{stcg_fit.max():.2f}", "0.01--0.08")

# concat-4 error-free annealings 2/10
efa = rrd[(rrd.config == "AReMActivities_concat4") & (rrd.method == "Regime-PCMCI")]["error_free_annealings"]
check("concat4 2/10 annealings", "2 of its 10 annealings", f"{int(efa.iloc[0])} of 10", "2 of 10")

# ---------------- real alignment ----------------
hyd = load("HydraulicSystems_state_label_alignment_summary.csv")
check("Hydraulic ARI", "ARI $0.735 \\pm 0.006$", f"{hyd['adjusted_rand'].mean():.3f}", "0.735")
check("Hydraulic ARI std", "ARI $0.735 \\pm 0.006$", f"{hyd['adjusted_rand'].std(ddof=1):.3f}", "0.006")
check("Hydraulic acc", "$0.902 \\pm 0.002$", f"{hyd['aligned_accuracy'].mean():.3f}", "0.902")

sel = load("HydraulicSystems_cooler_selectK_state_count_selection_summary.csv")
sel_k3 = sel[sel["selected_count"] == 5].iloc[0]
check("Hydraulic selectK K=3", "chooses $K{=}3$ on all five seeds", int(sel_k3["candidate_state_count"]), 3)
check("Hydraulic selectK BA", "$0.948 \\pm 0.004$", f3(float(sel_k3["balanced_accuracy_mean"])), "0.948")

hpred = load("HydraulicSystems_state_label_prediction_summary.csv")
check("Hydraulic centroid", "$0.959 \\pm 0.003$",
      f3(float(hpred[hpred.method == "GraphFeature Centroid"]["balanced_accuracy_mean"].iloc[0])), "0.959")
check("Hydraulic ceiling", "$0.994 \\pm 0.004$",
      f3(float(hpred[hpred.method == "SensorMeanStd Centroid"]["balanced_accuracy_mean"].iloc[0])), "0.994")

fwd = load("HydraulicSystems_forward_state_label_prediction_forward_summary.csv")
check("forward cooler", "$0.953 \\pm 0.003$",
      f3(float(fwd[fwd.method == "Forward STCG-v1 State Mapping"]["balanced_accuracy_mean"].iloc[0])), "0.953")
check("forward centroid", "centroids $0.966$",
      f3(float(fwd[fwd.method == "Forward GraphFeature Centroid"]["balanced_accuracy_mean"].iloc[0])), "0.966")

vfwd = load("HydraulicSystems_valve_forward_state_label_prediction_forward_summary.csv")
check("valve forward", "$0.250$ balanced accuracy, centroid features $0.323$",
      f3(float(vfwd[vfwd.method == "Forward STCG-v1 State Mapping"]["balanced_accuracy_mean"].iloc[0])), "0.250")
check("valve centroid",
      "centroid features $0.323$",
      f3(float(vfwd[vfwd.method == "Forward GraphFeature Centroid"]["balanced_accuracy_mean"].iloc[0])), "0.323")

arem = load("AReMActivities_state_label_alignment_summary.csv")
check("AReM ARI", "ARI $0.251$", f3(float(arem["adjusted_rand"].mean())), "0.251")
check("AReM acc", "$0.487$", f3(float(arem["aligned_accuracy"].mean())), "0.487")

ath = load("AReMActivities_state_threshold_forecasting_summary.csv")
k3 = ath[(ath.setting == "auto_k3_min_0.00") & (ath.method == "STCG-v1 Weighted Ridge")].iloc[0]
check("AReM K3 ARI", "ARI $0.309 \\pm 0.035$ at $K{=}3$", f3(float(k3["adjusted_rand_mean"])), "0.309")

ak4 = load("AReMActivities_K4_state_label_prediction_summary.csv")
check("AReM K4 centroid", "$0.623 \\pm 0.033$",
      f3(float(ak4[ak4.method == "GraphFeature Centroid"]["balanced_accuracy_mean"].iloc[0])), "0.623")
check("AReM K4 majority", "$0.143$ majority",
      f3(float(ak4[ak4.method == "Train Majority"]["balanced_accuracy_mean"].iloc[0])), "0.143")
check("AReM K4 mapping", "weaker ($0.421$)",
      f3(float(ak4[ak4.method == "STCG-v1 State Mapping"]["balanced_accuracy_mean"].iloc[0])), "0.421")
check("AReM K4 ceiling", "ceiling reaches $0.810$",
      f3(float(ak4[ak4.method == "SensorMeanStd Centroid"]["balanced_accuracy_mean"].iloc[0])), "0.810")

asel = load("AReMActivities_selectK_state_count_selection_summary.csv")
asel_k2 = asel[asel["selected_count"] == 5].iloc[0]
check("AReM selectK K=2", "picks coarse $K{=}2$ AReM states", int(asel_k2["candidate_state_count"]), 2)
check("AReM selectK", "$0.283 \\pm 0.005$", f3(float(asel_k2["balanced_accuracy_mean"])), "0.283")

occ = load("OccupancySensors_state_label_alignment_summary.csv")
check("Occupancy acc", "$0.737$ equal to the majority", f3(float(occ["aligned_accuracy"].mean())), "0.737")

har = load("UCIHARWindowSignals_state_label_alignment_summary.csv")
check("UCIHAR purity", "purity $0.230$", f3(float(har["mean_label_purity"].mean())), "0.230")
hth = load("UCIHARWindowSignals_state_threshold_forecasting_summary.csv")
check("UCIHAR best ARI", "only ARI $0.090$", f3(float(hth["adjusted_rand_mean"].max())), "0.090")

# ---------------- real forecasting ----------------
rf = load("real_benchmark_forecasting_summary.csv")


def rf_val(ds, method):
    return float(rf[(rf.dataset == ds) & (rf.method == method)]["rmse_mean"].iloc[0])


check("Hydraulic STCG mask", "$0.470$ versus $0.472$", f3(rf_val("HydraulicSystems", "STCG-v1 Graph Ridge")), "0.470")
check("Hydraulic DynVAR mask", "$0.470$ versus $0.472$", f3(rf_val("HydraulicSystems", "DynVAR-Lasso Graph Ridge")), "0.472")
check("ETTh1 STCG mask", "($0.760$)", f3(rf_val("ETTh1", "STCG-v1 Graph Ridge")), "0.760")

ws = load("real_weight_sweep_summary.csv")
gain = -float(ws[(ws.dataset == "AReMActivities") & (ws.method == "STCG-v1 Weighted Ridge")]["delta_vs_best_dense_rmse"].min())
check("AReM weight gain", "$0.0034$ RMSE", f"{gain:.4f}", "0.0034")

apt = load("AReMActivities_threshold_paired_tests.csv")
from scipy import stats as _stats
_d = apt["paired_delta_rmse"]
_t_p = _stats.ttest_1samp(_d, 0).pvalue
_w_p = _stats.wilcoxon(-_d, alternative="greater").pvalue  # one-sided "less" on deltas
check("AReM paired t", "$p=0.142$", f"{_t_p:.3f}", "0.142")
check("AReM wilcoxon", "$p=0.0625$", f"{_w_p:.4f}", "0.0625")
check("AReM 4/5 seeds", "four of five seeds improve", int((_d < 0).sum()), 4)
check("AReM K4 rmse", "$0.9062$ to $0.9041$",
      f"{apt['dynvar_lasso_weighted_rmse'].mean():.4f}-->{apt['stcg_v1_weighted_rmse'].mean():.4f}",
      "0.9062-->0.9041")

# ---------------- knob ablation ----------------
kg = load("stcg_knob_ablation_gates_summary.csv")
kgd = load("stcg_knob_ablation_gates_detail.csv")
check("knob min improvement", "$\\approx 0.270$", f"{kgd['state_improvement'].min():.3f}", "0.270")
check("knob gates flat", "reproduce the default $0.812$ Edge AUC exactly",
      f"{kg['edge_auc_mean'].min():.3f}--{kg['edge_auc_mean'].max():.3f}", "0.812--0.812")
check("knob gates K=2", "$K{=}2$ on all twenty seeds",
      f"{int(kg['state_k2_rate'].min() * 20)}/20", "20/20")
ka = load("stcg_knob_ablation_aggregators_summary.csv")
check("knob agg range", "cross-state p90, $0.814$", f"{ka['edge_auc_mean'].max():.3f}", "0.814")
kw = load("stcg_knob_ablation_windows_summary.csv")
check("knob h=300", "$0.781$ versus $0.748$",
      f3(float(kw[kw.config == "h=300"]["edge_auc_mean"].iloc[0])), "0.781")
check("knob h=150", "$0.748$ at $h{=}150$",
      f3(float(kw[kw.config == "h=150"]["edge_auc_mean"].iloc[0])), "0.748")
check("knob h=450", "$0.765$ at $h{=}450$",
      f3(float(kw[kw.config == "h=450"]["edge_auc_mean"].iloc[0])), "0.765")
check("knob ARI h=300", "$0.035$ and $0.050$",
      f3(float(kw[kw.config == "h=300"]["state_ari_mean"].iloc[0])), "0.035")
check("knob ARI h=450", "$0.035$ and $0.050$",
      f3(float(kw[kw.config == "h=450"]["state_ari_mean"].iloc[0])), "0.050")
check("knob ARI h=150", "$0.333$ at $h{=}150$",
      f3(float(kw[kw.config == "h=150"]["state_ari_mean"].iloc[0])), "0.333")

# Actual self-lag features, two-sided tests and post-pilot sensitivity.
kf = load("stcg_knob_ablation_features_summary.csv")
fa = kf[kf.config == "features=all"].iloc[0]
fd = kf[kf.config == "default"].iloc[0]
check("actual self-lag edge AUC", "Edge AUC $0.813$ versus the default $0.812$", f3(float(fa["edge_auc_mean"])), "0.813")
check("actual self-lag state ARI", "state ARI $0.730$ versus $0.764$", f3(float(fa["state_ari_mean"])), "0.730")
check("default feature state ARI", "state ARI $0.730$ versus $0.764$", f3(float(fd["state_ari_mean"])), "0.764")
ts = load("synthetic_rpcmci_two_sided_summary.csv")
ts_edge = ts[ts.metric == "edge_auc"].iloc[0]
check("RPCMCI two-sided t", "$p_t=0.668$", f3(float(ts_edge["paired_t_two_sided_p"])), "0.668")
check("RPCMCI two-sided W", "$p_W=0.596$", f3(float(ts_edge["wilcoxon_two_sided_p"])), "0.596")
post = load("synthetic_postpilot_paired_summary.csv")
pc = post[(post.generator == "switching_var") & (post.method_b == "cMLP") & (post.metric == "edge_auc")].iloc[0]
check("post-pilot n", "over fifteen common seeds", int(pc["n"]), 15)
check("post-pilot STCG", "$0.803$ for STCG-v1", f3(float(pc["mean_a"])), "0.803")
check("post-pilot cMLP", "$0.775$ for cMLP", f3(float(pc["mean_b"])), "0.775")
check("post-pilot difference", "$+0.0284$", f"{float(pc['mean_improvement']):+.4f}", "+0.0284")
check("post-pilot t", "$p_t=0.0086$", f"{float(pc['paired_t_greater_p']):.4f}", "0.0086")
check("post-pilot W", "$p_W=0.0051$", f"{float(pc['wilcoxon_greater_p']):.4f}", "0.0051")

# Purged-horizon warning results must carry the current protocol provenance.
warning = load("HydraulicSystems_transition_warning_detail.csv")
selected_warning = load("HydraulicSystems_warning_ablation_selected.csv")
for frame in (warning, selected_warning):
    assert (frame["split_policy"] == "purged_target_horizon").all(), "Legacy warning protocol"
    assert (frame["purge_horizon_cycles"] == frame["horizon_cycles"]).all(), "Wrong warning purge"

def warning_val(frame, label, horizon, method, metric="average_precision"):
    match = frame[(frame.label_column == label) & (frame.horizon_cycles == horizon) & (frame.method == method)]
    assert len(match) == 1, (label, horizon, method, len(match))
    return float(match[metric].iloc[0])

for desc, label, horizon, method, metric, expected, snippet in [
    ("valve graph AP", "valve_condition", 10, "Sensor+Graph Logistic", "average_precision", "0.901", "AP $0.901$ and ROC AUC $0.752$"),
    ("valve graph ROC", "valve_condition", 10, "Sensor+Graph Logistic", "roc_auc", "0.752", "ROC AUC $0.752$"),
    ("valve majority", "valve_condition", 10, "Train Majority", "average_precision", "0.742", "majority AP $0.742$"),
    ("stable sensor graph", "stable_flag", 10, "Sensor+Graph Logistic", "average_precision", "0.278", "sensor-plus-graph AP is $0.278$"),
    ("stable graph", "stable_flag", 10, "Graph Ridge Logistic", "average_precision", "0.275", "graph-only ridge AP $0.275$"),
    ("stable majority", "stable_flag", 10, "Train Majority", "average_precision", "0.244", "majority AP $0.244$"),
    ("short stable AP", "stable_flag", 5, "Sensor+STCG-v1 Logistic", "average_precision", "0.160", "5-cycle stable-flag $0.160$"),
    ("short valve AP", "valve_condition", 3, "Sensor+STCG-v1 Logistic", "average_precision", "0.239", "3-cycle valve-condition $0.239$"),
    ("valve STCG F1", "valve_condition", 10, "Sensor+STCG-v1 Logistic", "f1", "0.864", "warnings ($0.864$)"),
]:
    check(desc, snippet, f3(warning_val(warning, label, horizon, method, metric)), expected)
check("selected stable AP", "AP $0.272$ versus sensor-only $0.278$", f3(warning_val(selected_warning, "stable_flag", 10, "Sensor+STCG-v1 Logistic")), "0.272")
check("selected valve AP", "reaches $0.834$ on 10-cycle valve", f3(warning_val(selected_warning, "valve_condition", 10, "Sensor+STCG-v1 Logistic")), "0.834")

# ---------------- specialist stress extension + matched timing ----------------
sp = load("revision_specialist_paired.csv")
ssum = load("revision_specialist_summary.csv")
scov = load("revision_specialist_coverage.csv")
stim = load("revision_specialist_timing.csv")
sdet = load("revision_specialist_detail.csv")


def sp_row(axis, value):
    sid = ssum[(ssum.axis == axis) & (ssum.value == value)]["setting_id"].iloc[0]
    r = sp[(sp.phase == "stress") & (sp.setting_id == sid)
           & (sp.opponent == "Regime-PCMCI") & (sp.metric == "edge_auc")]
    assert len(r) == 1, (axis, value)
    return r.iloc[0]


def sci(p, digits=1):
    m, e = f"{p:.{digits}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


check("stress 13 unique settings", "Thirteen unique settings",
      ssum[["axis", "value", "setting_id"]].drop_duplicates()["setting_id"].nunique(), 13)
check("stress 1400 fits", "All $1400$ planned fits", len(sdet), 1400)
check("stress zero failures", "zero failures or timeouts",
      int(scov.failed.sum()) + int((scov.planned - scov.successful).sum()), 0)

base = sp_row("obs_noise", 0.0)
check("stress base means", "$0.827$ versus $0.839$",
      f"{base.mean_stcg:.3f}-->{base.mean_opponent:.3f}", "0.827-->0.839")
check("stress base diff", "$-0.0121$", f"{base.mean_improvement:.4f}", "-0.0121")
check("stress base CI", "$[-0.0248, 0.0022]$",
      f"[{base.ci_low:.4f}, {base.ci_high:.4f}]", "[-0.0248, 0.0022]")
check("stress base Holm t", "$p_t=0.571$", f"{base.paired_t_two_sided_p_holm13:.3f}", "0.571")

s150 = sp_row("switch_period", 150)
check("stress s150 diff", "$-0.1078$", f"{s150.mean_improvement:.4f}", "-0.1078")
check("stress s150 Holm t", "p_t=1.1\\times10^{-9}$", sci(s150.paired_t_two_sided_p_holm13), "1.1\\times10^{-9}")
s300 = sp_row("switch_period", 300)
check("stress s300 diff", "$-0.0375$", f"{s300.mean_improvement:.4f}", "-0.0375")
check("stress s300 Holm t", "p_t=4.6\\times10^{-4}$", sci(s300.paired_t_two_sided_p_holm13), "4.6\\times10^{-4}")
t1500 = sp_row("timesteps", 1500)
check("stress T1500 diff", "$-0.0240$", f"{t1500.mean_improvement:.4f}", "-0.0240")
check("stress T1500 Holm t", "$p_t=0.0027$", f"{t1500.paired_t_two_sided_p_holm13:.4f}", "0.0027")
t300 = sp_row("timesteps", 300)
check("stress T300 diff", "$+0.0398$", f"{t300.mean_improvement:+.4f}", "+0.0398")
check("stress T300 CI", "$[0.0131, 0.0666]$",
      f"[{t300.ci_low:.4f}, {t300.ci_high:.4f}]", "[0.0131, 0.0666]")
check("stress T300 Holm t", "$p_t=0.082$", f"{t300.paired_t_two_sided_p_holm13:.3f}", "0.082")
e03 = sp_row("edge_prob", 0.3)
check("stress density 0.3 diff", "$-0.0147$", f"{e03.mean_improvement:.4f}", "-0.0147")
check("stress density 0.3 Holm t", "$p_t=0.066$", f"{e03.paired_t_two_sided_p_holm13:.3f}", "0.066")


def stim_row(opponent, metric="fit_seconds"):
    r = stim[(stim.phase == "timing") & (stim.dedicated_exclusive_phase == True)
             & (stim.opponent == opponent) & (stim.metric == metric)]
    assert len(r) == 1 and bool(r.iloc[0].ratio_available)
    return r.iloc[0]


rpc_w = stim_row("Regime-PCMCI")
check("timing RPCMCI wall", "$5575\\times$", f"{rpc_w.geometric_mean_opponent_over_stcg:.0f}", "5575")
check("timing RPCMCI wall CI", "$[5246, 5984]$",
      f"[{rpc_w.ratio_ci_low:.0f}, {rpc_w.ratio_ci_high:.0f}]", "[5246, 5984]")
check("timing 5.6e3 phrasing", "$5.6\\times10^{3}$ times",
      f"{rpc_w.geometric_mean_opponent_over_stcg / 1000:.1f}", "5.6")
rpc_c = stim_row("Regime-PCMCI", "fit_cpu_seconds")
check("timing RPCMCI cpu", "$6293\\times$", f"{rpc_c.geometric_mean_opponent_over_stcg:.0f}", "6293")
cmlp_w = stim_row("cMLP")
check("timing cMLP", "$86.6\\times$", f"{cmlp_w.geometric_mean_opponent_over_stcg:.1f}", "86.6")
check("timing cMLP CI", "$[81.8, 92.1]$",
      f"[{cmlp_w.ratio_ci_low:.1f}, {cmlp_w.ratio_ci_high:.1f}]", "[81.8, 92.1]")
check("timing DynVAR", "$0.458$", f"{stim_row('DynVAR-Lasso').geometric_mean_opponent_over_stcg:.3f}", "0.458")
check("timing VAR-Lasso", "$0.116$", f"{stim_row('VAR-Lasso').geometric_mean_opponent_over_stcg:.3f}", "0.116")

# ---------------- joint MSVAR corrective comparison ----------------
jm = load("revision_joint_msvar_paired.csv")
jcov = load("revision_joint_msvar_coverage.csv")
jrow = jm[(jm.population == "selected_converged") & (jm.metric == "edge_auc")].iloc[0]
check("joint converged", "All twenty fits converge",
      f"{int(jcov.ok.iloc[0])}/{int(jcov.planned.iloc[0])} nc={int(jcov.nonconverged.iloc[0])}", "20/20 nc=0")
check("joint means", "$0.827$ versus $0.817$",
      f"{jrow.mean_stcg:.3f}-->{jrow.mean_opponent:.3f}", "0.827-->0.817")
check("joint diff", "$+0.0109$", f"{jrow.mean_improvement:+.4f}", "+0.0109")
check("joint CI", "$[-0.0174, 0.0463]$", f"[{jrow.ci_low:.4f}, {jrow.ci_high:.4f}]", "[-0.0174, 0.0463]")
check("joint t", "$p_t=0.527$", f"{jrow.paired_t_two_sided_p:.3f}", "0.527")
check("joint W", "$p_W=0.430$", f"{jrow.wilcoxon_two_sided_p:.3f}", "0.430")


# ---------------- variance-reduced score ensemble (STCG-v1e) ----------------
ens = load("stcg_ensemble_summary.csv")
ens_pair = load("stcg_ensemble_paired.csv")
ens_stress = load("stcg_ensemble_stress_detail.csv")


def ens_mean(gen, method):
    return float(ens[(ens.generator == gen) & (ens.method == method)]["edge_auc_mean"].iloc[0])


def ens_p(gen):
    return float(ens_pair[ens_pair.generator == gen]["edge_auc_p_t"].iloc[0])


check("ensemble stationary mean", "0.876\\rightarrow0.884$", f3(ens_mean("var", "STCG-v1e")), "0.884")
check("ensemble nonlinear mean", "0.970\\rightarrow0.971$", f3(ens_mean("nonlinear_var", "STCG-v1e")), "0.971")
check("ensemble switching mean", "$0.815$", f3(ens_mean("switching_var", "STCG-v1e")), "0.815")
check("ensemble switching p", "$p_t=0.032$", f"{ens_p('switching_var'):.3f}", "0.032")

ens_pivot = ens_stress.pivot_table(
    index=["setting_id", "seed"], columns="method", values="edge_auc"
)
ens_diff = ens_pivot["STCG-v1e"] - ens_pivot["STCG-v1"]
check("ensemble stress mean", "$0.801$", f3(float(ens_pivot["STCG-v1e"].mean())), "0.801")
check("ensemble stress wins", "190 of 260", int((ens_diff > 0).sum()), 190)
check("ensemble stress p", "$p_t=1.5\\times10^{-12}$",
      f"{_stats.ttest_rel(ens_pivot['STCG-v1e'], ens_pivot['STCG-v1']).pvalue:.1e}", "1.5e-12")
period300 = ens_stress[ens_stress.setting_id == "f3cb92c65405"]
p300 = period300.pivot_table(index="seed", columns="method", values="edge_auc")
check("ensemble period300 mean", "$0.831$", f3(float(p300["STCG-v1e"].mean())), "0.831")
check("ensemble period300 p", "$p_t=3.2\\times10^{-5}$",
      f"{_stats.ttest_rel(p300['STCG-v1e'], p300['STCG-v1']).pvalue:.1e}", "3.2e-05")


# ---------------- ensemble mechanism analysis ----------------
inst = load("stcg_ensemble_instability.csv")
inst_bin = load("stcg_ensemble_instability_summary.csv")
inst_stats = load("stcg_ensemble_instability_stats.csv").set_index("statistic")["value"]
inst_r = float(inst_stats["pearson_r"])
inst_p = float(inst_stats["pearson_p"])
base_r = float(inst_stats["baseline_r"])
within_r = float(inst_stats["within_condition_r"])
within_p = float(inst_stats["within_condition_p"])
check("instability correlation", "$r=0.22$", f"{inst_r:.2f}", "0.22")
check("instability p", "$p=4.1\\times10^{-4}$", f"{inst_p:.1e}", "4.1e-04")
check("instability baseline", "$r=0.11$", f"{base_r:.2f}", "0.11")
check("instability within", "$r=0.08$", f"{within_r:.2f}", "0.08")
check("instability within p", "$p=0.21$", f"{within_p:.2f}", "0.21")
check("instability stable quintile", "$+0.0026$",
      f"{inst_bin['gain'].iloc[0]:+.4f}", "+0.0026")
check("instability unstable quintile", "$+0.0097$",
      f"{inst_bin['gain'].iloc[-1]:+.4f}", "+0.0097")

seed_gain = ens_diff.groupby(level="seed").mean()
clustered_p = float(_stats.ttest_1samp(seed_gain, 0).pvalue)
check("ensemble stress clustered p", "$p=2.1\\times10^{-7}$",
      f"{clustered_p:.1e}", "2.1e-07")

sens = load("stcg_ensemble_b_sensitivity_summary.csv")
sens_switch = sens[(sens.generator == "switching_var") & (sens.members.isin([10, 20, 40]))]
check("B sensitivity mean", "stays at $0.815$",
      f3(float(sens_switch["edge_auc_mean"].min())), "0.815")
check("B sensitivity p range", "$0.032$ to $0.081$",
      f"{sens_switch['p_t'].min():.3f} to {sens_switch['p_t'].max():.3f}", "0.032 to 0.081")

# ---------------- change-point calibrated gate ----------------
cpg = load("change_point_gate_validation.csv")


def cpg_accept(group):
    subset = cpg[cpg.group == group]
    return f"{int(subset['change_point_accept'].sum())}/{len(subset)}"


check("cp gate main stationary accept", "0/20 stationary",
      cpg_accept("main_stationary"), "0/20")
check("cp gate few-window stationary accept", "0/20 few-window stationary",
      cpg_accept("stationary_few"), "0/20")
check("cp gate main switching accept", "20/20 switching",
      cpg_accept("main_switching"), "20/20")
check("cp gate short-switch accept", "20/20 short-switch",
      cpg_accept("short_switch"), "20/20")
check("cp gate switching auc", "$0.827$ under both gates",
      f3(float(cpg[cpg.group == "main_switching"]["edge_auc_new"].mean())), "0.827")
check("cp gate stationary current auc", "stationary fell from $0.890$ to $0.875$",
      f3(float(cpg[cpg.group == "main_stationary"]["edge_auc_current"].mean())), "0.890")
check("cp gate stationary new auc", "stationary fell from $0.890$ to $0.875$",
      f3(float(cpg[cpg.group == "main_stationary"]["edge_auc_new"].mean())), "0.875")

# ---------------- ensemble controls ----------------
ctrl = load("stcg_ensemble_controls_summary.csv")


def ctrl_mean(gen, method):
    row = ctrl[(ctrl.generator == gen) & (ctrl.method == method)]
    return f3(float(row["edge_auc_mean"].iloc[0]))


check("control bag stationary", "$0.886$ on stationary VAR",
      ctrl_mean("var", "bag_global"), "0.886")
check("control bag nonlinear", "$0.974$ on nonlinear VAR",
      ctrl_mean("nonlinear_var", "bag_global"), "0.974")
check("control bag switching", "$0.808$ on switching VAR",
      ctrl_mean("switching_var", "bag_global"), "0.808")
check("control random switching", "$0.813$ on switching",
      ctrl_mean("switching_var", "random_partition"), "0.813")
check("control permuted", "$0.885$ and $0.812$",
      ctrl_mean("var", "permuted_kmeans"), "0.885")
check("control permuted switching", "$0.885$ and $0.812$",
      ctrl_mean("switching_var", "permuted_kmeans"), "0.812")


# ---------------- real-data boundary check ----------------
real_h = load("real_change_point_gate_hydraulic_summary.csv").set_index("kind")
real_a = load("real_change_point_gate_arem_summary.csv").set_index("kind")
check("real hydraulic boundary", "34.2\\% of boundaries",
      f"{real_h.loc['boundary', 'change_point_accept'] * 100:.1f}\\%", "34.2\\%")
check("real hydraulic stable", "30.0\\% of stable segments",
      f"{real_h.loc['stable', 'change_point_accept'] * 100:.1f}\\%", "30.0\\%")
check("real arem boundary", "100\\% of both",
      f"{real_a.loc['boundary', 'change_point_accept'] * 100:.0f}\\%", "100\\%")
check("real arem stable", "100\\% of both",
      f"{real_a.loc['stable', 'change_point_accept'] * 100:.0f}\\%", "100\\%")

# ---------------- theoretical reading ----------------
bias = load("aggregation_bias.csv").set_index("mode")
check("toy shared mean", "from $0.934$ to $0.901$", f3(float(bias.loc["shared", "mean_stat"])), "0.934")
check("toy shared max", "from $0.934$ to $0.901$", f3(float(bias.loc["shared", "max_stat"])), "0.901")
check("toy specific mean", "from $0.774$ to $0.803$", f3(float(bias.loc["state_specific", "mean_stat"])), "0.774")
check("toy specific max", "from $0.774$ to $0.803$", f3(float(bias.loc["state_specific", "max_stat"])), "0.803")
variance = load("stcg_ensemble_variance.csv")
switch_rho = float(variance[variance.generator == "switching_var"]["mean_member_correlation"].mean())
check("ensemble member correlation", "$0.97$", f"{switch_rho:.2f}", "0.97")

print(f"checked {checked} numbers")
if failures:
    print(f"MISMATCHES: {len(failures)}")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("ALL NUMBERS MATCH")
