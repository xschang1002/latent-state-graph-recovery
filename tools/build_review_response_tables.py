"""Build compact manuscript tables from the additional review-response evidence.

This tool reads recorded results only. It does not select favorable conditions,
refit models, or overwrite the original revision tables.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def write_table(name, columns, heading, rows):
    lines = [rf"\begin{{tabular}}{{{columns}}}", r"\toprule", heading + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path = TABLES / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def build_gate_table():
    frame = pd.read_csv(TABLES / "revision_gate_stress_summary.csv")
    selected = frame[(frame.evidence_kind == "generator") & (frame.analysis == "gate") & (frame.variant == "default")].set_index("condition")
    conditions = {
        "stationary_many": "Stationary, long",
        "stationary_few": "Stationary, few windows",
        "strong_switch_long": "Strong switch, long",
        "weak_switch_long": "Weak switch, long",
        "mixed_periodic": "Mixed periodic windows",
        "short_switch": "Short switch",
        "measurement_noise_switch": "Noisy switch",
        "imbalanced_switch": "Rare second regime",
    }
    assert set(selected.index) == set(conditions), "Incomplete frozen gate grid"
    rows = []
    for name, label in conditions.items():
        row = selected.loc[name]
        n = int(row["n"])
        assert n == 20, (name, n)
        accepted = int(round(n * row.states_accepted_mean))
        fallback = int(round(n * row.fallback_accept_mean))
        rows.append([label, f"{accepted}/{n}", f"{fallback}/{n}", f"{row.edge_auc_mean:.3f}"])
    return write_table("revision_gate_paper_table.tex", "lrrr", "Condition & Accepted & Fallback only & Edge AUC", rows)


def build_forward_table():
    frame = pd.read_csv(TABLES / "revision_real_forward_summary.csv")
    methods = {
        "Train Majority": "Train majority",
        "Sensor Centroid": "Sensor centroid",
        "Graph Centroid": "Graph centroid",
        "Sensor Logistic": "Sensor logistic",
        "STCG Fixed K2": "STCG, fixed $K{=}2$",
        "STCG Train Silhouette K": "STCG, train-selected $K$",
    }
    labels = [("HydraulicSystems", "cooler_condition"), ("HydraulicSystems", "valve_condition"), ("AReMActivities", "activity")]
    assert len(frame) == len(methods) * len(labels), "Incomplete forward comparison"
    rows = []
    for method, display in methods.items():
        values = []
        for dataset, label in labels:
            match = frame[(frame.dataset == dataset) & (frame.label_column == label) & (frame.method == method)]
            assert len(match) == 1 and int(match.iloc[0].origins) == 4
            values.append(f"{match.iloc[0].balanced_accuracy_mean:.3f}")
        rows.append([display, *values])
    return write_table("revision_real_paper_table.tex", "lrrr", "Method & Cooler & Valve & AReM", rows)


def build_specialist_tables():
    """Require every planned attempt; failed fits remain visible through n."""
    detail = pd.read_csv(TABLES / "revision_specialist_detail.csv")
    coverage = pd.read_csv(TABLES / "revision_specialist_coverage.csv")
    pairs = pd.read_csv(TABLES / "revision_specialist_paired.csv")
    timing = pd.read_csv(TABLES / "revision_specialist_timing.csv")
    assert len(detail) == 1400 and coverage.attempted.eq(coverage.planned).all(), "Full stress and quiet timing required"
    assert not detail.duplicated(["phase", "setting_id", "method", "seed"]).any()
    frozen = json.loads((TABLES / "revision_specialist_protocol.json").read_text(encoding="utf-8"))
    cells = pd.DataFrame(frozen["specification"]["display_cells"])[["axis", "value", "setting_id"]]
    assert len(cells) == 15 and cells.setting_id.nunique() == 13
    methods = ["STCG-v1", "Regime-PCMCI", "cMLP", "DynVAR-Lasso", "VAR-Lasso"]
    axis_labels = {"obs_noise": "Noise", "timesteps": "$T$", "edge_prob": "Density", "switch_period": "Period"}
    rows, effects = [], []
    for cell in cells.itertuples(index=False):
        complete = detail[(detail.phase == "stress") & (detail.setting_id == cell.setting_id) & (detail.status == "ok")]
        matrix = complete.pivot(index="seed", columns="method", values="edge_auc").reindex(columns=methods).dropna()
        mean_values = [f"{matrix[method].mean():.3f}" if len(matrix) else "--" for method in methods]
        match = pairs[(pairs.phase == "stress") & (pairs.setting_id == cell.setting_id) &
                      (pairs.opponent == "Regime-PCMCI") & (pairs.metric == "edge_auc")]
        ci = "--"
        effect = dict(axis=cell.axis, value=cell.value, mean=np.nan, low=np.nan, high=np.nan, n=len(matrix))
        if len(match) == 1 and match.iloc[0].statistics_status == "ok":
            row = match.iloc[0]
            assert int(row.n) == len(matrix)
            ci = f"${row.mean_improvement:+.3f}$ $[{row.ci_low:+.3f},{row.ci_high:+.3f}]$"
            effect.update(mean=row.mean_improvement, low=row.ci_low, high=row.ci_high)
        effects.append(effect)
        rows.append([f"{axis_labels[cell.axis]} {cell.value:g}", str(len(matrix)), *mean_values, ci])
    accuracy_path = write_table("revision_specialist_paper_table.tex", "lrrrrrrl",
                               r"Setting & $n$ & STCG & RPCMCI & cMLP & DynVAR & VAR & $\Delta$ vs RPCMCI [95\% CI]", rows)
    quiet = detail[detail.phase == "timing"]
    assert len(quiet) == 100 and quiet.execution_workers.eq(1).all() and quiet.exclusive_timing.eq(True).all()
    timing_rows = []
    for method in methods:
        success = quiet[(quiet.method == method) & (quiet.status == "ok")]
        seconds = success.fit_seconds
        observed = "--" if not len(seconds) else f"{seconds.median():.4f} [{seconds.quantile(.25):.4f}, {seconds.quantile(.75):.4f}]"
        ratio = "Reference"
        ratio_n = "--"
        if method != "STCG-v1":
            match = timing[(timing.phase == "timing") & (timing.metric == "fit_seconds") & (timing.opponent == method)]
            ratio = "--"
            if len(match) == 1 and bool(match.iloc[0].ratio_available):
                row = match.iloc[0]
                ratio_n = str(int(row.n))
                ratio = f"{row.geometric_mean_opponent_over_stcg:.2f} [{row.ratio_ci_low:.2f}, {row.ratio_ci_high:.2f}]"
        timing_rows.append([method, f"{len(success)}/20", observed, ratio_n, ratio])
    timing_path = write_table("revision_specialist_timing_table.tex", "lrlrl",
                             "Method & Fits & Seconds: median [IQR] & Ratio $n$ & Geometric cost / STCG [95\\% CI]", timing_rows)
    return accuracy_path, timing_path, build_effect_figure(pd.DataFrame(effects))


def build_effect_figure(effects):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    effects = effects.reindex(columns=["axis", "value", "mean", "low", "high", "n"])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True, constrained_layout=True)
    groups = [("obs_noise", "Observation noise ratio"), ("timesteps", "Sequence length"),
              ("edge_prob", "Edge probability"), ("switch_period", "Switch period")]
    for ax, (axis, title) in zip(axes.flat, groups):
        group = effects[effects.axis == axis].sort_values("value")
        values = group["mean"].to_numpy(dtype=float) * 100
        left = values - group.low.to_numpy(dtype=float) * 100
        right = group.high.to_numpy(dtype=float) * 100 - values
        y = np.arange(len(group))
        ax.axvline(0, color="#68717b", linewidth=.8, linestyle="--")
        available = np.isfinite(values) & np.isfinite(left) & np.isfinite(right)
        if available.any():
            ax.errorbar(values[available], y[available], xerr=[left[available], right[available]], fmt="o", color="#245788", ecolor="#245788", capsize=3, markersize=4)
        for position in y[~available]:
            ax.text(.03, position, f"Not estimable (n={int(group.iloc[position]['n'])})", transform=ax.get_yaxis_transform(), fontsize=8, va="center", color="#60666d")
        if not len(group):
            ax.text(.5, .5, "No estimates", ha="center", transform=ax.transAxes, fontsize=9)
        ax.set_yticks(y, [f"{v:g}" for v in group.value])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x", alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.supxlabel("STCG minus Regime-PCMCI Edge AUC (percentage points)", fontsize=10)
    if not effects["mean"].notna().any():
        axes.flat[0].set_xlim(-1, 1)
    path = ROOT / "results/figures/revision_specialist_rpcmci_effect.png"
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-specialist", action="store_true", help="Require completed stress and quiet timing and build their tables/figure")
    args = parser.parse_args()
    outputs = [build_gate_table(), build_forward_table()]
    if args.include_specialist:
        outputs.extend(build_specialist_tables())
    for result in outputs:
        print(result.relative_to(ROOT))
