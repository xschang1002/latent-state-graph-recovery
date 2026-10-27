"""Build four main-text tables and four figures from retained study outputs.

This presentation builder never fits a scorer, reruns an experiment or modifies
the frozen A/B outputs. The deterministic 4-edge example is explicitly a finite
illustration, not an empirical estimate. All uncertainty uses simulation repeats.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/information_sciences"
A = OUT / "duration_confirmation"
B = OUT / "schedule_robustness"
KINDS = ("fixed_phase", "random_phase", "irregular")
CELLS = [(duration, kind) for duration in (300, 600) for kind in KINDS]
KIND_LABEL = {"fixed_phase": "Fixed phase", "random_phase": "Random phase", "irregular": "Irregular"}
METHODS = ("raw_window", "static_mean", "native_hard", "forced_hard", "matched_random", "oracle_label")
METHOD_LABEL = {"raw_window": "Raw windows", "static_mean": "Static mean", "native_hard": "Native hard",
                "forced_hard": "Forced hard", "matched_random": "Matched random", "oracle_label": "Oracle labels"}
BLUE, ORANGE, PURPLE = "#0072B2", "#D55E00", "#7B3294"
STYLE = {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
         "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
         "legend.fontsize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.linewidth": 0.7, "savefig.facecolor": "white", "figure.facecolor": "white"}
SKILL_CITATION = {
    "authors": ["Timothy Kassis", "Vinayak Agarwal", "Yuhuan He", "Darshil Patel", "Aubrey M. Brueckner"],
    "year": 2026, "title": "Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents",
    "doi": "https://doi.org/10.48550/arXiv.2609.00065", "metadata_verified": "2026-10-24",
    "metadata_source": "https://arxiv.org/abs/2609.00065",
    "version_observed": "v2, last revised 2026-09-02; citation uses the unversioned DOI",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs() -> dict:
    paths = {
        "a_detail": A / "detail.csv", "a_summary": A / "summary.csv", "a_tests": A / "contrasts.csv",
        "a_protocol": A / "protocol.json", "a_complete": A / "COMPLETE.json",
        "b_detail": B / "schedule_robustness_detail.csv",
        "b_diagnostics": B / "schedule_robustness_diagnostics.csv",
        "b_paired": B / "schedule_robustness_paired.csv",
        "b_tests": B / "schedule_robustness_secondary_tests.csv",
        "b_gate": B / "schedule_robustness_gate_summary.csv",
        "b_protocol": B / "schedule_robustness_protocol.json",
        "b_complete": B / "schedule_robustness_outputs.json",
        "ess": ROOT / "results/tables/ess_alignment.csv",
    }
    missing = [p.relative_to(ROOT).as_posix() for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Completed retained sources are required; no placeholder data: " + ", ".join(missing))
    frame = {name: pd.read_csv(path) for name, path in paths.items() if path.suffix == ".csv"}
    protocol = {name: json.loads(path.read_text()) for name, path in paths.items() if name.endswith("protocol")}
    if len(frame["a_detail"]) != 2400 or frame["a_detail"].repeat_seed.nunique() != 100:
        raise ValueError("Study A must retain all 100 planned repeats and 2400 method records")
    if len(frame["b_detail"]) != 1500 or frame["b_detail"].seed.nunique() != 50:
        raise ValueError("Study B must retain all 50 planned repeats and 1500 method records")
    if set(zip(frame["b_paired"].duration_mean, frame["b_paired"].schedule)) != set(CELLS):
        raise ValueError("All six schedule cells must remain visible")
    return {**frame, **protocol, "paths": paths}


def number(value, *, signed=False) -> str:
    return "--" if not np.isfinite(float(value)) else f"{value:+.3f}" if signed else f"{value:.3f}"


def interval(values) -> tuple[float, float, float, int]:
    """Pointwise bootstrap over independent repeats, never windows or edges."""
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan, 0
    if len(values) < 5:
        return float(values.mean()), np.nan, np.nan, len(values)
    rng = np.random.default_rng(20261024)
    estimates = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(values.mean()), float(low), float(high), len(values)


def point_interval(ax, y, mean, low, high, *, color=BLUE, marker="o", label=None):
    if not np.isfinite(mean):
        ax.text(0.02, y, "Not estimable", transform=ax.get_yaxis_transform(), va="center", fontsize=8.5)
        return
    if np.isfinite(low) and np.isfinite(high):
        ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt=marker, color=color,
                    markeredgecolor="black", markeredgewidth=0.4, markersize=5.1,
                    elinewidth=1.25, capsize=2.5, label=label, zorder=3)
    else:
        ax.plot(mean, y, marker=marker, color=color, markeredgecolor="black", markersize=5.1, label=label)
        ax.annotate("CI unavailable", (mean, y), xytext=(5, 0), textcoords="offset points", fontsize=8.5, va="center")


def forest(ax, labels, *, baseline=True):
    positions = np.arange(len(labels))
    ax.set(yticks=positions, yticklabels=labels, ylim=(len(labels) - 0.45, -0.55))
    ax.grid(axis="x", color="#D7D7D7", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    if baseline:
        ax.axvline(0, color="#525252", linestyle="--", linewidth=0.8, zorder=1)
    return positions


def scientific(value) -> str:
    if not np.isfinite(value):
        return "Not estimable"
    if value == 0:
        return "$p=0$"
    exponent = int(np.floor(np.log10(value)))
    return f"${value / 10 ** exponent:.2f}\\times10^{{{exponent}}}$" if exponent < -2 else f"{value:.3f}"


def difference_limits(*arrays) -> tuple[float, float]:
    finite = np.concatenate([np.asarray(values, dtype=float).ravel() for values in arrays])
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return -0.1, 0.1
    low, high = min(0.0, float(finite.min())), max(0.0, float(finite.max()))
    padding = max(0.015, 0.08 * (high - low))
    return min(-0.1, low - padding), max(0.1, high + padding)


def export(fig, output: Path, name: str, description: str, provenance: dict) -> dict:
    """Keep the exact 6.6-inch page, embedded TrueType fonts and opaque PNG."""
    fig.canvas.draw()
    fonts = [item.get_fontsize() for item in fig.findobj(Text) if item.get_visible() and item.get_text()]
    if fonts and min(fonts) < 8:
        raise ValueError(f"A label is smaller than 8 pt in {name}")
    pdf, png = output / f"{name}.pdf", output / f"{name}.png"
    fig.savefig(pdf, format="pdf", bbox_inches=None, transparent=False,
                metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None})
    fig.savefig(png, format="png", dpi=600, bbox_inches=None, transparent=False,
                metadata={"Software": "", "Description": description})
    with Image.open(png) as saved:
        pixel_size = saved.size; dpi = saved.info.get("dpi")
        if saved.mode == "RGBA" and saved.getextrema()[-1] != (255, 255):
            raise ValueError("Exported PNG must have an opaque white background")
    record = {"pdf": pdf.name, "png": png.name, "width_inches": float(fig.get_size_inches()[0]),
              "height_inches": float(fig.get_size_inches()[1]), "png_pixels": list(pixel_size),
              "png_dpi": list(dpi) if dpi else None, "minimum_nominal_font_pt": min(fonts),
              "pdf_fonttype": 42, "rasterized_artists": False,
              "description": description, **provenance}
    plt.close(fig)
    return record


def target_example(output: Path) -> tuple[dict, dict]:
    states = np.array([0] * 6 + [1] * 2)
    truth = np.stack([np.ones(8), states == 0, states == 1, np.zeros(8)]).astype(int)
    occupancy = truth.mean(axis=1)
    static = np.repeat(occupancy[:, None], 8, axis=1)
    pooled = float(roc_auc_score(truth.reshape(-1), static.reshape(-1)))
    temporal = float(np.mean([roc_auc_score(truth[k], static[k]) for k in (1, 2)]))
    edge_labels = ["Persistent", "Only state 0", "Only state 1", "Never"]
    rows = [{"centre": w + 1, "true_state": int(states[w]), "edge": e + 1,
             "edge_kind": edge_labels[e], "true_active": int(truth[e, w]),
             "static_occupancy_score": float(static[e, w])} for e in range(4) for w in range(8)]
    pd.DataFrame(rows).to_csv(output / "target_example.csv", index=False)
    metrics = {"role": "deterministic finite illustration, not an experimental estimate",
               "n_edges": 4, "n_centres": 8, "state0_centres": 6, "state1_centres": 2,
               "changing_edges": 2, "pooled_auroc": pooled, "changing_edge_temporal_auroc": temporal}
    pd.DataFrame([metrics]).to_csv(output / "target_example_metrics.csv", index=False)
    fig = plt.figure(figsize=(6.6, 2.9), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.15, 0.9], wspace=0.14)
    first, second, last = [fig.add_subplot(grid[0, k]) for k in range(3)]
    for index, (ax, values, title) in enumerate(((first, truth, "(a) Active-edge truth"),
                                                (second, static, "(b) Static occupancy score"))):
        ax.pcolormesh(np.arange(9), np.arange(5), values, shading="flat", cmap="Blues",
                      norm=Normalize(0, 1), edgecolors="#AAAAAA", linewidth=0.4, rasterized=False)
        for e in range(4):
            for w in range(8):
                value = values[e, w]
                if index == 0:
                    ax.text(w + 0.5, e + 0.5, str(int(value)), ha="center", va="center",
                            color="white" if value >= 0.75 else "black", fontsize=8.5)
            if index == 1:
                value = values[e, 0]
                label = str(int(value)) if value in (0, 1) else f"{value:.2f}"
                ax.text(4, e + 0.5, label + " (constant)", ha="center", va="center",
                        color="white" if value >= 0.75 else "black", fontsize=8.5,
                        bbox={"facecolor": plt.get_cmap("Blues")(value), "edgecolor": "none", "pad": 0.7})
        ax.set(xlim=(0, 8), ylim=(4, 0), xticks=np.arange(8) + 0.5, xticklabels=np.arange(1, 9),
               yticks=np.arange(4) + 0.5, yticklabels=edge_labels if index == 0 else [], xlabel="Window centre")
        ax.tick_params(axis="both", length=0)
        ax.set_title(title, loc="left", pad=22)
        for left, right, label in ((0, 6, "State 0"), (6, 8, "State 1")):
            ax.plot([left, right], [-0.2, -0.2], color="#333333", clip_on=False, linewidth=1)
            ax.text((left + right) / 2, -0.35, label, ha="center", va="bottom", fontsize=8.5)
    last.set_title("(c) Static-score\nAUROC", loc="left")
    for y, value, color, marker in ((1, pooled, BLUE, "o"), (0, temporal, ORANGE, "s")):
        last.plot(value, y, marker, color=color, markeredgecolor="black", markersize=6)
        last.annotate(f"{value:.3f}", (value, y), xytext=(-3, 9), textcoords="offset points", ha="center", fontsize=8.5)
    last.axvline(0.5, color="#666666", linestyle="--", linewidth=0.8)
    last.set(xlim=(0, 1.05), ylim=(-0.65, 1.65), yticks=[1, 0], yticklabels=["Pooled", "Changing\nedge"],
             xlabel="AUROC", xticks=[0, 0.5, 1])
    last.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    fig.suptitle("Finite illustration: 4 edges, 8 centres, state counts 6 / 2", fontsize=9.5)
    description = "A deterministic four-edge example: a constant occupancy score has pooled AUROC 0.9375 and changing-edge temporal AUROC 0.5, despite no time-varying prediction."
    record = export(fig, output, "fig_target_example", description,
                    {"unit": "finite edge/time table; no inferential replicate", "uncertainty": "none; exact finite calculation"})
    return record, metrics


def confirmation(data: dict, output: Path) -> dict:
    tests = data["a_tests"]
    names = ("native_hard_minus_raw_D300", "native_hard_minus_raw_D600",
             "native_hard_duration_interaction", "native_hard_minus_matched_random_D600")
    selected = tests[(tests.scorer == "lasso") & tests.primary].set_index("contrast").loc[list(names)].reset_index()
    selected.to_csv(output / "fig_confirmation_source.csv", index=False)
    fig = plt.figure(figsize=(6.6, 3.2), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[3.8, 1], wspace=0.03)
    ax, p_ax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    labels = ["D = 300: hard − raw", "D = 600: hard − raw", "Paired duration difference Γ", "D = 600: hard − matched"]
    forest(ax, labels)
    for y, row in selected.iterrows():
        point_interval(ax, y, row["mean"], row.ci95_low, row.ci95_high,
                       color=BLUE if y < 3 else PURPLE, marker=("o", "^", "s", "D")[y])
        p_ax.text(0.05, y, scientific(row.p_holm), va="center", fontsize=9)
    ax.set(xlim=difference_limits(selected.ci95_low, selected.ci95_high, selected["mean"]),
           xlabel="Paired difference in changing-edge temporal AUROC")
    p_ax.set(ylim=ax.get_ylim(), xlim=(0, 1)); p_ax.axis("off")
    p_ax.set_title("Holm4 p", loc="left")
    ax.set_title("Independent paired confirmation, n = 100", loc="left")
    fig.supxlabel("Pointwise 95% bootstrap intervals; the four tests form one frozen primary family", fontsize=8.5)
    return export(fig, output, "fig_confirmation",
                  "Four prespecified Lasso comparisons on 100 independent paired repeats: loss at D300, gain at D600, their paired duration difference, and D600 gain over size-matched random grouping.",
                  {"unit": "independent mechanism/noise repeat, n=100", "uncertainty": "pointwise 95% percentile bootstrap, 10000 resamples, seed20261024", "inference": "two-sided t tests, Holm4; OLS excluded from family"})


def b_plot_data(data: dict) -> pd.DataFrame:
    rows = []
    for duration, kind in CELLS:
        paired = data["b_paired"][(data["b_paired"].duration_mean == duration) & (data["b_paired"].schedule == kind)].sort_values("seed")
        diagnostic = data["b_diagnostics"][(data["b_diagnostics"].duration_mean == duration) & (data["b_diagnostics"].schedule == kind)].sort_values("seed")
        gate = data["b_gate"][(data["b_gate"].duration_mean == duration) & (data["b_gate"].schedule == kind)].iloc[0]
        row = {"duration_mean": duration, "schedule": kind, "label": f"Mean D={duration}  {KIND_LABEL[kind]}",
               "n_gate_observed": int(gate.n_gate_observed), "n_gate_accepted": int(gate.n_gate_accepted),
               "gate_acceptance": gate.gate_acceptance}
        for metric, values in (("forced", paired.forced_minus_raw), ("native", paired.native_minus_raw),
                               ("centre_switches", diagnostic.n_centre_switches), ("purity", diagnostic.window_purity_mean)):
            mean, low, high, n = interval(values)
            row.update({metric + "_mean": mean, metric + "_low": low, metric + "_high": high, metric + "_n": n})
        test = data["b_tests"][(data["b_tests"].family == "B_forced_minus_raw") &
                               (data["b_tests"].duration_mean == duration) & (data["b_tests"].schedule == kind)].iloc[0]
        for actual, expected in ((row["forced_mean"], test.mean_difference), (row["forced_low"], test.ci95_low),
                                 (row["forced_high"], test.ci95_high)):
            if not np.isclose(actual, expected, atol=1e-12, rtol=0, equal_nan=True):
                raise ValueError("Presentation recomputation disagrees with frozen B summary")
        row["forced_p_holm6"] = test.p_holm
        rows.append(row)
    return pd.DataFrame(rows)


def schedule_figure(plot: pd.DataFrame, output: Path) -> dict:
    plot.to_csv(output / "fig_schedule_source.csv", index=False)
    fig = plt.figure(figsize=(6.6, 4.1), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.55, 1, 1], wspace=0.08)
    axes = [fig.add_subplot(grid[0, k]) for k in range(3)]
    for index, (ax, metric, title) in enumerate(zip(axes, ("forced", "centre_switches", "purity"),
                                                   ("(a) Forced − raw", "(b) Centre switches", "(c) Window purity"))):
        forest(ax, plot.label.tolist() if index == 0 else [""] * 6, baseline=index == 0)
        for y, row in plot.iterrows():
            point_interval(ax, y, row[metric + "_mean"], row[metric + "_low"], row[metric + "_high"],
                           color=ORANGE if index == 0 else "#333333", marker="s" if index == 0 else "o")
        ax.axhline(2.5, color="#AAAAAA", linewidth=0.6)
        ax.set_title(title, loc="left")
        if index:
            ax.tick_params(axis="y", length=0)
    axes[0].set(xlabel="Temporal AUROC difference", xlim=difference_limits(plot.forced_low, plot.forced_high))
    axes[1].set(xlabel="Mean count / repeat", xlim=(0, max(12, plot.centre_switches_high.max() + 1)))
    axes[2].set(xlabel="Mean / repeat", xlim=(0.5, 1), xticks=[0.5, 0.75, 1])
    fig.suptitle("Schedule robustness: 50 independent repeats per cell, 23 windows per repeat", fontsize=9.5)
    fig.supxlabel("Intervals resample repeats; switch counts and purity are descriptive diagnostics", fontsize=8.5)
    return export(fig, output, "fig_schedule",
                  "Six schedule cells compare forced hard-minus-raw temporal AUROC and describe mean centre-switch count and mean within-repeat window purity. Repeats, not windows, define uncertainty.",
                  {"unit": "independent simulation repeat, n=50/cell", "uncertainty": "pointwise 95% paired-repeat percentile bootstrap, 10000 resamples, seed20261024", "inference": "forced differences are prespecified secondary Holm6; switch/purity intervals are descriptive"})


def gate_figure(plot: pd.DataFrame, output: Path) -> dict:
    plot.to_csv(output / "fig_gate_source.csv", index=False)
    fig = plt.figure(figsize=(6.6, 4.1), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[2.3, 1], wspace=0.07)
    ax, gates = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    forest(ax, plot.label.tolist()); forest(gates, [""] * 6, baseline=False)
    for y, row in plot.iterrows():
        point_interval(ax, y - 0.13, row.native_mean, row.native_low, row.native_high,
                       color=BLUE, marker="o", label="Native hard" if y == 0 else None)
        point_interval(ax, y + 0.13, row.forced_mean, row.forced_low, row.forced_high,
                       color=ORANGE, marker="s", label="Forced candidate" if y == 0 else None)
        if np.isfinite(row.gate_acceptance):
            gates.plot(row.gate_acceptance, y, marker="^", color="#333333", markersize=5.5)
            gates.annotate(f"{row.n_gate_accepted}/{row.n_gate_observed}", (row.gate_acceptance, y),
                           xytext=(7, 0), textcoords="offset points", va="center", fontsize=8.5)
        else:
            gates.text(0.02, y, "Not estimable", va="center", fontsize=8.5)
    ax.set(xlabel="Hard − raw temporal AUROC", xlim=difference_limits(
        plot.forced_low, plot.forced_high, plot.native_low, plot.native_high))
    gates.set(xlim=(-0.05, 1.4), xlabel="Gate acceptance", xticks=[0, 0.5, 1], xticklabels=["0%", "50%", "100%"])
    gates.tick_params(axis="y", length=0)
    for panel in (ax, gates):
        panel.axhline(2.5, color="#AAAAAA", linewidth=0.6)
    ax.set_title("(a) Native output and unchanged candidate", loc="left")
    gates.set_title("(b) Accepted / fitted", loc="left")
    ax.legend(loc="lower left", frameon=False, bbox_to_anchor=(0.0, 1.04), ncol=2, borderaxespad=0)
    fig.supxlabel("All planned repeats are retained; acceptance counts are descriptive", fontsize=8.5)
    return export(fig, output, "fig_gate",
                  "Native and forced hard-minus-raw temporal AUROC differences share six schedule rows. A separate axis reports gate counts without conditioning the performance analysis on acceptance.",
                  {"unit": "independent simulation repeat, n=50/cell", "uncertainty": "pointwise 95% repeat-bootstrap intervals for paired differences; gate fractions are counts, without inferential intervals", "selected_subgroups": "accepted-only performance omitted from main figure and retained in supplement"})


def table_protocol() -> str:
    return "\n".join([
        r"\begin{tabular}{llrrrl}", r"\toprule",
        r"Block & Schedule and duration & $T$ & $M$ & Repeats & Within-repeat pairing \\", r"\midrule",
        r"\shortstack[l]{Earlier\\exploratory} & \shortstack[l]{Fixed phase\\$D=150,300,600,900$} & 1200 & 7 & 20 & \shortstack[l]{Shared local scores\\across aggregators} \\",
        r"A: confirmation & \shortstack[l]{Fixed phase\\$D=300,600$} & 1200 & 7 & 100 & \shortstack[l]{Mechanism, history,\\innovations across $D$} \\",
        r"B: robustness & \shortstack[l]{Fixed/random phase, irregular\\$\bar D=300,600$} & 3600 & 23 & 50 & \shortstack[l]{Mechanism, history,\\innovations across six cells} \\",
        r"\bottomrule", r"\end{tabular}"])


def table_main(data: dict) -> str:
    frame = data["a_summary"]; frame = frame[frame.scorer == "lasso"]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{$D=300$} & \multicolumn{2}{c}{$D=600$} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Method & Pooled & Changing-edge & Pooled & Changing-edge \\", r"\midrule"]
    for method in METHODS:
        cells = []
        for duration in (300, 600):
            row = frame[(frame.method == method) & (frame.D == duration)].iloc[0]
            cells.extend([number(row.pooled_temporal_auroc_mean), number(row.temporal_delta_auroc_mean)])
        lines.append(METHOD_LABEL[method] + " & " + " & ".join(cells) + r" \\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"])


def factor_data(data: dict) -> pd.DataFrame:
    points = []
    for study, duration, kind in [("A", 300, "fixed_phase"), ("A", 600, "fixed_phase"),
                                   *[("B", d, k) for d, k in CELLS]]:
        frame = data["a_detail"] if study == "A" else data["b_detail"]
        if study == "A":
            frame = frame[(frame.scorer == "lasso") & (frame.D == duration)]
            key = "repeat_seed"
        else:
            frame = frame[(frame.duration_mean == duration) & (frame.schedule == kind)]
            key = "seed"
        candidate = frame[frame.method == "forced_hard"].set_index(key)
        raw = frame[frame.method == "raw_window"].set_index(key).temporal_delta_auroc
        selected = candidate[["J", "C_edge", "J_times_C", "temporal_delta_auroc"]].join(raw.rename("raw"))
        finite = np.isfinite(selected.to_numpy(dtype=float)).all(axis=1)
        group = selected[finite]
        points.append({"study": study, "duration": duration, "schedule": kind,
                       "n_planned": 100 if study == "A" else 50, "n": len(group),
                       "J": group.J.mean(), "C_edge": group.C_edge.mean(), "J_times_C": group.J_times_C.mean(),
                       "raw_reference": 2 * group.raw.mean() - 1,
                       "forced_minus_raw": (group.temporal_delta_auroc - group.raw).mean(),
                       "identity_max_gap": float(np.max(np.abs(group.temporal_delta_auroc - .5 - .5 * group.J_times_C))) if len(group) else np.nan})
    return pd.DataFrame(points)


def table_factors(frame: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Study/cell & $D$ or $\bar D$ & $n$ & $\overline J$ & $\overline C_{\rm edge}$ & $\overline{JC_{\rm edge}}$ & $2\overline T_{\rm raw}-1$ & Forced $\Delta$ \\", r"\midrule"]
    for row in frame.itertuples(index=False):
        lines.append(f"{row.study}: {KIND_LABEL[row.schedule]} & {row.duration} & {row.n} & " +
                     " & ".join(number(v) for v in (row.J, row.C_edge, row.J_times_C, row.raw_reference)) +
                     " & " + number(row.forced_minus_raw, signed=True) + r" \\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"])


def table_ess(data: dict) -> str:
    frame = data["ess"]
    labels = {"stcg_v1": "STCG-v1", "window_max": "Window max", "pelt_var_lasso": "PELT + refit", "lag_corr": "Lagged correlation"}
    lines = [r"\begin{tabular}{llrr}", r"\toprule",
             r"Method & Expert edges & Forward-fill AUROC & Four-imputation range \\", r"\midrule"]
    for method, label in labels.items():
        for category, edge_label in (("all_edges", "All"), ("strong_only", "Strong")):
            group = frame[(frame.method == method) & (frame.binarisation == category)]
            reference = group[group.imputation == "ffill"]
            if len(reference) != 1 or group.imputation.nunique() != 4:
                raise ValueError("ESS compact table needs all four observed imputations")
            lines.append(f"{label} & {edge_label} & {number(reference.auroc.iloc[0])} & "
                         f"[{number(group.auroc.min())}, {number(group.auroc.max())}]" + r" \\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"])


def build(output: Path = OUT) -> dict:
    data = inputs()
    output = Path(output).resolve()
    if output.is_relative_to((ROOT / "results/revision2").resolve()):
        raise ValueError("This builder never writes the historical revision2 outputs")
    if output in {A.resolve(), B.resolve()} or output.is_relative_to(A.resolve()) or output.is_relative_to(B.resolve()):
        raise ValueError("Presentation assets must remain outside frozen study output directories")
    output.mkdir(parents=True, exist_ok=True)
    before = {key: sha(path) for key, path in data["paths"].items()}
    factors = factor_data(data); factors.to_csv(output / "factor_table_source.csv", index=False)
    tables = {"tab_protocol.tex": table_protocol(), "tab_main_results.tex": table_main(data),
              "tab_factors.tex": table_factors(factors), "tab_ess_compact.tex": table_ess(data)}
    for name, content in tables.items():
        (output / name).write_text(content + "\n", encoding="utf-8")
    with plt.rc_context(STYLE):
        first, toy = target_example(output)
        second = confirmation(data, output)
        plot = b_plot_data(data)
        third = schedule_figure(plot, output)
        fourth = gate_figure(plot, output)
    after = {key: sha(path) for key, path in data["paths"].items()}
    if before != after:
        raise RuntimeError("Retained source outputs changed during presentation export")
    outputs = [*tables, *[f"{name}.{suffix}" for name in
               ("fig_target_example", "fig_confirmation", "fig_schedule", "fig_gate") for suffix in ("pdf", "png")],
               "target_example.csv", "target_example_metrics.csv", "factor_table_source.csv",
               "fig_confirmation_source.csv", "fig_schedule_source.csv", "fig_gate_source.csv"]
    manifest = {
        "asset_builder_sha256": sha(Path(__file__)),
        "source_sha256": {path.relative_to(ROOT).as_posix(): before[key] for key, path in data["paths"].items()},
        "output_sha256": {name: sha(output / name) for name in outputs},
        "main_table_count": 4, "main_figure_count": 4, "figures": [first, second, third, fourth],
        "toy": toy, "units": {"A": "100 independent paired mechanism/history/noise repeats; 7 windows per fit",
                                "B": "50 separate independent repeats paired over six schedules; 23 windows per fit",
                                "matched_random": "partition-specific metrics averaged within repeat, not extra replicates",
                                "factor_table": "jointly evaluable repeat set per row; mean per-repeat product, not product of means",
                                "ESS": "one observed hour; four-imputation range is sensitivity, not a confidence interval"},
        "uncertainty": "pointwise 95% percentile bootstrap, 10000 repeat resamples, seed20261024; missing CI not replaced by zero",
        "publisher_export": {"journal": "Information Sciences", "phase": "initial manuscript preparation",
                             "figure_width_inches": 6.6, "png_dpi": 600, "pdf_fonts": "embedded TrueType (Type42 request)",
                             "status": "general scientific figure export; no journal-compliance certification",
                             "live_guide_check": "2026-10-24, ScienceDirect guide-for-authors returned HTTP403"},
        "workflow_reference": SKILL_CITATION,
        "experiments_fitted": 0, "retained_source_hashes_unchanged": True,
    }
    (output / "assets_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    result = build(args.output_dir)
    print(json.dumps({"tables": result["main_table_count"], "figures": result["main_figure_count"],
                      "experiments_fitted": 0, "retained_source_hashes_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
