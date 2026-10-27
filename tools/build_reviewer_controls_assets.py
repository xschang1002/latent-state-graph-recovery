"""Deterministic presentation of completed post-review controls and robustness.

Read retained summaries only. No generation, fitting, parameter selection or
replacement of the original A/B results. All plots have paired-repeat pointwise
bootstrap intervals, and every fixed lambda/window remains in source files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("results/information_sciences")
DEST = BASE / "reviewer_assets"
SOURCE_FILES = (
    "reviewer_assets/prior_results_sha256.json",
    "score_preserving_controls/summary.csv", "score_preserving_controls/contrasts.csv",
    "score_preserving_controls/detail.csv", "score_preserving_controls/diagnostics.json",
    "temporal_localisation/summary.csv", "temporal_localisation/paired_vs_raw.csv",
    "temporal_localisation/detail.csv", "symmetric_sensitivity/symmetric_sensitivity_summary.csv",
    "symmetric_sensitivity/symmetric_sensitivity_gate.csv",
    "symmetric_controls/summary.csv", "symmetric_controls/contrasts.csv",
    "symmetric_localisation/summary.csv", "symmetric_localisation/paired_vs_raw.csv",
)
FIGURES = ("fig_fallback", "fig_symmetric", "fig_localisation")
TABLES = ("tab_score_controls", "tab_score_controls_aux", "tab_fallback_full",
          "tab_symmetric_full", "tab_localisation_full", "tab_stable_changes")
PROSE_FILES = ("reviewer_assets/results_controls.tex", "reviewer_assets/results_symmetric.tex",
               "reviewer_assets/results_localisation.tex", "abstract.tex", "conclusions.tex")
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#007F60"
LABELS = {"raw_window": "Raw", "forced_hard": "Hard", "native_hard": "Native",
          "raw_fallback": "Raw fallback", "residual_hard_l050": "Residual 0.5"}
SCHEDULES = ("fixed_phase", "random_phase", "irregular")
SCHEDULE_LABEL = {"fixed_phase": "Fixed", "random_phase": "Random", "irregular": "Irregular"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root: Path = ROOT) -> dict[str, pd.DataFrame | dict]:
    paths = {name: root / BASE / name for name in SOURCE_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Completed post-review evidence required: " + ", ".join(missing))
    return {name: (json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json"
                   else pd.read_csv(path, float_precision="round_trip")) for name, path in paths.items()}


def one(frame: pd.DataFrame, **where) -> pd.Series:
    selected = frame
    for key, value in where.items():
        selected = selected[selected[key] == value]
    if len(selected) != 1:
        raise ValueError(f"Expected one display record for {where}, found {len(selected)}")
    return selected.iloc[0]


def fmt(value, signed=False) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def interval(row) -> str:
    if not np.isfinite(row["mean"]):
        return "--"
    return "$" + fmt(row["mean"], True) + "\\;[" + fmt(row["ci95_low"]) + "," + fmt(row["ci95_high"]) + "]$"


def tabular(header: str, rows: list[str], spec: str) -> str:
    return "\n".join([r"\begin{tabular}{" + spec + "}", r"\toprule", header + r" \\",
                       r"\midrule", *[line + r" \\" for line in rows], r"\bottomrule", r"\end{tabular}"]) + "\n"


def control_table(data: dict, scorer="lasso") -> str:
    s = data["score_preserving_controls/summary.csv"]
    methods = ["raw_window", "forced_hard", "shrink_l025", "shrink_l050", "shrink_l075",
               "residual_hard_l025", "residual_hard_l050", "residual_hard_l075"]
    rows = []
    for method in methods:
        label = LABELS[method] if method in LABELS else (
            ("Shrink " if method.startswith("shrink") else "Residual ") + str(int(method[-3:]) / 100))
        values = [one(s, study="A", scorer=scorer, duration=d, method=method, metric=metric)["mean"]
                  for d in (300, 600) for metric in ("pooled_temporal_auroc", "temporal_delta_auroc")]
        rows.append(label + " & " + " & ".join(fmt(v) for v in values))
    return tabular(r"Emission & Pooled, $D=300$ & State, $300$ & Pooled, $600$ & State, $600$", rows, "lrrrr")


def fallback_table(data: dict) -> str:
    c, detail = data["score_preserving_controls/contrasts.csv"], data["score_preserving_controls/detail.csv"]
    rows = []
    for duration in (300, 600):
        for schedule in SCHEDULES:
            raw = detail[(detail.study == "B") & (detail.duration == duration) &
                         (detail.schedule == schedule) & (detail.method == "raw_window")]
            accepted = int(raw.gate_accepted.sum())
            values = [interval(one(c, study="B", duration=duration, schedule=schedule, method=m,
                         reference="raw_window", metric="temporal_delta_auroc"))
                      for m in ("native_hard", "raw_fallback", "forced_hard")]
            rows.append(f"{duration} & {SCHEDULE_LABEL[schedule]} & {accepted}/{len(raw)} & " + " & ".join(values))
    return tabular(r"$D$ & Schedule & Accepted & Native $\Delta$ & Raw-fallback $\Delta$ & Forced $\Delta$", rows, "rllrrr")


def symmetric_table(data: dict) -> str:
    s = data["symmetric_sensitivity/symmetric_sensitivity_summary.csv"]
    g = data["symmetric_sensitivity/symmetric_sensitivity_gate.csv"]
    c = data["symmetric_controls/contrasts.csv"]
    rows = []
    for duration in (300, 600):
        for h, stride in ((200, 100), (300, 150), (400, 200), (300, 75), (300, 300)):
            cell = f"D{duration}_h{h}_s{stride}"
            hard = one(s, cell=cell, method="forced_hard", metric="temporal_delta_auroc", comparison="minus_raw")
            gate = one(g, cell=cell)
            fallback = one(c, cell=cell, method="raw_fallback", reference="raw_window", metric="temporal_delta_auroc")
            rows.append(f"{duration} & {h}/{stride} & {int(hard.n_windows)} & "
                        f"{int(gate.n_accepted)}/{int(gate.n_planned)} & {interval(hard)} & {interval(fallback)}")
    return tabular(r"$D$ & $h/s$ & $M$ & Accepted & Forced $\Delta$ & Raw-fallback $\Delta$", rows, "rlrlrr")


def local_conditions():
    for d in (300, 600):
        yield "A", d, "fixed_phase", f"D{d}_fixed_phase", 300, 150
    for d in (300, 600):
        for schedule in SCHEDULES:
            yield "B", d, schedule, f"D{d}_{schedule}", 300, 150
    for d in (300, 600):
        for h, stride in ((200, 100), (300, 150), (400, 200), (300, 75), (300, 300)):
            yield "C", d, "fixed_phase", f"D{d}_h{h}_s{stride}", h, stride


def local_frames(data, study):
    prefix = "symmetric_localisation" if study == "C" else "temporal_localisation"
    return data[prefix + "/summary.csv"], data[prefix + "/paired_vs_raw.csv"]


def short_condition(study, duration, schedule, h, stride):
    if study == "C":
        return f"C {duration}, {h}/{stride}"
    return f"{study} {duration}, {SCHEDULE_LABEL[schedule]}"


def localisation_table(data) -> str:
    rows = []
    for study, d, schedule, cell, h, stride in local_conditions():
        s, c = local_frames(data, study)
        values = []
        count = None
        for metric in ("boundary_positive_change_ap", "boundary_slot_f1"):
            raw = one(s, study=study, scorer="lasso", cell=cell, method="raw_window", metric=metric)
            hard = one(c, study=study, scorer="lasso", cell=cell, method="forced_hard", metric=metric)
            values.extend((fmt(raw["mean"]), interval(hard)))
            if metric == "boundary_positive_change_ap":
                count = int(hard.n_paired)
        rows.append(short_condition(study, d, schedule, h, stride) + f" & {count} & " + " & ".join(values))
    return tabular(r"Condition ($D,h/s$) & $n_{\rm AP}$ & Raw AP & Hard--raw AP & Raw F1 & Hard--raw F1", rows, "lrrrrr")


def stable_table(data) -> str:
    rows = []
    for study, d, schedule, cell, h, stride in local_conditions():
        if study == "C" and (h, stride) != (300, 150):
            continue
        s, _ = local_frames(data, study)
        values = []
        for metric in ("persistent_false_change_rate", "absent_false_change_rate"):
            for method in ("raw_window", "forced_hard", "raw_fallback"):
                row = one(s, study=study, scorer="lasso", cell=cell, method=method, metric=metric)
                values.append(fmt(row["mean"]))
        rows.append(short_condition(study, d, schedule, h, stride) + " & " + " & ".join(values))
    return tabular(r"Condition & Persistent: raw & hard & fallback & Absent: raw & hard & fallback", rows, "lrrrrrr")


def tables(data) -> dict[str, str]:
    return {"tab_score_controls": control_table(data), "tab_score_controls_aux": control_table(data, "ols"),
            "tab_fallback_full": fallback_table(data), "tab_symmetric_full": symmetric_table(data),
            "tab_localisation_full": localisation_table(data), "tab_stable_changes": stable_table(data)}


def point(ax, x, y, low, high, color, marker, label=None):
    if not np.isfinite(x):
        return
    ax.errorbar(x, y, xerr=[[max(0., x-low)], [max(0., high-x)]], marker=marker,
                color=color, mec=color, linestyle="none", capsize=2.5, markersize=4.5,
                linewidth=1.1, label=label)


def fallback_figure(data):
    c = data["score_preserving_controls/contrasts.csv"]
    detail = data["score_preserving_controls/detail.csv"]
    fig, ax = plt.subplots(figsize=(7.2, 4.9), layout="constrained")
    labels = []
    for i, (d, schedule) in enumerate((d, q) for d in (300, 600) for q in SCHEDULES):
        raw = detail[(detail.study == "B") & (detail.duration == d) &
                     (detail.schedule == schedule) & (detail.method == "raw_window")]
        labels.append(f"D={d}  {SCHEDULE_LABEL[schedule]}  ({int(raw.gate_accepted.sum())}/{len(raw)})")
        for method, shift, color, marker in (("native_hard", -.20, ORANGE, "s"),
                ("forced_hard", 0, BLUE, "o"), ("raw_fallback", .20, GREEN, "^")):
            row = one(c, study="B", duration=d, schedule=schedule, method=method,
                      reference="raw_window", metric="temporal_delta_auroc")
            point(ax, row["mean"], i + shift, row.ci95_low, row.ci95_high,
                  color, marker, LABELS[method] if i == 0 else None)
    ax.axvline(0, color="#777777", lw=.8)
    ax.set(yticks=range(6), yticklabels=labels, xlabel="Paired change in temporal state-discrimination AUROC")
    ax.invert_yaxis(); ax.legend(loc="lower left", frameon=False, ncols=3)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def symmetric_figure(data):
    s = data["symmetric_sensitivity/symmetric_sensitivity_summary.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.1), layout="constrained", sharey=True)
    for d, color, marker in ((300, BLUE, "o"), (600, ORANGE, "s")):
        for ax, settings in ((axes[0], ((200,100),(300,150),(400,200))),
                             (axes[1], ((300,300),(300,150),(300,75)))):
            xx, yy = [], []
            for h, stride in settings:
                row = one(s, cell=f"D{d}_h{h}_s{stride}", method="forced_hard",
                          metric="temporal_delta_auroc", comparison="minus_raw")
                x = h/d if ax is axes[0] else row.n_windows
                ax.errorbar(x, row["mean"], yerr=[[row["mean"]-row.ci95_low], [row.ci95_high-row["mean"]]],
                            marker=marker, color=color, capsize=2.5, markersize=4.5, linestyle="none")
                xx.append(x); yy.append(row["mean"])
                if ax is axes[0]:
                    ax.annotate(f"M={int(row.n_windows)}", (x, row["mean"]), xytext=(4, 5),
                                textcoords="offset points", fontsize=7, color=color)
            ax.plot(xx, yy, color=color, marker=marker, lw=.9, label=f"D={d}")
    axes[0].set(xlabel="h / D  (stride = h / 2)", ylabel="Forced hard minus raw state AUROC",
                title="Proportional window/stride change")
    axes[1].set(xlabel="Window count M  (h = 300)", title="Stride change at fixed h", xticks=[12,23,45])
    for ax in axes:
        ax.axhline(0, color="#777777", lw=.8); ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)
    return fig


def localisation_figure(data):
    conditions = [row for row in local_conditions() if row[0] != "C" or row[4:] == (300,150)]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 6.2), layout="constrained", sharey=True)
    for i, (study, d, schedule, cell, h, stride) in enumerate(conditions):
        _, c = local_frames(data, study)
        for ax, metric in zip(axes, ("boundary_positive_change_ap", "boundary_slot_f1")):
            for method, shift, color, marker in (("forced_hard", -.2, BLUE, "o"),
                    ("raw_fallback", 0, GREEN, "^"), ("residual_hard_l050", .2, ORANGE, "s")):
                row = one(c, study=study, scorer="lasso", cell=cell, method=method, metric=metric)
                point(ax, row["mean"], i+shift, row.ci95_low, row.ci95_high, color, marker,
                      LABELS[method] if i == 0 else None)
    labels = [short_condition(a,b,c,e,f) for a,b,c,_,e,f in conditions]
    axes[0].set(yticks=range(len(labels)), yticklabels=labels, title="Positive-change AP", xlabel="Paired change from raw")
    axes[1].set(title="Boundary-slot F1", xlabel="Paired change from raw")
    axes[0].invert_yaxis()
    for ax in axes:
        ax.axvline(0, color="#777777", lw=.8); ax.spines[["top","right"]].set_visible(False)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="outside lower center", ncols=3, fontsize=9)
    return fig


def build(root: Path = ROOT) -> dict:
    data = load(root); output = root / DEST; output.mkdir(parents=True, exist_ok=True)
    before = {(BASE/name).as_posix(): sha(root/BASE/name) for name in (*SOURCE_FILES, *PROSE_FILES)}
    generated = []
    with plt.rc_context({"font.size": 9, "axes.titlesize": 10, "pdf.fonttype": 42,
                         "ps.fonttype":42, "savefig.facecolor":"white"}):
        for name, factory in zip(FIGURES, (fallback_figure, symmetric_figure, localisation_figure)):
            fig = factory(data)
            fig.savefig(output/f"{name}.pdf", metadata={"CreationDate":None,"ModDate":None,"Author":None,
                                                      "Creator":None,"Producer":None})
            fig.savefig(output/f"{name}.png", dpi=600)
            plt.close(fig); generated.extend((name+".pdf", name+".png"))
    for name, text in tables(data).items():
        (output/f"{name}.tex").write_text(text,encoding="utf-8",newline="\n"); generated.append(name+".tex")
    frames = []
    for name, frame in data.items():
        if isinstance(frame,pd.DataFrame) and name.endswith(("summary.csv","contrasts.csv","paired_vs_raw.csv")):
            frames.append(frame.assign(source_file=name))
    pd.concat(frames,ignore_index=True,sort=False).to_csv(output/"display_source.csv",index=False,lineterminator="\n")
    generated.append("display_source.csv")
    generated.extend(("prior_results_sha256.json", "results_controls.tex", "results_symmetric.tex", "results_localisation.tex"))
    assert before == {(BASE/name).as_posix():sha(root/BASE/name) for name in (*SOURCE_FILES, *PROSE_FILES)}
    manifest = {"role":"post-review descriptive presentation; no new fits or selected lambda/window",
        "source_sha256":{**before,"tools/build_reviewer_controls_assets.py":sha(root/"tools/build_reviewer_controls_assets.py")},
        "sha256":{name:sha(output/name) for name in generated},"figures":list(FIGURES),"tables":list(TABLES),
        "uncertainty":"pointwise paired-repeat 95% percentile bootstrap intervals; fixed 10000 resamples",
        "scope":"A/B controls reanalyse saved predictions; C is separate symmetric-generator/window robustness",
        "missing":"undefined AP slots are omitted with retained coverage; no zero substitution",
        "export":{"pdf":"vector, embedded TrueType fonts; no creation/modification dates","png_dpi":600},
        "workflow_reference":"Kassis et al. (2026), Scientific Agent Skills, doi:10.48550/arXiv.2609.00065"}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return manifest


def verify_assets(root: Path = ROOT) -> dict:
    output = root/DEST
    manifest = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    expected_sources = {(BASE/name).as_posix() for name in (*SOURCE_FILES, *PROSE_FILES)}|{"tools/build_reviewer_controls_assets.py"}
    if set(manifest["source_sha256"]) != expected_sources:
        raise ValueError("Incomplete reviewer asset source closure")
    for name, expected in manifest["source_sha256"].items():
        target=(root/name).resolve()
        if not target.is_relative_to(root.resolve()) or sha(target)!=expected:
            raise ValueError(f"Reviewer asset source checksum mismatch: {name}")
    expected_outputs=({n+ext for n in FIGURES for ext in (".pdf",".png")}|
                      {n+".tex" for n in TABLES}|{"display_source.csv", "prior_results_sha256.json",
                       "results_controls.tex", "results_symmetric.tex", "results_localisation.tex"})
    if set(manifest["sha256"]) != expected_outputs:
        raise ValueError("Incomplete reviewer asset output inventory")
    for name, expected in manifest["sha256"].items():
        target=(output/name).resolve()
        if not target.is_relative_to(output.resolve()) or sha(target)!=expected:
            raise ValueError(f"Reviewer asset output checksum mismatch: {name}")
    for name, expected in tables(load(root)).items():
        if (output/f"{name}.tex").read_text(encoding="utf-8")!=expected:
            raise ValueError(f"Reviewer table values no longer match retained outputs: {name}")
    return {"figures":len(FIGURES),"tables_reconstructed":len(TABLES),"outputs_verified":len(expected_outputs),
            "source_files":len(expected_sources),"new_fits":0}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    print(json.dumps(verify_assets() if args.verify else build(),indent=2))
