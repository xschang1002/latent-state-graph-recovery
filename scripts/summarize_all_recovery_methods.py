"""Grand summary over all synthetic recovery detail CSVs.

Merges every tracked synthetic detail file (STCG recovery sweep plus the seven
baseline-family files), de-duplicates shared (generator, seed, method) rows
after verifying they agree numerically, and writes:

- ``synthetic_all_methods_summary.csv`` / ``.md`` / ``synthetic_all_methods_table.tex``
  (rows = generator x method, mean +/- std for edge AUC, pair AUC, P@K, SHD,
  lag accuracy, plus mean fit time where the detail file tracked it).
- ``synthetic_switching_paired_matrix.csv`` / ``.md``: paired comparison of
  STCG-v1 against every other method on ``switching_var`` for all metrics,
  reusing the exact statistics of ``compare_recovery_methods.py`` (paired t /
  Wilcoxon one-sided "greater" p, bootstrap CI of the improvement, Cohen dz,
  wins/ties/losses).
- ``synthetic_all_methods_edge_auc.png``: per-generator dot + std panels.

The script also sanity-checks seed coverage and reconciles its aggregates
against each per-file ``*_summary.csv`` and the already tracked per-baseline
paired-comparison summaries. All checks are reported in the stdout JSON; any
discrepancy makes the script exit non-zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import fmt_na as fmt  # noqa: E402
from compare_recovery_methods import (  # noqa: E402
    DEFAULT_METRICS,
    LOWER_IS_BETTER,
    compare,
)

ROOT = Path(__file__).resolve().parents[1]

DETAIL_FILES = [
    "synthetic_recovery_detail.csv",
    "synthetic_pcmci_baseline_detail.csv",
    "synthetic_pcmciplus_baseline_detail.csv",
    "synthetic_msvar_baseline_detail.csv",
    "synthetic_rpcmci_baseline_detail.csv",
    "synthetic_neural_granger_baseline_detail.csv",
    "synthetic_dynotears_baseline_detail.csv",
]

METHOD_A = "STCG-v1"
MATRIX_GENERATOR = "switching_var"

# method_b label -> tracked per-baseline paired summary CSV that the matrix
# must reconcile with exactly.
EXISTING_PAIRED_FILES = {
    "DynVAR-Lasso": "synthetic_recovery_paired_comparison_summary.csv",
    "DynGranger-F": "synthetic_recovery_paired_granger_comparison_summary.csv",
    "PCMCI-ParCorr": "synthetic_recovery_paired_pcmci_comparison_summary.csv",
    "PCMCI+": "synthetic_pcmciplus_paired_comparison_summary.csv",
    "MSVAR": "synthetic_msvar_paired_comparison_summary.csv",
    "Regime-PCMCI": "synthetic_rpcmci_paired_comparison_summary.csv",
    "cMLP": "synthetic_neural_granger_paired_comparison_summary.csv",
    "DYNOTEARS": "synthetic_dynotears_paired_comparison_summary.csv",
}

SUMMARY_FILES = {
    "synthetic_recovery_detail.csv": "synthetic_recovery_summary.csv",
    "synthetic_pcmci_baseline_detail.csv": "synthetic_pcmci_baseline_summary.csv",
    "synthetic_pcmciplus_baseline_detail.csv": "synthetic_pcmciplus_baseline_summary.csv",
    "synthetic_msvar_baseline_detail.csv": "synthetic_msvar_baseline_summary.csv",
    "synthetic_rpcmci_baseline_detail.csv": "synthetic_rpcmci_baseline_summary.csv",
    "synthetic_neural_granger_baseline_detail.csv": "synthetic_neural_granger_baseline_summary.csv",
    "synthetic_dynotears_baseline_detail.csv": "synthetic_dynotears_baseline_summary.csv",
}

METRIC_LABELS = {
    "edge_auc": "Edge AUC",
    "pair_auc": "Pair AUC",
    "precision_at_k": "P@K",
    "shd": "SHD",
    "lag_accuracy": "Lag Acc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--output-prefix", default="synthetic_all_methods_summary")
    parser.add_argument("--matrix-prefix", default="synthetic_switching_paired_matrix")
    parser.add_argument("--figure-name", default="synthetic_all_methods_edge_auc.png")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--skip-figure", action="store_true")
    parser.add_argument("--skip-matrix", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def merge_details(
    tables_dir: Path, detail_files: list[str]
) -> tuple[list[dict[str, str]], list[str], dict[str, list[int]]]:
    """Merge detail CSVs, verifying that duplicated keys agree on every metric."""

    merged: dict[tuple[str, str, str], dict[str, str]] = {}
    problems: list[str] = []
    seeds_by_file: dict[str, list[int]] = {}
    for name in detail_files:
        path = tables_dir / name
        rows = read_rows(path)
        seeds_by_file[name] = sorted({int(row["seed"]) for row in rows})
        for row in rows:
            key = (row["generator"], row["seed"], row["method"])
            if key in merged:
                previous = merged[key]
                for metric in DEFAULT_METRICS:
                    if abs(float(previous[metric]) - float(row[metric])) > 1e-12:
                        problems.append(
                            f"metric mismatch for {key} in {name}: "
                            f"{metric} {previous[metric]} vs {row[metric]}"
                        )
            else:
                merged[key] = row
    rows = sorted(merged.values(), key=lambda r: (r["generator"], r["method"], int(r["seed"])))
    return rows, problems, seeds_by_file


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    value_mean = mean(values)
    return math.sqrt(sum((value - value_mean) ** 2 for value in values) / (len(values) - 1))


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["generator"], row["method"]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (generator, method), items in sorted(groups.items()):
        summary: dict[str, object] = {"generator": generator, "method": method, "n": len(items)}
        for metric in DEFAULT_METRICS:
            values = [float(item[metric]) for item in items]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = sample_std(values)
        fit_values = [
            float(item["fit_seconds"])
            for item in items
            if item.get("fit_seconds") not in {None, ""}
        ]
        summary["fit_seconds_mean"] = mean(fit_values) if fit_values else math.nan
        summary_rows.append(summary)
    return summary_rows


def reconcile_with_per_file_summaries(
    tables_dir: Path, detail_files: list[str], summary_rows: list[dict[str, object]], tol: float = 1e-9
) -> tuple[int, list[str]]:
    lookup = {(row["generator"], row["method"]): row for row in summary_rows}
    checked = 0
    problems: list[str] = []
    for detail_name in detail_files:
        summary_name = SUMMARY_FILES[detail_name]
        path = tables_dir / summary_name
        if not path.is_file():
            problems.append(f"missing per-file summary {summary_name}")
            continue
        for row in read_rows(path):
            key = (row["generator"], row["method"])
            merged = lookup.get(key)
            if merged is None:
                problems.append(f"{summary_name} row {key} absent from merged summary")
                continue
            if int(row["n"]) != int(merged["n"]):
                problems.append(f"{summary_name} {key}: n {row['n']} vs merged {merged['n']}")
            for metric in DEFAULT_METRICS:
                for stat in ("mean", "std"):
                    reference = float(row[f"{metric}_{stat}"])
                    value = float(merged[f"{metric}_{stat}"])
                    if abs(reference - value) > tol:
                        problems.append(
                            f"{summary_name} {key} {metric}_{stat}: {reference} vs {value}"
                        )
                    checked += 1
    return checked, problems


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["generator", "method", "n"]
    for metric in DEFAULT_METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    fieldnames.append("fit_seconds_mean")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit_cell(value: object) -> str:
    return fmt(value, 1)


def display_method(method: object) -> str:
    """Preserve CSV identities while correcting the historical model name."""
    return "MSAR-X (legacy)" if method == "MSVAR" else str(method)


def write_summary_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Synthetic Recovery — All Methods Summary",
        "",
        "Merged over all tracked detail CSVs (STCG recovery sweep plus PCMCI-ParCorr,",
        "PCMCI+, legacy MSAR-X (raw identifier MSVAR), Regime-PCMCI, cMLP and DYNOTEARS baseline files); shared seeds 0-19.",
        "The legacy MSAR-X exogenous-coefficient score is not a direct joint VAR(L) coefficient; its results do not substantiate superiority over joint MSVAR.",
        "",
        "| Generator | Method | N | Edge AUC | Pair AUC | P@K | SHD | Lag Acc | Fit Time (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {generator} | {method} | {n} | {eauc} | {pauc} | {pk} | {shd} | {lag} | {fit} |".format(
                generator=row["generator"],
                method=display_method(row["method"]),
                n=row["n"],
                eauc=f"{fmt(row['edge_auc_mean'], 3)} +/- {fmt(row['edge_auc_std'], 3)}",
                pauc=f"{fmt(row['pair_auc_mean'], 3)} +/- {fmt(row['pair_auc_std'], 3)}",
                pk=f"{fmt(row['precision_at_k_mean'], 3)} +/- {fmt(row['precision_at_k_std'], 3)}",
                shd=f"{fmt(row['shd_mean'], 1)} +/- {fmt(row['shd_std'], 1)}",
                lag=f"{fmt(row['lag_accuracy_mean'], 3)} +/- {fmt(row['lag_accuracy_std'], 3)}",
                fit=fit_cell(row["fit_seconds_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "Footnotes:",
            "",
            "- Regime-PCMCI on `nonlinear_var` has N=19: the seed-18 fit failed (see",
            "  `synthetic_rpcmci_baseline_detail.csv`); all other generator x method cells use N=20.",
            "- Fit Time (s) is the per-fit mean over seeds and is only tracked for the",
            "  legacy MSAR-X, Regime-PCMCI, cMLP and DYNOTEARS baseline runs (`fit_seconds` column);",
            "  `n/a` elsewhere.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_keys(rows: list[dict[str, object]]) -> set[tuple[str, str, str]]:
    """(generator, metric, method) keys holding the best mean per generator x metric."""

    best: set[tuple[str, str, str]] = set()
    generators = sorted({str(row["generator"]) for row in rows})
    for generator in generators:
        subset = [row for row in rows if row["generator"] == generator]
        for metric in DEFAULT_METRICS:
            key = f"{metric}_mean"
            if metric in LOWER_IS_BETTER:
                winner = min(subset, key=lambda row: float(row[key]))
            else:
                winner = max(subset, key=lambda row: float(row[key]))
            best.add((generator, metric, str(winner["method"])))
    return best


def metric_cell(row: dict[str, object], metric: str, best: set[tuple[str, str, str]]) -> str:
    places = 1 if metric == "shd" else 3
    text = f"{fmt(row[f'{metric}_mean'], places)} $\\pm$ {fmt(row[f'{metric}_std'], places)}"
    if (str(row["generator"]), metric, str(row["method"])) in best:
        return f"\\textbf{{{text}}}"
    return text


def latex_text(value: object) -> str:
    """Escape plain CSV fields without interpreting them as LaTeX markup."""
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def write_summary_latex(path: Path, rows: list[dict[str, object]]) -> None:
    best = best_keys(rows)
    lines = [
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Generator & Method & Edge AUC & Pair AUC & P@K & SHD & Lag Acc & Fit Time (s) \\\\",
        "\\midrule",
    ]
    previous_generator = None
    for row in rows:
        if previous_generator is not None and row["generator"] != previous_generator:
            lines.append("\\midrule")
        previous_generator = row["generator"]
        cells = [metric_cell(row, metric, best) for metric in DEFAULT_METRICS]
        lines.append(
            f"{latex_text(row['generator'])} & {latex_text(display_method(row['method']))} & "
            + " & ".join(cells)
            + f" & {fit_cell(row['fit_seconds_mean'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_paired_matrix(
    rows: list[dict[str, str]], bootstrap_samples: int, seed: int
) -> list[dict[str, object]]:
    """STCG-v1 vs every other method on switching_var, all metrics."""

    methods = sorted(
        {row["method"] for row in rows if row["generator"] == MATRIX_GENERATOR and row["method"] != METHOD_A}
    )
    summary_rows: list[dict[str, object]] = []
    for method_b in methods:
        args = argparse.Namespace(
            generators=[MATRIX_GENERATOR],
            method_a=METHOD_A,
            method_b=method_b,
            metrics=DEFAULT_METRICS,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        _, per_method_summary = compare(args, rows)
        summary_rows.extend(per_method_summary)
    return summary_rows


def reconcile_with_existing_paired(
    tables_dir: Path, matrix_rows: list[dict[str, object]], tol: float = 1e-9
) -> tuple[int, list[str]]:
    fields = [
        "n",
        "mean_a",
        "mean_b",
        "mean_improvement",
        "std_improvement",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "paired_t_greater_p",
        "wilcoxon_greater_p",
        "cohen_dz",
        "wins",
        "ties",
        "losses",
    ]
    lookup = {
        (str(row["generator"]), str(row["metric"]), str(row["method_b"])): row for row in matrix_rows
    }
    checked = 0
    problems: list[str] = []
    for method_b, filename in sorted(EXISTING_PAIRED_FILES.items()):
        path = tables_dir / filename
        if not path.is_file():
            problems.append(f"missing existing paired summary {filename}")
            continue
        for row in read_rows(path):
            key = (row["generator"], row["metric"], row["method_b"])
            new_row = lookup.get(key)
            if new_row is None:
                if row["generator"] != MATRIX_GENERATOR:
                    continue  # matrix only covers switching_var
                problems.append(f"{filename} row {key} absent from matrix")
                continue
            for field in fields:
                reference = float(row[field])
                value = float(new_row[field])
                if abs(reference - value) > tol:
                    problems.append(f"{filename} {key} {field}: {reference} vs {value}")
                checked += 1
    return checked, problems


def write_matrix_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "generator",
        "metric",
        "method_a",
        "method_b",
        "n",
        "mean_a",
        "mean_b",
        "mean_raw_diff_a_minus_b",
        "mean_improvement",
        "std_improvement",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "paired_t_greater_p",
        "wilcoxon_greater_p",
        "cohen_dz",
        "wins",
        "ties",
        "losses",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_p(value: object) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    if number < 0.0001:
        return "<1e-4"
    return f"{number:.4f}"


def write_matrix_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Paired Comparison Matrix — switching_var",
        "",
        f"`{METHOD_A}` (method A) vs every other method (method B), paired on the 20 shared",
        "seeds 0-19. Positive improvement means STCG-v1 is better; for SHD the sign is",
        "reversed because lower is better. Statistics identical to",
        "`compare_recovery_methods.py`: one-sided paired t / Wilcoxon p, 95% bootstrap CI",
        "of the mean improvement (10000 draws, seed 12345), Cohen dz, wins/ties/losses.",
        "MSAR-X is the legacy independent-equation autoregressive-error control, originally mislabeled MSVAR in raw CSVs; it is not a valid direct joint VAR(L) coefficient comparison.",
        "",
    ]
    for metric in DEFAULT_METRICS:
        places = 1 if metric == "shd" else 3
        lines.extend(
            [
                f"## {METRIC_LABELS[metric]}",
                "",
                "| Method B | N | STCG-v1 | Method B mean | Improvement | 95% CI | t p | Wilcoxon p | Cohen dz | W/T/L |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        subset = [row for row in rows if row["metric"] == metric]
        for row in sorted(subset, key=lambda r: float(r["mean_improvement"]), reverse=True):
            lines.append(
                "| {method_b} | {n} | {mean_a} | {mean_b} | {improvement} | [{ci_low}, {ci_high}] | "
                "{t_p} | {w_p} | {dz} | {wins}/{ties}/{losses} |".format(
                    method_b=display_method(row["method_b"]),
                    n=row["n"],
                    mean_a=fmt(row["mean_a"], places),
                    mean_b=fmt(row["mean_b"], places),
                    improvement=fmt(row["mean_improvement"], 4 if metric != "shd" else 1),
                    ci_low=fmt(row["bootstrap_ci_low"], 4 if metric != "shd" else 1),
                    ci_high=fmt(row["bootstrap_ci_high"], 4 if metric != "shd" else 1),
                    t_p=fmt_p(row["paired_t_greater_p"]),
                    w_p=fmt_p(row["wilcoxon_greater_p"]),
                    dz=fmt(row["cohen_dz"], 2),
                    wins=row["wins"],
                    ties=row["ties"],
                    losses=row["losses"],
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_edge_auc_figure(path: Path, rows: list[dict[str, object]]) -> None:
    generators = sorted({str(row["generator"]) for row in rows})
    overall: dict[str, float] = {}
    for row in rows:
        overall.setdefault(str(row["method"]), 0.0)
        overall[str(row["method"])] += float(row["edge_auc_mean"]) / len(generators)
    methods = sorted(overall, key=lambda method: overall[method], reverse=True)
    lookup = {(str(row["generator"]), str(row["method"])): row for row in rows}

    fig, axes = plt.subplots(1, len(generators), figsize=(12.4, 6.8), sharey=True)
    y_positions = list(range(len(methods)))[::-1]
    for ax, generator in zip(axes, generators):
        means = [float(lookup[(generator, method)]["edge_auc_mean"]) for method in methods]
        stds = [float(lookup[(generator, method)]["edge_auc_std"]) for method in methods]
        colors = ["#d62728" if method == METHOD_A else "#1f77b4" for method in methods]
        ax.errorbar(
            means,
            y_positions,
            xerr=stds,
            fmt="none",
            ecolor="#888888",
            elinewidth=1.0,
            capsize=3,
            zorder=2,
        )
        ax.scatter(means, y_positions, c=colors, s=34, zorder=3)
        ax.set_title(generator)
        ax.set_xlim(0.4, 1.0)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([display_method(method) for method in methods])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    fig.supxlabel("Lagged Edge AUC (mean +/- std over seeds 0-19)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows, merge_problems, seeds_by_file = merge_details(args.tables_dir, DETAIL_FILES)

    seed_problems = [
        f"{name}: seeds {seeds[0]}-{seeds[-1]} (n={len(seeds)})"
        for name, seeds in seeds_by_file.items()
        if seeds != list(range(20))
    ]

    summary_rows = aggregate(rows)
    n_summary_checks, summary_problems = reconcile_with_per_file_summaries(
        args.tables_dir, DETAIL_FILES, summary_rows
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / f"{args.output_prefix}.csv"
    summary_md = args.output_dir / f"{args.output_prefix}.md"
    summary_tex = args.output_dir / f"{args.output_prefix.replace('_summary', '_table')}.tex"
    write_summary_csv(summary_csv, summary_rows)
    write_summary_markdown(summary_md, summary_rows)
    write_summary_latex(summary_tex, summary_rows)

    matrix_csv = None
    matrix_md = None
    matrix_rows: list[dict[str, object]] = []
    n_paired_checks = 0
    paired_problems: list[str] = []
    if not args.skip_matrix:
        matrix_rows = build_paired_matrix(rows, args.bootstrap_samples, args.seed)
        n_paired_checks, paired_problems = reconcile_with_existing_paired(args.tables_dir, matrix_rows)
        matrix_csv = args.output_dir / f"{args.matrix_prefix}.csv"
        matrix_md = args.output_dir / f"{args.matrix_prefix}.md"
        write_matrix_csv(matrix_csv, matrix_rows)
        write_matrix_markdown(matrix_md, matrix_rows)

    figure_path = None
    if not args.skip_figure:
        figure_path = args.figures_dir / args.figure_name
        write_edge_auc_figure(figure_path, summary_rows)

    problems = merge_problems + seed_problems + summary_problems + paired_problems
    print(
        json.dumps(
            {
                "summary_csv": str(summary_csv),
                "summary_md": str(summary_md),
                "summary_tex": str(summary_tex),
                "matrix_csv": str(matrix_csv) if matrix_csv else None,
                "matrix_md": str(matrix_md) if matrix_md else None,
                "figure": str(figure_path) if figure_path else None,
                "n_detail_rows_merged": len(rows),
                "n_summary_rows": len(summary_rows),
                "n_matrix_rows": len(matrix_rows),
                "summary_reconciliation_checks": n_summary_checks,
                "paired_reconciliation_checks": n_paired_checks,
                "problems": problems,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
