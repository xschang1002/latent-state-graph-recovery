"""Paired statistical comparisons for synthetic recovery results."""

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

from _common import fmt_na as fmt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_METRICS = ["edge_auc", "pair_auc", "precision_at_k", "shd", "lag_accuracy"]
LOWER_IS_BETTER = {"shd"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail-csv", type=Path, default=ROOT / "results" / "tables" / "synthetic_recovery_detail.csv")
    parser.add_argument("--method-a", default="STCG-v1")
    parser.add_argument("--method-b", default="DynVAR-Lasso")
    parser.add_argument("--generators", nargs="+", default=["switching_var"])
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--output-prefix", default="synthetic_recovery_paired_comparison")
    parser.add_argument("--latex-generator", default="switching_var")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    value_mean = mean(values)
    return math.sqrt(sum((value - value_mean) ** 2 for value in values) / (len(values) - 1))


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    import numpy as np

    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    draws = rng.choice(array, size=(samples, array.size), replace=True)
    boot_means = draws.mean(axis=1)
    return float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))


def paired_tests(improvements: list[float]) -> tuple[float, float, float]:
    """Return paired t one-sided p, Wilcoxon one-sided p, and Cohen dz."""

    if not improvements:
        return math.nan, math.nan, math.nan

    from scipy import stats

    if all(abs(value) <= 1e-12 for value in improvements):
        return 1.0, 1.0, 0.0

    t_result = stats.ttest_1samp(improvements, popmean=0.0, alternative="greater")
    try:
        wilcoxon_result = stats.wilcoxon(improvements, alternative="greater", zero_method="wilcox")
        wilcoxon_p = float(wilcoxon_result.pvalue)
    except ValueError:
        wilcoxon_p = math.nan

    std = sample_std(improvements)
    dz = mean(improvements) / std if std > 0 else math.inf
    return float(t_result.pvalue), wilcoxon_p, float(dz)


def build_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], dict[str, str]]:
    lookup: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in rows:
        lookup[(row["generator"], int(row["seed"]), row["method"])] = row
    return lookup


def compare(args: argparse.Namespace, rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    lookup = build_lookup(rows)
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for generator in args.generators:
        seeds_a = {seed for gen, seed, method in lookup if gen == generator and method == args.method_a}
        seeds_b = {seed for gen, seed, method in lookup if gen == generator and method == args.method_b}
        seeds = sorted(seeds_a & seeds_b)
        if not seeds:
            raise ValueError(f"No paired seeds found for {generator}: {args.method_a} vs {args.method_b}")

        for metric in args.metrics:
            improvements: list[float] = []
            values_a: list[float] = []
            values_b: list[float] = []

            for seed in seeds:
                value_a = float(lookup[(generator, seed, args.method_a)][metric])
                value_b = float(lookup[(generator, seed, args.method_b)][metric])
                raw_diff = value_a - value_b
                improvement = -raw_diff if metric in LOWER_IS_BETTER else raw_diff
                values_a.append(value_a)
                values_b.append(value_b)
                improvements.append(improvement)
                detail_rows.append(
                    {
                        "generator": generator,
                        "seed": seed,
                        "metric": metric,
                        "method_a": args.method_a,
                        "method_b": args.method_b,
                        "value_a": value_a,
                        "value_b": value_b,
                        "raw_diff_a_minus_b": raw_diff,
                        "improvement": improvement,
                    }
                )

            ci_low, ci_high = bootstrap_ci(improvements, args.bootstrap_samples, args.seed)
            t_p, wilcoxon_p, dz = paired_tests(improvements)
            wins = sum(value > 1e-12 for value in improvements)
            ties = sum(abs(value) <= 1e-12 for value in improvements)
            losses = sum(value < -1e-12 for value in improvements)
            summary_rows.append(
                {
                    "generator": generator,
                    "metric": metric,
                    "method_a": args.method_a,
                    "method_b": args.method_b,
                    "n": len(seeds),
                    "mean_a": mean(values_a),
                    "mean_b": mean(values_b),
                    "mean_raw_diff_a_minus_b": mean([a - b for a, b in zip(values_a, values_b)]),
                    "mean_improvement": mean(improvements),
                    "std_improvement": sample_std(improvements),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "paired_t_greater_p": t_p,
                    "wilcoxon_greater_p": wilcoxon_p,
                    "cohen_dz": dz,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                }
            )
    return detail_rows, summary_rows


def write_detail(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "generator",
        "seed",
        "metric",
        "method_a",
        "method_b",
        "value_a",
        "value_b",
        "raw_diff_a_minus_b",
        "improvement",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_p(value: object) -> str:
    value_float = float(value)
    if not math.isfinite(value_float):
        return "n/a"
    if value_float < 0.0001:
        return "$<10^{-4}$"
    return f"{value_float:.4f}"


def metric_label(metric: object) -> str:
    labels = {
        "edge_auc": "Edge AUC",
        "pair_auc": "Pair AUC",
        "precision_at_k": "P@K",
        "shd": "SHD",
        "lag_accuracy": "Lag Acc",
    }
    return labels.get(str(metric), str(metric).replace("_", " "))


def value_digits(metric: object) -> int:
    return 1 if str(metric) == "shd" else 3


def improvement_digits(metric: object) -> int:
    return 1 if str(metric) == "shd" else 4


def method_label(method: object) -> str:
    labels = {
        "DynGranger-F": "DynGranger",
        "DynVAR-Lasso": "DynVAR",
        "DynVAR-Ridge": "DynVAR-Ridge",
        "Granger-F": "Granger-F",
        "STCG-v1": "STCG-v1",
    }
    return labels.get(str(method), str(method))


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Paired Synthetic Recovery Comparison",
        "",
        "Positive improvement means `method_a` is better than `method_b`; for SHD the sign is reversed because lower is better.",
        "",
        "| Generator | Metric | N | Method A | Method B | Mean Improvement | 95% bootstrap CI | t-test p | Wilcoxon p | Wins/Ties/Losses |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {generator} | {metric} | {n} | {method_a} ({mean_a}) | {method_b} ({mean_b}) | "
            "{improvement} | [{ci_low}, {ci_high}] | {t_p} | {w_p} | {wins}/{ties}/{losses} |".format(
                generator=row["generator"],
                metric=row["metric"],
                n=row["n"],
                method_a=row["method_a"],
                method_b=row["method_b"],
                mean_a=fmt(row["mean_a"], 3),
                mean_b=fmt(row["mean_b"], 3),
                improvement=fmt(row["mean_improvement"], 4),
                ci_low=fmt(row["bootstrap_ci_low"], 4),
                ci_high=fmt(row["bootstrap_ci_high"], 4),
                t_p=fmt(row["paired_t_greater_p"], 4),
                w_p=fmt(row["wilcoxon_greater_p"], 4),
                wins=row["wins"],
                ties=row["ties"],
                losses=row["losses"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: list[dict[str, object]], generator: str, method_a: str, method_b: str) -> None:
    selected = [row for row in rows if str(row["generator"]) == generator]
    if not selected:
        selected = rows
    lines = [
        "\\begin{tabular}{lrrrlrrl}",
        "\\toprule",
        f"Metric & {method_label(method_a)} & {method_label(method_b)} & Improve & 95\\% CI & $p_t$ & $p_W$ & W/T/L \\\\",
        "\\midrule",
    ]
    for row in selected:
        value_places = value_digits(row["metric"])
        improvement_places = improvement_digits(row["metric"])
        lines.append(
            (
                "{metric} & {mean_a} & {mean_b} & {improvement} & "
                "[{ci_low}, {ci_high}] & {t_p} & {w_p} & {wins}/{ties}/{losses} \\\\"
            ).format(
                metric=metric_label(row["metric"]),
                mean_a=fmt(row["mean_a"], value_places),
                mean_b=fmt(row["mean_b"], value_places),
                improvement=fmt(row["mean_improvement"], improvement_places),
                ci_low=fmt(row["bootstrap_ci_low"], improvement_places),
                ci_high=fmt(row["bootstrap_ci_high"], improvement_places),
                t_p=fmt_p(row["paired_t_greater_p"]),
                w_p=fmt_p(row["wilcoxon_greater_p"]),
                wins=row["wins"],
                ties=row["ties"],
                losses=row["losses"],
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.detail_csv)
    detail_rows, summary_rows = compare(args, rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / f"{args.output_prefix}_detail.csv"
    summary_path = args.output_dir / f"{args.output_prefix}_summary.csv"
    markdown_path = args.output_dir / f"{args.output_prefix}_summary.md"
    latex_path = args.output_dir / f"{args.output_prefix}_table.tex"
    write_detail(detail_path, detail_rows)
    write_summary(summary_path, summary_rows)
    write_markdown(markdown_path, summary_rows)
    write_latex(latex_path, summary_rows, args.latex_generator, args.method_a, args.method_b)

    print(
        json.dumps(
            {
            "detail_csv": str(detail_path),
            "latex_table": str(latex_path),
            "summary_csv": str(summary_path),
            "summary_md": str(markdown_path),
            "n_detail_rows": len(detail_rows),
            "n_summary_rows": len(summary_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
