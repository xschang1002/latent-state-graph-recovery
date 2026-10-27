"""Recompute two-sided RPCMCI tests and exclude cMLP pilot seeds from comparisons.

The original 20-seed results remain descriptive. The post-pilot sensitivity
analysis uses the intersection of seeds 5--19 across all methods per generator;
this excludes nonlinear seed 18 because its RPCMCI fit failed. It is not a new,
prospectively collected independent test set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_recovery_methods import DEFAULT_METRICS, compare
from summarize_all_recovery_methods import DETAIL_FILES, aggregate, merge_details


def common_postpilot_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Use identical, non-pilot seeds for every method within each generator."""
    selected = []
    for generator in sorted({r["generator"] for r in rows}):
        group = [r for r in rows if r["generator"] == generator]
        methods = sorted({r["method"] for r in group})
        seed_sets = [
            {int(r["seed"]) for r in group if r["method"] == m and 5 <= int(r["seed"]) <= 19}
            for m in methods
        ]
        common = set.intersection(*seed_sets)
        if not common:
            raise ValueError(f"No common post-pilot seeds for {generator}")
        selected.extend(r for r in group if int(r["seed"]) in common)
    return selected


def two_sided_p(values: list[float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or not values.size:
        raise ValueError("Paired differences must be finite and nonempty")
    if np.all(values == 0):
        return 1.0, 1.0
    return (
        float(stats.ttest_1samp(values, 0.0, alternative="two-sided").pvalue),
        float(stats.wilcoxon(values, alternative="two-sided").pvalue),
    )


def comparison(rows, generator, opponent):
    args = argparse.Namespace(
        generators=[generator], method_a="STCG-v1", method_b=opponent,
        metrics=DEFAULT_METRICS, bootstrap_samples=10000, seed=12345,
    )
    details, summaries = compare(args, rows)
    for summary in summaries:
        values = [r["improvement"] for r in details if r["metric"] == summary["metric"]]
        tp, wp = two_sided_p(values)
        summary.update(paired_t_two_sided_p=tp, wilcoxon_two_sided_p=wp)
    return details, summaries


def build(tables_dir: Path, output_dir: Path) -> dict:
    rows, problems, _ = merge_details(tables_dir, DETAIL_FILES)
    if problems:
        raise ValueError("Conflicting stored observations: " + repr(problems))
    output_dir.mkdir(parents=True, exist_ok=True)
    _detail, rpc = comparison(rows, "switching_var", "Regime-PCMCI")
    pd.DataFrame(rpc).to_csv(output_dir / "synthetic_rpcmci_two_sided_summary.csv", index=False)

    selected = common_postpilot_rows(rows)
    summary = aggregate(selected)
    pd.DataFrame(summary).to_csv(output_dir / "synthetic_postpilot_all_methods_summary.csv", index=False)
    pd.DataFrame(selected).to_csv(output_dir / "synthetic_postpilot_all_methods_detail.csv", index=False)
    paired = []
    seed_manifest = {}
    for generator in sorted({r["generator"] for r in selected}):
        seed_manifest[generator] = sorted({int(r["seed"]) for r in selected if r["generator"] == generator})
        opponents = sorted({r["method"] for r in selected if r["generator"] == generator} - {"STCG-v1"})
        for opponent in opponents:
            _, comparison_summary = comparison(selected, generator, opponent)
            paired.extend(comparison_summary)
    pd.DataFrame(paired).to_csv(output_dir / "synthetic_postpilot_paired_summary.csv", index=False)

    lines = [
        "# Post-pilot sensitivity and two-sided tests", "",
        "cMLP hyperparameters were selected on seeds 0--4 of all three generators. "
        "The original 20-seed tables therefore include pilot observations. "
        "This sensitivity analysis excludes them and uses common seeds for all 16 methods. "
        "It is retrospective, not a newly collected independent confirmation experiment.", "",
        "Nonlinear seed 18 is also omitted from every method in this sensitivity analysis "
        "because RPCMCI did not return a converged fit.", "",
        "Positive improvement favors STCG-v1; SHD is sign-reversed. "
        "The paired CSV reports both one-sided superiority and two-sided difference tests, "
        "plus percentile bootstrap intervals (10,000 draws, seed 12345). "
        "No equivalence or non-inferiority margin was specified, so nonsignificance is not equivalence.", "",
        "| Generator | Common seeds | STCG Edge AUC | cMLP Edge AUC | Difference | One-sided t p | One-sided W p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for generator, seeds in seed_manifest.items():
        row = next(r for r in paired if r["generator"] == generator and r["method_b"] == "cMLP" and r["metric"] == "edge_auc")
        lines.append(
            f"| {generator} | {', '.join(map(str, seeds))} | {row['mean_a']:.6f} | {row['mean_b']:.6f} "
            f"| {row['mean_improvement']:+.6f} | {row['paired_t_greater_p']:.6f} | {row['wilcoxon_greater_p']:.6f} |"
        )
    lines += ["", "## Full-sample STCG versus Regime-PCMCI (switching)", "",
              "| Metric | Difference | Two-sided t p | Two-sided W p |", "|---|---:|---:|---:|"]
    for row in rpc:
        lines.append(f"| {row['metric']} | {row['mean_improvement']:+.6f} | {row['paired_t_two_sided_p']:.6f} | {row['wilcoxon_two_sided_p']:.6f} |")
    (output_dir / "synthetic_postpilot_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"common_seeds": seed_manifest, "method_summary_rows": len(summary), "paired_rows": len(paired), "two_sided_rows": len(rpc)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    args = parser.parse_args()
    print(json.dumps(build(args.tables_dir, args.output_dir), indent=2))
