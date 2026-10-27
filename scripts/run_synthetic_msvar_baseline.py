"""Run a focused Markov-switching VAR (MSVAR) synthetic baseline comparison.

Fits the equation-wise statsmodels ``MarkovAutoregression`` baseline from
``stcg.msvar_scores`` across the synthetic grid and, on ``switching_var``,
scores regime-estimation quality by aligning the smoothed regime
probabilities (averaged across target equations) with the generator's
ground-truth hidden regimes (ARI + Hungarian alignment accuracy).

Every ``(generator, seed)`` row is checkpointed into
``experiments/logs/msvar_parts/`` (gitignored); re-running skips existing
partials, and the tracked result tables are (re)built once every requested
partial exists (or immediately with ``--aggregate-only``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Set before sklearn imports so joblib/loky does not oversubscribe CPU cores
# when KMeans runs on Windows.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _common  # noqa: E402
from run_synthetic_recovery import (  # noqa: E402
    DEFAULT_KINDS,
    METRICS,
    aggregate,
    row_for,
    write_edge_auc_figure,
    write_latex,
)
from stcg import generate_lagged_var, msvar_scores, recovery_summary  # noqa: E402


DEFAULT_REFERENCE_METHODS = [
    "STCG-v1",
    "DynVAR-Lasso",
    "DynGranger-F",
    "VAR-Lasso",
    "Granger-F",
]
EXTRA_FIELDNAMES = [
    "status",
    "n_failed_targets",
    "converged_fraction",
    "regime_ari",
    "regime_accuracy",
    "fit_seconds",
]
DETAIL_FIELDNAMES = [
    "generator",
    "seed",
    "method",
    *METRICS,
    "true_edges",
    "edge_density",
    *EXTRA_FIELDNAMES,
]
SUMMARY_FIELDNAMES = [
    "generator",
    "method",
    "n",
]
for _metric in METRICS:
    SUMMARY_FIELDNAMES.extend([f"{_metric}_mean", f"{_metric}_std"])
SUMMARY_FIELDNAMES.extend(
    [
        "regime_ari_mean",
        "regime_ari_std",
        "regime_accuracy_mean",
        "regime_accuracy_std",
        "converged_fraction_mean",
        "converged_fraction_std",
        "fit_seconds_mean",
        "fit_seconds_std",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinds", nargs="+", choices=DEFAULT_KINDS, default=DEFAULT_KINDS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--edge-prob", type=float, default=0.18)
    parser.add_argument("--noise-scale", type=float, default=0.10)
    parser.add_argument("--n-regimes", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--retry-maxiter", type=int, default=500)
    parser.add_argument("--search-reps", type=int, default=3)
    parser.add_argument("--reference-detail-csv", type=Path, default=ROOT / "results" / "tables" / "synthetic_recovery_detail.csv")
    parser.add_argument("--reference-methods", nargs="+", default=DEFAULT_REFERENCE_METHODS)
    parser.add_argument("--parts-dir", type=Path, default=ROOT / "experiments" / "logs" / "msvar_parts")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Only compute missing partials; never write tracked result tables.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--output-prefix", default="synthetic_msvar_baseline")
    return parser.parse_args()


def part_path(parts_dir: Path, kind: str, seed: int) -> Path:
    return parts_dir / f"{kind}_seed{seed:03d}.csv"


def read_part(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[0]
    return {key: (int(value) if key == "seed" else value) for key, value in row.items()}


def regime_quality(dataset, info: dict[str, object], kind: str) -> tuple[float, float]:
    """Align smoothed regime probabilities with ground-truth hidden regimes."""

    sequence = info.get("regime_sequence")
    if kind != "switching_var" or sequence is None:
        return math.nan, math.nan
    from sklearn.metrics import adjusted_rand_score

    offset = int(info["effective_offset"])
    true = dataset.regimes[offset:]
    predicted = sequence[: len(true)]
    true = true[: len(predicted)]
    if len(true) == 0:
        return math.nan, math.nan
    ari = float(adjusted_rand_score(true, predicted))
    _, _, accuracy = _common.align_labels(
        [str(int(value)) for value in true],
        [int(value) for value in predicted],
    )
    return ari, accuracy


def compute_row(kind: str, seed: int, args: argparse.Namespace) -> dict[str, object]:
    dataset = generate_lagged_var(
        n_nodes=args.n_nodes,
        timesteps=args.timesteps,
        max_lag=args.max_lag,
        edge_prob=args.edge_prob,
        noise_scale=args.noise_scale,
        seed=seed,
        nonlinear=kind == "nonlinear_var",
        switching=kind == "switching_var",
    )
    started = time.perf_counter()
    try:
        scores, info = msvar_scores(
            dataset.x,
            max_lag=args.max_lag,
            n_regimes=args.n_regimes,
            maxiter=args.maxiter,
            retry_maxiter=args.retry_maxiter,
            search_reps=args.search_reps,
            return_info=True,
            seed=seed,
        )
        summary = recovery_summary(scores, dataset.edge_truth)
        failed = list(info["failed_targets"])
        converged = list(info["converged"])
        status = "ok" if not failed else "partial"
        regime_ari, regime_accuracy = regime_quality(dataset, info, kind)
        extras: dict[str, object] = {
            "status": status,
            "n_failed_targets": len(failed),
            "converged_fraction": sum(converged) / len(converged) if converged else 0.0,
            "regime_ari": regime_ari,
            "regime_accuracy": regime_accuracy,
        }
    except Exception as exc:  # flagged NaN row keeps the grid resumable
        print(f"[msvar] {kind}/seed{seed} failed: {exc}", file=sys.stderr, flush=True)
        summary = {metric: math.nan for metric in METRICS}
        summary.update({"true_edges": math.nan, "edge_density": math.nan})
        extras = {
            "status": "failed",
            "n_failed_targets": args.n_nodes,
            "converged_fraction": 0.0,
            "regime_ari": math.nan,
            "regime_accuracy": math.nan,
        }
    extras["fit_seconds"] = time.perf_counter() - started
    row = row_for(kind, seed, "MSVAR", summary)
    row.update(extras)
    return row


def read_reference_rows(path: Path, kinds: set[str], seeds: set[int], methods: set[str]) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["generator"] not in kinds:
                continue
            if int(row["seed"]) not in seeds:
                continue
            if row["method"] not in methods:
                continue
            rows.append(
                {
                    key: (int(value) if key == "seed" else value)
                    for key, value in row.items()
                }
            )
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan, math.nan
    mean = sum(finite) / len(finite)
    if len(finite) <= 1:
        return mean, 0.0
    return mean, math.sqrt(sum((value - mean) ** 2 for value in finite) / (len(finite) - 1))


def extend_summary(summary_rows: list[dict[str, object]], detail_rows: list[dict[str, object]]) -> None:
    """Attach regime/convergence/runtime aggregates to MSVAR summary rows."""

    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        groups[(str(row["generator"]), str(row["method"]))].append(row)
    for summary in summary_rows:
        items = groups.get((str(summary["generator"]), str(summary["method"])), [])
        for column in ("regime_ari", "regime_accuracy", "converged_fraction", "fit_seconds"):
            values = [float(item[column]) for item in items if column in item]
            mean, std = mean_std(values)
            summary[f"{column}_mean"] = mean
            summary[f"{column}_std"] = std


def write_summary_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Synthetic Recovery Summary",
        "",
        "| Generator | Method | N | Edge AUC | Pair AUC | Precision@K | SHD | Lag Acc | Regime ARI | Regime Acc |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {generator} | {method} | {n} | {edge_auc_mean:.3f} +/- {edge_auc_std:.3f} | "
            "{pair_auc_mean:.3f} +/- {pair_auc_std:.3f} | "
            "{precision_at_k_mean:.3f} +/- {precision_at_k_std:.3f} | "
            "{shd_mean:.1f} +/- {shd_std:.1f} | "
            "{lag_accuracy_mean:.3f} +/- {lag_accuracy_std:.3f} | {regime_ari} | {regime_acc} |".format(
                regime_ari=_common.fmt_na(row.get("regime_ari_mean"), 3),
                regime_acc=_common.fmt_na(row.get("regime_accuracy_mean"), 3),
                **row,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    computed: list[str] = []
    if not args.aggregate_only:
        args.parts_dir.mkdir(parents=True, exist_ok=True)
        for kind in args.kinds:
            for seed in args.seeds:
                path = part_path(args.parts_dir, kind, seed)
                if read_part(path) is not None:
                    continue
                row = compute_row(kind, seed, args)
                _common.write_csv(path, [row], DETAIL_FIELDNAMES)
                computed.append(f"{kind}/seed{seed}")
                print(f"[msvar] wrote {path}", flush=True)

    missing = [
        f"{kind}/seed{seed}"
        for kind in args.kinds
        for seed in args.seeds
        if read_part(part_path(args.parts_dir, kind, seed)) is None
    ]
    if args.no_aggregate:
        print(
            json.dumps(
                {"status": "partial" if missing else "complete", "computed": computed, "missing": missing},
                indent=2,
                sort_keys=True,
            )
        )
        return
    if missing:
        if args.aggregate_only:
            raise SystemExit(f"Cannot aggregate; missing partials: {missing}")
        print(
            json.dumps(
                {"status": "partial", "computed": computed, "missing": missing},
                indent=2,
                sort_keys=True,
            )
        )
        return

    reference_rows = read_reference_rows(
        args.reference_detail_csv,
        kinds=set(args.kinds),
        seeds=set(args.seeds),
        methods=set(args.reference_methods),
    )
    method_rows: list[dict[str, object]] = []
    for kind in args.kinds:
        for seed in args.seeds:
            part = read_part(part_path(args.parts_dir, kind, seed))
            assert part is not None  # guaranteed by the completeness check
            part["seed"] = int(part["seed"])
            method_rows.append(part)
    rows = reference_rows + method_rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / f"{args.output_prefix}_detail.csv"
    summary_path = args.output_dir / f"{args.output_prefix}_summary.csv"
    markdown_path = args.output_dir / f"{args.output_prefix}_summary.md"
    latex_path = args.output_dir / f"{args.output_prefix}_table.tex"
    figure_path = args.figure_dir / f"{args.output_prefix}_edge_auc.png"

    summary_rows = aggregate(rows)
    extend_summary(summary_rows, method_rows)
    _common.write_csv(detail_path, rows, DETAIL_FIELDNAMES)
    _common.write_csv(summary_path, summary_rows, SUMMARY_FIELDNAMES)
    write_summary_markdown(markdown_path, summary_rows)
    write_latex(latex_path, summary_rows)
    write_edge_auc_figure(figure_path, summary_rows)

    print(
        json.dumps(
            {
                "detail_csv": str(detail_path),
                "summary_csv": str(summary_path),
                "summary_md": str(markdown_path),
                "table_tex": str(latex_path),
                "figure": str(figure_path),
                "computed": computed,
                "n_rows": len(rows),
                "n_summary_rows": len(summary_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
