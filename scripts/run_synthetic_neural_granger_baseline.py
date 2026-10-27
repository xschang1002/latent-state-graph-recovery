"""Run the cMLP neural-Granger synthetic baseline comparison.

Fits the component-wise MLP neural-Granger model of Tank et al. (IEEE TPAMI
44(8), 2022; arXiv:1802.05842; reference implementation
github.com/iancovert/Neural-GC, MIT) via ``stcg.cmlp_scores`` across the
synthetic grid: one MLP per target series with a group-lasso penalty over
the first-layer input columns grouped by ``(source, lag)``, trained with
ISTA-style SGD + proximal group soft-thresholding; post-training group
norms score each ``(lag, source, target)`` edge.

Every ``(generator, seed)`` row is checkpointed into
``experiments/logs/cmlp_parts/`` (gitignored); re-running skips existing
partials, and the tracked result tables are (re)built once every requested
partial exists (or immediately with ``--aggregate-only``). Failed fits stay
in the partials as flagged NaN rows but are excluded from the tracked
tables, mirroring ``run_synthetic_rpcmci_baseline.py``.

Compute budget: the training configuration (``--hidden``, ``--penalty``,
``--lr``, ``--epochs``, ``--momentum``, ``--optimizer``, tanh) is FIXED for
the whole grid and recorded in every detail row so the tables document it.
The defaults come from a pilot sweep on seeds 0-4 (see
``stcg.neural_granger.cmlp_scores``): ~0.6 s per ``(generator, seed)`` row
(8 target MLPs, 300 epochs each) on a single CPU thread, so the 60-row grid
finishes in well under a minute. SGD (not Adam) is load-bearing: Adam's
per-parameter gradient rescaling breaks the group-lasso shrinkage
equilibrium and inverts the score ranking on low-SNR linear VARs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

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
from stcg import cmlp_scores, generate_lagged_var, recovery_summary  # noqa: E402


METHOD_NAME = "cMLP"
DEFAULT_REFERENCE_METHODS = [
    "STCG-v1",
    "DynVAR-Lasso",
    "DynGranger-F",
    "VAR-Lasso",
    "Granger-F",
]
EXTRA_FIELDNAMES = [
    "status",
    "hidden",
    "penalty",
    "lr",
    "epochs",
    "activation",
    "optimizer",
    "momentum",
    "final_loss_mean",
    "final_loss_max",
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
        "final_loss_mean_mean",
        "final_loss_mean_std",
        "final_loss_max_mean",
        "final_loss_max_std",
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
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--penalty", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--reference-detail-csv", type=Path, default=ROOT / "results" / "tables" / "synthetic_recovery_detail.csv")
    parser.add_argument("--reference-methods", nargs="+", default=DEFAULT_REFERENCE_METHODS)
    parser.add_argument("--parts-dir", type=Path, default=ROOT / "experiments" / "logs" / "cmlp_parts")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Only compute missing partials; never write tracked result tables.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--output-prefix", default="synthetic_neural_granger_baseline")
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
        scores, info = cmlp_scores(
            dataset.x,
            max_lag=args.max_lag,
            hidden=args.hidden,
            penalty=args.penalty,
            lr=args.lr,
            epochs=args.epochs,
            activation=args.activation,
            optimizer=args.optimizer,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            seed=seed,
            return_info=True,
        )
        summary = recovery_summary(scores, dataset.edge_truth)
        losses = [float(value) for value in info["final_losses"]]
        extras: dict[str, object] = {
            "status": "ok",
            "hidden": info["hidden"],
            "penalty": info["penalty"],
            "lr": info["lr"],
            "epochs": info["epochs"],
            "activation": info["activation"],
            "optimizer": info["optimizer"],
            "momentum": info["momentum"],
            "final_loss_mean": sum(losses) / len(losses),
            "final_loss_max": max(losses),
        }
    except Exception as exc:  # flagged NaN row keeps the grid resumable
        print(f"[cmlp] {kind}/seed{seed} failed: {exc}", file=sys.stderr, flush=True)
        summary = {metric: math.nan for metric in METRICS}
        summary.update({"true_edges": math.nan, "edge_density": math.nan})
        extras = {
            "status": "failed",
            "hidden": args.hidden,
            "penalty": args.penalty,
            "lr": args.lr,
            "epochs": args.epochs,
            "activation": args.activation,
            "optimizer": args.optimizer,
            "momentum": args.momentum,
            "final_loss_mean": math.nan,
            "final_loss_max": math.nan,
        }
    extras["fit_seconds"] = time.perf_counter() - started
    row = row_for(kind, seed, METHOD_NAME, summary)
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
    """Attach final-loss/runtime aggregates to cMLP summary rows."""

    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        groups[(str(row["generator"]), str(row["method"]))].append(row)
    for summary in summary_rows:
        items = groups.get((str(summary["generator"]), str(summary["method"])), [])
        for column in ("final_loss_mean", "final_loss_max", "fit_seconds"):
            values = [float(item[column]) for item in items if column in item]
            mean, std = mean_std(values)
            summary[f"{column}_mean"] = mean
            summary[f"{column}_std"] = std


def write_summary_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Synthetic Recovery Summary",
        "",
        "| Generator | Method | N | Edge AUC | Pair AUC | Precision@K | SHD | Lag Acc | Final Loss | Fit s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {generator} | {method} | {n} | {edge_auc_mean:.3f} +/- {edge_auc_std:.3f} | "
            "{pair_auc_mean:.3f} +/- {pair_auc_std:.3f} | "
            "{precision_at_k_mean:.3f} +/- {precision_at_k_std:.3f} | "
            "{shd_mean:.1f} +/- {shd_std:.1f} | "
            "{lag_accuracy_mean:.3f} +/- {lag_accuracy_std:.3f} | {final_loss} | {fit_s} |".format(
                final_loss=_common.fmt_na(row.get("final_loss_mean_mean"), 3),
                fit_s=_common.fmt_na(row.get("fit_seconds_mean"), 2),
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
                print(f"[cmlp] wrote {path}", flush=True)

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
    failed_rows: list[str] = []
    for kind in args.kinds:
        for seed in args.seeds:
            part = read_part(part_path(args.parts_dir, kind, seed))
            assert part is not None  # guaranteed by the completeness check
            part["seed"] = int(part["seed"])
            if str(part["status"]) != "ok":
                # Keep failed fits out of the tracked tables: NaN metrics
                # would poison aggregate() and the paired comparison. The
                # flagged row stays in the gitignored partial; the summary
                # `n` (and paired-seed intersections) reflect converged fits.
                failed_rows.append(f"{kind}/seed{seed}")
                continue
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
                "failed_rows": failed_rows,
                "n_rows": len(rows),
                "n_summary_rows": len(summary_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
