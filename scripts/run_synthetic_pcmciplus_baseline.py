"""Run a focused PCMCI+ synthetic baseline comparison.

Mirrors ``run_synthetic_pcmci_baseline.py`` but scores edges with
Tigramite's ``run_pcmciplus`` (PCMCI+) and checkpoints every
``(generator, seed)`` row into ``experiments/logs/pcmciplus_parts/``
(gitignored) so long runs are resumable: re-running the script skips
existing partials, and the tracked result tables are (re)built once every
requested partial exists (or immediately with ``--aggregate-only``).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
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
    aggregate,
    row_for,
    write_csv,
    write_edge_auc_figure,
    write_latex,
    write_markdown,
    write_summary_csv,
)
from stcg import generate_lagged_var, pcmciplus_scores, recovery_summary  # noqa: E402


DEFAULT_REFERENCE_METHODS = [
    "STCG-v1",
    "DynVAR-Lasso",
    "DynGranger-F",
    "VAR-Lasso",
    "Granger-F",
]
DETAIL_FIELDNAMES = [
    "generator",
    "seed",
    "method",
    "edge_auc",
    "pair_auc",
    "precision_at_k",
    "shd",
    "lag_accuracy",
    "true_edges",
    "edge_density",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinds", nargs="+", choices=DEFAULT_KINDS, default=DEFAULT_KINDS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--edge-prob", type=float, default=0.18)
    parser.add_argument("--noise-scale", type=float, default=0.10)
    parser.add_argument("--pc-alpha", type=float, default=0.05)
    parser.add_argument(
        "--alpha-level",
        type=float,
        default=None,
        help="Forwarded to run_pcmciplus only when the installed tigramite supports it.",
    )
    parser.add_argument("--score", choices=["statistic", "p_value"], default="statistic")
    parser.add_argument("--reference-detail-csv", type=Path, default=ROOT / "results" / "tables" / "synthetic_recovery_detail.csv")
    parser.add_argument("--reference-methods", nargs="+", default=DEFAULT_REFERENCE_METHODS)
    parser.add_argument("--parts-dir", type=Path, default=ROOT / "experiments" / "logs" / "pcmciplus_parts")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Only compute missing partials; never write tracked result tables.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--output-prefix", default="synthetic_pcmciplus_baseline")
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
    scores = pcmciplus_scores(
        dataset.x,
        max_lag=args.max_lag,
        pc_alpha=args.pc_alpha,
        alpha_level=args.alpha_level,
        score=args.score,
    )
    return row_for(kind, seed, "PCMCI+", recovery_summary(scores, dataset.edge_truth))


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
                print(f"[pcmciplus] wrote {path}", flush=True)

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

    rows = read_reference_rows(
        args.reference_detail_csv,
        kinds=set(args.kinds),
        seeds=set(args.seeds),
        methods=set(args.reference_methods),
    )
    for kind in args.kinds:
        for seed in args.seeds:
            part = read_part(part_path(args.parts_dir, kind, seed))
            assert part is not None  # guaranteed by the completeness check
            part["seed"] = int(part["seed"])
            rows.append(part)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / f"{args.output_prefix}_detail.csv"
    summary_path = args.output_dir / f"{args.output_prefix}_summary.csv"
    markdown_path = args.output_dir / f"{args.output_prefix}_summary.md"
    latex_path = args.output_dir / f"{args.output_prefix}_table.tex"
    figure_path = args.figure_dir / f"{args.output_prefix}_edge_auc.png"

    summary_rows = aggregate(rows)
    write_csv(detail_path, rows)
    write_summary_csv(summary_path, summary_rows)
    write_markdown(markdown_path, summary_rows)
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
