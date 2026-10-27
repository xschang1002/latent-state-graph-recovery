"""Run downstream forecasting for CSV files that passed the real-data inventory."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Set before child scripts are spawned so joblib/loky does not oversubscribe
# CPU cores when KMeans runs on Windows (inherited by subprocesses).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-csv",
        type=Path,
        default=ROOT / "results" / "tables" / "real_data_inventory.csv",
    )
    parser.add_argument("--statuses", nargs="+", default=["ready", "warning"])
    parser.add_argument("--datasets", nargs="+", default=[])
    parser.add_argument("--output-root", type=Path, default=ROOT / "experiments" / "logs" / "real_benchmarks")
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=ROOT / "results" / "tables" / "real_benchmark_run_plan.md",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--lasso-alpha", type=float, default=0.005)
    parser.add_argument("--graph-fraction", type=float, default=0.25)
    parser.add_argument("--graph-min-weight", type=float, default=0.10)
    parser.add_argument("--graph-self-weight", type=float, default=1.0)
    parser.add_argument("--graph-weight-power", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--window-stride", type=int, default=150)
    parser.add_argument("--window-aggregate", choices=["max", "mean", "p90"], default="max")
    parser.add_argument("--state-aggregate", choices=["mean", "median", "p75", "max"], default="mean")
    parser.add_argument("--stcg-v1-states", type=int, default=2)
    parser.add_argument("--stcg-v1-min-improvement", type=float, default=0.12)
    parser.add_argument("--stcg-v1-force-states", action="store_true")
    parser.add_argument("--diagnostic-top-k", type=int, default=30)
    parser.add_argument("--missing", choices=["drop", "interpolate"], default="interpolate")
    return parser.parse_args()


def read_inventory(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Inventory CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def eligible_rows(rows: list[dict[str, str]], statuses: set[str], datasets: set[str]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        if row.get("status", "") not in statuses:
            continue
        if datasets and row.get("dataset", "") not in datasets:
            continue
        selected.append(row)
    return selected


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("._") or "dataset"


def input_path(row: dict[str, str]) -> Path:
    path = Path(row["file"])
    return path if path.is_absolute() else ROOT / path


def add_option(parts: list[str], name: str, value: object) -> None:
    parts.extend([name, str(value)])


def build_command(row: dict[str, str], args: argparse.Namespace) -> tuple[list[str], Path, Path]:
    dataset = row["dataset"]
    slug = slugify(dataset)
    output_dir = args.output_root / slug / "tables"
    figure_dir = args.output_root / slug / "figures"
    command = [
        sys.executable,
        "scripts\\run_downstream_forecasting.py",
        "--input-csv",
        str(input_path(row)),
        "--dataset-name",
        dataset,
        "--missing",
        args.missing,
    ]
    if row.get("time_column"):
        add_option(command, "--time-column", row["time_column"])
    if row.get("selected_columns"):
        add_option(command, "--columns", row["selected_columns"])
    command.extend(["--seeds", *[str(seed) for seed in args.seeds]])
    add_option(command, "--max-lag", args.max_lag)
    add_option(command, "--train-fraction", args.train_fraction)
    add_option(command, "--ridge-alpha", args.ridge_alpha)
    add_option(command, "--lasso-alpha", args.lasso_alpha)
    add_option(command, "--graph-fraction", args.graph_fraction)
    add_option(command, "--graph-min-weight", args.graph_min_weight)
    add_option(command, "--graph-self-weight", args.graph_self_weight)
    add_option(command, "--graph-weight-power", args.graph_weight_power)
    add_option(command, "--window-size", args.window_size)
    add_option(command, "--window-stride", args.window_stride)
    add_option(command, "--window-aggregate", args.window_aggregate)
    add_option(command, "--state-aggregate", args.state_aggregate)
    add_option(command, "--stcg-v1-states", args.stcg_v1_states)
    add_option(command, "--stcg-v1-min-improvement", args.stcg_v1_min_improvement)
    if args.stcg_v1_force_states:
        command.append("--stcg-v1-force-states")
    add_option(command, "--diagnostic-top-k", args.diagnostic_top_k)
    add_option(command, "--output-dir", output_dir)
    add_option(command, "--figure-dir", figure_dir)
    return command, output_dir, figure_dir


def display_command(command: list[str]) -> str:
    return subprocess.list2cmdline(["python", *command[1:]])


def write_plan(
    path: Path,
    rows: list[dict[str, str]],
    commands: list[tuple[dict[str, str], list[str], Path, Path]],
    execute: bool,
) -> None:
    lines = [
        "# Real Benchmark Run Plan",
        "",
        f"Mode: `{'execute' if execute else 'dry-run'}`.",
        "",
    ]
    if not rows:
        lines.extend(
            [
                "No inventory rows matched the requested statuses/datasets.",
                "",
                "Run the inventory first:",
                "",
                "```powershell",
                "python scripts\\check_real_data_inventory.py",
                "```",
            ]
        )
        while lines and lines[-1] == "":
            lines.pop()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.extend(
        [
            "| Dataset | Status | Rows | Numeric Columns | Output Dir |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row, _, output_dir, _ in commands:
        lines.append(
            "| {dataset} | {status} | {rows} | {numeric_columns} | `{output}` |".format(
                dataset=row["dataset"],
                status=row["status"],
                rows=row["rows"],
                numeric_columns=row["numeric_columns"],
                output=output_dir,
            )
        )
    lines.extend(["", "## Commands", ""])
    for row, command, _, _ in commands:
        lines.extend(
            [
                f"### {row['dataset']}",
                "",
                "```powershell",
                display_command(command),
                "```",
                "",
            ]
        )
    while lines and lines[-1] == "":
        lines.pop()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_command(command: list[str]) -> dict[str, object]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> int:
    args = parse_args()
    rows = read_inventory(args.inventory_csv)
    selected = eligible_rows(rows, set(args.statuses), set(args.datasets))
    commands = [(row, *build_command(row, args)) for row in selected]
    write_plan(args.plan_output, selected, commands, args.execute)

    run_results = []
    if args.execute:
        for row, command, output_dir, figure_dir in commands:
            output_dir.mkdir(parents=True, exist_ok=True)
            figure_dir.mkdir(parents=True, exist_ok=True)
            result = run_command(command)
            run_results.append(
                {
                    "dataset": row["dataset"],
                    "status": row["status"],
                    "returncode": result["returncode"],
                    "output_dir": str(output_dir),
                    "figure_dir": str(figure_dir),
                }
            )
            if result["returncode"] != 0:
                raise RuntimeError(
                    "Downstream forecasting failed for {dataset} with return code {returncode}:\n{stderr}".format(
                        dataset=row["dataset"],
                        returncode=result["returncode"],
                        stderr=result["stderr"],
                    )
                )

    print(
        json.dumps(
            {
                "execute": args.execute,
                "inventory_csv": str(args.inventory_csv),
                "n_inventory_rows": len(rows),
                "n_selected": len(selected),
                "plan_output": str(args.plan_output),
                "runs": run_results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
