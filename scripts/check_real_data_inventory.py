"""Inspect local real-data CSV files before downstream benchmarking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _common import parse_columns, write_csv  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
TIME_COLUMN_CANDIDATES = ["datetime", "date", "timestamp", "time", "time_index"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--columns", type=str, default="")
    parser.add_argument("--time-column", type=str, default="")
    parser.add_argument("--dataset-name", type=str, default="")
    parser.add_argument("--min-rows", type=int, default=300)
    parser.add_argument("--preferred-rows", type=int, default=1000)
    parser.add_argument("--min-numeric-columns", type=int, default=2)
    parser.add_argument("--preferred-numeric-columns", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    return parser.parse_args()


def csv_paths(args: argparse.Namespace) -> list[Path]:
    if args.input_csv is not None:
        return [args.input_csv]
    if not args.raw_dir.exists():
        return []
    return sorted(path for path in args.raw_dir.glob("*.csv") if path.is_file())


def dataset_name(path: Path, explicit_name: str) -> str:
    return explicit_name if explicit_name else path.stem


def command_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def detect_time_column(frame: pd.DataFrame, explicit: str) -> str:
    if explicit:
        return explicit
    lower_to_actual = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in lower_to_actual:
            return lower_to_actual[candidate]
    return ""


def command_for(path: Path, args: argparse.Namespace, columns: list[str], time_column: str) -> str:
    parts = [
        "python",
        "scripts\\run_downstream_forecasting.py",
        "--input-csv",
        command_path(path),
    ]
    if time_column:
        parts.extend(["--time-column", time_column])
    if args.columns:
        parts.extend(["--columns", ",".join(columns)])
    parts.extend(["--dataset-name", dataset_name(path, args.dataset_name)])
    return " ".join(parts)


def inspect_csv(path: Path, args: argparse.Namespace) -> dict[str, object]:
    row: dict[str, object] = {
        "file": str(path),
        "dataset": dataset_name(path, args.dataset_name),
        "status": "blocked",
        "rows": 0,
        "columns": 0,
        "numeric_columns": 0,
        "selected_columns": "",
        "missing_cells": 0,
        "missing_fraction": 0.0,
        "empty_numeric_rows": 0,
        "time_column": args.time_column,
        "timestamp_parse_failures": "",
        "duplicate_timestamps": "",
        "timestamp_monotonic": "",
        "issues": "",
        "suggested_command": "",
    }
    issues: list[str] = []
    warnings: list[str] = []

    if not path.exists():
        row["status"] = "error"
        row["issues"] = "file not found"
        return row

    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - error type depends on pandas parser.
        row["status"] = "error"
        row["issues"] = f"read failed: {exc}"
        return row

    row["rows"] = int(frame.shape[0])
    row["columns"] = int(frame.shape[1])

    time_column = detect_time_column(frame, args.time_column)
    row["time_column"] = time_column
    if time_column:
        if time_column not in frame.columns:
            issues.append(f"time column missing: {time_column}")
        else:
            timestamps = pd.to_datetime(frame[time_column], errors="coerce")
            failures = int(timestamps.isna().sum())
            row["timestamp_parse_failures"] = failures
            row["duplicate_timestamps"] = int(timestamps.duplicated().sum())
            row["timestamp_monotonic"] = bool(timestamps.is_monotonic_increasing)
            if failures:
                warnings.append(f"{failures} timestamp parse failures")
            if not bool(timestamps.is_monotonic_increasing):
                warnings.append("timestamps are not monotonic")
            if int(timestamps.duplicated().sum()):
                warnings.append("duplicate timestamps present")

    requested_columns = parse_columns(args.columns)
    if requested_columns:
        missing = [column for column in requested_columns if column not in frame.columns]
        if missing:
            issues.append(f"requested columns missing: {missing}")
        available_columns = [column for column in requested_columns if column in frame.columns]
        numeric = frame[available_columns].apply(pd.to_numeric, errors="coerce")
        selected_columns = available_columns
    else:
        drop_columns = [time_column] if time_column else []
        numeric = frame.drop(columns=drop_columns, errors="ignore").select_dtypes(include=["number"])
        selected_columns = [str(column) for column in numeric.columns]

    row["numeric_columns"] = int(numeric.shape[1])
    row["selected_columns"] = ",".join(selected_columns)

    if numeric.shape[1] > 0:
        missing_cells = int(numeric.isna().sum().sum())
        total_cells = int(numeric.shape[0] * numeric.shape[1])
        row["missing_cells"] = missing_cells
        row["missing_fraction"] = missing_cells / total_cells if total_cells else 0.0
        row["empty_numeric_rows"] = int(numeric.isna().all(axis=1).sum())

    if int(row["rows"]) < args.min_rows:
        issues.append(f"row count below minimum {args.min_rows}")
    elif int(row["rows"]) < args.preferred_rows:
        warnings.append(f"row count below preferred {args.preferred_rows}")

    if int(row["numeric_columns"]) < args.min_numeric_columns:
        issues.append(f"numeric column count below minimum {args.min_numeric_columns}")
    elif int(row["numeric_columns"]) < args.preferred_numeric_columns:
        warnings.append(f"numeric column count below preferred {args.preferred_numeric_columns}")

    if float(row["missing_fraction"]) > 0.2:
        warnings.append("more than 20% numeric cells are missing")
    if int(row["empty_numeric_rows"]) > 0:
        warnings.append("some rows have no numeric values")

    if issues:
        row["status"] = "blocked"
    elif warnings:
        row["status"] = "warning"
    else:
        row["status"] = "ready"
    row["issues"] = "; ".join(issues + warnings)
    row["suggested_command"] = command_for(path, args, selected_columns, time_column)
    return row


INVENTORY_FIELDNAMES = [
    "file",
    "dataset",
    "status",
    "rows",
    "columns",
    "numeric_columns",
    "selected_columns",
    "missing_cells",
    "missing_fraction",
    "empty_numeric_rows",
    "time_column",
    "timestamp_parse_failures",
    "duplicate_timestamps",
    "timestamp_monotonic",
    "issues",
    "suggested_command",
]


def write_markdown(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "# Real Data Inventory",
        "",
        f"Scanned path: `{args.input_csv or args.raw_dir}`.",
        "",
    ]
    if not rows:
        lines.extend(
            [
                "No CSV files were found.",
                "",
                "Place a multivariate time-series CSV under `data/raw/`, then rerun:",
                "",
                "```powershell",
                "python scripts\\check_real_data_inventory.py",
                "```",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.extend(
        [
            "| File | Status | Rows | Numeric Columns | Missing Fraction | Issues |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {file} | {status} | {rows} | {numeric_columns} | {missing_fraction:.4f} | {issues} |".format(
                **row
            )
        )

    runnable = [row for row in rows if row["status"] in {"ready", "warning"}]
    if runnable:
        lines.extend(["", "## Suggested Commands", ""])
        for row in runnable:
            lines.extend(
                [
                    f"### {row['dataset']}",
                    "",
                    "```powershell",
                    str(row["suggested_command"]),
                    "```",
                    "",
                ]
            )
    while lines and lines[-1] == "":
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = csv_paths(args)
    rows = [inspect_csv(path, args) for path in paths]

    csv_path = args.output_dir / "real_data_inventory.csv"
    markdown_path = args.output_dir / "real_data_inventory.md"
    write_csv(csv_path, rows, INVENTORY_FIELDNAMES)
    write_markdown(markdown_path, rows, args)

    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    print(
        json.dumps(
            {
                "inventory_csv": str(csv_path),
                "inventory_md": str(markdown_path),
                "n_csv": len(rows),
                "status_counts": counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
