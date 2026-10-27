"""Download and prepare UCI Occupancy Detection sensor time series."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _common import download, sha256  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip"
ZIP_SHA256 = "4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a"
FILES = ["datatest.txt", "datatraining.txt", "datatest2.txt"]
SENSOR_COLUMNS = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--label-output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--zip-name", type=str, default="occupancy_data.zip")
    parser.add_argument("--output-name", type=str, default="OccupancySensors.csv")
    parser.add_argument("--label-output-name", type=str, default="OccupancySensors_labels.csv")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_occupancy_file(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with archive.open(name) as f:
        frame = pd.read_csv(f)
    required = ["date", *SENSOR_COLUMNS, "Occupancy"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    frame = frame[required].copy()
    frame["datetime"] = pd.to_datetime(frame["date"], errors="raise")
    frame["source_split"] = name.replace(".txt", "")
    for column in [*SENSOR_COLUMNS, "Occupancy"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop(columns=["date"])


def prepare_occupancy_zip(
    zip_path: Path,
    output_path: Path,
    label_output_path: Path,
) -> dict[str, object]:
    frames = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in FILES:
            frames.append(read_occupancy_file(archive, name))

    frame = pd.concat(frames, ignore_index=True).sort_values("datetime")
    frame = frame.drop_duplicates(subset=["datetime"], keep="first")
    frame[SENSOR_COLUMNS] = frame[SENSOR_COLUMNS].interpolate(limit_direction="both")
    frame = frame.dropna(subset=[*SENSOR_COLUMNS, "Occupancy"])
    frame["Occupancy"] = frame["Occupancy"].astype(int)

    output = frame[["datetime", *SENSOR_COLUMNS]].copy()
    labels = frame[["datetime", "Occupancy", "source_split"]].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    labels.to_csv(label_output_path, index=False)

    occupancy_values = labels["Occupancy"].to_numpy()
    transitions = int((occupancy_values[1:] != occupancy_values[:-1]).sum()) if len(occupancy_values) > 1 else 0
    return {
        "output_csv": str(output_path),
        "label_csv": str(label_output_path),
        "rows": int(output.shape[0]),
        "sensor_columns": SENSOR_COLUMNS,
        "n_sensors": len(SENSOR_COLUMNS),
        "missing_cells": int(output[SENSOR_COLUMNS].isna().sum().sum()),
        "occupancy_fraction": float(labels["Occupancy"].mean()),
        "occupancy_transitions": transitions,
        "start": str(output["datetime"].iloc[0]),
        "end": str(output["datetime"].iloc[-1]),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.output_dir / args.zip_name
    output_path = args.output_dir / args.output_name
    label_output_path = args.label_output_dir / args.label_output_name
    if args.force or not zip_path.exists():
        download(URL, zip_path)
    actual_hash = sha256(zip_path)
    if actual_hash != ZIP_SHA256:
        raise ValueError(f"Hash mismatch for {zip_path}: expected {ZIP_SHA256}, got {actual_hash}")

    prepared = prepare_occupancy_zip(zip_path, output_path, label_output_path)
    print(
        json.dumps(
            {
                "source_url": URL,
                "source_zip": str(zip_path),
                "source_sha256": actual_hash,
                **prepared,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
