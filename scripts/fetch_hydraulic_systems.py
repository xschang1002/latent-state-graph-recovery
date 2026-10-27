"""Download and prepare UCI Hydraulic Systems condition-monitoring time series."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _common import download, sha256  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
ZENODO_URL = "https://zenodo.org/record/1323611/files/data.zip?download=1"
ZENODO_MD5 = "ac0b28f34686660219c4e43417053f9e"
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00447/data.zip"
UCI_STATIC_URL = "https://archive.ics.uci.edu/static/public/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems.zip"
PROFILE_COLUMNS = [
    "cooler_condition",
    "valve_condition",
    "internal_pump_leakage",
    "hydraulic_accumulator",
    "stable_flag",
]
SENSOR_RATES = {
    "PS1": 100,
    "PS2": 100,
    "PS3": 100,
    "PS4": 100,
    "PS5": 100,
    "PS6": 100,
    "EPS1": 100,
    "FS1": 10,
    "FS2": 10,
    "TS1": 1,
    "TS2": 1,
    "TS3": 1,
    "TS4": 1,
    "VS1": 1,
    "CE": 1,
    "CP": 1,
    "SE": 1,
}
DEFAULT_SENSORS = ["PS1", "PS2", "PS3", "PS4", "FS1", "FS2", "TS1", "TS2", "VS1", "EPS1"]
SECONDS_PER_CYCLE = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--label-output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--zip-name", type=str, default="hydraulic_systems.zip")
    parser.add_argument("--output-name", type=str, default="HydraulicSystems.csv")
    parser.add_argument("--label-output-name", type=str, default="HydraulicSystems_labels.csv")
    parser.add_argument("--sensors", type=str, default=",".join(DEFAULT_SENSORS))
    parser.add_argument("--source-url", type=str, default=ZENODO_URL)
    parser.add_argument("--skip-md5-check", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_sensors(raw: str) -> list[str]:
    sensors = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [sensor for sensor in sensors if sensor not in SENSOR_RATES]
    if unknown:
        raise ValueError(f"Unknown sensors requested: {unknown}")
    if len(sensors) < 2:
        raise ValueError("At least two sensors are required")
    return sensors


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_name(archive: zipfile.ZipFile, filename: str) -> str:
    lower_filename = filename.lower()
    for name in archive.namelist():
        if Path(name).name.lower() == lower_filename:
            return name
    raise ValueError(f"Archive is missing required file: {filename}")


def read_sensor(archive: zipfile.ZipFile, sensor: str) -> np.ndarray:
    name = archive_name(archive, f"{sensor}.txt")
    with archive.open(name) as f:
        frame = pd.read_csv(f, sep=r"\s+", header=None)
    values = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    rate = SENSOR_RATES[sensor]
    expected_columns = rate * SECONDS_PER_CYCLE
    if values.shape[1] != expected_columns:
        raise ValueError(f"{sensor}.txt has {values.shape[1]} columns; expected {expected_columns}")
    return values.reshape(values.shape[0], SECONDS_PER_CYCLE, rate).mean(axis=2)


def read_profile(archive: zipfile.ZipFile) -> pd.DataFrame:
    name = archive_name(archive, "profile.txt")
    with archive.open(name) as f:
        frame = pd.read_csv(f, sep=r"\s+", header=None, names=PROFILE_COLUMNS)
    if frame.shape[1] != len(PROFILE_COLUMNS):
        raise ValueError("profile.txt does not match the expected five label columns")
    for column in PROFILE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    return frame


def prepare_hydraulic_zip(
    zip_path: Path,
    output_path: Path,
    label_output_path: Path,
    sensors: list[str] | None = None,
) -> dict[str, object]:
    selected_sensors = sensors or DEFAULT_SENSORS
    sensor_arrays: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(zip_path) as archive:
        profile = read_profile(archive)
        for sensor in selected_sensors:
            sensor_arrays[sensor] = read_sensor(archive, sensor)

    n_cycles = int(profile.shape[0])
    for sensor, values in sensor_arrays.items():
        if values.shape[0] != n_cycles:
            raise ValueError(f"{sensor}.txt has {values.shape[0]} cycles; expected {n_cycles}")

    base = pd.DataFrame(
        {
            "cycle": np.repeat(np.arange(n_cycles, dtype=int), SECONDS_PER_CYCLE),
            "second": np.tile(np.arange(SECONDS_PER_CYCLE, dtype=int), n_cycles),
        }
    )
    base["time_index"] = base["cycle"] * SECONDS_PER_CYCLE + base["second"]
    for sensor in selected_sensors:
        base[sensor] = sensor_arrays[sensor].reshape(n_cycles * SECONDS_PER_CYCLE)

    labels = base[["time_index", "cycle", "second"]].copy()
    for column in PROFILE_COLUMNS:
        labels[column] = np.repeat(profile[column].to_numpy(dtype=int), SECONDS_PER_CYCLE)

    output = base[["time_index", *selected_sensors]].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    labels.to_csv(label_output_path, index=False)

    return {
        "output_csv": str(output_path),
        "label_csv": str(label_output_path),
        "rows": int(output.shape[0]),
        "cycles": n_cycles,
        "seconds_per_cycle": SECONDS_PER_CYCLE,
        "sensor_columns": selected_sensors,
        "n_sensors": len(selected_sensors),
        "missing_cells": int(output[selected_sensors].isna().sum().sum()),
        "label_columns": PROFILE_COLUMNS,
        "label_state_counts": {
            column: {str(key): int(value) for key, value in profile[column].value_counts().sort_index().items()}
            for column in PROFILE_COLUMNS
        },
    }


def main() -> int:
    args = parse_args()
    sensors = parse_sensors(args.sensors)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.output_dir / args.zip_name
    output_path = args.output_dir / args.output_name
    label_output_path = args.label_output_dir / args.label_output_name

    if args.force or not zip_path.exists():
        download(args.source_url, zip_path)
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"{zip_path} is not a complete zip file. Re-download from {args.source_url}")
    actual_md5 = md5(zip_path)
    if not args.skip_md5_check and args.source_url == ZENODO_URL and actual_md5 != ZENODO_MD5:
        raise ValueError(f"MD5 mismatch for {zip_path}: expected {ZENODO_MD5}, got {actual_md5}")

    prepared = prepare_hydraulic_zip(zip_path, output_path, label_output_path, sensors)
    print(
        json.dumps(
            {
                "source_url": args.source_url,
                "zenodo_url": ZENODO_URL,
                "uci_url": UCI_URL,
                "uci_static_url": UCI_STATIC_URL,
                "source_zip": str(zip_path),
                "source_md5": actual_md5,
                "source_sha256": sha256(zip_path),
                **prepared,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
