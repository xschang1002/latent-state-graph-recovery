"""Download and prepare UCI AReM activity multivariate time series."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _common import download, sha256  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/366/activity%2Brecognition%2Bsystem%2Bbased%2Bon%2Bmultisensor%2Bdata%2Bfusion%2Barem.zip"
ZIP_SHA256 = "a328bd9f4a4017546c2359b371c1a820441e9e83346922fde3c3b058c7209000"
SENSOR_COLUMNS = ["avg_rss12", "var_rss12", "avg_rss13", "var_rss13", "avg_rss23", "var_rss23"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--label-output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--zip-name", type=str, default="arem.zip")
    parser.add_argument("--output-name", type=str, default="AReMActivities.csv")
    parser.add_argument("--label-output-name", type=str, default="AReMActivities_labels.csv")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def natural_key(path: str) -> tuple[str, int, str]:
    parts = Path(path).parts
    activity = parts[0] if parts else ""
    match = re.search(r"dataset(\d+)\.csv$", path)
    index = int(match.group(1)) if match else -1
    return activity, index, path


def csv_members(archive: zipfile.ZipFile) -> list[str]:
    return sorted(
        [name for name in archive.namelist() if name.lower().endswith(".csv") and "/" in name],
        key=natural_key,
    )


def read_trial(archive: zipfile.ZipFile, name: str, sequence_id: int, offset: int) -> pd.DataFrame:
    activity = Path(name).parts[0]
    with archive.open(name) as f:
        frame = pd.read_csv(
            f,
            sep=r",|\s+",
            comment="#",
            header=None,
            names=["time", *SENSOR_COLUMNS],
            on_bad_lines="skip",
            engine="python",
        )
    for column in ["time", *SENSOR_COLUMNS]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[SENSOR_COLUMNS] = frame[SENSOR_COLUMNS].interpolate(limit_direction="both")
    frame = frame.dropna(subset=SENSOR_COLUMNS).reset_index(drop=True)
    frame["time_index"] = range(offset, offset + len(frame))
    frame["sample_in_sequence"] = range(len(frame))
    frame["sequence_id"] = sequence_id
    frame["activity"] = activity
    frame["source_file"] = name
    return frame


def prepare_arem_zip(zip_path: Path, output_path: Path, label_output_path: Path) -> dict[str, object]:
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = csv_members(archive)
        if not members:
            raise ValueError(f"{zip_path} does not contain AReM CSV trials")
        offset = 0
        for sequence_id, member in enumerate(members):
            frame = read_trial(archive, member, sequence_id, offset)
            frames.append(frame)
            offset += len(frame)

    combined = pd.concat(frames, ignore_index=True)
    output = combined[["time_index", *SENSOR_COLUMNS]].copy()
    labels = combined[["time_index", "activity", "sequence_id", "sample_in_sequence", "source_file"]].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    labels.to_csv(label_output_path, index=False)

    activities = labels["activity"].astype(str)
    transitions = int((activities.iloc[1:].to_numpy() != activities.iloc[:-1].to_numpy()).sum())
    return {
        "output_csv": str(output_path),
        "label_csv": str(label_output_path),
        "rows": int(output.shape[0]),
        "n_sequences": int(labels["sequence_id"].nunique()),
        "sensor_columns": SENSOR_COLUMNS,
        "n_sensors": len(SENSOR_COLUMNS),
        "missing_cells": int(output[SENSOR_COLUMNS].isna().sum().sum()),
        "activity_counts": {str(k): int(v) for k, v in activities.value_counts().sort_index().items()},
        "activity_transitions": transitions,
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

    prepared = prepare_arem_zip(zip_path, output_path, label_output_path)
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
