"""Download and prepare UCI HAR smartphone inertial window signals."""

from __future__ import annotations

import argparse
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
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
MIRROR_URL = "https://d396qusza40orc.cloudfront.net/getdata%2Fprojectfiles%2FUCI%20HAR%20Dataset.zip"
ZIP_SHA256 = "50dabbc800629611831a85b8b71c87040525ca4af6a152c6ba9360ecee6b92dc"
SIGNALS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--label-output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--zip-name", type=str, default="uci_har.zip")
    parser.add_argument("--output-name", type=str, default="UCIHARWindowSignals.csv")
    parser.add_argument("--label-output-name", type=str, default="UCIHARWindowSignals_labels.csv")
    parser.add_argument("--source-url", type=str, default=MIRROR_URL)
    parser.add_argument("--include-std", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_text_array(archive: zipfile.ZipFile, name: str, dtype: type = float) -> np.ndarray:
    with archive.open(name) as f:
        return np.loadtxt(f, dtype=dtype)


def read_activity_labels(archive: zipfile.ZipFile) -> dict[int, str]:
    with archive.open("UCI HAR Dataset/activity_labels.txt") as f:
        labels: dict[int, str] = {}
        for raw_line in f:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            activity_id, activity = line.split(maxsplit=1)
            labels[int(activity_id)] = activity
    return labels


def signal_feature_frame(archive: zipfile.ZipFile, split: str, include_std: bool) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for signal in SIGNALS:
        path = f"UCI HAR Dataset/{split}/Inertial Signals/{signal}_{split}.txt"
        values = read_text_array(archive, path, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        data[f"{signal}_mean"] = values.mean(axis=1)
        if include_std:
            data[f"{signal}_std"] = values.std(axis=1)
    return pd.DataFrame(data)


def prepare_har_zip(zip_path: Path, output_path: Path, label_output_path: Path, include_std: bool = False) -> dict[str, object]:
    frames: list[pd.DataFrame] = []
    label_frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as archive:
        activity_names = read_activity_labels(archive)
        offset = 0
        for split in ("train", "test"):
            features = signal_feature_frame(archive, split, include_std)
            subjects = read_text_array(archive, f"UCI HAR Dataset/{split}/subject_{split}.txt", dtype=int).reshape(-1)
            activity_ids = read_text_array(archive, f"UCI HAR Dataset/{split}/y_{split}.txt", dtype=int).reshape(-1)
            if len(features) != len(subjects) or len(features) != len(activity_ids):
                raise ValueError(f"UCI HAR {split} feature/label length mismatch")
            n_rows = len(features)
            features.insert(0, "time_index", np.arange(offset, offset + n_rows, dtype=int))
            frames.append(features)
            label_frames.append(
                pd.DataFrame(
                    {
                        "time_index": np.arange(offset, offset + n_rows, dtype=int),
                        "activity_id": activity_ids.astype(int),
                        "activity": [activity_names[int(label)] for label in activity_ids],
                        "subject": subjects.astype(int),
                        "source_split": split,
                        "window_id": np.arange(n_rows, dtype=int),
                    }
                )
            )
            offset += n_rows

    output = pd.concat(frames, ignore_index=True)
    labels = pd.concat(label_frames, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    labels.to_csv(label_output_path, index=False)

    activity_series = labels["activity"].astype(str)
    transitions = int((activity_series.iloc[1:].to_numpy() != activity_series.iloc[:-1].to_numpy()).sum())
    feature_columns = [column for column in output.columns if column != "time_index"]
    return {
        "output_csv": str(output_path),
        "label_csv": str(label_output_path),
        "rows": int(output.shape[0]),
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "n_subjects": int(labels["subject"].nunique()),
        "activity_counts": {str(k): int(v) for k, v in activity_series.value_counts().sort_index().items()},
        "activity_transitions": transitions,
        "include_std": include_std,
        "missing_cells": int(output[feature_columns].isna().sum().sum()),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.output_dir / args.zip_name
    output_path = args.output_dir / args.output_name
    label_output_path = args.label_output_dir / args.label_output_name
    if args.force or not zip_path.exists():
        download(args.source_url, zip_path)
    actual_hash = sha256(zip_path)
    if actual_hash != ZIP_SHA256:
        raise ValueError(f"Hash mismatch for {zip_path}: expected {ZIP_SHA256}, got {actual_hash}")

    prepared = prepare_har_zip(zip_path, output_path, label_output_path, include_std=args.include_std)
    print(
        json.dumps(
            {
                "source_url": args.source_url,
                "uci_url": UCI_URL,
                "mirror_url": MIRROR_URL,
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
