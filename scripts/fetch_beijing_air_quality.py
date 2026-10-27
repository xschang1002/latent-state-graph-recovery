"""Download and prepare Beijing multi-site PM2.5 station data."""

from __future__ import annotations

import argparse
import io
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
URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
ZIP_SHA256 = "b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4"
NESTED_ZIP = "PRSA2017_Data_20130301-20170228.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--zip-name", type=str, default="beijing_multi_site_air_quality.zip")
    parser.add_argument("--output-name", type=str, default="BeijingPM25.csv")
    parser.add_argument("--pollutant", type=str, default="PM2.5")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def station_csv_names(archive: zipfile.ZipFile) -> list[str]:
    return sorted(
        info.filename
        for info in archive.infolist()
        if info.filename.lower().endswith(".csv") and "PRSA_Data_" in info.filename
    )


def read_station_frame(archive: zipfile.ZipFile, name: str, pollutant: str) -> pd.DataFrame:
    with archive.open(name) as f:
        frame = pd.read_csv(f, na_values=["NA"])
    if pollutant not in frame.columns:
        raise ValueError(f"{name} does not contain pollutant column {pollutant!r}")
    station_values = frame["station"].dropna().astype(str).unique()
    if len(station_values) != 1:
        raise ValueError(f"{name} must contain exactly one station, got {station_values}")
    station = station_values[0]
    timestamps = pd.to_datetime(
        {
            "year": frame["year"],
            "month": frame["month"],
            "day": frame["day"],
            "hour": frame["hour"],
        },
        errors="raise",
    )
    values = pd.to_numeric(frame[pollutant], errors="coerce")
    return pd.DataFrame({"datetime": timestamps, station: values})


def prepare_pm25_zip(zip_path: Path, output_path: Path, pollutant: str = "PM2.5") -> dict[str, object]:
    with zipfile.ZipFile(zip_path) as outer:
        nested_bytes = outer.read(NESTED_ZIP)
    with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
        names = station_csv_names(nested)
        if not names:
            raise ValueError("No PRSA station CSV files found in nested archive")
        wide: pd.DataFrame | None = None
        stations: list[str] = []
        for name in names:
            station_frame = read_station_frame(nested, name, pollutant)
            station = station_frame.columns[1]
            stations.append(station)
            wide = station_frame if wide is None else wide.merge(station_frame, on="datetime", how="outer")

    if wide is None:
        raise ValueError("No station data was loaded")
    wide = wide.sort_values("datetime").drop_duplicates(subset=["datetime"])
    wide[stations] = wide[stations].interpolate(limit_direction="both")
    wide = wide.dropna(subset=stations)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(output_path, index=False)
    return {
        "output_csv": str(output_path),
        "rows": int(wide.shape[0]),
        "stations": stations,
        "n_stations": len(stations),
        "missing_cells": int(wide[stations].isna().sum().sum()),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.output_dir / args.zip_name
    output_path = args.output_dir / args.output_name
    if args.force or not zip_path.exists():
        download(URL, zip_path)
    actual_hash = sha256(zip_path)
    if actual_hash != ZIP_SHA256:
        raise ValueError(f"Hash mismatch for {zip_path}: expected {ZIP_SHA256}, got {actual_hash}")

    prepared = prepare_pm25_zip(zip_path, output_path, pollutant=args.pollutant)
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
