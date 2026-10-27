"""Build the ESS PV -> subsystem map from the reduced archive layout.

The reduced Zenodo archive stores one directory per subsystem and one CSV per
process variable, but does not ship ``systemoverview.csv``.  The directory
layout is therefore the authoritative mapping; this script records it as a CSV
so downstream scripts never re-derive it.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "raw" / "ess" / "accp_dataset_reduced.zip"
OUTPUT = ROOT / "data" / "external" / "essdata" / "pv_subsystem_map.csv"


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Missing ESS archive: {ARCHIVE}")
    rows = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in archive.namelist():
            match = re.match(
                r"accp_dataset_reduced/Period_(\d)/([^/]+)/(.+)\.csv$", name
            )
            if not match:
                continue
            period, subsystem, pv = match.groups()
            rows.setdefault(pv, {}).setdefault(subsystem, set()).add(int(period))

    frame = pd.DataFrame(
        [
            {
                "pv": pv,
                "subsystem": sorted(subsystems)[0],
                "n_subsystems": len(subsystems),
                "periods": ",".join(str(p) for p in sorted(set().union(*subsystems.values()))),
            }
            for pv, subsystems in rows.items()
        ]
    ).sort_values("pv")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    ambiguous = frame[frame["n_subsystems"] > 1]
    print(f"mapped {len(frame)} PVs, ambiguous {len(ambiguous)} -> {OUTPUT}")


if __name__ == "__main__":
    main()
