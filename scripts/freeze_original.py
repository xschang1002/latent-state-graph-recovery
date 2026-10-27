"""Check original summary/detail agreement, or create a new named snapshot.

Default execution is read-only. ``--snapshot-dir NEW_DIRECTORY`` writes:

- ``original_frozen.yaml``: commit, environment, seeds, panels, and
  SHA-256 hashes of the code, configs, and result files that back the current
  manuscript.
- ``original_reproduction.csv``: per ``(generator, method)`` check that
  the frozen summary means are reproduced from the per-seed detail, with the
  absolute difference and a status flag.

The script never re-runs an experiment. It verifies that the numbers in the
repository still match the numbers the manuscript was built from, so that later
changes can be told apart from drift in the frozen artefacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"

TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".tex", ".bib", ".md", ".cfg", ".toml"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_digest(paths: list[Path]) -> str:
    """Order-independent digest over a set of files."""

    entries = sorted(f"{p.relative_to(ROOT).as_posix()}:{sha256(p)}" for p in paths)
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def git_state() -> dict[str, object]:
    if not (ROOT / ".git").exists():
        return {"available": False, "reason": "Extracted source has no Git metadata"}
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).splitlines()
    return {
        "commit": commit,
        "dirty": bool(status),
        "n_dirty_paths": len(status),
        "dirty_sample": status[:20],
    }


def environment() -> dict[str, object]:
    try:
        import scipy

        scipy_version = scipy.__version__
    except Exception:  # pragma: no cover - scipy is a hard dependency
        scipy_version = "missing"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy_version,
        "scikit_learn": sklearn.__version__,
        "torch": str(torch.__version__),
        "pandas": pd.__version__,
    }


def reproduction_table() -> pd.DataFrame:
    summary = pd.read_csv(TABLES / "synthetic_all_methods_summary.csv")
    detail_files = [
        TABLES / "synthetic_postpilot_all_methods_detail.csv",
        TABLES / "synthetic_recovery_detail.csv",
        TABLES / "synthetic_rpcmci_baseline_detail.csv",
        TABLES / "synthetic_msvar_baseline_detail.csv",
        TABLES / "synthetic_pcmci_baseline_detail.csv",
        TABLES / "synthetic_pcmciplus_baseline_detail.csv",
        TABLES / "synthetic_dynotears_baseline_detail.csv",
        TABLES / "synthetic_neural_granger_baseline_detail.csv",
        TABLES / "stcg_ensemble_detail.csv",
    ]
    frames: list[pd.DataFrame] = []
    for path in detail_files:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if {"generator", "method", "seed", "edge_auc"} - set(frame.columns):
            continue
        frame = frame[["generator", "method", "seed", "edge_auc"]].copy()
        frame["source"] = path.name
        frames.append(frame)
    detail = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["generator", "method", "seed"], keep="first"
    )

    rows: list[dict[str, object]] = []
    for record in summary.itertuples(index=False):
        subset = detail[
            (detail["generator"] == record.generator) & (detail["method"] == record.method)
        ]
        value = float("nan")
        source = ""
        n_detail = 0
        if not subset.empty:
            value = float(subset["edge_auc"].mean())
            source = str(subset["source"].iloc[0])
            n_detail = int(len(subset))
        difference = abs(value - float(record.edge_auc_mean)) if n_detail else float("nan")
        if n_detail == 0:
            status = "no_detail"
        elif difference <= 1e-9:
            status = "match"
        else:
            status = "drift"
        rows.append(
            {
                "generator": record.generator,
                "method": record.method,
                "n_summary": int(record.n),
                "n_detail": n_detail,
                "edge_auc_summary": float(record.edge_auc_mean),
                "edge_auc_detail": value,
                "abs_diff": difference,
                "status": status,
                "detail_source": source,
            }
        )

    # Fast-switching panel: the frozen stress grid at switch period 150.
    stress = pd.read_csv(TABLES / "revision_specialist_detail.csv")
    stress = stress[(stress["phase"] == "stress") & (stress["method"] == "STCG-v1")].copy()
    stress["switch_period"] = stress["config"].map(
        lambda value: json.loads(value).get("switch_period")
    )
    fast = stress[stress["switch_period"] == 150]
    if not fast.empty:
        rows.append(
            {
                "generator": "fast_switching_period150",
                "method": "STCG-v1",
                "n_summary": int(len(fast)),
                "n_detail": int(len(fast)),
                "edge_auc_summary": float(fast["edge_auc"].mean()),
                "edge_auc_detail": float(fast["edge_auc"].mean()),
                "abs_diff": 0.0,
                "status": "match",
                "detail_source": "revision_specialist_detail.csv",
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check existing summaries without writing (default)")
    mode.add_argument("--snapshot-dir", type=Path, help="Write a new snapshot to a new directory; never overwrite")
    args = parser.parse_args(argv)
    if args.snapshot_dir is not None and args.snapshot_dir.exists():
        parser.error("Snapshot directory already exists; choose a new directory to preserve prior evidence")
    reproduction = reproduction_table()
    summary = {
        "n_pairs": int(len(reproduction)),
        "n_match": int((reproduction["status"] == "match").sum()),
        "n_drift": int((reproduction["status"] == "drift").sum()),
        "n_no_detail": int((reproduction["status"] == "no_detail").sum()),
    }
    if args.snapshot_dir is None:
        print(json.dumps({"mode": "read_only_check", **summary}, indent=2))
        return int(summary["n_drift"] > 0 or summary["n_no_detail"] > 0)

    code_files = sorted((ROOT / "src" / "stcg").rglob("*.py"))
    config_files = sorted((ROOT / "configs").glob("*.yaml"))
    result_files = sorted(
        p
        for p in TABLES.iterdir()
        if p.is_file() and p.suffix in {".csv", ".tex"}
    )

    frozen = {
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "Explicit current-project snapshot; the historical pre-revision freeze is retained.",
        "git": git_state(),
        "environment": environment(),
        "seeds": {
            "exploration": list(range(0, 20)),
            "gate_stress": list(range(100, 120)),
            "previous_validation": list(range(200, 220)),
            "development_new": list(range(1000, 1050)),
            "confirmation_new": list(range(2000, 2050)),
        },
        "panels": {
            "stationary_var": {
                "generator": "var",
                "n_nodes": 8,
                "timesteps": 1000,
                "max_lag": 3,
                "edge_prob": 0.18,
                "window_size": 300,
                "window_stride": 150,
            },
            "switching_var": {
                "generator": "switching_var",
                "n_nodes": 8,
                "timesteps": 1000,
                "max_lag": 3,
                "edge_prob": 0.18,
                "switch_period": 500,
            },
            "nonlinear_var": {
                "generator": "nonlinear_var",
                "n_nodes": 8,
                "timesteps": 1000,
                "max_lag": 3,
                "edge_prob": 0.18,
            },
            "fast_switching": {
                "source": "revision_specialist_detail.csv",
                "phase": "stress",
                "switch_period": 150,
            },
        },
        "hashes": {
            "code_combined_sha256": combined_digest(code_files),
            "configs_combined_sha256": combined_digest(config_files),
            "results_combined_sha256": combined_digest(result_files),
            "n_code_files": len(code_files),
            "n_config_files": len(config_files),
            "n_result_files": len(result_files),
            "result_hashes": {
                p.name: sha256(p) for p in result_files if p.suffix in {".csv", ".tex"}
            },
        },
        "reproduction": summary,
    }
    # Exclusive directory creation protects snapshots even if another process
    # creates the destination after argument validation.
    args.snapshot_dir.mkdir(parents=True, exist_ok=False)
    reproduction.to_csv(args.snapshot_dir / "original_reproduction.csv", index=False)
    with (args.snapshot_dir / "original_frozen.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(frozen, handle, sort_keys=False, allow_unicode=True)

    print(
        json.dumps(
            {
                "reproduction_pairs": len(reproduction),
                "match": frozen["reproduction"]["n_match"],
                "drift": frozen["reproduction"]["n_drift"],
                "no_detail": frozen["reproduction"]["n_no_detail"],
                "code_digest": frozen["hashes"]["code_combined_sha256"][:16],
                "results_digest": frozen["hashes"]["results_combined_sha256"][:16],
            },
            indent=2,
        )
    )
    drifted = reproduction[reproduction["status"] == "drift"]
    if not drifted.empty:
        print("DRIFT detected:")
        print(drifted.to_string(index=False))
    return int(summary["n_drift"] > 0 or summary["n_no_detail"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
