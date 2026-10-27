"""Shared helpers for the scripts in this directory.

Each function here is the canonical implementation of a helper that was
previously copy-pasted across multiple fetch/experiment scripts. The
extraction is behavior-preserving: no numeric logic, defaults, or output
formats changed. Helpers whose formatting semantics genuinely differed
between scripts (the ``fmt`` variants) are kept as separate, explicitly
named functions rather than being merged.

Scripts are invoked as ``python scripts/<name>.py`` from the repository root,
so this directory is on ``sys.path`` and ``import _common`` resolves directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from stcg import STCGStateConfig

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RESULTS_TABLES_DIR = ROOT / "results" / "tables"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    with urllib.request.urlopen(url) as response:
        path.write_bytes(response.read())


def parse_columns(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def interpolate_within_segments(frame: pd.DataFrame, split_points: list[int]) -> pd.DataFrame:
    """Interpolate each chronological segment independently.

    ``split_points`` are positional row indices at which a new segment
    starts (for example the train/test boundary of a chronological split).
    Running ``interpolate(limit_direction="both")`` on the full series
    before the split lets train gaps borrow test-period values (and vice
    versa); interpolating within each segment keeps every fill value inside
    its own segment. Rows that remain NaN inside their segment are left in
    place for the caller's usual ``dropna``.
    """
    points = sorted({int(point) for point in split_points if 0 < int(point) < len(frame)})
    if not points:
        return frame.interpolate(limit_direction="both")
    boundaries = [0, *points, len(frame)]
    out = frame.copy()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        out.iloc[start:end] = frame.iloc[start:end].interpolate(limit_direction="both")
    return out


def majority_label(values: list[str]) -> tuple[str, float]:
    if not values:
        raise ValueError("Cannot compute majority label for an empty window")
    counts = Counter(values)
    majority, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return str(majority), float(count / len(values))


def align_labels(
    true_labels: list[str],
    inferred_states: list[int],
) -> tuple[list[str], dict[int, str], float]:
    # Imports stay lazy so ``import _common`` does not require numpy/scipy for
    # scripts that only need the file/CSV helpers.
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if len(true_labels) != len(inferred_states):
        raise ValueError("true_labels and inferred_states must have the same length")
    true_values = sorted(set(str(value) for value in true_labels))
    inferred_values = sorted(set(int(value) for value in inferred_states))
    if not true_values or not inferred_values:
        return [], {}, math.nan

    true_index = {value: idx for idx, value in enumerate(true_values)}
    inferred_index = {value: idx for idx, value in enumerate(inferred_values)}
    matrix = np.zeros((len(true_values), len(inferred_values)), dtype=int)
    for true_label, inferred_state in zip(true_labels, inferred_states):
        matrix[true_index[str(true_label)], inferred_index[int(inferred_state)]] += 1

    row_ind, col_ind = linear_sum_assignment(-matrix)
    mapping = {inferred_values[col]: true_values[row] for row, col in zip(row_ind, col_ind)}
    fallback = true_values[0]
    aligned = [mapping.get(int(state), fallback) for state in inferred_states]
    accuracy = float(np.mean([str(a) == str(t) for a, t in zip(aligned, true_labels)]))
    return aligned, mapping, accuracy


def counts_text(values: list[object]) -> str:
    counts = Counter(str(value) for value in values)
    return ";".join(f"{label}:{count}" for label, count in sorted(counts.items()))


def fmt(value: object, digits: int = 3) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value_float):
        return "nan"
    return f"{value_float:.{digits}f}"


def fmt_na(value: object, digits: int = 3) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def fmt_na_lenient(value: object, digits: int = 3) -> str:
    if value in {None, ""}:
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def fmt_dash(value: object, digits: int = 3) -> str:
    value = float(value)
    return "--" if math.isnan(value) else f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def warn_if_results_tables_output(
    script: str,
    output_dir: Path,
    n_seeds: int,
    canonical_seeds: int,
) -> None:
    """Print a one-line stderr caution when a short run targets results/tables.

    The tracked main tables are produced with the canonical seed counts from
    docs/reproduction_manifest.md; quick checks should redirect output with
    ``--output-dir experiments\\logs\\quickcheck`` instead of overwriting them.
    Script defaults are unchanged; this is only a guardrail warning.
    """
    if n_seeds >= canonical_seeds:
        return
    if output_dir.resolve() != RESULTS_TABLES_DIR.resolve():
        return
    print(
        f"[{script}] caution: writing into results/tables with only "
        f"{n_seeds} seed(s); the tracked main tables use {canonical_seeds} "
        "seeds (docs/reproduction_manifest.md). Quick checks should pass "
        "--output-dir experiments\\logs\\quickcheck.",
        file=sys.stderr,
    )


def state_config(    args: argparse.Namespace,
    seed: int,
    n_states: int = 2,
    seed_offset: int = 70_000,
    auto_states: bool = True,
    min_state_improvement: float = 0.12,
) -> STCGStateConfig:
    """Build the lasso-window STCGStateConfig shared by the state experiments.

    ``seed_offset`` keeps each script's historical seed stream unchanged
    (70_000/80_000/90_000 depending on the caller). The stcg import stays
    lazy so that ``import _common`` does not require torch or sklearn.
    """
    from stcg import STCGStateConfig

    return STCGStateConfig(
        n_states=n_states,
        base_model="lasso",
        lasso_alpha=args.lasso_alpha,
        window_size=args.window_size,
        stride=args.window_stride,
        aggregate=args.window_aggregate,
        state_aggregate=args.state_aggregate,
        auto_states=auto_states,
        min_state_improvement=min_state_improvement,
        contrast_mix=0.0,
        seed=seed + seed_offset,
    )
