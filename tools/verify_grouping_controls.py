"""Read-only checks for exact matched partitions and state-factor provenance."""
from __future__ import annotations

import hashlib
import json
from math import comb
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_grouping_controls import analyse, coverage


def verify(root: Path = ROOT) -> dict:
    folder = root / "results/revision2"
    protocol = json.loads((folder / "grouping_controls_protocol.json").read_text())
    for source, expected in protocol["source_sha256"].items():
        assert hashlib.sha256((root / source).read_bytes()).hexdigest() == expected, source
    metadata = json.loads((folder / "grouping_replay_manifest.json").read_text())
    assert hashlib.sha256((folder / "grouping_replay_outputs.npz").read_bytes()).hexdigest() == metadata["cache_sha256"]
    for source, expected in metadata["fit_source_sha256"].items():
        assert hashlib.sha256((root / source).read_bytes()).hexdigest() == expected, source
    original = pd.read_csv(folder / "temporal_grid_detail.csv")
    core = pd.read_csv(folder / "state_decomposition.csv")
    with np.load(folder / "grouping_replay_outputs.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    factors, partitions, seed_summary = analyse(arrays, original, core)
    for name, replay in (("grouping_state_factors_detail.csv", factors),
                         ("matched_random_partitions.csv", partitions),
                         ("matched_random_seed_summary.csv", seed_summary),
                         ("temporal_grid_coverage.csv", coverage(original))):
        actual = pd.read_csv(folder / name, dtype={"membership": str})
        pd.testing.assert_frame_equal(replay, actual, check_exact=False, atol=1e-15, rtol=0)
    assert len(factors) == 60 and len(seed_summary) == 20
    assert factors.identity_gap.max() < 1e-14
    np.testing.assert_allclose(factors.hard_temporal_auroc, 0.5 + 0.5 * factors.J_times_C, atol=1e-15, rtol=0)
    for seed, group in partitions.groupby("seed"):
        row = seed_summary[seed_summary.seed == seed].iloc[0]
        assert len(group) == row.n_partitions == comb(7, int(row.minority_size))
        assert not group.membership.duplicated().any()
        assert all(value.count("1") == row.minority_size for value in group.membership)
        assert abs(group.temporal_auroc.mean() - row.matched_random_temporal_auroc) < 1e-15
    factor_summary = pd.read_csv(folder / "grouping_state_factors_summary.csv")
    for row in factor_summary.itertuples(index=False):
        values = factors[factors.condition == row.condition]
        for metric in ("J", "C_edge", "J_times_C", "hard_temporal_auroc", "raw_temporal_auroc"):
            assert abs(values[metric].mean() - getattr(row, metric)) < 1e-15
    return {"original_fits_reconstructed": 40, "new_seeds": 0, "factor_rows": len(factors),
            "seed_units": len(seed_summary), "enumerated_partitions": len(partitions),
            "maximum_identity_gap": float(factors.identity_gap.max()), "coverage_entries": 35}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
