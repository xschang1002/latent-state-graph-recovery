"""Verify grid pairing, protocol, historical agreement and subgroup accounting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def verify(root: Path = ROOT) -> dict:
    directory = root / "results/revision2"
    detail = pd.read_csv(directory / "temporal_grid_detail.csv")
    summary = pd.read_csv(directory / "temporal_grid_summary.csv")
    protocol = json.loads((directory / "temporal_grid_protocol.json").read_text())
    for source, expected in protocol["source_sha256"].items():
        assert hashlib.sha256((root / source).read_bytes()).hexdigest() == expected, source
    assert len(detail) == 3500 and len(summary) == 700
    assert detail.grid_entry.nunique() == 35 and detail.config_id.nunique() == 31
    assert set(detail.seed) == set(range(20))
    assert not detail.duplicated(["grid_entry", "seed", "method"]).any()
    assert (detail.groupby(["grid_entry", "seed"]).size() == 5).all()
    assert not summary.duplicated(["grid_entry", "method", "subset"]).any()
    assert (detail.groupby(["grid_entry", "seed"]).raw_temporal_delta_auroc.nunique() <= 1).all()
    assert (detail.groupby(["grid_entry", "seed"]).hard_gate_accepted.nunique() == 1).all()
    np.testing.assert_allclose(detail.delta_temporal, detail.temporal_delta_auroc - detail.raw_temporal_delta_auroc,
                               atol=1e-15, rtol=0, equal_nan=True)
    np.testing.assert_allclose(detail.h_over_D, detail.window_size / detail.switch_period)
    np.testing.assert_array_equal(detail.n_windows, (detail.timesteps - detail.window_size) // detail.stride + 1)
    np.testing.assert_array_equal(detail.evaluable, detail.temporal_delta_auroc.notna())
    hard = detail[detail.method == "kmeans_hard"]
    np.testing.assert_array_equal(hard.aggregation_executed, hard.hard_gate_accepted)
    assert not detail[detail.method == "raw_window"].aggregation_executed.any()
    assert detail[detail.method.isin(["gmm_soft", "oracle_label", "gmm_weighted_argmax"])].aggregation_executed.all()
    rejected = hard[~hard.hard_gate_accepted & hard.evaluable]
    assert (rejected.temporal_delta_auroc == 0.5).all()
    accepted = hard[hard.hard_gate_accepted]
    np.testing.assert_allclose(accepted.temporal_delta_auroc, accepted.candidate_temporal_delta_auroc,
                               atol=1e-15, rtol=0, equal_nan=True)
    # Duplicate contexts must be identical observations, not fresh replicates.
    value_columns = [c for c in detail if c not in ("grid_entry", "axis", "level", "config_json")]
    for _, group in detail.groupby(["config_id", "seed", "method"]):
        for _, row in group.iloc[1:].iterrows():
            pd.testing.assert_series_equal(group.iloc[0][value_columns], row[value_columns], check_names=False)
    for row in summary.itertuples(index=False):
        group = detail[(detail.grid_entry == row.grid_entry) & (detail.method == row.method)]
        if row.subset != "all":
            group = group[~group.hard_gate_accepted if row.subset == "hard_gate_rejected" else group[row.subset]]
        assert len(group) == row.n_seeds
        assert group.evaluable.sum() == row.n_evaluable
        for metric in ("temporal_delta_auroc", "delta_temporal", "candidate_delta_temporal"):
            expected = group[metric].mean()
            np.testing.assert_allclose(getattr(row, metric + "_mean"), expected, atol=1e-15, rtol=0, equal_nan=True)
        if row.n_evaluable < 5:
            assert np.isnan(row.delta_temporal_ci95_low) and np.isnan(row.delta_temporal_ci95_high)
    # Reuse all existing hard/oracle measurements on the original full grid.
    historical = pd.read_csv(root / "results/mechanism/mechanism_grid.csv")
    for method, metric in (("kmeans_hard", "temporal_delta_auroc"), ("oracle_label", "oracle_temporal_delta_auroc")):
        joined = detail[detail.method == method].merge(historical, on=["axis", "level", "seed"], suffixes=("_new", "_old"), validate="1:1")
        observed = joined.temporal_delta_auroc_new
        expected = joined.temporal_delta_auroc_old if method == "kmeans_hard" else joined[metric]
        np.testing.assert_allclose(observed, expected, atol=1e-15, rtol=0, equal_nan=True)
        np.testing.assert_array_equal(joined.hard_gate_accepted, joined.states_accepted)
    original = pd.read_csv(directory / "temporal_delta_detail.csv")
    for method in ("raw_window", "kmeans_hard", "oracle_label"):
        new = detail[(detail.axis == "core") & (detail.method == method)].set_index("seed").sort_index()
        old = original[original.method == method].set_index("seed").sort_index()
        np.testing.assert_allclose(new.temporal_delta_auroc, old.temporal_delta_auroc, atol=1e-15, rtol=0)
        np.testing.assert_allclose(new.pooled_temporal_auroc, old.pooled_temporal_auroc, atol=1e-15, rtol=0)
    aggregation = pd.read_csv(root / "results/aggregation/aggregation_controls.csv")
    legacy = aggregation[(aggregation.setting == "switching_core") & (aggregation.method == "kmeans_soft") & (aggregation.eta == 0)].set_index("seed").sort_index()
    new = detail[(detail.axis == "core") & (detail.method == "gmm_weighted_argmax")].set_index("seed").sort_index()
    np.testing.assert_allclose(new.pooled_temporal_auroc, legacy.temporal_graph_auroc, atol=1e-15, rtol=0)
    return {"detail_rows": len(detail), "summary_rows": len(summary), "distinct_fits": len(detail[["config_id", "seed"]].drop_duplicates()),
            "historical_hard_oracle_rows_matched": 1400, "core_reference_rows_matched": 60,
            "legacy_gmm_core_rows_matched": 20, "no_target_configuration_seeds": int((~detail[detail.method == "raw_window"].evaluable).sum())}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
