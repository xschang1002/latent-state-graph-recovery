"""Exact-partition, seed-unit and factorisation regression tests."""
from __future__ import annotations

from math import comb
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grouping_controls as controls
from reproduce_paper import verify_bundle


@pytest.mark.parametrize("small", [1, 2, 3])
def test_odd_window_partitions_match_group_sizes_without_duplicates(small):
    labels = np.array([0] * small + [1] * (7 - small))
    values = list(controls.matched_partitions(labels))
    assert len(values) == comb(7, small)
    assert len({tuple(row) for row in values}) == len(values)
    assert all(row.sum() == small for row in values)
    assert any(np.array_equal(row, labels) or np.array_equal(1 - row, labels) for row in values)


def test_equal_halves_are_unlabelled_complement_pairs():
    assert len(list(controls.matched_partitions(np.array([0, 0, 1, 1])))) == 3
    with pytest.raises(ValueError, match="two nonempty"):
        list(controls.matched_partitions(np.zeros(7)))


def _truth():
    support = np.zeros((2, 1, 2, 2))
    support[0, 0, 0, 1] = 1; support[1, 0, 1, 0] = 1
    return support


def test_partition_aucs_must_be_averaged_before_seed_inference():
    support = _truth(); regimes = np.array([0, 1, 1]); bounds = [(i, i + 1) for i in range(3)]
    windows = support[regimes]; values, predictions = [], []
    for labels in controls.matched_partitions(regimes):
        maps = controls._hard_state_scores(windows, labels)
        values.append(controls.temporal_auc(maps, labels, bounds, regimes, support))
        predictions.append(maps[labels])
    assert np.mean(values) == pytest.approx(5 / 6)
    averaged_prediction_auc = controls.temporal_auc(np.mean(predictions, axis=0), np.arange(3), bounds, regimes, support)
    assert averaged_prediction_auc == 1 and averaged_prediction_auc != np.mean(values)


def test_factorisation_is_exact_and_invariant_to_renaming_both_predicted_states():
    support = _truth(); regimes = np.array([0, 0, 0, 1, 1, 1])
    bounds = [(i, i + 1) for i in range(6)]; labels = np.array([0, 0, 1, 1, 1, 0])
    first = controls.state_factors(support, labels, bounds, regimes, support)
    second = controls.state_factors(support[::-1], 1 - labels, bounds, regimes, support)
    assert first["identity_gap"] < 1e-15
    assert second["J_times_C"] == pytest.approx(first["J_times_C"])
    assert first["hard_temporal_auroc"] == pytest.approx(2 / 3)


def test_bundle_verification_detects_corruption_and_path_escape(tmp_path):
    payload = tmp_path / "data.csv"; payload.write_text("a,b\n1,2\n")
    manifest = {"sha256": {"data.csv": hashlib.sha256(payload.read_bytes()).hexdigest()}}
    (tmp_path / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest))
    assert verify_bundle(tmp_path) == 1
    payload.write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_bundle(tmp_path)
    manifest["sha256"] = {"../escape.txt": "x"}
    (tmp_path / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid or missing"):
        verify_bundle(tmp_path)


def test_saved_controls_replay_from_retained_outputs():
    sys.path.insert(0, str(ROOT / "tools"))
    from verify_grouping_controls import verify
    result = verify()
    assert result["new_seeds"] == 0 and result["factor_rows"] == 60
