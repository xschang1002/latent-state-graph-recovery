import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("revision_real", ROOT / "scripts/run_revision_real_validation.py")
revision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(revision)


def test_plan_uses_only_length_and_all_test_blocks_are_disjoint():
    plan = revision.interval_plan(132300)
    forward = [p for p in plan if p["protocol"] == "forward"]
    segments = [p for p in plan if p["protocol"] == "segment"]
    assert len(forward) == len(segments) == 4
    assert [p["train_end"] for p in forward] == [26460, 52920, 79380, 105840]
    assert forward[-1]["end"] == 132300
    for left, right in zip(forward, forward[1:]):
        assert left["end"] == right["start"]
    for left, right in zip(segments, segments[1:]):
        assert left["end"] <= right["start"]
    for p in forward:
        train = revision.complete_windows(0, p["train_end"])
        test = revision.complete_windows(p["start"], p["end"])
        assert train[-1][1] <= p["start"] <= test[0][0]
        assert all(b - a == 300 for a, b in train + test)


def test_changing_test_features_does_not_change_train_k_or_train_states():
    random = np.random.default_rng(81)
    features = np.concatenate([random.normal(-3, 0.2, (15, 4)), random.normal(3, 0.2, (15, 4)), random.normal(size=(10, 4))])
    train = np.arange(30)
    modified = features.copy()
    modified[30:] = random.normal(1e8, 1e6, (10, 4))
    for choose_k in [False, True]:
        states, metadata = revision.train_only_states(features, train, 2, choose_k)
        changed_states, changed_metadata = revision.train_only_states(modified, train, 2, choose_k)
        np.testing.assert_array_equal(states[train], changed_states[train])
        for key in ["supplied_k", "state_improvement", "temporal_coherence", "inferred_train_k"]:
            assert metadata[key] == changed_metadata[key]


def test_test_labels_do_not_change_any_forward_prediction():
    random = np.random.default_rng(32)
    features = random.normal(size=(40, 6))
    train, test = np.arange(30), np.arange(30, 40)
    labels = ["A"] * 15 + ["B"] * 15 + ["A"] * 10
    modified = labels[:30] + ["previously unseen test class"] * 10
    predictions, metadata = revision.predict_from_training(features, features, labels, train, test, 0)
    changed, changed_metadata = revision.predict_from_training(features, features, modified, train, test, 0)
    assert predictions == changed
    for method in metadata:
        assert metadata[method]["state_mapping"] == changed_metadata[method]["state_mapping"]


def test_worker_really_terminates_at_budget_and_does_not_reuse_stale_output(tmp_path):
    output = tmp_path / "worker.json"
    output.write_text('{"status": "ok"}', encoding="utf-8")
    command = [sys.executable, "-c", "import time; time.sleep(10)"]
    start = time.perf_counter()
    result = revision.run_worker_with_budget(command, 0.1, output, tmp_path / "worker.log")
    assert result["status"] == "timeout"
    assert not output.exists()
    assert time.perf_counter() - start < 3.0


def test_failed_and_single_label_segments_are_retained_without_fake_metrics():
    labels = pd.DataFrame({"cooler_condition": [3] * 1500, "valve_condition": [100] * 1500})
    interval = dict(interval_id=0, start=0, end=1500)
    failed = revision.score_segment("HydraulicSystems", interval, labels, "Regime-PCMCI K2", dict(status="timeout", sequence=[]))
    assert len(failed) == 2
    assert all(row["status"] == "timeout" and "sample_ari" not in row for row in failed)
    assert all(row["error_free_annealings"] is None for row in failed)
    collapsed = revision.score_segment("HydraulicSystems", interval, labels, "STCG Fixed K2", dict(status="ok", sequence=[0] * 1500))
    assert all(row["true_window_k"] == 1 and row["state_collapse"] == 1 for row in collapsed)


def test_rpcmci_full_sequence_offset_is_not_shifted():
    labels = pd.DataFrame({"cooler_condition": [3] * 750 + [100] * 750, "valve_condition": [100] * 1500})
    interval = dict(interval_id=0, start=0, end=1500)
    result = dict(status="ok", sequence=[1] * 3 + [0] * 747 + [1] * 750, offset=3)
    rows = revision.score_segment("HydraulicSystems", interval, labels, "Regime-PCMCI K2", result)
    assert rows[0]["sample_ari"] == 1.0
    assert rows[0]["scored_samples"] == 1497
