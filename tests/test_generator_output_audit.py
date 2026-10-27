"""Tests of descriptive diagnostics without generating or fitting any series."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import audit_generator_outputs as audit


def test_endpoint_rms_is_across_nodes_and_distinct_endpoint_blocks():
    x = np.vstack([np.ones((4, 2)), np.full((4, 2), 100.0), np.full((4, 2), 3.0)])
    result = audit.trajectory_metrics(x, endpoint_size=4)
    assert result["early_rms"] == 1
    assert result["late_rms"] == 3
    assert result["late_to_early_rms_ratio"] == 3
    assert result["maximum_absolute_observed_value"] == 100
    assert result["all_observed_entries_finite"]
    assert result["observed_timesteps"] == 12
    assert result["n_nodes"] == 2


def test_rms_handles_large_finite_values_without_overflow():
    assert audit.rms(np.full((3, 2), 1e200)) == 1e200
    assert audit.rms(np.zeros((2, 2))) == 0


def test_undefined_or_nonfinite_endpoint_ratios_are_not_zero_or_a_pass_flag():
    x = np.vstack([np.zeros((4, 2)), np.ones((4, 2))])
    result = audit.trajectory_metrics(x, 4)
    assert np.isnan(result["late_to_early_rms_ratio"])
    assert not result["rms_ratio_evaluable"]
    x[-1, -1] = np.inf
    result = audit.trajectory_metrics(x, 4)
    assert not result["all_observed_entries_finite"]
    assert result["nonfinite_observed_entries"] == 1
    assert np.isnan(result["late_rms"])
    assert audit.json_safe(result)["maximum_absolute_observed_value"] is None


@pytest.mark.parametrize("shape,endpoint", [((3, 2), 2), ((8,), 2), ((8, 2), 0)])
def test_endpoint_blocks_must_exist_and_not_overlap(shape, endpoint):
    with pytest.raises(ValueError, match="nonoverlapping"):
        audit.trajectory_metrics(np.ones(shape), endpoint)


def retained_mechanism():
    coefficient = np.zeros((2, 1, 2, 2))
    coefficient[0, 0, 0, 1] = 0.2
    coefficient[1, 0, 0, 1] = 0.3
    coefficient[1, 0, 1, 1] = 0.4
    return {"coefficients": coefficient, "support": (coefficient != 0).astype(int),
            "spectral_radius": np.array([audit.spectral_radius(state) for state in coefficient]),
            "initial_values": np.zeros((1, 2)), "innovations": np.zeros((100, 2))}


def test_mechanism_diagnostics_keep_self_asymmetry_and_only_flag_compatible_fallbacks():
    result = audit.mechanism_metrics(retained_mechanism(), "fixture", 7)
    assert result["state0_self_edges"] == 0
    assert result["state1_self_edges"] == 1
    assert result["realised_retained_fraction"] == 1
    assert result["base_empty_fallback_compatible"]  # A one-edge sample could arise without fallback too.
    assert not result["state1_empty_fallback_compatible"]
    assert result["state0_companion_radius"] == 0
    assert result["state1_companion_radius"] == 0.4


def test_corrupted_retained_support_or_radius_cannot_silently_pass():
    inputs = retained_mechanism()
    inputs["support"][0, 0, 0, 0] = 1
    with pytest.raises(AssertionError):
        audit.mechanism_metrics(inputs, "fixture", 7)
    inputs = retained_mechanism()
    inputs["spectral_radius"][1] = 0.5
    with pytest.raises(AssertionError):
        audit.mechanism_metrics(inputs, "fixture", 7)


def test_frozen_output_guard_precedes_any_data_reads(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Data must not be read before output guard")
    monkeypatch.setattr(audit, "frozen_hashes", forbidden)
    with pytest.raises(ValueError, match="outside both frozen"):
        audit.audit(tmp_path, tmp_path / "results/information_sciences/duration_confirmation/diagnostics")


def test_declared_hash_validation_is_readonly_and_rejects_output_changes(tmp_path):
    root = tmp_path
    source = root / "source.py"
    source.write_text("retained source\n", encoding="utf-8")
    config = root / "configs/schedule_robustness.yaml"
    config.parent.mkdir()
    config.write_text("fixture: true\n", encoding="utf-8")
    for folder, protocol_name, output_name, protocol_key in (
        ("duration_confirmation", "protocol.json", "COMPLETE.json", "protocol_sha256"),
        ("schedule_robustness", "schedule_robustness_protocol.json", "schedule_robustness_outputs.json", "source_protocol_sha256"),
    ):
        directory = root / "results/information_sciences" / folder
        directory.mkdir(parents=True)
        payload = directory / "data.csv"
        payload.write_text("fixture\n", encoding="utf-8")
        protocol = {"source_sha256": {source.name: audit.sha(source)}, "configuration_sha256": audit.sha(config)}
        protocol_path = directory / protocol_name
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        (directory / output_name).write_text(json.dumps({protocol_key: audit.sha(protocol_path), "sha256": {payload.name: audit.sha(payload)}}), encoding="utf-8")
    before = audit.frozen_hashes(root)
    sources_before = audit.frozen_source_hashes(root)
    audit.check_declared_sources(root)
    assert audit.frozen_hashes(root) == before
    assert audit.frozen_source_hashes(root) == sources_before
    (root / "results/information_sciences/duration_confirmation/data.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="experimental output changed"):
        audit.check_declared_sources(root)
