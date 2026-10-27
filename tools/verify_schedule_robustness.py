"""Read-only replay of study B from retained exogenous inputs and emissions.

No VAR series or local scorer is refitted. Full schedules, state truth, scores,
state maps, emissions, gate, factors, metrics and secondary summaries are checked.
The output directory is hashed before and after; only a JSON report is printed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aggregation_study_common as common
import run_schedule_robustness as study
from stcg.evaluation import nonself_mask, window_purity, window_regime_labels
from stcg.mechanism import spectral_radius
from stcg.self_supervised import STCGStateConfig


def close(actual, expected, label: str) -> None:
    if expected is None or (isinstance(expected, (float, np.floating)) and np.isnan(expected)):
        if actual is not None:
            assert np.isnan(actual), label
    elif isinstance(expected, (bool, str, list, dict, int, np.integer)):
        assert actual == expected, (label, actual, expected)
    else:
        assert actual is not None, label
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0, err_msg=label)


def directory_hashes(output: Path) -> dict:
    return {path.relative_to(output).as_posix(): study.sha256(path)
            for path in sorted(output.rglob("*")) if path.is_file()}


def load_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def verify(output: Path = study.OUT, root: Path = ROOT) -> dict:
    before = directory_hashes(output)
    protocol = json.loads((output / "schedule_robustness_protocol.json").read_text(encoding="utf-8"))
    config = protocol["configuration"]
    for name, expected in protocol["source_sha256"].items():
        assert study.sha256(root / name) == expected, name
    assert study.sha256(root / "configs/schedule_robustness.yaml") == protocol["configuration_sha256"]
    manifest = json.loads((output / "schedule_robustness_outputs.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        assert before[name] == expected, name
    assert manifest["source_protocol_sha256"] == before["schedule_robustness_protocol.json"]
    records = []
    n_metrics, n_failed, n_windows, n_inputs = 0, 0, 0, 0
    rho_max, identity_max = 0.0, 0.0
    with threadpool_limits(limits=1):
        for seed in protocol["repeats"]:
            inputs_path = output / "inputs" / f"repeat_{seed}.npz"
            inputs = load_arrays(inputs_path)
            metadata = json.loads(inputs_path.with_suffix(".json").read_text(encoding="utf-8"))
            assert metadata["sha256"] == study.sha256(inputs_path)
            assert metadata["array_sha256"] == study.array_fingerprints(inputs)
            assert metadata["input_seed"] == str(study.domain_seed(seed, "base_inputs"))
            n_inputs += 1
            radii = np.array([spectral_radius(coefficients) for coefficients in inputs["coefficients"]])
            np.testing.assert_allclose(radii, inputs["spectral_radius"], atol=1e-12, rtol=0)
            rho_max = max(rho_max, float(radii.max()))
            for duration in config["durations"]:
                for kind in config["schedules"]:
                    cell = f"D{duration}_{kind}"
                    path = output / "parts" / f"repeat_{seed}_{cell}.json"
                    record = json.loads(path.read_text(encoding="utf-8"))
                    arrays = load_arrays(path.with_suffix(".npz"))
                    assert record["arrays_sha256"] == study.sha256(path.with_suffix(".npz"))
                    assert record["input_array_sha256"] == metadata["array_sha256"]
                    for name, other in (("coefficients", "coefficients"), ("state_support", "support")):
                        np.testing.assert_array_equal(arrays[name], inputs[other])
                    rng = np.random.default_rng(study.domain_seed(seed, "schedule", duration, kind))
                    full, definition = study.make_schedule(config["mechanism"]["timesteps"] + config["mechanism"]["burn_in"],
                                                           duration, kind, rng)
                    np.testing.assert_array_equal(arrays["regimes_full"], full)
                    np.testing.assert_array_equal(arrays["regimes"], full[config["mechanism"]["burn_in"]:])
                    assert definition == record["schedule_definition"]
                    np.testing.assert_array_equal(arrays["bounds"], study._bounds(config))
                    diagnostic, windows = study.describe_schedule(full, arrays["bounds"], config["mechanism"]["burn_in"])
                    for name, expected in diagnostic.items():
                        close(record["diagnostic"][name], expected, f"{cell}/{seed}/{name}")
                    identity = record["identity"]
                    assert record["windows"] == [{**identity, **row} for row in windows]
                    n_windows += len(windows)
                    if record["diagnostic"]["status"] != "ok":
                        n_failed += 1
                        assert all(row["fit_status"] == "failed" and row["temporal_delta_auroc"] is None
                                   for row in record["methods"])
                        records.append(record)
                        continue
                    assert np.isfinite(arrays["observed_series"]).all()
                    bounds = arrays["bounds"]
                    dataset = SimpleNamespace(regimes=arrays["regimes"], state_support=arrays["state_support"],
                                              union_support=arrays["state_support"].max(axis=0))
                    truth = window_regime_labels(bounds, dataset.regimes)
                    purity, _ = window_purity(bounds, dataset.regimes)
                    np.testing.assert_array_equal(arrays["true_window_labels"], truth)
                    np.testing.assert_allclose(arrays["window_purity"], purity, atol=0, rtol=0)
                    maps, forced = common.maps_and_emission(arrays["window_scores"], arrays["candidate_labels"])
                    oracle_maps, oracle = common.maps_and_emission(arrays["window_scores"], truth)
                    np.testing.assert_allclose(arrays["candidate_maps"], maps, atol=0, rtol=0)
                    np.testing.assert_allclose(arrays["oracle_maps"], oracle_maps, atol=0, rtol=0)
                    mask = nonself_mask(arrays["window_scores"].shape[1:])
                    features = arrays["window_scores"][:, mask]
                    centered = features - features.mean(axis=0, keepdims=True)
                    scaled = centered / (centered.std(axis=0, keepdims=True) + 1e-8)
                    labels = arrays["candidate_labels"]
                    _, fit_features = common.maps_and_emission(scaled, labels)
                    inertia = float(np.square(scaled - fit_features).sum())
                    improvement = common._state_improvement(scaled, inertia)
                    coherence = common._temporal_coherence(labels)
                    gate_config = STCGStateConfig()
                    accepted = bool(len(np.unique(labels)) == 2 and
                        (improvement >= gate_config.min_state_improvement or
                         (improvement >= gate_config.min_temporal_state_improvement and
                          coherence >= gate_config.min_temporal_coherence and
                          common._min_cluster_fraction(labels) >= gate_config.min_temporal_state_fraction)))
                    close(record["diagnostic"]["hard_gate_accepted"], accepted, "gate")
                    close(record["diagnostic"]["hard_gate_improvement"], improvement, "gate improvement")
                    close(record["diagnostic"]["hard_gate_coherence"], coherence, "gate coherence")
                    predictions = {"raw_window": arrays["window_scores"], "forced_hard": forced,
                                   "native_hard": forced if accepted else np.repeat(arrays["window_scores"].max(axis=0)[None], len(bounds), axis=0),
                                   "oracle_label": oracle,
                                   "static_mean": np.repeat(arrays["window_scores"].mean(axis=0)[None], len(bounds), axis=0)}
                    factors = common.hard_factors(maps, labels, bounds, dataset)
                    for name in ("J", "C_edge", "J_times_C"):
                        close(record["diagnostic"][name], factors[name], name)
                    if np.isfinite(factors["identity_gap"]):
                        identity_max = max(identity_max, factors["identity_gap"])
                    replay = deepcopy(record)
                    for row in replay["methods"]:
                        method = row["method"]
                        np.testing.assert_allclose(arrays[f"emission_{method}"], predictions[method], atol=0, rtol=0)
                        metrics = common.evaluate_predictions(predictions[method], bounds, dataset)
                        factor = factors if method == "forced_hard" or (method == "native_hard" and accepted) else (
                            common.hard_factors(oracle_maps, truth, bounds, dataset) if method == "oracle_label" else
                            {"J": np.nan, "C_edge": np.nan, "J_times_C": np.nan, "identity_gap": np.nan})
                        for name, expected in {**metrics, **factor}.items():
                            close(row[name], expected, f"{cell}/{seed}/{method}/{name}")
                            row[name] = expected
                        n_metrics += 1
                    records.append(replay)
        tables = study.summarise(records, config)
        for name, expected in tables.items():
            # CSV round-trip normalises list diagnostics and missing-value dtype.
            replay = pd.read_csv(io.StringIO(expected.to_csv(index=False)), float_precision="round_trip")
            actual = pd.read_csv(output / name, float_precision="round_trip")
            pd.testing.assert_frame_equal(actual, replay, check_exact=False, atol=1e-12, rtol=0)
    after = directory_hashes(output)
    assert before == after, "Read-only QA changed study files"
    return {"status": "passed", "planned_cases": 300, "verified_cases": len(records), "independent_input_repeats": n_inputs,
            "failed_cases_retained": n_failed, "method_metric_rows_recomputed": n_metrics,
            "window_diagnostics_recomputed": n_windows, "summary_csvs_recomputed": len(tables),
            "secondary_family_sizes": [6, 3], "maximum_identity_gap": identity_max,
            "maximum_actual_spectral_radius": rho_max, "all_study_files_unchanged": len(before),
            "protocol_sha256": before["schedule_robustness_protocol.json"],
            "fit_or_resimulation_performed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=study.OUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
