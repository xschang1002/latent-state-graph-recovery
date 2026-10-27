"""Read-only replay of the frozen duration confirmation from retained outputs.

Verifies every checksum, paired mechanism/noise reuse, schedule/series replay,
stored emissions, exact matched partitions, repeat-level contrasts and tests.
It never fits a scorer or changes an output file.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from aggregation_study_common import evaluate_predictions, fixed_schedule, maps_and_emission, simulate_schedule
from run_duration_confirmation import canonical_bytes, file_sha, read_repeat, summarise, validate_protocol
from stcg.mechanism import MechanismConfig


def _same_number(actual, expected, name: str) -> None:
    if actual is None:
        actual = np.nan
    if expected is None:
        expected = np.nan
    if not np.isclose(float(actual), float(expected), rtol=0, atol=2e-12, equal_nan=True):
        raise AssertionError(f"Cached value mismatch: {name}: {actual} != {expected}")


def verify(output_dir: Path) -> dict:
    output_dir = Path(output_dir).resolve()
    complete = json.loads((output_dir / "COMPLETE.json").read_text())
    for relative, expected in complete["sha256"].items():
        path = (output_dir / relative).resolve()
        if not path.is_relative_to(output_dir) or not path.is_file() or file_sha(path) != expected:
            raise ValueError(f"Output checksum mismatch or invalid path: {relative}")
    protocol = json.loads((output_dir / "protocol.json").read_text())
    validate_protocol(protocol)
    protocol_sha = hashlib.sha256(canonical_bytes(protocol)).hexdigest()
    if protocol_sha != complete["protocol_sha256"]:
        raise ValueError("Completion record and frozen protocol disagree")
    records = output_dir / "records"
    collected = []
    replayed_fits = failures = partitions = 0
    for seed in protocol["repeat_seeds"]:
        rows = read_repeat(records, seed, protocol_sha)
        collected.extend(rows)
        with np.load(records / f"repeat_{seed}_inputs.npz", allow_pickle=False) as saved:
            inputs = {k: saved[k] for k in saved.files}
        base = MechanismConfig(**protocol["base_config"])
        for duration in protocol["durations"]:
            data_path = records / f"repeat_{seed}_D{duration}_data.npz"
            expected_schedule = fixed_schedule(base.timesteps + base.burn_in, duration)
            with np.load(data_path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["regimes_full"], expected_schedule)
                config = replace(base, switch_period=duration, seed=seed)
                if "x" not in saved.files:
                    if any(r["status"] != "failed" for r in rows if r["D"] == duration):
                        raise AssertionError("Missing series must retain explicit simulation failures")
                    failures += len(protocol["scorers"])
                    continue
                dataset = simulate_schedule(inputs, expected_schedule, config)
                np.testing.assert_array_equal(saved["regimes"], dataset.regimes)
                np.testing.assert_array_equal(saved["x"], dataset.x)
            for scorer in protocol["scorers"]:
                fit_base = records / f"repeat_{seed}_D{duration}_{scorer}"
                fit_info = json.loads(fit_base.with_suffix(".json").read_text())
                fit_rows = [r for r in rows if r["D"] == duration and r["scorer"] == scorer]
                if fit_info["status"] == "failed":
                    if len(fit_rows) != 6 or any(r["status"] != "failed" for r in fit_rows):
                        raise AssertionError("A failed fit must retain all six failed method rows")
                    failures += 1
                    continue
                with np.load(fit_base.with_suffix(".npz"), allow_pickle=False) as saved:
                    scores = saved["window_scores"]; bounds = saved["bounds"]
                    candidate = saved["candidate_labels"]
                    _, forced = maps_and_emission(scores, candidate)
                    np.testing.assert_array_equal(saved["emission_forced_hard"], forced)
                    native = forced if fit_info["diagnostic"]["hard_gate_accepted"] else np.repeat(
                        scores.max(axis=0)[None], len(bounds), axis=0)
                    np.testing.assert_array_equal(saved["emission_native_hard"], native)
                    for row in fit_rows:
                        if row["method"] == "matched_random":
                            if "matched_random_partitions" not in saved.files:
                                if row["undefined_reason"] != "candidate_not_two_groups":
                                    raise AssertionError("Missing random control needs explicit reason")
                                continue
                            random_values = []
                            for assignment in saved["matched_random_partitions"]:
                                _, emitted = maps_and_emission(scores, assignment)
                                random_values.append(evaluate_predictions(emitted, bounds, dataset))
                            partitions += len(random_values)
                            if len(random_values) != row["n_partitions"]:
                                raise AssertionError("Matched enumeration count mismatch")
                            for metric in ("temporal_delta_auroc", "pooled_temporal_auroc", "union_auroc"):
                                values = np.asarray([v[metric] for v in random_values])
                                np.testing.assert_allclose(saved[f"matched_random_{metric}"], values, rtol=0, atol=2e-12,
                                                           equal_nan=True)
                                _same_number(row[metric], values.mean(), metric)
                        else:
                            metric_values = evaluate_predictions(saved[f"emission_{row['method']}"], bounds, dataset)
                            for metric in ("temporal_delta_auroc", "pooled_temporal_auroc", "union_auroc",
                                           "temporal_delta_edges", "temporal_delta_edges_total"):
                                _same_number(row[metric], metric_values[metric], metric)
                    if scorer == "ols":
                        if any(w["rank"] != w["n_features"] for w in fit_info["diagnostic"]["numerical"]):
                            raise AssertionError("A successful OLS fit must retain full design ranks")
                replayed_fits += 1
    frame = pd.DataFrame(collected)
    frame = frame[sorted(frame.columns)]
    stored = pd.read_csv(output_dir / "detail.csv")
    expected_detail = frame.copy()
    for column in expected_detail:
        if expected_detail[column].dtype == object:
            expected_detail[column] = expected_detail[column].map(
                lambda value: np.nan if value is None or value == "" else value)
    pd.testing.assert_frame_equal(stored, expected_detail, check_dtype=False, check_exact=False, atol=2e-12, rtol=0)
    rebuilt_summary, rebuilt_contrasts = summarise(frame, protocol)
    for name, expected in (("summary.csv", rebuilt_summary), ("contrasts.csv", rebuilt_contrasts)):
        observed = pd.read_csv(output_dir / name)
        pd.testing.assert_frame_equal(observed, expected, check_dtype=False, check_exact=False, atol=2e-12, rtol=0)
    if len(frame) != len(protocol["repeat_seeds"]) * 2 * 2 * 6:
        raise AssertionError("Planned repeat/condition/scorer/method rows were lost")
    return {"repeats": len(protocol["repeat_seeds"]), "replayed_successful_fits": replayed_fits,
            "retained_failed_fits": failures, "exact_partition_scores_replayed": partitions,
            "primary_contrasts": 4, "scorer_refits": 0, "read_only": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/information_sciences/duration_confirmation")
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
