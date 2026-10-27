"""Prespecified schedule robustness study, with an explicit freeze before execution.

The default prints the proposed protocol without generating data or fitting.
Use --freeze once the protocol is approved, then --run to execute it. Replaying
--aggregate reads retained case records and never fits an estimator. Study B is
secondary evidence, separate from the independent primary confirmation family.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/schedule_robustness.yaml"
OUT = ROOT / "results/information_sciences/schedule_robustness"
sys.path.insert(0, str(ROOT / "src"))
METHODS = ("raw_window", "native_hard", "forced_hard", "oracle_label", "static_mean")
KINDS = ("fixed_phase", "random_phase", "irregular")
RNG_NAMESPACE = "stcg_information_sciences_schedule_v1"
SOURCES = ("configs/schedule_robustness.yaml", "scripts/run_schedule_robustness.py",
           "scripts/aggregation_study_common.py", "src/stcg/mechanism.py",
           "src/stcg/self_supervised.py", "src/stcg/baselines.py", "src/stcg/evaluation.py",
           "scripts/run_aggregation_controls.py", "scripts/run_state_decomposition.py", "requirements-revision.txt")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def domain_seed(seed: int, domain: str, *parts: Any, bits: int = 128) -> int:
    """Stable domain separation, independent of Python hash/random global state."""
    if bits not in (32, 128):
        raise ValueError("Only 32-bit clustering or 128-bit stream seeds are supported")
    token = json.dumps([RNG_NAMESPACE, int(seed), domain, *parts], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:bits // 8], "little")


def make_schedule(total: int, duration: int, kind: str, rng: Any) -> tuple[np.ndarray, dict]:
    """Full-time labels, before the common burn-in; integer endpoints inclusive."""
    if total < 1 or duration < 1:
        raise ValueError("total and duration must be positive integers")
    if kind not in KINDS:
        raise ValueError(f"Unknown schedule: {kind}")
    phase = int(rng.integers(0, 2 * duration)) if kind == "random_phase" else 0
    segments = []
    if kind != "irregular":
        labels = ((np.arange(total, dtype=np.int64) + phase) // duration % 2).astype(np.int8)
        starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
        for index, start in enumerate(starts):
            stop = int(starts[index + 1]) if index + 1 < len(starts) else total
            segments.append({"start_full": int(start), "stop_full": stop, "state": int(labels[start]),
                             "nominal_duration": duration,
                             "left_truncated_at_origin": bool(index == 0 and phase % duration),
                             "right_truncated_at_end": bool(stop == total and (total + phase) % duration)})
    else:
        low, high = int(np.ceil(0.75 * duration)), int(np.floor(1.25 * duration))
        labels = np.empty(total, dtype=np.int8)
        position, state = 0, 0
        while position < total:
            length = int(rng.integers(low, high + 1))
            stop = min(total, position + length)
            labels[position:stop] = state
            segments.append({"start_full": position, "stop_full": stop, "state": state,
                             "nominal_duration": length, "left_truncated_at_origin": False,
                             "right_truncated_at_end": stop < position + length})
            position, state = stop, 1 - state
    return labels, {"kind": kind, "duration_mean": duration, "phase_full": phase,
                    "segments_full": segments,
                    "irregular_low": int(np.ceil(0.75 * duration)) if kind == "irregular" else None,
                    "irregular_high": int(np.floor(1.25 * duration)) if kind == "irregular" else None}


def describe_schedule(regimes_full: np.ndarray, bounds: np.ndarray, burn_in: int) -> tuple[dict, list[dict]]:
    regimes = np.asarray(regimes_full)[burn_in:]
    bounds = np.asarray(bounds, dtype=int)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or not len(bounds):
        raise ValueError("Nonempty half-open window bounds are required")
    if np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] > len(regimes)) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Window bounds must lie inside the observed sequence")
    centres = (bounds[:, 0] + bounds[:, 1] - 1) // 2
    labels = regimes[centres]
    windows = []
    purities = []
    for index, ((start, stop), centre, label) in enumerate(zip(bounds, centres, labels)):
        states, counts = np.unique(regimes[start:stop], return_counts=True)
        best = int(np.argmax(counts))
        purity = float(counts[best] / (stop - start))
        purities.append(purity)
        windows.append({"window": index, "start": int(start), "stop": int(stop), "centre": int(centre),
                        "centre_label": int(label), "majority_label": int(states[best]), "purity": purity})
    positions = np.flatnonzero(regimes[1:] != regimes[:-1]) + 1
    centre_positions = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    first_full = burn_in
    while first_full and regimes_full[first_full - 1] == regimes_full[burn_in]:
        first_full -= 1
    last_full = len(regimes_full) - 1
    while last_full > burn_in and regimes_full[last_full - 1] == regimes_full[-1]:
        last_full -= 1
    diagnostic = {"n_actual_switches": int(len(positions)), "actual_switch_positions": positions.tolist(),
                  "n_centre_switches": int(len(centre_positions)), "centre_switch_window_indices": centre_positions.tolist(),
                  "centre_label_sequence": labels.tolist(), "n_windows": len(bounds),
                  "centre_state0_windows": int((labels == 0).sum()), "centre_state1_windows": int((labels == 1).sum()),
                  "observed_state0_samples": int((regimes == 0).sum()), "observed_state1_samples": int((regimes == 1).sum()),
                  "first_observed_segment_left_truncated": bool(first_full < burn_in),
                  "first_observed_segment_start_full": first_full, "last_observed_segment_start_full": last_full,
                  "window_purity_mean": float(np.mean(purities)), "window_purity_min": float(np.min(purities)),
                  "window_purity_q25": float(np.quantile(purities, 0.25)), "window_purity_median": float(np.median(purities)),
                  "window_purity_q75": float(np.quantile(purities, 0.75)), "window_purity_max": float(np.max(purities))}
    return diagnostic, windows


def load_config(path: Path = CONFIG) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("threads") != 1:
        raise ValueError("Study B fixes BLAS/OpenMP thread pools to one thread")
    if config["durations"] != [300, 600] or tuple(config["schedules"]) != KINDS:
        raise ValueError("Study B has six fixed duration-by-schedule cells")
    if config["seed_start"] != 60000 or config["seed_count"] != 50:
        raise ValueError("Study B uses the fifty frozen repeats 60000--60049")
    if config["model"]["scorer"] != "lasso":
        raise ValueError("Study B uses the prespecified Lasso scorer only")
    if (config["mechanism"]["timesteps"], config["mechanism"]["burn_in"],
        config["window"]["size"], config["window"]["stride"]) != (3600, 100, 300, 150):
        raise ValueError("Study B requires T3600, burn-in100, h300 and stride150")
    if config["statistics"] != {"bootstrap_draws": 10000, "bootstrap_seed": 20261024,
                                "minimum_ci_n": 5, "alpha": 0.05}:
        raise ValueError("Study B uses the frozen pointwise bootstrap and inference settings")
    return config


def protocol(config: dict, config_path: Path = CONFIG) -> dict:
    files = [path for path in SOURCES if path != "configs/schedule_robustness.yaml"]
    sources = {path: sha256(ROOT / path) for path in files}
    return {"study": config["study"], "role": "prespecified secondary schedule robustness; separate from A confirmatory family",
            "configuration": config, "configuration_sha256": sha256(config_path), "source_sha256": sources,
            "repeats": list(range(config["seed_start"], config["seed_start"] + config["seed_count"])),
            "paired_unit": "independent mechanism/initialisation/innovation repeat, shared across all six cells",
            "rng": {"namespace": RNG_NAMESPACE, "seed_derivation": "little-endian first128 SHA256 bits of canonical compact JSON [namespace, repeat_seed, domain, *parts]; clustering first32 bits",
                    "domains": {"inputs": "base_inputs", "cluster": "clustering", "schedule": "schedule,D,kind"},
                    "base_helper": "SeedSequence([input_seed,domain]): domain0 coefficients, domain1 initial values, domain2 innovations",
                    "no_optional_stopping": True},
            "schedule": {"fixed_phase": "full time origin0, state=floor(t/D)%2, then discard100 burn-in",
                         "random_phase": "independent q~integerUniform[0,2D-1], state=floor((t+q)/D)%2 before burn-in",
                         "irregular": "full origin0 state0, alternating states, independent inclusive integerUniform[ceil(.75D),floor(1.25D)] lengths before burn-in",
                         "censoring": "retain sampled/full-time segments and flag burn-in/endpoint truncation"},
            "methods": list(METHODS), "expected_cells": 6, "expected_fits": 300, "expected_windows_per_fit": 23,
            "aggregation": {"features": "nonself absolute Lasso coefficients, standardised per feature across windows with std+1e-8",
                            "candidate": "KMeans K2, n_init10; unweighted mean state maps",
                            "gate": "two nonempty groups and (improvement>=.12 or (improvement>=.04 and coherence>=.55 and minimum_fraction>=.10))",
                            "native_fallback": "same window scores collapsed by static maximum",
                            "self_lags": "included in regression; excluded from scored edges and clustering features"},
            "primary_estimand_within_B": "forced_hard minus raw_window changing-edge temporal AUROC",
            "inference": {"B_forced_family": "six two-sided paired-repeat one-sample t tests; Holm6",
                          "B_duration_difference_family": "three paired Gamma=Delta(600)-Delta(300), one per schedule; two-sided t, Holm3",
                          "native_gate_factors": "descriptive; no accepted-only confirmatory tests",
                          "intervals": "pointwise paired-repeat percentile bootstrap; not simultaneous intervals",
                          "missing": "retain failures and undefined targets; finite complete pairs only, no 0.5 imputation; fixed family size retained"},
            "stability": "preserve existing one-pass scalar stabilisation; report actual spectral radii, no rejection/resampling",
            "retained": "base coefficients/support/initial values/innovations; full and observed schedules/series; window scores/labels/maps/purity; actual and centre switches; J,C,JC and edge denominators"}


def ensure_protocol(output: Path, proposed: dict, freeze: bool = False) -> None:
    path = output / "schedule_robustness_protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != jsonable(proposed):
            raise RuntimeError("Frozen protocol/source differs; use a new output directory and obtain approval")
    elif freeze:
        output.mkdir(parents=True, exist_ok=True)
        write_json(path, proposed)
    else:
        raise RuntimeError("No frozen protocol; review and --freeze before --run")


def array_fingerprints(arrays: dict) -> dict:
    return {key: hashlib.sha256(json.dumps([str(np.asarray(value).dtype), list(np.shape(value))]).encode() +
                               np.ascontiguousarray(value).tobytes()).hexdigest() for key, value in arrays.items()}


def _backend():
    return importlib.import_module("aggregation_study_common")


def _bounds(config: dict) -> np.ndarray:
    window, stride = config["window"]["size"], config["window"]["stride"]
    return np.asarray([(start, start + window) for start in range(0, config["mechanism"]["timesteps"] - window + 1, stride)])


def run_repeat(config: dict, seed: int, output: Path, backend: Any = None) -> list[dict]:
    """Execute six cells with one retained base draw, or reuse checked records."""
    from stcg.mechanism import MechanismConfig
    backend = _backend() if backend is None else backend
    inputs_dir, parts_dir = output / "inputs", output / "parts"
    inputs_dir.mkdir(parents=True, exist_ok=True); parts_dir.mkdir(parents=True, exist_ok=True)
    setting = MechanismConfig(**config["mechanism"], switch_period=300, seed=seed)
    base_seed = domain_seed(seed, "base_inputs")
    cluster_seed = domain_seed(seed, "clustering", bits=32)
    input_npz, input_json = inputs_dir / f"repeat_{seed}.npz", inputs_dir / f"repeat_{seed}.json"
    if input_npz.exists():
        saved = json.loads(input_json.read_text(encoding="utf-8"))
        if saved["sha256"] != sha256(input_npz):
            raise RuntimeError("Base input checksum mismatch")
        with np.load(input_npz, allow_pickle=False) as retained:
            inputs = {key: retained[key] for key in retained.files}
    else:
        inputs = backend.draw_base_inputs(setting, base_seed)
        if any(np.asarray(value).dtype.hasobject for value in inputs.values()):
            raise ValueError("Retained inputs must contain numeric arrays only")
        np.savez_compressed(input_npz, **inputs)
        write_json(input_json, {"seed": seed, "input_seed": str(base_seed), "sha256": sha256(input_npz),
                                "array_sha256": array_fingerprints(inputs)})
    shared_hashes = array_fingerprints(inputs)
    records = []
    for duration in config["durations"]:
        for kind in config["schedules"]:
            cell = f"D{duration}_{kind}"
            part_npz, part_json = parts_dir / f"repeat_{seed}_{cell}.npz", parts_dir / f"repeat_{seed}_{cell}.json"
            if part_json.exists():
                record = json.loads(part_json.read_text(encoding="utf-8"))
                if record["arrays_sha256"] != sha256(part_npz) or record["input_array_sha256"] != shared_hashes:
                    raise RuntimeError("Retained case or paired input checksum mismatch")
                records.append(record)
                continue
            schedule_seed = domain_seed(seed, "schedule", duration, kind)
            full, schedule = make_schedule(setting.timesteps + setting.burn_in, duration, kind,
                                           np.random.default_rng(schedule_seed))
            cell_config = replace(setting, switch_period=duration)
            base = {"seed": seed, "duration_mean": duration, "schedule": kind, "cell": cell,
                    "input_seed": str(base_seed), "cluster_seed": cluster_seed, "schedule_seed": str(schedule_seed)}
            arrays = {"regimes_full": full, "regimes": full[setting.burn_in:], "state_support": inputs["support"],
                      "coefficients": inputs["coefficients"], "bounds": _bounds(config)}
            status, error = "ok", None
            simulation_completed = False
            try:
                dataset = backend.simulate_schedule(inputs, full, cell_config)
                simulation_completed = True
                arrays["observed_series"] = dataset.x
                methods, fitted, fitted_diagnostics = backend.fit_aggregation(dataset, scorer="lasso", cluster_seed=cluster_seed,
                    exact_random=False, window_size=config["window"]["size"], stride=config["window"]["stride"],
                    lasso_alpha=config["model"]["lasso_alpha"], n_init=config["model"]["n_init"])
                if {row["method"] for row in methods} != set(METHODS) or len(methods) != len(METHODS):
                    raise ValueError("Unexpected study-B aggregation methods")
                arrays.update(fitted)
            except Exception as exception:
                status, error = "failed", f"{type(exception).__name__}: {exception}"
                (parts_dir / f"repeat_{seed}_{cell}_failure.log").write_text(traceback.format_exc(), encoding="utf-8")
                methods = [{"method": method, "temporal_delta_auroc": np.nan, "status": "failed"} for method in METHODS]
                fitted_diagnostics = getattr(exception, "diagnostic", {})
            if array_fingerprints(inputs) != shared_hashes:
                raise RuntimeError("Simulator or fitter mutated paired exogenous inputs")
            schedule_diagnostics, windows = describe_schedule(full, arrays["bounds"], setting.burn_in)
            if "true_window_labels" in arrays:
                np.testing.assert_array_equal(arrays["true_window_labels"], schedule_diagnostics["centre_label_sequence"])
            forced = next(row for row in methods if row["method"] == "forced_hard")
            radii = np.asarray(inputs["spectral_radius"])
            diagnostic = {**base, **fitted_diagnostics, **schedule_diagnostics, "status": status, "error": error,
                          "simulation_completed": simulation_completed,
                          "spectral_radius_state0": float(radii[0]), "spectral_radius_state1": float(radii[1]),
                          "unstable_state_count": int((radii >= 1).sum()),
                          "last_full_segment_right_truncated": schedule["segments_full"][-1]["right_truncated_at_end"],
                          "J": forced.get("J", np.nan), "C_edge": forced.get("C_edge", np.nan),
                          "J_times_C": forced.get("J_times_C", np.nan),
                          "temporal_delta_edges": forced.get("temporal_delta_edges", np.nan),
                          "temporal_delta_edges_total": forced.get("temporal_delta_edges_total", np.nan)}
            if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
                raise ValueError("Retained case outputs must contain numeric arrays only")
            np.savez_compressed(part_npz, **arrays)
            record = {"identity": base, "diagnostic": diagnostic, "schedule_definition": schedule,
                      "methods": [{**base, **row, "fit_status": status, "fit_error": error} for row in methods],
                      "windows": [{**base, **row} for row in windows], "arrays_sha256": sha256(part_npz),
                      "input_array_sha256": shared_hashes}
            write_json(part_json, record)
            records.append(jsonable(record))
    return records


def mean_ci(values: np.ndarray, config: dict) -> dict:
    values = np.asarray(values, dtype=float); values = values[np.isfinite(values)]
    result = {"n_pairs": len(values), "mean_difference": float(np.mean(values)) if len(values) else np.nan,
              "ci95_low": np.nan, "ci95_high": np.nan, "t_stat": np.nan, "p_raw": np.nan}
    settings = config["statistics"]
    if len(values) >= settings["minimum_ci_n"]:
        rng = np.random.default_rng(settings["bootstrap_seed"])
        means = values[rng.integers(len(values), size=(settings["bootstrap_draws"], len(values)))].mean(axis=1)
        result["ci95_low"], result["ci95_high"] = map(float, np.quantile(means, [0.025, 0.975]))
    if len(values) >= 2:
        if np.ptp(values) == 0:
            result.update(t_stat=0.0 if values[0] == 0 else np.copysign(np.inf, values[0]),
                          p_raw=1.0 if values[0] == 0 else 0.0)
        else:
            test = stats.ttest_1samp(values, 0.0)
            result.update(t_stat=float(test.statistic), p_raw=float(test.pvalue))
    return result


def holm_fixed(values: list[float]) -> list[float]:
    """Untestable slots keep their place in the prespecified family size."""
    p = np.asarray(values, dtype=float)
    finite = np.isfinite(p)
    ordered = np.argsort(np.where(finite, p, 1.0), kind="stable")
    adjusted = np.full(len(p), np.nan); running = 0.0
    for rank, index in enumerate(ordered):
        candidate = (len(p) - rank) * (p[index] if finite[index] else 1.0)
        running = max(running, candidate)
        if finite[index]: adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def summarise(records: list[dict], config: dict) -> dict[str, pd.DataFrame]:
    detail = pd.DataFrame([row for record in records for row in record["methods"]])
    diagnostics = pd.DataFrame([record["diagnostic"] for record in records])
    windows = pd.DataFrame([row for record in records for row in record["windows"]])
    keys = ["seed", "duration_mean", "schedule", "cell"]
    wide = detail.pivot(index=keys, columns="method", values="temporal_delta_auroc").reset_index()
    for column in METHODS:
        if column not in wide: wide[column] = np.nan
        wide[column] = pd.to_numeric(wide[column], errors="coerce")
    wide["forced_minus_raw"] = wide.forced_hard - wide.raw_window
    wide["native_minus_raw"] = wide.native_hard - wide.raw_window
    tests = []
    for duration in config["durations"]:
        for kind in config["schedules"]:
            group = wide[(wide.duration_mean == duration) & (wide.schedule == kind)].sort_values("seed")
            tests.append({"family": "B_forced_minus_raw", "family_size": 6, "duration_mean": duration,
                          "schedule": kind, "comparison": "forced_hard_minus_raw_window",
                          "n_planned": config["seed_count"], **mean_ci(group.forced_minus_raw.to_numpy(), config)})
    for kind in config["schedules"]:
        group = wide[wide.schedule == kind].pivot(index="seed", columns="duration_mean", values="forced_minus_raw")
        differences = group.reindex(columns=config["durations"])
        tests.append({"family": "B_duration_gamma", "family_size": 3, "duration_mean": np.nan,
                      "schedule": kind, "comparison": "forced_minus_raw_D600_minus_D300", "n_planned": config["seed_count"],
                      **mean_ci((differences[600] - differences[300]).to_numpy(), config)})
    tests_frame = pd.DataFrame(tests)
    tests_frame["p_holm"] = np.nan
    for family in tests_frame.family.unique():
        mask = tests_frame.family == family
        tests_frame.loc[mask, "p_holm"] = holm_fixed(tests_frame.loc[mask, "p_raw"].tolist())
    tests_frame["significant_holm"] = tests_frame.p_holm < config["statistics"]["alpha"]
    summary, gates = [], []
    for (duration, kind), frame in detail.groupby(["duration_mean", "schedule"], sort=False):
        diagnostic = diagnostics[(diagnostics.duration_mean == duration) & (diagnostics.schedule == kind)]
        for method, values in frame.groupby("method", sort=False):
            interval = mean_ci(pd.to_numeric(values.temporal_delta_auroc, errors="coerce").to_numpy(), config)
            interval.pop("p_raw"); interval.pop("t_stat")
            summary.append({"duration_mean": duration, "schedule": kind, "method": method,
                            "n_planned": config["seed_count"], "n_records": len(values),
                            "n_failed": int((values.fit_status != "ok").sum()), **interval})
        gate = diagnostic.get("hard_gate_accepted", pd.Series(np.nan, index=diagnostic.index)).astype("boolean")
        paired = wide[(wide.duration_mean == duration) & (wide.schedule == kind)].set_index("seed")
        accepted = diagnostic.loc[gate.fillna(False), "seed"]
        finite_gate = gate.notna()
        gates.append({"duration_mean": duration, "schedule": kind, "n_planned": config["seed_count"],
                      "n_gate_observed": int(finite_gate.sum()), "n_gate_accepted": int(gate.fillna(False).sum()),
                      "gate_acceptance": float(gate.mean()) if finite_gate.any() else np.nan,
                      "n_accepted_evaluable": int(np.isfinite(paired.loc[accepted, "forced_minus_raw"]).sum()),
                      "accepted_forced_minus_raw_mean": float(paired.loc[accepted, "forced_minus_raw"].mean()),
                      "all_forced_minus_raw_mean": float(paired.forced_minus_raw.mean()),
                      "all_native_minus_raw_mean": float(paired.native_minus_raw.mean()),
                      "J_mean": float(pd.to_numeric(diagnostic.J, errors="coerce").mean()),
                      "C_edge_mean": float(pd.to_numeric(diagnostic.C_edge, errors="coerce").mean()),
                      "J_times_C_mean": float(pd.to_numeric(diagnostic.J_times_C, errors="coerce").mean())})
    return {"schedule_robustness_detail.csv": detail, "schedule_robustness_diagnostics.csv": diagnostics,
            "schedule_robustness_windows.csv": windows, "schedule_robustness_paired.csv": wide,
            "schedule_robustness_secondary_tests.csv": tests_frame,
            "schedule_robustness_method_summary.csv": pd.DataFrame(summary),
            "schedule_robustness_gate_summary.csv": pd.DataFrame(gates)}


def aggregate(output: Path, config: dict) -> dict[str, pd.DataFrame]:
    records = []
    for seed in range(config["seed_start"], config["seed_start"] + config["seed_count"]):
        for duration in config["durations"]:
            for kind in config["schedules"]:
                file = output / "parts" / f"repeat_{seed}_D{duration}_{kind}.json"
                if not file.exists():
                    raise RuntimeError(f"Missing planned case: {file.name}; no silent seed deletion")
                record = json.loads(file.read_text(encoding="utf-8"))
                arrays = file.with_suffix(".npz")
                if sha256(arrays) != record["arrays_sha256"]:
                    raise RuntimeError(f"Case arrays checksum mismatch: {arrays.name}")
                records.append(record)
    tables = summarise(records, config)
    for name, table in tables.items():
        table.to_csv(output / name, index=False, lineterminator="\n")
    write_json(output / "schedule_robustness_outputs.json", {"source_protocol_sha256": sha256(output / "schedule_robustness_protocol.json"),
        "sha256": {path.relative_to(output).as_posix(): sha256(path) for path in sorted(output.rglob("*"))
                   if path.is_file() and path.name != "schedule_robustness_outputs.json"}})
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze", action="store_true", help="Write the approved immutable protocol; do not fit")
    mode.add_argument("--run", action="store_true", help="Execute only an already frozen, unchanged protocol")
    mode.add_argument("--aggregate", action="store_true", help="Recompute summaries from all retained planned cases; do not fit")
    args = parser.parse_args()
    config = load_config(args.config)
    proposed = protocol(config, args.config)
    if not (args.freeze or args.run or args.aggregate):
        print(json.dumps(proposed, indent=2, sort_keys=True))
        return
    ensure_protocol(args.output_dir, proposed, freeze=args.freeze)
    if args.freeze:
        print("Frozen protocol only:", sha256(args.output_dir / "schedule_robustness_protocol.json"))
        return
    with threadpool_limits(limits=config["threads"]):
        if args.run:
            for seed in proposed["repeats"]:
                run_repeat(config, seed, args.output_dir)
                print(f"Completed paired repeat {seed}: six schedule cells", flush=True)
        tables = aggregate(args.output_dir, config)
    print(tables["schedule_robustness_secondary_tests.csv"].to_string(index=False))


if __name__ == "__main__":
    main()
