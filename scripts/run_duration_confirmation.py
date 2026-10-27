"""Frozen, independent paired confirmation at durations 300 and 600.

Prepare the protocol with --freeze-protocol, review it, then run only through
--run-protocol. No simulation is performed by freezing. No optional stopping,
replacement repeats, source changes or unrecorded cache reuse are permitted.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stcg.mechanism import MechanismConfig
from aggregation_study_common import (
    COMMON_SOURCES, ScorerFailure, draw_base_inputs, fit_aggregation,
    fixed_schedule, simulate_schedule, source_hashes,
)

SOURCE_FILES = (*COMMON_SOURCES, "scripts/run_duration_confirmation.py", "tools/verify_duration_confirmation.py")
METHODS = ("raw_window", "native_hard", "forced_hard", "matched_random", "oracle_label", "static_mean")
PRIMARY = ("native_hard_minus_raw_D300", "native_hard_minus_raw_D600",
           "native_hard_duration_interaction", "native_hard_minus_matched_random_D600")


def canonical_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    """Preserve full float precision and encode undefined numbers as null."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def default_protocol() -> dict:
    base = MechanismConfig(n_nodes=8, timesteps=1200, max_lag=3, edge_prob=0.18,
                           switch_period=300, overlap=0.5, effect_scale=1.0,
                           noise_scale=0.10, observation_noise=0, burn_in=100)
    return {
        "study": "information_sciences_duration_confirmation_v1", "frozen": True,
        "repeat_seeds": list(range(50000, 50100)), "n_planned_repeats": 100,
        "durations": [300, 600], "base_config": asdict(base),
        "score_pipeline": {"window_size": 300, "stride": 150, "lasso_alpha": 0.005, "n_init": 10},
        "scorers": ["lasso", "ols"], "methods": list(METHODS),
        "pairing": "Within repeat: identical coefficients, support, initial history, full innovations and KMeans random_state across both D and scorers; repeats redraw all inputs independently",
        "random_streams": {"generator": "PCG64 via SeedSequence([repeat_seed,domain])",
                           "mechanism_domain": 0, "initial_history_domain": 1, "innovations_domain": 2,
                           "cluster_random_state": "repeat_seed (same across conditions and scorers)"},
        "schedule": {"kind": "fixed_periodic", "full_time_origin": 0, "phase": 0,
                     "observed_offset": 100, "first_state": 0,
                     "definition": "z_full[t] = floor(t/D) modulo 2; discard first 100 observations"},
        "generator": {"coefficient_rule": "call existing _build_coefficients without changes",
                      "stabilisation": "existing single scalar rescale; record actual per-state radii; no redraw"},
        "ols": {"solver": "numpy.linalg.lstsq(rcond=None)", "full_column_rank_required": True,
                "rank_failure": "retain failure and inputs, no ridge/pseudoinverse substitute or replacement repeat",
                "preprocessing": "same two-stage zscore and all self/cross-lag regression design as Lasso; self scores zeroed"},
        "hard_policy": "unchanged native STCG gate and static window-max fallback; forced bypasses gate only",
        "matched_random": "all fixed-group-size unlabelled bipartitions of 7 windows; calculate each partition AUROC before averaging within repeat",
        "primary_family": {"scorer": "lasso", "hard": "native_hard", "contrasts": list(PRIMARY),
                           "test": "two-sided one-sample t on paired-repeat contrasts versus zero", "adjustment": "Holm over all four, including unavailable tests as p=1 for adjustment"},
        "intervals": {"type": "pointwise percentile bootstrap over paired simulation repeats",
                      "draws": 10000, "seed": 20261024, "level": 0.95, "simultaneous": False, "minimum_n": 5},
        "failure_policy": "retain every planned repeat; no replacement, additional repeats, or zero/0.5 imputation; report missing-target and numerical failures with denominators; Gamma uses complete pairs",
        "auxiliary": "forced hard and all OLS contrasts are prespecified auxiliary, outside the four-test primary family",
        "threads": 1, "source_sha256": source_hashes(SOURCE_FILES),
    }


def freeze_protocol(path: Path) -> dict:
    protocol = default_protocol()
    path = Path(path)
    if path.exists():
        raise FileExistsError("A frozen protocol cannot be overwritten; choose a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(protocol))
    return protocol


def validate_protocol(protocol: dict) -> None:
    if not protocol.get("frozen") or protocol.get("study") != "information_sciences_duration_confirmation_v1":
        raise ValueError("An explicitly frozen duration protocol is required")
    if protocol != default_protocol():
        raise ValueError("Frozen protocol or source hashes differ from the reviewed fixed study")


def mean_interval(values, interval: dict) -> tuple[float, float, float, int]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    n = len(values)
    if not n:
        return np.nan, np.nan, np.nan, 0
    if n < interval["minimum_n"]:
        return float(values.mean()), np.nan, np.nan, n
    rng = np.random.default_rng(interval["seed"])
    means = values[rng.integers(n, size=(interval["draws"], n))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high), n


def holm(values) -> np.ndarray:
    """Keep the planned family size when a contrast has no computable test."""
    p = np.asarray(values, float)
    usable = np.where(np.isfinite(p), p, 1.0)
    order = np.argsort(usable, kind="stable")
    corrected = np.minimum(1, np.maximum.accumulate(usable[order] * (len(p) - np.arange(len(p)))))
    result = np.empty_like(corrected); result[order] = corrected
    result[~np.isfinite(p)] = np.nan
    return result


def test_zero(values) -> float:
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    if np.ptp(values) == 0:
        return 1.0 if values[0] == 0 else 0.0
    return float(stats.ttest_1samp(values, 0.0, alternative="two-sided").pvalue)


def _failed_rows(scorer: str, reason: str, error: str) -> list[dict]:
    return [{"scorer": scorer, "method": method, "status": "failed", "error": error,
             "undefined_reason": reason, "evaluable": False, "temporal_delta_auroc": np.nan,
             "pooled_temporal_auroc": np.nan, "union_auroc": np.nan,
             "temporal_delta_edges": 0, "temporal_delta_edges_total": np.nan,
             "J": np.nan, "C_edge": np.nan, "J_times_C": np.nan, "identity_gap": np.nan,
             "hard_gate_accepted": np.nan, "aggregation_executed": False} for method in METHODS]


def _record_hashes(folder: Path, paths) -> dict:
    return {Path(p).relative_to(folder).as_posix(): file_sha(Path(p)) for p in paths}


def read_repeat(folder: Path, seed: int, protocol_sha: str) -> list[dict]:
    record_path = folder / f"repeat_{seed}.json"
    record = json.loads(record_path.read_text())
    if record["protocol_sha256"] != protocol_sha:
        raise ValueError("Cached repeat belongs to a different frozen protocol")
    for relative, expected in record["sha256"].items():
        path = folder / relative
        if path.resolve().parent != folder.resolve() or not path.is_file() or file_sha(path) != expected:
            raise ValueError(f"Cached repeat checksum mismatch: {relative}")
    return record["rows"]


def run_repeat(protocol: dict, seed: int, folder: Path, protocol_sha: str) -> list[dict]:
    """Write inputs before fitting; save each cell's success/failure separately."""
    config = MechanismConfig(**protocol["base_config"])
    inputs = draw_base_inputs(config, seed)
    paths = []
    input_path = folder / f"repeat_{seed}_inputs.npz"
    np.savez_compressed(input_path, **inputs); paths.append(input_path)
    rows = []
    for duration in protocol["durations"]:
        condition = replace(config, switch_period=duration, seed=seed)
        schedule = fixed_schedule(config.timesteps + config.burn_in, duration)
        data_path = folder / f"repeat_{seed}_D{duration}_data.npz"
        try:
            dataset = simulate_schedule(inputs, schedule, condition)
            np.savez_compressed(data_path, x=dataset.x, regimes=dataset.regimes, regimes_full=schedule)
            paths.append(data_path)
            simulation_error = None
        except Exception as exc:
            dataset = None; simulation_error = exc
            np.savez_compressed(data_path, regimes_full=schedule); paths.append(data_path)
        for scorer in protocol["scorers"]:
            fit_path = folder / f"repeat_{seed}_D{duration}_{scorer}"
            try:
                if simulation_error is not None:
                    raise simulation_error
                fitted, arrays, info = fit_aggregation(dataset, scorer, seed, exact_random=True,
                                                       **protocol["score_pipeline"])
                np.savez_compressed(fit_path.with_suffix(".npz"), **arrays)
                paths.append(fit_path.with_suffix(".npz"))
                fit_path.with_suffix(".json").write_text(json.dumps(json_safe(
                    {"status": "success", "diagnostic": info}), indent=2, sort_keys=True, allow_nan=False) + "\n")
                for row in fitted:
                    row["status"] = "success"; row["error"] = ""
            except Exception as exc:
                failure = {"status": "failed", "exception_type": type(exc).__name__, "error": str(exc),
                           "diagnostic": getattr(exc, "diagnostic", {}), "traceback": traceback.format_exc()}
                fit_path.with_suffix(".json").write_text(json.dumps(json_safe(failure), indent=2, sort_keys=True,
                                                                   allow_nan=False) + "\n")
                fitted = _failed_rows(scorer, "simulation_failure" if simulation_error else "scorer_or_aggregation_failure", str(exc))
            paths.append(fit_path.with_suffix(".json"))
            rows.extend({"repeat_seed": seed, "D": duration, **row} for row in fitted)
    # Null is used only for JSON storage of missing numbers; CSV returns NaN.
    clean_rows = json_safe(rows)
    record = {"repeat_seed": seed, "protocol_sha256": protocol_sha, "rows": clean_rows,
              "sha256": _record_hashes(folder, paths)}
    (folder / f"repeat_{seed}.json").write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return clean_rows


def summarise(frame: pd.DataFrame, protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Infer only across repeats; edge/window/partition counts are diagnostics."""
    summary, tests = [], []
    for scorer in protocol["scorers"]:
        subset = frame[frame.scorer == scorer]
        pivot = subset.pivot(index="repeat_seed", columns=["D", "method"], values="temporal_delta_auroc")
        for (duration, method), group in subset.groupby(["D", "method"], sort=True):
            row = {"scorer": scorer, "D": duration, "method": method, "n_planned": len(protocol["repeat_seeds"]),
                   "n_failed": int((group.status == "failed").sum()),
                   "n_undefined_target": int(((group.status == "success") & ~group.evaluable.astype(bool)).sum()),
                   "n_gate_evaluable": int(group.hard_gate_accepted.notna().sum()),
                   "gate_acceptance": float(pd.to_numeric(group.hard_gate_accepted, errors="coerce").mean()),
                   "execution_rate": float(group.aggregation_executed.mean())}
            for metric in ("temporal_delta_auroc", "pooled_temporal_auroc", "union_auroc", "J", "C_edge", "J_times_C"):
                mean, low, high, n = mean_interval(group[metric].to_numpy(), protocol["intervals"])
                row.update({metric + "_mean": mean, metric + "_ci95_low": low, metric + "_ci95_high": high,
                            metric + "_n": n})
            difference = pivot[(duration, method)] - pivot[(duration, "raw_window")]
            mean, low, high, n = mean_interval(difference, protocol["intervals"])
            row.update(delta_temporal_mean=mean, delta_temporal_ci95_low=low, delta_temporal_ci95_high=high, n_paired=n)
            summary.append(row)
        for hard in ("native_hard", "forced_hard"):
            differences = {d: pivot[(d, hard)] - pivot[(d, "raw_window")] for d in protocol["durations"]}
            contrasts = {f"{hard}_minus_raw_D300": differences[300],
                         f"{hard}_minus_raw_D600": differences[600],
                         f"{hard}_duration_interaction": differences[600] - differences[300],
                         f"{hard}_minus_matched_random_D600": pivot[(600, hard)] - pivot[(600, "matched_random")]}
            for contrast, values in contrasts.items():
                mean, low, high, n = mean_interval(values, protocol["intervals"])
                tests.append({"scorer": scorer, "contrast": contrast, "n_planned": len(protocol["repeat_seeds"]),
                              "n_paired": n, "mean": mean, "ci95_low": low, "ci95_high": high,
                              "p_raw": test_zero(values), "primary": scorer == "lasso" and contrast in PRIMARY,
                              "interval_scope": "pointwise, not simultaneous"})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = np.nan
    selected = tests.primary
    if selected.sum() != 4:
        raise AssertionError("The primary family must retain exactly four planned contrasts")
    tests.loc[selected, "p_holm"] = holm(tests.loc[selected, "p_raw"])
    return pd.DataFrame(summary), tests


def run_study(protocol_path: Path, output_dir: Path, *, resume: bool = False) -> dict:
    protocol = json.loads(Path(protocol_path).read_text())
    validate_protocol(protocol)
    protocol_data = canonical_bytes(protocol); protocol_sha = hashlib.sha256(protocol_data).hexdigest()
    output_dir = Path(output_dir)
    frozen_path = output_dir / "protocol.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume or not frozen_path.is_file() or frozen_path.read_bytes() != protocol_data:
            raise ValueError("Output directory is not empty or protocol differs; refuse to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path.write_bytes(protocol_data)
    records = output_dir / "records"; records.mkdir(exist_ok=True)
    rows = []
    with threadpool_limits(limits=protocol["threads"]):
        for count, seed in enumerate(protocol["repeat_seeds"], start=1):
            if (records / f"repeat_{seed}.json").exists():
                rows.extend(read_repeat(records, seed, protocol_sha))
            else:
                rows.extend(run_repeat(protocol, seed, records, protocol_sha))
            print(f"Duration confirmation: {count}/{len(protocol['repeat_seeds'])} fixed repeats retained", flush=True)
    frame = pd.DataFrame(rows)
    frame = frame[sorted(frame.columns)]
    frame.to_csv(output_dir / "detail.csv", index=False)
    summary, tests = summarise(frame, protocol)
    summary.to_csv(output_dir / "summary.csv", index=False)
    tests.to_csv(output_dir / "contrasts.csv", index=False)
    environment = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                   "pandas": pd.__version__, "requested_repeats": len(protocol["repeat_seeds"]),
                   "retained_rows": len(frame), "numerical_failure_rows": int((frame.status == "failed").sum())}
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    files = [p for p in output_dir.rglob("*") if p.is_file() and p.name != "COMPLETE.json"]
    complete = {"protocol_sha256": protocol_sha, "n_repeats": len(protocol["repeat_seeds"]),
                "n_detail_rows": len(frame), "sha256": _record_hashes(output_dir, files)}
    (output_dir / "COMPLETE.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
    print(tests.to_string(index=False), flush=True)
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze-protocol", type=Path, help="Write reviewed default protocol and hashes only; no simulations")
    actions.add_argument("--run-protocol", type=Path, help="Execute exactly this source-locked frozen protocol")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/information_sciences/duration_confirmation")
    parser.add_argument("--resume", action="store_true", help="Reuse only hash-verified repeats from the identical protocol")
    args = parser.parse_args()
    if args.freeze_protocol:
        freeze_protocol(args.freeze_protocol)
        print(f"Protocol frozen without simulation: {args.freeze_protocol}")
    else:
        run_study(args.run_protocol, args.output_dir, resume=args.resume)


if __name__ == "__main__":
    main()
