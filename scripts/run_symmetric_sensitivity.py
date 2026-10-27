"""Study C: new symmetric-support generator and bounded window sensitivity.

The original generator, scorer helpers and Studies A/B are never edited.
--freeze writes a source/config locked protocol and real internal clock record.
--run requires that freeze; --aggregate replays retained records; --verify is
strictly read-only. Study C is post-review robustness, not preregistration or
an extension of the A/B confirmatory evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import aggregation_study_common as common
from stcg.evaluation import nonself_mask, window_purity, window_regime_labels
from stcg.mechanism import MechanismConfig, MechanismDataset, spectral_radius
from stcg.self_supervised import STCGStateConfig, _min_cluster_fraction, _state_improvement, _temporal_coherence

CONFIG = ROOT / "configs/symmetric_sensitivity.yaml"
OUT = ROOT / "results/information_sciences/symmetric_sensitivity"
LOGS = ROOT / "experiments/logs/reviewer_controls_2026-10-25"
PROTOCOL_NAME = "symmetric_sensitivity_protocol.json"
METHODS = ("raw_window", "native_hard", "forced_hard", "oracle_label", "static_mean")
NAMESPACE = "stcg_symmetric_sensitivity_v1"
SOURCES = ("scripts/run_symmetric_sensitivity.py", "tests/test_symmetric_sensitivity.py",
           "scripts/aggregation_study_common.py", "src/stcg/mechanism.py", "src/stcg/baselines.py",
           "src/stcg/self_supervised.py", "src/stcg/evaluation.py", "src/stcg/__init__.py",
           "requirements-revision.txt")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
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


def directory_hashes(output: Path) -> dict:
    return {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}


def load_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def fingerprints(arrays: dict) -> dict:
    return {key: hashlib.sha256(json.dumps([str(np.asarray(v).dtype), list(np.shape(v))]).encode()
                               + np.ascontiguousarray(v).tobytes()).hexdigest() for key, v in arrays.items()}


def domain_seed(seed: int, domain: str, bits: int = 128) -> int:
    token = json.dumps([NAMESPACE, int(seed), domain], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:bits // 8], "little")


def sample_supports(rng: np.random.Generator, n_nodes: int, max_lag: int,
                    probability: float, retention: float) -> np.ndarray:
    """Exactly exchangeable Bernoulli-p supports, with no empty-support fallback."""
    if not 0 < probability < 1 or not 0 <= retention <= 1:
        raise ValueError("Require 0<p<1 and 0<=retention<=1")
    addition = probability * (1 - retention) / (1 - probability)
    if addition > 1:
        raise ValueError("The requested Bernoulli marginals and retention are incompatible")
    shape = (max_lag, n_nodes, n_nodes)
    first = rng.random(shape) < probability
    first[:, np.arange(n_nodes), np.arange(n_nodes)] = False
    second = (first & (rng.random(shape) < retention)) | (~first & (rng.random(shape) < addition))
    second[:, np.arange(n_nodes), np.arange(n_nodes)] = False
    return np.stack([first, second])


def lag_aware_scale_pair(coefficients: np.ndarray, target: float = 0.9) -> dict:
    """A_l -> a**l A_l scales companion eigenvalues by a in both states.

    A shared positive factor for each lag preserves shared coefficients exactly.
    A recorded 1e-12 relative safety margin is used only when scaling is needed.
    State-wise stability does not establish stability under arbitrary switching.
    """
    raw = np.asarray(coefficients, dtype=float)
    if raw.ndim != 4 or raw.shape[0] != 2 or not np.isfinite(raw).all() or not 0 < target < 1:
        raise ValueError("Expected finite two-state lagged coefficient tensors and 0<target<1")
    initial = np.array([spectral_radius(c) for c in raw])
    maximum = float(initial.max())
    ideal = min(1.0, target / maximum) if maximum else 1.0
    scale = ideal * (1 - 1e-12) if ideal < 1 else 1.0
    lag_scale = scale ** np.arange(1, raw.shape[1] + 1, dtype=float)
    final = raw * lag_scale[None, :, None, None]
    radii = np.array([spectral_radius(c) for c in final])
    np.testing.assert_allclose(radii, scale * initial, atol=2e-11, rtol=2e-11)
    if np.any(radii > target):
        raise common.ScorerFailure("Final state radius exceeds fixed target; no resampling", {"radii": radii.tolist()})
    return {"coefficients": final, "rho_initial": initial, "rho_final": radii,
            "ideal_common_scale": np.asarray(ideal), "common_scale": np.asarray(scale), "lag_scale": lag_scale}


def draw_inputs(config: dict, seed: int) -> dict:
    m = config["mechanism"]
    rng = np.random.default_rng(domain_seed(seed, "mechanism"))
    support = sample_supports(rng, m["n_nodes"], m["max_lag"], m["edge_prob"], m["overlap"])
    shape = support.shape[1:]
    decay = np.arange(1, m["max_lag"] + 1)[:, None, None]
    first_values = rng.choice([-1.0, 1.0], size=shape) * rng.uniform(.08, .35, size=shape) / decay
    second_values = rng.choice([-1.0, 1.0], size=shape) * rng.uniform(.08, .35, size=shape) / decay
    raw = np.stack([support[0] * first_values,
                    np.where(support[0] & support[1], first_values, second_values) * support[1]])
    scaled = lag_aware_scale_pair(raw, m["target_radius"])
    total = m["timesteps"] + m["burn_in"]
    return {"support": support, "coefficients_raw": raw, **scaled,
            "initial_values": np.random.default_rng(domain_seed(seed, "initial_history")).normal(
                0, m["noise_scale"], (m["max_lag"], m["n_nodes"])),
            "innovations": np.random.default_rng(domain_seed(seed, "innovations")).normal(
                0, m["noise_scale"], (total, m["n_nodes"]))}


def setting(config: dict, seed: int, duration: int) -> MechanismConfig:
    m = config["mechanism"]
    return MechanismConfig(n_nodes=m["n_nodes"], timesteps=m["timesteps"], max_lag=m["max_lag"],
                           edge_prob=m["edge_prob"], overlap=m["overlap"], noise_scale=m["noise_scale"],
                           burn_in=m["burn_in"], switch_period=duration, seed=seed)


def simulate_full(inputs: dict, duration: int, config: dict) -> tuple[dict, MechanismDataset]:
    m = config["mechanism"]; total = m["timesteps"] + m["burn_in"]
    regimes_full = ((np.arange(total) // duration) % 2).astype(np.int8)
    x = np.empty((total, m["n_nodes"]), dtype=float)
    x[:m["max_lag"]] = inputs["initial_values"]
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(m["max_lag"], total):
            x[t] = inputs["innovations"][t]
            for lag in range(1, m["max_lag"] + 1):
                x[t] += x[t - lag] @ inputs["coefficients"][int(regimes_full[t]), lag - 1]
    observed = x[m["burn_in"]:].copy(); regimes = regimes_full[m["burn_in"]:].astype(int)
    support = inputs["support"]
    labels, counts = np.unique(regimes, return_counts=True)
    first_n = int(support[0].sum())
    dataset = MechanismDataset(x=observed, coefficients=inputs["coefficients"], state_support=support,
        union_support=support.max(axis=0), regimes=regimes, spectral_radius=inputs["rho_final"],
        overlap_realised=float((support[0] & support[1]).sum() / first_n) if first_n else np.nan,
        durations={int(k): int(v) for k, v in zip(labels, counts)})
    return {"full_series": x, "observed_series": observed, "regimes_full": regimes_full,
            "regimes": regimes, "coefficients": inputs["coefficients"], "state_support": support}, dataset


def bounds(config: dict, window: dict) -> np.ndarray:
    return np.asarray([(start, start + window["size"]) for start in range(
        0, config["mechanism"]["timesteps"] - window["size"] + 1, window["stride"])])


def cell_name(duration: int, window: dict) -> str:
    return f"D{duration}_h{window['size']}_s{window['stride']}"


def load_config(path: Path = CONFIG) -> dict:
    c = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (c["study"] != "information_sciences_symmetric_sensitivity_v1" or c["threads"] != 1 or
            c["generator"] != "exchangeable_support_shared_coefficients_lag_scaled_v1"):
        raise ValueError("Study C identity and one thread are fixed")
    if (c["seed_start"], c["seed_count"], c["durations"]) != (70000, 30, [300, 600]):
        raise ValueError("Study C retains the fixed 30 new seeds and both durations")
    if c["windows"] != [{"size": h, "stride": s} for h, s in [(200, 100), (300, 150), (400, 200), (300, 75), (300, 300)]]:
        raise ValueError("Study C has exactly five fixed window/stride settings")
    if c["mechanism"] != {"n_nodes": 8, "timesteps": 3600, "max_lag": 3, "edge_prob": .18,
            "overlap": .5, "noise_scale": .10, "burn_in": 100, "target_radius": .90}:
        raise ValueError("The symmetric generator specification is fixed")
    if c["model"] != {"scorer": "lasso", "lasso_alpha": .005, "n_init": 10}:
        raise ValueError("Study C retains the original local Lasso scoring definition")
    if c["statistics"] != {"bootstrap_draws": 10000, "bootstrap_seed": 20261025, "minimum_ci_n": 5}:
        raise ValueError("Study C freezes descriptive pointwise intervals only")
    return c


def protocol(config: dict, config_path: Path = CONFIG) -> dict:
    p = config["mechanism"]["edge_prob"]; o = config["mechanism"]["overlap"]
    return {"study": config["study"], "role": "Post-review robustness; not part of A/B confirmation or public preregistration",
        "configuration": config, "configuration_sha256": sha256(config_path),
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCES},
        "repeats": list(range(config["seed_start"], config["seed_start"] + config["seed_count"])),
        "expected_fits": config["seed_count"] * len(config["durations"]) * len(config["windows"]),
        "generator": {"name": config["generator"], "p_both_nonself": p, "p_B_given_A": o,
            "p_B_given_not_A": p * (1 - o) / (1 - p), "self": "absent in both states",
            "empty_support": "allowed without fallback or conditional resampling",
            "coefficients": "shared support has identical values; exclusive supports independently signed Uniform(.08,.35)/lag",
            "grouped_change": "self rule, marginal support density and shared coefficients change together; not a factorial attribution",
            "stability": "ideal a=min(1,.90/max initial state rho); when a<1 multiply a by (1-1e-12) as a recorded numerical safety margin; both states lag l multiplied by same a**l; verify final rho=a*initial rho and <=.90; no arbitrary-switching guarantee"},
        "pairing": "Within repeat, both D and all five window settings share coefficients, initial history and innovations; no replacement or extension",
        "rng": {"namespace": NAMESPACE, "address": "SHA256 compact JSON [namespace, repeat, domain], little-endian first128 bits; cluster first32 bits",
                "domains": ["mechanism", "initial_history", "innovations", "clustering"]},
        "schedule": "fixed full-time floor(t/D)%2, origin0; common100 burn-in discarded; no random phase",
        "window_design": "h/s proportional triplet plus stride-only contrasts at fixed h300; the triplet alone does not isolate h from stride or M",
        "methods": list(METHODS), "inference": "paired-repeat descriptive means and pointwise percentile bootstrap intervals; no new confirmatory p-value family; undefined targets/failures retain fixed planned denominators",
        "no_optional_stopping": True, "threads": 1}


def freeze(output: Path, proposed: dict, log_dir: Path = LOGS) -> dict:
    before = utc_now()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Freeze requires an empty new output directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / PROTOCOL_NAME, proposed)
    record = {"stage": "freeze_only", "before_freeze_utc": before, "after_freeze_utc": utc_now(),
              "protocol_sha256": sha256(output / PROTOCOL_NAME), "source_sha256": proposed["source_sha256"],
              "new_fits": 0, "new_simulations": 0, "clock_status": "internal unverified wall clock; not public registration"}
    write_json(output / "freeze_record.json", record)
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(log_dir / "symmetric_sensitivity_freeze.json", record)
    return record


def ensure_protocol(output: Path, proposed: dict) -> None:
    path = output / PROTOCOL_NAME
    if not path.exists() or json.loads(path.read_text(encoding="utf-8")) != jsonable(proposed):
        raise RuntimeError("An identical frozen protocol/config/source is required; no fitting allowed")
    record = json.loads((output / "freeze_record.json").read_text(encoding="utf-8"))
    if record["protocol_sha256"] != sha256(path):
        raise RuntimeError("Frozen protocol checksum differs")


def _npz(path: Path, arrays: dict) -> None:
    if any(np.asarray(v).dtype.hasobject for v in arrays.values()):
        raise ValueError("Retained arrays must be numerical, never pickled objects")
    np.savez_compressed(path, **arrays)


def run_repeat(config: dict, seed: int, output: Path, fitter=None) -> list[dict]:
    fitter = common.fit_aggregation if fitter is None else fitter
    for name in ("inputs", "series", "parts"):
        (output / name).mkdir(exist_ok=True, parents=True)
    path = output / "inputs" / f"repeat_{seed}.npz"
    if path.exists():
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if sha256(path) != metadata["sha256"]:
            raise RuntimeError("Input checksum mismatch")
        inputs = load_arrays(path)
    else:
        inputs = draw_inputs(config, seed); _npz(path, inputs)
        write_json(path.with_suffix(".json"), {"seed": seed, "sha256": sha256(path), "array_sha256": fingerprints(inputs)})
    input_sha = sha256(path); before_inputs = fingerprints(inputs)
    cluster_seed = domain_seed(seed, "clustering", bits=32)
    records = []
    for duration in config["durations"]:
        series_path = output / "series" / f"repeat_{seed}_D{duration}.npz"
        simulation_start = perf_counter()
        if series_path.exists():
            series_meta = json.loads(series_path.with_suffix(".json").read_text(encoding="utf-8"))
            if sha256(series_path) != series_meta["sha256"] or input_sha != series_meta["input_sha256"]:
                raise RuntimeError("Series checksum or paired input mismatch")
            series = load_arrays(series_path)
        else:
            series, _ = simulate_full(inputs, duration, config)
            _npz(series_path, series)
            write_json(series_path.with_suffix(".json"), {"seed": seed, "D": duration,
                "sha256": sha256(series_path), "input_sha256": input_sha,
                "simulation_seconds": perf_counter() - simulation_start,
                "full_finite": bool(np.isfinite(series["full_series"]).all()),
                "observed_finite": bool(np.isfinite(series["observed_series"]).all())})
        series_sha = sha256(series_path)
        support = inputs["support"]; regimes = series["regimes"]
        base_count = int(support[0].sum()); labels, counts = np.unique(regimes, return_counts=True)
        dataset = MechanismDataset(x=series["observed_series"], coefficients=inputs["coefficients"], state_support=support,
            union_support=support.max(axis=0), regimes=regimes, spectral_radius=inputs["rho_final"],
            overlap_realised=float((support[0] & support[1]).sum() / base_count) if base_count else np.nan,
            durations={int(k): int(v) for k, v in zip(labels, counts)}, config=setting(config, seed, duration))
        for window in config["windows"]:
            cell = cell_name(duration, window); part = output / "parts" / f"repeat_{seed}_{cell}.npz"
            if part.with_suffix(".json").exists():
                record = json.loads(part.with_suffix(".json").read_text(encoding="utf-8"))
                if sha256(part) != record["arrays_sha256"] or input_sha != record["input_sha256"] or series_sha != record["series_sha256"]:
                    raise RuntimeError("Retained fit checksum or pairing mismatch")
                records.append(record); continue
            identity = {"study": "C", "scorer": "lasso", "seed": seed, "D": duration, "duration": duration,
                        "h": window["size"], "stride": window["stride"], "cell": cell,
                        "n_windows": len(bounds(config, window)), "cluster_seed": cluster_seed}
            arrays = {"observed_series": dataset.x, "regimes": regimes, "state_support": support,
                      "coefficients": inputs["coefficients"], "bounds": bounds(config, window)}
            started = perf_counter(); status, error = "ok", None
            try:
                if not np.isfinite(series["full_series"]).all():
                    raise common.ScorerFailure("Nonfinite full trajectory; saved without replacement")
                rows, fitted, numerical = fitter(dataset, scorer="lasso", cluster_seed=cluster_seed, exact_random=False,
                    window_size=window["size"], stride=window["stride"], lasso_alpha=config["model"]["lasso_alpha"],
                    n_init=config["model"]["n_init"])
                if {row["method"] for row in rows} != set(METHODS):
                    raise RuntimeError("Unexpected aggregation method set")
                arrays.update(fitted)
            except Exception as exc:
                status, error = "failed", f"{type(exc).__name__}: {exc}"
                rows = [{"method": name, "temporal_delta_auroc": np.nan, "pooled_temporal_auroc": np.nan,
                         "union_auroc": np.nan, "evaluable": False, "undefined_reason": "fit_failed"} for name in METHODS]
                numerical = getattr(exc, "diagnostic", {})
            truth = window_regime_labels(arrays["bounds"], regimes)
            purity, _ = window_purity(arrays["bounds"], regimes)
            diagnostic = {**identity, "status": status, "error": error, "fit_seconds": perf_counter() - started,
                "M": len(arrays["bounds"]), "h_over_D": window["size"] / duration,
                "n_actual_switches": int(np.count_nonzero(regimes[1:] != regimes[:-1])),
                "n_centre_switches": int(np.count_nonzero(truth[1:] != truth[:-1])),
                "centre_state0": int((truth == 0).sum()), "centre_state1": int((truth == 1).sum()),
                "window_purity_mean": float(purity.mean()), "window_purity_min": float(purity.min()),
                "rho_initial": inputs["rho_initial"], "rho_final": inputs["rho_final"],
                "common_scale": inputs["common_scale"], "full_finite": bool(np.isfinite(series["full_series"]).all()),
                "observed_max_abs": float(np.max(np.abs(dataset.x))), "fitting": numerical}
            _npz(part, arrays)
            record = {"identity": identity, "diagnostic": diagnostic,
                "methods": [{**identity, **row, "fit_status": status, "fit_error": error} for row in rows],
                "arrays_sha256": sha256(part), "input_sha256": input_sha, "series_sha256": series_sha}
            write_json(part.with_suffix(".json"), record); records.append(jsonable(record))
    if fingerprints(inputs) != before_inputs:
        raise RuntimeError("An estimator mutated paired exogenous inputs")
    return records


def interval(values, config: dict) -> dict:
    a = np.asarray(values, dtype=float); a = a[np.isfinite(a)]
    result = {"n": len(a), "n_evaluable": len(a), "mean": float(a.mean()) if len(a) else np.nan,
              "ci95_low": np.nan, "ci95_high": np.nan}
    s = config["statistics"]
    if len(a) >= s["minimum_ci_n"]:
        rng = np.random.default_rng(s["bootstrap_seed"])
        means = a[rng.integers(len(a), size=(s["bootstrap_draws"], len(a)))].mean(axis=1)
        result["ci95_low"], result["ci95_high"] = map(float, np.quantile(means, [.025, .975]))
    return result


def collect_records(output: Path, config: dict) -> list[dict]:
    records = []
    for seed in range(config["seed_start"], config["seed_start"] + config["seed_count"]):
        for duration in config["durations"]:
            for window in config["windows"]:
                file = output / "parts" / f"repeat_{seed}_{cell_name(duration, window)}.json"
                if not file.exists():
                    raise RuntimeError(f"Missing planned case {file.name}; no silent deletion")
                record = json.loads(file.read_text(encoding="utf-8"))
                if sha256(file.with_suffix(".npz")) != record["arrays_sha256"]:
                    raise RuntimeError(f"Case checksum differs: {file.name}")
                records.append(record)
    return records


def summarise(records: list[dict], config: dict) -> dict[str, pd.DataFrame]:
    detail = pd.DataFrame([row for record in records for row in record["methods"]])
    diagnostic = pd.DataFrame([{**r["identity"], **{k: v for k, v in r["diagnostic"].items()
        if k not in {"fitting", "rho_initial", "rho_final"}}} for r in records])
    paired = detail.pivot(index=["seed", "D", "h", "stride", "cell"], columns="method", values="temporal_delta_auroc").reset_index()
    paired["duration"] = paired.D
    window_counts = dict(zip(diagnostic.cell, diagnostic.M))
    paired["n_windows"] = paired.cell.map(window_counts)
    paired["h_over_D"] = paired.h / paired.D
    for method in METHODS:
        paired[method] = pd.to_numeric(paired[method], errors="coerce")
        paired[method + "_minus_raw"] = paired[method] - paired.raw_window
    summary, gates = [], []
    for (duration, h, stride, cell), frame in detail.groupby(["D", "h", "stride", "cell"], sort=True):
        d = diagnostic[diagnostic.cell == cell]; paired_cell = paired[paired.cell == cell].set_index("seed")
        base = {"D": duration, "duration": duration, "h": h, "stride": stride, "cell": cell,
                "M": int(d.M.iloc[0]), "n_windows": int(d.M.iloc[0]),
                "h_over_D": h / duration, "n_planned": config["seed_count"]}
        for method, group in frame.groupby("method", sort=True):
            for metric in ("temporal_delta_auroc", "pooled_temporal_auroc", "union_auroc"):
                summary.append({**base, "method": method, "metric": metric, "comparison": "absolute",
                    "n_failed": int((group.fit_status != "ok").sum()), **interval(pd.to_numeric(group[metric], errors="coerce"), config)})
            summary.append({**base, "method": method, "metric": "temporal_delta_auroc", "comparison": "minus_raw",
                "n_failed": int((group.fit_status != "ok").sum()), **interval(paired_cell[method + "_minus_raw"], config)})
        accepted = [r["identity"]["seed"] for r in records if r["identity"]["cell"] == cell
                    and r["diagnostic"]["status"] == "ok" and r["diagnostic"]["fitting"].get("hard_gate_accepted", False)]
        observed = int((d.status == "ok").sum())
        gates.append({**base, "n_gate_observed": observed, "n_accepted": len(accepted),
            "acceptance_fraction": len(accepted) / observed if observed else np.nan,
            "n_accepted_evaluable": int(np.isfinite(paired_cell.loc[accepted, "forced_hard_minus_raw"]).sum()),
            "accepted_forced_minus_raw_mean": float(paired_cell.loc[accepted, "forced_hard_minus_raw"].mean()),
            "all_forced_minus_raw_mean": float(paired_cell.forced_hard_minus_raw.mean()),
            "all_native_minus_raw_mean": float(paired_cell.native_hard_minus_raw.mean())})
    return {"symmetric_sensitivity_detail.csv": detail, "symmetric_sensitivity_diagnostics.csv": diagnostic,
        "symmetric_sensitivity_paired.csv": paired, "symmetric_sensitivity_summary.csv": pd.DataFrame(summary),
        "symmetric_sensitivity_gate.csv": pd.DataFrame(gates)}


def aggregate(output: Path, config: dict) -> dict:
    tables = summarise(collect_records(output, config), config)
    for name, table in tables.items():
        table.to_csv(output / name, index=False, lineterminator="\n")
    seal = {"role": "post-review robustness; pointwise intervals and descriptive comparisons",
            "protocol_sha256": sha256(output / PROTOCOL_NAME),
            "sha256": {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*"))
                       if p.is_file() and p.name != "symmetric_sensitivity_outputs.json"}}
    write_json(output / "symmetric_sensitivity_outputs.json", seal)
    return tables


def iter_cases(root: Path = ROOT):
    """Yield pure-read adapters: identity, arrays, record, gate_accepted, fit_status.

    Downstream controls read case['arrays']['window_scores'] and the explicit
    emission_* arrays. This iterator never verifies by fitting or rewrites data.
    It retains failed records with gate_accepted=None instead of filtering them.
    """
    output = Path(root) / "results/information_sciences/symmetric_sensitivity"
    p = json.loads((output / PROTOCOL_NAME).read_text(encoding="utf-8"))
    for record in collect_records(output, p["configuration"]):
        identity = record["identity"]
        file = output / "parts" / f"repeat_{identity['seed']}_{identity['cell']}.npz"
        yield {"identity": identity, "arrays": load_arrays(file), "record": record,
               "gate_accepted": record["diagnostic"]["fitting"].get("hard_gate_accepted")
                  if record["diagnostic"]["status"] == "ok" else None,
               "fit_status": record["diagnostic"]["status"]}


def verify(output: Path = OUT, root: Path = ROOT) -> dict:
    """Recalculate retained emissions/metrics without generating series or fitting."""
    before = directory_hashes(output)
    p = json.loads((output / PROTOCOL_NAME).read_text(encoding="utf-8")); config = p["configuration"]
    for name, expected in p["source_sha256"].items():
        assert sha256(root / name) == expected, name
    assert sha256(root / "configs/symmetric_sensitivity.yaml") == p["configuration_sha256"]
    manifest = json.loads((output / "symmetric_sensitivity_outputs.json").read_text(encoding="utf-8"))
    assert manifest["protocol_sha256"] == before[PROTOCOL_NAME]
    assert set(manifest["sha256"]) == set(before) - {"symmetric_sensitivity_outputs.json"}
    for name, expected in manifest["sha256"].items():
        assert before[name] == expected, name
    records = collect_records(output, config); checked, failed, maximum_gap, maximum_rho = 0, 0, 0., 0.
    for seed in p["repeats"]:
        input_path = output / "inputs" / f"repeat_{seed}.npz"; inputs = load_arrays(input_path)
        metadata = json.loads(input_path.with_suffix(".json").read_text(encoding="utf-8"))
        assert metadata["sha256"] == sha256(input_path) and metadata["array_sha256"] == fingerprints(inputs)
        support = inputs["support"].astype(bool); shared = support[0] & support[1]
        replayed = draw_inputs(config, seed)
        for key in ("support", "coefficients_raw", "initial_values", "innovations"):
            np.testing.assert_array_equal(inputs[key], replayed[key])
        idx = np.arange(config["mechanism"]["n_nodes"])
        assert not support[:, :, idx, idx].any()
        np.testing.assert_array_equal(inputs["coefficients"][0][shared], inputs["coefficients"][1][shared])
        np.testing.assert_array_equal(inputs["coefficients"] != 0, support)
        scaled = lag_aware_scale_pair(inputs["coefficients_raw"], config["mechanism"]["target_radius"])
        for key in scaled:
            np.testing.assert_allclose(inputs[key], scaled[key], atol=2e-12, rtol=2e-11)
        maximum_rho = max(maximum_rho, float(inputs["rho_final"].max()))
        for duration in config["durations"]:
            series_path = output / "series" / f"repeat_{seed}_D{duration}.npz"; series = load_arrays(series_path)
            sm = json.loads(series_path.with_suffix(".json").read_text(encoding="utf-8"))
            assert sm["sha256"] == sha256(series_path) and sm["input_sha256"] == sha256(input_path)
            full = series["full_series"]; lag_count = config["mechanism"]["max_lag"]
            expected_labels = ((np.arange(len(full)) // duration) % 2).astype(np.int8)
            np.testing.assert_array_equal(series["regimes_full"], expected_labels)
            np.testing.assert_array_equal(full[:lag_count], inputs["initial_values"])
            assert bool(np.isfinite(full).all()) == sm["full_finite"]
            if sm["full_finite"]:
                expected = inputs["innovations"][lag_count:].copy()
                for lag in range(1, lag_count + 1):
                    expected += np.einsum("ti,tij->tj", full[lag_count - lag:len(full) - lag],
                        inputs["coefficients"][expected_labels[lag_count:], lag - 1])
                np.testing.assert_allclose(full[lag_count:], expected, atol=2e-15, rtol=2e-14)
            np.testing.assert_array_equal(series["observed_series"], full[config["mechanism"]["burn_in"]:])
    for record in records:
        identity = record["identity"]; seed = identity["seed"]; window = {"size": identity["h"], "stride": identity["stride"]}
        file = output / "parts" / f"repeat_{seed}_{identity['cell']}.npz"; a = load_arrays(file)
        input_path = output / "inputs" / f"repeat_{seed}.npz"; inputs = load_arrays(input_path)
        series_path = output / "series" / f"repeat_{seed}_D{identity['D']}.npz"; series = load_arrays(series_path)
        assert record["input_sha256"] == sha256(input_path) and record["series_sha256"] == sha256(series_path)
        for key, expected in (("coefficients", inputs["coefficients"]), ("state_support", inputs["support"]),
                              ("observed_series", series["observed_series"]), ("regimes", series["regimes"]),
                              ("bounds", bounds(config, window))):
            np.testing.assert_array_equal(a[key], expected)
        if record["diagnostic"]["status"] != "ok":
            failed += 1
            assert all(row["fit_status"] == "failed" and row["temporal_delta_auroc"] is None for row in record["methods"])
            continue
        dataset = type("RetainedTruth", (), {"regimes": a["regimes"], "state_support": a["state_support"],
                                            "union_support": a["state_support"].max(axis=0)})()
        truth = window_regime_labels(a["bounds"], a["regimes"])
        np.testing.assert_array_equal(a["true_window_labels"], truth)
        purity, _ = window_purity(a["bounds"], a["regimes"])
        np.testing.assert_allclose(a["window_purity"], purity, atol=0, rtol=0)
        maps, forced = common.maps_and_emission(a["window_scores"], a["candidate_labels"])
        oracle_maps, oracle = common.maps_and_emission(a["window_scores"], truth)
        np.testing.assert_array_equal(a["candidate_maps"], maps); np.testing.assert_array_equal(a["oracle_maps"], oracle_maps)
        features = a["window_scores"][:, nonself_mask(a["window_scores"].shape[1:])]
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        labels = a["candidate_labels"]
        inertia = sum(np.square(features[labels == k] - features[labels == k].mean(axis=0)).sum() for k in np.unique(labels))
        improvement = _state_improvement(features, float(inertia)); coherence = _temporal_coherence(labels); gate = STCGStateConfig()
        accepted = bool(len(np.unique(labels)) == 2 and (improvement >= gate.min_state_improvement or
            (improvement >= gate.min_temporal_state_improvement and coherence >= gate.min_temporal_coherence and
             _min_cluster_fraction(labels) >= gate.min_temporal_state_fraction)))
        assert accepted == record["diagnostic"]["fitting"]["hard_gate_accepted"]
        emissions = {"raw_window": a["window_scores"], "forced_hard": forced, "oracle_label": oracle,
            "static_mean": np.repeat(a["window_scores"].mean(axis=0)[None], len(truth), axis=0),
            "native_hard": forced if accepted else np.repeat(a["window_scores"].max(axis=0)[None], len(truth), axis=0)}
        for row in record["methods"]:
            method = row["method"]; np.testing.assert_array_equal(a["emission_" + method], emissions[method])
            recalculated = common.evaluate_predictions(emissions[method], a["bounds"], dataset)
            for metric, expected in recalculated.items():
                if isinstance(expected, (str, bool, int)):
                    assert row[metric] == expected, (identity, metric)
                else:
                    target = np.nan if row[metric] is None else row[metric]
                    np.testing.assert_allclose(target, expected, atol=2e-14, rtol=0, equal_nan=True)
            factor = common.hard_factors(maps, labels, a["bounds"], dataset) if method == "forced_hard" or (
                method == "native_hard" and accepted) else (common.hard_factors(oracle_maps, truth, a["bounds"], dataset)
                if method == "oracle_label" else {k: np.nan for k in ["J", "C_edge", "J_times_C", "identity_gap"]})
            for key, value in factor.items():
                np.testing.assert_allclose(np.nan if row[key] is None else row[key], value,
                                           atol=2e-14, rtol=0, equal_nan=True)
            checked += 1
        factor = common.hard_factors(maps, labels, a["bounds"], dataset)
        if np.isfinite(factor["identity_gap"]):
            maximum_gap = max(maximum_gap, factor["identity_gap"])
    tables = summarise(records, config)
    for name, expected in tables.items():
        actual = pd.read_csv(output / name)
        # Normalize CSV's empty-string/null convention and discarded index names.
        # The expected numbers still come from independently replayed case rows.
        expected = pd.read_csv(io.StringIO(expected.to_csv(index=False, lineterminator="\n")))
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=2e-13, rtol=2e-13)
    assert directory_hashes(output) == before, "Read-only verification changed an output"
    return {"passed": True, "role": p["role"], "independent_repeats": len(p["repeats"]),
        "planned_fits": len(records), "failed_fits": failed, "metric_rows_recomputed": checked,
        "maximum_final_radius": maximum_rho, "maximum_factor_identity_gap": maximum_gap,
        "new_simulations": 0, "new_fits": 0, "files_unchanged": len(before)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--log-dir", type=Path, default=LOGS)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--freeze", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--aggregate", action="store_true")
    modes.add_argument("--verify", action="store_true")
    args = parser.parse_args(); config = load_config(args.config); proposed = protocol(config, args.config)
    if args.freeze:
        print(json.dumps(freeze(args.output_dir, proposed, args.log_dir), indent=2)); return
    if not (args.run or args.aggregate or args.verify):
        print(json.dumps(proposed, indent=2)); return
    ensure_protocol(args.output_dir, proposed)
    with threadpool_limits(limits=1):
        if args.verify:
            print(json.dumps(verify(args.output_dir), indent=2)); return
        if args.run:
            start = perf_counter(); stage = {"pre_run_utc": utc_now(), "protocol_sha256": sha256(args.output_dir / PROTOCOL_NAME),
                "expected_fits": proposed["expected_fits"], "repeat_count": len(proposed["repeats"]),
                "mode": "fixed planned execution with checksum-verified resume; no optional sample extension"}
            args.log_dir.mkdir(parents=True, exist_ok=True)
            attempt = len(list(args.output_dir.glob("run_record_*.json"))) + 1
            record_path = args.output_dir / f"run_record_{attempt:03d}.json"
            write_json(record_path, stage)
            write_json(args.log_dir / f"symmetric_sensitivity_pre_run_{attempt:03d}.json", stage)
            environment = {"python": platform.python_version(), "packages": {name: importlib.metadata.version(name)
                for name in ["numpy", "pandas", "scipy", "scikit-learn", "PyYAML", "threadpoolctl"]}, "threads": 1}
            if not (args.output_dir / "environment.json").exists():
                write_json(args.output_dir / "environment.json", environment)
            for index, seed in enumerate(proposed["repeats"], 1):
                records = run_repeat(config, seed, args.output_dir)
                print(json.dumps({"repeat": seed, "completed_repeats": index, "cells": len(records),
                                  "elapsed_seconds": round(perf_counter() - start, 3)}), flush=True)
            stage.update(post_run_utc=utc_now(), wall_seconds=perf_counter() - start, execution_complete=True)
            write_json(record_path, stage)
        tables = aggregate(args.output_dir, config)
    print(tables["symmetric_sensitivity_gate.csv"].to_string(index=False))


if __name__ == "__main__":
    main()
