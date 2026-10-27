"""Frozen new-seed specialist stress grid and separately scheduled matched timing.

All writes use results/tables/revision_specialist_* by default. This controller
runs bounded, isolated one-thread worker processes and never imports or reuses
historical experiment caches. See docs/revision_specialist_protocol_2026-09-13.md.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PREFIX = "revision_specialist"
PROTOCOL_DOC = ROOT / "docs" / "revision_specialist_protocol_2026-09-13.md"
METHODS = ("STCG-v1", "DynVAR-Lasso", "VAR-Lasso", "Regime-PCMCI", "cMLP")
METRICS = ("edge_auc", "pair_auc", "precision_at_k", "shd", "lag_accuracy")
SEEDS = tuple(range(100, 120))
THREAD_ENV = {
    name: "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
        "LOKY_MAX_CPU_COUNT",
    )
}
SOURCE_FILES = (
    "scripts/run_revision_specialist_stress.py", "src/stcg/baselines.py",
    "src/stcg/self_supervised.py", "src/stcg/synthetic.py",
    "src/stcg/neural_granger.py", "src/stcg/metrics.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def json_safe(value):
    """Undefined diagnostics are null; they do not turn valid fits into failures."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def finite_metric(row: dict, metric: str) -> bool:
    value = row.get(metric)
    return value is not None and math.isfinite(float(value))


def function_ast_hashes() -> dict[str, str]:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    names = ("generate_dataset", "fit_method", "make_grid", "fixed_spec", "paired_statistics")
    return {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names}


def make_grid() -> list[dict]:
    """Preserve 15 displayed cells and deduplicate only exact experiment aliases."""
    cells = []
    axes = (("obs_noise", (0.0, 0.05, 0.10, 0.20)),
            ("timesteps", (300, 600, 1000, 1500)),
            ("edge_prob", (0.10, 0.18, 0.30, 0.45)),
            ("switch_period", (150, 300, 600)))
    for axis, values in axes:
        for value in values:
            periodic = axis == "switch_period"
            timesteps = int(value) if axis == "timesteps" else (1200 if periodic else 1000)
            window = 300 if periodic else min(300, max(120, timesteps // 2))
            config = {
                "n_nodes": 8, "max_lag": 3, "timesteps": timesteps,
                "edge_prob": float(value) if axis == "edge_prob" else 0.18,
                "noise_scale": 0.10,
                "observation_noise": float(value) if axis == "obs_noise" else 0.0,
                "switch_period": int(value) if periodic else None,
                "window_size": window, "window_stride": window // 2,
                "state_seed_offset": 90000 if periodic else 50000,
            }
            cells.append({"axis": axis, "value": value, "setting_id": digest(config)[:12], "config": config})
    return cells


def fixed_spec() -> dict:
    return {
        "schema": 1, "seeds": list(SEEDS), "methods": list(METHODS),
        "display_cells": make_grid(), "timeout_seconds": 3600,
        "thread_environment": THREAD_ENV, "dispatch_seed": 20260913,
        "bootstrap_seed": 20260913, "bootstrap_samples": 10000,
        "stcg": {"n_states": 2, "auto_states": True, "lasso_alpha": 0.005,
                 "state_aggregate": "mean", "aggregate": "max", "contrast_mix": 0.0},
        "rpcmci": {"num_regimes": 2, "num_iterations": 20, "max_anneal": 10,
                   "max_transitions": 10, "switch_thres": 0.05, "pc_alpha": 0.05,
                   "alpha_level": 0.01, "score": "statistic", "n_jobs": 1},
        "cmlp": {"hidden": 8, "penalty": 0.15, "lr": 0.05, "epochs": 300,
                 "activation": "tanh", "optimizer": "sgd", "momentum": 0.9, "weight_decay": 0.0},
        "source_hashes": source_hashes(),
        "protocol_document_sha256": hashlib.sha256(PROTOCOL_DOC.read_bytes()).hexdigest(),
    }


def protocol_path(output_dir: Path) -> Path:
    return output_dir / f"{PREFIX}_protocol.json"


def freeze(output_dir: Path) -> dict:
    path = protocol_path(output_dir)
    specification = fixed_spec()
    if path.exists():
        return load_protocol(output_dir)
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "tigramite", "ortools", "threadpoolctl"):
        versions[package] = importlib.metadata.version(package)
    frozen = {"frozen_at_utc": utc_now(), "protocol_hash": digest(specification),
              "specification": specification,
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                              "dependencies": versions}}
    atomic_json(path, frozen)
    return frozen


def load_protocol(output_dir: Path) -> dict:
    path = protocol_path(output_dir)
    if not path.exists():
        raise ValueError("Freeze the protocol with --freeze before fitting any new data.")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    current = fixed_spec()
    if frozen["specification"] != current:
        # One explicit storage/analysis-only amendment permits already successful
        # fits to survive a JSON NaN recording bug. Estimation, generation, grid,
        # all parameters, metrics source, and paired tests must remain identical.
        amendment_path = output_dir / f"{PREFIX}_storage_amendment.json"
        if not amendment_path.exists():
            raise ValueError("Frozen protocol/source fingerprint mismatch; refusing to mix changed code and cached fits.")
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        runner_name = "scripts/run_revision_specialist_stress.py"
        original_hash = frozen["specification"]["source_hashes"][runner_name]
        if (amendment.get("old_runner_sha256") != original_hash
                or amendment.get("new_runner_sha256") != current["source_hashes"][runner_name]
                or amendment.get("frozen_function_ast_hashes") != function_ast_hashes()):
            raise ValueError("Runner change is not the explicitly approved estimation-identical storage amendment")
        current["source_hashes"][runner_name] = original_hash
        if current != frozen["specification"]:
            raise ValueError("Storage amendment cannot authorize estimator, input, budget, seed, or protocol changes")
    return frozen


def unique_settings(cells: list[dict]) -> dict[str, dict]:
    return {cell["setting_id"]: cell["config"] for cell in cells}


def base_setting(cells: list[dict]) -> str:
    return next(cell["setting_id"] for cell in cells if cell["axis"] == "obs_noise" and cell["value"] == 0)


def make_jobs(frozen: dict, phase: str, calibrate: bool = False) -> list[dict]:
    spec = frozen["specification"]
    settings = unique_settings(spec["display_cells"])
    if phase == "timing" or calibrate:
        key = base_setting(spec["display_cells"])
        settings = {key: settings[key]}
    seeds = [spec["seeds"][0]] if calibrate else spec["seeds"]
    jobs = [{"phase": phase, "setting_id": setting_id, "config": config,
             "seed": seed, "method": method, "protocol_hash": frozen["protocol_hash"]}
            for setting_id, config in sorted(settings.items())
            for seed in seeds for method in spec["methods"]]
    random.Random(spec["dispatch_seed"]).shuffle(jobs)
    return jobs


def job_key(job: dict) -> str:
    method = job["method"].replace("-", "").lower()
    return f"{job['phase']}_{job['setting_id']}_seed{job['seed']:03d}_{method}"


def result_path(output_dir: Path, job: dict) -> Path:
    return output_dir / f"{PREFIX}_parts" / (job_key(job) + ".json")


def valid_cached(path: Path, job: dict) -> dict | None:
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    for key in ("protocol_hash", "phase", "setting_id", "seed", "method", "config"):
        if row.get(key) != job[key]:
            raise ValueError(f"Cached {path.name} disagrees on {key}")
    if row.get("status") not in {"ok", "failed", "timeout", "crashed"}:
        raise ValueError(f"Incomplete cache record: {path}")
    return row


def array_hash(array) -> str:
    import numpy as np
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(str((contiguous.shape, contiguous.dtype.str)).encode() + contiguous.tobytes()).hexdigest()


def generate_dataset(config: dict, seed: int):
    from dataclasses import replace
    import numpy as np
    from stcg import generate_lagged_var
    dataset = generate_lagged_var(
        n_nodes=config["n_nodes"], timesteps=config["timesteps"], max_lag=config["max_lag"],
        edge_prob=config["edge_prob"], noise_scale=config["noise_scale"], seed=seed,
        switching=True, switch_period=config["switch_period"],
    )
    ratio = config["observation_noise"]
    if ratio > 0:
        noise = np.random.default_rng(seed + 60000).normal(
            0.0, ratio * float(np.std(dataset.x)), size=dataset.x.shape)
        dataset = replace(dataset, x=dataset.x + noise)
    return dataset


def fit_method(method: str, x, config: dict, seed: int, spec: dict):
    """Called wholly inside timing; wrappers retain their native standardization."""
    from stcg import (STCGStateConfig, stcg_v1_diagnostics, dynamic_lasso_var_scores,
                      lasso_var_scores, rpcmci_scores, cmlp_scores)
    lag = config["max_lag"]
    if method == "STCG-v1":
        state_config = STCGStateConfig(
            **spec["stcg"], window_size=config["window_size"], stride=config["window_stride"],
            seed=seed + config["state_seed_offset"],
        )
        info = stcg_v1_diagnostics(x, lag, state_config)
        return info.scores, info
    if method == "DynVAR-Lasso":
        return dynamic_lasso_var_scores(x, lag, alpha=0.005, window_size=config["window_size"],
                                        stride=config["window_stride"], aggregate="max"), None
    if method == "VAR-Lasso":
        return lasso_var_scores(x, lag, alpha=0.005), None
    if method == "Regime-PCMCI":
        return rpcmci_scores(x, lag, **spec["rpcmci"], seed=seed, return_info=True)
    if method == "cMLP":
        return cmlp_scores(x, lag, **spec["cmlp"], seed=seed, return_info=True)
    raise ValueError(method)


def worker(output_dir: Path, job: dict) -> None:
    os.environ.update(THREAD_ENV)
    os.environ["PYTHONHASHSEED"] = "0"
    import numpy as np
    import torch
    import sklearn.cluster
    import sklearn.linear_model
    import tigramite.rpcmci
    from sklearn.metrics import adjusted_rand_score
    from threadpoolctl import threadpool_limits, threadpool_info
    from stcg import recovery_summary

    frozen = load_protocol(output_dir)
    if frozen["protocol_hash"] != job["protocol_hash"]:
        raise ValueError("Worker job protocol does not match manifest")
    dataset = generate_dataset(job["config"], job["seed"])
    row = dict(job, started_at_utc=utc_now(), worker_pid=os.getpid(),
               input_sha256=array_hash(dataset.x), truth_sha256=array_hash(dataset.edge_truth),
               thread_environment=dict(THREAD_ENV), execution_workers=job["execution_workers"],
               exclusive_timing=job["exclusive_timing"],
               recording_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        row["threadpools"] = [{k: pool.get(k) for k in ("user_api", "internal_api", "num_threads")}
                              for pool in threadpool_info()]
        cpu_started = time.process_time()
        started = time.perf_counter()
        try:
            scores, info = fit_method(job["method"], np.array(dataset.x, dtype=float, copy=True),
                                      job["config"], job["seed"], frozen["specification"])
            fit_seconds = time.perf_counter() - started
            fit_cpu_seconds = time.process_time() - cpu_started
            if scores.shape != dataset.edge_truth.shape or not np.isfinite(scores).all():
                raise ValueError("Invalid/nonfinite output scores")
            row.update(status="ok", fit_seconds=fit_seconds, fit_cpu_seconds=fit_cpu_seconds,
                       scores_sha256=array_hash(scores))
            row.update({key: float(value) for key, value in recovery_summary(scores, dataset.edge_truth).items()})
            row["undefined_metrics"] = [metric for metric in METRICS if not finite_metric(row, metric)]
            if job["method"] == "STCG-v1":
                truth = [int(np.bincount(dataset.regimes[start:end]).argmax()) for start, end in info.bounds]
                row.update(state_count=int(np.unique(info.labels).size),
                           states_accepted=bool(info.states_accepted),
                           state_improvement=float(info.state_improvement),
                           temporal_coherence=float(info.temporal_coherence),
                           window_regime_ari=float(adjusted_rand_score(truth, info.labels)))
            if job["method"] == "Regime-PCMCI":
                offset = int(info["effective_offset"])
                seq = np.asarray(info["regime_sequence"])
                predicted = seq[offset:] if len(seq) == len(dataset.regimes) else seq
                truth = dataset.regimes[offset:offset + len(predicted)]
                row.update(error_free_annealings=int(info["error_free_annealings"]),
                           sample_regime_ari=float(adjusted_rand_score(truth, predicted)))
        except Exception as exc:
            row.update(status="failed", fit_seconds=time.perf_counter() - started,
                       fit_cpu_seconds=time.process_time() - cpu_started,
                       error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            for metric in METRICS:
                row[metric] = None
    row["finished_at_utc"] = utc_now()
    atomic_json(result_path(output_dir, job), row)
    print(json.dumps({"job": job_key(job), "status": row["status"], "fit_seconds": row["fit_seconds"]}), flush=True)


def run_controller(output_dir: Path, phase: str, workers: int, calibrate: bool, exclusive: bool) -> dict:
    if workers < 1 or workers > 8:
        raise ValueError("Use 1--8 explicitly bounded workers")
    if phase == "timing" and (workers != 1 or not exclusive):
        raise ValueError("Dedicated timing requires --workers 1 --exclusive-timing on a quiet machine")
    frozen = load_protocol(output_dir)
    jobs = make_jobs(frozen, phase, calibrate)
    for job in jobs:
        job.update(execution_workers=workers, exclusive_timing=exclusive)
    pending = [job for job in jobs if valid_cached(result_path(output_dir, job), job) is None]
    logs = output_dir / f"{PREFIX}_logs"
    logs.mkdir(parents=True, exist_ok=True)
    active = []
    completed = len(jobs) - len(pending)
    status_path = output_dir / f"{PREFIX}_{phase}_status.json"
    started = time.perf_counter()
    last_report = 0.0
    try:
        while pending or active:
            while pending and len(active) < workers:
                job = pending.pop(0)
                logfile = (logs / (job_key(job) + ".log")).open("w", encoding="utf-8")
                command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker-job",
                           json.dumps(job, separators=(",", ":")), "--output-dir", str(output_dir)]
                env = dict(os.environ, **THREAD_ENV, PYTHONHASHSEED="0")
                proc = subprocess.Popen(command, cwd=ROOT, stdout=logfile, stderr=subprocess.STDOUT,
                                        env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                active.append({"job": job, "process": proc, "log": logfile, "started": time.perf_counter()})
            for item in list(active):
                proc, job = item["process"], item["job"]
                elapsed = time.perf_counter() - item["started"]
                timeout = elapsed > frozen["specification"]["timeout_seconds"]
                if timeout and proc.poll() is None:
                    proc.kill()
                    proc.wait()
                if proc.poll() is not None:
                    item["log"].close()
                    path = result_path(output_dir, job)
                    if not path.exists():
                        failed = dict(job, status="timeout" if timeout else "crashed", exit_code=proc.returncode,
                                      dispatch_seconds=elapsed, error="See retained worker log", finished_at_utc=utc_now())
                        failed.update({metric: None for metric in METRICS})
                        atomic_json(path, failed)
                    valid_cached(path, job)
                    active.remove(item)
                    completed += 1
            if time.perf_counter() - last_report >= 5 or not (pending or active):
                status = {"phase": phase, "controller_pid": os.getpid(), "protocol_hash": frozen["protocol_hash"],
                          "planned_jobs": len(jobs), "completed_jobs": completed, "pending_jobs": len(pending),
                          "workers": workers, "elapsed_seconds": time.perf_counter() - started,
                          "updated_at_utc": utc_now(), "calibration": calibrate,
                          "active": [{"pid": x["process"].pid, "job": job_key(x["job"]),
                                      "elapsed_seconds": time.perf_counter() - x["started"]} for x in active]}
                atomic_json(status_path, status)
                if time.perf_counter() - last_report >= 60 or completed == len(jobs):
                    print(json.dumps(status), flush=True)
                last_report = time.perf_counter()
            time.sleep(0.2)
    finally:
        for item in active:
            if item["process"].poll() is None:
                item["process"].terminate()
                item["process"].wait()
            item["log"].close()
    return {"phase": phase, "jobs": len(jobs), "completed": completed}


def holm_adjust(p_values: list[float]) -> list[float]:
    ordered = sorted(range(len(p_values)), key=lambda i: p_values[i])
    output = [0.0] * len(p_values)
    bound = 0.0
    for rank, index in enumerate(ordered):
        bound = max(bound, min(1.0, (len(p_values) - rank) * p_values[index]))
        output[index] = bound
    return output


def paired_statistics(a: list[float], b: list[float], metric: str, seed: int = 20260913) -> dict:
    import numpy as np
    from scipy import stats
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(x) != len(y) or len(x) < 2 or not np.isfinite(np.r_[x, y]).all():
        raise ValueError("At least two finite, identically paired values are required")
    diff = (y - x) if metric == "shd" else (x - y)
    draws = np.random.default_rng(seed).choice(diff, size=(10000, len(diff)), replace=True).mean(axis=1)
    all_zero = bool(np.all(diff == 0))
    return {"n": len(x), "mean_stcg": float(x.mean()), "mean_opponent": float(y.mean()),
            "mean_improvement": float(diff.mean()), "sd_improvement": float(diff.std(ddof=1)),
            "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
            "paired_t_two_sided_p": 1.0 if all_zero else float(stats.ttest_1samp(diff, 0).pvalue),
            "wilcoxon_two_sided_p": 1.0 if all_zero else float(stats.wilcoxon(diff).pvalue),
            "wins": int(np.sum(diff > 1e-12)), "ties": int(np.sum(np.abs(diff) <= 1e-12)),
            "losses": int(np.sum(diff < -1e-12))}


def paired_cost_ratio(stcg: list[float], opponent: list[float]) -> dict:
    import numpy as np
    a, b = np.asarray(stcg), np.asarray(opponent)
    if len(a) < 2 or len(a) != len(b) or np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("Positive paired timings required")
    log_ratio = np.log(b / a)
    boot = np.random.default_rng(20260913).choice(log_ratio, size=(10000, len(a)), replace=True).mean(axis=1)
    return {"geometric_mean_opponent_over_stcg": float(np.exp(log_ratio.mean())),
            "ratio_ci_low": float(np.exp(np.quantile(boot, 0.025))),
            "ratio_ci_high": float(np.exp(np.quantile(boot, 0.975)))}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def aggregate_results(output_dir: Path) -> dict:
    import numpy as np
    frozen = load_protocol(output_dir)
    spec = frozen["specification"]
    cells = spec["display_cells"]
    all_rows, coverage, summaries, pairs, runtimes = [], [], [], [], []
    for phase in ("stress", "timing"):
        jobs = make_jobs(frozen, phase)
        by_setting = defaultdict(list)
        for job in jobs:
            row = valid_cached(result_path(output_dir, job), job)
            if row:
                all_rows.append(row)
                by_setting[job["setting_id"]].append(row)
        relevant = unique_settings(cells) if phase == "stress" else {base_setting(cells): {}}
        for setting_id in relevant:
            rows = by_setting[setting_id]
            successes = {method: {int(r["seed"]): r for r in rows if r["method"] == method and r["status"] == "ok"}
                         for method in METHODS}
            common = sorted(set.intersection(*(set(group) for group in successes.values())))
            for seed in common:
                if len({group[seed]["input_sha256"] for group in successes.values()}) != 1:
                    raise ValueError("Methods did not receive identical input arrays")
            for method in METHODS:
                attempted = [r for r in rows if r["method"] == method]
                cov = {"phase": phase, "setting_id": setting_id, "method": method,
                       "planned": len(SEEDS), "attempted": len(attempted), "successful": len(successes[method]),
                       "failed": sum(r["status"] != "ok" for r in attempted), "common_n": len(common),
                       "common_seeds": common,
                       "failed_seeds": [r["seed"] for r in attempted if r["status"] != "ok"]}
                cov["defined_counts_by_metric"] = {
                    metric: sum(finite_metric(r, metric) for r in successes[method].values()) for metric in METRICS}
                coverage.append(cov)
                if successes[method]:
                    summary = dict(cov)
                    for metric in (*METRICS, "fit_seconds", "fit_cpu_seconds"):
                        values = [r[metric] for r in successes[method].values() if finite_metric(r, metric)]
                        summary[metric + "_n"] = len(values)
                        summary[metric + "_mean"] = float(np.mean(values)) if values else None
                        summary[metric + "_std"] = (float(np.std(values, ddof=1)) if len(values) > 1
                                                    else (0.0 if values else None))
                    summaries.append(summary)
            if len(common) < 2:
                continue
            for opponent in METHODS[1:]:
                for metric in METRICS:
                    metric_common = [s for s in common if all(finite_metric(group[s], metric) for group in successes.values())]
                    a = [successes["STCG-v1"][s][metric] for s in metric_common]
                    b = [successes[opponent][s][metric] for s in metric_common]
                    pair = {"phase": phase, "setting_id": setting_id, "opponent": opponent,
                            "metric": metric, "common_seeds": metric_common,
                            "successful_fit_common_n": len(common),
                            "undefined_metric_seeds": [s for s in common if s not in metric_common]}
                    if len(metric_common) >= 2:
                        pair.update(statistics_status="ok", **paired_statistics(a, b, metric))
                    else:
                        pair.update(n=len(metric_common), statistics_status="fewer than two defined common observations",
                                    mean_stcg=float(np.mean(a)) if a else None,
                                    mean_opponent=float(np.mean(b)) if b else None)
                    pairs.append(pair)
                for metric in ("fit_seconds", "fit_cpu_seconds"):
                    a = [successes["STCG-v1"][s][metric] for s in common]
                    b = [successes[opponent][s][metric] for s in common]
                    runtimes.append({"phase": phase, "setting_id": setting_id, "opponent": opponent,
                                     "metric": metric, "n": len(common),
                                     "dedicated_exclusive_phase": phase == "timing",
                                     **paired_cost_ratio(a, b)})
    for p_column in ("paired_t_two_sided_p", "wilcoxon_two_sided_p"):
        primary = [r for r in pairs if r["phase"] == "stress" and r["metric"] == "edge_auc"
                   and r["opponent"] == "Regime-PCMCI" and r["statistics_status"] == "ok"]
        broader = [r for r in pairs if r["phase"] == "stress" and r["metric"] == "edge_auc" and r["statistics_status"] == "ok"]
        # Missing setting tests receive p=1 in the frozen-size family, so partial
        # aggregation cannot make a small interim subset look more significant.
        for selected, size, suffix in ((primary, 13, "_holm13"), (broader, 52, "_holm52")):
            adjusted = holm_adjust([r[p_column] for r in selected] + [1.0] * (size - len(selected)))
            for row, value in zip(selected, adjusted):
                row[p_column + suffix] = value
    displayed = []
    for cell in cells:
        for row in summaries:
            if row["phase"] == "stress" and row["setting_id"] == cell["setting_id"]:
                displayed.append({"axis": cell["axis"], "value": cell["value"], **row})
    write_csv(output_dir / f"{PREFIX}_detail.csv", all_rows)
    write_csv(output_dir / f"{PREFIX}_coverage.csv", coverage)
    write_csv(output_dir / f"{PREFIX}_summary.csv", displayed)
    write_csv(output_dir / f"{PREFIX}_paired.csv", pairs)
    write_csv(output_dir / f"{PREFIX}_timing.csv", runtimes)
    lines = ["# Specialist stress extension: frozen new seeds 100--119", "",
             "15 displayed settings, 13 unique inputs per seed; fixed method budgets and supplied K=2. "
             "Every axis is reported. Pairwise tests use the same complete-case seeds across all five methods; "
             "coverage records expose failures and unfinished jobs. Missing primary tests count as p=1 in Holm families.", "",
             "Stress-phase time ratios were collected under concurrent load and are descriptive. "
             "Only the separately scheduled timing phase supports a matched quiet-machine latency report. "
             "There is no equivalence margin, post-hoc practical-gain cutoff, or winner selection.", "",
             "| Axis | Value | Common n | STCG Edge AUC | Regime-PCMCI | Difference [95% CI] | Two-sided t Holm(13) | Two-sided W Holm(13) |",
             "|---|---:|---:|---:|---:|---|---:|---:|"]
    for cell in cells:
        row = next((r for r in pairs if r["phase"] == "stress" and r["setting_id"] == cell["setting_id"]
                    and r["opponent"] == "Regime-PCMCI" and r["metric"] == "edge_auc"), None)
        if row and row["statistics_status"] == "ok":
            lines.append(f"| {cell['axis']} | {cell['value']} | {row['n']} | {row['mean_stcg']:.4f} | "
                         f"{row['mean_opponent']:.4f} | {row['mean_improvement']:+.4f} "
                         f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
                         f"{row['paired_t_two_sided_p_holm13']:.4g} | {row['wilcoxon_two_sided_p_holm13']:.4g} |")
        else:
            lines.append(f"| {cell['axis']} | {cell['value']} | pending | -- | -- | -- | -- | -- |")
    lines += ["", "## cMLP and window/global comparators", "",
              "All 4 opponents x 5 recovery metrics appear in the paired CSV. "
              "The 52-test Holm columns cover all Edge-AUC opponents and unique settings. "
              "Secondary metrics are exploratory; sign-reversed SHD improvements mean fewer errors.", "",
              "## Coverage and completion", "",
              f"Stored fits: {len(all_rows)}; successful: {sum(r['status'] == 'ok' for r in all_rows)}. "
              f"Frozen full plan: {13 * 20 * 5} stress fits and {20 * 5} separate timing fits.", "",
              "Use the coverage CSV's planned, attempted, successful, failed, common_n and failed_seeds "
              "columns before interpreting any effect. A failed specialist is not a performance win."]
    (output_dir / f"{PREFIX}_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"stored_fits": len(all_rows), "paired_rows": len(pairs), "summary_rows": len(displayed), "timing_rows": len(runtimes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--phase", choices=("stress", "timing"), default="stress")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--exclusive-timing", action="store_true")
    parser.add_argument("--worker-job", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_job:
        worker(args.output_dir, json.loads(args.worker_job))
        return
    if args.freeze:
        result = freeze(args.output_dir)
        print(json.dumps({"protocol_hash": result["protocol_hash"], "frozen_at_utc": result["frozen_at_utc"]}), flush=True)
    if args.run or args.calibrate:
        print(json.dumps(run_controller(args.output_dir, args.phase, 1 if args.calibrate else args.workers,
                                        args.calibrate, args.exclusive_timing)), flush=True)
        print(json.dumps(aggregate_results(args.output_dir)), flush=True)
    elif args.aggregate:
        print(json.dumps(aggregate_results(args.output_dir)), flush=True)
    elif not args.freeze:
        parser.error("Choose --freeze, --calibrate, --run or --aggregate")


if __name__ == "__main__":
    main()
