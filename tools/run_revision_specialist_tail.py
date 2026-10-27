"""Finish the remaining frozen specialist fits without a controller.

Scheduling-only tail driver for interactive terminals (no harness timeout).
It re-enumerates pending jobs from the frozen protocol, runs the stress tail
with a few workers, then the remaining dedicated-timing fits strictly
sequentially on a quiet machine, and finally aggregates. Estimation,
generation, metrics and per-fit budgets are the fingerprinted runner's,
unchanged; this file only replaces the drained/blocked controllers.

Usage (from repo root, venv python):
    python tools/run_revision_specialist_tail.py --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "revision_specialist_runner", ROOT / "scripts/run_revision_specialist_stress.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def pending_jobs(output_dir: Path, phase: str, workers: int, exclusive: bool) -> list[dict]:
    frozen = runner.load_protocol(output_dir)
    jobs = runner.make_jobs(frozen, phase)
    for job in jobs:
        job.update(execution_workers=workers, exclusive_timing=exclusive)
    return [job for job in jobs
            if runner.valid_cached(runner.result_path(output_dir, job), job) is None]


def run_one(output_dir: Path, job: dict) -> dict:
    command = [sys.executable, "-B", str(ROOT / "scripts/run_revision_specialist_stress.py"),
               "--worker-job", json.dumps(job, separators=(",", ":")),
               "--output-dir", str(output_dir)]
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                          env=dict(os.environ, **runner.THREAD_ENV, PYTHONHASHSEED="0"))
    elapsed = time.perf_counter() - started
    key = runner.job_key(job)
    row = runner.valid_cached(runner.result_path(output_dir, job), job)
    if row is None:
        raise RuntimeError(f"{key} produced no valid record after {elapsed:.0f}s: {proc.stdout[-500:]}")
    print(f"[{key}] status={row['status']} fit={row.get('fit_seconds')}s wall={elapsed:.0f}s", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    stress = pending_jobs(output_dir, "stress", args.workers, False)
    timing = pending_jobs(output_dir, "timing", 1, True)
    print(f"pending: stress={len(stress)} timing={len(timing)}", flush=True)

    amendment = {
        "recorded_at_utc": runner.utc_now(),
        "amendment_type": "Scheduling only: terminal tail driver, no harness timeout",
        "stress_workers": args.workers,
        "timing_workers": 1,
        "thread_environment": runner.THREAD_ENV,
        "estimation_runner_sha256": hashlib.sha256(
            (ROOT / "scripts/run_revision_specialist_stress.py").read_bytes()).hexdigest(),
        "tail_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pending_at_start": {"stress": len(stress), "timing": len(timing)},
        "reason": "Harness calls cap at ~290s; drained controller blocked by a recycled-PID "
                  "liveness false positive. Fits run as the same fingerprinted worker with "
                  "unchanged protocol, budgets, per-fit threads and timing boundaries.",
        "unchanged": "Frozen protocol, inputs, seeds, method parameters/budgets, metrics, "
                     "statistics and the single-worker exclusive timing phase",
    }
    runner.atomic_json(output_dir / "revision_specialist_tail_amendment.json", amendment)

    t0 = time.perf_counter()
    # Stress tail: a few at a time; these are the slow Regime-PCMCI fits.
    queue = list(stress)
    while queue:
        batch, queue = queue[: args.workers], queue[args.workers :]
        procs = []
        for job in batch:
            logpath = output_dir / "revision_specialist_logs" / (runner.job_key(job) + ".tail.log")
            logpath.parent.mkdir(exist_ok=True)
            log = logpath.open("w", encoding="utf-8")
            procs.append((subprocess.Popen(
                [sys.executable, "-B", str(ROOT / "scripts/run_revision_specialist_stress.py"),
                 "--worker-job", json.dumps(job, separators=(",", ":")),
                 "--output-dir", str(output_dir)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, **runner.THREAD_ENV, PYTHONHASHSEED="0")), job, log))
        for proc, job, log in procs:
            proc.wait()
            log.close()
            row = runner.valid_cached(runner.result_path(output_dir, job), job)
            print(f"[{runner.job_key(job)}] status={row['status'] if row else 'MISSING'}", flush=True)

    # Timing tail: strictly sequential, quiet machine.
    for job in timing:
        run_one(output_dir, job)

    print(json.dumps(runner.aggregate_results(output_dir)), flush=True)
    print(f"tail finished in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
