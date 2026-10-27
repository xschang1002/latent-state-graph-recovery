"""Scheduling-only extension: preserve frozen workers, allow up to 16 fits.

This controller is separate from the fingerprinted estimation/recording runner.
It refuses to overlap the drained controller or any of its surviving workers.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("revision_specialist_runner", ROOT / "scripts/run_revision_specialist_stress.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def process_active(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # PID no longer exists.
            return False
        raise OSError(error, f"Cannot establish whether prior process {pid} is alive")
    try:
        code = ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise OSError(ctypes.get_last_error(), "GetExitCodeProcess failed")
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


def verify_drain(output_dir: Path) -> dict:
    path = output_dir / "revision_specialist_schedule_drain.json"
    drain = json.loads(path.read_text(encoding="utf-8"))
    pids = [drain["old_controller_pid"], *(r["pid"] for r in drain["running_children_to_drain"])]
    active = [pid for pid in pids if process_active(pid)]
    if active:
        raise RuntimeError(f"Wait for previous controller/workers to exit: {active}")
    # Windows venv python.exe can launch a second interpreter: Popen.pid is
    # then the launcher PID, while worker_pid records the actual interpreter.
    # The stable pre-stop status maps each enumerated launcher to its job key.
    mapping = {r["pid"]: r["job"] for r in drain["old_status"]["active"]}
    expected = [mapping[r["pid"]] for r in drain["running_children_to_drain"]]
    completed = {}
    for key in expected:
        path = output_dir / "revision_specialist_parts" / (key + ".json")
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if runner.job_key(row) != key or row.get("status") not in {"ok", "failed", "crashed", "timeout"}:
                raise ValueError(f"Invalid prior-worker completion: {key}")
            if row.get("worker_pid") is not None and process_active(row["worker_pid"]):
                raise RuntimeError(f"Wait for previous interpreter to exit: {row['worker_pid']}")
            completed[key] = row
    missing = [key for key in expected if key not in completed]
    if missing:
        raise RuntimeError(f"Prior workers have no retained completion record; investigate logs: {missing}")
    return {"prior_controller": drain["old_controller_pid"],
            "preserved_worker_completions": expected}


def run(output_dir: Path, workers: int) -> dict:
    if not 1 <= workers <= 16:
        raise ValueError("Scheduling extension permits 1--16 isolated one-thread workers")
    drained = verify_drain(output_dir)
    frozen = runner.load_protocol(output_dir)
    jobs = runner.make_jobs(frozen, "stress")
    for job in jobs:
        job.update(execution_workers=workers, exclusive_timing=False)
    pending = [job for job in jobs if runner.valid_cached(runner.result_path(output_dir, job), job) is None]
    provenance = {
        "recorded_at_utc": runner.utc_now(), "protocol_hash": frozen["protocol_hash"],
        "amendment_type": "Scheduling only after natural completion of all previous active fits",
        "workers": workers, "thread_environment": runner.THREAD_ENV,
        "estimation_runner_sha256": hashlib.sha256((ROOT / "scripts/run_revision_specialist_stress.py").read_bytes()).hexdigest(),
        "scheduling_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "already_completed_fits": len(jobs) - len(pending), "drain_verification": drained,
        "reason": "Other heavy experiments completed; parent authorized up to 16 workers on 24 logical CPUs",
        "unchanged": "Frozen protocol, all input generation, seeds, method parameters/budgets, per-fit threads, timing boundary, timeout, metrics and statistical procedures",
        "timing_limit": "These stress times remain concurrent-load descriptions; dedicated timing still requires one worker on a quiet machine",
    }
    runner.atomic_json(output_dir / "revision_specialist_schedule_16_amendment.json", provenance)
    logs = output_dir / "revision_specialist_logs"
    logs.mkdir(exist_ok=True)
    active, completed = [], len(jobs) - len(pending)
    started, last_status, last_stdout = time.perf_counter(), 0.0, 0.0
    try:
        while pending or active:
            while pending and len(active) < workers:
                job = pending.pop(0)
                logfile = (logs / (runner.job_key(job) + ".log")).open("w", encoding="utf-8")
                command = [sys.executable, "-B", str(ROOT / "scripts/run_revision_specialist_stress.py"),
                           "--worker-job", json.dumps(job, separators=(",", ":")), "--output-dir", str(output_dir)]
                proc = subprocess.Popen(command, cwd=ROOT, stdout=logfile, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, **runner.THREAD_ENV, PYTHONHASHSEED="0"),
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
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
                    path = runner.result_path(output_dir, job)
                    if not path.exists():
                        failed = dict(job, status="timeout" if timeout else "crashed", exit_code=proc.returncode,
                                      dispatch_seconds=elapsed, error="See retained worker log", finished_at_utc=runner.utc_now())
                        failed.update({metric: None for metric in runner.METRICS})
                        runner.atomic_json(path, failed)
                    runner.valid_cached(path, job)
                    active.remove(item)
                    completed += 1
            now = time.perf_counter()
            if now - last_status >= 5 or not (pending or active):
                status = {"phase": "stress", "controller_pid": os.getpid(), "protocol_hash": frozen["protocol_hash"],
                          "planned_jobs": len(jobs), "completed_jobs": completed, "pending_jobs": len(pending),
                          "workers": workers, "elapsed_seconds": now - started, "updated_at_utc": runner.utc_now(),
                          "scheduling_tool": "tools/run_revision_specialist_schedule.py",
                          "active": [{"pid": x["process"].pid, "job": runner.job_key(x["job"]),
                                      "elapsed_seconds": now - x["started"]} for x in active]}
                runner.atomic_json(output_dir / "revision_specialist_stress_status.json", status)
                if now - last_stdout >= 60 or completed == len(jobs):
                    print(json.dumps(status), flush=True)
                    last_stdout = now
                last_status = now
            time.sleep(0.2)
    finally:
        for item in active:
            if item["process"].poll() is None:
                item["process"].terminate()
                item["process"].wait()
            item["log"].close()
    return {"phase": "stress", "jobs": len(jobs), "completed": completed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/tables")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--check-drain", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_drain(args.output_dir) if args.check_drain else run(args.output_dir, args.workers)), flush=True)
