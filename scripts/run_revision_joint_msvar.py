"""Independent corrective shared-chain MSVAR comparison; never edits C7 records."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from _joint_msvar import JointMSVARConfig, joint_msvar_scores

SPEC = importlib.util.spec_from_file_location("corrective_c7_reference", ROOT / "scripts" / "run_revision_specialist_stress.py")
C7 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = C7
SPEC.loader.exec_module(C7)

PREFIX = "revision_joint_msvar"
METHOD = "Joint MSVAR"
SEEDS = tuple(range(100, 120))
DOC = ROOT / "docs" / "revision_joint_msvar_protocol_2026-09-13.md"
SOURCE_FILES = ("scripts/_joint_msvar.py", "scripts/run_revision_joint_msvar.py",
                "scripts/run_revision_specialist_stress.py", "src/stcg/synthetic.py",
                "src/stcg/metrics.py", "src/stcg/baselines.py", "src/stcg/self_supervised.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_spec():
    reference = C7.load_protocol(ROOT / "results" / "tables")
    sid = C7.base_setting(reference["specification"]["display_cells"])
    config = C7.unique_settings(reference["specification"]["display_cells"])[sid]
    return reference, sid, config


def fixed_spec():
    reference, sid, config = reference_spec()
    return {
        "schema": 1, "experiment_stage": "postfinding_corrective_comparison",
        "method": METHOD, "reference_method": "STCG-v1", "seeds": list(SEEDS),
        "model": asdict(JointMSVARConfig()), "setting_id": sid, "input_config": config,
        "reference_protocol_hash": reference["protocol_hash"],
        "source_hashes": {name: sha(ROOT / name) for name in SOURCE_FILES},
        "protocol_document_sha256": sha(DOC), "timeout_seconds": 3600,
        "thread_environment": C7.THREAD_ENV,
        "primary_metric": "edge_auc", "bootstrap_seed": 20260913,
        "bootstrap_samples": 10000,
        "primary_population": "selected-start-converged joint MSVAR and successful STCG",
        "sensitivity_population": "all finite budget fits and successful STCG",
        "multiplicity": "one corrective primary Edge AUC contrast; other metrics exploratory; separate from C7",
    }


def load_frozen(output_dir: Path):
    path = output_dir / f"{PREFIX}_protocol.json"
    if not path.exists():
        raise ValueError("Freeze this corrective protocol before any evaluation fit")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen["specification"] != fixed_spec():
        raise ValueError("Corrective protocol/source mismatch; refusing changed code or cache mixing")
    if frozen["protocol_hash"] != C7.digest(frozen["specification"]):
        raise ValueError("Invalid corrective protocol hash")
    return frozen


def freeze(output_dir: Path):
    path = output_dir / f"{PREFIX}_protocol.json"
    if path.exists():
        return load_frozen(output_dir)
    specification = fixed_spec()
    frozen = {"frozen_at_utc": now(), "protocol_hash": C7.digest(specification),
              "specification": specification, "python": sys.version,
              "dependencies": {name: importlib.metadata.version(name)
                               for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")}}
    C7.atomic_json(path, frozen)
    return frozen


def part_path(output_dir: Path, seed: int):
    return output_dir / f"{PREFIX}_parts" / f"seed{seed:03d}.json"


def valid_part(output_dir: Path, seed: int, frozen: dict):
    path = part_path(output_dir, seed)
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    if row["seed"] != seed or row["protocol_hash"] != frozen["protocol_hash"]:
        raise ValueError(f"Wrong cache provenance: {path}")
    if row["status"] not in ("ok", "nonconverged", "failed", "timeout", "crashed"):
        raise ValueError(f"Incomplete cache status: {path}")
    if row.get("fit_artifact"):
        artifact = output_dir / row["fit_artifact"]
        if not artifact.exists() or sha(artifact) != row["fit_artifact_sha256"]:
            raise ValueError(f"Corrupt fitted artifact: {artifact}")
    return row


def reference_row(seed: int, frozen: dict):
    reference, sid, config = reference_spec()
    job = {"phase": "stress", "setting_id": sid, "seed": seed, "method": "STCG-v1",
           "config": config, "protocol_hash": reference["protocol_hash"]}
    row = C7.valid_cached(C7.result_path(ROOT / "results" / "tables", job), job)
    if row is None:
        raise ValueError(f"Missing frozen C7 STCG reference for seed {seed}")
    if row["protocol_hash"] != frozen["specification"]["reference_protocol_hash"]:
        raise ValueError("Wrong C7 reference protocol")
    return row


def worker(output_dir: Path, seed: int):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score
    from threadpoolctl import threadpool_limits, threadpool_info
    from stcg import recovery_summary
    frozen = load_frozen(output_dir)
    spec = frozen["specification"]
    dataset = C7.generate_dataset(spec["input_config"], seed)
    reference = reference_row(seed, frozen)
    if C7.array_hash(dataset.x) != reference["input_sha256"]:
        raise ValueError("Corrective and C7 raw input hash mismatch")
    if C7.array_hash(dataset.edge_truth) != reference["truth_sha256"]:
        raise ValueError("Corrective and C7 truth hash mismatch")
    row = {"seed": seed, "method": METHOD, "protocol_hash": frozen["protocol_hash"],
           "reference_protocol_hash": spec["reference_protocol_hash"],
           "setting_id": spec["setting_id"], "input_sha256": reference["input_sha256"],
           "truth_sha256": reference["truth_sha256"], "started_at_utc": now(),
           "worker_pid": os.getpid(), "experiment_stage": spec["experiment_stage"],
           "runtime_scope": "concurrent corrective fitting; no comparative speed claim"}
    with threadpool_limits(limits=1):
        row["threadpools"] = [{key: p.get(key) for key in ("user_api", "internal_api", "num_threads")}
                              for p in threadpool_info()]
        start = time.perf_counter()
        cpu_start = time.process_time()
        try:
            scores, info = joint_msvar_scores(np.array(dataset.x, dtype=float, copy=True),
                spec["input_config"]["max_lag"], seed=seed, config=JointMSVARConfig(**spec["model"]))
            row.update(fit_seconds=time.perf_counter() - start,
                       fit_cpu_seconds=time.process_time() - cpu_start)
            if scores.shape != dataset.edge_truth.shape or not np.isfinite(scores).all():
                raise ValueError("Invalid score tensor")
            row.update(status="ok" if info["converged"] else "nonconverged",
                       selected_start=info["start_index"], selected_converged=info["converged"],
                       selected_loglik=info["loglik"], selected_iterations=info["iterations"],
                       n_successful_starts=info["n_successful_starts"],
                       n_converged_starts=info["n_converged_starts"], starts=info["starts"],
                       effective_counts=info["effective_counts"],
                       sample_regime_ari=float(adjusted_rand_score(
                           dataset.regimes[info["effective_offset"]:], info["regime_sequence"])),
                       scores_sha256=C7.array_hash(scores))
            row.update({key: float(value) for key, value in recovery_summary(scores, dataset.edge_truth).items()})
            artifact = part_path(output_dir, seed).with_suffix(".npz")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            temporary = artifact.with_suffix(".tmp")
            with temporary.open("wb") as file:
                np.savez_compressed(file, scores=scores, coefficients=info["coefficients"],
                    covariance=info["covariance"], transition=info["transition"], initial=info["initial"],
                    gamma=info["gamma"], scale=info["scale"])
            os.replace(temporary, artifact)
            row.update(fit_artifact=artifact.relative_to(output_dir).as_posix(),
                       fit_artifact_sha256=sha(artifact))
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                       fit_seconds=time.perf_counter() - start,
                       fit_cpu_seconds=time.process_time() - cpu_start)
    row["finished_at_utc"] = now()
    C7.atomic_json(part_path(output_dir, seed), row)
    print(json.dumps({key: row.get(key) for key in
                      ("seed", "status", "fit_seconds", "selected_iterations", "n_converged_starts")}), flush=True)


def run(output_dir: Path, workers: int, calibrate: bool):
    if workers not in (1, 2):
        raise ValueError("Corrective comparison allows at most two workers")
    frozen = load_frozen(output_dir)
    seeds = [SEEDS[0]] if calibrate else list(SEEDS)
    todo = [seed for seed in seeds if valid_part(output_dir, seed, frozen) is None]
    for seed in todo:
        reference_row(seed, frozen)  # stop before launching if any reference is unavailable
    logs = ROOT / "experiments" / "logs" / PREFIX
    logs.mkdir(parents=True, exist_ok=True)
    active = {}
    env = dict(os.environ, **C7.THREAD_ENV, PYTHONHASHSEED="0")
    completed = len(seeds) - len(todo)
    try:
        while todo or active:
            while todo and len(active) < workers:
                seed = todo.pop(0)
                log = (logs / f"seed{seed:03d}.log").open("a", encoding="utf-8")
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(seed),
                           "--output-dir", str(output_dir)]
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                active[seed] = (process, log, time.monotonic())
            for seed, (process, log, started) in list(active.items()):
                expired = time.monotonic() - started > frozen["specification"]["timeout_seconds"]
                if expired:
                    process.kill()
                    process.wait()
                code = process.poll()
                if code is None:
                    continue
                log.close()
                row = valid_part(output_dir, seed, frozen)
                if row is None:
                    row = {"seed": seed, "method": METHOD, "protocol_hash": frozen["protocol_hash"],
                           "status": "timeout" if expired else "crashed", "exit_code": code,
                           "elapsed_dispatch_seconds": time.monotonic() - started,
                           "finished_at_utc": now()}
                    C7.atomic_json(part_path(output_dir, seed), row)
                completed += 1
                del active[seed]
                print(f"{completed}/{len(seeds)} seed={seed} status={row['status']} "
                      f"fit_seconds={row.get('fit_seconds')}", flush=True)
            C7.atomic_json(output_dir / f"{PREFIX}_status.json",
                {"controller_pid": os.getpid(), "updated_at_utc": now(), "completed": completed,
                 "planned": len(seeds), "active": {str(seed): p.pid for seed, (p, _, _) in active.items()},
                 "protocol_hash": frozen["protocol_hash"]})
            if todo or active:
                time.sleep(.2)
    finally:
        for process, log, _ in active.values():
            if process.poll() is None:
                process.kill()
                process.wait()
            log.close()


def aggregate(output_dir: Path):
    import numpy as np
    frozen = load_frozen(output_dir)
    rows = [row for seed in SEEDS if (row := valid_part(output_dir, seed, frozen)) is not None]
    starts = [{"seed": row["seed"], **start} for row in rows for start in row.get("starts", [])]
    C7.write_csv(output_dir / f"{PREFIX}_detail.csv", [{k: v for k, v in row.items() if k != "starts"} for row in rows])
    C7.write_csv(output_dir / f"{PREFIX}_starts.csv", starts)
    coverage = {"planned": len(SEEDS), "completed": len(rows),
                **{status: sum(row["status"] == status for row in rows)
                   for status in ("ok", "nonconverged", "failed", "timeout", "crashed")},
                "unfinished_seeds": [seed for seed in SEEDS if not any(row["seed"] == seed for row in rows)]}
    C7.write_csv(output_dir / f"{PREFIX}_coverage.csv", [coverage])
    if len(rows) != len(SEEDS):
        print(json.dumps({"complete": False, **coverage}), flush=True)
        return
    pairs, summaries = [], []
    for population, statuses in (("selected_converged", {"ok"}), ("all_finite_budget_fits", {"ok", "nonconverged"})):
        eligible = [row for row in rows if row["status"] in statuses]
        refs = {row["seed"]: reference_row(row["seed"], frozen) for row in eligible}
        eligible = [row for row in eligible if refs[row["seed"]]["status"] == "ok"]
        for row in eligible:
            if row["input_sha256"] != refs[row["seed"]]["input_sha256"]:
                raise ValueError("Paired input mismatch")
        for method in ("STCG-v1", METHOD):
            method_rows = [refs[row["seed"]] if method == "STCG-v1" else row for row in eligible]
            summary = {"population": population, "method": method, "n": len(method_rows),
                       "seeds": [row["seed"] for row in eligible]}
            for metric in C7.METRICS:
                values = [r[metric] for r in method_rows if C7.finite_metric(r, metric)]
                summary.update({metric + "_n": len(values), metric + "_mean": float(np.mean(values)) if values else None,
                                metric + "_std": float(np.std(values, ddof=1)) if len(values) > 1 else None})
            summaries.append(summary)
        for metric in C7.METRICS:
            valid = [row for row in eligible if C7.finite_metric(row, metric) and C7.finite_metric(refs[row["seed"]], metric)]
            pair = {"population": population, "opponent": METHOD, "metric": metric,
                    "common_seeds": [row["seed"] for row in valid], "n": len(valid),
                    "primary": population == "selected_converged" and metric == "edge_auc",
                    "c7_multiplicity_family_member": False}
            if len(valid) >= 2:
                pair.update(C7.paired_statistics([refs[row["seed"]][metric] for row in valid],
                                                [row[metric] for row in valid], metric))
            pairs.append(pair)
    C7.write_csv(output_dir / f"{PREFIX}_summary.csv", summaries)
    C7.write_csv(output_dir / f"{PREFIX}_paired.csv", pairs)
    lines = ["# Corrective shared-chain joint MSVAR comparison", "",
             "Post-finding corrective comparison, frozen before its fit results. Historical MSAR-proxy rows and the "
             "C7 grid/Holm families are unchanged. Exactly the C7 base input hashes, seeds 100–119, are used. "
             "The new model has direct VAR(3) means, a single shared Markov chain, and full regime covariance.", "",
             f"Coverage: {coverage}", "",
             "Ten starts per seed, 200 EM iterations per start, relative likelihood tolerance 1e-6. "
             "Primary rows require convergence of the highest-likelihood selected start; all finite budget fits "
             "are separately shown. Max-iteration and failed starts are never imputed as chance recovery.", "",
             "| Population | n | STCG Edge AUC | Joint MSVAR Edge AUC | STCG minus joint [95% CI] | Two-sided t p | Wilcoxon p |",
             "|---|---:|---:|---:|---|---:|---:|"]
    for row in pairs:
        if row["metric"] != "edge_auc" or "mean_stcg" not in row:
            continue
        lines.append(f"| {row['population']} | {row['n']} | {row['mean_stcg']:.6f} | {row['mean_opponent']:.6f} | "
                     f"{row['mean_improvement']:.6f} [{row['ci_low']:.6f}, {row['ci_high']:.6f}] | "
                     f"{row['paired_t_two_sided_p']:.6g} | {row['wilcoxon_two_sided_p']:.6g} |")
    lines += ["", "One corrective primary Edge AUC contrast; four other recovery metrics and the finite-fit population "
              "are sensitivity evidence. No equivalence or non-inferiority claim is available. Runtime records here "
              "were collected during other work and do not establish a comparative speed ratio.", "",
              "Source and protocol hashes: `revision_joint_msvar_protocol.json`; full likelihood histories and all "
              "starts: `revision_joint_msvar_starts.csv`; fitted matrices/posteriors: `revision_joint_msvar_parts/*.npz`."]
    (output_dir / f"{PREFIX}_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, **coverage}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--worker", type=int, choices=SEEDS)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.worker is not None:
        worker(args.output_dir, args.worker)
        return
    if args.freeze:
        print(json.dumps(freeze(args.output_dir)), flush=True)
    if args.run or args.calibrate:
        run(args.output_dir, args.workers, args.calibrate)
    if args.aggregate:
        aggregate(args.output_dir)


if __name__ == "__main__":
    main()
