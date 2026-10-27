"""Apply frozen post-review boundary-slot metrics to retained Study C outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_temporal_localisation as loc

OUT = ROOT / "results/information_sciences/symmetric_localisation"
STAGE = ROOT / "experiments/logs/reviewer_controls_2026-10-25"


def input_hashes(root: Path) -> dict:
    directory = root / "results/information_sciences/symmetric_sensitivity"
    return {path.relative_to(root).as_posix(): loc.sha(path) for path in sorted(directory.rglob("*")) if path.is_file()}


def protocol(root: Path) -> dict:
    root = Path(root).resolve()
    definition_path = root / "results/information_sciences/temporal_localisation/protocol.json"
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    c_protocol = json.loads((root / "results/information_sciences/symmetric_sensitivity/symmetric_sensitivity_protocol.json").read_text(encoding="utf-8"))
    sources = dict(c_protocol["source_sha256"])
    sources["configs/symmetric_sensitivity.yaml"] = c_protocol["configuration_sha256"]
    for name in ("scripts/analyze_temporal_localisation.py", "scripts/analyze_score_preserving_controls.py", "scripts/analyze_symmetric_localisation.py"):
        sources[name] = loc.sha(root / name)
    for name, digest in sources.items():
        if loc.sha(root / name) != digest:
            raise ValueError(f"Frozen C source changed: {name}")
    return {"study": "post_review_symmetric_boundary_slot_localisation_v2", "role": "post hoc descriptive retained-output analysis",
            "QA_amendment": {"superseded_protocol_sha256": "ebfdc6b48edefdb701516942980730b452c27df2a45533faffe497f18a29d9fe",
                             "reason": "Successful input records contain error=null; normalize to an empty string rather than the literal None, which CSV parsing treats as missing",
                             "scope": "Serialization-only correction after the initial run and failed text-column verification; no metric, parameter, seed, emission or sample changes; v1 preserved in the local audit stage"},
            "definition_protocol_path": definition_path.relative_to(root).as_posix(), "definition_protocol_sha256": loc.sha(definition_path),
            **{key: definition[key] for key in ("methods", "boundary_grid", "boundary_AP", "boundary_F1", "stable_metric", "threshold", "interpretation", "statistics")},
            "frozen_C_input_sha256": input_hashes(root), "source_sha256": sources,
            "protected_AB_input_sha256": loc.frozen_hashes(root), "protected_AB_source_sha256": loc.frozen_sources(root),
            "new_fits": 0, "new_simulations": 0}


def freeze(root: Path = ROOT, output: Path = OUT, stage: Path = STAGE) -> dict:
    root, output, stage = Path(root).resolve(), Path(output).resolve(), Path(stage).resolve()
    value = protocol(root)
    if output.exists() and any(output.iterdir()):
        path = output / "protocol.json"
        if not path.is_file() or path.read_bytes() != loc.json_bytes(value):
            raise ValueError("Refusing to replace a different/existing C localisation protocol")
        return value
    before = datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_bytes(loc.json_bytes(value))
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "symmetric_localisation_freeze.json").write_bytes(loc.json_bytes({"before_protocol_write_utc": before,
        "after_protocol_write_before_analysis_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": loc.sha(output / "protocol.json"), "definition_protocol_sha256": value["definition_protocol_sha256"],
        "new_fits": 0, "new_simulations": 0}))
    return value


def _check(root: Path, output: Path) -> dict:
    value = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if value != protocol(root):
        raise ValueError("C localisation protocol/source/input closure drift")
    return value


def retained_cases(root: Path):
    from analyze_score_preserving_controls import build_control_emissions
    from run_symmetric_sensitivity import iter_cases
    for case in iter_cases(root):
        identity, arrays = case["identity"], case["arrays"]
        meta = {"study": "C", "scorer": "lasso", "repeat_seed": identity["seed"],
                "duration": identity["duration"], "schedule": "fixed_phase", "cell": identity["cell"],
                "h": identity["h"], "stride": identity["stride"],
                "status": case["fit_status"], "error": "" if case["record"]["diagnostic"].get("error") is None else str(case["record"]["diagnostic"]["error"])}
        if case["fit_status"] != "ok":
            yield meta, None, None
            continue
        raw = arrays["window_scores"]
        emissions = build_control_emissions(raw, arrays["emission_forced_hard"], arrays["emission_native_hard"], case["gate_accepted"])
        emissions["oracle_label"] = arrays["emission_oracle_label"]
        emissions["static_mean"] = arrays["emission_static_mean"]
        if tuple(emissions) != loc.METHODS:
            raise ValueError("C control-emission method inventory drift")
        rows, edge_arrays, grid = loc.evaluate_localisation(emissions, raw, arrays["bounds"], arrays["regimes"], arrays["state_support"])
        yield meta, rows, edge_arrays


def _filename(meta: dict) -> str:
    return f"C_{meta['repeat_seed']}_{meta['cell']}.npz"


def _rows(meta: dict, rows) -> list[dict]:
    if rows is None:
        rows = [{"method": method, "status": "retained_fit_failure", **{key: np.nan for key in loc.METRICS}} for method in loc.METHODS]
    return [{**meta, **row} for row in rows]


def summarise(detail: pd.DataFrame):
    summary, paired = loc.summarise(detail)
    settings = detail[[*loc.GROUP, "h", "stride"]].drop_duplicates()
    return tuple(frame.merge(settings, on=list(loc.GROUP), how="left", validate="many_to_one") for frame in (summary, paired))


def _preserved(root: Path, locked: dict) -> bool:
    return (input_hashes(root) == locked["frozen_C_input_sha256"]
            and loc.frozen_hashes(root) == locked["protected_AB_input_sha256"]
            and loc.frozen_sources(root) == locked["protected_AB_source_sha256"])


def run(root: Path = ROOT, output: Path = OUT, stage: Path = STAGE) -> dict:
    root, output, stage = Path(root).resolve(), Path(output).resolve(), Path(stage).resolve()
    locked = _check(root, output)
    if any(path.name != "protocol.json" for path in output.iterdir()):
        raise ValueError("Refusing to overwrite existing C localisation outputs")
    started = datetime.now(timezone.utc).isoformat()
    records = output / "records"; records.mkdir()
    details = []; cases = failures = 0
    for meta, rows, edge_arrays in retained_cases(root):
        failures += int(rows is None)
        details.extend(_rows(meta, rows))
        if edge_arrays is not None:
            np.savez_compressed(records / _filename(meta), **edge_arrays)
        cases += 1
        if cases % 50 == 0:
            print(f"retained C cases processed: {cases}", flush=True)
    detail = pd.DataFrame(details)
    summary, paired = summarise(detail)
    for name, frame in (("detail.csv", detail), ("summary.csv", summary), ("paired_vs_raw.csv", paired)):
        frame.to_csv(output / name, index=False)
    if not _preserved(root, locked):
        raise AssertionError("Frozen C/A/B files changed during localisation analysis")
    preservation = {"C_files_unchanged": len(locked["frozen_C_input_sha256"]), "AB_files_unchanged": len(locked["protected_AB_input_sha256"]),
        "AB_source_files_unchanged": len(locked["protected_AB_source_sha256"]), "cases_retained": cases, "failed_cases_retained": failures,
        "new_simulations": 0, "new_fits": 0}
    (output / "preservation.json").write_bytes(loc.json_bytes(preservation))
    manifest = {"sha256": {path.relative_to(output).as_posix(): loc.sha(path) for path in sorted(output.rglob("*")) if path.is_file()},
                "protocol_sha256": loc.sha(output / "protocol.json"), "source_sha256": locked["source_sha256"], **preservation}
    (output / "manifest.json").write_bytes(loc.json_bytes(manifest))
    report = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "cases": cases,
        "retained_failures": failures, "method_rows": len(detail), "summary_rows": len(summary), "paired_rows": len(paired),
        "protocol_sha256": loc.sha(output / "protocol.json"), "manifest_sha256": loc.sha(output / "manifest.json"), "new_fits": 0, "new_simulations": 0}
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "symmetric_localisation_run.json").write_bytes(loc.json_bytes(report))
    return report


def verify(root: Path = ROOT, output: Path = OUT) -> dict:
    root, output = Path(root).resolve(), Path(output).resolve()
    locked = _check(root, output)
    before = {p.relative_to(output).as_posix(): loc.sha(p) for p in output.rglob("*") if p.is_file()}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest["sha256"]) != set(before) - {"manifest.json"} or any(before[name] != digest for name, digest in manifest["sha256"].items()):
        raise ValueError("C localisation output inventory/hash drift")
    if manifest["protocol_sha256"] != before["protocol.json"] or manifest["source_sha256"] != locked["source_sha256"]:
        raise ValueError("C localisation source/protocol manifest drift")
    expected_rows = []; cases = failures = 0
    for meta, rows, edge_arrays in retained_cases(root):
        failures += int(rows is None)
        expected_rows.extend(_rows(meta, rows))
        if edge_arrays is not None:
            with np.load(output / "records" / _filename(meta), allow_pickle=False) as saved:
                if set(saved.files) != set(edge_arrays):
                    raise ValueError("C localisation per-edge array inventory drift")
                for key, value in edge_arrays.items():
                    if value.dtype.kind in "US":
                        np.testing.assert_array_equal(saved[key], value)
                    else:
                        np.testing.assert_allclose(saved[key], value, rtol=1e-12, atol=1e-14, equal_nan=True)
        cases += 1
    expected = pd.DataFrame(expected_rows)
    for column in expected.select_dtypes(include="object"):
        expected[column] = expected[column].map(lambda value: np.nan if isinstance(value, str) and value == "" else value)
    detail = pd.read_csv(output / "detail.csv")
    pd.testing.assert_frame_equal(detail, expected, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-14)
    summary, paired = summarise(detail)
    for name, frame in (("summary.csv", summary), ("paired_vs_raw.csv", paired)):
        pd.testing.assert_frame_equal(pd.read_csv(output / name), frame, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-14)
    after = {p.relative_to(output).as_posix(): loc.sha(p) for p in output.rglob("*") if p.is_file()}
    if before != after or not _preserved(root, locked):
        raise AssertionError("Read-only C localisation verification changed retained files")
    return {"status": "passed", "read_only": True, "cases": cases, "retained_failures": failures,
        "method_rows_recomputed": len(detail), "summary_rows_recomputed": len(summary), "paired_rows_recomputed": len(paired),
        "files_unchanged": len(before), "C_files_unchanged": len(locked["frozen_C_input_sha256"]),
        "frozen_AB_files_unchanged": len(locked["protected_AB_input_sha256"]), "new_fits": 0, "new_simulations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "run", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--stage-dir", type=Path, default=STAGE)
    args = parser.parse_args()
    if args.mode == "freeze":
        value = freeze(args.root, args.output_dir, args.stage_dir)
        report = {"status": "frozen", "C_input_files": len(value["frozen_C_input_sha256"]), "methods": len(value["methods"])}
    elif args.mode == "run":
        report = run(args.root, args.output_dir, args.stage_dir)
    else:
        report = verify(args.root, args.output_dir)
    print(json.dumps(loc.clean(report), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
