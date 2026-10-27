"""Thin post hoc application of the frozen score controls to retained Study C.

No generation or fitting is invoked and the symmetric_sensitivity directory is
read only. All lambdas and statistical choices are inherited without selection.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import analyze_score_preserving_controls as core
import run_symmetric_sensitivity as study

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/information_sciences/symmetric_controls"
C_DIR = "results/information_sciences/symmetric_sensitivity"
SOURCES = ("scripts/analyze_symmetric_controls.py", "tests/test_symmetric_controls.py",
           "scripts/analyze_score_preserving_controls.py", "scripts/run_symmetric_sensitivity.py")
WINDOW_ATTRS = ["cell", "h", "stride", "M", "h_over_D"]


def input_snapshot(root: Path) -> dict:
    directory = root / C_DIR
    protocol = json.loads((directory / study.PROTOCOL_NAME).read_text(encoding="utf-8"))
    seal = json.loads((directory / "symmetric_sensitivity_outputs.json").read_text(encoding="utf-8"))
    for relative, expected in seal["sha256"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or core.sha(path) != expected:
            raise ValueError("Study C sealed input changed: " + relative)
    sources = protocol["source_sha256"].copy()
    sources["configs/symmetric_sensitivity.yaml"] = protocol["configuration_sha256"]
    for relative, expected in sources.items():
        if core.sha(root / relative) != expected: raise ValueError("Study C frozen source changed: " + relative)
    return {"study_C_sha256": {f"{C_DIR}/{name}": value for name, value in core.directory_hashes(directory).items()},
            "study_C_frozen_sources_sha256": sources, "original_A_B_guard": core.input_snapshot(root)}


def protocol(input_sha: str, root: Path = ROOT) -> dict:
    return {"analysis": "symmetric_score_preserving_controls_v1",
        "role": "post hoc application to completed Study C outputs; not a new confirmation or public preregistration",
        "input_manifest_sha256": input_sha, "source_sha256": {name: core.sha(root / name) for name in SOURCES},
        "fixed_score_control_source_sha256": "a5ac44d412d2edeb0a9699ce95b954ba06f5cb96170c7b21ab48b1f50bd7a7c4",
        "lambda_positive": list(core.LAMBDAS), "residual_endpoints": [0., 1.], "no_lambda_selection": True,
        "studies": "all 30 seeds70000-70029, D300/600, five prespecified window/stride cells; 300 retained fits",
        "methods": [r["method"] for r in core.method_spec()], "metrics": list(core.METRICS),
        "statistics": {"bootstrap_draws": core.BOOTSTRAP_DRAWS, "bootstrap_seed": core.STAT_SEED,
                       "scope": "95% pointwise paired-repeat descriptive intervals; no hypothesis tests or family assignment",
                       "minimum_n": 5, "references": ["raw_window", "forced_hard"]},
        "missing": "retain failed cases/methods and undefined targets, finite complete pairs only; no imputation/replacement",
        "threads": 1, "new_fits": 0, "new_simulations": 0}


def freeze(output: Path = OUT, root: Path = ROOT) -> dict:
    output, root = Path(output), Path(root)
    if output.exists() and any(output.iterdir()): raise ValueError("Freeze requires new empty output directory")
    snapshot = input_snapshot(root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "input_manifest.json").write_bytes(core.json_bytes(snapshot))
    p = protocol(core.sha(output / "input_manifest.json"), root)
    assert p["source_sha256"]["scripts/analyze_score_preserving_controls.py"] == p["fixed_score_control_source_sha256"]
    (output / "protocol.json").write_bytes(core.json_bytes(p))
    return p


def validate_frozen(output: Path, root: Path) -> dict:
    before = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    if before != input_snapshot(root): raise ValueError("Original or Study C frozen inputs changed")
    p = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if p != protocol(core.sha(output / "input_manifest.json"), root): raise ValueError("Post hoc protocol/source changed")
    return p


def adapt_case(case: dict) -> dict:
    identity, a = case["identity"], case["arrays"]
    metadata = {"study": "C", "scorer": "lasso", "repeat_seed": identity["seed"],
                "duration": identity["duration"], "schedule": "fixed_phase", "cell": identity["cell"],
                "h": identity["h"], "stride": identity["stride"], "M": len(a["bounds"]),
                "h_over_D": identity["h"] / identity["duration"]}
    if case["fit_status"] != "ok":
        return {**metadata, "status": "failed", "error": case["record"]["diagnostic"].get("error", "fit_failed")}
    truth = a["regimes"][(a["bounds"][:, 0] + a["bounds"][:, 1] - 1) // 2]
    np.testing.assert_array_equal(truth, a["true_window_labels"])
    support = a["state_support"]
    return {**metadata, "status": "ok", "error": "", "raw": a["window_scores"],
            "forced": a["emission_forced_hard"], "native": a["emission_native_hard"],
            "gate_accepted": case["gate_accepted"], "active_truth": support[truth].astype(bool),
            "changing_mask": np.any(support != support[0], axis=0)}


def attach_window_attributes(frame: pd.DataFrame, detail: pd.DataFrame) -> pd.DataFrame:
    attributes = detail[WINDOW_ATTRS].drop_duplicates()
    if attributes.cell.duplicated().any(): raise ValueError("Cell aliases different window configurations")
    return frame.merge(attributes, on="cell", how="left", validate="many_to_one")


def compute(root: Path = ROOT) -> tuple[dict[str, pd.DataFrame], dict]:
    details, checks = [], []
    cases = failures = baseline_checks = 0
    for item in study.iter_cases(root):
        case = adapt_case(item); cases += 1
        identity = {key: case[key] for key in [*core.GROUP, "repeat_seed", *WINDOW_ATTRS[1:]]}
        if case["status"] == "failed":
            failures += 1
            details.extend({**identity, **spec, **{metric: np.nan for metric in core.METRICS},
                "status": "failed", "undefined_reason": case["error"], "gate_accepted": None,
                "temporal_delta_edges": 0, "temporal_delta_edges_total": 0, "scored_edges": 0,
                "equals_raw": False, "equals_forced": False} for spec in core.method_spec())
            continue
        rows, _, diagnostic = core.evaluate_controls(case["raw"], case["forced"], case["native"],
            case["gate_accepted"], case["active_truth"], case["changing_mask"])
        original = {row["method"]: row for row in item["record"]["methods"]}
        for row in rows:
            if row["method"] in ("raw_window", "native_hard", "forced_hard"):
                for metric in core.METRICS:
                    value = original[row["method"]][metric]
                    np.testing.assert_allclose(row[metric], np.nan if value is None else value, atol=2e-12, rtol=0, equal_nan=True)
                baseline_checks += 1
        details.extend({**identity, **row} for row in rows)
        checks.extend({**identity, **row} for row in diagnostic)
    detail = pd.DataFrame(details).sort_values(core.GROUP + ["repeat_seed", "method"]).reset_index(drop=True)
    columns = [*core.GROUP, "repeat_seed", *WINDOW_ATTRS[1:], "lambda", "lag", "source", "target", "time_pairs", "finite",
               "changing_edge", "temporal_evaluable", "raw_tied_pairs", "shrink_tied_pairs", "new_tied_pairs", "lost_tied_pairs",
               "reversed_pairs", "raw_edge_auroc", "shrink_edge_auroc", "edge_auroc_difference", "undefined_reason"]
    affine = pd.DataFrame(checks, columns=columns).sort_values(core.GROUP + ["repeat_seed", "lambda", "lag", "source", "target"]).reset_index(drop=True)
    affine["finite"] = affine.finite.astype(bool); affine["temporal_evaluable"] = affine.temporal_evaluable.astype(bool)
    finite = affine[affine.finite]
    difference = finite.edge_auroc_difference.astype(float)
    events = finite[(finite.new_tied_pairs > 0) | (finite.lost_tied_pairs > 0) | (finite.reversed_pairs > 0) |
                    (np.isfinite(difference) & (difference != 0))].reset_index(drop=True)
    valid = finite[finite.temporal_evaluable]
    summary, contrasts = core.summarise(detail)
    report = {"role": "post hoc score-preserving controls on Study C", "cases": cases, "failed_cases": failures,
        "method_rows": len(detail), "method_failed_rows": int((detail.status == "failed").sum()),
        "method_undefined_rows": int((detail.status == "undefined").sum()), "reference_method_rows_checked": baseline_checks,
        "affine_edge_checks": len(affine), "temporal_evaluable_edge_checks": len(valid), "affine_events": len(events),
        "new_tied_pairs": int(finite.new_tied_pairs.sum()), "lost_tied_pairs": int(finite.lost_tied_pairs.sum()),
        "reversed_pairs": int(finite.reversed_pairs.sum()), "changed_edge_auc_count": int((valid.edge_auroc_difference != 0).sum()),
        "new_fits": 0, "new_simulations": 0}
    return {"detail.csv": detail, "summary.csv": attach_window_attributes(summary, detail),
            "contrasts.csv": attach_window_attributes(contrasts, detail),
            "affine_checks.csv": affine, "affine_events.csv": events}, report


def _write(output: Path, root: Path, p: dict) -> dict:
    with threadpool_limits(limits=1): tables, report = compute(root)
    for name, frame in tables.items(): frame.to_csv(output / name, index=False, lineterminator="\n")
    (output / "diagnostics.json").write_bytes(core.json_bytes(report))
    validate_frozen(output, root)
    seal = {"protocol_sha256": core.sha(output / "protocol.json"), "source_sha256": p["source_sha256"],
            "sha256": {name: value for name, value in core.directory_hashes(output).items() if name != "manifest.json"},
            "new_fits": 0, "new_simulations": 0}
    (output / "manifest.json").write_bytes(core.json_bytes(seal))
    return report


def run(output: Path = OUT, root: Path = ROOT) -> dict:
    output, root = Path(output), Path(root)
    p = validate_frozen(output, root)
    if set(core.directory_hashes(output)) != {"input_manifest.json", "protocol.json"}:
        raise ValueError("Run refuses to overwrite prior derived outputs")
    return _write(output, root, p)


def rebuild(output: Path = OUT, root: Path = ROOT) -> dict:
    output, root = Path(output), Path(root)
    p = validate_frozen(output, root)
    seal = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    before = core.directory_hashes(output)
    assert seal["sha256"] == {name: value for name, value in before.items() if name != "manifest.json"}
    result = _write(output, root, p)
    assert before["protocol.json"] == core.sha(output / "protocol.json")
    assert before["input_manifest.json"] == core.sha(output / "input_manifest.json")
    return result


def verify(output: Path = OUT, root: Path = ROOT) -> dict:
    output, root = Path(output), Path(root)
    before = core.directory_hashes(output)
    p = validate_frozen(output, root)
    seal = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert seal["protocol_sha256"] == core.sha(output / "protocol.json")
    assert seal["source_sha256"] == p["source_sha256"]
    assert seal["sha256"] == {name: value for name, value in before.items() if name != "manifest.json"}
    with threadpool_limits(limits=1): tables, report = compute(root)
    for name, frame in tables.items():
        actual = pd.read_csv(output / name)
        expected = pd.read_csv(io.StringIO(frame.to_csv(index=False, lineterminator="\n")))
        pd.testing.assert_frame_equal(actual, expected, rtol=0, atol=2e-12)
    assert report == json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
    assert before == core.directory_hashes(output)
    validate_frozen(output, root)
    return {"status": "passed", "read_only": True, "files_unchanged": len(before), **report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    actions = parser.add_mutually_exclusive_group(required=True)
    for name in ("freeze", "run", "verify", "rebuild"): actions.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    action = freeze if args.freeze else run if args.run else rebuild if args.rebuild else verify
    print(json.dumps(action(args.output_dir, args.root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
