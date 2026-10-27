"""Post hoc diagnostics of retained generator outputs; no simulation or fitting.

Only observed series were retained in A/B. Endpoint RMS summaries therefore
cover those saved observations, not the unretained burn-in trajectory. These
descriptive measurements apply no stability threshold or pass/fail criterion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stcg.mechanism import spectral_radius


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def frozen_hashes(root: Path) -> dict:
    base = root / "results/information_sciences"
    return {path.relative_to(root).as_posix(): sha(path)
            for study in ("duration_confirmation", "schedule_robustness")
            for path in sorted((base / study).rglob("*")) if path.is_file()}


def frozen_source_hashes(root: Path) -> dict:
    base = root / "results/information_sciences"
    paths = {"configs/schedule_robustness.yaml"}
    for name in ("duration_confirmation/protocol.json", "schedule_robustness/schedule_robustness_protocol.json"):
        protocol = json.loads((base / name).read_text(encoding="utf-8"))
        paths.update(protocol["source_sha256"])
    return {name: sha(root / name) for name in sorted(paths)}


def check_declared_sources(root: Path) -> None:
    base = root / "results/information_sciences"
    for folder, protocol_name, output_name in (
        ("duration_confirmation", "protocol.json", "COMPLETE.json"),
        ("schedule_robustness", "schedule_robustness_protocol.json", "schedule_robustness_outputs.json"),
    ):
        output = base / folder
        protocol = json.loads((output / protocol_name).read_text(encoding="utf-8"))
        for name, expected in protocol["source_sha256"].items():
            if sha(root / name) != expected:
                raise ValueError(f"Frozen source changed: {name}")
        manifest = json.loads((output / output_name).read_text(encoding="utf-8"))
        protocol_key = "protocol_sha256" if folder == "duration_confirmation" else "source_protocol_sha256"
        if sha(output / protocol_name) != manifest[protocol_key]:
            raise ValueError(f"Frozen protocol changed: {folder}/{protocol_name}")
        if folder == "schedule_robustness" and sha(root / "configs/schedule_robustness.yaml") != protocol["configuration_sha256"]:
            raise ValueError("Frozen B configuration changed")
        for name, expected in manifest["sha256"].items():
            if sha(output / name) != expected:
                raise ValueError(f"Frozen experimental output changed: {folder}/{name}")


def rms(values: np.ndarray) -> float:
    """RMS over all entries, scaled to avoid overflow from squaring large values."""
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        return np.nan
    scale = float(np.max(np.abs(values)))
    return 0.0 if scale == 0 else float(scale * np.sqrt(np.mean(np.square(values / scale))))


def trajectory_metrics(x: np.ndarray, endpoint_size: int = 300) -> dict:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or endpoint_size < 1 or len(x) < 2 * endpoint_size:
        raise ValueError("Observed series must contain two nonoverlapping endpoint blocks")
    early, late = rms(x[:endpoint_size]), rms(x[-endpoint_size:])
    return {
        "observed_timesteps": len(x), "n_nodes": x.shape[1],
        "all_observed_entries_finite": bool(np.isfinite(x).all()),
        "nonfinite_observed_entries": int((~np.isfinite(x)).sum()),
        "maximum_absolute_observed_value": float(np.max(np.abs(x))),
        "observed_rms": rms(x), "early_rms": early, "late_rms": late,
        "late_to_early_rms_ratio": float(late / early) if np.isfinite(early) and early > 0 else np.nan,
        "rms_ratio_evaluable": bool(np.isfinite(early) and early > 0 and np.isfinite(late)),
        "endpoint_block_timesteps": endpoint_size,
    }


def mechanism_metrics(inputs: dict, study: str, seed: int) -> dict:
    coefficients, support = inputs["coefficients"], inputs["support"]
    if coefficients.shape != support.shape or coefficients.shape[0] != 2:
        raise ValueError("Expected the retained two-state coefficient/support tensors")
    np.testing.assert_array_equal(support, np.abs(coefficients) > 0)
    computed = np.array([spectral_radius(state) for state in coefficients])
    np.testing.assert_allclose(computed, inputs["spectral_radius"], atol=1e-12, rtol=0)
    _, lags, nodes, _ = support.shape
    diagonal = np.zeros((lags, nodes, nodes), bool)
    for node in range(nodes):
        diagonal[:, node, node] = True
    base, second = support.astype(bool)
    base_nonself, second_nonself = int(base[~diagonal].sum()), int(second[~diagonal].sum())
    shared = int((base & second & ~diagonal).sum())
    return {"study": study, "repeat_seed": seed,
            "state0_companion_radius": computed[0], "state1_companion_radius": computed[1],
            "stored_radius_max_abs_gap": float(np.max(np.abs(computed - inputs["spectral_radius"]))),
            "state0_nonself_edges": base_nonself, "state1_nonself_edges": second_nonself,
            "state0_self_edges": int(base[diagonal].sum()), "state1_self_edges": int(second[diagonal].sum()),
            "nonself_positions": int((~diagonal).sum()), "self_positions": int(diagonal.sum()),
            "shared_nonself_edges": shared, "realised_retained_fraction": shared / base_nonself if base_nonself else np.nan,
            "state0_nonself_density": float(base[~diagonal].mean()),
            "state1_nonself_density": float(second[~diagonal].mean()),
            "state1_self_density": float(second[diagonal].mean()),
            "base_empty_fallback_compatible": bool(base.sum() == 1),
            "state1_empty_fallback_compatible": bool(np.array_equal(base, second)),
            "initial_history_timesteps_saved": int(inputs["initial_values"].shape[0]),
            "innovation_timesteps_saved": int(inputs["innovations"].shape[0])}


def collect(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = root / "results/information_sciences"
    a = base / "duration_confirmation"; b = base / "schedule_robustness"
    a_protocol = json.loads((a / "protocol.json").read_text(encoding="utf-8"))
    b_protocol = json.loads((b / "schedule_robustness_protocol.json").read_text(encoding="utf-8"))
    mechanisms, trajectories = [], []
    for study, seeds, config in (("A", a_protocol["repeat_seeds"], a_protocol["base_config"]),
                                 ("B", b_protocol["repeats"], b_protocol["configuration"]["mechanism"])):
        burn, timesteps = int(config["burn_in"]), int(config["timesteps"])
        for seed in seeds:
            inputs = arrays(a / "records" / f"repeat_{seed}_inputs.npz" if study == "A" else b / "inputs" / f"repeat_{seed}.npz")
            mechanisms.append(mechanism_metrics(inputs, study, seed))
            cells = [(duration, "fixed_phase") for duration in a_protocol["durations"]] if study == "A" else [
                (duration, kind) for duration in b_protocol["configuration"]["durations"] for kind in b_protocol["configuration"]["schedules"]]
            for duration, kind in cells:
                path = a / "records" / f"repeat_{seed}_D{duration}_data.npz" if study == "A" else b / "parts" / f"repeat_{seed}_D{duration}_{kind}.npz"
                saved = arrays(path)
                series_key = "x" if study == "A" else "observed_series"
                if series_key not in saved:
                    raise ValueError(f"No observed series retained in planned cell: {path.name}")
                x = saved[series_key]
                if x.shape != (timesteps, int(config["n_nodes"])):
                    raise ValueError(f"Unexpected observed-series coverage: {path.name}")
                np.testing.assert_array_equal(saved["regimes"], saved["regimes_full"][burn:])
                if len(saved["regimes_full"]) != timesteps + burn:
                    raise ValueError("The retained full schedule must include the burn-in")
                trajectories.append({"study": study, "repeat_seed": seed, "duration": duration, "schedule": kind,
                                     "source": path.relative_to(root).as_posix(),
                                     "observed_start_full_time": burn, "observed_stop_full_time_exclusive": timesteps + burn,
                                     "burn_in_trajectory_values_saved": False,
                                     **trajectory_metrics(x)})
    return pd.DataFrame(mechanisms), pd.DataFrame(trajectories)


def summaries(mechanisms: pd.DataFrame, trajectories: pd.DataFrame) -> pd.DataFrame:
    result = []
    for study, m in mechanisms.groupby("study", sort=True):
        t = trajectories[trajectories.study == study]
        radii = m[["state0_companion_radius", "state1_companion_radius"]].to_numpy().reshape(-1)
        row = {"study": study, "independent_mechanisms": len(m), "state_coefficient_tensors": len(radii),
               "observed_trajectories": len(t), "finite_observed_trajectories": int(t.all_observed_entries_finite.sum()),
               "postconstruction_radius_min": float(radii.min()), "postconstruction_radius_max": float(radii.max()),
               "maximum_absolute_observed_value": float(t.maximum_absolute_observed_value.max()),
               "early_rms_min": float(t.early_rms.min()), "early_rms_max": float(t.early_rms.max()),
               "late_rms_min": float(t.late_rms.min()), "late_rms_max": float(t.late_rms.max()),
               "late_to_early_rms_ratio_min": float(t.late_to_early_rms_ratio.min()),
               "late_to_early_rms_ratio_max": float(t.late_to_early_rms_ratio.max()),
               "rms_ratio_evaluable_trajectories": int(t.rms_ratio_evaluable.sum()),
               "state0_self_edges_min": int(m.state0_self_edges.min()), "state0_self_edges_max": int(m.state0_self_edges.max()),
               "state1_self_edges_min": int(m.state1_self_edges.min()), "state1_self_edges_max": int(m.state1_self_edges.max()),
               "realised_retained_fraction_min": float(m.realised_retained_fraction.min()),
               "realised_retained_fraction_max": float(m.realised_retained_fraction.max()),
               "base_empty_fallback_compatible_mechanisms": int(m.base_empty_fallback_compatible.sum()),
               "state1_empty_fallback_compatible_mechanisms": int(m.state1_empty_fallback_compatible.sum())}
        result.append(row)
    return pd.DataFrame(result)


def table(frame: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Study & Mechanisms & Paths & Actual radius range & Max $|X|$ & Late/early RMS range & Finite paths \\", r"\midrule"]
    for r in frame.itertuples(index=False):
        lines.append(f"{r.study} & {r.independent_mechanisms} & {r.observed_trajectories} & "
                     f"[{r.postconstruction_radius_min:.3f}, {r.postconstruction_radius_max:.3f}] & "
                     f"{r.maximum_absolute_observed_value:.3f} & "
                     f"[{r.late_to_early_rms_ratio_min:.3f}, {r.late_to_early_rms_ratio_max:.3f}] & "
                     f"{r.finite_observed_trajectories}/{r.observed_trajectories}" + r" \\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"])


def audit(root: Path = ROOT, output: Path | None = None) -> dict:
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root / "results/information_sciences/generator_audit"
    frozen_dirs = [root / "results/information_sciences" / name for name in ("duration_confirmation", "schedule_robustness")]
    if any(output.is_relative_to(directory) for directory in frozen_dirs):
        raise ValueError("Post hoc diagnostics must stay outside both frozen study directories")
    before = frozen_hashes(root)
    source_before = frozen_source_hashes(root)
    check_declared_sources(root)
    mechanisms, trajectories = collect(root)
    summary = summaries(mechanisms, trajectories)
    after = frozen_hashes(root)
    source_after = frozen_source_hashes(root)
    if before != after or source_before != source_after:
        raise RuntimeError("The read-only audit changed frozen experimental files")
    output.mkdir(parents=True, exist_ok=True)
    mechanisms.to_csv(output / "mechanisms.csv", index=False)
    trajectories.to_csv(output / "observed_trajectories.csv", index=False)
    summary.to_csv(output / "study_summary.csv", index=False)
    (output / "tab_generator_diagnostics.tex").write_text(table(summary) + "\n", encoding="utf-8")
    for name, value in (("frozen_before_sha256.json", before), ("frozen_after_sha256.json", after)):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, value in (("frozen_sources_before_sha256.json", source_before), ("frozen_sources_after_sha256.json", source_after)):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "role": "post hoc descriptive audit of retained outputs; no preset stability threshold or general stability claim",
        "new_simulations": 0, "new_fits": 0, "frozen_files_unchanged": len(before), "frozen_source_files_unchanged": len(source_before),
        "definitions": {"radius": "companion-matrix radius recomputed from each retained final state coefficient tensor; independent mechanisms counted once",
                        "max_abs": "maximum absolute value over all saved observed time points and nodes",
                        "RMS": "sqrt(mean(X**2)) over every node and the first/last 300 saved observed time points, calculated with overflow-safe scaling",
                        "growth_ratio": "late-block RMS divided by early-block RMS; undefined if early RMS is zero; descriptive endpoint ratio, not a fitted growth rate",
                        "thresholds": "none; endpoint block size 300 selected post hoc to match the analysis window, not a prespecified diagnostic cutoff"},
        "coverage": {"A": "saved observed X covers full times [100,1300), T=1200; 100 burn-in X values are not retained",
                     "B": "saved observed X covers full times [100,3700), T=3600; 100 burn-in X values are not retained",
                     "exogenous_inputs": "initial history at full times0..2 and full innovations/schedules are retained; missing burn-in X is not reconstructed here",
                     "runtime_check": "the frozen simulator checks finiteness before discarding burn-in, but this post hoc audit independently inspects only saved observed X",
                     "pre_rescale_radius": "not saved; the number of applied rescales is not inferred from final radii alone"},
        "support_rule": {"additions_eligibility": "absent in the original state-A base support, not absent after retention deletions",
                         "p_A_nonself": 0.18, "o_conditional_retention_probability": 0.5,
                         "p_add_if_originally_absent": 0.18 * (1 - 0.5),
                         "p_B_nonself_ignoring_empty_fallbacks": 0.18 * 0.5 + (1 - 0.18) * 0.18 * (1 - 0.5),
                         "self_edges": "state-A diagonals are cleared; state-B additions include those originally absent self positions at probability0.09",
                         "base_empty_fallback": "force one nonself lag1 edge from random source to its cyclic successor",
                         "B_empty_fallback": "if the entire proposed B support is empty, copy original A support",
                         "overlap": "o is per-base-edge retention probability, not a guarantee that realised shared/A edge fraction equals0.5",
                         "fallback_compatibility": "flags only possible activation from final support; false excludes activation, true does not prove it"},
        "study_summary": summary.to_dict("records"),
        "source_sha256": {"src/stcg/mechanism.py": sha(root / "src/stcg/mechanism.py"),
                          "scripts/aggregation_study_common.py": sha(root / "scripts/aggregation_study_common.py")},
    }
    (output / "diagnostic_summary.json").write_text(json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    names = ["mechanisms.csv", "observed_trajectories.csv", "study_summary.csv", "tab_generator_diagnostics.tex",
             "frozen_before_sha256.json", "frozen_after_sha256.json", "frozen_sources_before_sha256.json",
             "frozen_sources_after_sha256.json", "diagnostic_summary.json"]
    manifest = {"audit_builder_sha256": sha(Path(__file__)), "role": report["role"], "frozen_files_unchanged": len(before),
                "frozen_source_files_unchanged": len(source_before), "frozen_source_sha256": source_after,
                "before_after_frozen_hashes_equal": True, "before_after_frozen_source_hashes_equal": True,
                "sha256": {name: sha(output / name) for name in names}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json_safe(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = audit(args.root, args.output_dir)
    print(json.dumps({"role": report["role"], "frozen_files_unchanged": report["frozen_files_unchanged"],
                      "study_summary": report["study_summary"], "new_simulations": 0, "new_fits": 0}, indent=2))


if __name__ == "__main__":
    main()
