"""Regressions for the post hoc controls; no simulation or fitting."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import json
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_score_preserving_controls as controls


def example():
    raw = np.zeros((4, 1, 2, 2))
    raw[:, 0, 0, 1] = [0, 1, 1, 3]
    raw[:, 0, 1, 0] = [5, 5, 4, 4]
    forced = raw.copy()
    forced[:2] = raw[:2].mean(axis=0)
    forced[2:] = raw[2:].mean(axis=0)
    native = np.broadcast_to(raw.max(axis=0), raw.shape).copy()
    truth = np.zeros_like(raw, dtype=bool)
    truth[:, 0, 0, 1] = [0, 0, 1, 1]
    truth[:, 0, 1, 0] = True
    changed = np.zeros(raw.shape[1:], dtype=bool)
    changed[0, 0, 1] = True
    return raw, forced, native, truth, changed


def test_methods_and_byte_exact_duplicate_endpoints():
    raw, hard, native, _, _ = example()
    methods = controls.build_control_emissions(raw, hard, native, False)
    assert len(methods) == len(controls.method_spec()) == 13
    for name in ("shrink_l100", "residual_hard_l100", "raw_fallback"):
        assert methods[name].tobytes() == raw.tobytes()
    assert methods["residual_hard_l000"].tobytes() == hard.tobytes()
    assert not np.shares_memory(methods["raw_fallback"], raw)


def test_gate_acceptance_controls_only_raw_fallback_branch():
    raw, hard, native, _, _ = example()
    accepted = controls.build_control_emissions(raw, hard, native, True)
    rejected = controls.build_control_emissions(raw, hard, native, False)
    np.testing.assert_array_equal(accepted["raw_fallback"], hard)
    np.testing.assert_array_equal(rejected["raw_fallback"], raw)
    np.testing.assert_array_equal(rejected["native_hard"], native)


def test_actual_shrink_and_residual_formula():
    raw, hard, native, _, _ = example()
    result = controls.build_control_emissions(raw, hard, native, False)
    np.testing.assert_array_equal(result["shrink_l025"], .25 * raw + .75 * raw.mean(axis=0))
    np.testing.assert_array_equal(result["residual_hard_l075"], .75 * raw + .25 * hard)


def test_positive_affine_preserves_ties_and_per_edge_auc():
    raw, hard, native, truth, changed = example()
    rows, _, checks = controls.evaluate_controls(raw, hard, native, False, truth, changed)
    by_name = {row["method"]: row for row in rows}
    for lam in controls.LAMBDAS:
        assert by_name[f"shrink_l{round(lam*100):03d}"]["temporal_delta_auroc"] == by_name["raw_window"]["temporal_delta_auroc"]
    assert not any(row["new_tied_pairs"] or row["lost_tied_pairs"] or row["reversed_pairs"] for row in checks)
    assert by_name["native_hard"]["temporal_delta_auroc"] == .5
    assert by_name["raw_fallback"]["temporal_delta_auroc"] != .5


def test_floating_point_tie_collapse_is_reported_not_repaired():
    raw = np.zeros((3, 1, 2, 2))
    raw[:, 0, 0, 1] = [1., np.nextafter(1., np.inf), 2.]
    truth = np.zeros_like(raw, dtype=bool)
    truth[:, 0, 0, 1] = [0, 1, 1]
    changed = np.zeros(raw.shape[1:], dtype=bool); changed[0, 0, 1] = True
    rows, _, checks = controls.evaluate_controls(raw, raw, raw, True, truth, changed)
    diagnostic = next(row for row in checks if row["lambda"] == .25 and row["source"] == 0)
    assert diagnostic["new_tied_pairs"] > 0
    assert diagnostic["edge_auroc_difference"] < 0
    by_name = {row["method"]: row for row in rows}
    assert by_name["shrink_l025"]["temporal_delta_auroc"] < by_name["raw_window"]["temporal_delta_auroc"]


def test_subnormal_and_extreme_comparisons_do_not_use_overflowing_subtraction():
    scores = np.array([[-1e308, 0.], [1e308, np.nextafter(0., 1.)]])
    truth = np.array([[0, 0], [1, 1]], dtype=bool)
    np.testing.assert_array_equal(controls.edge_aurocs(truth, scores), [1., 1.])


def test_edge_and_pooled_metrics_match_public_auc_with_ties():
    raw, _, _, truth, changed = example()
    value = controls.score_metrics(raw, truth, changed)
    mask = controls.nonself(raw.shape[1:])
    assert value["pooled_temporal_auroc"] == roc_auc_score(truth[:, mask].ravel(), raw[:, mask].ravel())
    assert value["temporal_delta_auroc"] == roc_auc_score(truth[:, 0, 0, 1], raw[:, 0, 0, 1])
    raw[:, 0, 0, 0] = 1e100
    assert controls.score_metrics(raw, truth, changed)["pooled_temporal_auroc"] == value["pooled_temporal_auroc"]


def test_shrink_can_change_pooled_order_without_temporal_change():
    rng = np.random.default_rng(36)
    raw = rng.uniform(0, 10, (7, 1, 3, 3))
    truth = rng.integers(0, 2, raw.shape).astype(bool)
    changed = np.ones(raw.shape[1:], dtype=bool)
    rows, _, _ = controls.evaluate_controls(raw, raw, raw, True, truth, changed)
    d = {r["method"]: r for r in rows}
    assert d["shrink_l025"]["pooled_temporal_auroc"] != d["raw_window"]["pooled_temporal_auroc"]
    assert d["shrink_l025"]["temporal_delta_auroc"] == d["raw_window"]["temporal_delta_auroc"]


def test_undefined_targets_and_nonfinite_scores_remain_explicit():
    raw, _, _, truth, changed = example()
    none = controls.score_metrics(raw, truth, np.zeros_like(changed))
    assert none["status"] == "undefined" and np.isnan(none["temporal_delta_auroc"])
    assert none["undefined_reason"] == "no_support_change"
    raw[0, 0, 0, 1] = np.inf
    bad = controls.score_metrics(raw, truth, changed)
    assert bad["status"] == "failed" and np.isnan(bad["pooled_temporal_auroc"])
    assert bad["undefined_reason"] == "nonfinite_scored_predictions"


@pytest.mark.parametrize("mutation", ["shape", "gate", "lambda"])
def test_invalid_api_inputs_fail_closed(mutation):
    raw, hard, native, _, _ = example()
    with pytest.raises(ValueError):
        controls.build_control_emissions(raw, hard[:2] if mutation == "shape" else hard, native,
                                        1 if mutation == "gate" else True,
                                        lambdas=(.5,) if mutation == "lambda" else controls.LAMBDAS)


def test_bootstrap_uses_repeat_differences_and_keeps_missing_denominators():
    records = []
    for seed in range(8):
        for name, value in [("raw_window", seed/10), ("forced_hard", seed/10+.1), ("shrink_l025", seed/10)]:
            records.append({"study": "A", "scorer": "lasso", "duration": 300, "schedule": "fixed_phase", "cell": "D300_fixed_phase",
                "repeat_seed": seed, "method": name, "temporal_delta_auroc": value if seed != 7 else np.nan,
                "pooled_temporal_auroc": value, "status": "ok" if seed != 7 else "undefined"})
    summary, contrasts = controls.summarise(pd.DataFrame(records))
    row = contrasts[(contrasts.method == "forced_hard") & (contrasts.reference == "raw_window") &
                    (contrasts.metric == "temporal_delta_auroc")].iloc[0]
    assert row.n_planned == 8 and row.n_paired == 7 and row.n_missing_pairs == 1
    np.testing.assert_allclose([row["mean"], row.ci95_low, row.ci95_high], [.1,.1,.1])
    assert not any("p_" in name for name in contrasts)
    assert summary.n_missing.max() == 1


def test_failed_case_retains_every_method_even_if_all_cases_fail(monkeypatch):
    failure = {"study": "A", "scorer": "ols", "duration": 300, "schedule": "fixed_phase", "cell": "D300_fixed_phase",
               "repeat_seed": 50000, "status": "failed", "error": "retained rank failure"}
    monkeypatch.setattr(controls, "iter_retained_cases", lambda root: iter([failure]))
    tables, report = controls.compute(Path('.'))
    assert len(tables["detail.csv"]) == 13
    assert (tables["detail.csv"].status == "failed").all()
    assert report["failed_cases"] == 1 and report["affine_edge_checks"] == 0


def test_adapter_is_lazy_and_does_not_fit_or_write():
    generator = controls.iter_retained_cases(Path("nonexistent-test-root"))
    assert iter(generator) is generator
    with pytest.raises(FileNotFoundError): next(generator)


@pytest.fixture
def miniature_analysis(tmp_path, monkeypatch):
    # Isolate the filesystem/protocol lifecycle from scientific input loading.
    # The numerical controls still run on real synthetic arrays, not mocked metrics.
    raw, hard, native, truth, changed = example()
    snapshot = {"frozen_A_B_sha256": {str(k): "0" * 64 for k in range(1915)},
                "frozen_source_sha256": {str(k): "0" * 64 for k in range(12)}}
    monkeypatch.setattr(controls, "input_snapshot", lambda root: snapshot)
    case = {"study": "A", "scorer": "lasso", "duration": 300, "schedule": "fixed_phase", "cell": "D300_fixed_phase",
            "seed": 50000, "repeat_seed": 50000, "status": "ok", "error": "", "raw": raw, "forced": hard,
            "native": native, "active_truth": truth, "changing_mask": changed, "gate_accepted": False}
    monkeypatch.setattr(controls, "iter_retained_cases", lambda root: iter([case]))
    monkeypatch.setattr(controls, "check_reference_metrics", lambda detail, root: 3)
    output = tmp_path / "analysis"
    controls.freeze(output)
    return output, snapshot


def test_freeze_is_only_protocol_and_run_refuses_overwrite(miniature_analysis):
    output, _ = miniature_analysis
    assert set(controls.directory_hashes(output)) == {"protocol.json", "input_manifest.json"}
    with pytest.raises(ValueError, match="empty"):
        controls.freeze(output)
    controls.run(output)
    with pytest.raises(ValueError, match="refuse to overwrite"):
        controls.run(output)


def test_rebuild_and_verifier_preserve_every_byte(miniature_analysis):
    output, _ = miniature_analysis
    controls.run(output)
    before = controls.directory_hashes(output)
    report = controls.verify(output)
    assert report["read_only"] and report["new_fits"] == 0
    assert before == controls.directory_hashes(output)
    controls.rebuild(output)
    assert before == controls.directory_hashes(output)


def test_hash_valid_but_numerically_altered_metric_is_rejected(miniature_analysis):
    output, _ = miniature_analysis
    controls.run(output)
    detail = pd.read_csv(output / "detail.csv")
    detail.loc[0, "temporal_delta_auroc"] = .123
    detail.to_csv(output / "detail.csv", index=False)
    seal = json.loads((output / "manifest.json").read_text())
    seal["sha256"]["detail.csv"] = controls.sha(output / "detail.csv")
    (output / "manifest.json").write_bytes(controls.json_bytes(seal))
    with pytest.raises(AssertionError): controls.verify(output)


def test_frozen_input_changes_are_rejected_before_any_rebuild(miniature_analysis):
    output, snapshot = miniature_analysis
    controls.run(output)
    snapshot["frozen_source_sha256"]["0"] = "1" * 64
    with pytest.raises(ValueError, match="inputs changed"):
        controls.rebuild(output)


@pytest.mark.parametrize("all_targets_unobserved", [False, True])
def test_undefined_auc_is_not_a_nonzero_event(monkeypatch, all_targets_unobserved):
    raw, hard, native, truth, changed = example()
    if all_targets_unobserved: truth[:] = False
    case = {"study": "A", "scorer": "lasso", "duration": 300, "schedule": "fixed_phase", "cell": "D300_fixed_phase",
            "seed": 50000, "repeat_seed": 50000, "status": "ok", "error": "", "raw": raw, "forced": hard,
            "native": native, "active_truth": truth, "changing_mask": changed, "gate_accepted": False}
    monkeypatch.setattr(controls, "iter_retained_cases", lambda root: iter([case]))
    tables, diagnostics = controls.compute(Path('.'))
    assert tables["affine_checks.csv"].edge_auroc_difference.isna().any()
    assert tables["affine_events.csv"].empty
    assert diagnostics["affine_event_rows"] == 0
