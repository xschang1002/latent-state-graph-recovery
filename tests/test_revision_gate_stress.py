"""Validate production equivalence and label-blind stress interventions."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_revision_gate_stress as stress
from stcg import STCGStateConfig, generate_lagged_var, stcg_v1_diagnostics
from _state_selection import select_state_candidate_index
import run_real_state_count_selection as real_selector


def test_modified_generator_matches_production_and_uses_actual_mixed_truth():
    reference = generate_lagged_var(timesteps=300, switching=True, seed=123)
    extended = stress.modified_switch(123, 300)
    np.testing.assert_array_equal(extended.x, reference.x)
    np.testing.assert_array_equal(extended.coefficients, reference.coefficients)
    np.testing.assert_array_equal(extended.regimes, reference.regimes)
    weak = stress.modified_switch(123, 300, mixture=0.25)
    np.testing.assert_allclose(weak.coefficients[1], 0.75 * reference.coefficients[0] + 0.25 * reference.coefficients[1])
    np.testing.assert_array_equal(weak.edge_truth, np.any(np.abs(weak.coefficients) > 0, axis=0))
    assert np.max(np.abs(weak.coefficients).sum(axis=(1, 2))) <= 0.85 + 1e-12
    rare = stress.modified_switch(123, 300, switch_fraction=0.95)
    assert np.count_nonzero(rare.regimes) == 15


def test_cached_pipeline_is_identical_to_direct_production_for_all_gate_variants():
    x = generate_lagged_var(timesteps=420, switching=True, seed=124).x
    base = STCGStateConfig(window_size=100, stride=50, seed=30124)
    with threadpool_limits(limits=1):
        local = stcg_v1_diagnostics(x, 3, base).window_scores
        for changes in stress.GATE_VARIANTS.values():
            config = replace(base, **changes)
            direct = stcg_v1_diagnostics(x, 3, config)
            cached = stress.cached_diagnostics(x, local, config)
            np.testing.assert_array_equal(direct.scores, cached.scores)
            np.testing.assert_array_equal(direct.labels, cached.labels)
            assert direct.states_accepted == cached.states_accepted
            assert direct.state_improvement == cached.state_improvement


def test_full_selector_scores_call_production_and_factor_removals_are_isolated():
    candidate = {"state_improvement": 0.2, "temporal_coherence": 0.7,
                 "min_cluster_fraction": 0.2, "state_k": 3}
    for objective, production in (("stable", stress.state_count_selection_score),
                                  ("warning", stress.stcg_transition_selection_score)):
        full, balance = production(0.2, 0.7, 0.2, 3)
        assert stress.selector_score(candidate, objective, "full") == full
        assert stress.selector_score(candidate, objective, "no_balance") == full / balance
        assert stress.selector_score(candidate, objective, "no_parsimony") == full * 2
    low_coherence = {**candidate, "temporal_coherence": 0.3}
    assert stress.selector_score(low_coherence, "stable", "full") == 0
    assert stress.selector_score(low_coherence, "stable", "no_coherence_weight") == 0
    assert stress.selector_score(low_coherence, "stable", "no_coherence_screen") > 0
    assert stress.selector_score(low_coherence, "stable", "no_coherence_all") > stress.selector_score(low_coherence, "stable", "no_coherence_screen")


def test_selector_keeps_production_zero_score_tie_and_is_blind_to_audit_metrics():
    candidates = [{"seed": 100, "candidate_state_count": k, "state_k": k,
                   "state_improvement": 0.03, "temporal_coherence": coherence,
                   "min_cluster_fraction": 1 / k, "state_ari": ari}
                  for k, coherence, ari in ((1, 0, 1), (2, 0.4, 0), (3, 0.3, 0.5))]
    selected, all_zero = stress.select_candidate(candidates, "stable", "full")
    assert all_zero and selected["candidate_state_count"] == 2
    changed = [{**row, "state_ari": 1e6 * row["candidate_state_count"], "edge_auc": -row["candidate_state_count"]} for row in candidates]
    again, _ = stress.select_candidate(changed, "stable", "full")
    assert again["candidate_state_count"] == selected["candidate_state_count"]
    corrected, corrected_zero = stress.select_candidate(candidates, "stable", "positive_fallback")
    assert corrected_zero and corrected["candidate_state_count"] == 1


def test_safe_selector_rejects_all_nonpositive_scores_and_preserves_positive_order():
    all_zero = [(0.0, 0.8, 3), (0.0, 0.0, 1), (0.0, 0.6, 2)]
    assert select_state_candidate_index(all_zero) == 1
    assert select_state_candidate_index(all_zero, "legacy") == 0
    positive = [(0.0, 0.0, 1), (0.2, 0.8, 3), (0.2, 0.8, 2)]
    assert select_state_candidate_index(positive) == select_state_candidate_index(positive, "legacy") == 2
    with pytest.raises(ValueError, match="include a fitted K=1"):
        select_state_candidate_index([(0.0, 0.8, 2), (0.0, 0.9, 3)])
    with pytest.raises(ValueError, match="finite"):
        select_state_candidate_index([(float("nan"), 0.8, 2), (0.0, 0.0, 1)])


def test_real_mark_selected_defaults_to_rejection_without_using_audit_labels():
    rows = [{"seed": 0, "candidate_state_count": k, "selection_score": 0,
             "temporal_coherence": k / 10, "balanced_accuracy": k * 100, "macro_f1": k * 100}
            for k in (1, 2, 3)]
    real_selector.mark_selected(rows)
    assert [row["candidate_state_count"] for row in rows if row["selected"]] == [1]
    real_selector.mark_selected(rows, policy="legacy")
    assert [row["candidate_state_count"] for row in rows if row["selected"]] == [3]


def test_real_selector_cli_requires_k1_for_safe_policy(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_real_state_count_selection.py", "--state-counts", "2", "3"])
    with pytest.raises(SystemExit) as error:
        real_selector.parse_args()
    assert error.value.code == 2
    monkeypatch.setattr(sys, "argv", ["run_real_state_count_selection.py", "--state-counts", "2", "3", "--selection-policy", "legacy"])
    assert real_selector.parse_args().selection_policy == "legacy"


def test_controlled_fixtures_are_nonnegative_nonself_tensors_with_no_causal_truth():
    for name in stress.FIXTURES:
        tensors, labels = stress.fixture_scores(name, 100)
        assert tensors.shape == (120, 3, 8, 8) and len(labels) == 120
        assert np.min(tensors) >= 0
        assert not tensors[:, :, np.arange(8), np.arange(8)].any()
    with threadpool_limits(limits=1):
        rows, _ = stress.run_one("coherent_weak", 100, None)
    assert all(row["evidence_kind"] == "controlled_score_tensor" for row in rows)
    assert all(np.isnan(row["edge_auc"]) for row in rows)


def test_paired_changes_use_identical_seeds_and_preserve_direction():
    rows = [{"evidence_kind": "generator", "condition": "test", "analysis": "gate", "objective": "partition",
             "variant": variant, "seed": seed, "edge_auc": score}
            for seed in (100, 101) for variant, score in (("default", 0.7), ("no_fallback", 0.6))]
    results = stress.paired_differences(rows)
    assert len(results) == 1
    assert results[0]["n"] == 2 and results[0]["changed_seed_count"] == 2
    np.testing.assert_allclose([results[0]["mean_difference"], results[0]["ci_low"], results[0]["ci_high"]], -0.1)
