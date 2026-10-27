"""Regression checks for the ESS refit and labelled-degree null."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_ess_alignment as ess
from stcg.randomization import directed_degree_null


def _support():
    graph = np.zeros((6, 6), dtype=int)
    for source, target in [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 0), (5, 2)]:
        graph[source, target] = 1
    return graph


def test_null_preserves_every_labelled_degree_without_modifying_input():
    graph = _support()
    original = graph.copy()
    draws = directed_degree_null(graph, 30, seed=12)
    np.testing.assert_array_equal(graph, original)
    assert not np.diagonal(draws, axis1=1, axis2=2).any()
    assert (draws.sum(axis=1) == graph.sum(axis=0)).all()
    assert (draws.sum(axis=2) == graph.sum(axis=1)).all()
    assert len(np.unique(draws.reshape(30, -1), axis=0)) > 1
    np.testing.assert_array_equal(draws, directed_degree_null(graph, 30, seed=12))
    assert not np.array_equal(draws, directed_degree_null(graph, 30, seed=13))


def test_directed_triangle_can_reverse_despite_no_valid_two_edge_switch():
    graph = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    draws = directed_degree_null(graph, 80, seed=3)
    assert any(np.array_equal(sample, graph) for sample in draws)
    assert any(np.array_equal(sample, graph.T) for sample in draws)


def test_diagnostics_preserve_random_stream_and_count_rejected_proposals():
    graph = _support()
    counts = {}
    expected = directed_degree_null(graph, 35, seed=3, burn_in_per_edge=7, thinning_per_edge=4)
    actual = directed_degree_null(graph, 35, seed=3, burn_in_per_edge=7,
                                  thinning_per_edge=4, diagnostics=counts)
    np.testing.assert_array_equal(actual, expected)
    for phase, proposals in (("burn_in", 7 * graph.sum()), ("sampling", 35 * 4 * graph.sum())):
        assert sum(move["attempted"] for move in counts[phase].values()) == proposals
        assert all(0 <= move["accepted"] <= move["attempted"] for move in counts[phase].values())
        assert sum(move["accepted"] for move in counts[phase].values()) < proposals
    assert counts["sampling"]["three_cycle"]["attempted"] > 0


def test_triangle_diagnostics_distinguish_cycle_reversals_from_edge_switches():
    graph = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    counts = {}
    directed_degree_null(graph, 80, seed=3, diagnostics=counts)
    assert counts["sampling"]["two_edge"]["accepted"] == 0
    assert counts["sampling"]["three_cycle"]["accepted"] > 0


def test_chain_diagnostics_detect_correlation_and_leave_constant_tail_undefined():
    from diagnose_ess_null import chain_statistics
    rng = np.random.default_rng(28)
    independent = rng.normal(size=5000)
    correlated = np.zeros(5000)
    for i in range(1, len(correlated)):
        correlated[i] = 0.9 * correlated[i - 1] + independent[i]
    assert chain_statistics(correlated)["ess"] < chain_statistics(independent)["ess"] / 5
    assert all(np.isnan(value) for value in chain_statistics(np.zeros(5000)).values())


@pytest.mark.parametrize("graph", [np.zeros((4, 4), int), np.ones((4, 4), int) - np.eye(4, dtype=int)])
def test_fixed_degree_classes_terminate_without_invalid_edges(graph):
    draws = directed_degree_null(graph, 4)
    assert (draws == graph).all()
    # AUROC is undefined for the empty/complete off-diagonal truth; it must not
    # be interpreted as a significant comparison (the caller omits empty truth).


def test_unique_nonempty_degree_class_has_conservative_p_value():
    graph = np.zeros((4, 4), int)
    graph[0, 1:] = 1
    scores = graph.astype(float)
    mean, p = ess.permutation_null(scores, graph, 10)
    assert mean == 1.0 and p == 1.0


def test_permutation_test_uses_valid_samples_and_plus_one_correction():
    graph = _support()
    scores = np.random.default_rng(9).random(graph.shape)
    draws = directed_degree_null(graph, 20, seed=6)
    mean, p = ess.permutation_null(scores, graph, 20, null_supports=draws)
    observed = ess.edge_auc(scores[None], graph[None])
    values = [ess.edge_auc(scores[None], sample[None]) for sample in draws]
    assert mean == pytest.approx(np.mean(values))
    assert p == pytest.approx((1 + sum(value >= observed for value in values)) / 21)
    invalid = draws.copy()
    invalid[0, 0, 0] = True
    with pytest.raises(ValueError, match="preserve"):
        ess.permutation_null(scores, graph, 20, null_supports=invalid)


@pytest.mark.parametrize("score_mode", ["tied", "constant", "negative"])
def test_vectorized_rank_sum_matches_scalar_auc_with_ties(score_mode):
    graph = _support()
    scores = np.arange(graph.size).reshape(graph.shape).astype(float)
    if score_mode == "tied":
        scores %= 3
    elif score_mode == "constant":
        scores[:] = 1
    else:
        scores *= -1
    draws = directed_degree_null(graph, 40, seed=8)
    observed = ess.edge_auc(scores[None], graph[None])
    scalar = np.array([ess.edge_auc(scores[None], sample[None]) for sample in draws])
    mean, p = ess.permutation_null(scores, graph, 40, null_supports=draws)
    assert mean == pytest.approx(scalar.mean(), abs=1e-15)
    assert p == (1 + int((scalar >= observed).sum())) / 41


def test_default_protocol_is_5000_draws_and_all_five_null_seeds():
    args = ess.parse_args([])
    assert args.draws == 5000
    assert args.null_seeds == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("arguments", [
    ["--draws", "0"], ["--null-seeds", "1", "2"],
    ["--null-seeds", "0", "0"], ["--null-seeds", "0", "-1"],
])
def test_cli_rejects_invalid_draws_or_missing_reference_seed(arguments):
    with pytest.raises(SystemExit) as error:
        ess.parse_args(arguments)
    assert error.value.code == 2


def test_holm_families_are_separate_for_each_monte_carlo_seed():
    rows = []
    for seed, small in [(0, 0.01), (1, 0.02)]:
        for imputation in ess.IMPUTATIONS:
            for cell in range(8):
                rows.append({"null_seed": seed, "imputation": imputation,
                             "p_raw": small if imputation == "ffill" and cell == 0 else 1.0})
    result = ess.correct_families(pd.DataFrame(rows))
    for seed, small in [(0, 0.01), (1, 0.02)]:
        cell = result[(result.null_seed == seed) & (result.p_raw == small)].iloc[0]
        assert cell.p_holm == pytest.approx(8 * small)
        assert cell.p_holm_all_imputations == pytest.approx(32 * small)
    assert result[result.imputation != "ffill"].p_holm.isna().all()


def test_ess_pipeline_refits_each_detected_segment(monkeypatch):
    x = pd.DataFrame(np.random.default_rng(4).normal(size=(600, 4)), columns=list("abcd"))
    adjacency = pd.DataFrame(np.array([[0, 2, 1, 0], [0, 0, 2, 0], [0, 0, 0, 2], [1, 0, 0, 0]]))
    monkeypatch.setattr(ess, "load_subsystem_series", lambda strategy: (x, adjacency, {}))
    rows = ess.run_strategy("ffill", SimpleNamespace(max_lag=1, draws=4))
    pelt = [row for row in rows if row["method"] == "pelt_var_lasso"]
    assert len(pelt) == 2
    assert all(row["n_segments"] > 1 for row in pelt)
    assert all(row["n_refit_ranges"] == row["n_segments"] for row in pelt)
    assert all(row["null_model"] == "labelled_in_out_degree_switch_chain" for row in rows)


def test_null_graphs_are_shared_across_methods_and_imputations(monkeypatch):
    x = pd.DataFrame(np.random.default_rng(4).normal(size=(600, 4)), columns=list("abcd"))
    adjacency = pd.DataFrame(np.array([[0, 2, 1, 0], [0, 0, 2, 0], [0, 0, 0, 2], [1, 0, 0, 0]]))
    monkeypatch.setattr(ess, "load_subsystem_series", lambda strategy: (x, adjacency.copy(), {}))
    calls = []

    def sample(support, draws, seed):
        calls.append((int(support.sum()), seed))
        return directed_degree_null(support, draws, seed)

    monkeypatch.setattr(ess, "directed_degree_null", sample)
    args = SimpleNamespace(max_lag=1, draws=4, null_seeds=[0, 1])
    cache = {}
    first = ess.run_strategy("ffill", args, cache)
    second = ess.run_strategy("bfill", args, cache)
    assert len(calls) == 4  # Two edge sets x two seeds, independent of methods/imputations.
    assert len(first) == len(second) == 16
    assert set(row["null_seed"] for row in first) == {0, 1}


def test_current_ess_results_record_corrected_protocol_and_holm_family():
    frame = pd.read_csv(Path(__file__).resolve().parents[1] / "results/tables/ess_alignment.csv")
    assert len(frame) == 32
    assert not frame.duplicated(["method", "imputation", "binarisation"]).any()
    assert set(frame.null_model) == {"labelled_in_out_degree_switch_chain"}
    pelt = frame[frame.method == "pelt_var_lasso"]
    assert (pelt.n_segments == 23).all()
    assert (pelt.n_refit_ranges == pelt.n_segments).all()
    primary = frame[frame.imputation == "ffill"]
    np.testing.assert_allclose(primary.p_holm, ess.holm(primary.p_raw.tolist()))
    assert frame[frame.imputation != "ffill"].p_holm.isna().all()


def test_seed_sensitivity_results_have_every_seed_and_preserve_reference_report():
    root = Path(__file__).resolve().parents[1]
    reference = pd.read_csv(root / "results/tables/ess_alignment.csv")
    all_seeds = pd.read_csv(root / "results/tables/ess_alignment_seed_sensitivity.csv")
    assert len(all_seeds) == 160
    assert set(all_seeds.null_seed) == set(range(5))
    assert (all_seeds.null_draws == 5000).all()
    assert not all_seeds.duplicated(["method", "imputation", "binarisation", "null_seed"]).any()
    pd.testing.assert_frame_equal(reference, all_seeds[all_seeds.null_seed == 0].reset_index(drop=True))
    np.testing.assert_allclose(all_seeds.p_raw, (all_seeds.null_exceedances + 1) / 5001)
    for _, seed_frame in all_seeds.groupby("null_seed"):
        primary = seed_frame[seed_frame.imputation == "ffill"]
        np.testing.assert_allclose(primary.p_holm, ess.holm(primary.p_raw.tolist()))
        np.testing.assert_allclose(seed_frame.p_holm_all_imputations, ess.holm(seed_frame.p_raw.tolist()))


def test_saved_diagnostic_traces_agree_with_counts_and_primary_results():
    root = Path(__file__).resolve().parents[1]
    traces = pd.read_csv(root / "results/tables/ess_null_traces.csv")
    diagnostics = pd.read_csv(root / "results/tables/ess_null_diagnostics.csv")
    primary = pd.read_csv(root / "results/tables/ess_alignment_seed_sensitivity.csv")
    assert len(diagnostics) == 12 and len(traces) == 60000
    for row in diagnostics.itertuples(index=False):
        trace = traces[(traces.binarisation == row.binarisation) & (traces.null_seed == row.null_seed)
                       & (traces.burn_in_per_edge == row.burn_in_per_edge)]
        np.testing.assert_array_equal(trace.draw, np.arange(1, 5001))
        np.testing.assert_array_equal(trace.tail_event, (trace.null_auroc >= row.observed_auroc).astype(int))
        assert trace.tail_event.sum() == row.null_exceedances
        assert trace.null_auroc.mean() == pytest.approx(row.null_auroc_mean, abs=1e-15)
        assert row.p_raw == pytest.approx((1 + row.null_exceedances) / 5001)
        assert row.sampling_two_edge_attempted + row.sampling_three_cycle_attempted == 5000 * 10 * row.n_edges
        if row.burn_in_per_edge == 100:
            expected = primary[(primary.method == "lag_corr") & (primary.imputation == "ffill")
                               & (primary.null_seed == row.null_seed) & (primary.binarisation == row.binarisation)].iloc[0]
            assert row.null_exceedances == expected.null_exceedances
            assert row.p_raw == expected.p_raw


@pytest.mark.parametrize("support", [np.eye(3), np.ones((2, 3)), np.array([[0, 2], [0, 0]])])
def test_null_rejects_invalid_adjacencies(support):
    with pytest.raises(ValueError):
        directed_degree_null(support, 2)
