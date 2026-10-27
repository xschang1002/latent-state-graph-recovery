"""Independently reconcile additional review-response records and manuscript tables.

This verifier reads saved evidence, never refits models or treats initialization
seeds on the same real observations as independent data replicates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
CHECKS: list[str] = []
INPUTS: dict[str, str] = {}


def require(condition, description):
    if not bool(condition):
        raise AssertionError(description)
    CHECKS.append(description)


def equal(actual, expected, description):
    require(np.allclose(actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True), description)


def read(name, **kwargs):
    path = TABLES / name
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    kwargs.setdefault("float_precision", "round_trip")
    return pd.read_csv(path, **kwargs)


def verify_gate():
    detail = read("revision_gate_stress_detail.csv")
    summary = read("revision_gate_stress_summary.csv")
    paired = read("revision_gate_paired_changes.csv")
    metadata = read("revision_gate_run_metadata.csv").iloc[0]
    failures = read("revision_gate_failures.csv")
    candidates = read("revision_gate_candidates.csv")
    require(len(detail) == int(metadata.detail_rows) == 5320, "complete gate decision grid")
    require(len(candidates) == int(metadata.candidate_rows) == 1120, "complete gate candidate grid")
    require(len(failures) == int(metadata.failed_cases) == 0, "gate failure ledger is empty")
    keys = ["evidence_kind", "condition", "analysis", "objective", "variant"]
    require(not detail.duplicated(keys + ["seed"]).any(), "unique gate decisions")
    for key, group in detail.groupby(keys, sort=False):
        require(set(group.seed) == set(range(100, 120)), f"gate new-seed coverage: {key}")
    means = detail.groupby(keys).mean(numeric_only=True)
    stds = detail.groupby(keys).std(numeric_only=True)
    indexed = summary.set_index(keys)
    require(set(means.index) == set(indexed.index), "gate summary covers every condition and variant")
    for column in summary:
        suffix = "_mean" if column.endswith("_mean") else "_std" if column.endswith("_std") else None
        if suffix:
            metric = column[:-len(suffix)]
            source = means if suffix == "_mean" else stds
            equal(indexed.loc[source.index, column], source[metric], f"gate raw-to-summary {column}")
    decision_index = detail.set_index(keys + ["seed"]).sort_index()
    for row in paired.itertuples(index=False):
        prefix = (row.evidence_kind, row.condition, row.analysis, row.objective)
        candidate = decision_index.loc[(*prefix, row.variant), row.metric].sort_index()
        reference = decision_index.loc[(*prefix, row.reference), row.metric].sort_index()
        difference = (candidate - reference).dropna()
        require(len(difference) == row.n, f"gate paired denominator {prefix}/{row.variant}/{row.metric}")
        equal(difference.mean(), row.mean_difference, f"gate paired effect {prefix}/{row.variant}/{row.metric}")
    for condition, expected in [("stationary_many", 1), ("stationary_few", 13)]:
        subset = detail[(detail.condition == condition) & (detail.analysis == "selector") &
                        (detail.objective == "stable") & (detail.variant == "positive_fallback")]
        require(int(subset.false_split.sum()) == expected, f"corrected stable false splits: {condition}")
    audit = read("revision_gate_real_selector_policy_check.csv")
    require(len(audit) == 10 and not audit.choice_changed.any(), "all ten existing positive real selections unchanged")
    return {"decisions": len(detail), "candidates": len(candidates), "failures": len(failures)}


def verify_forward():
    detail = read("revision_real_forward_detail.csv")
    predictions = read("revision_real_forward_predictions.csv", dtype={"true_label": str, "predicted_label": str})
    origins = read("revision_real_forward_origins.csv")
    summary = read("revision_real_forward_summary.csv")
    plan = read("revision_real_plan.csv")
    keys = ["dataset", "label_column", "method"]
    origin_keys = keys + ["interval_id"]
    run_keys = origin_keys + ["seed"]
    require(len(detail) == 360 and detail.status.eq("ok").all(), "all 360 forward method/seed/origin fits recorded")
    require(len(predictions) == 24480, "all 24480 forward per-window predictions recorded")
    require(not detail.duplicated(run_keys).any(), "unique forward fit rows")
    require(not predictions.duplicated(run_keys + ["start", "end"]).any(), "unique forward predictions")
    indexed = detail.set_index(run_keys)
    for key, group in predictions.groupby(run_keys, sort=False):
        row = indexed.loc[key]
        require(len(group) == row.test_windows, f"forward prediction count: {key}")
        ordered = group.sort_values("start")
        require(((ordered.end - ordered.start) == 300).all(), f"complete 300-row test windows: {key}")
        require((ordered.start >= row.train_end).all() and (ordered.end <= row.test_end).all(), f"forward boundary isolation: {key}")
        require((ordered.start.to_numpy()[1:] >= ordered.end.to_numpy()[:-1]).all(), f"nonoverlapping forward windows: {key}")
        true, pred = group.true_label, group.predicted_label
        metrics = {"accuracy": accuracy_score(true, pred),
                   "balanced_accuracy": recall_score(true, pred, labels=sorted(set(true)), average="macro", zero_division=0),
                   "macro_f1": f1_score(true, pred, average="macro", zero_division=0)}
        for metric, value in metrics.items():
            equal(value, row[metric], f"prediction-to-metric {key}/{metric}")
        equal(len(set(true)), row.true_test_k, f"test label count {key}")
    for key, group in detail.groupby(origin_keys):
        require(set(group.seed) == set(range(5)), f"all initialization seeds at origin: {key}")
    actual_origins = detail.groupby(origin_keys)[["accuracy", "balanced_accuracy", "macro_f1"]].mean()
    saved_origins = origins.set_index(origin_keys)
    require(set(actual_origins.index) == set(saved_origins.index), "complete origin-level summary")
    for metric in actual_origins:
        equal(actual_origins[metric], saved_origins.loc[actual_origins.index, metric], f"average seeds within origin: {metric}")
    saved_summary = summary.set_index(keys)
    for metric in actual_origins:
        for statistic in ["mean", "min", "max"]:
            values = getattr(actual_origins.groupby(keys)[metric], statistic)()
            equal(values, saved_summary.loc[values.index, f"{metric}_{statistic}"], f"origin-level {metric}/{statistic}")
    unique = detail.drop_duplicates(["dataset", "label_column", "interval_id"])
    for row in unique.itertuples(index=False):
        planned = plan[(plan.protocol == "forward") & (plan.dataset == row.dataset) & (plan.interval_id == row.interval_id)]
        require(len(planned) == 1 and row.train_end == planned.iloc[0].train_end and row.test_start == planned.iloc[0].start and row.test_end == planned.iloc[0].end,
                f"frozen forward coordinates: {row.dataset}/{row.interval_id}")
    arem = unique[unique.dataset == "AReMActivities"]
    require(int(arem.test_recording_boundary_windows.sum()) == 70 and int(arem.test_windows.sum()) == 112,
            "AReM boundary limitation retained: 70/112 windows")
    return {"fits": len(detail), "predictions": len(predictions), "origins_per_label": 4,
            "distinct_label_windows": int(unique.test_windows.sum())}


def verify_segments():
    detail = read("revision_real_segment_detail.csv")
    summary = read("revision_real_segment_summary.csv")
    plan = read("revision_real_plan.csv")
    keys = ["dataset", "label_column", "interval_id", "method"]
    require(len(detail) == 36 and not detail.duplicated(keys).any(), "all 36 segment/label/method rows")
    fit_keys = ["dataset", "interval_id", "method"]
    fits = detail.drop_duplicates(fit_keys)
    require(len(fits) == 24, "24 distinct segment fits; Hydraulic labels reuse fits")
    rpc = fits[fits.method == "Regime-PCMCI K2"]
    require(len(rpc) == 8 and rpc.status.eq("timeout").all(), "all eight actual RPC wall timeouts retained")
    require(rpc.timeout_budget_seconds.eq(120).all() and rpc.worker_wall_seconds.ge(120).all(), "prespecified real RPC time budgets")
    require(rpc.error_free_annealings.isna().all() and rpc.window_ari.isna().all(), "no imputed success diagnostic or recovery score after timeout")
    for (dataset, interval), group in fits.groupby(["dataset", "interval_id"]):
        scheduled = plan[(plan.dataset == dataset) & (plan.interval_id == interval) & (plan.protocol == "segment")].iloc[0]
        require(group.start.eq(scheduled.start).all() and group.end.eq(scheduled.end).all(), f"all methods use frozen segment coordinates: {dataset}/{interval}")
        require(group.supplied_k.eq(2).all() and group.n_windows.eq(5).all(), f"fixed K and nonoverlapping-window count: {dataset}/{interval}")
    for row in summary.itertuples(index=False):
        group = detail[(detail.dataset == row.dataset) & (detail.label_column == row.label_column) & (detail.method == row.method)]
        successful = group[group.status == "ok"]
        multi = successful[successful.true_window_k > 1]
        equal([len(group), len(successful), int(group.status.eq("failed").sum()), int(group.status.eq("timeout").sum()), len(multi)],
              [row.planned, row.completed, row.failed, row.timed_out, row.multistate_window_segments], f"segment summary denominators: {row.dataset}/{row.label_column}/{row.method}")
        equal(multi.window_ari.mean(), row.multistate_window_ari_mean, f"segment multistate ARI: {row.dataset}/{row.label_column}/{row.method}")
        equal(successful.sample_ari.mean(), row.all_sample_ari_mean, f"segment full sample ARI: {row.dataset}/{row.label_column}/{row.method}")
    return {"fit_attempts": 24, "label_rows": 36, "rpc_timeouts": 8}


def verify_specialists():
    """Recompute final paired tests from CSV, independently of the job runner."""
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    detail = read("revision_specialist_detail.csv")
    coverage = read("revision_specialist_coverage.csv")
    pairs = read("revision_specialist_paired.csv")
    summary = read("revision_specialist_summary.csv")
    timing = read("revision_specialist_timing.csv")
    protocol_path = TABLES / "revision_specialist_protocol.json"
    INPUTS[protocol_path.relative_to(ROOT).as_posix()] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cells = protocol["specification"]["display_cells"]
    methods = {"STCG-v1", "Regime-PCMCI", "cMLP", "DynVAR-Lasso", "VAR-Lasso"}
    metrics = ["edge_auc", "pair_auc", "precision_at_k", "shd", "lag_accuracy"]
    stress_settings = {cell["setting_id"] for cell in cells}
    timing_setting = next(cell["setting_id"] for cell in cells if cell["axis"] == "obs_noise" and cell["value"] == 0)
    settings = [("stress", setting) for setting in stress_settings] + [("timing", timing_setting)]
    require(len(cells) == 15 and len(stress_settings) == 13, "frozen display and unique setting populations")
    job_keys = ["phase", "setting_id", "method", "seed"]
    expected_jobs = {(phase, setting, method, seed) for phase, setting in settings for method in methods for seed in range(100, 120)}
    require(set(detail[job_keys].itertuples(index=False, name=None)) == expected_jobs, "exact frozen job identities; no substituted setting or seed")
    require(len(detail) == 1400 and not detail.duplicated(job_keys).any(), "all 1400 unique frozen fits attempted")
    require(len(detail[detail.phase == "stress"]) == 1300 and len(detail[detail.phase == "timing"]) == 100, "separate stress and timing populations")
    require(coverage.planned.eq(20).all() and coverage.attempted.eq(20).all(), "complete planned coverage including failures")
    require(set(detail.method) == methods and set(detail.seed) == set(range(100, 120)), "new independent seed and method populations")
    quiet = detail[detail.phase == "timing"]
    require(quiet.execution_workers.eq(1).all() and quiet.exclusive_timing.eq(True).all(), "quiet timing uses one worker")
    successful = detail[detail.status == "ok"]
    coverage_keys = ["phase", "setting_id", "method"]
    expected_coverage = {(phase, setting, method) for phase, setting in settings for method in methods}
    require(not coverage.duplicated(coverage_keys).any() and set(coverage[coverage_keys].itertuples(index=False, name=None)) == expected_coverage,
            "exact coverage key set including every failed method")
    expected_pairs, expected_timings = set(), set()
    common_by_setting = {}
    for phase, setting in settings:
        group = successful[(successful.phase == phase) & (successful.setting_id == setting)]
        common = set.intersection(*(set(group[group.method == method].seed) for method in methods))
        common_by_setting[(phase, setting)] = common
        if len(common) >= 2:
            expected_pairs.update((phase, setting, opponent, metric) for opponent in methods - {"STCG-v1"} for metric in metrics)
            expected_timings.update((phase, setting, opponent, metric) for opponent in methods - {"STCG-v1"} for metric in ["fit_seconds", "fit_cpu_seconds"])
    paired_keys = ["phase", "setting_id", "opponent", "metric"]
    require(not pairs.duplicated(paired_keys).any() and set(pairs[paired_keys].itertuples(index=False, name=None)) == expected_pairs,
            "exact paired-test key set including undefined metrics")
    require(not timing.duplicated(paired_keys).any() and set(timing[paired_keys].itertuples(index=False, name=None)) == expected_timings,
            "exact timing-comparison key set")
    summary_keys = ["axis", "value", "phase", "setting_id", "method"]
    successful_keys = set(successful[coverage_keys].itertuples(index=False, name=None))
    expected_summary = {(cell["axis"], cell["value"], "stress", cell["setting_id"], method)
                        for cell in cells for method in methods if ("stress", cell["setting_id"], method) in successful_keys}
    require(not summary.duplicated(summary_keys).any() and set(summary[summary_keys].itertuples(index=False, name=None)) == expected_summary,
            "exact display summary key set; repeated baseline aliases retained")
    for row in coverage.itertuples(index=False):
        group = detail[(detail.phase == row.phase) & (detail.setting_id == row.setting_id) & (detail.method == row.method)]
        ok = group[group.status == "ok"]
        common = sorted(common_by_setting[(row.phase, row.setting_id)])
        equal([len(group), len(ok), len(group) - len(ok), len(common)], [row.attempted, row.successful, row.failed, row.common_n], "raw-to-coverage success and failure counts")
        require(json.loads(row.common_seeds) == common and sorted(json.loads(row.failed_seeds)) == sorted(group[group.status != "ok"].seed),
                "coverage failure and common-seed identities")
        require(json.loads(row.defined_counts_by_metric) == {metric: int(ok[metric].notna().sum()) for metric in metrics}, "coverage per-metric defined denominators")
    for key, group in detail.groupby(["phase", "setting_id", "method"]):
        require(set(group.seed) == set(range(100, 120)), f"complete method seed coverage {key}")
    for key, group in successful.groupby(["phase", "setting_id", "seed"]):
        require(group.input_sha256.nunique() == 1 and group.truth_sha256.nunique() == 1, f"same input and evaluation truth {key}")
    for value in successful.threadpools:
        require(all(pool["num_threads"] == 1 for pool in json.loads(value)), "one computational thread in recorded pools")
    for row in pairs.itertuples(index=False):
        group = successful[(successful.phase == row.phase) & (successful.setting_id == row.setting_id)]
        matrix = group.pivot(index="seed", columns="method", values=row.metric).reindex(columns=sorted(methods)).dropna()
        require(list(matrix.index) == json.loads(row.common_seeds), f"per-metric common finite seeds {row.phase}/{row.setting_id}/{row.metric}/{row.opponent}")
        require(len(matrix) == row.n, "per-metric paired sample size")
        successful_common = common_by_setting[(row.phase, row.setting_id)]
        require(row.successful_fit_common_n == len(successful_common) and
                sorted(json.loads(row.undefined_metric_seeds)) == sorted(successful_common - set(matrix.index)),
                "undefined metric seeds remain separate from fit failures")
        if len(matrix) < 2:
            require(row.statistics_status != "ok", "undefined metric is not promoted to a test")
            continue
        a, b = matrix["STCG-v1"].to_numpy(), matrix[row.opponent].to_numpy()
        delta = b - a if row.metric == "shd" else a - b
        equal([a.mean(), b.mean(), delta.mean(), delta.std(ddof=1)],
              [row.mean_stcg, row.mean_opponent, row.mean_improvement, row.sd_improvement], "paired effect moments")
        index = np.random.default_rng(20260913).integers(0, len(delta), size=(10000, len(delta)))
        equal(np.quantile(delta[index].mean(axis=1), [.025, .975]), [row.ci_low, row.ci_high], "paired bootstrap confidence interval")
        p_t = 1.0 if np.all(delta == 0) else stats.ttest_rel(a, b).pvalue
        p_w = 1.0 if np.all(delta == 0) else stats.wilcoxon(delta).pvalue
        equal([p_t, p_w], [row.paired_t_two_sided_p, row.wilcoxon_two_sided_p], "two-sided paired significance tests")
        equal([np.sum(delta > 1e-12), np.sum(np.abs(delta) <= 1e-12), np.sum(delta < -1e-12)],
              [row.wins, row.ties, row.losses], "paired win/tie/loss counts")
    for column in ["paired_t_two_sided_p", "wilcoxon_two_sided_p"]:
        eligible = pairs[(pairs.phase == "stress") & (pairs.metric == "edge_auc") & (pairs.statistics_status == "ok")]
        for size, family in [(13, eligible[eligible.opponent == "Regime-PCMCI"]), (52, eligible)]:
            padded = [*family[column], *([1.] * (size - len(family)))]
            corrected = multipletests(padded, method="holm")[1][:len(family)]
            equal(corrected, family[column + f"_holm{size}"], f"Holm family {size}/{column}")
    for row in summary.itertuples(index=False):
        group = successful[(successful.phase == row.phase) & (successful.setting_id == row.setting_id) & (successful.method == row.method)]
        for metric in metrics:
            defined = group[metric].dropna()
            equal([len(defined), defined.mean()], [getattr(row, metric + "_n"), getattr(row, metric + "_mean")], "all-successful per-method means and defined counts")
    for row in timing.itertuples(index=False):
        group = successful[(successful.phase == row.phase) & (successful.setting_id == row.setting_id)]
        matrix = group.pivot(index="seed", columns="method", values=row.metric).reindex(columns=sorted(methods)).dropna()
        require(len(matrix) == row.n, "timing paired denominator")
        a, b = matrix["STCG-v1"].to_numpy(), matrix[row.opponent].to_numpy()
        if (a == 0).any() or (b == 0).any():
            require(not row.ratio_available and pd.isna(row.geometric_mean_opponent_over_stcg), "zero CPU clock observation retained without ratio imputation")
        else:
            log_ratio = np.log(b / a)
            index = np.random.default_rng(20260913).integers(0, len(log_ratio), size=(10000, len(log_ratio)))
            interval = np.exp(np.quantile(log_ratio[index].mean(axis=1), [.025, .975]))
            equal([np.exp(log_ratio.mean()), *interval], [row.geometric_mean_opponent_over_stcg, row.ratio_ci_low, row.ratio_ci_high], "paired geometric cost ratio and confidence interval")
    return {"fits": len(detail), "successful": len(successful), "failed_or_timeout": len(detail) - len(successful), "paired_rows": len(pairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-specialist", action="store_true", help="Require complete stress and quiet timing evidence")
    args = parser.parse_args()
    evidence = {"gate": verify_gate(), "forward": verify_forward(), "segments": verify_segments()}
    if args.include_specialist:
        evidence["specialists"] = verify_specialists()
    result = {"status": "passed", "checks": len(CHECKS), "evidence": evidence, "input_sha256": INPUTS}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
