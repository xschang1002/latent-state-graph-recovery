"""Run the frozen independent-seed gate/selector stress protocol.

See docs/revision_gate_protocol_2026-09-13.md. No production defaults are changed.
Local score caching reuses only identical Lasso outputs; production inference and
aggregation are executed for every candidate and gate intervention.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from _common import write_csv
from run_hydraulic_transition_warning import stcg_transition_selection_score
from run_real_state_count_selection import mark_selected, state_count_selection_score
from stcg import STCGStateConfig, generate_lagged_var, recovery_summary, stcg_v1_diagnostics
from stcg.self_supervised import _min_cluster_fraction
from stcg.synthetic import SyntheticDataset, _sample_coefficients

PROTOCOL = ROOT / "docs/revision_gate_protocol_2026-09-13.md"
SEEDS = tuple(range(100, 120))
EDGE_METRICS = ("edge_auc", "pair_auc", "precision_at_k", "shd", "lag_accuracy")
GATE_VARIANTS = {
    "default": {}, "always_accept": {"auto_states": False},
    "always_reject": {"min_state_improvement": math.inf, "min_temporal_state_improvement": math.inf},
    "no_fallback": {"min_temporal_state_improvement": math.inf},
    "no_primary": {"min_state_improvement": math.inf},
    "no_coherence": {"min_temporal_coherence": 0.0},
    "no_min_share": {"min_temporal_state_fraction": 0.0},
}
SELECTOR_VARIANTS = {
    "stable": ("full", "no_balance", "no_parsimony", "no_coherence_weight", "no_coherence_screen", "no_coherence_all", "positive_fallback"),
    "warning": ("full", "no_balance", "no_parsimony", "no_coherence_weight", "positive_fallback"),
}


@dataclass(frozen=True)
class Condition:
    name: str
    timesteps: int = 9000
    window: int = 300
    stride: int = 150
    switching: bool = True
    period: int | None = None
    mixture: float = 1.0
    switch_fraction: float | None = None
    observation_noise: float = 0.0


CONDITIONS = (
    Condition("stationary_many", switching=False),
    Condition("stationary_few", timesteps=1000, switching=False),
    Condition("strong_switch_long"),
    Condition("weak_switch_long", mixture=0.25),
    Condition("mixed_periodic", period=150),
    Condition("short_switch", timesteps=600, window=150, stride=75),
    Condition("measurement_noise_switch", observation_noise=0.20),
    Condition("imbalanced_switch", switch_fraction=0.95),
)
FIXTURES = ("coherent_weak", "shuffled_weak", "tiny_coherent", "primary_incoherent", "hierarchical_balanced", "hierarchical_rare")


def modified_switch(seed: int, timesteps: int, mixture: float = 1.0, switch_fraction: float | None = None) -> SyntheticDataset:
    """Production recurrence/RNG, extending only coefficient mixing/switch time."""
    rng = np.random.default_rng(seed)
    max_lag, n_nodes, burn_in = 3, 8, 100
    coeff_a = _sample_coefficients(rng, n_nodes, max_lag, 0.18)
    sampled_b = _sample_coefficients(rng, n_nodes, max_lag, 0.18)
    coeff_b = (1.0 - mixture) * coeff_a + mixture * sampled_b
    coefficients = np.stack([coeff_a, coeff_b])
    assert np.max(np.abs(coefficients).sum(axis=(1, 2))) <= 0.85 + 1e-12
    total_steps = timesteps + burn_in
    x = rng.normal(0.0, 0.10, size=(total_steps, n_nodes))
    regimes = np.zeros(total_steps, dtype=int)
    switch_point = total_steps // 2 if switch_fraction is None else burn_in + int(timesteps * switch_fraction)
    for t in range(max_lag, total_steps):
        regime = int(t >= switch_point)
        regimes[t] = regime
        value = np.zeros(n_nodes)
        for lag in range(1, max_lag + 1):
            value += x[t - lag] @ coefficients[regime, lag - 1]
        x[t] = value + rng.normal(0.0, 0.10, size=n_nodes)
    truth = np.any(np.abs(coefficients) > 0, axis=0).astype(int)
    return SyntheticDataset(x[burn_in:], coefficients, truth, regimes[burn_in:])


def generate_condition(condition: Condition, seed: int) -> SyntheticDataset:
    if condition.mixture != 1.0 or condition.switch_fraction is not None:
        dataset = modified_switch(seed, condition.timesteps, condition.mixture, condition.switch_fraction)
    else:
        dataset = generate_lagged_var(seed=seed, timesteps=condition.timesteps,
                                      switching=condition.switching, switch_period=condition.period)
    if condition.observation_noise:
        noise = np.random.default_rng(600000 + seed).normal(0, condition.observation_noise, dataset.x.shape)
        dataset = SyntheticDataset(dataset.x + noise, dataset.coefficients, dataset.edge_truth, dataset.regimes)
    return dataset


def cached_diagnostics(x: np.ndarray, window_scores: np.ndarray, config: STCGStateConfig):
    """Only bypass repeated regression; execute the complete production pipeline."""
    with patch("stcg.self_supervised._window_reconstruction_scores", side_effect=[s.copy() for s in window_scores]) as scorer:
        result = stcg_v1_diagnostics(x, 3, config)
    assert scorer.call_count == len(window_scores)
    return result


def fixture_scores(name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(700000 + seed)
    labels = np.repeat([0, 1], 60)
    if name in {"shuffled_weak", "primary_incoherent"}:
        rng.shuffle(labels)
    elif name == "tiny_coherent":
        labels = np.r_[np.zeros(114, dtype=int), np.ones(6, dtype=int)]
    elif name.startswith("hierarchical"):
        sizes = [30, 30, 30, 30] if name == "hierarchical_balanced" else [60, 30, 24, 6]
        labels = np.repeat(np.arange(4), sizes)
    features = 4.0 + rng.normal(0, 0.2, (120, 168))
    if name.startswith("hierarchical"):
        features[:, :30] += 10.0 * (labels >= 2)[:, None]
        features[:, 30:40] += 10.0 * (labels % 2)[:, None]
    else:
        n_signal = 36 if name == "primary_incoherent" else 12
        features[:, :n_signal] += 10.0 * labels[:, None]
    scores = np.zeros((120, 3, 8, 8))
    mask = np.ones((3, 8, 8), dtype=bool)
    mask[:, np.arange(8), np.arange(8)] = False
    scores[:, mask] = features
    return scores, labels


def selector_score(candidate: dict[str, object], objective: str, variant: str) -> float:
    """Exact production full score, with explicitly isolated factor removals."""
    improvement, coherence, fraction, count = (float(candidate[k]) for k in ("state_improvement", "temporal_coherence", "min_cluster_fraction", "state_k"))
    k = int(count)
    if objective == "stable":
        func = state_count_selection_score
        score, balance = func(improvement, coherence, fraction, k)
    elif objective == "warning":
        score, balance = stcg_transition_selection_score(improvement, coherence, fraction, k)
    else:
        raise ValueError(objective)
    if variant in {"full", "positive_fallback"}:
        return score
    if variant == "no_balance":
        return score / balance if balance > 0 else 0.0
    if variant == "no_parsimony":
        return score * math.log2(k + 1)
    if variant == "no_coherence_weight":
        weight = coherence if objective == "stable" else 1.0 - 0.5 * min(1.0, max(0.0, coherence))
        return score / weight if weight > 0 else 0.0
    if variant == "no_coherence_screen" and objective == "stable":
        return func(improvement, coherence, fraction, k, min_temporal_coherence=0.0)[0]
    if variant == "no_coherence_all" and objective == "stable":
        return func(improvement, 1.0, fraction, k)[0]
    raise ValueError((objective, variant))


def select_candidate(candidates: list[dict[str, object]], objective: str, variant: str) -> tuple[dict[str, object], bool]:
    # Calls the production selector, including the all-zero-score tie behavior.
    rows = [{**candidate, "selection_score": selector_score(candidate, objective, variant)} for candidate in candidates]
    mark_selected(rows, policy="reject_nonpositive" if variant == "positive_fallback" else "legacy")
    return next(row for row in rows if row["selected"]), all(row["selection_score"] == 0 for row in rows)


def evaluate_decision(diagnostic, labels: np.ndarray, truth_count: int, edge_truth: np.ndarray | None) -> dict[str, object]:
    count = len(np.unique(diagnostic.labels))
    row = {"state_k": count, "state_ari": float(adjusted_rand_score(labels, diagnostic.labels)),
           "false_split": int(count > 1) if truth_count == 1 else math.nan,
           "count_correct": int(count == truth_count), "under_split": int(count < truth_count),
           "over_split": int(count > truth_count)}
    row.update(recovery_summary(diagnostic.scores, edge_truth) if edge_truth is not None else {metric: math.nan for metric in EDGE_METRICS})
    return row


def run_one(name: str, seed: int, condition: Condition | None) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    config = STCGStateConfig(window_size=condition.window if condition else 5,
                             stride=condition.stride if condition else 1, seed=30000 + seed)
    if condition:
        dataset = generate_condition(condition, seed)
        x, edge_truth = dataset.x, dataset.edge_truth
        default = stcg_v1_diagnostics(x, 3, config)
        scores = default.window_scores
        labels, purities = [], []
        for start, end in default.bounds:
            counts = np.bincount(dataset.regimes[start:end])
            labels.append(int(np.argmax(counts)))
            purities.append(float(counts.max() / (end - start)))
        labels = np.array(labels)
        truth_count = len(np.unique(dataset.regimes))
    else:
        scores, labels = fixture_scores(name, seed)
        x, edge_truth = np.zeros((len(scores) + 4, 8)), None
        default = cached_diagnostics(x, scores, config)
        truth_count, purities = len(np.unique(labels)), [1.0]
    shared = {"evidence_kind": "generator" if condition else "controlled_score_tensor", "condition": name,
              "seed": seed, "truth_state_count": truth_count, "window_truth_state_count": len(np.unique(labels)),
              "n_windows": len(scores), "window_purity": float(np.mean(purities))}
    # Decisions never take labels or edge truth. The audit-only evaluation occurs below.
    diagnostics = {k: cached_diagnostics(x, scores, replace(config, n_states=k, auto_states=False)) for k in (1, 2, 3, 4)}
    raw = diagnostics[2]
    gate_decisions = {variant: default if variant == "default" else cached_diagnostics(x, scores, replace(config, **changes)) for variant, changes in GATE_VARIANTS.items()}
    candidates = [{**shared, "candidate_state_count": k, "state_k": len(np.unique(d.labels)),
                   "state_improvement": d.state_improvement, "temporal_coherence": d.temporal_coherence,
                   "min_cluster_fraction": _min_cluster_fraction(d.labels)} for k, d in diagnostics.items()]
    selections = {(objective, variant): select_candidate(candidates, objective, variant)
                  for objective, variants in SELECTOR_VARIANTS.items() for variant in variants}
    rows = []
    for variant, diagnostic in gate_decisions.items():
        cfg = replace(config, **GATE_VARIANTS[variant])
        if not cfg.auto_states:
            path = "forced_accept"
        elif not diagnostic.states_accepted:
            path = "reject"
        elif raw.state_improvement >= cfg.min_state_improvement:
            path = "primary"
        else:
            path = "fallback"
        row = {**shared, "analysis": "gate", "objective": "partition", "variant": variant,
               "experiment_stage": "frozen_legacy",
               "gate_path": path, "states_accepted": int(diagnostic.states_accepted),
               "primary_accept": int(path == "primary"), "fallback_accept": int(path == "fallback"),
               "reject": int(path == "reject"), "forced_accept": int(path == "forced_accept"),
               "state_improvement": raw.state_improvement, "temporal_coherence": raw.temporal_coherence,
               "min_cluster_fraction": _min_cluster_fraction(raw.labels)}
        row.update(evaluate_decision(diagnostic, labels, truth_count, edge_truth))
        rows.append(row)
    for (objective, variant), (candidate, all_zero) in selections.items():
        k = int(candidate["candidate_state_count"])
        row = {**shared, "analysis": "selector", "objective": objective, "variant": variant,
               "experiment_stage": "postfinding_logic_fix" if variant == "positive_fallback" else "frozen_legacy",
               "candidate_state_count": k, "selection_score": candidate["selection_score"],
               "all_scores_zero": int(all_zero), "state_improvement": candidate["state_improvement"],
               "temporal_coherence": candidate["temporal_coherence"], "min_cluster_fraction": candidate["min_cluster_fraction"]}
        row.update(evaluate_decision(diagnostics[k], labels, truth_count, edge_truth))
        rows.append(row)
    # Candidate metrics are reporting only and cannot affect the completed selections.
    for candidate in candidates:
        k = int(candidate["candidate_state_count"])
        candidate.update(evaluate_decision(diagnostics[k], labels, truth_count, edge_truth))
        candidate["stable_score"] = selector_score(candidate, "stable", "full")
        candidate["warning_score"] = selector_score(candidate, "warning", "full")
    return rows, candidates


def save_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(path, rows, fields)


def audit_saved_real_selections(output_dir: Path) -> list[dict[str, object]]:
    """Recompute scoring/selection for the two tracked default-protocol panels.

    The correction does not refit candidates. If selection changes, this audit
    exposes it rather than silently rewriting the historical real-data result.
    """
    out = []
    for filename in ("AReMActivities_selectK_state_count_selection_detail.csv",
                     "HydraulicSystems_cooler_selectK_state_count_selection_detail.csv"):
        path = ROOT / "results/tables" / filename
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            score, _ = state_count_selection_score(float(row["state_improvement"]), float(row["temporal_coherence"]),
                                                   float(row["min_cluster_fraction"]), int(row["inferred_state_count"]))
            if abs(score - float(row["selection_score"])) >= 1e-12:
                raise ValueError(f"Saved selector score disagrees with production: {filename}, seed {row['seed']}")
        legacy, safe = [dict(row) for row in rows], [dict(row) for row in rows]
        mark_selected(legacy, policy="legacy")
        mark_selected(safe, policy="reject_nonpositive")
        for seed in sorted({row["seed"] for row in rows}):
            original = next(row for row in rows if row["seed"] == seed and row["selected"] == "True")
            before = next(row for row in legacy if row["seed"] == seed and row["selected"])
            after = next(row for row in safe if row["seed"] == seed and row["selected"])
            if original["candidate_state_count"] != before["candidate_state_count"]:
                raise ValueError(f"Saved legacy choice disagrees: {filename}, seed {seed}")
            out.append({"source_file": filename, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "seed": seed, "legacy_k": before["candidate_state_count"], "corrected_k": after["candidate_state_count"],
                        "positive_score": after["selection_score"],
                        "choice_changed": int(before["candidate_state_count"] != after["candidate_state_count"]),
                        "score_recomputed": True})
    save_rows(output_dir / "revision_gate_real_selector_policy_check.csv", out)
    return out


def wilson(values: list[float]) -> tuple[float, float]:
    n, z = len(values), 1.959963984540054
    p = float(np.mean(values))
    denominator = 1 + z * z / n
    middle = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1-p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, middle-half), min(1.0, middle+half)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = ("evidence_kind", "condition", "analysis", "objective", "variant")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    metrics = (*EDGE_METRICS, "state_ari", "state_k", "false_split", "count_correct", "under_split", "over_split",
               "states_accepted", "primary_accept", "fallback_accept", "reject", "forced_accept", "all_scores_zero",
               "state_improvement", "temporal_coherence", "min_cluster_fraction", "selection_score", "window_purity")
    for key, items in groups.items():
        summary = {**dict(zip(keys, key)), "n": len(items)}
        for metric in metrics:
            values = [float(row[metric]) for row in items if metric in row and np.isfinite(float(row[metric]))]
            summary[f"{metric}_mean"] = float(np.mean(values)) if values else math.nan
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            if metric in {"false_split", "states_accepted", "all_scores_zero", "count_correct"}:
                lo, hi = wilson(values) if values else (math.nan, math.nan)
                summary[f"{metric}_ci_low"], summary[f"{metric}_ci_high"] = lo, hi
        result.append(summary)
    return result


def paired_differences(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = ("evidence_kind", "condition", "analysis", "objective")
    grouped = defaultdict(dict)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].setdefault(str(row["variant"]), {})[int(row["seed"])] = row
    result = []
    for key, variants in grouped.items():
        reference = "default" if key[2] == "gate" else "full"
        baseline = variants[reference]
        for variant, seed_rows in variants.items():
            if variant == reference:
                continue
            common = sorted(set(seed_rows) & set(baseline))
            for metric in ("edge_auc", "state_ari", "state_k", "false_split", "states_accepted", "all_scores_zero"):
                differences = [float(seed_rows[s][metric]) - float(baseline[s][metric]) for s in common
                               if metric in seed_rows[s] and np.isfinite(float(seed_rows[s][metric])) and np.isfinite(float(baseline[s][metric]))]
                if not differences:
                    continue
                values = np.array(differences)
                rng = np.random.default_rng(90913)
                means = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
                lo, hi = np.quantile(means, [0.025, 0.975])
                result.append({**dict(zip(keys, key)), "variant": variant, "reference": reference,
                               "metric": metric, "n": len(values), "mean_difference": float(values.mean()),
                               "ci_low": float(lo), "ci_high": float(hi), "changed_seed_count": int(np.count_nonzero(np.abs(values) > 1e-12))})
    return result


def write_report(path: Path, summaries: list[dict[str, object]], paired: list[dict[str, object]], elapsed: float) -> None:
    lines = ["# Independent gate and selector stress results", "", "Frozen protocol: `docs/revision_gate_protocol_2026-09-13.md`.",
             "Seeds 100–119; all stated thresholds and candidate sets are fixed. Controlled fixtures are shown separately from generator evidence.",
             "Core STCG gate thresholds are unchanged. Paired differences use 10000 paired bootstrap draws; intervals are descriptive, not adjusted hypothesis tests.", ""]
    lines += ["The original full and factor-removal selectors explicitly use legacy selection. Additional `positive_fallback` rows apply the post-finding all-nonpositive-score logic fix and are not part of the prospectively frozen factor experiment. Core STCG partition-gate defaults remain unchanged.", ""]
    for kind in ("generator", "controlled_score_tensor"):
        lines += [f"## {kind}: default partition gate", "", "| Condition | N | Accept | Primary | Fallback | False split | Edge AUC | Window ARI |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in summaries:
            if row["evidence_kind"] == kind and row["analysis"] == "gate" and row["variant"] == "default":
                def show(key):
                    value = float(row[key + "_mean"])
                    return f"{value:.3f}" if np.isfinite(value) else "—"
                lines.append(f"| {row['condition']} | {row['n']} | {show('states_accepted')} | {show('primary_accept')} | {show('fallback_accept')} | {show('false_split')} | {show('edge_auc')} | {show('state_ari')} |")
        lines += ["", f"## {kind}: full selectors", "", "| Condition | Objective | Mean K | All scores zero | False split | Edge AUC | Window ARI |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for row in summaries:
            if row["evidence_kind"] == kind and row["analysis"] == "selector" and row["variant"] == "full":
                def show(key):
                    value = float(row[key + "_mean"])
                    return f"{value:.3f}" if np.isfinite(value) else "—"
                lines.append(f"| {row['condition']} | {row['objective']} | {show('state_k')} | {show('all_scores_zero')} | {show('false_split')} | {show('edge_auc')} | {show('state_ari')} |")
        lines += ["", f"## {kind}: interventions that change selected/accepted counts", "", "| Condition | Analysis/objective | Removal | Metric | Changed seeds | Mean difference | 95% interval |",
                  "|---|---|---|---|---:|---:|---|"]
        for row in paired:
            target_metric = "states_accepted" if row["analysis"] == "gate" else "state_k"
            if row["evidence_kind"] == kind and row["metric"] == target_metric and row["changed_seed_count"]:
                lines.append(f"| {row['condition']} | {row['analysis']}/{row['objective']} | {row['variant']} | {row['metric']} | {row['changed_seed_count']}/{row['n']} | {row['mean_difference']:.3f} | [{row['ci_low']:.3f}, {row['ci_high']:.3f}] |")
        lines.append("")
        lines += [f"## {kind}: corrected positive-score fallback", "", "| Condition | Objective | Mean K | All scores zero | False split | Edge AUC | Window ARI |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for row in summaries:
            if row["evidence_kind"] == kind and row["analysis"] == "selector" and row["variant"] == "positive_fallback":
                def show(key):
                    value = float(row[key + "_mean"])
                    return f"{value:.3f}" if np.isfinite(value) else "—"
                lines.append(f"| {row['condition']} | {row['objective']} | {show('state_k')} | {show('all_scores_zero')} | {show('false_split')} | {show('edge_auc')} | {show('state_ari')} |")
        lines.append("")
    lines += ["Frozen legacy rows preserve the original `(score, coherence, -K)` tie-break when all scores are zero; those historical rows do not imply a single-state rejection. The post-finding `positive_fallback` rows use the current production default: return the fitted K=1 candidate when no score is positive.",
              "In mixed_periodic, every window contains exactly equal portions of the two regimes (purity 0.500). The deterministic majority-label tie-break produces one window label; ARI=1 for a one-state prediction is therefore degenerate and is not evidence of recovering the two true regimes. The detail file records both the true time-series count and the window-label count.",
              "False splitting is defined only on stationary generators; fixture graph-recovery metrics are absent because these fixtures have no empirical causal ground truth.",
              "Complete per-seed decisions, candidates, summary/Wilson intervals and paired changes are in the accompanying CSV files.", f"Elapsed wall time (single BLAS/OpenMP thread): {elapsed:.2f} seconds.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/tables")
    parser.add_argument("--audit-real-only", action="store_true", help="Only recheck the two stored real-data selector panels")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.audit_real_only:
        audit = audit_saved_real_selections(args.output_dir)
        print(f"Rechecked {len(audit)} real-data seeds; changed selections: {sum(row['choice_changed'] for row in audit)}")
        return 0
    started, rows, candidates, failures = time.perf_counter(), [], [], []
    with threadpool_limits(limits=1):
        for condition in [*CONDITIONS, *FIXTURES]:
            name = condition.name if isinstance(condition, Condition) else condition
            for seed in SEEDS:
                try:
                    decisions, candidate_rows = run_one(name, seed, condition if isinstance(condition, Condition) else None)
                except Exception as exc:
                    failures.append({"condition": name, "seed": seed, "error": f"{type(exc).__name__}: {exc}"})
                    print(f"FAILED {name} seed {seed}: {exc}", flush=True)
                    continue
                rows.extend(decisions)
                candidates.extend(candidate_rows)
            save_rows(args.output_dir / "revision_gate_stress_detail.csv", rows)
            save_rows(args.output_dir / "revision_gate_candidates.csv", candidates)
            print(f"Finished {name}: {len(SEEDS)} seeds; {time.perf_counter()-started:.1f}s elapsed", flush=True)
    summaries, paired = summarize(rows), paired_differences(rows)
    elapsed = time.perf_counter()-started
    save_rows(args.output_dir / "revision_gate_stress_summary.csv", summaries)
    save_rows(args.output_dir / "revision_gate_paired_changes.csv", paired)
    write_report(args.output_dir / "revision_gate_stress_summary.md", summaries, paired, elapsed)
    write_csv(args.output_dir / "revision_gate_failures.csv", failures, ["condition", "seed", "error"])
    real_audit = audit_saved_real_selections(args.output_dir)
    save_rows(args.output_dir / "revision_gate_run_metadata.csv", [{
        "protocol_id": "revision-gate-2026-09-13-v1-amendment-A", "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "initial_protocol_sha256": "62b4e67710e6f027b0ba2c16cc3a866375a5a9bbef08c1aabbdbef8b328929e3",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_state_sha256": hashlib.sha256((ROOT / "src/stcg/self_supervised.py").read_bytes()).hexdigest(),
        "production_generator_sha256": hashlib.sha256((ROOT / "src/stcg/synthetic.py").read_bytes()).hexdigest(),
        "production_stable_selector_sha256": hashlib.sha256((ROOT / "scripts/run_real_state_count_selection.py").read_bytes()).hexdigest(),
        "selection_helper_sha256": hashlib.sha256((ROOT / "scripts/_state_selection.py").read_bytes()).hexdigest(),
        "production_warning_selector_sha256": hashlib.sha256((ROOT / "scripts/run_hydraulic_transition_warning.py").read_bytes()).hexdigest(),
        "seeds": "100-119", "blas_openmp_threads": 1, "elapsed_seconds": elapsed,
        "generator_conditions": len(CONDITIONS), "controlled_fixtures": len(FIXTURES),
        "detail_rows": len(rows), "candidate_rows": len(candidates), "failed_cases": len(failures),
        "real_selector_seeds_checked": len(real_audit),
        "real_selector_choices_changed": sum(row["choice_changed"] for row in real_audit)}])
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
