"""Head-to-head real-data regime recovery: STCG-v1 states vs Regime-PCMCI vs MSVAR.

This is the real-data counterpart of ``run_synthetic_rpcmci_baseline.py`` /
``run_synthetic_msvar_baseline.py``: it asks whether the *regime sequences*
recovered by each method match ground-truth operating-mode labels on the real
datasets that carry them (HydraulicSystems component conditions, AReM activity
labels), using the same alignment protocol as the existing state-label
alignment harness (``compare_real_state_labels.py`` / ``_common.align_labels``:
Hungarian best-match accuracy + sklearn adjusted Rand index).

Compute-constrained protocol (documented, identical input for every method):

- RPCMCI costs ``max_anneal`` x ``num_iterations`` x ``num_regimes``
  PCMCI+MIP fits and does not fit a full-length real series into a single
  process invocation, so each dataset is evaluated on a CAPPED window:
  - ``segment`` mode: a contiguous ``cap_t``-sample segment centered on the
    FIRST ground-truth label transition (``start = first_transition - cap_t //
    2``). Segment placement is label-driven (oracle-sighted), chosen once from
    the label map before any method ran and frozen for all methods.
  - ``blocks`` mode: the first ``block_n`` samples of each requested label
    block, concatenated in the given order (block boundaries are real regime
    transitions between recordings). Used for AReM so that more than two
    activity regimes fit inside the compute cap.
- STCG-v1 states are RECOMPUTED on the same capped window with the forward
  harness machinery (``run_real_state_label_prediction_forward`` window
  features + train-fitted KMeans assignment, fitted on all windows of the
  capped segment), never reused from the full-series runs.
- RPCMCI and MSVAR receive the TRUE regime count ``K`` (oracle-sighted, i.e.
  generous to the baselines: they never have to estimate the regime count).
  Per-config ``K`` and the number of regimes actually present in the capped
  window are both recorded in every row.
- RPCMCI parameters match the synthetic runner (``pc_alpha=0.05``,
  ``alpha_level=0.01``, ``tau_max=3``, ``num_iterations=20``,
  ``max_transitions=10``, ``switch_thres=0.05``, ``max_anneal=10``). The
  ``--max-anneal 5`` fallback exists in case compute forbids; the tracked
  tables record the budget actually used in every row.

Metrics are reported at two granularities:

- window level (primary; the same protocol as the tracked real-data alignment
  tables): window-majority true labels vs. per-window inferred states.
  RPCMCI/MSVAR per-sample assignments are windowed by majority vote.
- sample level (secondary; matches the synthetic ``regime_quality`` path):
  per-sample true labels vs. assignments from each method's effective offset
  onward. STCG window states are tiled to samples by nearest window center.

Every ``(config, method, seed)`` row is checkpointed into
``experiments/logs/real_regime_parts/`` (gitignored); re-running skips existing
partials, and the tracked result tables are (re)built once every requested
partial exists (or immediately with ``--aggregate-only``). Failed fits stay in
the partials as flagged NaN rows and are excluded from the tracked aggregates.

Note on parallelism: the imported real-data harness modules set
``LOKY_MAX_CPU_COUNT=1`` at import time to keep sklearn KMeans from
oversubscribing cores. When a run does not include the STCG method, this
script pops that guard before fitting so RPCMCI/MSVAR keep full joblib
parallelism (the guard is read by loky at executor creation, not at import).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

import _common  # noqa: E402
import run_real_state_label_prediction as base  # noqa: E402
import run_real_state_label_prediction_forward as forward  # noqa: E402
from stcg import STCGStateConfig, rpcmci_scores, sliding_window_bounds  # noqa: E402


METHOD_STCG = "STCG-v1"
METHOD_RPCMCI = "Regime-PCMCI"
METHOD_MSVAR = "MSVAR"
METHODS = [METHOD_STCG, METHOD_RPCMCI, METHOD_MSVAR]

# Capped evaluation configs. ``oracle_k`` is the ground-truth regime count
# handed to RPCMCI/MSVAR and used as the STCG state count (matching the
# forward harness choice of 3 for HydraulicSystems cooler condition).
CONFIGS: dict[str, dict[str, object]] = {
    "HydraulicSystems_cooler_transition": {
        "input_csv": ROOT / "data" / "raw" / "HydraulicSystems.csv",
        "labels_csv": ROOT / "data" / "processed" / "HydraulicSystems_labels.csv",
        "label_column": "cooler_condition",
        "columns": "TS1,TS2,TS3,TS4,VS1,CE,CP,SE",
        "cap_mode": "segment",
        "cap_t": 1500,
        "oracle_k": 3,
    },
    "HydraulicSystems_valve_transition": {
        "input_csv": ROOT / "data" / "raw" / "HydraulicSystems.csv",
        "labels_csv": ROOT / "data" / "processed" / "HydraulicSystems_labels.csv",
        "label_column": "valve_condition",
        "columns": "TS1,TS2,TS3,TS4,VS1,CE,CP,SE",
        "cap_mode": "segment",
        "cap_t": 1500,
        "oracle_k": 4,
    },
    # Generosity controls: same capped segments, but the baselines receive the
    # number of regimes actually PRESENT in the segment (easier than the
    # dataset-level true K used in the primary configs).
    "HydraulicSystems_cooler_transition_K2present": {
        "input_csv": ROOT / "data" / "raw" / "HydraulicSystems.csv",
        "labels_csv": ROOT / "data" / "processed" / "HydraulicSystems_labels.csv",
        "label_column": "cooler_condition",
        "columns": "TS1,TS2,TS3,TS4,VS1,CE,CP,SE",
        "cap_mode": "segment",
        "cap_t": 1500,
        "oracle_k": 2,
    },
    "HydraulicSystems_valve_transition_K3present": {
        "input_csv": ROOT / "data" / "raw" / "HydraulicSystems.csv",
        "labels_csv": ROOT / "data" / "processed" / "HydraulicSystems_labels.csv",
        "label_column": "valve_condition",
        "columns": "TS1,TS2,TS3,TS4,VS1,CE,CP,SE",
        "cap_mode": "segment",
        "cap_t": 1500,
        "oracle_k": 3,
    },
    "AReMActivities_bending_transition": {
        "input_csv": ROOT / "data" / "raw" / "AReMActivities.csv",
        "labels_csv": ROOT / "data" / "processed" / "AReMActivities_labels.csv",
        "label_column": "activity",
        "columns": "avg_rss12,var_rss12,avg_rss13,var_rss13,avg_rss23,var_rss23",
        "cap_mode": "segment",
        "cap_t": 1500,
        "oracle_k": 2,
    },
    "AReMActivities_concat4": {
        "input_csv": ROOT / "data" / "raw" / "AReMActivities.csv",
        "labels_csv": ROOT / "data" / "processed" / "AReMActivities_labels.csv",
        "label_column": "activity",
        "columns": "avg_rss12,var_rss12,avg_rss13,var_rss13,avg_rss23,var_rss23",
        "cap_mode": "blocks",
        "block_labels": ["bending1", "cycling", "lying", "walking"],
        "block_n": 375,
        "oracle_k": 4,
    },
}

DETAIL_FIELDNAMES = [
    "config",
    "method",
    "seed",
    "status",
    "oracle_k",
    "k_present",
    "cap_mode",
    "cap_description",
    "n_timesteps",
    "n_channels",
    "n_windows",
    "true_window_counts",
    "pred_window_counts",
    "window_ari",
    "window_aligned_accuracy",
    "sample_ari",
    "sample_aligned_accuracy",
    "n_samples_scored",
    "state_mapping",
    "error_free_annealings",
    "converged_fraction",
    "num_iterations",
    "max_anneal",
    "max_transitions",
    "fit_seconds",
]
SUMMARY_FIELDNAMES = [
    "config",
    "method",
    "n",
    "oracle_k",
    "k_present",
    "n_timesteps",
    "n_windows",
    "window_ari_mean",
    "window_ari_std",
    "window_aligned_accuracy_mean",
    "window_aligned_accuracy_std",
    "sample_ari_mean",
    "sample_ari_std",
    "sample_aligned_accuracy_mean",
    "sample_aligned_accuracy_std",
    "fit_seconds_mean",
    "fit_seconds_std",
    "cap_description",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", choices=sorted(CONFIGS), default=sorted(CONFIGS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--stcg-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--lasso-alpha", type=float, default=0.005)
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--window-stride", type=int, default=150)
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--max-anneal", type=int, default=10)
    parser.add_argument("--max-transitions", type=int, default=10)
    parser.add_argument("--switch-thres", type=float, default=0.05)
    parser.add_argument("--pc-alpha", type=float, default=0.05)
    parser.add_argument("--alpha-level", type=float, default=0.01)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--msvar-maxiter", type=int, default=200)
    parser.add_argument("--msvar-retry-maxiter", type=int, default=500)
    parser.add_argument("--msvar-search-reps", type=int, default=3)
    parser.add_argument("--missing", choices=["drop", "interpolate"], default="interpolate")
    parser.add_argument("--parts-dir", type=Path, default=ROOT / "experiments" / "logs" / "real_regime_parts")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--aggregate-partial",
        action="store_true",
        help="Aggregate whatever partials exist in the plan; missing combos are listed "
        "in the output JSON and a markdown footnote instead of blocking aggregation.",
    )
    parser.add_argument(
        "--mark-budget-failure",
        nargs=3,
        metavar=("CONFIG", "METHOD", "SEED"),
        action="append",
        default=[],
        help="Write a failed partial recording that this fit was genuinely attempted but "
        "repeatedly killed by the per-call process budget (fit_seconds records the budget). "
        "Delete the partial to retry the fit.",
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=295.0,
        help="Per-call process budget recorded by --mark-budget-failure.",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Only compute missing partials; never write tracked result tables.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tables")
    parser.add_argument("--output-prefix", default="real_regime_discovery_comparison")
    return parser.parse_args()


def load_config_data(config: dict[str, object], args: argparse.Namespace) -> tuple[np.ndarray, list[str]]:
    """Load and clean a config's series exactly like the canonical real runs."""

    loader_args = argparse.Namespace(
        input_csv=Path(str(config["input_csv"])),
        labels_csv=Path(str(config["labels_csv"])),
        dataset_name="",
        label_column=str(config["label_column"]),
        columns=str(config["columns"]),
        time_column="time_index",
        missing=args.missing,
    )
    _name, x, _columns, labels = base.load_series_and_labels(loader_args)
    return x, labels


def first_transition(labels: list[str]) -> int:
    for idx in range(1, len(labels)):
        if labels[idx] != labels[idx - 1]:
            return idx
    raise ValueError("Label sequence has no transition")


def apply_cap(
    config: dict[str, object],
    x: np.ndarray,
    labels: list[str],
) -> tuple[np.ndarray, list[str], str]:
    """Return the capped window shared by every method, plus a description."""

    if config["cap_mode"] == "segment":
        cap_t = int(config["cap_t"])
        start = max(0, first_transition(labels) - cap_t // 2)
        end = min(len(labels), start + cap_t)
        description = f"segment [{start}, {end}) centered on first label transition"
        return x[start:end], labels[start:end], description

    block_labels = [str(item) for item in config["block_labels"]]  # type: ignore[index]
    block_n = int(config["block_n"])
    index_chunks: list[np.ndarray] = []
    for block in block_labels:
        positions = np.flatnonzero(np.asarray([value == block for value in labels]))
        if positions.size < block_n:
            raise ValueError(f"Label block {block!r} has only {positions.size} samples (< {block_n})")
        index_chunks.append(positions[:block_n])
    indices = np.concatenate(index_chunks)
    description = f"block-concat first {block_n} samples of {block_labels}"
    return x[indices], [labels[int(idx)] for idx in indices], description


def stcg_states(
    x: np.ndarray,
    bounds: tuple[tuple[int, int], ...],
    oracle_k: int,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Recompute STCG-v1 states on the capped window via the forward harness."""

    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    config = STCGStateConfig(
        n_states=oracle_k,
        base_model="lasso",
        lasso_alpha=args.lasso_alpha,
        window_size=args.window_size,
        stride=args.window_stride,
        auto_states=True,
        min_state_improvement=0.0,
        contrast_mix=0.0,
        seed=seed + 95_000,
    )
    features = forward.window_features(x, bounds, args.max_lag, config)
    all_indices = np.arange(len(bounds), dtype=int)
    states, improvement, coherence, accepted = forward.assign_train_fitted_states(features, all_indices, config)
    info = {
        "state_improvement": improvement,
        "temporal_coherence": coherence,
        "states_accepted": accepted,
    }
    return np.asarray(states, dtype=int), info


def tile_states_to_samples(states: np.ndarray, bounds: tuple[tuple[int, int], ...], n: int) -> np.ndarray:
    """Assign each sample the state of the nearest window center."""

    centers = np.asarray([(start + end) / 2.0 for start, end in bounds], dtype=float)
    positions = np.arange(n, dtype=float)
    nearest = np.abs(positions[:, None] - centers[None, :]).argmin(axis=1)
    return np.asarray(states, dtype=int)[nearest]


def window_majority_of_sequence(
    sequence: np.ndarray,
    offset: int,
    bounds: tuple[tuple[int, int], ...],
    n: int,
) -> list[int]:
    """Majority predicted regime per window over a per-sample assignment.

    ``sequence`` is aligned to ``t >= offset`` (RPCMCI: ``max_lag``; MSVAR:
    ``2 * max_lag``); earlier samples are invalid and excluded from the vote.
    """

    full = np.full(n, -1, dtype=int)
    full[offset : offset + len(sequence)] = sequence[: max(0, n - offset)]
    window_states: list[int] = []
    for start, end in bounds:
        valid = full[start:end][full[start:end] >= 0]
        if valid.size == 0:
            window_states.append(-1)
            continue
        counts = np.bincount(valid)
        window_states.append(int(np.flatnonzero(counts == counts.max())[0]))
    return window_states


def alignment_metrics(true: list[str], predicted: list[int]) -> tuple[float, float, str]:
    """Hungarian aligned accuracy + ARI, the ``compare_real_state_labels`` protocol."""

    from sklearn.metrics import adjusted_rand_score

    _aligned, mapping, accuracy = _common.align_labels([str(value) for value in true], [int(value) for value in predicted])
    ari = float(adjusted_rand_score([str(value) for value in true], [int(value) for value in predicted]))
    return ari, accuracy, json.dumps({int(k): str(v) for k, v in mapping.items()}, sort_keys=True)


def compute_row(config_name: str, method: str, seed: int, args: argparse.Namespace) -> dict[str, object]:
    config = CONFIGS[config_name]
    x, labels = load_config_data(config, args)
    x, labels, cap_description = apply_cap(config, x, labels)
    bounds = tuple(sliding_window_bounds(x.shape[0], args.max_lag, args.window_size, args.window_stride))
    true_window_labels, _purities = base.window_majority_labels(labels, bounds)
    oracle_k = int(config["oracle_k"])
    k_present = len(set(labels))

    started = time.perf_counter()
    status = "ok"
    extras: dict[str, object] = {
        "error_free_annealings": "",
        "converged_fraction": "",
        "num_iterations": "",
        "max_anneal": "",
        "max_transitions": "",
    }
    try:
        if method == METHOD_STCG:
            window_states, _info = stcg_states(x, bounds, oracle_k, args, seed)
            sample_states = tile_states_to_samples(window_states, bounds, x.shape[0])
            sample_offset = 0
        elif method == METHOD_RPCMCI:
            _scores, info = rpcmci_scores(
                x,
                max_lag=args.max_lag,
                num_regimes=oracle_k,
                max_transitions=args.max_transitions,
                switch_thres=args.switch_thres,
                num_iterations=args.num_iterations,
                max_anneal=args.max_anneal,
                pc_alpha=args.pc_alpha,
                alpha_level=args.alpha_level,
                seed=seed,
                n_jobs=args.n_jobs,
                return_info=True,
            )
            sample_states = np.asarray(info["regime_sequence"], dtype=int)
            sample_offset = int(info["effective_offset"])
            window_states = window_majority_of_sequence(sample_states, sample_offset, bounds, x.shape[0])
            extras.update(
                {
                    "error_free_annealings": info["error_free_annealings"],
                    "num_iterations": info["num_iterations"],
                    "max_anneal": info["max_anneal"],
                    "max_transitions": info["max_transitions"],
                }
            )
        elif method == METHOD_MSVAR:
            from stcg import msvar_scores

            _scores, info = msvar_scores(
                x,
                max_lag=args.max_lag,
                n_regimes=oracle_k,
                maxiter=args.msvar_maxiter,
                retry_maxiter=args.msvar_retry_maxiter,
                search_reps=args.msvar_search_reps,
                return_info=True,
                seed=seed,
            )
            sequence = info["regime_sequence"]
            if sequence is None:
                raise RuntimeError("MSVAR returned no regime posteriors (all target equations failed)")
            sample_states = np.asarray(sequence, dtype=int)
            sample_offset = int(info["effective_offset"])
            window_states = window_majority_of_sequence(sample_states, sample_offset, bounds, x.shape[0])
            converged = list(info["converged"])
            extras["converged_fraction"] = sum(converged) / len(converged) if converged else 0.0
            if list(info["failed_targets"]):
                status = "partial"
        else:  # pragma: no cover - argparse choices prevent this
            raise ValueError(f"Unknown method: {method}")
    except Exception as exc:  # flagged NaN row keeps the grid resumable
        print(f"[real-regime] {config_name}/{method}/seed{seed} failed: {exc}", file=sys.stderr, flush=True)
        status = "failed"
        window_states = [-1] * len(bounds)
        sample_states = np.zeros(0, dtype=int)
        sample_offset = 0
        if method == METHOD_RPCMCI:
            # rpcmci_scores raises only when every annealing failed.
            extras.update(
                {
                    "error_free_annealings": 0,
                    "num_iterations": args.num_iterations,
                    "max_anneal": args.max_anneal,
                    "max_transitions": args.max_transitions,
                }
            )

    fit_seconds = time.perf_counter() - started

    if status == "failed":
        window_ari = window_accuracy = sample_ari = sample_accuracy = math.nan
        mapping_text = "{}"
        n_samples_scored = 0
        pred_window_counts = ""
    else:
        window_ari, window_accuracy, mapping_text = alignment_metrics(true_window_labels, list(window_states))
        pred_window_counts = _common.counts_text(list(window_states))
        true_samples = labels[sample_offset : sample_offset + len(sample_states)]
        scored = sample_states[: len(true_samples)]
        if len(true_samples) and len(scored):
            sample_ari, sample_accuracy, _mapping = alignment_metrics(true_samples, [int(v) for v in scored])
            n_samples_scored = len(scored)
        else:
            sample_ari = sample_accuracy = math.nan
            n_samples_scored = 0

    row: dict[str, object] = {
        "config": config_name,
        "method": method,
        "seed": seed,
        "status": status,
        "oracle_k": oracle_k,
        "k_present": k_present,
        "cap_mode": str(config["cap_mode"]),
        "cap_description": cap_description,
        "n_timesteps": int(x.shape[0]),
        "n_channels": int(x.shape[1]),
        "n_windows": len(bounds),
        "true_window_counts": _common.counts_text(true_window_labels),
        "pred_window_counts": pred_window_counts,
        "window_ari": window_ari,
        "window_aligned_accuracy": window_accuracy,
        "sample_ari": sample_ari,
        "sample_aligned_accuracy": sample_accuracy,
        "n_samples_scored": n_samples_scored,
        "state_mapping": mapping_text,
        "fit_seconds": fit_seconds,
    }
    row.update(extras)
    return row


def part_path(parts_dir: Path, config_name: str, method: str, seed: int) -> Path:
    safe_method = method.replace(" ", "").replace("-", "").lower()
    return parts_dir / f"{config_name}__{safe_method}__seed{seed:03d}.csv"


def read_part(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        import csv

        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[0]
    return {key: (int(value) if key == "seed" else value) for key, value in row.items()}


def method_seeds(method: str, args: argparse.Namespace) -> list[int]:
    if method == METHOD_STCG:
        return list(args.stcg_seeds)
    return list(args.seeds)


def mean_std(values: list[float]) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan, math.nan
    mean = sum(finite) / len(finite)
    if len(finite) <= 1:
        return mean, 0.0
    return mean, math.sqrt(sum((value - mean) ** 2 for value in finite) / (len(finite) - 1))


def aggregate(detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in detail_rows:
        groups.setdefault((str(row["config"]), str(row["method"])), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (config_name, method), items in sorted(groups.items()):
        summary: dict[str, object] = {
            "config": config_name,
            "method": method,
            "n": len(items),
            "oracle_k": items[0]["oracle_k"],
            "k_present": items[0]["k_present"],
            "n_timesteps": items[0]["n_timesteps"],
            "n_windows": items[0]["n_windows"],
            "cap_description": items[0]["cap_description"],
        }
        for column in (
            "window_ari",
            "window_aligned_accuracy",
            "sample_ari",
            "sample_aligned_accuracy",
            "fit_seconds",
        ):
            mean, std = mean_std([float(item[column]) for item in items])
            summary[f"{column}_mean"] = mean
            summary[f"{column}_std"] = std
        summary_rows.append(summary)
    return summary_rows


def write_summary_markdown(
    path: Path,
    rows: list[dict[str, object]],
    missing: list[str] | None = None,
    failed_parts: list[dict[str, object]] | None = None,
) -> None:
    lines = [
        "# Real Regime Discovery Comparison",
        "",
        "Head-to-head regime recovery on real datasets with ground-truth operating-mode labels.",
        "All methods see the SAME capped window per config; RPCMCI/MSVAR receive the true regime",
        "count (oracle-sighted). Window-level metrics follow the tracked state-label alignment",
        "protocol (Hungarian aligned accuracy + ARI over window-majority labels); sample-level",
        "metrics follow the synthetic regime-quality path (per-sample, from each method's",
        "effective offset). STCG states are recomputed on the capped window with the forward",
        "harness machinery (lasso window features + KMeans, window 300 / stride 150).",
        "",
        "| Config | Method | N | K (present) | T | Windows | Window ARI | Window Acc | Sample ARI | Sample Acc | Fit s |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {config} | {method} | {n} | {oracle_k} ({k_present}) | {n_timesteps} | {n_windows} | "
            "{window_ari} | {window_acc} | {sample_ari} | {sample_acc} | {fit_s} |".format(
                window_ari=_common.fmt_na(row["window_ari_mean"], 3)
                + " +/- "
                + _common.fmt_na(row["window_ari_std"], 3),
                window_acc=_common.fmt_na(row["window_aligned_accuracy_mean"], 3)
                + " +/- "
                + _common.fmt_na(row["window_aligned_accuracy_std"], 3),
                sample_ari=_common.fmt_na(row["sample_ari_mean"], 3)
                + " +/- "
                + _common.fmt_na(row["sample_ari_std"], 3),
                sample_acc=_common.fmt_na(row["sample_aligned_accuracy_mean"], 3)
                + " +/- "
                + _common.fmt_na(row["sample_aligned_accuracy_std"], 3),
                fit_s=_common.fmt_na(row["fit_seconds_mean"], 1),
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Protocol notes",
            "",
            "- Cap protocol per config is recorded in the `cap_description` column of the CSV.",
            "- RPCMCI budget matches the synthetic runner: num_iterations=20, max_anneal=10,",
            "  pc_alpha=0.05, alpha_level=0.01, tau_max=3, max_transitions=10, switch_thres=0.05.",
            "- MSVAR budget where a fit completed: maxiter=100, retry_maxiter=200, search_reps=1",
            "  (trimmed from the synthetic 200/500/3 so a fit finishes inside the per-call",
            "  process budget).",
            "- Rows whose fit failed stay in the gitignored partials and are excluded here;",
            "  `fit_seconds=295` on a failed MSVAR partial means the process was killed at the",
            "  per-call budget, not that the fit finished.",
        ]
    )
    if failed_parts:
        lines.extend(
            [
                "",
                "## Failed fits (excluded from the aggregates above)",
                "",
                "| Config | Method | Seed | Fit s | Recorded cause |",
                "|---|---|---:|---:|---|",
            ]
        )
        for part in failed_parts:
            if str(part.get("error_free_annealings", "")) == "0":
                cause = "no RPCMCI annealing converged (tigramite `run_rpcmci` returned None)"
            else:
                cause = str(part.get("cap_description", "fit failed"))
            lines.append(
                "| {config} | {method} | {seed} | {fit_s} | {cause} |".format(
                    config=part["config"],
                    method=part["method"],
                    seed=part["seed"],
                    fit_s=_common.fmt_na_lenient(part.get("fit_seconds"), 1),
                    cause=cause,
                )
            )
    if missing:
        lines.extend(
            [
                "",
                "## Not-run plan combos",
                "",
                "Aggregated with `--aggregate-partial`; these planned combos have no partial:",
                "",
            ]
            + [f"- `{item}`" for item in missing]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def budget_failure_row(config_name: str, method: str, seed: int, args: argparse.Namespace) -> dict[str, object]:
    """Failed-fit row for a fit that genuinely exceeded the per-call budget."""

    config = CONFIGS[config_name]
    row: dict[str, object] = {
        "config": config_name,
        "method": method,
        "seed": seed,
        "status": "failed",
        "oracle_k": int(config["oracle_k"]),
        "k_present": "",
        "cap_mode": str(config["cap_mode"]),
        "cap_description": f"killed at per-call budget of {args.budget_seconds:.0f}s (fit did not finish)",
        "n_timesteps": "",
        "n_channels": "",
        "n_windows": "",
        "true_window_counts": "",
        "pred_window_counts": "",
        "window_ari": math.nan,
        "window_aligned_accuracy": math.nan,
        "sample_ari": math.nan,
        "sample_aligned_accuracy": math.nan,
        "n_samples_scored": 0,
        "state_mapping": "{}",
        "error_free_annealings": "",
        "converged_fraction": "",
        "num_iterations": "",
        "max_anneal": "",
        "max_transitions": "",
        "fit_seconds": float(args.budget_seconds),
    }
    return row


def main() -> None:
    args = parse_args()
    if METHOD_RPCMCI in args.methods and METHOD_STCG not in args.methods:
        # The imported real-data harness modules set LOKY_MAX_CPU_COUNT=1 for
        # sklearn KMeans; loky reads it at executor creation, so popping it
        # here restores RPCMCI's joblib annealing parallelism.
        os.environ.pop("LOKY_MAX_CPU_COUNT", None)

    plan = [
        (config_name, method, seed)
        for config_name in args.configs
        for method in args.methods
        for seed in method_seeds(method, args)
    ]

    computed: list[str] = []
    if not args.aggregate_only:
        args.parts_dir.mkdir(parents=True, exist_ok=True)
        for config_name, method, seed_text in args.mark_budget_failure:
            path = part_path(args.parts_dir, config_name, method, int(seed_text))
            if read_part(path) is not None:
                continue
            row = budget_failure_row(config_name, method, int(seed_text), args)
            _common.write_csv(path, [row], DETAIL_FIELDNAMES)
            computed.append(f"{config_name}/{method}/seed{seed_text}(budget-failure)")
            print(f"[real-regime] wrote {path}", flush=True)
        for config_name, method, seed in plan:
            path = part_path(args.parts_dir, config_name, method, seed)
            if read_part(path) is not None:
                continue
            row = compute_row(config_name, method, seed, args)
            _common.write_csv(path, [row], DETAIL_FIELDNAMES)
            computed.append(f"{config_name}/{method}/seed{seed}")
            print(f"[real-regime] wrote {path}", flush=True)

    missing = [
        f"{config_name}/{method}/seed{seed}"
        for config_name, method, seed in plan
        if read_part(part_path(args.parts_dir, config_name, method, seed)) is None
    ]
    if args.no_aggregate:
        print(
            json.dumps(
                {"status": "partial" if missing else "complete", "computed": computed, "missing": missing},
                indent=2,
                sort_keys=True,
            )
        )
        return
    if missing and not args.aggregate_partial:
        if args.aggregate_only:
            raise SystemExit(f"Cannot aggregate; missing partials: {missing}")
        print(
            json.dumps(
                {"status": "partial", "computed": computed, "missing": missing},
                indent=2,
                sort_keys=True,
            )
        )
        return

    detail_rows: list[dict[str, object]] = []
    failed_rows: list[str] = []
    failed_parts: list[dict[str, object]] = []
    for config_name, method, seed in plan:
        part = read_part(part_path(args.parts_dir, config_name, method, seed))
        if part is None:
            continue  # aggregate-partial mode; listed in `missing`
        part["seed"] = int(part["seed"])
        if str(part["status"]) == "failed":
            # Keep failed fits out of the tracked metric aggregates; the
            # flagged rows are still surfaced in the markdown's failed-fits
            # section and stay in the gitignored partials.
            failed_rows.append(f"{config_name}/{method}/seed{seed}")
            failed_parts.append(part)
            continue
        detail_rows.append(part)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / f"{args.output_prefix}_detail.csv"
    summary_path = args.output_dir / f"{args.output_prefix}.csv"
    markdown_path = args.output_dir / f"{args.output_prefix}.md"

    summary_rows = aggregate(detail_rows)
    _common.write_csv(detail_path, detail_rows, DETAIL_FIELDNAMES)
    _common.write_csv(summary_path, summary_rows, SUMMARY_FIELDNAMES)
    write_summary_markdown(markdown_path, summary_rows, missing=missing, failed_parts=failed_parts)

    print(
        json.dumps(
            {
                "detail_csv": str(detail_path),
                "summary_csv": str(summary_path),
                "summary_md": str(markdown_path),
                "computed": computed,
                "failed_rows": failed_rows,
                "missing": missing,
                "n_detail_rows": len(detail_rows),
                "n_summary_rows": len(summary_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
