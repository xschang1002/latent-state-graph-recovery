"""Frozen-study primitives; the existing estimator and generator stay unchanged.

The duration and schedule studies share mechanisms, initial values and complete
innovation arrays within a simulation repeat. A schedule is the only input that
differs between paired conditions. Failures are explicit, never regularised away.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path
import sys
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stcg import STCGStateConfig
from stcg.baselines import _zscore, lasso_var_scores, make_lagged_design, sliding_window_bounds
from stcg.evaluation import match_states, nonself_mask, window_purity, window_regime_labels
from stcg.mechanism import MechanismConfig, MechanismDataset, _build_coefficients, spectral_radius
from stcg.self_supervised import _min_cluster_fraction, _state_improvement, _temporal_coherence

COMMON_SOURCES = (
    "scripts/aggregation_study_common.py", "src/stcg/mechanism.py",
    "src/stcg/baselines.py", "src/stcg/self_supervised.py", "src/stcg/evaluation.py",
)


class ScorerFailure(RuntimeError):
    """A fit failed the frozen numerical rule; retain its diagnostic details."""

    def __init__(self, message: str, diagnostic: dict | None = None):
        super().__init__(message)
        self.diagnostic = diagnostic or {}


def source_hashes(paths=COMMON_SOURCES) -> dict[str, str]:
    return {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def rng_for(repeat_seed: int, domain: int) -> np.random.Generator:
    """Independent, addressable streams; never depend on execution/cell order."""
    return np.random.default_rng(np.random.SeedSequence([int(repeat_seed), int(domain)]))


def draw_base_inputs(config: MechanismConfig, repeat_seed: int) -> dict[str, np.ndarray]:
    """Draw all exogenous inputs once, then reuse them for every paired schedule.

    Domains 0/1/2 are respectively mechanism/initial history/full innovations.
    The original support, signed coefficient and one-step stability construction
    is called directly. Actual radii are retained, without resampling or capping.
    """
    if config.control is not None or config.n_states != 2:
        raise ValueError("These studies require two-state support-change mechanisms")
    if config.noise_kind != "gaussian" or config.observation_noise != 0:
        raise ValueError("The frozen common-input simulator uses Gaussian innovations and no observation noise")
    coefficients, support = _build_coefficients(config, rng_for(repeat_seed, 0))
    total = config.timesteps + config.burn_in
    return {
        "coefficients": coefficients,
        "support": support,
        "initial_values": rng_for(repeat_seed, 1).normal(
            0.0, config.noise_scale, (config.max_lag, config.n_nodes)),
        "innovations": rng_for(repeat_seed, 2).normal(
            0.0, config.noise_scale, (total, config.n_nodes)),
        "spectral_radius": np.array([spectral_radius(c) for c in coefficients]),
        "repeat_seed_words": np.asarray(
            [(int(repeat_seed) >> (32 * k)) & 0xffffffff for k in range(4)], dtype=np.uint32),
    }


def fixed_schedule(total: int, duration: int, phase: int = 0) -> np.ndarray:
    if total < 1 or duration < 1 or phase < 0:
        raise ValueError("Schedule length and duration must be positive, phase nonnegative")
    return ((np.arange(total, dtype=np.int64) + int(phase)) // int(duration) % 2).astype(np.int8)


def simulate_schedule(inputs: dict[str, np.ndarray], regimes_full: np.ndarray,
                      config: MechanismConfig) -> MechanismDataset:
    """Apply a full-time schedule, then discard the same burn-in for all cells."""
    total = config.timesteps + config.burn_in
    schedule = np.asarray(regimes_full)
    coefficients = np.asarray(inputs["coefficients"])
    support = np.asarray(inputs["support"])
    expected = (2, config.max_lag, config.n_nodes, config.n_nodes)
    if schedule.shape != (total,) or not set(np.unique(schedule)).issubset({0, 1}):
        raise ValueError("Expected a 0/1 full-time schedule including the burn-in")
    if coefficients.shape != expected or support.shape != expected:
        raise ValueError("Mechanism inputs do not match the simulation configuration")
    initial = np.asarray(inputs["initial_values"])
    innovations = np.asarray(inputs["innovations"])
    if initial.shape != (config.max_lag, config.n_nodes) or innovations.shape != (total, config.n_nodes):
        raise ValueError("Exogenous arrays do not match the full-time configuration")
    x = np.empty((total, config.n_nodes), dtype=float)
    x[:config.max_lag] = initial
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(config.max_lag, total):
            value = np.zeros(config.n_nodes)
            for lag in range(1, config.max_lag + 1):
                value += x[t - lag] @ coefficients[int(schedule[t]), lag - 1]
            x[t] = value + innovations[t]
    if not np.isfinite(x).all():
        raise ScorerFailure("Nonfinite simulation; exogenous inputs retained", {"simulation_nonfinite": True})
    regimes = schedule[config.burn_in:].astype(int, copy=True)
    labels, counts = np.unique(regimes, return_counts=True)
    base_edges = int(support[0].sum())
    return MechanismDataset(
        x=x[config.burn_in:].copy(), coefficients=coefficients.copy(),
        state_support=support.copy(), union_support=support.max(axis=0), regimes=regimes,
        spectral_radius=np.array([spectral_radius(c) for c in coefficients]),
        overlap_realised=float(np.sum(support[0] & support[1]) / base_edges) if base_edges else np.nan,
        durations={int(k): int(v) for k, v in zip(labels, counts)}, config=config,
    )


def matched_partitions(labels: np.ndarray):
    """Enumerate fixed-size unlabelled bipartitions, including the fitted one."""
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) != 2:
        raise ValueError("Matched random grouping needs two nonempty candidate groups")
    size = int(counts.min())
    for members in combinations(range(len(labels)), size):
        if 2 * size == len(labels) and 0 not in members:
            continue
        result = np.zeros(len(labels), dtype=np.int8)
        result[list(members)] = 1
        yield result


def maps_and_emission(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    levels, inverse = np.unique(labels, return_inverse=True)
    maps = np.stack([scores[labels == label].mean(axis=0) for label in levels])
    return maps, maps[inverse]


def evaluate_predictions(predictions: np.ndarray, bounds, dataset: MechanismDataset,
                         union_scores: np.ndarray | None = None) -> dict:
    """Evaluate explicit window emissions; exact pair counts preserve all ties.

    Vectorisation avoids thousands of repeated per-edge sklearn calls during
    exact enumeration. Tests compare both temporal targets with public metrics.
    """
    predictions = np.asarray(predictions)
    if predictions.shape != (len(bounds), *dataset.state_support.shape[1:]):
        raise ValueError("Prediction array and windows/support shape disagree")
    mask = nonself_mask(dataset.state_support.shape[1:])
    truth = window_regime_labels(bounds, dataset.regimes)
    active = dataset.state_support[truth][:, mask].astype(bool)
    scores = predictions[:, mask]
    changed = np.any(dataset.state_support[:, mask] != dataset.state_support[0, mask], axis=0)
    counts = active.sum(axis=0)
    usable = changed & (counts > 0) & (counts < len(bounds))
    if usable.any():
        y, s = active[:, usable], scores[:, usable]
        pairs = y[:, None, :] & ~y[None, :, :]
        differences = s[:, None, :] - s[None, :, :]
        credit = ((differences > 0) + 0.5 * (differences == 0)) * pairs
        edge_aucs = credit.sum(axis=(0, 1)) / pairs.sum(axis=(0, 1))
        temporal = float(edge_aucs.mean())
    else:
        temporal = np.nan
    pooled = float(roc_auc_score(active.reshape(-1), scores.reshape(-1))) if np.unique(active).size == 2 else np.nan
    union = predictions.max(axis=0) if union_scores is None else np.asarray(union_scores)
    union_truth = dataset.union_support[mask]
    union_auc = float(roc_auc_score(union_truth, union[mask])) if np.unique(union_truth).size == 2 else np.nan
    return {
        "temporal_delta_auroc": temporal, "temporal_delta_edges": int(usable.sum()),
        "temporal_delta_edges_total": int(changed.sum()), "pooled_temporal_auroc": pooled,
        "union_auroc": union_auc, "evaluable": bool(np.isfinite(temporal)),
        "undefined_reason": "" if usable.any() else ("no_support_change" if not changed.any() else "no_observed_flip"),
    }


def hard_factors(maps: np.ndarray, labels: np.ndarray, bounds, dataset: MechanismDataset) -> dict:
    """Only defined for two observed true states and two predicted maps."""
    empty = {"J": np.nan, "C_edge": np.nan, "J_times_C": np.nan, "identity_gap": np.nan}
    true = window_regime_labels(bounds, dataset.regimes)
    if len(np.unique(labels)) != 2 or set(np.unique(true)) != {0, 1}:
        return empty
    mask = nonself_mask(dataset.state_support.shape[1:])
    delta = dataset.state_support[1].astype(int) - dataset.state_support[0].astype(int)
    changing = (delta != 0) & mask
    if not changing.any():
        return empty
    levels = np.unique(labels)
    mapping = match_states(labels, true, n_truth_states=2)["mapping"]
    aligned_labels = np.array([mapping[int(v)] for v in labels])
    j = float((aligned_labels[true == 1] == 1).mean() - (aligned_labels[true == 0] == 1).mean())
    aligned_maps = np.empty_like(maps)
    for index, label in enumerate(levels):
        aligned_maps[int(mapping[int(label)])] = maps[index]
    concordance = float(np.sign(delta[changing] * (aligned_maps[1] - aligned_maps[0])[changing]).mean())
    lookup = {int(k): i for i, k in enumerate(levels)}
    emission = maps[[lookup[int(k)] for k in labels]]
    actual = evaluate_predictions(emission, bounds, dataset)["temporal_delta_auroc"]
    return {"J": j, "C_edge": concordance, "J_times_C": j * concordance,
            "identity_gap": float(abs(actual - (0.5 + 0.5 * j * concordance)))}


def _window_scores(x: np.ndarray, max_lag: int, bounds, scorer: str, lasso_alpha: float):
    scores, numerical = [], []
    for window, (start, end) in enumerate(bounds):
        window_x = x[start:end]
        if scorer == "lasso":
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                score = lasso_var_scores(window_x, max_lag, alpha=lasso_alpha)
            convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
            if convergence:
                raise ScorerFailure("Lasso convergence warning", {"window": window, "warnings": convergence})
            numerical.append({"window": window, "warnings": [str(w.message) for w in caught]})
        elif scorer == "ols":
            # Exactly the Lasso preprocessing, with the same self-lag design.
            design, target = make_lagged_design(_zscore(window_x), max_lag)
            design, target = _zscore(design), _zscore(target)
            coeff, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
            info = {"window": window, "rank": int(rank), "n_features": int(design.shape[1]),
                    "n_rows": int(design.shape[0]), "singular_min": float(singular[-1]),
                    "singular_max": float(singular[0])}
            numerical.append(info)
            if rank != design.shape[1]:
                raise ScorerFailure("OLS design is rank deficient; no fallback", {"windows": numerical})
            score = np.abs(coeff.reshape(max_lag, x.shape[1], x.shape[1]))
            index = np.arange(x.shape[1]); score[:, index, index] = 0
        else:
            raise ValueError("Unknown scorer; expected lasso or ols")
        if not np.isfinite(score).all():
            raise ScorerFailure("Nonfinite window scores", {"window": window})
        scores.append(score)
    return np.stack(scores), numerical


def fit_aggregation(dataset: MechanismDataset, scorer: str = "lasso", cluster_seed: int = 0,
                    exact_random: bool = True, *, window_size: int = 300, stride: int = 150,
                    lasso_alpha: float = 0.005, n_init: int = 10) -> tuple[list[dict], dict, dict]:
    """Fit one score pass, share it across every aggregation and save emissions."""
    bounds = tuple(sliding_window_bounds(len(dataset.x), dataset.coefficients.shape[1], window_size, stride))
    if exact_random and len(bounds) > 12:
        raise ValueError("Exact random enumeration is restricted to at most twelve windows")
    scores, numerical = _window_scores(dataset.x, dataset.coefficients.shape[1], bounds, scorer, lasso_alpha)
    mask = nonself_mask(scores.shape[1:])
    features = scores[:, mask]
    centered = features - features.mean(axis=0, keepdims=True)
    scaled = centered / (centered.std(axis=0, keepdims=True) + 1e-8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = KMeans(n_clusters=2, n_init=n_init, random_state=int(cluster_seed))
        candidate = model.fit_predict(scaled)
    improvement = _state_improvement(scaled, float(model.inertia_))
    coherence = _temporal_coherence(candidate)
    config = STCGStateConfig()
    accepted = bool(len(np.unique(candidate)) == 2 and
                    (improvement >= config.min_state_improvement or
                     (improvement >= config.min_temporal_state_improvement and
                      coherence >= config.min_temporal_coherence and
                      _min_cluster_fraction(candidate) >= config.min_temporal_state_fraction)))
    candidate_maps, forced = maps_and_emission(scores, candidate)
    native = forced if accepted else np.repeat(scores.max(axis=0)[None], len(bounds), axis=0)
    truth = window_regime_labels(bounds, dataset.regimes)
    oracle_maps, oracle = maps_and_emission(scores, truth)
    static_mean = np.repeat(scores.mean(axis=0)[None], len(bounds), axis=0)
    predictions = {"raw_window": scores, "native_hard": native, "forced_hard": forced,
                   "oracle_label": oracle, "static_mean": static_mean}
    purity, _ = window_purity(bounds, dataset.regimes)
    factors = hard_factors(candidate_maps, candidate, bounds, dataset)
    info = {"hard_gate_accepted": accepted, "hard_gate_improvement": improvement,
            "hard_gate_coherence": coherence, "candidate_group_count": len(np.unique(candidate)),
            "candidate_group0_n": int((candidate == 0).sum()), "candidate_group1_n": int((candidate == 1).sum()),
            "n_windows": len(bounds), "window_purity_mean": float(purity.mean()),
            "window_purity_min": float(purity.min()), "n_switches": dataset.n_switches,
            "center_switches": int(np.sum(truth[1:] != truth[:-1])),
            "true_window_state0_n": int((truth == 0).sum()), "true_window_state1_n": int((truth == 1).sum()),
            "candidate_J": factors["J"], "candidate_C_edge": factors["C_edge"],
            "candidate_J_times_C": factors["J_times_C"], "candidate_identity_gap": factors["identity_gap"],
            "numerical": numerical, "clustering_warnings": [str(w.message) for w in caught]}
    rows = []
    for method, values in predictions.items():
        metric = evaluate_predictions(values, bounds, dataset)
        factor = factors if method == "forced_hard" or (method == "native_hard" and accepted) else (
            hard_factors(oracle_maps, truth, bounds, dataset) if method == "oracle_label" else
            {"J": np.nan, "C_edge": np.nan, "J_times_C": np.nan, "identity_gap": np.nan})
        rows.append({"method": method, "scorer": scorer, **metric, **factor,
                     "aggregation_executed": method in {"forced_hard", "oracle_label"} or
                     (method == "native_hard" and accepted),
                     **{k: v for k, v in info.items() if k not in {"numerical", "clustering_warnings"}}})
    arrays = {"window_scores": scores, "candidate_labels": candidate, "candidate_maps": candidate_maps,
              "true_window_labels": truth, "bounds": np.asarray(bounds), "window_purity": purity,
              "oracle_maps": oracle_maps,
              **{f"emission_{method}": values for method, values in predictions.items()}}
    if exact_random and len(np.unique(candidate)) == 2:
        assignments, values = [], []
        for partition in matched_partitions(candidate):
            _, emitted = maps_and_emission(scores, partition)
            assignments.append(partition); values.append(evaluate_predictions(emitted, bounds, dataset))
        # Average AUROC within repeat, never average emissions before AUROC.
        metrics = {key: float(np.mean([v[key] for v in values])) for key in
                   ("temporal_delta_auroc", "pooled_temporal_auroc", "union_auroc")}
        rows.append({**rows[0], **metrics, "method": "matched_random", "aggregation_executed": True,
                     "n_partitions": len(values)})
        arrays["matched_random_partitions"] = np.stack(assignments)
        for key in ("temporal_delta_auroc", "pooled_temporal_auroc", "union_auroc"):
            arrays[f"matched_random_{key}"] = np.asarray([v[key] for v in values])
        info["n_random_partitions"] = len(values)
    elif exact_random:
        rows.append({**rows[0], "method": "matched_random", "temporal_delta_auroc": np.nan,
                     "pooled_temporal_auroc": np.nan, "union_auroc": np.nan, "evaluable": False,
                     "undefined_reason": "candidate_not_two_groups", "n_partitions": 0})
        info["n_random_partitions"] = 0
    return rows, arrays, info
