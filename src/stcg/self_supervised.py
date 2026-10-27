"""Self-supervised lag-aware graph scoring models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .baselines import (
    _zscore,
    lasso_var_scores,
    make_lagged_design,
    ridge_var_coefficients,
    sliding_window_bounds,
)
from .metrics import lagged_correlation_scores


@dataclass(frozen=True)
class STCGConfig:
    """Training settings for the first self-supervised graph scorer."""

    steps: int = 80
    learning_rate: float = 0.03
    ridge_alpha: float = 1.0
    mask_rate: float = 0.35
    sparsity_weight: float = 0.015
    l1_weight: float = 0.002
    entropy_weight: float = 0.001
    contrast_weight: float = 0.0
    consistency_weight: float = 0.01
    score_prior_mix: float = 0.0
    window_size: int | None = None
    stride: int | None = None
    aggregate: str = "max"
    seed: int = 0


@dataclass(frozen=True)
class STCGStateConfig:
    """Settings for graph-state-aware self-supervised scoring."""

    n_states: int = 2
    base_model: str = "lasso"
    ridge_alpha: float = 1.0
    lasso_alpha: float = 0.005
    window_size: int | None = None
    stride: int | None = None
    aggregate: str = "max"
    state_aggregate: str = "mean"
    state_features: str = "nonself"
    auto_states: bool = True
    min_state_improvement: float = 0.12
    min_temporal_state_improvement: float = 0.04
    min_temporal_coherence: float = 0.55
    min_temporal_state_fraction: float = 0.10
    contrast_mix: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class STCGStateDiagnostics:
    """Intermediate STCG-v1 state inference outputs for analysis figures."""

    scores: np.ndarray
    bounds: tuple[tuple[int, int], ...]
    window_scores: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    state_scores: np.ndarray
    state_improvement: float = 0.0
    temporal_coherence: float = 0.0
    states_accepted: bool = False


def _aggregate_window_scores(window_scores: list[np.ndarray], aggregate: str) -> np.ndarray:
    stacked = np.stack(window_scores, axis=0)
    if aggregate == "max":
        return stacked.max(axis=0)
    if aggregate == "mean":
        return stacked.mean(axis=0)
    if aggregate == "p90":
        return np.quantile(stacked, 0.90, axis=0)
    raise ValueError(f"Unknown aggregate mode: {aggregate}")


def _nonself_flatten(scores: np.ndarray) -> np.ndarray:
    _, n_nodes, _ = scores.shape
    mask = np.ones_like(scores, dtype=bool)
    for node in range(n_nodes):
        mask[:, node, node] = False
    return scores[mask]


def _window_state_features(
    x: np.ndarray,
    bounds: tuple[tuple[int, int], ...],
    window_scores: list[np.ndarray],
    mode: str,
) -> np.ndarray:
    """Build per-window state features for graph-state clustering.

    ``nonself`` (default) keeps only non-self lagged edge scores, so window
    autocorrelation cannot dominate the state representation. ``all`` adds the
    self-lag diagonal, and ``nonself_stats`` additionally appends per-window
    per-node mean/std of the raw series.
    """

    rows: list[np.ndarray] = []
    for (start, end), scores in zip(bounds, window_scores):
        if mode == "nonself":
            row = _nonself_flatten(scores)
        elif mode == "all":
            row = scores.reshape(-1)
        elif mode == "nonself_stats":
            window = x[start:end]
            row = np.concatenate(
                [_nonself_flatten(scores), window.mean(axis=0), window.std(axis=0)]
            )
        else:
            raise ValueError(f"Unknown state_features mode: {mode}")
        rows.append(row)
    return np.stack(rows, axis=0)


def _window_reconstruction_scores(
    x: np.ndarray,
    max_lag: int,
    config: STCGStateConfig,
) -> np.ndarray:
    if config.base_model == "ridge":
        scores = np.abs(ridge_var_coefficients(
            x, max_lag=max_lag, alpha=config.ridge_alpha,
            include_self=config.state_features == "all",
        ))
    elif config.base_model == "lasso":
        scores = lasso_var_scores(
            x, max_lag=max_lag, alpha=config.lasso_alpha,
            include_self=config.state_features == "all",
        )
    else:
        raise ValueError(f"Unknown STCG-v1 base_model: {config.base_model}")
    return scores


def _state_aggregate(scores: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return scores.mean(axis=0)
    if mode == "median":
        return np.median(scores, axis=0)
    if mode == "p75":
        return np.quantile(scores, 0.75, axis=0)
    if mode == "max":
        return scores.max(axis=0)
    raise ValueError(f"Unknown state aggregate mode: {mode}")


def _state_improvement(scaled: np.ndarray, inertia: float) -> float:
    inertia_one = float(np.square(scaled - scaled.mean(axis=0, keepdims=True)).sum())
    if inertia_one <= 1e-8:
        return 0.0
    return max(0.0, (inertia_one - inertia) / inertia_one)


def _temporal_coherence(labels: np.ndarray) -> float:
    if labels.size < 2 or np.unique(labels).size <= 1:
        return 0.0
    observed = float(np.mean(labels[1:] == labels[:-1]))
    _, counts = np.unique(labels, return_counts=True)
    proportions = counts.astype(float) / float(labels.size)
    expected = float(np.square(proportions).sum())
    if expected >= 1.0 - 1e-8:
        return 0.0
    return max(0.0, (observed - expected) / (1.0 - expected))


def _min_cluster_fraction(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.min() / labels.size)


def _infer_state_labels(features: np.ndarray, config: STCGStateConfig) -> tuple[np.ndarray, float, float, bool]:
    if features.shape[0] < 2 or config.n_states <= 1:
        return np.zeros(features.shape[0], dtype=int), 0.0, 0.0, False

    from sklearn.cluster import KMeans

    n_states = min(config.n_states, features.shape[0])
    centered = features - features.mean(axis=0, keepdims=True)
    scaled = centered / (centered.std(axis=0, keepdims=True) + 1e-8)
    model = KMeans(n_clusters=n_states, n_init=10, random_state=config.seed)
    labels = model.fit_predict(scaled)
    improvement = _state_improvement(scaled, float(model.inertia_))
    temporal_coherence = _temporal_coherence(labels)

    if config.auto_states:
        inertia_pass = improvement >= config.min_state_improvement
        temporal_pass = (
            improvement >= config.min_temporal_state_improvement
            and temporal_coherence >= config.min_temporal_coherence
            and _min_cluster_fraction(labels) >= config.min_temporal_state_fraction
        )
        if not (inertia_pass or temporal_pass):
            return np.zeros(features.shape[0], dtype=int), improvement, temporal_coherence, False
        return labels, improvement, temporal_coherence, True

    return labels, improvement, temporal_coherence, True


def _initial_gate_logits(coefficients: np.ndarray) -> np.ndarray:
    magnitudes = np.abs(coefficients)
    positive = magnitudes[magnitudes > 0]
    scale = np.quantile(positive, 0.75) if positive.size else 1.0
    probabilities = np.clip(0.05 + 0.90 * magnitudes / (scale + 1e-8), 0.05, 0.95)
    return np.log(probabilities / (1.0 - probabilities))


def _prepare_supervised_view(x: np.ndarray, max_lag: int) -> tuple[torch.Tensor, torch.Tensor]:
    design, target = make_lagged_design(_zscore(x), max_lag)
    design = _zscore(design)
    target = _zscore(target)
    n_nodes = target.shape[1]
    design = design.reshape(design.shape[0], max_lag, n_nodes)
    return (
        torch.as_tensor(design, dtype=torch.float32),
        torch.as_tensor(target, dtype=torch.float32),
    )


def _normalized_lag_prior(x: np.ndarray, max_lag: int) -> np.ndarray:
    prior = lagged_correlation_scores(x, max_lag)
    positive = prior[prior > 0]
    scale = np.quantile(positive, 0.90) if positive.size else 1.0
    return np.clip(prior / (scale + 1e-8), 0.0, 1.0)


def _fit_window_scores(
    x: np.ndarray,
    max_lag: int,
    config: STCGConfig,
    seed: int,
    previous_gate: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if config.steps <= 0:
        raise ValueError("STCGConfig.steps must be positive")

    torch.manual_seed(seed)
    design, target = _prepare_supervised_view(x, max_lag)
    n_nodes = target.shape[1]
    nonself = torch.ones((max_lag, n_nodes, n_nodes), dtype=torch.float32)
    diag = torch.arange(n_nodes)
    nonself[:, diag, diag] = 0.0

    init_coeff = ridge_var_coefficients(x, max_lag=max_lag, alpha=config.ridge_alpha)
    lag_prior = torch.as_tensor(_normalized_lag_prior(x, max_lag), dtype=torch.float32)
    weights = torch.nn.Parameter(torch.as_tensor(init_coeff, dtype=torch.float32))
    logits = torch.nn.Parameter(torch.as_tensor(_initial_gate_logits(init_coeff), dtype=torch.float32))
    previous = (
        torch.as_tensor(previous_gate, dtype=torch.float32)
        if previous_gate is not None
        else None
    )

    optimizer = torch.optim.Adam([weights, logits], lr=config.learning_rate)
    for _ in range(config.steps):
        optimizer.zero_grad()
        gate = torch.sigmoid(logits) * nonself
        effective = gate * weights * nonself
        prediction = torch.einsum("tli,lij->tj", design, effective)

        reconstruction_mask = (torch.rand_like(target) < config.mask_rate).float()
        if float(reconstruction_mask.sum()) == 0.0:
            reconstruction_mask = torch.ones_like(target)
        recon = ((prediction - target) ** 2 * reconstruction_mask).sum() / reconstruction_mask.sum()

        sparsity = gate.sum() / nonself.sum()
        l1 = effective.abs().sum() / nonself.sum()
        clipped_gate = gate.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(
            clipped_gate * torch.log(clipped_gate)
            + (1.0 - clipped_gate) * torch.log(1.0 - clipped_gate)
        ).sum() / nonself.sum()
        consistency = torch.tensor(0.0)
        if previous is not None:
            consistency = ((gate - previous) ** 2 * nonself).sum() / nonself.sum()
        contrast = ((gate - lag_prior) ** 2 * nonself).sum() / nonself.sum()

        loss = (
            recon
            + config.sparsity_weight * sparsity
            + config.l1_weight * l1
            + config.entropy_weight * entropy
            + config.contrast_weight * contrast
            + config.consistency_weight * consistency
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            weights.mul_(nonself)
            logits[:, diag, diag] = -20.0

    with torch.no_grad():
        gate = torch.sigmoid(logits) * nonself
        learned = gate * weights.abs() * nonself
        learned_scale = torch.quantile(learned[nonself.bool()], 0.90).clamp_min(1e-8)
        prior = lag_prior * nonself
        score = (
            (1.0 - config.score_prior_mix) * learned / learned_scale
            + config.score_prior_mix * prior
        ) * nonself
    return score.numpy(), gate.numpy()


def stcg_v0_scores(x: np.ndarray, max_lag: int, config: STCGConfig | None = None) -> np.ndarray:
    """Train self-supervised lagged gates over windows and return edge scores.

    This v0 model uses only temporal reconstruction as supervision. It does not
    consume synthetic edge labels or regime labels.
    """

    if config is None:
        config = STCGConfig()
    timesteps, _ = x.shape
    bounds = sliding_window_bounds(timesteps, max_lag, config.window_size, config.stride)

    window_scores: list[np.ndarray] = []
    previous_gate: np.ndarray | None = None
    for window_idx, (start, end) in enumerate(bounds):
        scores, previous_gate = _fit_window_scores(
            x[start:end],
            max_lag=max_lag,
            config=config,
            seed=config.seed + window_idx,
            previous_gate=previous_gate,
        )
        window_scores.append(scores)

    return _aggregate_window_scores(window_scores, config.aggregate)


def stcg_v1_diagnostics(
    x: np.ndarray,
    max_lag: int,
    config: STCGStateConfig | None = None,
) -> STCGStateDiagnostics:
    """Infer graph states from windowed reconstruction scores and aggregate by state.

    This diagnostic is self-supervised in the sense that graph states are inferred
    from reconstruction coefficients only. It does not use synthetic edges or
    provided regime labels.
    """

    if config is None:
        config = STCGStateConfig()

    timesteps, _ = x.shape
    bounds = tuple(sliding_window_bounds(timesteps, max_lag, config.window_size, config.stride))
    window_scores = [
        _window_reconstruction_scores(x[start:end], max_lag=max_lag, config=config)
        for start, end in bounds
    ]
    features = _window_state_features(x, bounds, window_scores, config.state_features)
    # Only clustering may use the optional self-lag coefficients. Recovery
    # scores and all graph diagnostics keep the non-self edge convention.
    if config.state_features == "all":
        for scores in window_scores:
            diag = np.arange(scores.shape[1])
            scores[:, diag, diag] = 0.0
    stacked = np.stack(window_scores, axis=0)

    if len(window_scores) == 1:
        labels = np.zeros(1, dtype=int)
        return STCGStateDiagnostics(
            scores=window_scores[0],
            bounds=bounds,
            window_scores=stacked,
            features=features,
            labels=labels,
            state_scores=stacked.copy(),
            state_improvement=0.0,
            temporal_coherence=0.0,
            states_accepted=False,
        )

    labels, state_improvement, temporal_coherence, states_accepted = _infer_state_labels(features, config)

    if np.unique(labels).size <= 1:
        scores = _aggregate_window_scores(window_scores, config.aggregate)
        return STCGStateDiagnostics(
            scores=scores,
            bounds=bounds,
            window_scores=stacked,
            features=features,
            labels=labels,
            state_scores=np.expand_dims(scores, axis=0),
            state_improvement=state_improvement,
            temporal_coherence=temporal_coherence,
            states_accepted=states_accepted,
        )

    state_scores: list[np.ndarray] = []
    for state in sorted(np.unique(labels)):
        members = stacked[labels == state]
        state_score = _state_aggregate(members, config.state_aggregate)
        if config.contrast_mix > 0 and np.unique(labels).size > 1:
            others = stacked[labels != state]
            other_score = _state_aggregate(others, config.state_aggregate)
            specificity = np.maximum(state_score - other_score, 0.0)
            state_score = state_score + config.contrast_mix * specificity
        state_scores.append(state_score)

    scores = _aggregate_window_scores(state_scores, config.aggregate)
    return STCGStateDiagnostics(
        scores=scores,
        bounds=bounds,
        window_scores=stacked,
        features=features,
        labels=labels,
        state_scores=np.stack(state_scores, axis=0),
        state_improvement=state_improvement,
        temporal_coherence=temporal_coherence,
        states_accepted=states_accepted,
    )


def stcg_v1_scores(x: np.ndarray, max_lag: int, config: STCGStateConfig | None = None) -> np.ndarray:
    """Infer graph states from windowed reconstruction scores and aggregate by state."""

    return stcg_v1_diagnostics(x, max_lag=max_lag, config=config).scores
