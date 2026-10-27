"""Controlled switching-VAR systems with known graphs and negative controls.

The generator separates three things that the earlier panel conflated:

1. a change in the *coefficient* matrix,
2. a change in the *support* of the coefficient matrix,
3. a change in something else (mean, variance, trend, lag, sign).

Only (2) changes the regime-specific graph.  A negative control that changes
(1) or (3) but not (2) must not be credited as graph-structure recovery.

Every generated system reports the companion-matrix spectral radius of each
state, the realised support overlap, the realised state durations, and the
per-window purity for a requested window length, so that a result can be
attributed to a configuration rather than to an accidental instability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "MechanismConfig",
    "MechanismDataset",
    "generate_mechanism",
    "spectral_radius",
    "misassign_labels",
    "structure_counts",
    "CONTROL_KINDS",
]

CONTROL_KINDS = (
    None,
    "same_support",
    "variance_only",
    "mean_only",
    "trend",
    "sign_flip",
    "lag_shift",
    "stationary",
)


@dataclass(frozen=True)
class MechanismConfig:
    """Configuration of a controlled switching VAR.

    ``overlap`` is the fraction of state-0 edges that also appear in state 1.
    New state-1-specific edges are added at rate ``edge_prob * (1 - overlap)``
    so that the union density stays roughly constant as overlap varies.
    """

    n_nodes: int = 8
    timesteps: int = 1200
    max_lag: int = 3
    edge_prob: float = 0.18
    n_states: int = 2
    switch_period: int = 300
    overlap: float = 0.5
    effect_scale: float = 1.0
    noise_scale: float = 0.10
    observation_noise: float = 0.0
    noise_kind: str = "gaussian"
    state_proportions: tuple[float, ...] | None = None
    control: str | None = None
    control_scale: float = 2.0
    burn_in: int = 100
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_nodes < 2:
            raise ValueError("n_nodes must be at least 2")
        if self.timesteps <= self.max_lag + 1:
            raise ValueError("timesteps must exceed max_lag + 1")
        if not 0.0 <= self.overlap <= 1.0:
            raise ValueError("overlap must lie in [0, 1]")
        if self.control not in CONTROL_KINDS:
            raise ValueError(f"Unknown control: {self.control}")
        if self.n_states < 1:
            raise ValueError("n_states must be at least 1")


@dataclass(frozen=True)
class MechanismDataset:
    """Generated series plus every quantity needed to audit it."""

    x: np.ndarray
    coefficients: np.ndarray  # (n_states, max_lag, nodes, nodes)
    state_support: np.ndarray  # (n_states, max_lag, nodes, nodes)
    union_support: np.ndarray  # (max_lag, nodes, nodes)
    regimes: np.ndarray  # (timesteps,)
    spectral_radius: np.ndarray  # (n_states,)
    overlap_realised: float
    durations: dict[int, int] = field(default_factory=dict)
    config: MechanismConfig | None = None

    @property
    def n_switches(self) -> int:
        return int(np.sum(self.regimes[1:] != self.regimes[:-1]))


def _sample_support(
    rng: np.random.Generator, n_nodes: int, max_lag: int, edge_prob: float
) -> np.ndarray:
    support = np.zeros((max_lag, n_nodes, n_nodes), dtype=bool)
    for lag in range(max_lag):
        mask = rng.random((n_nodes, n_nodes)) < edge_prob
        np.fill_diagonal(mask, False)
        support[lag] = mask
    if not support.any():
        source = int(rng.integers(0, n_nodes))
        target = (source + 1) % n_nodes
        support[0, source, target] = True
    return support


def _coefficients_from_support(
    rng: np.random.Generator, support: np.ndarray, effect_scale: float
) -> np.ndarray:
    max_lag = support.shape[0]
    signs = rng.choice([-1.0, 1.0], size=support.shape)
    magnitudes = rng.uniform(0.08, 0.35, size=support.shape)
    decay = np.arange(1, max_lag + 1, dtype=float).reshape(-1, 1, 1)
    return support.astype(float) * signs * magnitudes * effect_scale / decay


def spectral_radius(coefficients: np.ndarray) -> float:
    """Spectral radius of the companion matrix of a lagged coefficient tensor.

    ``coefficients[lag-1, source, target]`` is the effect of ``source(t-lag)``
    on ``target(t)``, so the column-vector state form uses the transpose of
    each block.
    """

    max_lag, n_nodes, _ = coefficients.shape
    if max_lag == 1:
        return float(np.max(np.abs(np.linalg.eigvals(coefficients[0]))))
    companion = np.zeros((n_nodes * max_lag, n_nodes * max_lag), dtype=float)
    for lag in range(max_lag):
        companion[:n_nodes, lag * n_nodes : (lag + 1) * n_nodes] = coefficients[lag].T
    companion[n_nodes:, : n_nodes * (max_lag - 1)] = np.eye(n_nodes * (max_lag - 1))
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def _stabilise(coefficients: np.ndarray, target: float = 0.90) -> np.ndarray:
    radius = spectral_radius(coefficients)
    if radius <= target or radius == 0.0:
        return coefficients
    return coefficients * (target / radius)


def _regime_schedule(config: MechanismConfig, total: int) -> tuple[np.ndarray, dict[int, int]]:
    """Build a repeating cycle of states with optional unequal proportions."""

    if config.n_states == 1 or config.control == "stationary":
        return np.zeros(total, dtype=int), {0: total}

    durations: dict[int, int] = {}
    if config.state_proportions is None:
        lengths = [int(config.switch_period)] * config.n_states
    else:
        proportions = np.asarray(config.state_proportions, dtype=float)
        if proportions.size != config.n_states:
            raise ValueError("state_proportions must have one entry per state")
        if not np.isclose(proportions.sum(), 1.0):
            raise ValueError("state_proportions must sum to one")
        if np.any(proportions <= 0):
            raise ValueError("state_proportions must be positive")
        cycle = int(config.switch_period) * config.n_states
        lengths = [max(config.max_lag + 2, int(round(cycle * p))) for p in proportions]

    regimes = np.zeros(total, dtype=int)
    position = 0
    state = 0
    while position < total:
        length = lengths[state % len(lengths)]
        end = min(total, position + length)
        regimes[position:end] = state % config.n_states
        durations[state % config.n_states] = durations.get(state % config.n_states, 0) + (end - position)
        position = end
        state += 1
    return regimes, durations


def _build_coefficients(
    config: MechanismConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    base_support = _sample_support(rng, config.n_nodes, config.max_lag, config.edge_prob)
    base = _coefficients_from_support(rng, base_support, config.effect_scale)

    if config.n_states == 1 or config.control == "stationary":
        return base[None, ...], base_support[None, ...]

    if config.control == "same_support":
        second_support = base_support.copy()
    elif config.control == "lag_shift":
        second_support = np.zeros_like(base_support)
        for lag in range(config.max_lag):
            second_support[(lag + 1) % config.max_lag] = base_support[lag]
    elif config.control in ("variance_only", "mean_only", "trend"):
        # Nuisance-only controls keep the coefficient matrix identical.
        second_support = base_support.copy()
    else:
        keep = rng.random(base_support.shape) < config.overlap
        second_support = base_support & keep
        additions = (~base_support) & (
            rng.random(base_support.shape) < config.edge_prob * (1.0 - config.overlap)
        )
        second_support |= additions
        if not second_support.any():
            second_support = base_support.copy()

    if config.control == "sign_flip":
        second = -base
    elif config.control in ("variance_only", "mean_only", "trend"):
        second = base.copy()
    else:
        second = _coefficients_from_support(rng, second_support, config.effect_scale)

    coefficients = np.stack([base, second], axis=0)
    support = np.stack([base_support, second_support], axis=0)
    for state in range(coefficients.shape[0]):
        coefficients[state] = _stabilise(coefficients[state])
    support = (np.abs(coefficients) > 0).astype(int)
    return coefficients, support


def _noise(
    rng: np.random.Generator, size: int, scale: float, kind: str, phase: float = 0.0
) -> np.ndarray:
    if kind == "gaussian":
        return rng.normal(0.0, scale, size=size)
    if kind == "t":
        return rng.standard_t(df=5, size=size) * scale / np.sqrt(5.0 / 3.0)
    if kind == "heteroscedastic":
        base = rng.normal(0.0, scale, size=size)
        modulation = 1.0 + 0.5 * np.sin(phase / 25.0)
        return base * modulation
    raise ValueError(f"Unknown noise kind: {kind}")


def generate_mechanism(config: MechanismConfig | None = None) -> MechanismDataset:
    """Generate a controlled switching VAR with auditable graph truth."""

    if config is None:
        config = MechanismConfig()

    rng = np.random.default_rng(config.seed)
    total = config.timesteps + config.burn_in
    coefficients, state_support = _build_coefficients(config, rng)
    n_states = coefficients.shape[0]

    regimes_full, durations = _regime_schedule(config, total)
    if n_states == 1:
        regimes_full = np.zeros(total, dtype=int)

    n_nodes = config.n_nodes
    x = rng.normal(0.0, config.noise_scale, size=(total, n_nodes))
    mean_offset = np.zeros((n_states, n_nodes))
    if config.control == "mean_only":
        mean_offset = rng.normal(0.0, config.control_scale, size=(n_states, n_nodes))
        mean_offset[0] = 0.0
    trend = np.zeros(n_nodes)
    if config.control == "trend":
        trend = rng.normal(0.0, config.control_scale, size=n_nodes)

    noise_scale = np.full(n_states, config.noise_scale)
    if config.control == "variance_only":
        noise_scale[1] = config.noise_scale * config.control_scale

    for t in range(config.max_lag, total):
        state = int(regimes_full[t])
        value = np.zeros(n_nodes)
        for lag in range(1, config.max_lag + 1):
            # coefficients[..., source, target] applied to source(t - lag)
            value += x[t - lag] @ coefficients[state, lag - 1]
        value += mean_offset[state]
        if config.control == "trend" and state == 1:
            # A state-specific linear trend: state 1 drifts, state 0 does not.
            # This keeps a genuine dynamical difference (so the two regimes are
            # not label-only) while leaving the graph support untouched.
            value += trend * (t - config.max_lag) / max(1, total - config.max_lag)
        x[t] = value + _noise(
            rng, n_nodes, float(noise_scale[state]), config.noise_kind, phase=t
        )

    observed = slice(config.burn_in, total)
    regimes = regimes_full[observed]
    x_observed = x[observed].copy()
    if config.observation_noise > 0:
        # Measurement contamination. Innovation scale alone is not a usable
        # axis: a linear VAR driven by c*eps has the same standardised series
        # for every c, so scaling eps leaves z-scored Lasso scores unchanged.
        contamination = float(np.std(x_observed))
        x_observed = x_observed + rng.normal(
            0.0,
            config.observation_noise * contamination,
            size=x_observed.shape,
        )
    radii = np.array([spectral_radius(coefficients[state]) for state in range(n_states)])

    union = state_support.max(axis=0)
    shared = int(np.sum(state_support[0] & state_support[1])) if n_states > 1 else int(state_support.sum())
    base_edges = int(state_support[0].sum()) if n_states else 0
    realised_overlap = float(shared / base_edges) if base_edges else float("nan")

    labels, counts = np.unique(regimes, return_counts=True)
    realised_durations = {int(k): int(v) for k, v in zip(labels, counts)}
    return MechanismDataset(
        x=x_observed,
        coefficients=coefficients,
        state_support=state_support,
        union_support=union,
        regimes=regimes,
        spectral_radius=radii,
        overlap_realised=realised_overlap,
        durations=realised_durations,
        config=config,
    )



def structure_counts(dataset: MechanismDataset) -> dict[str, int]:
    """Count dynamical states and distinct binary graph supports.

    ``k_dyn`` is the number of distinct data-generating mechanisms. ``k_graph``
    is the number of distinct binary supports; the two differ whenever states
    share a support but differ in sign, magnitude, noise, or mean. The target of
    a regime-specific binary-support metric is ``k_graph``, not ``k_dyn``.
    """

    coefficients = dataset.coefficients.reshape(dataset.coefficients.shape[0], -1)
    supports = dataset.state_support.reshape(dataset.state_support.shape[0], -1)
    k_coefficient = len({tuple(np.round(row, 12)) for row in coefficients})
    k_graph = len({tuple(int(v) for v in row) for row in supports})
    control = dataset.config.control if dataset.config is not None else None
    if control in ("variance_only", "mean_only", "trend"):
        # These controls change the data distribution per state (variance, mean,
        # or a state-specific trend) while the coefficient matrix and support
        # stay identical, so there are two dynamical mechanisms but one graph.
        k_dyn = 2
    elif control == "stationary":
        k_dyn = 1
    else:
        k_dyn = k_coefficient
    return {"k_dyn": int(k_dyn), "k_graph": int(k_graph)}

def misassign_labels(
    labels: np.ndarray, eta: float, seed: int = 0, n_states: int | None = None
) -> np.ndarray:
    """Corrupt a fraction ``eta`` of state labels uniformly at random.

    Used to trace performance against state-assignment error rather than
    against a clustering method's own accuracy.  ``eta=0`` returns the input.
    """

    labels = np.asarray(labels).astype(int)
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0, 1]")
    if eta == 0.0:
        return labels.copy()
    rng = np.random.default_rng(seed)
    states = np.unique(labels) if n_states is None else np.arange(n_states)
    corrupted = labels.copy()
    flip = rng.random(labels.size) < eta
    for index in np.flatnonzero(flip):
        alternatives = states[states != corrupted[index]]
        if alternatives.size:
            corrupted[index] = int(rng.choice(alternatives))
    return corrupted
