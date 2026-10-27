"""Classical lagged-edge scoring baselines for synthetic recovery."""

from __future__ import annotations

import numpy as np


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)


def make_lagged_design(x: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a lag-major design matrix for VAR-style regression."""

    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (timesteps, n_nodes), got {x.shape}")
    timesteps, _ = x.shape
    if timesteps <= max_lag + 1:
        raise ValueError("Need more timesteps than max_lag + 1")

    features = []
    for lag in range(1, max_lag + 1):
        features.append(x[max_lag - lag : timesteps - lag])
    design = np.concatenate(features, axis=1)
    target = x[max_lag:]
    return design, target


def random_scores(n_nodes: int, max_lag: int, seed: int = 0) -> np.ndarray:
    """Random non-self lagged edge scores for a chance baseline."""

    rng = np.random.default_rng(seed)
    scores = rng.random((max_lag, n_nodes, n_nodes))
    for node in range(n_nodes):
        scores[:, node, node] = 0.0
    return scores


def ridge_var_scores(x: np.ndarray, max_lag: int, alpha: float = 1.0) -> np.ndarray:
    """Fit a ridge VAR and use absolute coefficients as lagged edge scores."""

    scores = np.abs(ridge_var_coefficients(x, max_lag=max_lag, alpha=alpha))
    for node in range(scores.shape[1]):
        scores[:, node, node] = 0.0
    return scores


def ridge_var_coefficients(
    x: np.ndarray, max_lag: int, alpha: float = 1.0, *, include_self: bool = False
) -> np.ndarray:
    """Fit a ridge VAR and return signed lagged coefficients.

    Self-lag coefficients are retained only when explicitly requested for
    state features; the default non-self graph-scoring behavior is unchanged.
    """

    _, n_nodes = x.shape
    design, target = make_lagged_design(_zscore(x), max_lag)
    design = _zscore(design)
    target = _zscore(target)

    gram = design.T @ design
    penalty = alpha * np.eye(gram.shape[0])
    coeff = np.linalg.solve(gram + penalty, design.T @ target)
    scores = coeff.reshape(max_lag, n_nodes, n_nodes)
    if not include_self:
        for node in range(n_nodes):
            scores[:, node, node] = 0.0
    return scores


def lasso_var_scores(
    x: np.ndarray, max_lag: int, alpha: float = 0.005, *, include_self: bool = False
) -> np.ndarray:
    """Fit Lasso regressions and score absolute coefficients.

    ``include_self=True`` retains fitted self-lag coefficients for state-feature
    ablations. Public graph scores exclude self-lags by default.
    """

    from sklearn.linear_model import Lasso

    _, n_nodes = x.shape
    design, target = make_lagged_design(_zscore(x), max_lag)
    design = _zscore(design)
    target = _zscore(target)

    coeff = np.zeros((max_lag * n_nodes, n_nodes), dtype=float)
    for target_idx in range(n_nodes):
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000, selection="cyclic")
        model.fit(design, target[:, target_idx])
        coeff[:, target_idx] = model.coef_

    scores = np.abs(coeff.reshape(max_lag, n_nodes, n_nodes))
    if not include_self:
        for node in range(n_nodes):
            scores[:, node, node] = 0.0
    return scores


def _ols_rss(design: np.ndarray, target: np.ndarray) -> float:
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    return float(residual.T @ residual)


def granger_f_scores(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Score lagged edges with conditional Granger-style F statistics.

    Each score compares the full lagged linear model for a target node against
    a restricted model that removes one source-lag feature while conditioning
    on every other lagged source. This provides a classical hypothesis-test
    family beside coefficient-magnitude VAR baselines.
    """

    _, n_nodes = x.shape
    design, target = make_lagged_design(_zscore(x), max_lag)
    design = _zscore(design)
    target = _zscore(target)
    n_rows, n_features = design.shape
    df_den = max(1, n_rows - n_features)
    scores = np.zeros((max_lag, n_nodes, n_nodes), dtype=float)

    for target_idx in range(n_nodes):
        y = target[:, target_idx]
        full_rss = max(_ols_rss(design, y), 1e-12)
        full_mse = full_rss / float(df_den)
        for lag in range(max_lag):
            for source_idx in range(n_nodes):
                if source_idx == target_idx:
                    continue
                feature_idx = lag * n_nodes + source_idx
                restricted = np.delete(design, feature_idx, axis=1)
                restricted_rss = _ols_rss(restricted, y)
                numerator = max(0.0, restricted_rss - full_rss)
                scores[lag, source_idx, target_idx] = numerator / full_mse
    return scores


def pcmci_parcorr_scores(
    x: np.ndarray,
    max_lag: int,
    pc_alpha: float = 0.05,
    score: str = "statistic",
) -> np.ndarray:
    """Score lagged edges with Tigramite PCMCI using ParCorr tests.

    Tigramite returns matrices indexed as ``source, target, tau``. This
    function converts them to the repository convention
    ``lag, source, target`` so that recovery metrics and downstream graph
    utilities can consume PCMCI scores without special handling.
    """

    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except ImportError as exc:
        raise ImportError(
            "PCMCI-ParCorr requires tigramite; install it with "
            "`python -m pip install tigramite`."
        ) from exc

    if score not in {"statistic", "p_value"}:
        raise ValueError("score must be 'statistic' or 'p_value'")

    dataframe = pp.DataFrame(_zscore(x))
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=0,
    )
    results = pcmci.run_pcmci(tau_min=1, tau_max=max_lag, pc_alpha=pc_alpha)

    if score == "statistic":
        source_target_lag = np.abs(results["val_matrix"][:, :, 1 : max_lag + 1])
    else:
        p_values = np.clip(results["p_matrix"][:, :, 1 : max_lag + 1], 1e-300, 1.0)
        source_target_lag = -np.log10(p_values)

    scores = np.moveaxis(source_target_lag, 2, 0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=300.0, neginf=0.0)
    for node in range(scores.shape[1]):
        scores[:, node, node] = 0.0
    return scores


def pcmciplus_scores(
    x: np.ndarray,
    max_lag: int,
    pc_alpha: float = 0.05,
    alpha_level: float | None = None,
    score: str = "statistic",
) -> np.ndarray:
    """Score lagged edges with Tigramite PCMCI+ using ParCorr tests.

    PCMCI+ extends PCMCI with a full skeleton phase over contemporaneous
    adjacencies plus collider/rule orientation, so it discovers the lagged
    causal graph under weaker assumptions. The returned ``graph`` encodes
    orientations; edge scores here are derived from the ``val_matrix`` /
    ``p_matrix`` MCI test results exactly like :func:`pcmci_parcorr_scores`,
    so both PCMCI variants stay comparable on the recovery metrics.

    Tigramite returns matrices indexed as ``source, target, tau``. This
    function converts them to the repository convention
    ``lag, source, target``, drops the contemporaneous ``tau = 0`` slice
    (only lagged edges are evaluated), and zeroes self-edges.

    ``alpha_level`` is forwarded to ``run_pcmciplus`` only when the installed
    Tigramite supports it (it was added after 5.2.10); otherwise ``pc_alpha``
    alone controls all steps, matching the installed behavior.
    """

    try:
        import inspect

        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except ImportError as exc:
        raise ImportError(
            "PCMCI+ requires tigramite; install it with "
            "`python -m pip install tigramite`."
        ) from exc

    if score not in {"statistic", "p_value"}:
        raise ValueError("score must be 'statistic' or 'p_value'")

    dataframe = pp.DataFrame(_zscore(x))
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=0,
    )
    run_kwargs: dict[str, object] = {
        "tau_min": 1,
        "tau_max": max_lag,
        "pc_alpha": pc_alpha,
    }
    if alpha_level is not None and "alpha_level" in inspect.signature(
        PCMCI.run_pcmciplus
    ).parameters:
        run_kwargs["alpha_level"] = alpha_level
    results = pcmci.run_pcmciplus(**run_kwargs)

    if score == "statistic":
        source_target_lag = np.abs(results["val_matrix"][:, :, 1 : max_lag + 1])
    else:
        p_values = np.clip(results["p_matrix"][:, :, 1 : max_lag + 1], 1e-300, 1.0)
        source_target_lag = -np.log10(p_values)

    scores = np.moveaxis(source_target_lag, 2, 0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=300.0, neginf=0.0)
    for node in range(scores.shape[1]):
        scores[:, node, node] = 0.0
    return scores


def rpcmci_scores(
    x: np.ndarray,
    max_lag: int,
    num_regimes: int = 2,
    max_transitions: int = 10,
    switch_thres: float = 0.05,
    num_iterations: int = 20,
    max_anneal: int = 10,
    pc_alpha: float = 0.05,
    alpha_level: float = 0.01,
    score: str = "statistic",
    seed: int = 0,
    n_jobs: int = -1,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Score lagged edges with Tigramite Regime-PCMCI (RPCMCI).

    RPCMCI (Saggioro et al., Chaos 30, 113115, 2020) alternates between an
    optimization-based regime-assignment step (a MIP over the one-hot regime
    matrix ``gamma`` with a per-regime cap of ``max_transitions`` switches)
    and per-regime PCMCI causal discovery, so it recovers regime-dependent
    lagged graphs together with the hidden regime sequence. It is the
    regime-aware counterpart of :func:`pcmci_parcorr_scores` and the direct
    competitor baseline for switching ground-truth synthetics.

    ``pc_alpha`` / ``alpha_level`` default to the exact settings used by
    :func:`pcmci_parcorr_scores` / :func:`pcmciplus_scores` (``run_pcmci``
    defaults) so all Tigramite-based baselines are compared fairly.

    The per-regime PCMCI ``val_matrix`` / ``p_matrix`` results are aggregated
    into a single score tensor: the maximum over regimes of ``|val_matrix|``
    (``score="statistic"``, matching the other wrappers) or the minimum over
    regimes of the p-matrix mapped to ``-log10(p)`` (``score="p_value"``).
    Tigramite indexes matrices as ``source, target, tau``; this function
    converts to the repository convention ``lag, source, target``, drops the
    contemporaneous ``tau = 0`` slice, zeroes self-edges, and cleans NaNs
    exactly like :func:`pcmciplus_scores`, keeping AUC comparisons
    apples-to-apples.

    When ``return_info`` is true, a diagnostics dict is returned alongside
    the scores: the hard regime assignment (argmax over the returned
    ``gamma`` rows, length ``T``), the ``gamma`` probabilities transposed to
    ``(T, num_regimes)``, the number of annealings that converged without
    error, and the fixed compute budget (``num_iterations``, ``max_anneal``,
    ``max_transitions``). ``effective_offset`` equals ``max_lag``: RPCMCI
    predictions only exist from ``tau_max`` onward, so the first ``max_lag``
    gamma columns are unconstrained (all-zero residuals) and are excluded
    from regime-alignment metrics.

    Raises ``RuntimeError`` when no annealing converges (``run_rpcmci``
    returns ``None``), so runners can record a flagged NaN row.
    """

    try:
        from sklearn.linear_model import LinearRegression
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.rpcmci import RPCMCI
    except ImportError as exc:
        raise ImportError(
            "RPCMCI requires tigramite and ortools; install them with "
            "`python -m pip install tigramite ortools`."
        ) from exc

    if score not in {"statistic", "p_value"}:
        raise ValueError("score must be 'statistic' or 'p_value'")
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (timesteps, n_nodes), got {x.shape}")
    timesteps, _ = x.shape
    if timesteps <= 3 * max_lag + 1:
        raise ValueError("Need more timesteps than 3 * max_lag + 1 for RPCMCI fits")
    if num_regimes < 2:
        raise ValueError("num_regimes must be at least 2")

    dataframe = pp.DataFrame(_zscore(x))
    rpcmci = RPCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(significance="analytic"),
        prediction_model=LinearRegression(),
        seed=seed,
        verbosity=-1,
    )
    result = rpcmci.run_rpcmci(
        num_regimes=num_regimes,
        max_transitions=max_transitions,
        switch_thres=switch_thres,
        num_iterations=num_iterations,
        max_anneal=max_anneal,
        tau_min=1,
        tau_max=max_lag,
        pc_alpha=pc_alpha,
        alpha_level=alpha_level,
        n_jobs=n_jobs,
    )
    if result is None:
        raise RuntimeError(
            "RPCMCI failed: no annealing converged "
            f"(num_regimes={num_regimes}, max_transitions={max_transitions})."
        )

    causal_results = result["causal_results"]
    regime_keys = sorted(causal_results)
    if score == "statistic":
        stacked = np.stack(
            [np.abs(causal_results[key]["val_matrix"]) for key in regime_keys], axis=0
        )
        source_target_lag = stacked.max(axis=0)[:, :, 1 : max_lag + 1]
    else:
        stacked = np.stack(
            [causal_results[key]["p_matrix"] for key in regime_keys], axis=0
        )
        p_values = np.clip(stacked.min(axis=0)[:, :, 1 : max_lag + 1], 1e-300, 1.0)
        source_target_lag = -np.log10(p_values)

    scores = np.moveaxis(source_target_lag, 2, 0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=300.0, neginf=0.0)
    for node in range(scores.shape[1]):
        scores[:, node, node] = 0.0

    if not return_info:
        return scores

    gamma = np.asarray(result["regimes"], dtype=float)
    info: dict[str, object] = {
        "regime_probabilities": gamma.T,
        "regime_sequence": gamma.argmax(axis=0),
        "effective_offset": max_lag,
        "error_free_annealings": int(result.get("error_free_annealings", 0)),
        "num_iterations": num_iterations,
        "max_anneal": max_anneal,
        "max_transitions": max_transitions,
    }
    return scores, info


def _msvar_target_exog(x: np.ndarray, max_lag: int, target_idx: int) -> np.ndarray:
    """Build the switching-exog design for one MSVAR target equation.

    Columns are ordered lag-major: for each lag ``1..max_lag``, the lagged
    values of every non-target node in increasing node order, aligned so row
    ``t`` (of the trimmed series ``x[max_lag:]``) holds ``x(t - lag)``.
    """

    _, n_nodes = x.shape
    others = [idx for idx in range(n_nodes) if idx != target_idx]
    blocks = [
        x[max_lag - lag : x.shape[0] - lag][:, others] for lag in range(1, max_lag + 1)
    ]
    return np.concatenate(blocks, axis=1)


def _fit_msvar_target(
    x: np.ndarray,
    max_lag: int,
    target_idx: int,
    n_regimes: int,
    switching_variance: bool,
    maxiter: int,
    retry_maxiter: int,
    search_reps: int,
    seed: int = 0,
):
    """Fit one Markov-switching AR equation for ``target_idx``.

    The target's own lags enter through the switching AR part while lagged
    values of every other node enter as switching exogenous regressors, so
    each regime carries its own full lagged coefficient row. Returns the
    statsmodels result object, or ``None`` when both fit attempts fail.
    """

    import inspect
    import warnings

    from statsmodels.tsa.regime_switching.markov_autoregression import (
        MarkovAutoregression,
    )

    endog = x[max_lag:, target_idx]
    exog = _msvar_target_exog(x, max_lag, target_idx)
    model = MarkovAutoregression(
        endog,
        k_regimes=n_regimes,
        order=max_lag,
        exog=exog,
        switching_ar=True,
        switching_exog=True,
        switching_variance=switching_variance,
        trend="c",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = model.fit(disp=0, maxiter=maxiter)
        except Exception:
            result = None
        if result is not None and not bool(result.mle_retvals.get("converged", False)):
            # statsmodels >=0.15 accepts a local RNG; older releases use the
            # legacy NumPy RNG. Support both without consuming caller state.
            retry_kwargs = {
                "disp": 0,
                "maxiter": retry_maxiter,
                "em_iter": 20,
                "search_reps": search_reps,
                "search_iter": 10,
            }
            local_rng = "rng" in inspect.signature(model.fit).parameters
            random_state = None if local_rng else np.random.get_state()
            try:
                if local_rng:
                    retry_kwargs["rng"] = seed
                else:
                    np.random.seed(seed)
                result = model.fit(**retry_kwargs)
            except Exception:
                result = None
            finally:
                if random_state is not None:
                    np.random.set_state(random_state)
    return result


def _align_regime_probabilities(
    smoothed: list[np.ndarray], n_regimes: int
) -> np.ndarray:
    """Average smoothed regime probabilities after label alignment.

    Each target equation is fit independently, so its regime labels carry
    their own arbitrary permutation; averaging the raw probabilities would
    cancel the signal. Every equation's columns are permuted to maximize
    argmax-label agreement with the first (reference) equation before the
    probabilities are averaged.
    """

    import itertools

    reference_labels = smoothed[0].argmax(axis=1)
    aligned = [smoothed[0]]
    for probs in smoothed[1:]:
        labels = probs.argmax(axis=1)
        best_permutation: tuple[int, ...] | None = None
        best_agreement = -1.0
        for permutation in itertools.permutations(range(n_regimes)):
            remapped = np.asarray([permutation[label] for label in labels])
            agreement = float(np.mean(remapped == reference_labels))
            if agreement > best_agreement:
                best_agreement = agreement
                best_permutation = permutation
        # labels map old column -> new regime; invert to reorder columns.
        inverse = np.argsort(np.asarray(best_permutation))
        aligned.append(probs[:, inverse])
    return np.mean(np.stack(aligned, axis=0), axis=0)


def msvar_scores(
    x: np.ndarray,
    max_lag: int,
    n_regimes: int = 2,
    switching_variance: bool = True,
    maxiter: int = 200,
    retry_maxiter: int = 500,
    search_reps: int = 3,
    return_info: bool = False,
    seed: int = 0,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Score lagged edges with an equation-wise Markov-switching VAR.

    statsmodels ships a univariate ``MarkovAutoregression``, so the VAR is
    fit equation by equation: for each target node the model has a switching
    intercept, switching own-lag AR terms, switching coefficients on the
    lagged values of every other node, and (optionally) switching variance.
    The edge score for ``(lag, source, target)`` is the maximum absolute
    coefficient across regimes, which credits edges that are active in any
    single regime — the regime-aware analogue of the plain VAR baselines.

    Self-edges are zeroed. When ``return_info`` is true, a diagnostics dict
    is returned alongside the scores with per-target convergence flags, the
    number of failed target equations (their columns stay zero), the
    smoothed regime probabilities averaged across fitted targets, and the
    offset that aligns those probabilities with the input timeline. ``seed``
    controls randomized retry starts, with a separate stream per target and
    restoration of the caller's NumPy RNG state. With statsmodels versions
    before 0.15, run fits sequentially or in separate processes, rather than
    threads that share NumPy's global RNG.
    """

    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (timesteps, n_nodes), got {x.shape}")
    timesteps, n_nodes = x.shape
    if timesteps <= 3 * max_lag + 1:
        raise ValueError("Need more timesteps than 3 * max_lag + 1 for MSVAR fits")
    if n_regimes < 2:
        raise ValueError("n_regimes must be at least 2")

    z = _zscore(x)
    scores = np.zeros((max_lag, n_nodes, n_nodes), dtype=float)
    converged: list[bool] = []
    failed_targets: list[int] = []
    smoothed: list[np.ndarray] = []

    for target_idx in range(n_nodes):
        result = _fit_msvar_target(
            z,
            max_lag=max_lag,
            target_idx=target_idx,
            n_regimes=n_regimes,
            switching_variance=switching_variance,
            maxiter=maxiter,
            retry_maxiter=retry_maxiter,
            search_reps=search_reps,
            seed=int(np.random.SeedSequence([seed, target_idx]).generate_state(1)[0]),
        )
        if result is None:
            converged.append(False)
            failed_targets.append(target_idx)
            continue

        params = np.asarray(result.params, dtype=float)
        names = list(result.model.param_names)
        values = {name: value for name, value in zip(names, params)}
        others = [idx for idx in range(n_nodes) if idx != target_idx]
        for lag in range(1, max_lag + 1):
            for position, source_idx in enumerate(others):
                column = (lag - 1) * len(others) + position + 1
                regime_coefs = [
                    abs(values.get(f"x{column}[{regime}]", np.nan))
                    for regime in range(n_regimes)
                ]
                regime_coefs = [c for c in regime_coefs if np.isfinite(c)]
                if regime_coefs:
                    scores[lag - 1, source_idx, target_idx] = max(regime_coefs)

        converged.append(bool(result.mle_retvals.get("converged", False)))
        probs = np.asarray(result.smoothed_marginal_probabilities, dtype=float)
        if probs.ndim == 2 and probs.shape[1] == n_regimes:
            smoothed.append(probs)

    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    for node in range(n_nodes):
        scores[:, node, node] = 0.0

    if not return_info:
        return scores

    regime_probabilities = None
    regime_sequence = None
    if smoothed:
        regime_probabilities = _align_regime_probabilities(smoothed, n_regimes)
        regime_sequence = regime_probabilities.argmax(axis=1)
    info: dict[str, object] = {
        "converged": converged,
        "failed_targets": failed_targets,
        "regime_probabilities": regime_probabilities,
        "regime_sequence": regime_sequence,
        # The trimmed endog starts at t = max_lag and the filter drops the
        # first `order` rows, so probabilities align with t >= 2 * max_lag.
        "effective_offset": 2 * max_lag,
        "seed": seed,
    }
    return scores, info


def sliding_window_bounds(
    timesteps: int,
    max_lag: int,
    window_size: int | None = None,
    stride: int | None = None,
) -> list[tuple[int, int]]:
    """Return deterministic sliding-window bounds for local temporal scoring."""

    min_window = max(4 * max_lag + 20, max_lag + 2)
    if window_size is None:
        window_size = max(min_window, min(timesteps // 2, 300))
    if stride is None:
        stride = max(1, window_size // 2)
    if window_size <= max_lag + 1:
        raise ValueError("window_size must be greater than max_lag + 1")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if timesteps < window_size:
        return [(0, timesteps)]

    bounds: list[tuple[int, int]] = []
    start = 0
    while start + window_size <= timesteps:
        bounds.append((start, start + window_size))
        start += stride

    final = (timesteps - window_size, timesteps)
    if bounds[-1] != final:
        bounds.append(final)
    return bounds


def _aggregate_window_scores(window_scores: list[np.ndarray], aggregate: str) -> np.ndarray:
    stacked = np.stack(window_scores, axis=0)
    if aggregate == "max":
        return stacked.max(axis=0)
    if aggregate == "mean":
        return stacked.mean(axis=0)
    if aggregate == "p90":
        return np.quantile(stacked, 0.90, axis=0)
    raise ValueError(f"Unknown aggregate mode: {aggregate}")


def dynamic_ridge_var_scores(
    x: np.ndarray,
    max_lag: int,
    alpha: float = 1.0,
    window_size: int | None = None,
    stride: int | None = None,
    aggregate: str = "max",
) -> np.ndarray:
    """Score edges by aggregating local ridge VAR fits over sliding windows."""

    timesteps, _ = x.shape
    scores = [
        ridge_var_scores(x[start:end], max_lag=max_lag, alpha=alpha)
        for start, end in sliding_window_bounds(timesteps, max_lag, window_size, stride)
    ]
    return _aggregate_window_scores(scores, aggregate)


def dynamic_lasso_var_scores(
    x: np.ndarray,
    max_lag: int,
    alpha: float = 0.005,
    window_size: int | None = None,
    stride: int | None = None,
    aggregate: str = "max",
) -> np.ndarray:
    """Score edges by aggregating local Lasso VAR fits over sliding windows."""

    timesteps, _ = x.shape
    scores = [
        lasso_var_scores(x[start:end], max_lag=max_lag, alpha=alpha)
        for start, end in sliding_window_bounds(timesteps, max_lag, window_size, stride)
    ]
    return _aggregate_window_scores(scores, aggregate)


def dynamic_granger_f_scores(
    x: np.ndarray,
    max_lag: int,
    window_size: int | None = None,
    stride: int | None = None,
    aggregate: str = "max",
) -> np.ndarray:
    """Score edges by aggregating local Granger F-tests over sliding windows."""

    timesteps, _ = x.shape
    scores = [
        granger_f_scores(x[start:end], max_lag=max_lag)
        for start, end in sliding_window_bounds(timesteps, max_lag, window_size, stride)
    ]
    return _aggregate_window_scores(scores, aggregate)
