"""Direct VAR emissions with one shared Markov chain; corrective baseline.

This module deliberately does not wrap statsmodels MarkovAutoregression. For
row t and regime k, the entire response vector has mean design[t] @ beta[k]
and full covariance covariance[k]. Lagged predictors include every node,
including the target, exactly once at each lag 1..L. No autoregressive errors
or separate target-specific latent chains are fitted.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_triangular


@dataclass(frozen=True)
class JointMSVARConfig:
    n_regimes: int = 2
    n_starts: int = 10
    max_iter: int = 200
    tolerance: float = 1e-6
    covariance_floor: float = 1e-6
    probability_floor: float = 1e-8
    lstsq_rcond: float = 1e-12
    coefficient_ridge: float = 0.0
    initial_stay_probability: float = 0.98
    initialization_seed_salt: int = 20260913
    monotonicity_tolerance: float = 1e-8


def lag_design(x: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Return [intercept, lag1/all nodes, ..., lagL/all nodes], response."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or max_lag < 1 or len(x) <= max_lag:
        raise ValueError("Expected finite (T,N) data with T > max_lag >= 1")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite observations")
    design = np.column_stack([np.ones(len(x) - max_lag)] +
        [x[max_lag - lag:len(x) - lag] for lag in range(1, max_lag + 1)])
    return design, x[max_lag:].copy()


def probability_mle(counts: np.ndarray, floor: float) -> np.ndarray:
    """Maximize sum counts*log(p) subject to sum(p)=1 and p>=floor."""
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1 or (counts < 0).any() or not np.isfinite(counts).all():
        raise ValueError("Invalid categorical counts")
    if floor <= 0 or len(counts) * floor >= 1:
        raise ValueError("Invalid probability floor")
    if counts.sum() <= 0:
        return np.full(len(counts), 1. / len(counts))
    free = np.ones(len(counts), dtype=bool)
    result = np.full(len(counts), floor)
    while free.any():
        mass = 1. - floor * (~free).sum()
        free_counts = counts[free]
        proposal = mass * free_counts / free_counts.sum()
        below = proposal < floor
        if not below.any():
            result[free] = proposal
            break
        indices = np.flatnonzero(free)
        free[indices[below]] = False
    return result


def forward_backward(log_emissions: np.ndarray, transition: np.ndarray,
                     initial: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Scaled HMM recursion, returning (loglik, gamma[T,K], xi[T-1,K,K]).

    transition[i,j] is Pr(S_t=j | S_{t-1}=i). Emission logs have one
    multivariate Gaussian value per time/regime, shared by all targets.
    Per-time emission maxima avoid exponentiating large absolute log densities.
    """
    log_emissions = np.asarray(log_emissions, dtype=float)
    transition = np.asarray(transition, dtype=float)
    initial = np.asarray(initial, dtype=float)
    if log_emissions.ndim != 2 or len(log_emissions) == 0:
        raise ValueError("Empty/invalid emissions")
    timesteps, n_regimes = log_emissions.shape
    if transition.shape != (n_regimes, n_regimes) or initial.shape != (n_regimes,):
        raise ValueError("Invalid shared-chain dimensions")
    if (not np.isfinite(log_emissions).all() or not np.isfinite(transition).all()
            or not np.isfinite(initial).all() or (transition <= 0).any()
            or (initial <= 0).any() or not np.allclose(transition.sum(axis=1), 1)
            or not np.isclose(initial.sum(), 1)):
        raise ValueError("Invalid emissions or positive stochastic probabilities")
    shifts = log_emissions.max(axis=1)
    emissions = np.exp(log_emissions - shifts[:, None])
    alpha = np.empty_like(emissions)
    scales = np.empty(timesteps)
    alpha[0] = initial * emissions[0]
    scales[0] = alpha[0].sum()
    alpha[0] /= scales[0]
    for t in range(1, timesteps):
        alpha[t] = (alpha[t - 1] @ transition) * emissions[t]
        scales[t] = alpha[t].sum()
        alpha[t] /= scales[t]
    if (scales <= 0).any() or not np.isfinite(scales).all():
        raise FloatingPointError("Invalid HMM normalization")
    beta = np.ones_like(emissions)
    for t in range(timesteps - 2, -1, -1):
        beta[t] = transition @ (emissions[t + 1] * beta[t + 1]) / scales[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = (alpha[:-1, :, None] * transition[None, :, :] *
          (emissions[1:] * beta[1:])[:, None, :])
    if timesteps > 1:
        xi /= xi.sum(axis=(1, 2), keepdims=True)
    return float(np.sum(np.log(scales) + shifts)), gamma, xi


def emission_log_probabilities(design: np.ndarray, response: np.ndarray,
                               coefficients: np.ndarray,
                               covariance: np.ndarray) -> np.ndarray:
    """Joint, full-covariance Gaussian emission log densities [T,K]."""
    n_obs, n_nodes = response.shape
    values = np.empty((n_obs, len(coefficients)))
    for regime in range(len(coefficients)):
        residual = response - design @ coefficients[regime]
        chol = np.linalg.cholesky(covariance[regime])
        white = solve_triangular(chol, residual.T, lower=True, check_finite=False)
        logdet = 2. * np.log(np.diag(chol)).sum()
        values[:, regime] = -0.5 * (n_nodes * np.log(2. * np.pi) + logdet +
                                   np.square(white).sum(axis=0))
    return values


def weighted_regression(design: np.ndarray, response: np.ndarray,
                        gamma: np.ndarray, covariance_floor: float = 1e-6,
                        lstsq_rcond: float = 1e-12
                        ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Exact multivariate WLS; covariance MLE constrained Sigma >= floor*I.

    Coefficients [K,1+N*L,N] use no ridge penalty. A common full covariance
    within each regime cancels from the WLS normal equations. Spectral clipping
    is the constrained covariance MLE, unlike adding a diagonal ridge afterward.
    """
    if covariance_floor <= 0:
        raise ValueError("Covariance floor must be positive")
    n_regimes = gamma.shape[1]
    n_nodes = response.shape[1]
    coefficients = np.empty((n_regimes, design.shape[1], n_nodes))
    covariance = np.empty((n_regimes, n_nodes, n_nodes))
    ranks = []
    for regime in range(n_regimes):
        weights = gamma[:, regime]
        mass = float(weights.sum())
        if mass <= 1e-10 or (weights < 0).any() or not np.isfinite(weights).all():
            raise FloatingPointError("Regime has no usable posterior mass")
        root = np.sqrt(weights)
        coef, _, rank, _ = np.linalg.lstsq(design * root[:, None],
                                           response * root[:, None], rcond=lstsq_rcond)
        coefficients[regime] = coef
        residual = response - design @ coef
        cov = (residual.T * weights) @ residual / mass
        eigenvalues, eigenvectors = np.linalg.eigh((cov + cov.T) / 2.)
        covariance[regime] = ((eigenvectors * np.maximum(eigenvalues, covariance_floor))
                              @ eigenvectors.T)
        ranks.append(int(rank))
    return coefficients, covariance, ranks


def weighted_m_step(design: np.ndarray, response: np.ndarray,
                    gamma: np.ndarray, xi: np.ndarray,
                    covariance_floor: float = 1e-6,
                    probability_floor: float = 1e-8,
                    lstsq_rcond: float = 1e-12) -> dict:
    coefficients, covariance, ranks = weighted_regression(
        design, response, gamma, covariance_floor, lstsq_rcond)
    transition = np.vstack([probability_mle(row, probability_floor)
                            for row in xi.sum(axis=0)])
    initial = probability_mle(gamma[0], probability_floor)
    return {"coefficients": coefficients, "covariance": covariance,
            "transition": transition, "initial": initial, "design_ranks": ranks}


def scores_from_coefficients(coefficients: np.ndarray, max_lag: int) -> np.ndarray:
    """Direct lag/source/target coefficients, max absolute over regimes."""
    coefficients = np.asarray(coefficients, dtype=float)
    n_nodes = coefficients.shape[-1]
    if coefficients.ndim != 3 or coefficients.shape[1] != 1 + max_lag * n_nodes:
        raise ValueError("Coefficient dimensions do not match direct VAR(L)")
    scores = np.abs(coefficients[:, 1:].reshape(-1, max_lag, n_nodes, n_nodes)).max(axis=0)
    for node in range(n_nodes):
        scores[:, node, node] = 0.
    return scores


def _one_start(design: np.ndarray, response: np.ndarray, seed: int,
               start_index: int, config: JointMSVARConfig) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence(
        [seed, config.initialization_seed_salt, start_index]))
    n_regimes = config.n_regimes
    off_diag = (1. - config.initial_stay_probability) / (n_regimes - 1)
    transition = np.full((n_regimes, n_regimes), off_diag)
    np.fill_diagonal(transition, config.initial_stay_probability)
    initial = np.full(n_regimes, 1. / n_regimes)
    path = np.empty(len(response), dtype=int)
    path[0] = rng.integers(n_regimes)
    for t in range(1, len(path)):
        path[t] = rng.choice(n_regimes, p=transition[path[t - 1]])
    gamma = np.eye(n_regimes)[path]
    coefficients, covariance, ranks = weighted_regression(
        design, response, gamma, config.covariance_floor, config.lstsq_rcond)
    params = {"coefficients": coefficients, "covariance": covariance,
              "transition": transition, "initial": initial, "design_ranks": ranks}
    loglik, gamma, xi = forward_backward(
        emission_log_probabilities(design, response, coefficients, covariance), transition, initial)
    history = [loglik]
    converged = False
    for iteration in range(1, config.max_iter + 1):
        candidate = weighted_m_step(design, response, gamma, xi,
                                   config.covariance_floor, config.probability_floor,
                                   config.lstsq_rcond)
        new_loglik, new_gamma, new_xi = forward_backward(
            emission_log_probabilities(design, response, candidate["coefficients"],
                                      candidate["covariance"]),
            candidate["transition"], candidate["initial"])
        delta = new_loglik - loglik
        if delta < -config.monotonicity_tolerance * (1. + abs(loglik)):
            raise FloatingPointError(f"Observed likelihood decreased: {delta:.12g}")
        history.append(new_loglik)
        params, loglik, gamma, xi = candidate, new_loglik, new_gamma, new_xi
        if abs(delta) <= config.tolerance * (1. + abs(history[-2])):
            converged = True
            break
    return {**params, "gamma": gamma, "loglik": loglik,
            "loglik_history": history, "iterations": iteration,
            "converged": converged, "start_index": start_index,
            "effective_counts": gamma.sum(axis=0).tolist()}


def joint_msvar_scores(x: np.ndarray, max_lag: int, seed: int = 0,
                        config: JointMSVARConfig | None = None
                        ) -> tuple[np.ndarray, dict]:
    """Fit all fixed starts and select the highest finite observed likelihood.

    Max-iteration fits remain available with converged=False; callers must not
    silently count them as converged evidence. An all-failed run raises rather
    than returning a zero graph. No labels, truth, or reference scores are used.
    """
    config = config or JointMSVARConfig()
    if (config.n_regimes < 2 or config.n_starts < 1 or config.max_iter < 1
            or config.tolerance <= 0 or config.coefficient_ridge != 0):
        raise ValueError("Invalid config; this implementation uses unpenalized WLS")
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite matrix")
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z = (x - x.mean(axis=0)) / scale
    design, response = lag_design(z, max_lag)
    if len(response) <= design.shape[1]:
        raise ValueError("Insufficient observations for direct VAR design")
    starts = []
    best = None
    for index in range(config.n_starts):
        try:
            fitted = _one_start(design, response, seed, index, config)
            starts.append({key: fitted[key] for key in (
                "start_index", "loglik", "loglik_history", "iterations", "converged",
                "effective_counts", "design_ranks")})
            starts[-1]["status"] = "converged" if fitted["converged"] else "max_iter"
            if best is None or fitted["loglik"] > best["loglik"]:
                best = fitted
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            starts.append({"start_index": index, "status": "failed", "error": str(exc)})
    if best is None:
        raise RuntimeError(f"All {config.n_starts} starts failed: {starts}")
    scores = scores_from_coefficients(best["coefficients"], max_lag)
    info = {**best, "starts": starts, "effective_offset": max_lag,
            "regime_sequence": best["gamma"].argmax(axis=1),
            "n_successful_starts": sum(row["status"] != "failed" for row in starts),
            "n_converged_starts": sum(row["status"] == "converged" for row in starts),
            "seed": seed, "scale": scale}
    return scores, info
