"""DYNOTEARS baseline: NOTEARS-style structure learning for time series.

The optimization core is a clean, minimal port of
``causalnex/structure/dynotears.py`` (causalnex 0.12.2,
github.com/quantumblacklabs/causalnex), which is

    Copyright 2019-2020 QuantumBlack Visual Analytics Limited,
    licensed under the Apache License, Version 2.0
    (http://www.apache.org/licenses/LICENSE-2.0),

and implements

    Pamfil, Sriwattanaworachai, Desai, Pilgerstorfer, Georgatzis, Beaumont,
    Aragam (2020). "DYNOTEARS: Structure Learning from Time-Series Data."
    AISTATS, PMLR 108:1595-1605. https://arxiv.org/abs/2002.00498

The port drops the pandas/``StructureModel`` I/O layer and tabu-edge support,
z-scores the data like the other baselines in this repository, adds an
explicit inner L-BFGS-B iteration budget, and returns a
``(max_lag, source, target)`` score tensor of absolute lagged weights with
self-edges zeroed, matching ``stcg.metrics.recovery_summary``.

The contemporaneous (intra-slice) matrix W is still optimized under the
NOTEARS acyclicity constraint ``h(W) = tr(e^{W . W}) - d = 0``, faithful to
the method; on the purely-lagged synthetic ground truth used here it acts as
a nuisance/DAG regularizer and is not scored.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt

from .baselines import _zscore, make_lagged_design


def _reshape_wa(wa_vec: np.ndarray, d_vars: int, p_orders: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the flat plus/minus weight vector into W (d x d) and A (p*d x d).

    The vector layout is ``[W+ ; W- ; A+_1 ; A-_1 ; ... ; A+_p ; A-_p]`` with
    each block flattened row-major, exactly as in the causalnex reference.
    """

    w_tilde = wa_vec.reshape([2 * (p_orders + 1) * d_vars, d_vars])
    w_plus = w_tilde[:d_vars, :]
    w_minus = w_tilde[d_vars : 2 * d_vars, :]
    w_mat = w_plus - w_minus
    a_plus = (
        w_tilde[2 * d_vars :]
        .reshape(2 * p_orders, d_vars**2)[::2]
        .reshape(d_vars * p_orders, d_vars)
    )
    a_minus = (
        w_tilde[2 * d_vars :]
        .reshape(2 * p_orders, d_vars**2)[1::2]
        .reshape(d_vars * p_orders, d_vars)
    )
    a_mat = a_plus - a_minus
    return w_mat, a_mat


def acyclicity_h(w_mat: np.ndarray) -> float:
    """NOTEARS acyclicity surrogate ``tr(expm(W . W)) - d`` (0 iff DAG)."""

    d_vars = w_mat.shape[0]
    return float(np.trace(slin.expm(w_mat * w_mat)) - d_vars)


def _learn_dynamic_structure(
    X: np.ndarray,
    Xlags: np.ndarray,
    lambda_w: float,
    lambda_a: float,
    max_iter: int,
    h_tol: float,
    lbfgs_maxiter: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Minimize the DYNOTEARS augmented Lagrangian with L-BFGS-B.

    Objective over the contemporaneous W (d x d) and lagged A (p*d x d):

        0.5/n * ||X(I - W) - Xlags A||_F^2
        + lambda_w * |W|_1 + lambda_a * |A|_1
        + 0.5 * rho * h(W)^2 + alpha * h(W),

    where the L1 terms are smoothed through the plus/minus split
    ``W = W+ - W-``, ``A = A+ - A-`` with non-negativity bounds, the diagonal
    of W is pinned to zero (no contemporaneous self-loops), rho is escalated
    by x10 whenever h(W) does not shrink by a factor of 4, and alpha follows
    dual ascent. This mirrors the causalnex reference implementation.
    """

    if X.size == 0:
        raise ValueError("Input data X is empty, cannot learn any structure")
    if Xlags.size == 0:
        raise ValueError("Input data Xlags is empty, cannot learn any structure")
    if X.shape[0] != Xlags.shape[0]:
        raise ValueError("Input data X and Xlags must have the same number of rows")
    if Xlags.shape[1] % X.shape[1] != 0:
        raise ValueError("Number of columns of Xlags must be a multiple of number of columns of X")

    n, d_vars = X.shape
    p_orders = Xlags.shape[1] // d_vars

    # W diagonal pinned to 0 (no intra-slice self-loops); plus/minus parts
    # non-negative. Lagged self-weights stay free, as in the reference.
    bnds_w = 2 * [
        (0, 0) if i == j else (0, None)
        for i in range(d_vars)
        for j in range(d_vars)
    ]
    bnds_a: list[tuple[float, float | None]] = []
    for _ in range(1, p_orders + 1):
        bnds_a.extend(2 * [(0, None) for _ in range(d_vars**2)])
    bnds = bnds_w + bnds_a

    def _h(wa_vec: np.ndarray) -> float:
        w_mat, _ = _reshape_wa(wa_vec, d_vars, p_orders)
        return acyclicity_h(w_mat)

    def _func(wa_vec: np.ndarray) -> float:
        w_mat, a_mat = _reshape_wa(wa_vec, d_vars, p_orders)
        loss = (
            0.5
            / n
            * np.square(
                np.linalg.norm(
                    X.dot(np.eye(d_vars, d_vars) - w_mat) - Xlags.dot(a_mat), "fro"
                )
            )
        )
        h_value = _h(wa_vec)
        l1_penalty = lambda_w * (wa_vec[: 2 * d_vars**2].sum()) + lambda_a * (
            wa_vec[2 * d_vars**2 :].sum()
        )
        return loss + 0.5 * rho * h_value * h_value + alpha * h_value + l1_penalty

    def _grad(wa_vec: np.ndarray) -> np.ndarray:
        w_mat, a_mat = _reshape_wa(wa_vec, d_vars, p_orders)
        e_mat = slin.expm(w_mat * w_mat)
        loss_grad_w = (
            -1.0
            / n
            * (X.T.dot(X.dot(np.eye(d_vars, d_vars) - w_mat) - Xlags.dot(a_mat)))
        )
        obj_grad_w = (
            loss_grad_w
            + (rho * (np.trace(e_mat) - d_vars) + alpha) * e_mat.T * w_mat * 2
        )
        obj_grad_a = (
            -1.0
            / n
            * (Xlags.T.dot(X.dot(np.eye(d_vars, d_vars) - w_mat) - Xlags.dot(a_mat)))
        )
        grad_vec_w = np.append(
            obj_grad_w, -obj_grad_w, axis=0
        ).flatten() + lambda_w * np.ones(2 * d_vars**2)
        grad_vec_a = obj_grad_a.reshape(p_orders, d_vars**2)
        grad_vec_a = np.hstack(
            (grad_vec_a, -grad_vec_a)
        ).flatten() + lambda_a * np.ones(2 * p_orders * d_vars**2)
        return np.append(grad_vec_w, grad_vec_a, axis=0)

    wa_est = np.zeros(2 * (p_orders + 1) * d_vars**2)
    wa_new = np.zeros(2 * (p_orders + 1) * d_vars**2)
    rho, alpha, h_value, h_new = 1.0, 0.0, np.inf, np.inf
    n_lbfgs_solves = 0
    n_lbfgs_iters = 0
    n_outer = 0

    for n_iter in range(max_iter):
        n_outer = n_iter + 1
        while (rho < 1e20) and (h_new > 0.25 * h_value or h_new == np.inf):
            result = sopt.minimize(
                _func,
                wa_est,
                method="L-BFGS-B",
                jac=_grad,
                bounds=bnds,
                options={"maxiter": lbfgs_maxiter},
            )
            wa_new = result.x
            n_lbfgs_solves += 1
            n_lbfgs_iters += int(result.nit)
            h_new = _h(wa_new)
            if h_new > 0.25 * h_value:
                rho *= 10

        wa_est = wa_new
        h_value = h_new
        alpha += rho * h_value
        if h_value <= h_tol:
            break

    w_est, a_est = _reshape_wa(wa_est, d_vars, p_orders)
    info: dict[str, object] = {
        "h_value": float(h_value),
        "converged": bool(h_value <= h_tol),
        "n_outer_iters": n_outer,
        "n_lbfgs_solves": n_lbfgs_solves,
        "n_lbfgs_iters": n_lbfgs_iters,
        "rho_final": float(rho),
    }
    return w_est, a_est, info


def dynotears_scores(
    x: np.ndarray,
    max_lag: int,
    lambda_w: float = 0.05,
    lambda_a: float = 0.05,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    lbfgs_maxiter: int = 1000,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Score lagged edges with DYNOTEARS (Pamfil et al. 2020).

    Fits the joint least-squares + L1 objective over the contemporaneous
    matrix W (under the NOTEARS acyclicity constraint) and the lagged slabs
    A, then scores ``(lag, source, target)`` by the absolute lagged weight
    ``|A_lag[source, target]|`` with self-edges zeroed. ``lambda_w`` /
    ``lambda_a`` default to 0.05 as recommended for small graphs in the
    DYNOTEARS paper. When ``return_info`` is true, a diagnostics dict is
    returned alongside the scores with the final acyclicity value, the
    convergence flag, iteration counts, and the fitted W matrix.
    """

    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (timesteps, n_nodes), got {x.shape}")
    timesteps, n_nodes = x.shape
    if timesteps <= max_lag + 1:
        raise ValueError("Need more timesteps than max_lag + 1")

    design, target = make_lagged_design(_zscore(x), max_lag)
    X = _zscore(target)
    Xlags = _zscore(design)

    w_mat, a_mat, info = _learn_dynamic_structure(
        X,
        Xlags,
        lambda_w=lambda_w,
        lambda_a=lambda_a,
        max_iter=max_iter,
        h_tol=h_tol,
        lbfgs_maxiter=lbfgs_maxiter,
    )

    scores = np.abs(a_mat.reshape(max_lag, n_nodes, n_nodes))
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    for node in range(n_nodes):
        scores[:, node, node] = 0.0

    if not return_info:
        return scores
    info["w_matrix"] = w_mat
    return scores, info
