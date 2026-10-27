"""Check the two analytical claims used in the manuscript.

Claim 1 (assignment-error threshold).  With two equal-size states, coefficient
gap ``delta``, per-window estimate variance ``sigma2``, ``M`` windows and a
symmetric label-error rate ``eta``:

    MSE_group = eta^2 * delta^2 + 2 * sigma2 / M
    MSE_pool  = delta^2 / 4 + sigma2 / M
    MSE_group < MSE_pool  <=>  (1/4 - eta^2) delta^2 > sigma2 / M

The script evaluates both formulas and a Monte-Carlo version of the same
experiment on a grid of ``delta``, ``sigma2/M`` and ``eta``.

Claim 2 (max-selection bias).  For two independent standard-normal null scores,
E[max] = 1/sqrt(pi) ~ 0.564 sigma.  A toy edge-ranking experiment shows that
max aggregation hurts AUC when the edge is shared by both states and helps when
the edge is state-specific; both directions are reported, not just the
favourable one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "theory"


def auc_1d(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank AUC for a flat score vector (no self-edge masking)."""

    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    positives = int(y.sum())
    negatives = int((~y).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, s.size + 1)
    unique, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    if unique.size < s.size:
        sums = np.zeros(unique.size)
        np.add.at(sums, inverse, ranks)
        ranks = sums[inverse] / counts[inverse]
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def analytic_grid() -> pd.DataFrame:
    rows = []
    for delta in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        for sigma2_over_m in (0.0025, 0.01, 0.04):
            for eta in np.linspace(0.0, 0.5, 11):
                mse_group = eta**2 * delta**2 + 2 * sigma2_over_m
                mse_pool = delta**2 / 4 + sigma2_over_m
                threshold = np.sqrt(max(0.0, 0.25 - sigma2_over_m / max(delta**2, 1e-12)))
                rows.append(
                    {
                        "delta": delta,
                        "sigma2_over_M": sigma2_over_m,
                        "eta": float(eta),
                        "mse_group": mse_group,
                        "mse_pool": mse_pool,
                        "advantage": mse_pool - mse_group,
                        "predicted_max_eta": float(min(threshold, 0.5)),
                    }
                )
    return pd.DataFrame(rows)


def monte_carlo(reps: int = 400, m_windows: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for delta in (0.1, 0.35, 0.75):
        for sigma in (0.1, 0.3):
            for eta in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                group_errors, pool_errors = [], []
                for _ in range(reps):
                    theta = np.array([-delta / 2, delta / 2])
                    truth = np.repeat([0, 1], m_windows // 2)
                    noisy = theta[truth] + rng.normal(0.0, sigma, size=truth.size)
                    labels = truth.copy()
                    # Flip exactly round(eta * M) labels so that group sizes
                    # stay fixed and the formula is compared on its own terms.
                    n_flip = int(round(eta * truth.size))
                    if n_flip:
                        flip = rng.choice(truth.size, size=n_flip, replace=False)
                        labels[flip] = 1 - labels[flip]
                    group_means = np.array(
                        [noisy[labels == k].mean() if np.any(labels == k) else 0.0 for k in (0, 1)]
                    )
                    # Align groups to states by which is closer to the truth
                    # only for scoring the estimator error, not for fitting.
                    pooled = noisy.mean()
                    group_errors.append(float(np.mean((group_means - theta) ** 2)))
                    pool_errors.append(float(np.mean((pooled - theta) ** 2)))
                sigma2_over_m = sigma**2 / m_windows
                rows.append(
                    {
                        "delta": delta,
                        "sigma": sigma,
                        "sigma2_over_M": sigma2_over_m,
                        "eta": eta,
                        "empirical_mse_group": float(np.mean(group_errors)),
                        "empirical_mse_pool": float(np.mean(pool_errors)),
                        "mse_group": eta**2 * delta**2 + 2 * sigma2_over_m,
                        "mse_pool": delta**2 / 4 + sigma2_over_m,
                    }
                )
    return pd.DataFrame(rows)


def max_bias_toy(
    n_edges: int = 2000,
    prevalence: float = 0.05,
    signal: float = 1.5,
    reps: int = 400,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for mode in ("shared", "state_specific"):
        mean_auc, max_auc = [], []
        for _ in range(reps):
            truth = (rng.random(n_edges) < prevalence).astype(int)
            noise_a = rng.normal(0.0, 1.0, n_edges)
            noise_b = rng.normal(0.0, 1.0, n_edges)
            if mode == "shared":
                score_a = signal * truth + noise_a
                score_b = signal * truth + noise_b
            else:
                score_a = signal * truth + noise_a
                score_b = noise_b
            mean_auc.append(auc_1d(truth, (score_a + score_b) / 2))
            max_auc.append(auc_1d(truth, np.maximum(score_a, score_b)))
        rows.append(
            {
                "mode": mode,
                "n_edges": n_edges,
                "prevalence": prevalence,
                "mean_auc": float(np.mean(mean_auc)),
                "max_auc": float(np.mean(max_auc)),
                "difference": float(np.mean(max_auc) - np.mean(mean_auc)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    analytic = analytic_grid()
    analytic.to_csv(out / "theory_mse_grid.csv", index=False)

    empirical = monte_carlo()
    empirical.to_csv(out / "theory_mse_simulation.csv", index=False)

    bias = max_bias_toy()
    bias.to_csv(out / "theory_max_bias.csv", index=False)

    merged = empirical.copy()
    merged["group_formula_error"] = (
        merged["empirical_mse_group"] - merged["mse_group"]
    ).abs()
    merged["pool_formula_error"] = (merged["empirical_mse_pool"] - merged["mse_pool"]).abs()
    merged.to_csv(out / "theory_mse_comparison.csv", index=False)

    print(
        merged[["delta", "sigma2_over_M", "eta", "empirical_mse_group", "mse_group",
                "empirical_mse_pool", "mse_pool"]]
        .head(12)
        .round(4)
        .to_string(index=False)
    )
    print(f"max |group formula error| {merged['group_formula_error'].max():.4f}")
    print(f"max |pool formula error| {merged['pool_formula_error'].max():.4f}")
    print(bias.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
