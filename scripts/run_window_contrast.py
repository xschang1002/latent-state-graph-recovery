"""Window link: does the operating point sit at the mixing null?

Section ``sec:window-average`` claims the manuscript's ``h=300`` "sits exactly at
such a null". That null is the ``D=150, h=2D`` fast-switching regime of the
*exogenous-driver toy*, not the main generator. The main generator uses
``switch_period=300`` (each state lasts ``D=300`` steps, full cycle ``P=600``),
so a window of ``h=300`` is ``h=D``: windows alternate pure / 50-50 mixed and the
state contrast is *not* erased, only halved on the mixed windows.

This script verifies that distinction with the real pipeline. It freezes one
real coefficient tensor and state schedule, re-simulates the innovation noise
many times, and separates the population window mixture ``beta_w*`` (Eq. (5))
from the estimation noise. It reports the effective regression rows (297, not
300: ``make_lagged_design`` drops the first ``max_lag`` rows), the theoretical
signal (variance of ``beta_w*`` across windows over changing edges) and the
noise (per-replica estimation scatter), under both ``D=300`` (main) and ``D=150``
(fast switch, the toy null).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stcg.baselines import _zscore, make_lagged_design, sliding_window_bounds  # noqa: E402
from stcg.mechanism import MechanismConfig, _build_coefficients, _regime_schedule, generate_mechanism  # noqa: E402

OUT = ROOT / "results" / "revision2"
CONFIG = ROOT / "configs" / "aggregation_controls.yaml"


def simulate_fixed(coefficients: np.ndarray, regimes: np.ndarray, config: MechanismConfig, noise_seed: int) -> np.ndarray:
    """Re-simulate the VAR with fixed coefficients and schedule, fresh noise."""
    rng = np.random.default_rng(noise_seed)
    total = config.timesteps + config.burn_in
    n_nodes = config.n_nodes
    x = rng.normal(0.0, config.noise_scale, size=(total, n_nodes))
    for t in range(config.max_lag, total):
        state = int(regimes[t])
        value = np.zeros(n_nodes)
        for lag in range(1, config.max_lag + 1):
            value += x[t - lag] @ coefficients[state, lag - 1]
        x[t] = value + rng.normal(0.0, config.noise_scale, size=n_nodes)
    return x[config.burn_in :].copy()


def window_ols_coefficients(x: np.ndarray, max_lag: int, bounds) -> np.ndarray:
    """Per-window OLS coefficients (z-scored), shape (n_windows, max_lag, n, n)."""
    rows = []
    for start, end in bounds:
        design, target = make_lagged_design(_zscore(x[start:end]), max_lag)
        design = _zscore(design)
        target = _zscore(target)
        gram = design.T @ design
        coeff = np.linalg.solve(gram + 1e-8 * np.eye(gram.shape[0]), design.T @ target)
        rows.append(coeff.reshape(max_lag, target.shape[1], target.shape[1]))
    return np.stack(rows, axis=0)


def window_fractions(regimes: np.ndarray, bounds, n_states: int, max_lag: int) -> np.ndarray:
    """Effective per-window state fractions q_{w,k}, counting only rows that
    enter the regression (the first ``max_lag`` rows are dropped by the design)."""
    fracs = np.zeros((len(bounds), n_states))
    for w, (start, end) in enumerate(bounds):
        # The lagged design uses rows [start+max_lag, end).
        seg = regimes[start + max_lag : end]
        for k in range(n_states):
            fracs[w, k] = float(np.mean(seg == k))
    return fracs


def state_design_moment(coefficients: np.ndarray, state: int, config: MechanismConfig, length: int = 30000) -> np.ndarray:
    """Empirical E[z z^T] of the z-scored lagged design under a single state."""
    rng = np.random.default_rng(1000 + state)
    n_nodes = config.n_nodes
    x = rng.normal(0.0, config.noise_scale, size=(length, n_nodes))
    for t in range(config.max_lag, length):
        value = np.zeros(n_nodes)
        for lag in range(1, config.max_lag + 1):
            value += x[t - lag] @ coefficients[state, lag - 1]
        x[t] = value + rng.normal(0.0, config.noise_scale, size=n_nodes)
    design, _ = make_lagged_design(_zscore(x), config.max_lag)
    design = _zscore(design)
    return design.T @ design / design.shape[0]


def theoretical_beta_w(coefficients: np.ndarray, fracs: np.ndarray, config: MechanismConfig) -> np.ndarray:
    """Eq. (5): beta_w* = (sum q Q)^-1 (sum q Q beta), per window."""
    n_states, max_lag, n_nodes, _ = coefficients.shape
    qs = [state_design_moment(coefficients, k, config) for k in range(n_states)]
    out = []
    for w in range(fracs.shape[0]):
        gram = np.zeros((max_lag * n_nodes, max_lag * n_nodes))
        rhs = np.zeros((max_lag * n_nodes, n_nodes))
        for k in range(n_states):
            beta = coefficients[k].reshape(max_lag * n_nodes, n_nodes)
            gram += fracs[w, k] * qs[k]
            rhs += fracs[w, k] * qs[k] @ beta
        beta_w = np.linalg.solve(gram + 1e-8 * np.eye(gram.shape[0]), rhs)
        out.append(beta_w.reshape(max_lag, n_nodes, n_nodes))
    return np.stack(out, axis=0)


def changing_edge_mask(state_support: np.ndarray) -> np.ndarray:
    """Boolean mask over (lag, source, target) for edges that differ across states."""
    n_states = state_support.shape[0]
    delta = np.zeros(state_support.shape[1:], dtype=bool)
    for first in range(n_states):
        for second in range(first + 1, n_states):
            delta |= state_support[first] != state_support[second]
    for node in range(delta.shape[1]):
        delta[:, node, node] = False
    return delta


def run_schedule(coefficients: np.ndarray, state_support: np.ndarray, regimes_full: np.ndarray,
                 config: MechanismConfig, n_replicas: int, n_states: int) -> dict:
    regimes = regimes_full[config.burn_in :]
    bounds = sliding_window_bounds(config.timesteps, config.max_lag, 300, 150)
    fracs = window_fractions(regimes, bounds, n_states, config.max_lag)
    beta_w_star = theoretical_beta_w(coefficients, fracs, config)
    mask = changing_edge_mask(state_support)

    # Estimation replicas.
    replica_betas = np.stack(
        [window_ols_coefficients(simulate_fixed(coefficients, regimes_full, config, 2000 + m),
                                 config.max_lag, bounds)
         for m in range(n_replicas)],
        axis=0,
    )  # (n_replicas, n_windows, max_lag, n, n)

    mean_beta = replica_betas.mean(axis=0)
    noise = replica_betas.std(axis=0)

    # Theoretical signal: std of beta_w* across windows, over changing edges.
    theory_signal = float(np.std(beta_w_star[:, mask], axis=0).mean())
    # Estimated signal: std of mean_beta across windows, over changing edges.
    est_signal = float(np.std(mean_beta[:, mask], axis=0).mean())
    # Estimation noise: mean per-edge std across replicas (averaged over windows).
    est_noise = float(noise[:, mask].mean())

    # How far the estimated mean tracks the theoretical mixture (per edge, corr).
    theory_flat = beta_w_star[:, mask]
    est_flat = mean_beta[:, mask]
    tracking = float(np.corrcoef(theory_flat.reshape(-1), est_flat.reshape(-1))[0, 1])

    return {
        "n_windows": len(bounds),
        "effective_rows_per_window": bounds[0][1] - bounds[0][0] - config.max_lag,
        "min_state_frac_min": float(fracs.min(axis=1).min()),
        "min_state_frac_max": float(fracs.min(axis=1).max()),
        "min_state_frac_mean": float(fracs.min(axis=1).mean()),
        "theory_signal": theory_signal,
        "est_signal": est_signal,
        "est_noise": est_noise,
        "snr_estimated": est_signal / max(est_noise, 1e-12),
        "snr_theory": theory_signal / max(est_noise, 1e-12),
        "tracking_corr": tracking,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--n-replicas", type=int, default=50)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    setting = config["settings"]["switching_core"]
    base_cfg = MechanismConfig(**setting, seed=0)

    # Freeze one real coefficient tensor and support.
    frozen = generate_mechanism(base_cfg)
    coefficients = frozen.coefficients
    state_support = frozen.state_support
    n_states = coefficients.shape[0]

    rows = []
    for switch_period in (300, 150):
        cfg_setting = dict(setting)
        cfg_setting["switch_period"] = switch_period
        cfg = MechanismConfig(**cfg_setting, seed=0)
        regimes_full, _ = _regime_schedule(cfg, cfg.timesteps + cfg.burn_in)
        result = run_schedule(coefficients, state_support, regimes_full, cfg, args.n_replicas, n_states)
        result["switch_period"] = switch_period
        result["state_duration_D"] = switch_period
        result["full_cycle_P"] = 2 * switch_period
        rows.append(result)

    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "window_contrast.csv", index=False)
    print(frame.round(4).to_string(index=False))

    print("\nReading:")
    print("  D=300 (main): h=D, windows alternate pure / mixed -> state contrast HALVED on mixed windows, kept on pure windows.")
    print("  D=150 (toy null): h=2D=P, every window spans a full cycle -> contrast erased.")
    print("  theory_signal = std of beta_w* across windows (changing edges) = size of the population contrast.")
    print("  est_noise     = per-replica estimation scatter (changing edges).")
    print("  tracking_corr = corr between the estimated mean and the theoretical mixture per edge/window.")


if __name__ == "__main__":
    main()
