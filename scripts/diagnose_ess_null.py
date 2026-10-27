"""Replay the fixed ESS null chains with move counts and tail diagnostics.

No new data or fitted estimator is introduced. The default ten chains reproduce
the published 5,000-draw seeds 0--4; two seed-0 chains use tenfold burn-in as a
descriptive sensitivity check. Diagnostics do not establish adequate mixing.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stcg.randomization import directed_degree_null
from stcg.metrics import edge_auc, lagged_correlation_scores
from run_ess_alignment import load_subsystem_series


def autocorrelation(values: np.ndarray) -> np.ndarray:
    """Biased sample ACF, computed by FFT; undefined for constant traces."""
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    if not np.any(centered):
        return np.full(len(values), np.nan)
    size = 1 << (2 * len(values) - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=size)
    covariance = np.fft.irfft(spectrum * spectrum.conj(), n=size)[:len(values)]
    return covariance / covariance[0]


def chain_statistics(values: np.ndarray) -> dict:
    """Initial positive, monotone paired-ACF ESS and 20-batch MCSE.

    ESS is capped at n (no claim of super-efficiency). A constant tail indicator
    has undefined ESS/MCSE, rather than an asserted zero Monte Carlo error.
    These are descriptive single-chain estimates, especially noisy for rare tails.
    """
    values = np.asarray(values, dtype=float)
    acf = autocorrelation(values)
    if not np.isfinite(acf[0]):
        return {key: np.nan for key in ("acf1", "acf10", "ess", "mcse", "batch_mcse")}
    paired_sum = 0.0
    previous = np.inf
    for lag in range(1, len(acf) - 1, 2):
        pair = acf[lag] + acf[lag + 1]
        if pair <= 0:
            break
        previous = min(previous, pair)
        paired_sum += previous
    tau = max(1.0, 1 + 2 * paired_sum)
    effective = len(values) / tau
    batches = min(20, len(values) // 2)
    width = len(values) // batches if batches else 0
    batch_mcse = (values[:batches * width].reshape(batches, width).mean(axis=1)
                  .std(ddof=1) / np.sqrt(batches)) if batches > 1 else np.nan
    return {"acf1": acf[1], "acf10": acf[10] if len(acf) > 10 else np.nan,
            "ess": effective, "mcse": np.sqrt(values.var(ddof=1) / effective),
            "batch_mcse": batch_mcse}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/tables")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    series, adjacency, _ = load_subsystem_series("ffill")
    scores = lagged_correlation_scores(series.to_numpy(float), 3).max(axis=0)
    expert = adjacency.to_numpy().copy()
    np.fill_diagonal(expert, 0)
    mask = ~np.eye(len(expert), dtype=bool)
    ranks = rankdata(scores[mask], method="average")
    reference = pd.read_csv(ROOT / "results/tables/ess_alignment_seed_sensitivity.csv")
    rows, traces = [], []
    schedule = [(seed, 100) for seed in range(5)] + [(0, 1000)]
    for edge_set, support in (("all_edges", expert > 0), ("strong_only", expert == 2)):
        positive = int(support.sum()); negative = int(mask.sum()) - positive
        observed = edge_auc(scores[None], support[None])
        for seed, burn in schedule:
            counts = {}
            nulls = directed_degree_null(support, 5000, seed, burn_in_per_edge=burn,
                                        diagnostics=counts)
            values = (nulls[:, mask].astype(float) @ ranks - positive * (positive + 1) / 2) / (positive * negative)
            tail = (values >= observed).astype(float)
            row = {"binarisation": edge_set, "null_seed": seed, "burn_in_per_edge": burn,
                   "null_draws": len(values), "n_edges": positive, "observed_auroc": observed,
                   "null_auroc_mean": values.mean(), "null_exceedances": int(tail.sum()),
                   "p_raw": (1 + tail.sum()) / (1 + len(values))}
            for phase, moves in counts.items():
                for move, count in moves.items():
                    prefix = phase + "_" + move
                    row.update({prefix + "_" + key: value for key, value in count.items()})
                    row[prefix + "_acceptance"] = count["accepted"] / count["attempted"] if count["attempted"] else np.nan
            for name, data in (("auroc", values), ("tail", tail)):
                row.update({name + "_" + key: value for key, value in chain_statistics(data).items()})
            if burn == 100:
                expected = reference[(reference.method == "lag_corr") & (reference.imputation == "ffill")
                                     & (reference.binarisation == edge_set) & (reference.null_seed == seed)].iloc[0]
                assert row["null_exceedances"] == expected.null_exceedances
                assert np.isclose(row["null_auroc_mean"], expected.null_auroc_mean, atol=1e-15, rtol=0)
                assert np.isclose(observed, expected.auroc, atol=1e-15, rtol=0)
            rows.append(row)
            traces.append(pd.DataFrame({"binarisation": edge_set, "null_seed": seed,
                                       "burn_in_per_edge": burn, "draw": np.arange(1, len(values) + 1),
                                       "null_auroc": values, "tail_event": tail.astype(int)}))
            print(f"{edge_set} seed={seed} burn={burn}: b={row['null_exceedances']}, "
                  f"tail ESS={row['tail_ess']:.0f}, MCSE={row['tail_mcse']:.6f}", flush=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "ess_null_diagnostics.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(args.output_dir / "ess_null_traces.csv", index=False)


if __name__ == "__main__":
    main()
