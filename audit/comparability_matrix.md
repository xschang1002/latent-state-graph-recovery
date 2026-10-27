# Baseline comparability matrix

This audit records whether each candidate baseline can be scored on the same
estimand as the two-stage estimator, and why a method is included or excluded.
The rule is: a method enters the main comparison only if it emits a lag-resolved
edge score over the same variables; otherwise it is reported as a different
estimand, or excluded with the reason below.

| Method | Input | Lag-resolved output | Regime output | Needs K | Status |
|---|---|---|---|---|---|
| STCG-v1 | single series | yes, $(L,N,N)$ | yes, per-window labels | yes (Track I) | included |
| Window max | single series | yes | no | no | included (floor) |
| PELT + VAR-Lasso | single series | yes, per segment | segments, not recurring states | no | included |
| Regime-PCMCI (Saggioro et al., 2020) | single series | yes | yes | yes (regime count) | included on the frozen panel; timing reused |
| CASTOR (Rahmani and Frossard, 2025) | single series | no: instantaneous DAG over stacked copies | yes | number of regimes via window length | **excluded**, see below |
| PCMCI / PCMCI+ (Runge et al., 2019, 2020) | single series | yes | no | no | excluded here; reported in the archived panel |
| SPACETIME (Mameche et al., 2025) | single series + environment labels | yes | segmentation by MDL | no | not run (time); same estimand in principle |
| VARDetect (Bai et al., 2021) | single series | segment-level VAR | change points | no | not run (R toolchain) |
| SDCI (Balsells-Rodas et al., 2025) | single series | instantaneous | latent state | no | not run; different estimand |

## Why CASTOR is excluded from the main table

1. **Different estimand.** CASTOR learns an instantaneous DAG among the
   variables, optionally with lag-stacked copies. Its graph object is a
   $pN\times pN$ matrix of contemporaneous relations over stacked copies, not a
   per-lag $(L,N,N)$ score tensor. Extracting lagged edges from it requires an
   adapter that decides which cross-copy blocks count as lags, and the
   within-copy block is constrained to be acyclic, which our generator does not
   satisfy by construction.
2. **Different regime model.** `run_linear` sets `N_regime = m // window`, i.e.
   contiguous equal-length blocks. Our generator produces alternating,
   recurring states of unequal length, and the estimator's state count is
   controlled by `window` rather than supplied.
3. **Environment conflict.** The official requirements pin `numpy<1.25`,
   `scipy==1.10.1`, and `torch<2.5`; the project environment runs
   `numpy 2.5.3`, `scipy 1.17.1`, `torch 2.14.0+cpu`. Reproducing CASTOR
   faithfully needs a separate environment, and its optimizer is the same
   dynotears solver that our archived panel already contains as DYNOTEARS.

The honest conclusion is that a CASTOR comparison would be a comparison of two
estimands, not of two estimators. We therefore keep the joint-method comparison
on Regime-PCMCI, which emits a lag-resolved score, and we do not report CASTOR
numbers that would invite an apples-to-oranges reading.

## Why PELT is the right low-cost comparator

PELT reuses the same window Lasso pass and differs only in the partitioning and
the re-estimation rule. It therefore isolates the contribution of clustering,
which is the question of the paper. Its cost is within 6% of the two-stage
estimator on the controlled grid, so any accuracy difference is not a budget
artefact.

## Timing boundary

All times are end-to-end single-worker wall clock on the same machine, with the
window pass included for every method that needs it. Regime-PCMCI's
$5.6\times10^{3}$ ratio is taken from the archived matched-timing run on the
switching panel, not re-measured here.
