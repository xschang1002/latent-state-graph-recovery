# Frozen specialist stress and matched timing protocol

Frozen before any new-seed recovery results on 2026-09-13. This is a prospective
extension of the original robustness and switching-difficulty axes. Historical
tables remain unchanged. The separate machine-readable protocol records source
hashes, dependency versions, execution settings, and this document's hash.

## Questions and fixed population

1. Does the state-aggregation improvement extend across all original stress axes
   when the strongest regime specialist and a neural-Granger baseline are included?
2. What are the end-to-end fitting costs on identical generated inputs, with a
   common preprocessing boundary and one computational thread per method?

All methods use the 20 new generator seeds **100--119**. These do not overlap the
original 0--19 evaluation seeds or cMLP's 0--4 tuning seeds. Hyperparameters are
not selected or changed using these new outcomes. The grid is frozen in full:

| Axis | Values | Other settings |
|---|---|---|
| Observation noise ratio | 0, 0.05, 0.10, 0.20 | T=1000, edge probability=0.18 |
| Sequence length T | 300, 600, 1000, 1500 | observation noise=0, edge probability=0.18 |
| Edge probability | 0.10, 0.18, 0.30, 0.45 | T=1000, observation noise=0 |
| Switch period | 150, 300, 600 | T=1200, observation noise=0, edge probability=0.18 |

The original generator is unchanged: 8 nodes, lag 3, innovation noise 0.10,
100-step burn-in, switching enabled. Robustness uses the generator's original
midpoint switch. Difficulty uses its alternating two-regime periodic switch.
Observation noise follows the original ratio times the series standard deviation
with RNG seed `dataset_seed + 60000`. Robustness window size is
`min(300,max(120,T//2))`, stride half that size. Difficulty uses h=300, stride=150.
The three robustness baseline cells have identical input/settings and are
deduplicated computationally, then displayed under all three original axes.
Thus 15 reported cells require 13 unique settings x 20 seeds = 260 input series.

## Methods and unchanged fit budgets

- STCG-v1: supplied candidate K=2, original auto gate, Lasso alpha=0.005,
  within-state mean and cross-state max, no contrast; state RNG is dataset seed
  +50000 on robustness and +90000 on difficulty, matching each original runner.
- DynVAR-Lasso: identical windows/stride/alpha and cross-window max.
- Global VAR-Lasso: alpha=0.005, one full-series fit.
- Regime-PCMCI: K=2, 20 iterations x 10 annealings, max_transitions=10,
  switch_thres=0.05, pc_alpha=0.05, alpha_level=0.01, lag range 1--3,
  statistic scores, RNG=dataset seed. `n_jobs=1` changes only computational
  parallelism; it does not reduce optimization budgets. Even the fastest
  switching grid has no more than 7 true transitions, within the fixed cap.
- cMLP: hidden=8, penalty=0.15, learning rate=0.05, 300 epochs, tanh,
  SGD+prox with momentum=0.9, no weight decay, RNG=dataset seed. These are the
  already-fixed main-paper hyperparameters, with no new tuning.

Every algorithm receives the same copied raw array. Its existing standardization,
lag-design construction, target fits, regime inference, and score aggregation
are all inside the measured call. No method receives precomputed windows,
regimes, design matrices, or normalization parameters from another method.
Graph truth is accessed only after fitting to compute metrics. Data generation,
library/process startup, evaluation metrics, and result serialization are outside
the measured fit boundary. Libraries are imported before measuring each call;
this is warm-library, fresh-model timing, not cold-process latency.

## Execution, independence, failures, and caches

The first new-seed base-setting execution is a resource calibration only. Its
results remain part of the frozen grid; performance does not determine inclusion.
No seed or axis is dropped based on results. At most four independent workers
run initially. Each fit has OMP/BLAS/MKL/NumExpr/PyTorch thread limit 1, RPCMCI
`n_jobs=1`, and loky CPU count 1. All environment limits and both wall and process
CPU fit times are recorded. Resource-calibration wall times only determine
scheduling, never algorithm settings. The full grid is not shortened if slow.

Worker jobs contain one (setting, seed, method) and run in separate processes.
Their deterministic shuffled dispatch order is fixed by a scheduling-only seed.
Results do not depend on scheduling; no process shares method RNG state.
Workers checkpoint atomic JSON records and a matching input SHA-256. Resume
validates the frozen protocol and source fingerprint, skips completed records,
and never silently mixes results from changed implementations or parameters.
Worker logs are retained. No historical partial-result cache is used.

An unchanged per-fit safety ceiling of 3600 seconds applies to every method.
Timeouts, solver failures, invalid scores, and process crashes are retained with
their reason and elapsed dispatch time; they are never imputed as chance AUC or
counted as a performance victory. There is no outcome-dependent retry. A retry
would require a documented protocol amendment retaining the original failure.

## Two distinct runtime reports

Stress-grid timings are collected under the explicitly recorded concurrent
worker load and are descriptive. They must not be promoted to an exclusive
latency claim. After other heavy experiments finish, a separate **timing** phase
runs the base T=1000/density=0.18/no-observation-noise setting on all 20 new seeds
with all five methods and one worker, using the same fit boundary and budgets.
Its dedicated cache cannot reuse stress timing observations. Method dispatch
order is deterministically shuffled. This final phase is scheduled explicitly
once the parent task can provide a quiet machine; it is not launched in parallel
with this stress grid. Recorded process CPU times additionally expose scheduling
or background-load differences.

## Analysis fixed before seeing outcomes

All five original recovery metrics are retained. Primary outcome is lagged Edge
AUC. For each unique setting we compare STCG against each of the four opponents
on the same successfully completed seeds, report the planned/attempted/successful
counts and failure rates, paired mean difference, sample SD, win/tie/loss counts,
two-sided paired t and Wilcoxon tests, and a 95% percentile paired bootstrap CI
(10000 resamples; analysis RNG seed 20260913). SHD signs are reversed so positive
means STCG improves. Comparisons use common complete cases across all five
methods for that setting; per-method means over all successful fits are also
shown separately. Missingness is disclosed, not hidden by unequal sample sizes.

The 13 unique setting comparisons against Regime-PCMCI form the primary
multiple-testing family (Holm correction). A second reported Holm family covers
all 52 Edge-AUC comparisons. Other metrics are exploratory. Repeated baseline
axis aliases do not count as extra independent tests. No equivalence or
non-inferiority margin is introduced. A nonsignificant difference does not
establish equivalence. Practical interpretation must use the effect magnitude
and interval, compare it with the original +0.0143 Edge-AUC effect, and include
failures and scope; there is no post-hoc minimum-benefit cutoff or winner-based
selection. All settings, including reversals and null results, are reported.

The timing phase reports paired geometric mean wall/CPU cost ratios with paired
bootstrap intervals and all fit counts. Accuracy remains the same unchanged
estimator; timing is not used to retune any method.

## Reproduction

Use `.venv/Scripts/python.exe scripts/run_revision_specialist_stress.py --freeze`
to record the immutable protocol. Run `--calibrate` once for the base setting
and seed 100; then `--run --workers 4` for the full stress grid. The run command
resumes automatically. `--aggregate` regenerates only `revision_specialist_*`
tables from validated checkpoints. After the machine is quiet, run
`--run --phase timing --workers 1 --exclusive-timing` and aggregate again.
The tool prints progress and records controller PID, active child PIDs, planned
jobs, completed jobs, protocol hash, and source hash in its status JSON.
