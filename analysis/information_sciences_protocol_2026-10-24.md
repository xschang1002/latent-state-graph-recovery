# Fixed scope for the Information Sciences revision

Recorded before execution of the new studies on 2026-10-24 (Asia/Shanghai).
The executable protocols also freeze source/configuration hashes. The old
exploratory grid and prior confirmation blocks remain separate inputs.

## Study A: independent paired duration confirmation

100 independent mechanism/noise repeats, numeric labels 50000--50099, each at
D=300 and 600 with N=8, L=3, T=1200, h=300, stride=150, burn-in=100. Each pair
shares coefficients, support, initial history, full innovations and clustering
seed. Across repeats all these stochastic inputs are redrawn. Addressable
SeedSequence domains separate mechanism, initialization and innovations. The
original coefficient/support construction and Lasso preprocessing are reused.
The periodic schedule starts at state zero at full-time origin zero; the first
100 observations are removed, retaining the old observed phase convention.

Primary family: Lasso native-hard minus raw at D=300; the same at D=600;
Gamma=Delta(600)-Delta(300) within the same repeat; and native-hard minus exact
size-matched random at D=600. Use two-sided t tests of repeat-level contrasts
against zero and Holm over these four slots. Missing tests retain their family
slot. Ordinary 95% percentile bootstrap intervals use 10,000 repeat resamples
with seed 20261024 and have pointwise, not simultaneous, coverage.

Controls: raw, native hard, forced hard (bypass only the gate), exhaustive
group-size-matched random, oracle window labels and static window mean. Score
each random partition before averaging AUROCs within repeat; partitions and
overlapping windows are never new replicates. Save changing-edge temporal,
pooled time-indexed and union AUROC, all exogenous inputs, scores, labels,
true schedule, gates, edge denominators and J/C/JC diagnostics.

OLS is a prespecified auxiliary scorer on exactly the same paired datasets,
with the same preprocessing and aggregation. Require full column rank under
the fixed numpy.linalg.lstsq(rcond=None) rule, retain rank diagnostics and
failures, and do not replace failure with ridge or a different solver. Forced
and OLS comparisons are reported separately from the four primary tests.

## Study B: phase and irregular-schedule robustness

50 new independent repeats, numeric labels 60000--60049 in a separate random
namespace. Within each repeat six cells share coefficients/support/history/
innovations: Dmean=300,600 crossed with fixed phase, uniform random phase and
irregular alternating durations. T=3600, h=300, stride=150, M=23; Lasso only.
Fixed phase follows Study A. Random phase q is uniform on integers 0..2D-1
before burn-in. Irregular lengths are independent inclusive integer-uniform
draws from ceil(.75D)..floor(1.25D), starting state zero at full-time origin;
first/last observed runs can be truncated. Schedule RNG is domain-separated.

Prespecified secondary families, separate from A: six forced-hard-minus-raw
comparisons with two-sided t tests and Holm6; three paired duration Gamma
contrasts, one per schedule, with Holm3. Pointwise bootstrap intervals use the
same fixed 10,000 draws/seed. Native output, gate rates, actual execution,
state/centre switch counts, centre-state counts, window-purity distributions
and J/C/JC are reported as diagnostics, without causal attribution to the two
factors or claims of a universal h/D threshold.

## Fixed stopping and failure rules

No pilot on these new repeats, optional stopping, sample-size extension,
replacement mechanism, new hyperparameter selection or favorable-cell
selection is allowed. Retain every planned record and input even on failure.
Undefined temporal targets remain missing, with reasons and denominators;
failure is never imputed as temporal AUROC .5. Static fallback legitimately
has .5 only when its changing-edge target is evaluable. Gamma uses complete
paired repeats; all reports include planned/evaluable/failed counts.

Any result, including disappearance of the exploratory positive gain, is a
completed outcome. The paper's claim and title follow the evidence. A general
benchmark, new algorithm, nonlinear expansion and additional ESS draws are
outside this stage. Public-repository deposition is not performed: the author
requested an upload-ready archive and will supply its persistent link later.
