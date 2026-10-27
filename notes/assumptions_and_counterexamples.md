# Assumptions and counterexamples

This note lists the assumptions the two-stage estimator needs, and the
counterexamples that violate them. Each counterexample is implemented as a
negative control in `src/stcg/mechanism.py` and measured in
`results/mechanism/mechanism_grid.csv`.

## Assumptions

A1. **Window purity.** Each window's score belongs to one state graph. Violated
when `h` is comparable to the state duration; measured by `window_purity_mean`.

A2. **Score separability.** Two states with different graphs produce different
window-score vectors. Violated when the graphs differ only in a way the score
discards.

A3. **State recurrence.** States repeat across the series, so K-means has more
than one window per state. Violated by a single changepoint with short segments.

A4. **Correct state count.** `K` is supplied and correct (Track I). Unknown `K`
is out of scope here.

A5. **Stationarity within state.** Coefficients are constant inside a state.
Violated by trends and within-regime drift.

A6. **Graph truth equals support.** The estimand is edge support, not sign,
magnitude, or lag-specific coefficient value.

## Counterexamples

### C1. Sign flip (violates A2)

State 1 has `B[a,b] = +c` at lag 1; state 2 has `-c` at the same lag. The
absolute window score is identical, so the state features are identical and the
partition is arbitrary. Measured: state ARI `0.166`, regime AUPRC `0.466`,
against `0.570` for a genuine switch. The gate still accepts.

### C2. Lag shift (violates A2 at the lag level)

The same pair moves from lag 1 to lag 2. Pair-level support is unchanged but the
lag-specific support changes. Measured: state ARI `0.459`, regime AUPRC `0.553`.
A method evaluated only at the pair level cannot see this at all; the lag-level
evaluation can.

### C3. Same support, different coefficients (violates A2 in the other direction)

Both states share the support but resample coefficient magnitudes. The union
graph is unchanged and the clustering has no support difference to find.
Measured: state ARI `-0.121`, union AUROC `0.852` (higher than the genuine
switch, because sharing edges makes the union easier).

### C4. Variance only (violates A1/A2)

Coefficients are identical; only the innovation variance changes. Window scores
separate slightly, the gate accepts, and state ARI is `-0.116`.

### C5. Mean only (violates A5)

Coefficients are identical; a constant offset is added in state 2. Standardised
window scores are contaminated by the mean shift, union AUROC drops to `0.657`,
and state ARI is `-0.115`. This is the only nuisance control that also damages
union recovery, because z-scoring mixes the offset into the design.

### C6. Trend (violates A5)

A linear trend is added across the whole series. The gate accepts a split at an
arbitrary point, state ARI is `-0.141`, and union AUROC is `0.787`.

### C7. Impure windows (violates A1)

At switch period 150 with `h = 300`, every window straddles a boundary. Purity
is `0.50`, union AUROC `0.742`, regime AUPRC `0.458`. No partitioning algorithm
can recover states from scores that mix two graphs; the fix is a shorter window
or a boundary-aware window, not a better clusterer.

### C8. Rare state (violates A3)

With state proportions 95:5, the rare state occupies about 60 time points, fewer
than one window. State ARI is `0.000` and the gate still accepts.

## What would fix each

| Counterexample | Candidate fix | Status |
|---|---|---|
| C1 sign flip | signed scores or per-state sign features | not implemented |
| C2 lag shift | lag-level state features and lag-level evaluation | partially: lag-level metrics exist |
| C3 same support | coefficient-magnitude features, not support clustering | not implemented |
| C4 variance only | variance-normalised window scores | not implemented |
| C5 mean only | per-window centring before scoring | not implemented |
| C6 trend | detrending inside each window | not implemented |
| C7 impure windows | adaptive window length or boundary-aware scoring | not implemented |
| C8 rare state | hierarchical or changepoint-based states | not implemented |

None of these fixes is claimed in the manuscript. They are the natural next
step, and the negative controls exist so that a future fix can be tested against
them rather than against a favourable setting.
