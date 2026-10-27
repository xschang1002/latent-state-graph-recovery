# Toy theory: when does grouping beat pooling?

This note records the derivation behind Section 5 of the manuscript. It is a
two-state, Gaussian, equal-variance model. It is deliberately small enough to be
checked by simulation and it is **not** a risk bound for Lasso or for AUROC.

## Setup

- Two states of equal size, coefficients `theta_1`, `theta_2`, gap
  `Delta = theta_2 - theta_1`.
- `M` windows total, `M/2` per state, independent window estimates with variance
  `sigma^2` and zero bias before label noise.
- Symmetric label error `eta`: a window of state 1 is assigned to group 2 with
  probability `eta`, and vice versa.

## Grouped estimator

Group `k` contains `(1-eta) M/2` windows from state `k` and `eta M/2` from the
other state, so

```
E[theta_hat_1] = (1-eta) theta_1 + eta theta_2
E[theta_hat_2] = (1-eta) theta_2 + eta theta_1
bias_k         = ± eta * Delta
Var(theta_hat_k) = 2 sigma^2 / M
```

Averaging the two groups:

```
MSE_group = eta^2 Delta^2 + 2 sigma^2 / M
```

## Pooled estimator

```
E[theta_hat_pool] = (theta_1 + theta_2) / 2
bias              = ± Delta / 2
Var               = sigma^2 / M
MSE_pool          = Delta^2 / 4 + sigma^2 / M
```

## Threshold

```
MSE_group < MSE_pool
<=> eta^2 Delta^2 + 2 sigma^2/M < Delta^2/4 + sigma^2/M
<=> (1/4 - eta^2) Delta^2 > sigma^2 / M
<=> eta < sqrt(1/4 - sigma^2 / (M Delta^2))
```

Consequences:

1. Grouping needs a real state difference: as `Delta -> 0` the condition fails
   for every `eta`.
2. Grouping tolerates more label error when windows are many or quiet.
3. The tolerable error reaches zero at `Delta^2/4 = sigma^2/M`; beyond that
   point even a perfect partition is worse than pooling, because splitting the
   sample costs more variance than the bias it removes.

## Simulation check

`scripts/run_theory_check.py` flips exactly `round(eta M)` labels so that group
sizes stay fixed, and compares the empirical mean squared error with the
formulas over `Delta in {0.1, 0.35, 0.75}`, `sigma in {0.1, 0.3}`, and
`eta in {0, 0.1, ..., 0.5}` with 400 replications and `M = 40`.

Maximum absolute deviation: `0.0015` for the grouped formula and `0.0003` for
the pooled formula. The residual is Monte-Carlo error in the variance term.

## What is not covered

- Lasso shrinkage: `E[theta_hat] != theta` and the bias is coefficient-specific.
- Overlapping windows: the `M/2` estimates are correlated, so `Var` is larger
  than `2 sigma^2 / M`; the threshold is optimistic in `M`.
- Absolute scores: the pipeline clusters `|beta|`, so `+c` and `-c` are the same
  feature and the derivation's sign is lost.
- Max aggregation: the union score is a maximum over states, which is a
  different functional from the per-state mean analysed here.

The empirical sections of the manuscript therefore use the theory as an
organizing device, not as a guarantee: it predicts *where* grouping can help and
*which* failures are outside its scope, and the experiments confirm both.
