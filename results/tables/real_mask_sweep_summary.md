# Real Hard-Mask Sweep Summary

Lower RMSE is better. `Nonself Dense Ridge` keeps all non-self lagged features and drops self-lags, matching the feature universe available to graph masks at `graph_fraction=1.0`.

## Baseline Ceiling

| Dataset | Method | RMSE | Delta vs Dense | Delta vs Persistence |
|---|---|---:|---:|---:|
| ETTh1 | Persistence | 0.439 | 0.031 | 0.000 |
| ETTh1 | Dense Ridge | 0.409 | 0.000 | -0.031 |
| ETTh1 | Nonself Dense Ridge | 0.778 | 0.369 | 0.339 |
| ETTh2 | Persistence | 0.246 | 0.018 | 0.000 |
| ETTh2 | Dense Ridge | 0.227 | 0.000 | -0.018 |
| ETTh2 | Nonself Dense Ridge | 0.743 | 0.516 | 0.497 |

## Best Hard Graph Masks

| Dataset | Method | RMSE | Graph Fraction | Lasso Alpha | Window | Beats Dense | Beats Persistence |
|---|---|---:|---:|---:|---:|---|---|
| ETTh1 | DynVAR-Lasso Graph Ridge | 0.727 | 0.25 | 0.003 | 300 | no | no |
| ETTh2 | DynVAR-Lasso Graph Ridge | 0.743 | 0.75 | 0.010 | 600 | no | no |
| ETTh1 | STCG-v1 Graph Ridge | 0.740 | 0.25 | 0.001 | 600 | no | no |
| ETTh2 | STCG-v1 Graph Ridge | 0.741 | 0.25 | 0.005 | 600 | no | no |

## Decision

- Best STCG hard masks beat Dense Ridge: no.
- Best STCG hard masks beat Persistence: no.
- Current hard graph masks are not sufficient for ETT-style dense forecasting.
