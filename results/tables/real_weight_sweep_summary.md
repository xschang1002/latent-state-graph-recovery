# Real Weighted-Prior Sweep Summary

Lower RMSE is better. Weighted ridge keeps all lagged features and uses graph scores only as target-specific diagonal ridge-prior weights.
Improvements smaller than `0.001` RMSE are treated as numerical-scale rather than meaningful gains.

## Best Dense Ridge Baseline

| Dataset | Persistence RMSE | Best Dense RMSE | Dense Alpha |
|---|---:|---:|---:|
| AReMActivities | 1.128005 | 0.909116 | 100.000 |
| BeijingPM25 | 0.266148 | 0.233290 | 0.100 |
| ETTh1 | 0.439275 | 0.408458 | 30.000 |
| ETTh2 | 0.245591 | 0.227127 | 0.100 |
| ETTm1 | 0.253205 | 0.242435 | 3.000 |
| ETTm2 | 0.156106 | 0.149406 | 3.000 |
| HydraulicSystems | 0.167408 | 0.155234 | 0.100 |
| OccupancySensors | 0.045817 | 0.045936 | 3.000 |
| UCIHARWindowSignals | 0.851518 | 0.679655 | 100.000 |

## Best Weighted Priors

| Dataset | Method | RMSE | Alpha | Min Weight | Power | Delta vs Same Dense | Delta vs Best Dense | Raw Beats Best | Meaningful Beat |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| AReMActivities | DynVAR-Lasso Weighted Ridge | 0.906191 | 100.000 | 0.01 | 4.00 | -0.002924 | -0.002924 | yes | yes |
| AReMActivities | STCG-v1 Weighted Ridge | 0.905759 | 100.000 | 0.01 | 4.00 | -0.003356 | -0.003356 | yes | yes |
| BeijingPM25 | DynVAR-Lasso Weighted Ridge | 0.233282 | 1.000 | 0.01 | 4.00 | -0.000019 | -0.000008 | yes | no |
| BeijingPM25 | STCG-v1 Weighted Ridge | 0.233282 | 1.000 | 0.01 | 4.00 | -0.000019 | -0.000008 | yes | no |
| ETTh1 | DynVAR-Lasso Weighted Ridge | 0.408033 | 100.000 | 0.01 | 4.00 | -0.000753 | -0.000425 | yes | no |
| ETTh1 | STCG-v1 Weighted Ridge | 0.407319 | 100.000 | 0.01 | 4.00 | -0.001467 | -0.001139 | yes | yes |
| ETTh2 | DynVAR-Lasso Weighted Ridge | 0.226941 | 3.000 | 0.03 | 4.00 | -0.000219 | -0.000186 | yes | no |
| ETTh2 | STCG-v1 Weighted Ridge | 0.226941 | 3.000 | 0.03 | 4.00 | -0.000219 | -0.000186 | yes | no |
| ETTm1 | DynVAR-Lasso Weighted Ridge | 0.242435 | 3.000 | 0.60 | 4.00 | -0.000000 | -0.000000 | yes | no |
| ETTm1 | STCG-v1 Weighted Ridge | 0.242435 | 3.000 | 0.60 | 4.00 | -0.000000 | -0.000000 | yes | no |
| ETTm2 | DynVAR-Lasso Weighted Ridge | 0.149387 | 1.000 | 0.01 | 4.00 | -0.000020 | -0.000019 | yes | no |
| ETTm2 | STCG-v1 Weighted Ridge | 0.149387 | 1.000 | 0.01 | 4.00 | -0.000020 | -0.000019 | yes | no |
| HydraulicSystems | DynVAR-Lasso Weighted Ridge | 0.155237 | 0.100 | 0.60 | 0.50 | 0.000003 | 0.000003 | no | no |
| HydraulicSystems | STCG-v1 Weighted Ridge | 0.155237 | 0.100 | 0.60 | 0.50 | 0.000003 | 0.000003 | no | no |
| OccupancySensors | DynVAR-Lasso Weighted Ridge | 0.045931 | 3.000 | 0.01 | 2.00 | -0.000005 | -0.000005 | yes | no |
| OccupancySensors | STCG-v1 Weighted Ridge | 0.045931 | 3.000 | 0.01 | 2.00 | -0.000005 | -0.000005 | yes | no |
| UCIHARWindowSignals | DynVAR-Lasso Weighted Ridge | 0.678984 | 100.000 | 0.03 | 4.00 | -0.000671 | -0.000671 | yes | no |
| UCIHARWindowSignals | STCG-v1 Weighted Ridge | 0.678984 | 100.000 | 0.03 | 4.00 | -0.000671 | -0.000671 | yes | no |

## Decision

- Best STCG weighted priors raw-beat best tuned Dense Ridge: yes.
- Best STCG weighted priors meaningfully beat best tuned Dense Ridge: yes.
- Best STCG weighted priors meaningfully beat same-alpha Dense Ridge: yes.
