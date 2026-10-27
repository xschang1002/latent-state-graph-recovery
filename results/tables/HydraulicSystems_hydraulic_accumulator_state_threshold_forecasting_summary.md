# HydraulicSystems_hydraulic_accumulator State-Threshold Forecasting Sweep

This sweep tests whether relaxing the STCG-v1 auto-state acceptance threshold converts external mode-label alignment into downstream forecasting gains.
Improvements smaller than `0.001` RMSE are treated as numerical-scale changes.

## State Threshold Curve

| Setting | K | Pass Rate | KMeans Improvement | Temporal Coherence | Inferred States | ARI | Aligned Acc. | RMSE (Best STCG) | Delta vs DynVAR |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| auto_k2_min_0.00 | 2 | 1.00 | 0.048 | 0.906 | `{"2": 1}` | 0.045 +/- 0.000 | 0.436 +/- 0.000 | 0.155652 | 0.000114 |
| forced_k2 | 2 | 1.00 | 0.048 | 0.906 | `{"2": 1}` | 0.045 +/- 0.000 | 0.436 +/- 0.000 | 0.155652 | 0.000114 |
| auto_k3_min_0.00 | 3 | 1.00 | 0.071 | 0.908 | `{"3": 1}` | 0.151 +/- 0.000 | 0.525 +/- 0.000 | 0.155613 | 0.000076 |
| forced_k3 | 3 | 1.00 | 0.071 | 0.908 | `{"3": 1}` | 0.151 +/- 0.000 | 0.525 +/- 0.000 | 0.155613 | 0.000076 |
| auto_k4_min_0.00 | 4 | 1.00 | 0.089 | 0.906 | `{"4": 1}` | 0.132 +/- 0.000 | 0.491 +/- 0.000 | 0.155585 | 0.000048 |
| forced_k4 | 4 | 1.00 | 0.089 | 0.906 | `{"4": 1}` | 0.132 +/- 0.000 | 0.491 +/- 0.000 | 0.155585 | 0.000048 |

## Best Forecasting Rows

| Method | Setting | RMSE | Alpha | Min Weight | Power | Delta vs Best Dense | Delta vs DynVAR | Meaningful vs DynVAR |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Dense Ridge | auto_k2_min_0.00 | 0.155411 | 1.000 | n/a | n/a | 0.000000 | n/a | no |
| DynVAR-Lasso Weighted Ridge | auto_k2_min_0.00 | 0.155538 | 1.000 | 0.10 | 1.00 | 0.000126 | 0.000000 | no |
| STCG-v1 Weighted Ridge | auto_k4_min_0.00 | 0.155585 | 1.000 | 0.10 | 1.00 | 0.000174 | 0.000048 | no |

## Decision

- Any non-collapsed STCG setting tested: yes.
- Best STCG setting meaningfully beats DynVAR-Lasso weighted ridge: no.
- Best STCG setting: `auto_k4_min_0.00` with RMSE `0.155585` and ARI `0.132`.
