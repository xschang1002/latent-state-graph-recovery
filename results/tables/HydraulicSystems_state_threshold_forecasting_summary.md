# HydraulicSystems State-Threshold Forecasting Sweep

This sweep tests whether relaxing the STCG-v1 auto-state acceptance threshold converts external mode-label alignment into downstream forecasting gains.
Improvements smaller than `0.001` RMSE are treated as numerical-scale changes.

## State Threshold Curve

| Setting | K | Pass Rate | KMeans Improvement | Temporal Coherence | Inferred States | ARI | Aligned Acc. | RMSE (Best STCG) | Delta vs DynVAR |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| auto_k2_min_0.00 | 2 | 1.00 | 0.048 | 0.914 | `{"2": 5}` | -0.001 +/- 0.000 | 0.507 +/- 0.003 | 0.155653 | 0.000115 |
| forced_k2 | 2 | 1.00 | 0.048 | 0.914 | `{"2": 5}` | -0.001 +/- 0.000 | 0.507 +/- 0.003 | 0.155653 | 0.000115 |
| auto_k2_min_0.08 | 2 | 1.00 | 0.048 | 0.914 | `{"2": 5}` | -0.001 +/- 0.000 | 0.507 +/- 0.003 | 0.155653 | 0.000115 |
| auto_k2_min_0.12 | 2 | 1.00 | 0.048 | 0.914 | `{"2": 5}` | -0.001 +/- 0.000 | 0.507 +/- 0.003 | 0.155653 | 0.000115 |
| auto_k2_min_0.16 | 2 | 1.00 | 0.048 | 0.914 | `{"2": 5}` | -0.001 +/- 0.000 | 0.507 +/- 0.003 | 0.155653 | 0.000115 |
| auto_k3_min_0.00 | 3 | 1.00 | 0.071 | 0.903 | `{"3": 5}` | 0.055 +/- 0.001 | 0.688 +/- 0.001 | 0.155611 | 0.000073 |
| forced_k3 | 3 | 1.00 | 0.071 | 0.903 | `{"3": 5}` | 0.055 +/- 0.001 | 0.688 +/- 0.001 | 0.155611 | 0.000073 |
| auto_k3_min_0.08 | 3 | 1.00 | 0.071 | 0.903 | `{"3": 5}` | 0.055 +/- 0.001 | 0.688 +/- 0.001 | 0.155611 | 0.000073 |
| auto_k3_min_0.12 | 3 | 1.00 | 0.071 | 0.903 | `{"3": 5}` | 0.055 +/- 0.001 | 0.688 +/- 0.001 | 0.155611 | 0.000073 |
| auto_k3_min_0.16 | 3 | 1.00 | 0.071 | 0.903 | `{"3": 5}` | 0.055 +/- 0.001 | 0.688 +/- 0.001 | 0.155611 | 0.000073 |

## Best Forecasting Rows

| Method | Setting | RMSE | Alpha | Min Weight | Power | Delta vs Best Dense | Delta vs DynVAR | Meaningful vs DynVAR |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Dense Ridge | auto_k2_min_0.00 | 0.155411 | 1.000 | n/a | n/a | 0.000000 | n/a | no |
| DynVAR-Lasso Weighted Ridge | auto_k2_min_0.00 | 0.155538 | 1.000 | 0.10 | 1.00 | 0.000126 | 0.000000 | no |
| STCG-v1 Weighted Ridge | auto_k3_min_0.00 | 0.155611 | 1.000 | 0.10 | 1.00 | 0.000200 | 0.000073 | no |

## Decision

- Any non-collapsed STCG setting tested: yes.
- Best STCG setting meaningfully beats DynVAR-Lasso weighted ridge: no.
- Best STCG setting: `auto_k3_min_0.00` with RMSE `0.155611` and ARI `0.055`.
