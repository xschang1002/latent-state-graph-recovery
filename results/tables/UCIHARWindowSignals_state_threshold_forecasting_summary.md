# UCIHARWindowSignals State-Threshold Forecasting Sweep

This sweep tests whether relaxing the STCG-v1 auto-state acceptance threshold converts external mode-label alignment into downstream forecasting gains.
Improvements smaller than `0.001` RMSE are treated as numerical-scale changes.

## State Threshold Curve

| Setting | K | Pass Rate | KMeans Improvement | Temporal Coherence | Inferred States | ARI | Aligned Acc. | RMSE (Best STCG) | Delta vs DynVAR |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| auto_k2_min_0.00 | 2 | 1.00 | 0.067 | 0.000 | `{"2": 5}` | 0.086 +/- 0.006 | 0.353 +/- 0.008 | 0.678990 | -0.000210 |
| forced_k2 | 2 | 1.00 | 0.067 | 0.000 | `{"2": 5}` | 0.086 +/- 0.006 | 0.353 +/- 0.008 | 0.678990 | -0.000210 |
| auto_k2_min_0.04 | 2 | 1.00 | 0.067 | 0.000 | `{"2": 5}` | 0.086 +/- 0.006 | 0.353 +/- 0.008 | 0.678990 | -0.000210 |
| auto_k2_min_0.08 | 2 | 0.00 | 0.067 | 0.000 | `{"1": 5}` | 0.000 +/- 0.000 | 0.267 +/- 0.000 | 0.678670 | 0.000000 |
| auto_k2_min_0.12 | 2 | 0.00 | 0.067 | 0.000 | `{"1": 5}` | 0.000 +/- 0.000 | 0.267 +/- 0.000 | 0.678670 | 0.000000 |
| auto_k2_min_0.16 | 2 | 0.00 | 0.067 | 0.000 | `{"1": 5}` | 0.000 +/- 0.000 | 0.267 +/- 0.000 | 0.678670 | 0.000000 |
| auto_k3_min_0.00 | 3 | 1.00 | 0.085 | 0.059 | `{"3": 5}` | 0.090 +/- 0.007 | 0.360 +/- 0.013 | 0.678971 | -0.000228 |
| forced_k3 | 3 | 1.00 | 0.085 | 0.059 | `{"3": 5}` | 0.090 +/- 0.007 | 0.360 +/- 0.013 | 0.678971 | -0.000228 |
| auto_k3_min_0.04 | 3 | 1.00 | 0.085 | 0.059 | `{"3": 5}` | 0.090 +/- 0.007 | 0.360 +/- 0.013 | 0.678971 | -0.000228 |
| auto_k3_min_0.08 | 3 | 1.00 | 0.085 | 0.059 | `{"3": 5}` | 0.090 +/- 0.007 | 0.360 +/- 0.013 | 0.678971 | -0.000228 |
| auto_k3_min_0.12 | 3 | 0.00 | 0.085 | 0.059 | `{"1": 5}` | 0.000 +/- 0.000 | 0.267 +/- 0.000 | 0.678670 | 0.000000 |
| auto_k3_min_0.16 | 3 | 0.00 | 0.085 | 0.059 | `{"1": 5}` | 0.000 +/- 0.000 | 0.267 +/- 0.000 | 0.678670 | 0.000000 |
| auto_k4_min_0.00 | 4 | 1.00 | 0.098 | 0.029 | `{"4": 5}` | 0.087 +/- 0.010 | 0.366 +/- 0.015 | 0.679076 | 0.000405 |
| forced_k4 | 4 | 1.00 | 0.098 | 0.029 | `{"4": 5}` | 0.087 +/- 0.010 | 0.366 +/- 0.015 | 0.679076 | 0.000405 |
| auto_k4_min_0.04 | 4 | 1.00 | 0.098 | 0.029 | `{"4": 5}` | 0.087 +/- 0.010 | 0.366 +/- 0.015 | 0.679076 | 0.000405 |
| auto_k4_min_0.08 | 4 | 1.00 | 0.098 | 0.029 | `{"4": 5}` | 0.087 +/- 0.010 | 0.366 +/- 0.015 | 0.679076 | 0.000405 |
| auto_k4_min_0.12 | 4 | 0.00 | 0.098 | 0.029 | `{"1": 5}` | 0.000 +/- 0.000 | 0.267 +/- 0.000 | 0.678670 | 0.000000 |
| auto_k4_min_0.16 | 4 | 0.00 | 0.098 | 0.029 | `{"1": 5}` | 0.000 +/- 0.000 | 0.267 +/- 0.000 | 0.678670 | 0.000000 |

## Best Forecasting Rows

| Method | Setting | RMSE | Alpha | Min Weight | Power | Delta vs Best Dense | Delta vs DynVAR | Meaningful vs DynVAR |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Dense Ridge | auto_k2_min_0.00 | 0.679655 | 100.000 | n/a | n/a | 0.000000 | n/a | no |
| DynVAR-Lasso Weighted Ridge | auto_k2_min_0.00 | 0.678670 | 30.000 | 0.01 | 4.00 | -0.000985 | 0.000000 | no |
| STCG-v1 Weighted Ridge | auto_k2_min_0.08 | 0.678670 | 30.000 | 0.01 | 4.00 | -0.000985 | 0.000000 | no |

## Decision

- Any non-collapsed STCG setting tested: yes.
- Best STCG setting meaningfully beats DynVAR-Lasso weighted ridge: no.
- Best STCG setting: `auto_k2_min_0.08` with RMSE `0.678670` and ARI `0.000`.
