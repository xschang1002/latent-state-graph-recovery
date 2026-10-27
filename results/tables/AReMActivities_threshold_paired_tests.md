# AReMActivities Threshold Forecasting — Paired Seed Tests

Machine-readable companion: `AReMActivities_threshold_paired_tests.csv` (per-seed RMSEs and deltas).

Scope: STCG-v1 Weighted Ridge vs DynVAR-Lasso Weighted Ridge one-step RMSE on AReMActivities,
setting `auto_k4_min_0.00`, ridge alpha `100.0`, graph min weight `0.01`, weight power `4.0`,
five seeds (0-4). Source rows: `AReMActivities_state_threshold_forecasting_detail.csv`
(regenerated 2026-09-09 with leakage-free per-segment interpolation; per-seed values identical
to the 2026-05-14 audit in `docs/audit/arem_state_threshold_forecasting_2026-05-14.md`).

## Per-seed paired deltas (STCG minus DynVAR)

| Seed | STCG-v1 RMSE | DynVAR RMSE | Delta |
|---:|---:|---:|---:|
| 0 | 0.901711 | 0.906191 | -0.004480 |
| 1 | 0.905970 | 0.906191 | -0.000221 |
| 2 | 0.900868 | 0.906191 | -0.005323 |
| 3 | 0.905693 | 0.906191 | -0.000498 |
| 4 | 0.906203 | 0.906191 | +0.000011 |

## Statistics

| Statistic | Value |
|---|---:|
| Mean paired delta (RMSE) | -0.002102 |
| Std of paired deltas (ddof=1) | 0.002579 |
| Seeds improving over DynVAR-Lasso | 4 / 5 |
| Paired t-test vs zero (two-sided) | t = -1.8225, p = 0.1425 |
| Wilcoxon signed-rank (one-sided, less) | W = 1.0000, p = 0.0625 |

Verdict: the tuned gain is meaningful by the repo's `0.001` RMSE threshold (mean delta
`-0.002102`) but not statistically confirmed over five seeds (paired t `p=0.142`,
one-sided Wilcoxon `p=0.0625`). Non-confirmatory evidence only.
