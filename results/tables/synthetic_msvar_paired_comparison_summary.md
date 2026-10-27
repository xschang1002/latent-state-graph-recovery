# Paired Synthetic Recovery Comparison

Positive improvement means `method_a` is better than `method_b`; for SHD the sign is reversed because lower is better.

| Generator | Metric | N | Method A | Method B | Mean Improvement | 95% bootstrap CI | t-test p | Wilcoxon p | Wins/Ties/Losses |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| var | edge_auc | 20 | STCG-v1 (0.876) | MSVAR (0.743) | 0.1339 | [0.1131, 0.1554] | 0.0000 | 0.0000 | 20/0/0 |
| var | pair_auc | 20 | STCG-v1 (0.874) | MSVAR (0.662) | 0.2114 | [0.1838, 0.2381] | 0.0000 | 0.0000 | 20/0/0 |
| var | precision_at_k | 20 | STCG-v1 (0.665) | MSVAR (0.400) | 0.2650 | [0.2289, 0.3024] | 0.0000 | 0.0000 | 20/0/0 |
| var | shd | 20 | STCG-v1 (19.900) | MSVAR (35.700) | 15.8000 | [13.6000, 18.2000] | 0.0000 | 0.0000 | 20/0/0 |
| var | lag_accuracy | 20 | STCG-v1 (0.845) | MSVAR (0.680) | 0.1650 | [0.1141, 0.2150] | 0.0000 | 0.0001 | 17/1/2 |
| nonlinear_var | edge_auc | 20 | STCG-v1 (0.969) | MSVAR (0.895) | 0.0738 | [0.0617, 0.0867] | 0.0000 | 0.0000 | 20/0/0 |
| nonlinear_var | pair_auc | 20 | STCG-v1 (0.964) | MSVAR (0.827) | 0.1377 | [0.1129, 0.1630] | 0.0000 | 0.0000 | 20/0/0 |
| nonlinear_var | precision_at_k | 20 | STCG-v1 (0.860) | MSVAR (0.627) | 0.2328 | [0.1967, 0.2682] | 0.0000 | 0.0000 | 20/0/0 |
| nonlinear_var | shd | 20 | STCG-v1 (8.900) | MSVAR (22.600) | 13.7000 | [11.6000, 15.8000] | 0.0000 | 0.0000 | 20/0/0 |
| nonlinear_var | lag_accuracy | 20 | STCG-v1 (0.976) | MSVAR (0.915) | 0.0605 | [0.0401, 0.0801] | 0.0000 | 0.0002 | 16/3/1 |
| switching_var | edge_auc | 20 | STCG-v1 (0.812) | MSVAR (0.613) | 0.1996 | [0.1752, 0.2241] | 0.0000 | 0.0000 | 20/0/0 |
| switching_var | pair_auc | 20 | STCG-v1 (0.839) | MSVAR (0.610) | 0.2287 | [0.1912, 0.2657] | 0.0000 | 0.0000 | 20/0/0 |
| switching_var | precision_at_k | 20 | STCG-v1 (0.673) | MSVAR (0.409) | 0.2642 | [0.2364, 0.2934] | 0.0000 | 0.0000 | 20/0/0 |
| switching_var | shd | 20 | STCG-v1 (35.500) | MSVAR (64.200) | 28.7000 | [25.6000, 31.8000] | 0.0000 | 0.0000 | 20/0/0 |
| switching_var | lag_accuracy | 20 | STCG-v1 (0.811) | MSVAR (0.568) | 0.2429 | [0.2010, 0.2858] | 0.0000 | 0.0000 | 20/0/0 |
