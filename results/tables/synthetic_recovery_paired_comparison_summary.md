# Paired Synthetic Recovery Comparison

Positive improvement means `method_a` is better than `method_b`; for SHD the sign is reversed because lower is better.

| Generator | Metric | N | Method A | Method B | Mean Improvement | 95% bootstrap CI | t-test p | Wilcoxon p | Wins/Ties/Losses |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| var | edge_auc | 20 | STCG-v1 (0.876) | DynVAR-Lasso (0.866) | 0.0106 | [0.0051, 0.0164] | 0.0011 | 0.0008 | 16/0/4 |
| var | pair_auc | 20 | STCG-v1 (0.874) | DynVAR-Lasso (0.849) | 0.0250 | [0.0072, 0.0427] | 0.0071 | 0.0077 | 14/0/6 |
| var | precision_at_k | 20 | STCG-v1 (0.665) | DynVAR-Lasso (0.645) | 0.0199 | [-0.0024, 0.0435] | 0.0583 | 0.0760 | 9/7/4 |
| var | shd | 20 | STCG-v1 (19.900) | DynVAR-Lasso (21.200) | 1.3000 | [-0.1000, 2.8000] | 0.0483 | 0.0548 | 9/7/4 |
| var | lag_accuracy | 20 | STCG-v1 (0.845) | DynVAR-Lasso (0.847) | -0.0024 | [-0.0216, 0.0161] | 0.5915 | 0.5624 | 7/6/7 |
| nonlinear_var | edge_auc | 20 | STCG-v1 (0.969) | DynVAR-Lasso (0.956) | 0.0134 | [0.0092, 0.0183] | 0.0000 | 0.0000 | 20/0/0 |
| nonlinear_var | pair_auc | 20 | STCG-v1 (0.964) | DynVAR-Lasso (0.942) | 0.0219 | [0.0138, 0.0312] | 0.0001 | 0.0001 | 18/1/1 |
| nonlinear_var | precision_at_k | 20 | STCG-v1 (0.860) | DynVAR-Lasso (0.820) | 0.0398 | [0.0250, 0.0563] | 0.0001 | 0.0005 | 14/6/0 |
| nonlinear_var | shd | 20 | STCG-v1 (8.900) | DynVAR-Lasso (11.300) | 2.4000 | [1.5000, 3.3000] | 0.0000 | 0.0004 | 14/6/0 |
| nonlinear_var | lag_accuracy | 20 | STCG-v1 (0.976) | DynVAR-Lasso (0.969) | 0.0069 | [0.0015, 0.0137] | 0.0220 | 0.0339 | 4/16/0 |
| switching_var | edge_auc | 20 | STCG-v1 (0.812) | DynVAR-Lasso (0.798) | 0.0143 | [0.0067, 0.0213] | 0.0007 | 0.0014 | 17/0/3 |
| switching_var | pair_auc | 20 | STCG-v1 (0.839) | DynVAR-Lasso (0.830) | 0.0092 | [-0.0081, 0.0272] | 0.1683 | 0.2045 | 11/0/9 |
| switching_var | precision_at_k | 20 | STCG-v1 (0.673) | DynVAR-Lasso (0.646) | 0.0274 | [0.0150, 0.0403] | 0.0003 | 0.0012 | 14/4/2 |
| switching_var | shd | 20 | STCG-v1 (35.500) | DynVAR-Lasso (38.600) | 3.1000 | [1.7000, 4.6000] | 0.0003 | 0.0007 | 14/4/2 |
| switching_var | lag_accuracy | 20 | STCG-v1 (0.811) | DynVAR-Lasso (0.797) | 0.0143 | [-0.0055, 0.0366] | 0.1054 | 0.1896 | 9/6/5 |
