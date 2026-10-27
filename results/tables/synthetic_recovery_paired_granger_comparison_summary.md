# Paired Synthetic Recovery Comparison

Positive improvement means `method_a` is better than `method_b`; for SHD the sign is reversed because lower is better.

| Generator | Metric | N | Method A | Method B | Mean Improvement | 95% bootstrap CI | t-test p | Wilcoxon p | Wins/Ties/Losses |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| var | edge_auc | 20 | STCG-v1 (0.876) | DynGranger-F (0.861) | 0.0155 | [0.0089, 0.0222] | 0.0001 | 0.0001 | 18/0/2 |
| var | pair_auc | 20 | STCG-v1 (0.874) | DynGranger-F (0.846) | 0.0273 | [0.0091, 0.0446] | 0.0041 | 0.0050 | 14/0/6 |
| var | precision_at_k | 20 | STCG-v1 (0.665) | DynGranger-F (0.639) | 0.0254 | [0.0038, 0.0475] | 0.0205 | 0.0250 | 11/5/4 |
| var | shd | 20 | STCG-v1 (19.900) | DynGranger-F (21.500) | 1.6000 | [0.3000, 3.0000] | 0.0193 | 0.0238 | 11/5/4 |
| var | lag_accuracy | 20 | STCG-v1 (0.845) | DynGranger-F (0.838) | 0.0068 | [-0.0127, 0.0265] | 0.2600 | 0.2651 | 7/8/5 |
| nonlinear_var | edge_auc | 20 | STCG-v1 (0.969) | DynGranger-F (0.975) | -0.0062 | [-0.0162, 0.0019] | 0.8880 | 0.7392 | 9/0/11 |
| nonlinear_var | pair_auc | 20 | STCG-v1 (0.964) | DynGranger-F (0.973) | -0.0084 | [-0.0181, -0.0001] | 0.9559 | 0.9634 | 6/1/13 |
| nonlinear_var | precision_at_k | 20 | STCG-v1 (0.860) | DynGranger-F (0.878) | -0.0178 | [-0.0441, 0.0048] | 0.9124 | 0.8829 | 6/4/10 |
| nonlinear_var | shd | 20 | STCG-v1 (8.900) | DynGranger-F (7.600) | -1.3000 | [-3.3000, 0.2000] | 0.9104 | 0.9063 | 6/4/10 |
| nonlinear_var | lag_accuracy | 20 | STCG-v1 (0.976) | DynGranger-F (0.964) | 0.0120 | [0.0034, 0.0231] | 0.0177 | 0.0216 | 5/15/0 |
| switching_var | edge_auc | 20 | STCG-v1 (0.812) | DynGranger-F (0.797) | 0.0158 | [0.0080, 0.0230] | 0.0004 | 0.0007 | 17/0/3 |
| switching_var | pair_auc | 20 | STCG-v1 (0.839) | DynGranger-F (0.830) | 0.0090 | [-0.0099, 0.0288] | 0.1948 | 0.2262 | 11/0/9 |
| switching_var | precision_at_k | 20 | STCG-v1 (0.673) | DynGranger-F (0.650) | 0.0231 | [0.0110, 0.0356] | 0.0011 | 0.0026 | 13/6/1 |
| switching_var | shd | 20 | STCG-v1 (35.500) | DynGranger-F (38.100) | 2.6000 | [1.3000, 4.0000] | 0.0010 | 0.0024 | 13/6/1 |
| switching_var | lag_accuracy | 20 | STCG-v1 (0.811) | DynGranger-F (0.791) | 0.0204 | [0.0025, 0.0395] | 0.0247 | 0.0609 | 12/2/6 |
