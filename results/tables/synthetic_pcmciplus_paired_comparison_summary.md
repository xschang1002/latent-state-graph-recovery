# Paired Synthetic Recovery Comparison

Positive improvement means `method_a` is better than `method_b`; for SHD the sign is reversed because lower is better.

| Generator | Metric | N | Method A | Method B | Mean Improvement | 95% bootstrap CI | t-test p | Wilcoxon p | Wins/Ties/Losses |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| var | edge_auc | 20 | STCG-v1 (0.876) | PCMCI+ (0.921) | -0.0443 | [-0.0573, -0.0321] | 1.0000 | 1.0000 | 1/0/19 |
| var | pair_auc | 20 | STCG-v1 (0.874) | PCMCI+ (0.912) | -0.0385 | [-0.0576, -0.0207] | 0.9996 | 0.9993 | 5/1/14 |
| var | precision_at_k | 20 | STCG-v1 (0.665) | PCMCI+ (0.753) | -0.0876 | [-0.1063, -0.0698] | 1.0000 | 0.9999 | 0/1/19 |
| var | shd | 20 | STCG-v1 (19.900) | PCMCI+ (14.600) | -5.3000 | [-6.5000, -4.2000] | 1.0000 | 0.9999 | 0/1/19 |
| var | lag_accuracy | 20 | STCG-v1 (0.845) | PCMCI+ (0.905) | -0.0602 | [-0.0908, -0.0320] | 0.9995 | 0.9992 | 2/3/15 |
| nonlinear_var | edge_auc | 20 | STCG-v1 (0.969) | PCMCI+ (0.919) | 0.0499 | [0.0395, 0.0614] | 0.0000 | 0.0000 | 20/0/0 |
| nonlinear_var | pair_auc | 20 | STCG-v1 (0.964) | PCMCI+ (0.944) | 0.0201 | [0.0070, 0.0349] | 0.0062 | 0.0053 | 15/0/5 |
| nonlinear_var | precision_at_k | 20 | STCG-v1 (0.860) | PCMCI+ (0.837) | 0.0224 | [0.0040, 0.0415] | 0.0175 | 0.0222 | 10/6/4 |
| nonlinear_var | shd | 20 | STCG-v1 (8.900) | PCMCI+ (10.300) | 1.4000 | [0.3000, 2.6000] | 0.0154 | 0.0176 | 10/6/4 |
| nonlinear_var | lag_accuracy | 20 | STCG-v1 (0.976) | PCMCI+ (0.901) | 0.0748 | [0.0521, 0.0980] | 0.0000 | 0.0001 | 17/3/0 |
| switching_var | edge_auc | 20 | STCG-v1 (0.812) | PCMCI+ (0.769) | 0.0433 | [0.0272, 0.0591] | 0.0000 | 0.0000 | 17/0/3 |
| switching_var | pair_auc | 20 | STCG-v1 (0.839) | PCMCI+ (0.808) | 0.0313 | [0.0099, 0.0531] | 0.0060 | 0.0028 | 15/0/5 |
| switching_var | precision_at_k | 20 | STCG-v1 (0.673) | PCMCI+ (0.618) | 0.0548 | [0.0323, 0.0747] | 0.0000 | 0.0006 | 17/0/3 |
| switching_var | shd | 20 | STCG-v1 (35.500) | PCMCI+ (41.600) | 6.1000 | [3.6000, 8.4000] | 0.0001 | 0.0005 | 17/0/3 |
| switching_var | lag_accuracy | 20 | STCG-v1 (0.811) | PCMCI+ (0.756) | 0.0552 | [0.0296, 0.0816] | 0.0003 | 0.0009 | 15/2/3 |
