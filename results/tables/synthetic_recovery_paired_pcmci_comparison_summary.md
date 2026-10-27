# Paired Synthetic Recovery Comparison

Positive improvement means `method_a` is better than `method_b`; for SHD the sign is reversed because lower is better.

| Generator | Metric | N | Method A | Method B | Mean Improvement | 95% bootstrap CI | t-test p | Wilcoxon p | Wins/Ties/Losses |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| var | edge_auc | 20 | STCG-v1 (0.876) | PCMCI-ParCorr (0.920) | -0.0437 | [-0.0557, -0.0312] | 1.0000 | 1.0000 | 3/0/17 |
| var | pair_auc | 20 | STCG-v1 (0.874) | PCMCI-ParCorr (0.915) | -0.0415 | [-0.0585, -0.0254] | 0.9999 | 1.0000 | 1/0/19 |
| var | precision_at_k | 20 | STCG-v1 (0.665) | PCMCI-ParCorr (0.749) | -0.0843 | [-0.1060, -0.0622] | 1.0000 | 0.9999 | 1/1/18 |
| var | shd | 20 | STCG-v1 (19.900) | PCMCI-ParCorr (14.900) | -5.0000 | [-6.3000, -3.7000] | 1.0000 | 0.9999 | 1/1/18 |
| var | lag_accuracy | 20 | STCG-v1 (0.845) | PCMCI-ParCorr (0.873) | -0.0283 | [-0.0580, 0.0035] | 0.9523 | 0.9621 | 4/3/13 |
| nonlinear_var | edge_auc | 20 | STCG-v1 (0.969) | PCMCI-ParCorr (0.984) | -0.0145 | [-0.0261, -0.0052] | 0.9906 | 0.9984 | 5/0/15 |
| nonlinear_var | pair_auc | 20 | STCG-v1 (0.964) | PCMCI-ParCorr (0.985) | -0.0202 | [-0.0293, -0.0111] | 0.9998 | 0.9999 | 2/0/18 |
| nonlinear_var | precision_at_k | 20 | STCG-v1 (0.860) | PCMCI-ParCorr (0.924) | -0.0643 | [-0.0857, -0.0432] | 1.0000 | 0.9998 | 1/2/17 |
| nonlinear_var | shd | 20 | STCG-v1 (8.900) | PCMCI-ParCorr (4.900) | -4.0000 | [-5.6000, -2.6000] | 1.0000 | 0.9998 | 1/2/17 |
| nonlinear_var | lag_accuracy | 20 | STCG-v1 (0.976) | PCMCI-ParCorr (0.971) | 0.0046 | [-0.0077, 0.0179] | 0.2501 | 0.3060 | 4/13/3 |
| switching_var | edge_auc | 20 | STCG-v1 (0.812) | PCMCI-ParCorr (0.779) | 0.0334 | [0.0161, 0.0492] | 0.0005 | 0.0004 | 17/0/3 |
| switching_var | pair_auc | 20 | STCG-v1 (0.839) | PCMCI-ParCorr (0.809) | 0.0294 | [0.0099, 0.0484] | 0.0043 | 0.0016 | 17/0/3 |
| switching_var | precision_at_k | 20 | STCG-v1 (0.673) | PCMCI-ParCorr (0.621) | 0.0520 | [0.0296, 0.0709] | 0.0001 | 0.0007 | 18/1/1 |
| switching_var | shd | 20 | STCG-v1 (35.500) | PCMCI-ParCorr (41.300) | 5.8000 | [3.3000, 7.9000] | 0.0001 | 0.0006 | 18/1/1 |
| switching_var | lag_accuracy | 20 | STCG-v1 (0.811) | PCMCI-ParCorr (0.760) | 0.0511 | [0.0240, 0.0775] | 0.0008 | 0.0028 | 15/3/2 |
