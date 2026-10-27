# Paired Synthetic Recovery Comparison

Positive improvement means `method_a` is better than `method_b`; for SHD the sign is reversed because lower is better.

| Generator | Metric | N | Method A | Method B | Mean Improvement | 95% bootstrap CI | t-test p | Wilcoxon p | Wins/Ties/Losses |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| var | edge_auc | 20 | STCG-v1 (0.876) | cMLP (0.918) | -0.0413 | [-0.0529, -0.0294] | 1.0000 | 1.0000 | 2/0/18 |
| var | pair_auc | 20 | STCG-v1 (0.874) | cMLP (0.909) | -0.0352 | [-0.0511, -0.0203] | 0.9998 | 0.9999 | 3/0/17 |
| var | precision_at_k | 20 | STCG-v1 (0.665) | cMLP (0.749) | -0.0846 | [-0.0997, -0.0689] | 1.0000 | 0.9999 | 0/1/19 |
| var | shd | 20 | STCG-v1 (19.900) | cMLP (14.800) | -5.1000 | [-6.1000, -4.1000] | 1.0000 | 0.9999 | 0/1/19 |
| var | lag_accuracy | 20 | STCG-v1 (0.845) | cMLP (0.880) | -0.0349 | [-0.0721, 0.0004] | 0.9614 | 0.9695 | 7/2/11 |
| nonlinear_var | edge_auc | 20 | STCG-v1 (0.969) | cMLP (0.980) | -0.0107 | [-0.0162, -0.0052] | 0.9992 | 0.9993 | 3/0/17 |
| nonlinear_var | pair_auc | 20 | STCG-v1 (0.964) | cMLP (0.974) | -0.0099 | [-0.0201, -0.0002] | 0.9629 | 0.9818 | 5/1/14 |
| nonlinear_var | precision_at_k | 20 | STCG-v1 (0.860) | cMLP (0.886) | -0.0263 | [-0.0495, -0.0025] | 0.9772 | 0.9609 | 4/5/11 |
| nonlinear_var | shd | 20 | STCG-v1 (8.900) | cMLP (7.100) | -1.8000 | [-3.4000, -0.3000] | 0.9797 | 0.9796 | 4/5/11 |
| nonlinear_var | lag_accuracy | 20 | STCG-v1 (0.976) | cMLP (0.985) | -0.0095 | [-0.0173, -0.0031] | 0.9890 | 0.9789 | 0/15/5 |
| switching_var | edge_auc | 20 | STCG-v1 (0.812) | cMLP (0.780) | 0.0322 | [0.0140, 0.0480] | 0.0007 | 0.0004 | 18/0/2 |
| switching_var | pair_auc | 20 | STCG-v1 (0.839) | cMLP (0.809) | 0.0296 | [0.0110, 0.0482] | 0.0033 | 0.0042 | 16/0/4 |
| switching_var | precision_at_k | 20 | STCG-v1 (0.673) | cMLP (0.624) | 0.0487 | [0.0230, 0.0693] | 0.0004 | 0.0011 | 18/1/1 |
| switching_var | shd | 20 | STCG-v1 (35.500) | cMLP (40.900) | 5.4000 | [2.6000, 7.6025] | 0.0003 | 0.0010 | 18/1/1 |
| switching_var | lag_accuracy | 20 | STCG-v1 (0.811) | cMLP (0.766) | 0.0453 | [0.0193, 0.0694] | 0.0012 | 0.0042 | 15/2/3 |
