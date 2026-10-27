# Paired Synthetic Recovery Comparison

Positive improvement means `method_a` is better than `method_b`; for SHD the sign is reversed because lower is better.

| Generator | Metric | N | Method A | Method B | Mean Improvement | 95% bootstrap CI | t-test p | Wilcoxon p | Wins/Ties/Losses |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| var | edge_auc | 20 | STCG-v1 (0.876) | Regime-PCMCI (0.809) | 0.0676 | [0.0218, 0.1172] | 0.0070 | 0.0164 | 13/0/7 |
| var | pair_auc | 20 | STCG-v1 (0.874) | Regime-PCMCI (0.749) | 0.1248 | [0.0644, 0.1866] | 0.0005 | 0.0007 | 15/0/5 |
| var | precision_at_k | 20 | STCG-v1 (0.665) | Regime-PCMCI (0.514) | 0.1513 | [0.0659, 0.2391] | 0.0017 | 0.0063 | 12/1/7 |
| var | shd | 20 | STCG-v1 (19.900) | Regime-PCMCI (29.500) | 9.6000 | [4.2000, 15.4000] | 0.0020 | 0.0049 | 12/1/7 |
| var | lag_accuracy | 20 | STCG-v1 (0.845) | Regime-PCMCI (0.756) | 0.0886 | [0.0185, 0.1632] | 0.0147 | 0.0221 | 11/3/6 |
| nonlinear_var | edge_auc | 19 | STCG-v1 (0.968) | Regime-PCMCI (0.767) | 0.2016 | [0.1338, 0.2671] | 0.0000 | 0.0001 | 16/0/3 |
| nonlinear_var | pair_auc | 19 | STCG-v1 (0.964) | Regime-PCMCI (0.698) | 0.2662 | [0.1822, 0.3461] | 0.0000 | 0.0000 | 17/0/2 |
| nonlinear_var | precision_at_k | 19 | STCG-v1 (0.858) | Regime-PCMCI (0.468) | 0.3899 | [0.2806, 0.4918] | 0.0000 | 0.0002 | 16/2/1 |
| nonlinear_var | shd | 19 | STCG-v1 (9.053) | Regime-PCMCI (32.632) | 23.5789 | [16.7342, 30.1053] | 0.0000 | 0.0002 | 16/2/1 |
| nonlinear_var | lag_accuracy | 19 | STCG-v1 (0.976) | Regime-PCMCI (0.704) | 0.2720 | [0.1851, 0.3586] | 0.0000 | 0.0001 | 17/2/0 |
| switching_var | edge_auc | 20 | STCG-v1 (0.812) | Regime-PCMCI (0.816) | -0.0031 | [-0.0165, 0.0110] | 0.6659 | 0.7147 | 8/0/12 |
| switching_var | pair_auc | 20 | STCG-v1 (0.839) | Regime-PCMCI (0.832) | 0.0071 | [-0.0184, 0.0339] | 0.3054 | 0.4347 | 9/0/11 |
| switching_var | precision_at_k | 20 | STCG-v1 (0.673) | Regime-PCMCI (0.661) | 0.0117 | [-0.0089, 0.0329] | 0.1491 | 0.1429 | 12/2/6 |
| switching_var | shd | 20 | STCG-v1 (35.500) | Regime-PCMCI (36.600) | 1.1000 | [-1.1000, 3.3000] | 0.1803 | 0.2022 | 12/2/6 |
| switching_var | lag_accuracy | 20 | STCG-v1 (0.811) | Regime-PCMCI (0.823) | -0.0120 | [-0.0388, 0.0167] | 0.7922 | 0.8173 | 6/4/10 |
