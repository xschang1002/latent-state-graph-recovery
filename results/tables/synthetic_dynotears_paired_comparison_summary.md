# Paired Synthetic Recovery Comparison

Positive improvement means `method_a` is better than `method_b`; for SHD the sign is reversed because lower is better.

| Generator | Metric | N | Method A | Method B | Mean Improvement | 95% bootstrap CI | t-test p | Wilcoxon p | Wins/Ties/Losses |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| switching_var | edge_auc | 20 | STCG-v1 (0.812) | DYNOTEARS (0.746) | 0.0664 | [0.0532, 0.0802] | 0.0000 | 0.0000 | 20/0/0 |
| switching_var | pair_auc | 20 | STCG-v1 (0.839) | DYNOTEARS (0.795) | 0.0436 | [0.0236, 0.0643] | 0.0003 | 0.0004 | 16/0/4 |
| switching_var | precision_at_k | 20 | STCG-v1 (0.673) | DYNOTEARS (0.586) | 0.0873 | [0.0645, 0.1095] | 0.0000 | 0.0001 | 19/0/1 |
| switching_var | shd | 20 | STCG-v1 (35.500) | DYNOTEARS (45.200) | 9.7000 | [7.0000, 12.4000] | 0.0000 | 0.0001 | 19/0/1 |
| switching_var | lag_accuracy | 20 | STCG-v1 (0.811) | DYNOTEARS (0.697) | 0.1144 | [0.0953, 0.1337] | 0.0000 | 0.0000 | 20/0/0 |
