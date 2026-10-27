# STCG-v1 Knob Ablation: Window size (stride = h/2, switch period 300)

Generator: switching_var with periodic switch period 300 (n=8, T=1000, max_lag=3), 20 seeds.

| Config | Knob | N | Edge AUC | Pair AUC | P@K | SHD | Lag Acc | Accept rate | State ARI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h=150 | window size | 20 | 0.748 +/- 0.046 | 0.767 +/- 0.077 | 0.582 +/- 0.050 | 45.5 +/- 7.0 | 0.744 +/- 0.080 | 1.00 | 0.333 +/- 0.231 |
| h=300 | window size | 20 | 0.781 +/- 0.052 | 0.795 +/- 0.065 | 0.616 +/- 0.061 | 41.9 +/- 8.3 | 0.779 +/- 0.071 | 1.00 | 0.035 +/- 0.200 |
| h=450 | window size | 20 | 0.765 +/- 0.048 | 0.771 +/- 0.051 | 0.597 +/- 0.046 | 43.9 +/- 6.7 | 0.762 +/- 0.063 | 1.00 | 0.050 +/- 0.347 |
