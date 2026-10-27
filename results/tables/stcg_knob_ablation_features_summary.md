# STCG-v1 Knob Ablation: State-feature construction

Generator: switching_var, midpoint switch (n=8, T=1000, max_lag=3), 20 seeds. The default row reconciles with synthetic_recovery_detail.csv STCG-v1.

| Config | Knob | N | Edge AUC | Pair AUC | P@K | SHD | Lag Acc | Accept rate | State ARI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | default | 20 | 0.812 +/- 0.047 | 0.839 +/- 0.058 | 0.673 +/- 0.059 | 35.5 +/- 6.5 | 0.811 +/- 0.062 | 1.00 | 0.764 +/- 0.331 |
| features=all | state features | 20 | 0.813 +/- 0.047 | 0.840 +/- 0.056 | 0.674 +/- 0.058 | 35.4 +/- 6.6 | 0.813 +/- 0.062 | 1.00 | 0.730 +/- 0.340 |
| features=+winstats | state features | 20 | 0.812 +/- 0.047 | 0.839 +/- 0.058 | 0.673 +/- 0.059 | 35.5 +/- 6.5 | 0.811 +/- 0.062 | 1.00 | 0.764 +/- 0.331 |
