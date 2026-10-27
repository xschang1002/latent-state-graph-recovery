# STCG-v1 Knob Ablation: Within-/cross-state aggregators

Generator: switching_var, midpoint switch (n=8, T=1000, max_lag=3), 20 seeds. The default row reconciles with synthetic_recovery_detail.csv STCG-v1.

| Config | Knob | N | Edge AUC | Pair AUC | P@K | SHD | Lag Acc | Accept rate | State ARI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | default | 20 | 0.812 +/- 0.047 | 0.839 +/- 0.058 | 0.673 +/- 0.059 | 35.5 +/- 6.5 | 0.811 +/- 0.062 | 1.00 | 0.764 +/- 0.331 |
| state=median | within-state agg | 20 | 0.804 +/- 0.054 | 0.828 +/- 0.064 | 0.653 +/- 0.064 | 37.5 +/- 6.1 | 0.808 +/- 0.072 | 1.00 | 0.764 +/- 0.331 |
| state=p75 | within-state agg | 20 | 0.808 +/- 0.045 | 0.834 +/- 0.052 | 0.663 +/- 0.055 | 36.7 +/- 6.8 | 0.800 +/- 0.063 | 1.00 | 0.764 +/- 0.331 |
| state=max | within-state agg | 20 | 0.798 +/- 0.040 | 0.830 +/- 0.058 | 0.646 +/- 0.047 | 38.6 +/- 6.4 | 0.797 +/- 0.058 | 1.00 | 0.764 +/- 0.331 |
| cross=mean | cross-state agg | 20 | 0.810 +/- 0.048 | 0.833 +/- 0.064 | 0.665 +/- 0.060 | 36.5 +/- 7.1 | 0.814 +/- 0.058 | 1.00 | 0.764 +/- 0.331 |
| cross=p90 | cross-state agg | 20 | 0.814 +/- 0.048 | 0.840 +/- 0.058 | 0.676 +/- 0.059 | 35.2 +/- 6.8 | 0.816 +/- 0.058 | 1.00 | 0.764 +/- 0.331 |
