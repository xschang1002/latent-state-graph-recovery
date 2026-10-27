# STCG-v1 Knob Ablation: Auto-accept gate thresholds

Generator: switching_var, midpoint switch (n=8, T=1000, max_lag=3), 20 seeds. The default row reconciles with synthetic_recovery_detail.csv STCG-v1.

| Config | Knob | N | Edge AUC | Pair AUC | P@K | SHD | Lag Acc | Accept rate | State ARI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | default | 20 | 0.812 +/- 0.047 | 0.839 +/- 0.058 | 0.673 +/- 0.059 | 35.5 +/- 6.5 | 0.811 +/- 0.062 | 1.00 | 0.764 +/- 0.331 |
| inertia=0.06 | inertia improvement | 20 | 0.812 +/- 0.047 | 0.839 +/- 0.058 | 0.673 +/- 0.059 | 35.5 +/- 6.5 | 0.811 +/- 0.062 | 1.00 | 0.764 +/- 0.331 |
| inertia=0.18 | inertia improvement | 20 | 0.812 +/- 0.047 | 0.839 +/- 0.058 | 0.673 +/- 0.059 | 35.5 +/- 6.5 | 0.811 +/- 0.062 | 1.00 | 0.764 +/- 0.331 |
| coherence=0.45 | temporal coherence | 20 | 0.812 +/- 0.047 | 0.839 +/- 0.058 | 0.673 +/- 0.059 | 35.5 +/- 6.5 | 0.811 +/- 0.062 | 1.00 | 0.764 +/- 0.331 |
| coherence=0.65 | temporal coherence | 20 | 0.812 +/- 0.047 | 0.839 +/- 0.058 | 0.673 +/- 0.059 | 35.5 +/- 6.5 | 0.811 +/- 0.062 | 1.00 | 0.764 +/- 0.331 |
| share=0.05 | min state share | 20 | 0.812 +/- 0.047 | 0.839 +/- 0.058 | 0.673 +/- 0.059 | 35.5 +/- 6.5 | 0.811 +/- 0.062 | 1.00 | 0.764 +/- 0.331 |
| share=0.15 | min state share | 20 | 0.812 +/- 0.047 | 0.839 +/- 0.058 | 0.673 +/- 0.059 | 35.5 +/- 6.5 | 0.811 +/- 0.062 | 1.00 | 0.764 +/- 0.331 |
