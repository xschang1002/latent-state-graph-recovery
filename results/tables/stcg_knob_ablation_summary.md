# STCG-v1 Knob Ablation Summary (switching_var, 20 seeds)

One-factor-at-a-time isolation of the heuristic knobs mapped in the paper's
knob table (tab:knob-map). Edge AUC is mean +/- std over seeds; Accept rate is
the fraction of seeds where the auto-accept gate fires (K=2 states chosen);
State ARI compares inferred window states with the ground-truth regime.

| Group | Config | Edge AUC | Accept rate | State ARI |
|---|---|---:|---:|---:|
| gates | default | 0.812 +/- 0.047 | 1.00 | 0.764 +/- 0.331 |
| gates | inertia=0.06 | 0.812 +/- 0.047 | 1.00 | 0.764 +/- 0.331 |
| gates | inertia=0.18 | 0.812 +/- 0.047 | 1.00 | 0.764 +/- 0.331 |
| gates | coherence=0.45 | 0.812 +/- 0.047 | 1.00 | 0.764 +/- 0.331 |
| gates | coherence=0.65 | 0.812 +/- 0.047 | 1.00 | 0.764 +/- 0.331 |
| gates | share=0.05 | 0.812 +/- 0.047 | 1.00 | 0.764 +/- 0.331 |
| gates | share=0.15 | 0.812 +/- 0.047 | 1.00 | 0.764 +/- 0.331 |
| aggregators | default | 0.812 +/- 0.047 | 1.00 | 0.764 +/- 0.331 |
| aggregators | state=median | 0.804 +/- 0.054 | 1.00 | 0.764 +/- 0.331 |
| aggregators | state=p75 | 0.808 +/- 0.045 | 1.00 | 0.764 +/- 0.331 |
| aggregators | state=max | 0.798 +/- 0.040 | 1.00 | 0.764 +/- 0.331 |
| aggregators | cross=mean | 0.810 +/- 0.048 | 1.00 | 0.764 +/- 0.331 |
| aggregators | cross=p90 | 0.814 +/- 0.048 | 1.00 | 0.764 +/- 0.331 |
| features | default | 0.812 +/- 0.047 | 1.00 | 0.764 +/- 0.331 |
| features | features=all | 0.813 +/- 0.047 | 1.00 | 0.730 +/- 0.340 |
| features | features=+winstats | 0.812 +/- 0.047 | 1.00 | 0.764 +/- 0.331 |
| windows | h=150 | 0.748 +/- 0.046 | 1.00 | 0.333 +/- 0.231 |
| windows | h=300 | 0.781 +/- 0.052 | 1.00 | 0.035 +/- 0.200 |
| windows | h=450 | 0.765 +/- 0.048 | 1.00 | 0.050 +/- 0.347 |
