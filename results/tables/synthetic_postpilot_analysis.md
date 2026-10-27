# Post-pilot sensitivity and two-sided tests

cMLP hyperparameters were selected on seeds 0--4 of all three generators. The original 20-seed tables therefore include pilot observations. This sensitivity analysis excludes them and uses common seeds for all 16 methods. It is retrospective, not a newly collected independent confirmation experiment.

Nonlinear seed 18 is also omitted from every method in this sensitivity analysis because RPCMCI did not return a converged fit.

Positive improvement favors STCG-v1; SHD is sign-reversed. The paired CSV reports both one-sided superiority and two-sided difference tests, plus percentile bootstrap intervals (10,000 draws, seed 12345). No equivalence or non-inferiority margin was specified, so nonsignificance is not equivalence.

| Generator | Common seeds | STCG Edge AUC | cMLP Edge AUC | Difference | One-sided t p | One-sided W p |
|---|---|---:|---:|---:|---:|---:|
| nonlinear_var | 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19 | 0.971092 | 0.983320 | -0.012228 | 0.997076 | 0.997375 |
| switching_var | 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 | 0.803164 | 0.774757 | +0.028407 | 0.008597 | 0.005127 |
| var | 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 | 0.880514 | 0.916956 | -0.036442 | 0.999931 | 0.999786 |

## Full-sample STCG versus Regime-PCMCI (switching)

| Metric | Difference | Two-sided t p | Two-sided W p |
|---|---:|---:|---:|
| edge_auc | -0.003105 | 0.668223 | 0.595819 |
| pair_auc | +0.007121 | 0.610834 | 0.869488 |
| precision_at_k | +0.011743 | 0.298191 | 0.285865 |
| shd | +1.100000 | 0.360558 | 0.404447 |
| lag_accuracy | -0.012034 | 0.415643 | 0.365436 |
