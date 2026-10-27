# Synthetic Oracle-State Upper Bound

Oracle-state aggregation uses true synthetic regime labels only for analysis.
It is an upper-bound diagnostic, not a supervised variant of STCG-v1.

| Method | N | Edge AUC | P@K | SHD | Lag Acc. | Inferred K | Oracle K | Oracle purity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DynVAR-Lasso | 20 | 0.798 +/- 0.040 | 0.646 +/- 0.047 | 38.6 +/- 6.4 | 0.797 +/- 0.058 | -- | -- | -- |
| Oracle-state upper bound | 20 | 0.813 +/- 0.045 | 0.672 +/- 0.056 | 35.6 +/- 6.1 | 0.800 +/- 0.052 | -- | 2.00 | 0.917 |
| STCG-v1/inferred-state | 20 | 0.812 +/- 0.047 | 0.673 +/- 0.059 | 35.5 +/- 6.5 | 0.811 +/- 0.062 | 2.00 | -- | -- |
| STCG-v1/no-state | 20 | 0.798 +/- 0.040 | 0.646 +/- 0.047 | 38.6 +/- 6.4 | 0.797 +/- 0.058 | -- | -- | -- |
