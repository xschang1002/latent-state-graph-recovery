# Synthetic Switch-Difficulty Sweep

Smaller switch periods create more frequent graph changes and more mixed sliding windows.

| Switch period | Method | N | Edge AUC | P@K | SHD | State K | State improvement | Temporal coherence | Oracle purity |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 150 | DynVAR-Lasso | 10 | 0.733 +/- 0.076 | 0.578 +/- 0.080 | 45.4 +/- 10.3 | -- | -- | -- | -- |
| 150 | Oracle-state upper bound | 10 | 0.742 +/- 0.075 | 0.589 +/- 0.069 | 44.4 +/- 9.7 | -- | -- | -- | 0.500 |
| 150 | STCG-v1 | 10 | 0.733 +/- 0.076 | 0.564 +/- 0.065 | 47.0 +/- 9.7 | 2.00 | 0.221 | 0.524 | -- |
| 300 | DynVAR-Lasso | 10 | 0.819 +/- 0.050 | 0.643 +/- 0.051 | 38.2 +/- 5.5 | -- | -- | -- | -- |
| 300 | Oracle-state upper bound | 10 | 0.821 +/- 0.039 | 0.657 +/- 0.051 | 36.8 +/- 6.1 | -- | -- | -- | 0.786 |
| 300 | STCG-v1 | 10 | 0.816 +/- 0.052 | 0.664 +/- 0.058 | 36.2 +/- 7.8 | 2.00 | 0.220 | 0.066 | -- |
| 600 | DynVAR-Lasso | 10 | 0.835 +/- 0.050 | 0.671 +/- 0.064 | 35.4 +/- 7.8 | -- | -- | -- | -- |
| 600 | Oracle-state upper bound | 10 | 0.849 +/- 0.050 | 0.696 +/- 0.055 | 32.6 +/- 6.7 | -- | -- | -- | 0.929 |
| 600 | STCG-v1 | 10 | 0.847 +/- 0.051 | 0.695 +/- 0.072 | 32.8 +/- 8.7 | 2.00 | 0.277 | 0.660 | -- |
