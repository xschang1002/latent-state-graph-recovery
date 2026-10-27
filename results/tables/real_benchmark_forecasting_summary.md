# Real Benchmark Forecasting Summary

| Dataset | Method | N | RMSE | MAE | Mean Node RMSE | Max Node RMSE |
|---|---|---:|---:|---:|---:|---:|
| AReMActivities | Persistence | 5 | 1.128 +/- 0.000 | 0.676 +/- 0.000 | 1.014 +/- 0.000 | 1.858 +/- 0.000 |
| AReMActivities | Dense Ridge | 5 | 0.912 +/- 0.000 | 0.574 +/- 0.000 | 0.842 +/- 0.000 | 1.455 +/- 0.000 |
| AReMActivities | RandomGraph Ridge | 5 | 1.113 +/- 0.015 | 0.753 +/- 0.008 | 1.039 +/- 0.014 | 1.822 +/- 0.067 |
| AReMActivities | DynVAR-Lasso Graph Ridge | 5 | 1.259 +/- 0.000 | 0.841 +/- 0.000 | 1.147 +/- 0.000 | 2.127 +/- 0.000 |
| AReMActivities | DynVAR-Lasso Weighted Ridge | 5 | 0.912 +/- 0.000 | 0.574 +/- 0.000 | 0.842 +/- 0.000 | 1.455 +/- 0.000 |
| AReMActivities | STCG-v1 Graph Ridge | 5 | 1.230 +/- 0.000 | 0.833 +/- 0.000 | 1.131 +/- 0.000 | 2.102 +/- 0.000 |
| AReMActivities | STCG-v1 Weighted Ridge | 5 | 0.912 +/- 0.000 | 0.574 +/- 0.000 | 0.842 +/- 0.000 | 1.455 +/- 0.000 |
| BeijingPM25 | Persistence | 5 | 0.266 +/- 0.000 | 0.133 +/- 0.000 | 0.266 +/- 0.000 | 0.289 +/- 0.000 |
| BeijingPM25 | Dense Ridge | 5 | 0.233 +/- 0.000 | 0.122 +/- 0.000 | 0.232 +/- 0.000 | 0.267 +/- 0.000 |
| BeijingPM25 | RandomGraph Ridge | 5 | 0.410 +/- 0.015 | 0.222 +/- 0.011 | 0.403 +/- 0.016 | 0.541 +/- 0.027 |
| BeijingPM25 | DynVAR-Lasso Graph Ridge | 5 | 0.367 +/- 0.000 | 0.193 +/- 0.000 | 0.359 +/- 0.000 | 0.501 +/- 0.000 |
| BeijingPM25 | DynVAR-Lasso Weighted Ridge | 5 | 0.233 +/- 0.000 | 0.122 +/- 0.000 | 0.232 +/- 0.000 | 0.267 +/- 0.000 |
| BeijingPM25 | STCG-v1 Graph Ridge | 5 | 0.367 +/- 0.000 | 0.193 +/- 0.000 | 0.359 +/- 0.000 | 0.501 +/- 0.000 |
| BeijingPM25 | STCG-v1 Weighted Ridge | 5 | 0.233 +/- 0.000 | 0.122 +/- 0.000 | 0.232 +/- 0.000 | 0.267 +/- 0.000 |
| ETTh1 | Persistence | 5 | 0.439 +/- 0.000 | 0.269 +/- 0.000 | 0.406 +/- 0.000 | 0.574 +/- 0.000 |
| ETTh1 | Dense Ridge | 5 | 0.409 +/- 0.000 | 0.262 +/- 0.000 | 0.380 +/- 0.000 | 0.555 +/- 0.000 |
| ETTh1 | RandomGraph Ridge | 5 | 0.956 +/- 0.128 | 0.701 +/- 0.119 | 0.873 +/- 0.137 | 1.540 +/- 0.016 |
| ETTh1 | DynVAR-Lasso Graph Ridge | 5 | 0.728 +/- 0.000 | 0.506 +/- 0.000 | 0.636 +/- 0.000 | 1.489 +/- 0.000 |
| ETTh1 | DynVAR-Lasso Weighted Ridge | 5 | 0.409 +/- 0.000 | 0.262 +/- 0.000 | 0.380 +/- 0.000 | 0.555 +/- 0.000 |
| ETTh1 | STCG-v1 Graph Ridge | 5 | 0.760 +/- 0.018 | 0.528 +/- 0.007 | 0.661 +/- 0.009 | 1.524 +/- 0.066 |
| ETTh1 | STCG-v1 Weighted Ridge | 5 | 0.409 +/- 0.000 | 0.262 +/- 0.000 | 0.380 +/- 0.000 | 0.555 +/- 0.000 |
| ETTh2 | Persistence | 5 | 0.246 +/- 0.000 | 0.150 +/- 0.000 | 0.224 +/- 0.000 | 0.336 +/- 0.000 |
| ETTh2 | Dense Ridge | 5 | 0.227 +/- 0.000 | 0.146 +/- 0.000 | 0.204 +/- 0.000 | 0.312 +/- 0.000 |
| ETTh2 | RandomGraph Ridge | 5 | 0.805 +/- 0.076 | 0.641 +/- 0.071 | 0.765 +/- 0.080 | 1.138 +/- 0.132 |
| ETTh2 | DynVAR-Lasso Graph Ridge | 5 | 0.888 +/- 0.000 | 0.709 +/- 0.000 | 0.842 +/- 0.000 | 1.179 +/- 0.000 |
| ETTh2 | DynVAR-Lasso Weighted Ridge | 5 | 0.227 +/- 0.000 | 0.146 +/- 0.000 | 0.204 +/- 0.000 | 0.312 +/- 0.000 |
| ETTh2 | STCG-v1 Graph Ridge | 5 | 0.888 +/- 0.000 | 0.709 +/- 0.000 | 0.842 +/- 0.000 | 1.179 +/- 0.000 |
| ETTh2 | STCG-v1 Weighted Ridge | 5 | 0.227 +/- 0.000 | 0.146 +/- 0.000 | 0.204 +/- 0.000 | 0.312 +/- 0.000 |
| ETTm1 | Persistence | 5 | 0.253 +/- 0.000 | 0.144 +/- 0.000 | 0.238 +/- 0.000 | 0.293 +/- 0.000 |
| ETTm1 | Dense Ridge | 5 | 0.242 +/- 0.000 | 0.142 +/- 0.000 | 0.227 +/- 0.000 | 0.289 +/- 0.000 |
| ETTm1 | RandomGraph Ridge | 5 | 0.900 +/- 0.157 | 0.628 +/- 0.142 | 0.779 +/- 0.177 | 1.537 +/- 0.024 |
| ETTm1 | DynVAR-Lasso Graph Ridge | 5 | 0.695 +/- 0.000 | 0.436 +/- 0.000 | 0.536 +/- 0.000 | 1.602 +/- 0.000 |
| ETTm1 | DynVAR-Lasso Weighted Ridge | 5 | 0.242 +/- 0.000 | 0.142 +/- 0.000 | 0.227 +/- 0.000 | 0.289 +/- 0.000 |
| ETTm1 | STCG-v1 Graph Ridge | 5 | 0.695 +/- 0.000 | 0.436 +/- 0.000 | 0.536 +/- 0.000 | 1.602 +/- 0.000 |
| ETTm1 | STCG-v1 Weighted Ridge | 5 | 0.242 +/- 0.000 | 0.142 +/- 0.000 | 0.227 +/- 0.000 | 0.289 +/- 0.000 |
| ETTm2 | Persistence | 5 | 0.156 +/- 0.000 | 0.085 +/- 0.000 | 0.139 +/- 0.000 | 0.226 +/- 0.000 |
| ETTm2 | Dense Ridge | 5 | 0.149 +/- 0.000 | 0.088 +/- 0.000 | 0.132 +/- 0.000 | 0.214 +/- 0.000 |
| ETTm2 | RandomGraph Ridge | 5 | 0.796 +/- 0.074 | 0.632 +/- 0.071 | 0.751 +/- 0.080 | 1.141 +/- 0.122 |
| ETTm2 | DynVAR-Lasso Graph Ridge | 5 | 0.802 +/- 0.000 | 0.640 +/- 0.000 | 0.746 +/- 0.000 | 1.207 +/- 0.000 |
| ETTm2 | DynVAR-Lasso Weighted Ridge | 5 | 0.149 +/- 0.000 | 0.088 +/- 0.000 | 0.132 +/- 0.000 | 0.214 +/- 0.000 |
| ETTm2 | STCG-v1 Graph Ridge | 5 | 0.802 +/- 0.000 | 0.640 +/- 0.000 | 0.746 +/- 0.000 | 1.207 +/- 0.000 |
| ETTm2 | STCG-v1 Weighted Ridge | 5 | 0.149 +/- 0.000 | 0.088 +/- 0.000 | 0.132 +/- 0.000 | 0.214 +/- 0.000 |
| HydraulicSystems | Persistence | 5 | 0.167 +/- 0.000 | 0.037 +/- 0.000 | 0.086 +/- 0.000 | 0.451 +/- 0.000 |
| HydraulicSystems | Dense Ridge | 5 | 0.155 +/- 0.000 | 0.046 +/- 0.000 | 0.080 +/- 0.000 | 0.417 +/- 0.000 |
| HydraulicSystems | RandomGraph Ridge | 5 | 0.500 +/- 0.049 | 0.288 +/- 0.055 | 0.345 +/- 0.053 | 0.937 +/- 0.011 |
| HydraulicSystems | DynVAR-Lasso Graph Ridge | 5 | 0.472 +/- 0.000 | 0.236 +/- 0.000 | 0.296 +/- 0.000 | 0.940 +/- 0.000 |
| HydraulicSystems | DynVAR-Lasso Weighted Ridge | 5 | 0.156 +/- 0.000 | 0.046 +/- 0.000 | 0.080 +/- 0.000 | 0.418 +/- 0.000 |
| HydraulicSystems | STCG-v1 Graph Ridge | 5 | 0.470 +/- 0.000 | 0.234 +/- 0.000 | 0.295 +/- 0.000 | 0.940 +/- 0.000 |
| HydraulicSystems | STCG-v1 Weighted Ridge | 5 | 0.156 +/- 0.000 | 0.046 +/- 0.000 | 0.080 +/- 0.000 | 0.418 +/- 0.000 |
| OccupancySensors | Persistence | 5 | 0.046 +/- 0.000 | 0.014 +/- 0.000 | 0.039 +/- 0.000 | 0.068 +/- 0.000 |
| OccupancySensors | Dense Ridge | 5 | 0.046 +/- 0.000 | 0.016 +/- 0.000 | 0.039 +/- 0.000 | 0.069 +/- 0.000 |
| OccupancySensors | RandomGraph Ridge | 5 | 0.776 +/- 0.142 | 0.495 +/- 0.134 | 0.644 +/- 0.174 | 1.296 +/- 0.042 |
| OccupancySensors | DynVAR-Lasso Graph Ridge | 5 | 0.744 +/- 0.000 | 0.361 +/- 0.000 | 0.511 +/- 0.000 | 1.445 +/- 0.000 |
| OccupancySensors | DynVAR-Lasso Weighted Ridge | 5 | 0.046 +/- 0.000 | 0.016 +/- 0.000 | 0.039 +/- 0.000 | 0.069 +/- 0.000 |
| OccupancySensors | STCG-v1 Graph Ridge | 5 | 0.744 +/- 0.000 | 0.361 +/- 0.000 | 0.511 +/- 0.000 | 1.445 +/- 0.000 |
| OccupancySensors | STCG-v1 Weighted Ridge | 5 | 0.046 +/- 0.000 | 0.016 +/- 0.000 | 0.039 +/- 0.000 | 0.069 +/- 0.000 |
| UCIHARWindowSignals | Persistence | 5 | 0.852 +/- 0.000 | 0.402 +/- 0.000 | 0.771 +/- 0.000 | 1.165 +/- 0.000 |
| UCIHARWindowSignals | Dense Ridge | 5 | 0.680 +/- 0.000 | 0.322 +/- 0.000 | 0.629 +/- 0.000 | 0.915 +/- 0.000 |
| UCIHARWindowSignals | RandomGraph Ridge | 5 | 0.843 +/- 0.019 | 0.505 +/- 0.020 | 0.833 +/- 0.019 | 1.039 +/- 0.005 |
| UCIHARWindowSignals | DynVAR-Lasso Graph Ridge | 5 | 0.811 +/- 0.000 | 0.479 +/- 0.000 | 0.802 +/- 0.000 | 0.990 +/- 0.000 |
| UCIHARWindowSignals | DynVAR-Lasso Weighted Ridge | 5 | 0.680 +/- 0.000 | 0.322 +/- 0.000 | 0.629 +/- 0.000 | 0.915 +/- 0.000 |
| UCIHARWindowSignals | STCG-v1 Graph Ridge | 5 | 0.811 +/- 0.000 | 0.479 +/- 0.000 | 0.802 +/- 0.000 | 0.990 +/- 0.000 |
| UCIHARWindowSignals | STCG-v1 Weighted Ridge | 5 | 0.680 +/- 0.000 | 0.322 +/- 0.000 | 0.629 +/- 0.000 | 0.915 +/- 0.000 |
