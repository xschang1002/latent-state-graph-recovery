# Specialist stress extension: frozen new seeds 100--119

15 displayed settings, 13 unique inputs per seed; fixed method budgets and supplied K=2. Every axis is reported. Pairwise tests use the same complete-case seeds across all five methods; coverage records expose failures and unfinished jobs. Missing primary tests count as p=1 in Holm families.

Stress-phase time ratios were collected under concurrent load and are descriptive. Only the separately scheduled timing phase supports a matched quiet-machine latency report. There is no equivalence margin, post-hoc practical-gain cutoff, or winner selection.

| Axis | Value | Common n | STCG Edge AUC | Regime-PCMCI | Difference [95% CI] | Two-sided t Holm(13) | Two-sided W Holm(13) |
|---|---:|---:|---:|---:|---|---:|---:|
| obs_noise | 0.0 | 20 | 0.8274 | 0.8394 | -0.0121 [-0.0248, +0.0022] | 0.5706 | 0.1225 |
| obs_noise | 0.05 | 20 | 0.8273 | 0.8380 | -0.0108 [-0.0218, +0.0010] | 0.5706 | 0.2643 |
| obs_noise | 0.1 | 20 | 0.8267 | 0.8430 | -0.0163 [-0.0269, -0.0051] | 0.08172 | 0.1225 |
| obs_noise | 0.2 | 20 | 0.8206 | 0.8303 | -0.0096 [-0.0198, +0.0015] | 0.5706 | 0.4216 |
| timesteps | 300 | 20 | 0.6574 | 0.6175 | +0.0398 [+0.0131, +0.0666] | 0.08172 | 0.1087 |
| timesteps | 600 | 20 | 0.7632 | 0.7594 | +0.0038 [-0.0082, +0.0163] | 0.726 | 0.996 |
| timesteps | 1000 | 20 | 0.8274 | 0.8394 | -0.0121 [-0.0248, +0.0022] | 0.5706 | 0.1225 |
| timesteps | 1500 | 20 | 0.8519 | 0.8759 | -0.0240 [-0.0342, -0.0135] | 0.002741 | 0.004343 |
| edge_prob | 0.1 | 20 | 0.8283 | 0.8155 | +0.0127 [-0.0127, +0.0401] | 0.726 | 0.996 |
| edge_prob | 0.18 | 20 | 0.8274 | 0.8394 | -0.0121 [-0.0248, +0.0022] | 0.5706 | 0.1225 |
| edge_prob | 0.3 | 20 | 0.8017 | 0.8163 | -0.0147 [-0.0237, -0.0053] | 0.06567 | 0.09436 |
| edge_prob | 0.45 | 20 | 0.7650 | 0.7726 | -0.0075 [-0.0188, +0.0036] | 0.6597 | 0.8863 |
| switch_period | 150 | 20 | 0.7415 | 0.8493 | -0.1078 [-0.1231, -0.0909] | 1.145e-09 | 2.48e-05 |
| switch_period | 300 | 20 | 0.8152 | 0.8526 | -0.0375 [-0.0498, -0.0233] | 0.0004588 | 0.003868 |
| switch_period | 600 | 20 | 0.8342 | 0.8480 | -0.0138 [-0.0294, +0.0046] | 0.5706 | 0.2658 |

## cMLP and window/global comparators

All 4 opponents x 5 recovery metrics appear in the paired CSV. The 52-test Holm columns cover all Edge-AUC opponents and unique settings. Secondary metrics are exploratory; sign-reversed SHD improvements mean fewer errors.

## Coverage and completion

Stored fits: 1400; successful: 1400. Frozen full plan: 1300 stress fits and 100 separate timing fits.

Use the coverage CSV's planned, attempted, successful, failed, common_n and failed_seeds columns before interpreting any effect. A failed specialist is not a performance win.


## CPU-clock resolution amendment

Windows process CPU clocks can round very short fits to zero. These raw observations are retained. A geometric timing ratio that includes a zero duration is explicitly unavailable, with zero-clock counts in the timing CSV. No positive duration is imputed and no zero pair is dropped. Positive wall-time ratios and all recovery statistics are unchanged. Use `tools/summarize_revision_specialist.py` for this analysis-only amendment; estimation functions and the original protocol remain unchanged; any recording-only runner revision is separately logged.

## Undefined-metric recording amendment

At high edge density some truths contain no negative directed pairs, so Pair AUC is mathematically undefined. Successful fits retain status=ok and null for that metric. Each metric uses its own finite common seeds across all five methods, with defined counts and excluded undefined-metric seeds recorded. An undefined Pair AUC never removes an otherwise valid Edge-AUC observation. Original serialization failures and interrupted jobs are retained in the storage amendment/history; estimator, generator, grid and statistical-test functions were checked by AST hash to remain unchanged.
