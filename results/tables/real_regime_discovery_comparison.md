# Real Regime Discovery Comparison

Head-to-head regime recovery on real datasets with ground-truth operating-mode labels.
All methods see the SAME capped window per config; RPCMCI/MSVAR receive the true regime
count (oracle-sighted). Window-level metrics follow the tracked state-label alignment
protocol (Hungarian aligned accuracy + ARI over window-majority labels); sample-level
metrics follow the synthetic regime-quality path (per-sample, from each method's
effective offset). STCG states are recomputed on the capped window with the forward
harness machinery (lasso window features + KMeans, window 300 / stride 150).

| Config | Method | N | K (present) | T | Windows | Window ARI | Window Acc | Sample ARI | Sample Acc | Fit s |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| AReMActivities_bending_transition | MSVAR | 2 | 2 (2) | 1500 | 9 | 0.000 +/- 0.000 | 0.556 +/- 0.000 | 0.002 +/- 0.003 | 0.517 +/- 0.020 | 259.9 |
| AReMActivities_bending_transition | Regime-PCMCI | 2 | 2 (2) | 1500 | 9 | 0.000 +/- 0.000 | 0.556 +/- 0.000 | 0.002 +/- 0.002 | 0.524 +/- 0.011 | 78.6 |
| AReMActivities_bending_transition | STCG-v1 | 5 | 2 (2) | 1500 | 9 | 0.241 +/- 0.000 | 0.778 +/- 0.000 | 0.160 +/- 0.000 | 0.700 +/- 0.000 | 0.0 |
| AReMActivities_concat4 | Regime-PCMCI | 2 | 4 (4) | 1500 | 9 | 0.571 +/- 0.000 | 0.778 +/- 0.000 | 0.533 +/- 0.000 | 0.709 +/- 0.000 | 96.7 |
| AReMActivities_concat4 | STCG-v1 | 5 | 4 (4) | 1500 | 9 | 0.094 +/- 0.053 | 0.556 +/- 0.000 | 0.380 +/- 0.041 | 0.631 +/- 0.046 | 0.0 |
| HydraulicSystems_cooler_transition | Regime-PCMCI | 2 | 3 (2) | 1500 | 9 | 0.000 +/- 0.000 | 0.556 +/- 0.000 | 0.017 +/- 0.000 | 0.568 +/- 0.000 | 60.4 |
| HydraulicSystems_cooler_transition | STCG-v1 | 5 | 3 (2) | 1500 | 9 | 0.769 +/- 0.000 | 0.778 +/- 0.000 | 0.612 +/- 0.000 | 0.751 +/- 0.000 | 0.0 |
| HydraulicSystems_cooler_transition_K2present | Regime-PCMCI | 2 | 2 (2) | 1500 | 9 | 0.000 +/- 0.000 | 0.556 +/- 0.000 | 0.000 +/- 0.000 | 0.507 +/- 0.000 | 60.0 |
| HydraulicSystems_cooler_transition_K2present | STCG-v1 | 5 | 2 (2) | 1500 | 9 | 0.822 +/- 0.243 | 0.956 +/- 0.061 | 0.684 +/- 0.176 | 0.911 +/- 0.055 | 0.1 |
| HydraulicSystems_valve_transition | STCG-v1 | 5 | 4 (3) | 1500 | 9 | 0.171 +/- 0.000 | 0.778 +/- 0.000 | 0.519 +/- 0.000 | 0.901 +/- 0.000 | 0.0 |
| HydraulicSystems_valve_transition_K3present | STCG-v1 | 5 | 3 (3) | 1500 | 9 | 0.282 +/- 0.131 | 0.844 +/- 0.061 | 0.490 +/- 0.142 | 0.711 +/- 0.109 | 0.0 |

## Protocol notes

- Cap protocol per config is recorded in the `cap_description` column of the CSV.
- RPCMCI budget matches the synthetic runner: num_iterations=20, max_anneal=10,
  pc_alpha=0.05, alpha_level=0.01, tau_max=3, max_transitions=10, switch_thres=0.05.
- MSVAR budget where a fit completed: maxiter=100, retry_maxiter=200, search_reps=1
  (trimmed from the synthetic 200/500/3 so a fit finishes inside the per-call
  process budget).
- Rows whose fit failed stay in the gitignored partials and are excluded here;
  `fit_seconds=295` on a failed MSVAR partial means the process was killed at the
  per-call budget, not that the fit finished.

## Failed fits (excluded from the aggregates above)

| Config | Method | Seed | Fit s | Recorded cause |
|---|---|---:|---:|---|
| HydraulicSystems_cooler_transition | MSVAR | 0 | 295.0 | killed at per-call budget of 295s (fit did not finish) |
| HydraulicSystems_cooler_transition | MSVAR | 1 | 295.0 | killed at per-call budget of 295s (fit did not finish) |
| HydraulicSystems_valve_transition | Regime-PCMCI | 0 | 63.3 | no RPCMCI annealing converged (tigramite `run_rpcmci` returned None) |
| HydraulicSystems_valve_transition | Regime-PCMCI | 1 | 64.7 | no RPCMCI annealing converged (tigramite `run_rpcmci` returned None) |
| HydraulicSystems_valve_transition_K3present | Regime-PCMCI | 0 | 47.8 | no RPCMCI annealing converged (tigramite `run_rpcmci` returned None) |
| HydraulicSystems_valve_transition_K3present | Regime-PCMCI | 1 | 47.5 | no RPCMCI annealing converged (tigramite `run_rpcmci` returned None) |

## Not-run plan combos

Aggregated with `--aggregate-partial`; these planned combos have no partial:

- `AReMActivities_concat4/MSVAR/seed0`
- `AReMActivities_concat4/MSVAR/seed1`
- `HydraulicSystems_cooler_transition_K2present/MSVAR/seed0`
- `HydraulicSystems_cooler_transition_K2present/MSVAR/seed1`
- `HydraulicSystems_valve_transition/MSVAR/seed0`
- `HydraulicSystems_valve_transition/MSVAR/seed1`
- `HydraulicSystems_valve_transition_K3present/MSVAR/seed0`
- `HydraulicSystems_valve_transition_K3present/MSVAR/seed1`
