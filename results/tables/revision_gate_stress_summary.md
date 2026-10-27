# Independent gate and selector stress results

Frozen protocol: `docs/revision_gate_protocol_2026-09-13.md`.
Seeds 100–119; all stated thresholds and candidate sets are fixed. Controlled fixtures are shown separately from generator evidence.
Core STCG gate thresholds are unchanged. Paired differences use 10000 paired bootstrap draws; intervals are descriptive, not adjusted hypothesis tests.

The original full and factor-removal selectors explicitly use legacy selection. Additional `positive_fallback` rows apply the post-finding all-nonpositive-score logic fix and are not part of the prospectively frozen factor experiment. Core STCG partition-gate defaults remain unchanged.

## generator: default partition gate

| Condition | N | Accept | Primary | Fallback | False split | Edge AUC | Window ARI |
|---|---:|---:|---:|---:|---:|---:|---:|
| stationary_many | 20 | 0.000 | 0.000 | 0.000 | 0.000 | 0.943 | 1.000 |
| stationary_few | 20 | 1.000 | 1.000 | 0.000 | 1.000 | 0.886 | 0.000 |
| strong_switch_long | 20 | 1.000 | 0.750 | 0.250 | — | 0.957 | 1.000 |
| weak_switch_long | 20 | 0.100 | 0.000 | 0.100 | — | 0.783 | 0.066 |
| mixed_periodic | 20 | 0.000 | 0.000 | 0.000 | — | 0.803 | 1.000 |
| short_switch | 20 | 1.000 | 1.000 | 0.000 | — | 0.741 | 0.684 |
| measurement_noise_switch | 20 | 0.000 | 0.000 | 0.000 | — | 0.599 | 0.000 |
| imbalanced_switch | 20 | 0.050 | 0.000 | 0.050 | — | 0.835 | 0.011 |

## generator: full selectors

| Condition | Objective | Mean K | All scores zero | False split | Edge AUC | Window ARI |
|---|---|---:|---:|---:|---:|---:|
| stationary_many | stable | 2.550 | 0.950 | 1.000 | 0.970 | 0.000 |
| stationary_many | warning | 3.650 | 0.000 | 1.000 | 0.964 | 0.000 |
| stationary_few | stable | 2.300 | 0.350 | 1.000 | 0.886 | 0.000 |
| stationary_few | warning | 3.450 | 0.000 | 1.000 | 0.882 | 0.000 |
| strong_switch_long | stable | 2.000 | 0.000 | — | 0.957 | 1.000 |
| strong_switch_long | warning | 2.400 | 0.000 | — | 0.956 | 0.899 |
| weak_switch_long | stable | 2.550 | 0.850 | — | 0.807 | 0.214 |
| weak_switch_long | warning | 3.500 | 0.000 | — | 0.816 | 0.163 |
| mixed_periodic | stable | 2.650 | 1.000 | — | 0.849 | 0.000 |
| mixed_periodic | warning | 3.350 | 0.000 | — | 0.851 | 0.000 |
| short_switch | stable | 2.000 | 0.000 | — | 0.741 | 0.684 |
| short_switch | warning | 3.250 | 0.000 | — | 0.736 | 0.343 |
| measurement_noise_switch | stable | 2.750 | 1.000 | — | 0.612 | 0.018 |
| measurement_noise_switch | warning | 3.450 | 0.000 | — | 0.620 | 0.020 |
| imbalanced_switch | stable | 2.350 | 0.700 | — | 0.875 | 0.494 |
| imbalanced_switch | warning | 3.500 | 0.050 | — | 0.881 | 0.161 |

## generator: interventions that change selected/accepted counts

| Condition | Analysis/objective | Removal | Metric | Changed seeds | Mean difference | 95% interval |
|---|---|---|---|---:|---:|---|
| stationary_many | gate/partition | always_accept | states_accepted | 20/20 | 1.000 | [1.000, 1.000] |
| stationary_many | selector/stable | no_coherence_screen | state_k | 14/20 | 1.100 | [0.750, 1.450] |
| stationary_many | selector/stable | no_coherence_all | state_k | 16/20 | 1.050 | [0.600, 1.500] |
| stationary_many | selector/stable | positive_fallback | state_k | 19/20 | -1.450 | [-1.900, -1.050] |
| stationary_many | selector/warning | no_balance | state_k | 6/20 | 0.300 | [0.100, 0.500] |
| stationary_many | selector/warning | no_parsimony | state_k | 3/20 | 0.150 | [0.000, 0.300] |
| stationary_many | selector/warning | no_coherence_weight | state_k | 1/20 | -0.050 | [-0.150, 0.000] |
| stationary_few | gate/partition | always_reject | states_accepted | 20/20 | -1.000 | [-1.000, -1.000] |
| stationary_few | gate/partition | no_primary | states_accepted | 7/20 | -0.350 | [-0.550, -0.150] |
| stationary_few | selector/stable | no_coherence_screen | state_k | 7/20 | 0.400 | [0.150, 0.650] |
| stationary_few | selector/stable | no_coherence_all | state_k | 16/20 | 1.150 | [0.850, 1.450] |
| stationary_few | selector/stable | positive_fallback | state_k | 7/20 | -0.650 | [-1.050, -0.250] |
| stationary_few | selector/warning | no_balance | state_k | 11/20 | 0.550 | [0.350, 0.750] |
| stationary_few | selector/warning | no_parsimony | state_k | 10/20 | 0.500 | [0.300, 0.700] |
| strong_switch_long | gate/partition | always_reject | states_accepted | 20/20 | -1.000 | [-1.000, -1.000] |
| strong_switch_long | gate/partition | no_fallback | states_accepted | 5/20 | -0.250 | [-0.450, -0.100] |
| strong_switch_long | selector/warning | no_balance | state_k | 16/20 | 1.400 | [1.050, 1.750] |
| strong_switch_long | selector/warning | no_parsimony | state_k | 8/20 | 0.700 | [0.300, 1.100] |
| strong_switch_long | selector/warning | no_coherence_weight | state_k | 4/20 | -0.400 | [-0.800, -0.100] |
| weak_switch_long | gate/partition | always_accept | states_accepted | 18/20 | 0.900 | [0.750, 1.000] |
| weak_switch_long | gate/partition | always_reject | states_accepted | 2/20 | -0.100 | [-0.250, 0.000] |
| weak_switch_long | gate/partition | no_fallback | states_accepted | 2/20 | -0.100 | [-0.250, 0.000] |
| weak_switch_long | gate/partition | no_coherence | states_accepted | 1/20 | 0.050 | [0.000, 0.150] |
| weak_switch_long | selector/stable | no_balance | state_k | 1/20 | 0.050 | [0.000, 0.150] |
| weak_switch_long | selector/stable | no_coherence_screen | state_k | 12/20 | 0.750 | [0.350, 1.150] |
| weak_switch_long | selector/stable | no_coherence_all | state_k | 14/20 | 0.750 | [0.350, 1.150] |
| weak_switch_long | selector/stable | positive_fallback | state_k | 17/20 | -1.350 | [-1.750, -1.000] |
| weak_switch_long | selector/warning | no_balance | state_k | 9/20 | 0.500 | [0.250, 0.750] |
| weak_switch_long | selector/warning | no_parsimony | state_k | 2/20 | 0.150 | [0.000, 0.400] |
| weak_switch_long | selector/warning | no_coherence_weight | state_k | 3/20 | -0.200 | [-0.450, 0.000] |
| mixed_periodic | gate/partition | always_accept | states_accepted | 20/20 | 1.000 | [1.000, 1.000] |
| mixed_periodic | selector/stable | no_coherence_screen | state_k | 14/20 | 0.750 | [0.350, 1.150] |
| mixed_periodic | selector/stable | no_coherence_all | state_k | 16/20 | 0.700 | [0.250, 1.100] |
| mixed_periodic | selector/stable | positive_fallback | state_k | 20/20 | -1.650 | [-2.000, -1.350] |
| mixed_periodic | selector/warning | no_balance | state_k | 13/20 | 0.650 | [0.450, 0.850] |
| mixed_periodic | selector/warning | no_parsimony | state_k | 1/20 | 0.050 | [0.000, 0.150] |
| short_switch | gate/partition | always_reject | states_accepted | 20/20 | -1.000 | [-1.000, -1.000] |
| short_switch | selector/stable | no_coherence_screen | state_k | 11/20 | 0.550 | [0.350, 0.750] |
| short_switch | selector/stable | no_coherence_all | state_k | 20/20 | 1.250 | [1.100, 1.450] |
| short_switch | selector/warning | no_balance | state_k | 15/20 | 0.750 | [0.550, 0.900] |
| short_switch | selector/warning | no_parsimony | state_k | 13/20 | 0.650 | [0.450, 0.850] |
| measurement_noise_switch | gate/partition | always_accept | states_accepted | 20/20 | 1.000 | [1.000, 1.000] |
| measurement_noise_switch | selector/stable | no_coherence_screen | state_k | 15/20 | 0.600 | [0.200, 1.000] |
| measurement_noise_switch | selector/stable | no_coherence_all | state_k | 18/20 | 0.700 | [0.250, 1.150] |
| measurement_noise_switch | selector/stable | positive_fallback | state_k | 20/20 | -1.750 | [-2.100, -1.450] |
| measurement_noise_switch | selector/warning | no_balance | state_k | 7/20 | 0.350 | [0.150, 0.550] |
| imbalanced_switch | gate/partition | always_accept | states_accepted | 19/20 | 0.950 | [0.850, 1.000] |
| imbalanced_switch | gate/partition | always_reject | states_accepted | 1/20 | -0.050 | [-0.150, 0.000] |
| imbalanced_switch | gate/partition | no_fallback | states_accepted | 1/20 | -0.050 | [-0.150, 0.000] |
| imbalanced_switch | gate/partition | no_coherence | states_accepted | 3/20 | 0.150 | [0.000, 0.300] |
| imbalanced_switch | gate/partition | no_min_share | states_accepted | 9/20 | 0.450 | [0.250, 0.650] |
| imbalanced_switch | selector/stable | no_coherence_screen | state_k | 11/20 | 0.850 | [0.500, 1.250] |
| imbalanced_switch | selector/stable | no_coherence_all | state_k | 15/20 | 1.100 | [0.750, 1.450] |
| imbalanced_switch | selector/stable | positive_fallback | state_k | 14/20 | -1.000 | [-1.400, -0.650] |
| imbalanced_switch | selector/warning | no_balance | state_k | 3/20 | 0.050 | [-0.100, 0.200] |
| imbalanced_switch | selector/warning | no_coherence_weight | state_k | 1/20 | -0.050 | [-0.150, 0.000] |
| imbalanced_switch | selector/warning | positive_fallback | state_k | 1/20 | -0.050 | [-0.150, 0.000] |

## generator: corrected positive-score fallback

| Condition | Objective | Mean K | All scores zero | False split | Edge AUC | Window ARI |
|---|---|---:|---:|---:|---:|---:|
| stationary_many | stable | 1.100 | 0.950 | 0.050 | 0.943 | 0.950 |
| stationary_many | warning | 3.650 | 0.000 | 1.000 | 0.964 | 0.000 |
| stationary_few | stable | 1.650 | 0.350 | 0.650 | 0.886 | 0.350 |
| stationary_few | warning | 3.450 | 0.000 | 1.000 | 0.882 | 0.000 |
| strong_switch_long | stable | 2.000 | 0.000 | — | 0.957 | 1.000 |
| strong_switch_long | warning | 2.400 | 0.000 | — | 0.956 | 0.899 |
| weak_switch_long | stable | 1.200 | 0.850 | — | 0.783 | 0.095 |
| weak_switch_long | warning | 3.500 | 0.000 | — | 0.816 | 0.163 |
| mixed_periodic | stable | 1.000 | 1.000 | — | 0.803 | 1.000 |
| mixed_periodic | warning | 3.350 | 0.000 | — | 0.851 | 0.000 |
| short_switch | stable | 2.000 | 0.000 | — | 0.741 | 0.684 |
| short_switch | warning | 3.250 | 0.000 | — | 0.736 | 0.343 |
| measurement_noise_switch | stable | 1.000 | 1.000 | — | 0.599 | 0.000 |
| measurement_noise_switch | warning | 3.450 | 0.000 | — | 0.620 | 0.020 |
| imbalanced_switch | stable | 1.350 | 0.700 | — | 0.844 | 0.164 |
| imbalanced_switch | warning | 3.450 | 0.050 | — | 0.879 | 0.111 |

## controlled_score_tensor: default partition gate

| Condition | N | Accept | Primary | Fallback | False split | Edge AUC | Window ARI |
|---|---:|---:|---:|---:|---:|---:|---:|
| coherent_weak | 20 | 1.000 | 0.000 | 1.000 | — | — | 1.000 |
| shuffled_weak | 20 | 0.000 | 0.000 | 0.000 | — | — | 0.000 |
| tiny_coherent | 20 | 0.000 | 0.000 | 0.000 | — | — | 0.000 |
| primary_incoherent | 20 | 1.000 | 1.000 | 0.000 | — | — | 1.000 |
| hierarchical_balanced | 20 | 1.000 | 1.000 | 0.000 | — | — | 0.494 |
| hierarchical_rare | 20 | 1.000 | 1.000 | 0.000 | — | — | 0.493 |

## controlled_score_tensor: full selectors

| Condition | Objective | Mean K | All scores zero | False split | Edge AUC | Window ARI |
|---|---|---:|---:|---:|---:|---:|
| coherent_weak | stable | 2.000 | 0.000 | — | — | 1.000 |
| coherent_weak | warning | 2.650 | 0.000 | — | — | 0.840 |
| shuffled_weak | stable | 1.750 | 1.000 | — | — | 0.628 |
| shuffled_weak | warning | 2.000 | 0.000 | — | — | 1.000 |
| tiny_coherent | stable | 2.000 | 0.000 | — | — | 1.000 |
| tiny_coherent | warning | 3.950 | 0.000 | — | — | 0.105 |
| primary_incoherent | stable | 2.200 | 1.000 | — | — | 0.648 |
| primary_incoherent | warning | 2.000 | 0.000 | — | — | 1.000 |
| hierarchical_balanced | stable | 2.000 | 0.000 | — | — | 0.494 |
| hierarchical_balanced | warning | 2.000 | 0.000 | — | — | 0.494 |
| hierarchical_rare | stable | 3.000 | 0.000 | — | — | 0.956 |
| hierarchical_rare | warning | 3.250 | 0.000 | — | — | 0.887 |

## controlled_score_tensor: interventions that change selected/accepted counts

| Condition | Analysis/objective | Removal | Metric | Changed seeds | Mean difference | 95% interval |
|---|---|---|---|---:|---:|---|
| coherent_weak | gate/partition | always_reject | states_accepted | 20/20 | -1.000 | [-1.000, -1.000] |
| coherent_weak | gate/partition | no_fallback | states_accepted | 20/20 | -1.000 | [-1.000, -1.000] |
| coherent_weak | selector/warning | no_balance | state_k | 14/20 | 1.250 | [0.850, 1.600] |
| coherent_weak | selector/warning | no_parsimony | state_k | 11/20 | 1.000 | [0.600, 1.400] |
| coherent_weak | selector/warning | no_coherence_weight | state_k | 7/20 | -0.650 | [-1.050, -0.300] |
| shuffled_weak | gate/partition | always_accept | states_accepted | 20/20 | 1.000 | [1.000, 1.000] |
| shuffled_weak | gate/partition | no_coherence | states_accepted | 20/20 | 1.000 | [1.000, 1.000] |
| shuffled_weak | selector/stable | no_coherence_all | state_k | 8/20 | 0.250 | [-0.050, 0.550] |
| shuffled_weak | selector/stable | positive_fallback | state_k | 13/20 | -0.750 | [-1.050, -0.450] |
| shuffled_weak | selector/warning | no_parsimony | state_k | 4/20 | 0.400 | [0.100, 0.800] |
| tiny_coherent | gate/partition | always_accept | states_accepted | 20/20 | 1.000 | [1.000, 1.000] |
| tiny_coherent | gate/partition | no_min_share | states_accepted | 20/20 | 1.000 | [1.000, 1.000] |
| tiny_coherent | selector/stable | no_coherence_all | state_k | 20/20 | 1.950 | [1.850, 2.000] |
| tiny_coherent | selector/warning | no_balance | state_k | 7/20 | -0.350 | [-0.550, -0.150] |
| primary_incoherent | gate/partition | always_reject | states_accepted | 20/20 | -1.000 | [-1.000, -1.000] |
| primary_incoherent | gate/partition | no_primary | states_accepted | 20/20 | -1.000 | [-1.000, -1.000] |
| primary_incoherent | selector/stable | no_coherence_screen | state_k | 3/20 | -0.150 | [-0.300, 0.000] |
| primary_incoherent | selector/stable | no_coherence_all | state_k | 11/20 | -0.200 | [-0.650, 0.200] |
| primary_incoherent | selector/stable | positive_fallback | state_k | 15/20 | -1.200 | [-1.650, -0.800] |
| primary_incoherent | selector/warning | no_parsimony | state_k | 1/20 | 0.100 | [0.000, 0.300] |
| hierarchical_balanced | gate/partition | always_reject | states_accepted | 20/20 | -1.000 | [-1.000, -1.000] |
| hierarchical_balanced | selector/stable | no_parsimony | state_k | 20/20 | 2.000 | [2.000, 2.000] |
| hierarchical_balanced | selector/warning | no_parsimony | state_k | 20/20 | 2.000 | [2.000, 2.000] |
| hierarchical_rare | gate/partition | always_reject | states_accepted | 20/20 | -1.000 | [-1.000, -1.000] |
| hierarchical_rare | selector/stable | no_coherence_weight | state_k | 1/20 | 0.050 | [0.000, 0.150] |
| hierarchical_rare | selector/stable | no_coherence_all | state_k | 1/20 | 0.050 | [0.000, 0.150] |
| hierarchical_rare | selector/warning | no_balance | state_k | 4/20 | 0.200 | [0.050, 0.400] |
| hierarchical_rare | selector/warning | no_coherence_weight | state_k | 4/20 | -0.200 | [-0.400, -0.050] |

## controlled_score_tensor: corrected positive-score fallback

| Condition | Objective | Mean K | All scores zero | False split | Edge AUC | Window ARI |
|---|---|---:|---:|---:|---:|---:|
| coherent_weak | stable | 2.000 | 0.000 | — | — | 1.000 |
| coherent_weak | warning | 2.650 | 0.000 | — | — | 0.840 |
| shuffled_weak | stable | 1.000 | 1.000 | — | — | 0.000 |
| shuffled_weak | warning | 2.000 | 0.000 | — | — | 1.000 |
| tiny_coherent | stable | 2.000 | 0.000 | — | — | 1.000 |
| tiny_coherent | warning | 3.950 | 0.000 | — | — | 0.105 |
| primary_incoherent | stable | 1.000 | 1.000 | — | — | 0.000 |
| primary_incoherent | warning | 2.000 | 0.000 | — | — | 1.000 |
| hierarchical_balanced | stable | 2.000 | 0.000 | — | — | 0.494 |
| hierarchical_balanced | warning | 2.000 | 0.000 | — | — | 0.494 |
| hierarchical_rare | stable | 3.000 | 0.000 | — | — | 0.956 |
| hierarchical_rare | warning | 3.250 | 0.000 | — | — | 0.887 |

Frozen legacy rows preserve the original `(score, coherence, -K)` tie-break when all scores are zero; those historical rows do not imply a single-state rejection. The post-finding `positive_fallback` rows use the current production default: return the fitted K=1 candidate when no score is positive.
In mixed_periodic, every window contains exactly equal portions of the two regimes (purity 0.500). The deterministic majority-label tie-break produces one window label; ARI=1 for a one-state prediction is therefore degenerate and is not evidence of recovering the two true regimes. The detail file records both the true time-series count and the window-label count.
False splitting is defined only on stationary generators; fixture graph-recovery metrics are absent because these fixtures have no empirical causal ground truth.
Complete per-seed decisions, candidates, summary/Wilson intervals and paired changes are in the accompanying CSV files.
Elapsed wall time (single BLAS/OpenMP thread): 32.03 seconds.
