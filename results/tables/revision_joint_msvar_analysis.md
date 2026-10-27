# Corrective shared-chain joint MSVAR comparison

Post-finding corrective comparison, frozen before its fit results. Historical MSAR-proxy rows and the C7 grid/Holm families are unchanged. Exactly the C7 base input hashes, seeds 100–119, are used. The new model has direct VAR(3) means, a single shared Markov chain, and full regime covariance.

Coverage: {'planned': 20, 'completed': 20, 'ok': 20, 'nonconverged': 0, 'failed': 0, 'timeout': 0, 'crashed': 0, 'unfinished_seeds': []}

Ten starts per seed, 200 EM iterations per start, relative likelihood tolerance 1e-6. Primary rows require convergence of the highest-likelihood selected start; all finite budget fits are separately shown. Max-iteration and failed starts are never imputed as chance recovery.

| Population | n | STCG Edge AUC | Joint MSVAR Edge AUC | STCG minus joint [95% CI] | Two-sided t p | Wilcoxon p |
|---|---:|---:|---:|---|---:|---:|
| selected_converged | 20 | 0.827373 | 0.816502 | 0.010871 [-0.017448, 0.046272] | 0.526681 | 0.430433 |
| all_finite_budget_fits | 20 | 0.827373 | 0.816502 | 0.010871 [-0.017448, 0.046272] | 0.526681 | 0.430433 |

One corrective primary Edge AUC contrast; four other recovery metrics and the finite-fit population are sensitivity evidence. No equivalence or non-inferiority claim is available. Runtime records here were collected during other work and do not establish a comparative speed ratio.

Source and protocol hashes: `revision_joint_msvar_protocol.json`; full likelihood histories and all starts: `revision_joint_msvar_starts.csv`; fitted matrices/posteriors: `revision_joint_msvar_parts/*.npz`.
