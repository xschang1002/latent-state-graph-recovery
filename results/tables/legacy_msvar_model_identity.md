# Historical MSVAR identifier: model-identity correction

The archived `synthetic_msvar_*` observations, `MSVAR` method keys in merged
CSVs, historical real-segment results and the `stcg.msvar_scores` implementation
refer to an independent-equation statsmodels MarkovAutoregression with lagged
exogenous regressors. The corrected human-readable name is **MSAR-X (legacy)**.
Historical observations and identifiers are preserved rather than silently
substituted with results from a different model.

MarkovAutoregression applies its AR polynomial to the regression residual. When
the exogenous design itself contains lagged cross-series observations, this
introduces products of AR and exogenous coefficients and cross-lags up to 2L.
Consequently the stored exogenous beta coefficients are not the direct
coefficients of VAR(L). Posterior labels were aligned before averaging across
target equations, but such post-hoc averaging does not produce a joint model
with one shared Markov chain and transition matrix.

The old superiority and runtime interpretation against joint MSVAR is withdrawn.
These results remain a historical control, with corrected display labels in the
main manuscript tables and aggregate reports. The corrective shared-chain model
uses a separate implementation, frozen protocol, result names and independent
seeds; it must never be merged into an old `MSVAR` result row.

See `docs/review_response_assessment_2026-09-13.md` for the current corrective
comparison status. This model correction does not change Regime-PCMCI, cMLP,
STCG or the frozen specialist stress inputs and estimator settings.
