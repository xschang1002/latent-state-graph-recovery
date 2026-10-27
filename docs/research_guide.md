# Research guide

## Question and scope

When does sharing local graph estimates improve recovery of edges at the times
they are active? The study separates pooled structural ranking,
within-changing-edge temporal state discrimination and adjacent-window
boundary detection. A gain on one target need not establish a gain on another.

The mathematical results concern finite score/label arrays; the two-emission
factorisation additionally assumes two states and two emitted maps. The empirical
results concern the specified linear switching systems, local linear scorers
and aggregation rules. Known ranking principles motivate the controls; the
diagnostic identities and controlled comparisons show which interpretations
survive them. Gate behaviour is an implementation-specific case.

Start with the [main article](../paper/main.pdf)
and [supplement](../paper/supplement.pdf).
Their sources are [main.tex](../paper/main.tex) and
[supplement.tex](../paper/supplement.tex). In the accompanying archive the PDFs
are `paper/main.pdf` and `paper/supplement.pdf`.

## Current supplement map

This map follows the scientific organisation of the 27 October narrative
version. Labels identify the corresponding TeX locations; detailed table
numbers should be read from the current PDF.

| Section | Scientific content | Source label |
|---|---|---|
| S1 | Targets, notation and output definitions | `supp:targets` |
| S2 | Generators, local estimation and aggregation procedures | `supp:procedures` |
| S3 | Ranking and risk derivations, diagnostic identities and implementation-specific gate geometry | `supp:derivations` |
| S4 | Designs and results for Studies A, B and C | `supp:studies-abc` |
| S5 | Emission controls, boundary definitions, both AP conventions, threshold sensitivity and residual pair accounting | `supp:emission-sensitivity` |
| S6 | Additional comparisons and the PELT segment-refit adapter | `supp:additional` |
| S7 | ESS static expert alignment and degree-null analysis | `supp:industrial` |
| S8 | Implementation, generator-output diagnostics, selected fresh-fit checks and reproduction | `supp:implementation` |

## Read the evidence by question

| Question | Evidence and outputs | What the evidence supports |
|---|---|---|
| Can pooled ranking and temporal discrimination disagree? | A/B paired outputs; `duration_confirmation/` and `schedule_robustness/`; S3–S4 | The two targets can favour different emissions on the same local scores. Pooled AUROC does not alone identify temporal recovery. |
| Under which tested conditions does hard output improve on raw? | A's 100 independent paired repeats; B's 50 different repeats across six schedule cells; S4 | Contrasts at selected duration configurations. Duration also changes switch coverage; the results do not give a universal duration rule. A's primary Holm4 and B's secondary families retain their specified scope. |
| Does the contrast depend on the generator or window grid? | C's separately specified 30 new independent repeats over two durations and five window/stride settings; `symmetric_sensitivity/`; S4 | Robustness within the same broad linear framework. Generator features change together; this is not a factorial attribution of each change or an extension of the A/B test families. |
| Does pooled gain require inferred states, and what does hard compression discard? | `score_preserving_controls/`, `symmetric_controls/`; S5 | Additional fixed-weight shrinkage/residual and raw-fallback comparisons reuse fitted scores. Near-zero residual AUROC gains can arise from breaking within-group ties, without better state assignment. |
| Are state discrimination and boundary detection the same? | `temporal_localisation/`, `symmetric_localisation/`, `boundary_sensitivity/`; S5 | No. Both standard AP and positive-change AP are reported, as are all fixed raw-shared/method-own quantiles. Reference changes can change boundary assessments; method-own quantiles are not independent false-alarm calibration. |
| Does the distinction apply beyond recurrent clustering? | `segmentation_validation/`; S6 | One PELT plus non-overlapping segment VAR-Lasso adapter improves pooled ranking without a detected within-edge temporal AUROC advantage. The intervals do not establish equivalence. It shares the local linear features and does not establish generality over all methods. |
| What has been reproduced from scratch? | Eight original-seed A/C pipeline cases; `segmentation_validation/fresh/` and `fresh_comparisons.csv`; S8 | All 492 recorded comparisons are exact in the recorded environment. This is a selected computational check, not new independent seeds or a fresh full-grid run. |
| What does ESS validate? | ESS inputs/results; S7 | Alignment with a coarse static expert graph. It cannot validate changing-edge timing. Finite-chain degree-null qualifications and the distinction from between-method tests remain explicit. |

Result directories in this table are under `results/information_sciences/`.
Windows, edges, method rows and fitted cells are not independent repeats.
Undefined targets and failures retain their reported denominators. Post hoc
controls/sensitivity use pointwise paired-repeat intervals and do not enlarge
the original confirmatory families.

## Output policies

**Raw window** retains the local score sequence. **Forced hard** always emits
the candidate hard-group output. **Gated hard (static-max fallback)** uses that candidate on
acceptance and the edgewise maximum over raw windows, repeated through time,
on rejection: its fallback is static max. **Raw fallback** instead preserves the
raw sequence on rejection. These policies share the gate decision but differ
in what happens after rejection. All-case effects, acceptance rates and
accepted-case effects should be read separately. Oracle labels supply a
diagnostic control only in simulations with known states.

`STCG`, `STCG-v1` and `src/stcg/` are retained implementation identifiers.
They do not change the paper's positioning as a study of evaluation and
conditional aggregation effects.

## Reproduction and historical records

The [reproduction guide](information_sciences_artifact_readme.md) gives exact
commands for read-only verification, display rebuilding, analysis regeneration
and fresh fitting. Writing routes use separate output and execution-record
directories. The [script index](../scripts/README.md) maps entry points to
scientific tasks. [reproduction_manifest.md](reproduction_manifest.md) preserves
the longer command history after its current-stage introduction.

The following records retain their original dates, claims, checksums and section
numbering. Their section/table numbers describe those versions and are
superseded as current reading locations by the map above. They remain available
for tracing development and execution; their existence is not a new scientific
finding or public preregistration.

- [Original A/B evidence index](information_sciences_evidence_index_2026-10-24.md)
  and [public aggregation-literature notes](information_sciences_aggregation_literature_2026-10-24.md).
- [Emission-controls/C evidence index](reviewer_controls_evidence_index_2026-10-25.md)
  and [dated response record](reviewer_response_2026-10-25.md).
- [AP/threshold/comparator evidence index](evaluation_revision_evidence_index_2026-10-25.md),
  [dated response](evaluation_revision_response_2026-10-25.md) and
  [public-source/bibliography audit](evaluation_sensitivity_literature_2026-10-25.md).
- [Earlier audit records](../audit/) and the normalized execution exports under
  `results/information_sciences/provenance/`, `reviewer_provenance/` and
  `evaluation_provenance/`.

The prior narrative presentation checkpoint is recorded in
[narrative_cleanup_2026-10-27.md](narrative_cleanup_2026-10-27.md).
The same ZIP filename has been rebuilt across versions; use its actual manifest
and file hash to identify a delivered copy. Public deposition, persistent
identifiers and journal upload belong to the
[submission/release checklist](../submission_materials/information_sciences/README.md).
