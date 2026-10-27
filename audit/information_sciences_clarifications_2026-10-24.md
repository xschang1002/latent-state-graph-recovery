# Information Sciences clarification audit — 2026-10-24

This stage clarifies the existing study rather than changing its generator or
running another experiment. It preserves prior working-tree changes, including
the existing staged deletion, and makes no commit, public deposit or submission.
The preceding research-stage audit and dated PDFs remain as historical records.

## Generator and retained-output diagnostics

State B additions are drawn only at positions absent from the original A
support. An A edge deleted by the retention draw cannot be added back. Ignoring
the empty-tensor fallbacks, the nonself marginal is
`0.18*0.5 + (1-0.18)*0.09 = 0.1638`. The parameter 0.5 is a conditional
retention probability, not a fixed realised overlap. State A clears every
diagonal; B additions can include those originally absent self positions.
If all of A is empty, one lag-1 edge from a uniformly sampled node to its
cyclic successor is forced. If the entire B proposal is empty, A is copied.
Neither fallback is compatible with any of the 150 retained mechanisms.
The generator, scorer, frozen protocols and all 1,915 study files are unchanged.

A one-pass scaling of VAR coefficient blocks is not a scalar scaling of the
whole VAR(3) companion matrix. The diagnostic recomputes the companion radius
from retained final coefficients, without inferring pre-rescale radii or how
often scaling occurred:

| Study | Independent mechanisms | Saved observed paths | Final radius range | Largest absolute value | Late/early RMS range |
|---|---:|---:|---:|---:|---:|
| A | 100 | 200 | [0.380546, 0.675987] | 0.532854 | [0.930770, 1.050883] |
| B | 50 | 300 | [0.403785, 0.627269] | 0.555313 | [0.936157, 1.063834] |

All 500 saved observed paths are finite. RMS pools every node and the first
or last 300 saved observed samples. These endpoint ratios are post hoc
descriptive diagnostics, not fitted growth rates or prespecified stability
tests. The unsaved 100-sample burn-in is not reconstructed by this audit;
the original simulator's full-trajectory finiteness guard is separately
identified. Final radii below one do not by themselves establish stability
of arbitrary switching products. No general stability guarantee is claimed.
Section S14.1 and Table S23 give the definitions and coverage. Machine-readable
mechanism/path rows and before/after hashes are in
`results/information_sciences/generator_audit/`.

## Interpretation, definitions and evidence access

The motivation now connects post-estimation grouping to the local-connectivity
workflows of Allen et al. (2014), Shakil et al. (2016), Abrol et al. (2017)
and Chang et al. (2026), using primary full texts. These studies concern
functional association and use different summaries; several already validate
state transitions or temporal summaries. The manuscript does not claim that
temporal validation or duration sensitivity is new. Its narrower comparison
holds local scores fixed and studies paired aggregation-minus-raw changing-edge
ranking. The finite-score identities and the two-state/two-map factorisation
are distinguished from empirical results on the supplied prototype. The gate
remains a case of one implementation choice.

Definitions now cover the window count M (legacy W denotes the same count),
positive/negative window counts in the temporal AUROC, uniform integer random
phase, candidate-label temporal coherence, and both empty-support fallbacks.
Table 2 warns that paired differences are calculated before display rounding:
the actual D=300 native and raw means are 0.6385365723 and 0.7074056527,
with paired difference -0.0688690804, correctly displayed as -0.069.
The historical mechanism figure now says configured support retention rather
than implying realised overlap; only its axis/title and caption changed.

Vague supplement pointers are replaced by section, table and figure locators.
The reviewer entry `docs/information_sciences_evidence_index_2026-10-24.md`
maps Study A seeds 50000–50099 and Study B seeds 60000–60049 to protocols,
execution entry points, per-repeat arrays, summaries and verification commands.
The package includes eight byte-preserved exported internal records and machine
indices in `results/information_sciences/provenance/`. The chronology separates
internal launcher clocks, post hoc file times and subsequent verification.
There is no independent external timestamp or public preregistration. Study B
has a protocol-only prelaunch record, not a separately retained freeze-event
timestamp. No stronger timing record is invented.

## Validation and deliverables

All 353 regression tests pass, with three existing SWIG deprecation warnings.
All 237 historical number checks pass. The combined verifier checks the new
diagnostic computations, 12 frozen sources, 1,915 study files, eight exported
records, 17 manuscript locators and the existing A/B analysis/display checks.
A further independent retained-array check verifies 700 candidate groupings
and 300 B schedules against the stated coherence, window and phase definitions.
The new diagnostic tool performs no generation or fitting. The existing Study A
verifier deterministically replays saved stochastic inputs to check cached
observations; this is distinct from drawing new experimental data.

Both PDFs compile with zero overfull boxes or undefined references/citations,
and all fonts embedded. All 24 main and 36 supplement pages were rendered and
inspected, with individual checks of the generator, rounding note, execution
records and new diagnostic table. Known nonfatal compiler cache/font/BibTeX
notices remain; compilation is not described as warning-free.

The deliverables are the dated `information_sciences_clarified` main and
supplement PDFs and `submission_package/information_sciences_reproducibility.zip`.
The tested archive contains 4,019 payloads plus its manifest, including actual
inputs, outputs, source, protocols, internal records and reproduction routes.
Two fresh extractions passed `--verify` (73.759 seconds)
and `--rebuild` (109.455 seconds), using the existing
isolated Python 3.12 environment with all 47 pinned versions and `pip check`
confirmed. Verification leaves all 4,020 delivered files unchanged. Rebuilding
preserves all 27 regenerated tables, all 62 delivered table files, 1,915 frozen
study files, 12 scientific sources and all 239 prior result CSVs. Four main
figure PDFs and the corrected mechanism PNG are byte-identical; any changed
legacy PDF retains identical drawing streams and page boxes. The clean
commands introduce no new experimental fitting. The final export updates only
this completed audit and its JSON from the tested candidate; its current-byte
and candidate-difference checks are kept in an external execution record.
Public repository/DOI allocation remains outside this local preparation. The
ZIP checksum stays outside its own contents to avoid a self-referential digest.

The figure-label workflow used the scientific-visualization skill, with the
existing asset provenance in `audit/information_sciences_assets_2026-10-24.md`.
Required workflow attribution: Kassis et al. (2026), *Scientific Agent Skills*,
[DOI:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
Its current arXiv record was checked; this is not a manuscript research source.
