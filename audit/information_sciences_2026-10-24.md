# Information Sciences revision audit — 2026-10-24

This revision retains the evaluation/mechanism research line, adds the two
fixed paired studies, and reorganises the article for Information Sciences.
It preserves the prior working-tree changes and creates no Git commit.
The previous anonymous archive and dated PDFs remain intact. Exact study,
source, PDF and validation records accompany this report in the JSON file.

## Evidence and scope

Study A froze its configuration and source hashes before generating 100 new
independent mechanism/noise repeats. The two durations share stochastic inputs
within repeat; Lasso and prespecified auxiliary OLS share those same datasets.
All 400 condition/scorer fits succeed. The four primary Lasso comparisons use
two-sided repeat-level t tests with Holm4; percentile-bootstrap intervals have
pointwise 95% coverage. The protocol was locally frozen, not publicly registered.

| A primary comparison | Mean difference | Pointwise 95% interval |
|---|---:|---:|
| Native hard − raw, D=300 | −0.069 | [−0.090, −0.048] |
| Native hard − raw, D=600 | +0.038 | [0.023, 0.053] |
| Paired duration difference | +0.107 | [0.079, 0.134] |
| Native hard − size-matched random, D=600 | +0.226 | [0.210, 0.243] |

All four pass the frozen primary correction (largest adjusted p=2.93e−6).
Both durations accept all 100 candidates, so native and forced output coincide.
OLS reproduces the direction of the effects; its 1,400 window designs all have
rank 24. OLS and forced-hard auxiliary rows do not receive primary-family Holm
values. Random partitions are evaluated separately and averaged within repeat.

Study B freezes 50 different independent repeats, each paired across two duration
scales and three schedules at 23 windows: 300 successful condition fits. Forced
hard gains about +0.072, +0.072 and +0.070 at mean D=600 for fixed phase, random
phase and irregular runs; all pass the secondary Holm6 family. The three paired
duration differences pass a separate secondary Holm3 family. D=300 intervals
cross zero; this is not evidence of equivalence. Native gains remain negative
in all six cells. Gate acceptance is 2/50, 0/50, 3/50 at D=300 and 6/50, 6/50,
7/50 at D=600. Accepted-only estimates retain their own raw baselines and are
descriptive conditional means, with explicit small or zero denominators.

These studies confirm the selected condition contrast and probe specified
schedule changes. Realised switch counts, coverage and purity also change.
They establish neither a universal duration threshold nor broad superiority
outside the tested switching-VAR, supplied-two-state and local-linear setting.
The J/C diagnostic reconstructs realised scores; it is not independent causal
evidence or a rule for choosing duration without truth.

## Manuscript and submission preparation

The new main article has seven sections, a 182-word abstract, three contributions,
four figures and four tables. The separate supplement retains extended theory,
earlier controls, GMM/segmentation sensitivities, ESS diagnostics and provenance,
and adds three detailed A/B tables. The main article is 21 double-spaced pages;
the supplement is 35 pages. Both compile with zero overfull boxes or undefined
references/citations, and embedded fonts. All pages were rendered and inspected;
the new tables and repaired reproduction paths received individual review.
The offline compiler emits nonfatal cache/BibTeX rerun notices; no claim of a
warning-free compiler run or byte-identical PDF rebuild is made.

The generator description records self terms in simulation/regression, the
non-self scoring mask, unchanged coefficient/support sampling, one-pass radius
rescaling and burn-in phase. AI assistance for research code is disclosed in
Methods, with a separate manuscript-preparation statement. The article identifies
STCG-v1 as this project's controlled prototype and scopes the nearest evaluation
work to its i.i.d. target. Author, funding, interests and publication-history facts
use the author's direct replies. Highlights, cover letter, declarations, factual
biography and deposit metadata are supplied separately.

No additional ESS draws or new benchmark algorithms were introduced. All 218
historical result CSVs agree byte-for-byte with the retained previous archive,
including ESS. The earlier mechanism figure's duration/overlap labels were
corrected; its numerical source was not changed. Existing dirty changes and the
pre-existing staged deletion remain present.

## Verification and archive boundary

All 336 regression tests pass, including the A/B, manifest/path, tiny-p-value,
author-context, complete-array collection and atomic archive tests. All 237
historical manuscript-number checks pass. The current read-only gate checks
the A/B protocols and arrays, 11,928 exact matched-partition scores, 1,500 B
method rows, 6,900 window diagnostics, numerical interpretation, display hashes
and both LaTeX dependency closures. No new scorer is fitted during verification.
The presentation builders preserve all 1,915 frozen A/B files.
An independent implementation using SciPy, statsmodels Holm correction and
NumPy bootstrap also reconstructs all 13 primary/secondary A/B comparisons;
the largest absolute discrepancy is 2.22e−16. A separate ESS check verifies
the plus-one formula, Holm8/Holm32 and all eight displayed AUROC/range rows.

The local archive contains both PDFs, source, required ESS data, complete A/B
inputs/outputs, old retained evidence, reproduction tools and a payload manifest.
Its README distinguishes read-only verification, display rebuilds from saved
outputs, and fresh generation/fitting. Older Hydraulic/AReM acquisition remains
separate. The archive is tested in an isolated pinned environment; exact commands,
before/after hashes and final export checks are recorded in the companion audit
and execution logs. Rebuilding may alter PDF metadata; original exported bytes
are checked before rebuilding, and all scientific integrity checks remain active.

The tested export has 3,990 payloads plus its manifest. The read-only route
passes in 73.332 seconds with all 3,991 files unchanged. The separate rebuild
passes in 103.814 seconds: all 26 rebuilt tables and all 61 retained table files match byte-for-byte,
all 1,915 frozen study files and 218 historical result CSVs are unchanged,
and the four new figure PDFs are byte-identical. The sole changed legacy
vector PDF differs only in CreationDate; all five inspected PDFs retain
identical drawing streams and page boxes. Both command stderr logs are empty.
The final export adds the completed audit and script-index documentation;
its payload comparison must show no code, scientific-input/output or PDF
change from this tested snapshot. The final ZIP checksum is kept in a separate
execution record to avoid a self-referential archive checksum.

The archive is prepared locally; nothing has been uploaded, publicly deposited
or submitted. A genuine public repository identifier remains to be added after
author-controlled deposition. No author photograph has been supplied. The author
must review the scientific text and submission declarations before submission.

The figure workflow materially used the scientific-visualization skill. Its
required attribution and provenance are in `audit/information_sciences_assets_2026-10-24.md`:
Kassis et al. (2026), *Scientific Agent Skills*,
[DOI:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
This workflow reference is not evidence for a manuscript research claim.
