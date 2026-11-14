# State Aggregation in Dynamic Graph Recovery

When does replacing local graph estimates by shared state-level summaries help
recover edges at the times they are active? This repository accompanies
**State Aggregation in Dynamic Graph Recovery: Separating Structural Ranking
from Temporal Discrimination**, by Xuansong Chang.

Aggregation can improve the ranking of edges across a record while reducing the
ability to distinguish the active and inactive times of an individual edge.
The study holds local scores and simulated inputs fixed to compare these
questions separately. It combines diagnostic identities with paired simulation
studies, score controls, boundary diagnostics and an industrial example with
static expert truth.

## Read the study

- [Main article](paper/main.pdf)
- [Supplement](paper/supplement.pdf)
- [Research guide: questions, evidence and current supplement locations](docs/research_guide.md)
- [Reproduction instructions](docs/knowledge_based_systems_artifact_readme.md)
- [Main and supplement sources](paper/)

The results concern the tested two-state linear systems and local linear
scorers. The negative/positive hard-minus-raw contrast at two duration
configurations is supported by independent paired simulations. Duration changes
also change switch coverage, and window/stride studies bound the contrast; the
study does not identify a universal duration threshold. Non-clustering shrinkage
and residual controls help distinguish pooled-ranking gains, temporal ordering
and tie-breaking. Standard AP and positive-change AP, together with fixed
threshold references, show why a single boundary statistic can change the
assessment of the same predictions. A PELT segment-refit comparison applies the
evaluation protocol beyond recurrent clustering.

## Outputs being compared

| Output | What is emitted at each window |
|---|---|
| Raw window | That window's local scores. |
| Forced hard | The candidate hard-group output, regardless of the gate decision. |
| Gated hard (static-max fallback) | The candidate when accepted; otherwise the edgewise maximum over all raw windows, repeated at every window. This is the static-max fallback. |
| Raw fallback | The candidate when accepted; the original window scores when rejected. |
| Oracle | An output constructed using known simulation state labels; a diagnostic control. |

Gated, forced and raw-fallback effects answer different questions. Gate
acceptance and effects conditional on actual execution are reported separately.
`STCG`, `STCG-v1` and `src/stcg/` remain implementation and historical identifiers;
the paper's claim is about evaluation and the tested aggregation rules.

## Reproduce the supplied results

Use Python 3.12 and the pinned environment, then verify the retained evidence:

```text
python -m pip install -r requirements-revision.txt
python -B scripts/reproduce_paper.py --verify
```

Verification is read-only and does not fit a scorer. To rebuild displays from
retained results in a separate working copy:

```text
python -B scripts/reproduce_paper.py --rebuild
```

Display rebuilding, regenerating analysis CSVs and new generation/fitting are
separate operations. The [full instructions](docs/knowledge_based_systems_artifact_readme.md)
give independent output and execution-record paths for each. Eight selected
same-seed A/C pipelines have also been generated and fitted from scratch; their
492 recorded comparisons are exact in the recorded environment. They are a
bounded computational check, not new independent seeds or a fresh full-grid run.

The [complete reproducibility archive](https://github.com/xschang1002/latent-state-graph-recovery/releases/download/v2026.11.15-kbs/knowledge_based_systems_reproducibility.zip)
in [release v2026.11.15-kbs](https://github.com/xschang1002/latent-state-graph-recovery/releases/tag/v2026.11.15-kbs)
contains the current KBS submission PDFs and sources with the unchanged
scientific inputs, fitted outputs, protocols and analysis code. Preceding
[release v2026.10.28-layout.1](https://github.com/xschang1002/latent-state-graph-recovery/releases/tag/v2026.10.28-layout.1)
remains preserved with the earlier presentation and the same scientific
material. The [layout revision map](docs/ins_layout_revision_2026-10-28.md)
remains the record of the unchanged figure/table organisation.
ESS has static graph truth. Some older real-data studies require separate
acquisition; see the artifact instructions and [third-party notices](docs/third_party_notices.md).

## Repository map

| Path | Contents |
|---|---|
| `paper/` | Main article and supplement sources. |
| `results/information_sciences/` | Paired studies, controls, boundary analyses, comparator outputs and display assets. |
| `src/stcg/` | Reusable simulation, scoring, aggregation and evaluation code. |
| `scripts/` | [Experiment and analysis entry points](scripts/README.md). |
| `tools/` | Verification, display and archive builders. |
| `configs/`, `tests/` | Study configurations and regression checks. |
| `docs/research_guide.md` | Current scientific reading and result map. |
| `analysis/`, `audit/`, dated documents | Retained development and execution provenance; see the guide's history index. |

The [research guide](docs/research_guide.md) distinguishes current scientific
instructions from historical records. Dated protocols, evidence indices and
responses retain their original section numbers and scope. The same archive
filename has been rebuilt across revisions; identify a copy with its actual
manifest and file hash.

## Knowledge-Based Systems transfer preparation

[Current transfer materials and checklist](submission_materials/knowledge_based_systems/README.md)
contain the contribution-led cover letter, Highlights, confirmed author
information and declarations. The author confirmed the prior Information
Sciences rejection and transfer through Elsevier. The local delivery copy is in
`submission_package/knowledge_based_systems/`; its compact submission kit and
complete reproducibility archive are separate ZIPs. Public release version:
`v2026.11.15-kbs`.

The [source repository](https://github.com/xschang1002/latent-state-graph-recovery)
and [public release v2026.11.15-kbs](https://github.com/xschang1002/latent-state-graph-recovery/releases/tag/v2026.11.15-kbs)
provide the current KBS presentation alongside the unchanged scientific
material. Preceding [release v2026.10.28-layout.1](https://github.com/xschang1002/latent-state-graph-recovery/releases/tag/v2026.10.28-layout.1),
earlier public INS materials, and historical evidence records remain
preserved unchanged. No DOI has been assigned, and a GitHub release does not
constitute submission to the journal. Current KBS-specific format checks and
final author approval follow the actual transfer system; see the
[requirements record](docs/knowledge_based_systems_requirements_2026-10-29.md).

Project code is MIT licensed. Third-party components and data retain their own
notices and conditions; see [LICENSE](LICENSE) and
[third-party notices](docs/third_party_notices.md).
