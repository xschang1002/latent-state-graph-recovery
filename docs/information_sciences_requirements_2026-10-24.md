# Information Sciences: source-checked submission requirements

Checked on **2026-10-24 (Asia/Shanghai)** for the requested venue change.
This is a preparation record, not a claim that a submission has been completed
or that the editors have approved the article's fit.

## Primary sources and access

- [Information Sciences, Guide for authors](https://www.sciencedirect.com/journal/information-sciences/publish/guide-for-authors):
  **read directly in the background browser** on the date above. The ordinary
  web-fetch channel returned HTTP 403; the browser displayed the official guide,
  including Article types, Peer review, Title page, Abstract, Highlights,
  Declaration of generative AI use, Research data and Data statement. No mirror
  is used as the authority for the requirements below.
- [Elsevier, Generative AI policies for journals](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals):
  read directly; the page identifies its policy update as June 2026. Relevant
  locators are For authors / Disclosure, research-methods guidance, and the FAQ
  concerning AI-assisted code.
- [Elsevier, Highlights](https://www.elsevier.com/researcher/author/tools-and-resources/highlights):
  read directly. Its general instructions are subordinate to journal-specific
  requirements.
- [Elsevier, preparing files for Editorial Manager](https://www.elsevier.support/publishing/answer/how-do-i-prepare-my-files-for-submission-in-editorial-manager):
  official support page, updated June 15, 2026. Journal-specific instructions
  take precedence over these general file-preparation instructions.

## Requirements relevant to this revision

| Topic | Directly observed requirement | Preparation decision |
|---|---|---|
| Article length and displays | Under Article types, the guide gives experimental articles a guideline of 40 double-spaced typed pages with **8 figures plus tables**; theoretical articles have suggested figures of 45 pages and 10 combined displays. | Target 4 figures and 4 tables in the main article, and prepare a separate supplement. These are journal guidance, not evidence that any excess automatically causes rejection. A compact 35-page PDF cannot be equated with 35 double-spaced pages. |
| Abstract | The same live page contains an internal discrepancy: Article types gives **up to 200 words**, while Writing and formatting / Abstract gives **250 words**. | Use **180–195 words, never above 200**, satisfying the stricter journal statement and the user's requested limit. Do not report that the page contains one consistent abstract limit. |
| Review model | Peer review explicitly identifies **single-anonymized** review. | Supply the actual author information for a real submission. An internal draft can retain conspicuous missing-information markers, but should not be described as ready for upload while it says Anonymous Authors. |
| Title page | Author names and order, full affiliations including postal addresses and countries, and a named corresponding author with email are requested. | Obtain the real details from the author; do not infer affiliations from this computer or repository. |
| Highlights | The journal **encourages** highlights at submission. They are a separate editable file, named with highlights, containing **3–5 items, at most 85 characters per item including spaces**. | Prepare 4 checked items; do not label them mandatory. The general Elsevier highlights page additionally describes Word format. |
| Interests and funding | The declarations tool is to be completed even when there is nothing to declare; its generated declaration is uploaded as .doc/.docx. Funding and sponsors' roles must be stated accurately. | Draft the document structure, and obtain the actual declarations. Do not infer no funding or no interests from their absence in the manuscript. |
| Contributions | The Article structure section requires co-author contributions expressed using CRediT roles. | Obtain the real division of contributions; do not attribute authorship to an AI tool. |
| Availability | Research data / deposit, citation and linking explicitly selects **Option C**: deposit data in a relevant repository and cite/link it, or explain why it cannot be shared. A data-availability statement is required at submission. | Keep the local artifact and its verified reproduction commands, but do not equate a local ZIP with an already deposited, persistently linked repository. Preparing files does not authorize public upload. |
| Supplement | Supplementary files must be cited, captioned, accurate, and supplied alongside the article. They are distributed as received rather than copyedited. | Package the standalone supplement and source/data artifact consistently, with explicit generation, fit, summary, verify and rebuild entry points. |
| Author biographies | Writing and formatting / Article structure / Vitae asks for an editable biography of at most 100 words per author and a separate passport-type photograph. | Treat these as author-supplied materials. Do not create biographical facts or photographs. |

## AI disclosure and research-code provenance

The live journal guide requires a manuscript-preparation AI statement before
the references when applicable. Elsevier's linked policy separately requires
research uses, including AI-assisted code development, to be described in
Methods. Record the actual tool, purpose and human checking; code validity and
reproducibility standards remain unchanged. Data visualisations must faithfully
derive from underlying data through reproducible methods. AI involvement in
creating their analysis or plotting code belongs in the Methods disclosure.

For this project the disclosed scope should cover assistance with code,
regression tests, numerical checking, plotting and manuscript revision, where
actually performed. The corresponding author must review the final wording and
accept responsibility. Do not substitute a claim of language polishing alone
for the work recorded in the repository. Do not invent a model version that was
not recorded. Retain the operational provenance separately from the scientific
claims.

## Confirmed author facts and remaining submission steps

The author has confirmed the following facts for this manuscript:

- **Sole author and corresponding author:** Xuansong Chang.
- **Affiliation:** Stony Brook Institute at Anhui University, Hefei, China.
  The English institutional name is corroborated by the
  [Anhui University official bilingual page](https://www.ahu.edu.cn/_t992/2022/0701/c15904a289486/page.htm)
  and [Stony Brook University's Office of Global Affairs](https://www.stonybrook.edu/commcms/iaps/global-initiatives/sbiahu).
- **Correspondence:** alec.xs.chang@gmail.com.
- **Funding and interests:** no external funding and no competing interests.
- **Publication status:** no formal previous publication and no overlapping
  manuscript under review elsewhere.

The identified title page, local declarations, sole-author CRediT statement and
short factual biography use these confirmations. No postal code, ORCID, academic
rank or degree has been inferred. The remaining preparation steps are the real
repository deposition and its persistent identifier, an author-supplied
passport-type photograph, and final author review/approval of the article and
submission materials. The author has authorised preparation of an upload-ready
archive, not public deposition or journal submission. No repository DOI or URL
is claimed before it actually exists.

## Local evidence for STCG-v1's role

This is a source-local audit, not an external publication search:

- `pyproject.toml` identifies the `stcg` project as a preparation scaffold,
  version 0.1.0.
- `docs/audit/stcg_v1_recovery_run_2026-05-14.md`, Purpose, records that the
  project added STCG-v1 to its own synthetic recovery loop. The description
  identifies clustering of windowed lagged reconstruction scores and state-wise
  aggregation, rather than the reproduction of a cited external STCG-v1 paper.
- `docs/audit/current_status_and_next_plan_2026-05-14.md`, Main Risks and Gaps,
  characterises the then-current implementation as graph-state sparse
  reconstruction and cautions against calling it a full neural framework.
- `src/stcg/self_supervised.py` contains `STCGStateConfig`,
  `stcg_v1_diagnostics` and `stcg_v1_scores`. The manuscript describes the same
  local-window scoring, clustering, aggregation and fallback components.

Together with the author's prior-publication confirmation, these records
support describing STCG-v1 here as a **project research prototype used as a
controlled vehicle for studying aggregation**, rather than presenting it as an
externally published algorithm being reproduced. They do not justify an
unqualified claim about the absence of all public versions or about the novelty
of its individual components.

Suggested method sentence:

> We use STCG-v1, a research prototype developed in this project, as a controlled
> framework for isolating the effects of state aggregation on a shared set of
> local graph scores.

The scientific contribution should then be stated through the evaluation
targets, condition-dependent aggregation results and mechanism diagnostics.

