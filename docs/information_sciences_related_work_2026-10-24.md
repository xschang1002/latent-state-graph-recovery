# Nearest-neighbour literature check for the Information Sciences revision

Checked on **2026-10-24 (Asia/Shanghai)**. This scoped check addresses the paper
identified by the user, not a systematic or exhaustive novelty search. It does
not change the manuscript or bibliography automatically.

## Verified publication identity

**Georg Velev and Stefan Lessmann (2026).** *Interpretable, multidimensional
evaluation framework for causal discovery from observational i.i.d. data.*
**Information Sciences 723, 122641.**
[DOI: 10.1016/j.ins.2025.122641](https://doi.org/10.1016/j.ins.2025.122641).

The [official article page](https://www.sciencedirect.com/science/article/pii/S0020025525007741)
was read directly in the background browser. It supplies the exact title,
authors, **January 2026** issue date, volume, article number and DOI. Copyright
is 2025; the DOI also contains 2025. Neither is a reason to change the citation
year from the issue year **2026**.

The ordinary web-fetch channel initially returned HTTP 403. A source-specific
search recovered the publisher's indexed abstract, and the browser subsequently
displayed the publisher record directly. Its accessible content includes the
abstract, highlights and Data and code availability. It does **not** expose the
entire final article in the displayed page; no final-article section or table
not inspected is claimed here.

The [authors' preprint, arXiv:2409.19377v2](https://arxiv.org/html/2409.19377v2),
dated 16 December 2024, was also read, specifically its abstract, introduction,
and Section 4. Its older title uses “multi-dimensional”. Keep the final journal
metadata and the inspected preprint version distinct. In particular, the
preprint and final abstract do not describe their best-performing method
families identically, so do not copy comparative outcome claims between them.

## Supported comparison

The final publisher abstract describes a six-dimensional distance-to-optimal-
solution evaluation framework for causal discovery from observational **i.i.d.**
data. Its dimensions assess structural recovery and suitability of recovered
graphs for causal inference; the simulation varies identifiability assumptions
and nonlinearity. This supports citing the work as a close example of
interpretable evaluation methodology for structure learning.

The distinction needed here is the **evaluation object and index**: the current
study separates rankings across edges in time-varying graphs from temporal
discrimination within edges that actually change, and studies how aggregating
the same local scores alters those targets. The cited framework's stated
i.i.d. scope does not establish a result about recurring temporal regimes or
state aggregation. This is a scoped comparison, not evidence that no other
work studies temporal evaluation, and not an argument that publication in the
same journal guarantees suitability or acceptance.

Proposed related-work paragraph, to integrate near the evaluation literature:

> Multidimensional evaluation can expose differences that a single structural
> score conceals. Velev and Lessmann (2026) develop an interpretable framework
> for causal discovery from observational i.i.d. data, combining structural
> recovery with the suitability of estimated graphs for causal inference under
> varying assumptions. Our evaluation object is a time-varying graph: we
> separate ranking across edges from localisation within changing edges, and
> examine how state aggregation of the same local scores affects these two
> capabilities.

Use the formal journal citation for this high-level comparison. Further detail
about the final paper's individual criteria or empirical ranking should wait
until the relevant final full-text passages are inspected.

## BibTeX for integration

```bibtex
@article{velev2026evaluation,
  author  = {Velev, Georg and Lessmann, Stefan},
  title   = {Interpretable, multidimensional evaluation framework for causal discovery from observational {i.i.d.} data},
  journal = {Information Sciences},
  volume  = {723},
  pages   = {122641},
  year    = {2026},
  doi     = {10.1016/j.ins.2025.122641},
  url     = {https://doi.org/10.1016/j.ins.2025.122641}
}
```

No author, year, DOI or article-number field is inferred from memory. No
affiliation or funding details from this external work are imported into the
current manuscript.
