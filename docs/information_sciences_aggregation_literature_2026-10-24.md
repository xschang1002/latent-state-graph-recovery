# Direct literature on estimating local connectivity, then grouping states

Search and reading date: 2026-10-24. This is a targeted four-paper comparison, not a systematic or exhaustive review. It uses original articles and their publisher/PMC records. The new BibTeX entries are in `results/information_sciences/aggregation_literature.bib`; existing manuscript sources and references were not edited.

## Selection and access boundary

Inclusion required estimated local connectivity matrices followed by clustering/decomposition into recurring connectivity patterns. Joint segmentation/graph estimation and temporal penalties are a separate family. Search terms included the exact Allen, Shakil and Abrol titles, sliding-window connectivity with K-means, state-transition evaluation, reproducibility, and task-linked FNC. A recent task study was included to avoid describing the literature only through early resting-state work.

Direct PMC HTML retrieval returned a browser-check page. For Shakil, the complete main-article text, figure captions and Tables 1–2 were read through NCBI's public BioC API. Abrol's main article was read through Europe PMC's public full-text XML API. Allen's relevant main-text sections were available in the indexed original PMC article; the separate supplementary file was not retrieved. Chang's publisher HTML was retrieved and its Methods, Results, figure captions and citation metadata were read. The source copies remain in the local analysis log, outside the reproducibility inputs. No claim below depends on an unread supplementary result.

## Four directly relevant studies

### Allen et al. (2014): local estimates followed by recurring-state clustering

Elena A. Allen, Eswar Damaraju, Sergey M. Plis, Erik B. Erhardt, Tom Eichele and Vince D. Calhoun. *Tracking Whole-Brain Connectivity Dynamics in the Resting State*. **Cerebral Cortex 24(3):663–676**. DOI **10.1093/cercor/bhs352**. Final issue year is 2014; online publication was 2012. [Original article](https://pmc.ncbi.nlm.nih.gov/articles/PMC3920766/), [publication metadata](https://pubmed.ncbi.nlm.nih.gov/23146964/).

- Pipeline: ICA time courses → regularized covariance estimates in tapered sliding windows → Manhattan-distance clustering → representative cluster-median connectivity maps. It is not the present paper's squared-Euclidean K-means plus arithmetic-mean emission.
- Evaluation: resampling/split-half reproducibility, surrogate checks, state occurrence and modular organisation. The main text also describes a SimTB check of states **and transitions** in Supplementary Figure S7; the supplementary result itself was not independently inspected here.
- Relevance: an established instance of clustering already estimated local connectivity. Reproducible representatives and accurate timing remain distinct questions; this does not demonstrate an error in Allen's reported results.
- Support locations read: Figure 1; “FC Estimation and Temporal Variability”; “Clustering Analysis”; Figure 5; discussion of modelling choices. Read scope: those main-text passages, abstract and metadata; separate supplement not read.

### Shakil, Lee and Keilholz (2016): an essential temporal-evaluation predecessor

Sadia Shakil, Chin-Hui Lee and Shella Dawn Keilholz. *Evaluation of sliding window correlation performance for characterizing dynamic functional connectivity and brain states*. **NeuroImage 133:111–128**. DOI **10.1016/j.neuroimage.2016.02.074**. Use the final 2016 publication year, not a later manuscript-availability date. [Original article](https://pmc.ncbi.nlm.nih.gov/articles/PMC4889509/), [publication metadata](https://pubmed.ncbi.nlm.nih.gov/26952197/), [NCBI text API](https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/PMC4889509/unicode).

- Pipeline: splice real fMRI segments into networks with controlled transitions → pairwise sliding-window correlations → K-means state assignments; raw-signal clustering is also checked.
- Evaluation: correlation fidelity, full-state identification percentages, detected-transition counts, and correct-transition percentages relative to detected and actual counts. Periodic, quasi-periodic and random schedules are considered.
- Relevance: this paper already tests the relation between window length, state duration and temporal recovery. The present contribution should focus on paired **aggregation-minus-raw changing-edge ranking** and its factorisation, not claim the first temporal evaluation or first window/duration sensitivity study. Their fMRI-derived correlation targets differ from directed lagged support.
- Support locations read: Methods “Simulated networks and sliding window correlation”, “Ground truth” and “Clustering”; Results “Dependence of state identification on window length”; Figures 6–9; Tables 1–2; Discussion “Considerations for future SWC studies”. Read scope: main article and supplied table/caption text, not supplementary files.

### Abrol et al. (2017): replication, temporal summaries and careful surrogate interpretation

Anees Abrol, Eswar Damaraju, Robyn L. Miller, Julia M. Stephen, Eric D. Claus, Andrew R. Mayer and Vince D. Calhoun. *Replicability of time-varying connectivity patterns in large resting state fMRI samples*. **NeuroImage 163:160–176**. DOI **10.1016/j.neuroimage.2017.09.020**. [Original article](https://pmc.ncbi.nlm.nih.gov/articles/PMC5775892/), [publication metadata](https://pubmed.ncbi.nlm.nih.gov/28916181/), [full-text XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5775892/fullTextXML).

- Pipeline: sliding-window correlation vectors → hard Manhattan-distance clustering; a second branch decomposes the same windowed connectivity into fuzzy meta-states using ICA/PCA/clustering bases.
- Evaluation: matched state-profile correlations, occurrence, dwell/fractional time, transition probabilities, meta-state summary measures, and phase-randomized surrogates across independent groups.
- Relevance: temporal summaries are explicitly evaluated. Their reproducibility is not equivalent to recovery of latent changing-edge truth. The authors themselves explain that preserved lagged cross-covariance accounts for much, but not all, of the clustering; surrogate rejection does not isolate nonstationarity from nonlinearity or non-Gaussianity.
- Support locations read: Sections 2.4–2.4.2, 3.2.2 and 3.2.4–3.2.5; Figures 4, 6–7; Sections 3.3.1–3.3.3. Read scope: complete main-article XML, including captions; videos/supplementary files not read. The title/abstract says 7,500 scans, while Section 2.1 describes exclusions; no manuscript claim should casually treat 7,500 as the analysed participant count.

### Chang et al. (2026): task alignment and external behavioural relevance

Xiao Chang, Tianye Jia, Zening Fu et al. *Neurocognitive characterization of behaviour and mental illness through time-varying brain network analysis*. **Nature Communications 17:1353**; published **3 February 2026**. DOI **10.1038/s41467-025-67398-w**. The DOI contains 2025, but the actual publication is 2026. [Publisher article](https://www.nature.com/articles/s41467-025-67398-w).

- Pipeline: ICA time courses → sliding-window graphical-Lasso-based covariance estimates → clustering of exemplar windows and all windowed FNC matrices → recurrent-state centroids and assignments.
- Evaluation: spatial-profile correlations, graph metrics, state/task-event correlations, dwell-time/behaviour associations, model fit including adjusted R²/log-likelihood, and an independent clinical cohort.
- Relevance: this is direct evidence that temporal and behavioural validation is already practiced. Task-state coupling tests a meaningful external association; it does not supply per-edge causal-support truth. Our simulation metric answers a different target question and does not invalidate that association.
- Support locations read: Figure 1; Results “FNC states are differentially modulated by ongoing task events”; Methods “Time-varying functional network connectivity (FNC) estimation”, “FNC states derived from clustering on time-varying FNC matrices”, task-event and behavioural analyses. Read scope: main publisher HTML, not supplementary PDFs.

## Taxonomy that should survive the rewrite

| Family | What is shared or grouped | Comparison boundary |
|---|---|---|
| Estimate, then group | Previously estimated local connection matrices are clustered/decomposed and summarized. The four articles above belong here. | This motivates studying the aggregation layer. They use undirected functional association and heterogeneous summaries, not identical copies of STCG's directed-support estimator. |
| Joint segmentation and graphical state estimation | Assignments and cluster-specific graphical models are optimized together. TICC explicitly uses alternating estimation of these variables. | Do not call this post hoc clustering of fixed local graphs, or treat a constrained PELT adapter as its implementation. |
| Penalized sharing across time | Network estimates are coupled during fitting by a temporal regularizer, as in TVGL. | Changing fitted local estimates and aggregating fixed estimates are different interventions. |

Primary checks for the last two rows: [TICC author paper, v2](https://arxiv.org/html/1706.03161v2), Section 2 Eq. (1) and Section 3's alternating-minimization formulation; [TVGL author paper, v2](https://arxiv.org/html/1703.01958v2), Section 2 Eq. (2), whose temporal penalty couples successive precision matrices. These sources were inspected for classification only; no numerical performance comparisons or superiority claims are drawn from them. Existing Regime-PCMCI/CASTOR/SPACETIME discussion can remain in the joint-estimation family.

## Paste-ready manuscript suggestions

The following paragraphs use the four verified keys in the accompanying temporary BibTeX file. They are suggestions for the main author to integrate; no manuscript source was changed.

### Introduction motivation: one paragraph

```latex
Estimating local connectivity and then grouping recurring patterns is an established
analysis strategy. Sliding-window functional-connectivity studies cluster estimated
connection matrices into representative states, using them to describe recurrence,
reproducibility and task-related variation
\citep{allen2014tracking,abrol2017replicability,chang2026neurocognitive}.
Such applications motivate a separate question for dynamic graph recovery: when does
replacing local estimates by state-level summaries improve the recovery of edges at
the times they are active? A representative graph can be reproducible while the
timing of particular edges remains inaccurate. We study this distinction by keeping
local scores fixed and comparing aggregation rules against the same time-indexed
truth, separating ranking across edges from discrimination across time within
changing edges.
```

### Related work: two paragraphs

```latex
Post-estimation grouping is exemplified by dynamic functional-connectivity analysis.
\citet{allen2014tracking} cluster regularized windowed covariance
matrices into recurring states. \citet{abrol2017replicability} examine
replication of state profiles and temporal summaries, including dwell times and
transitions, with hard clustering and fuzzy meta-states. More recently, task-fMRI
work relates clustered connectivity states to ongoing events and behaviour
\citep{chang2026neurocognitive}. These studies establish useful validation targets;
functional association, state recurrence and behavioural relevance differ from
recovering the temporal support of individual directed lagged edges.

Temporal validation also predates the present study. Using simulated networks with
known transitions, \citet{shakil2016evaluation} evaluate correlation
fidelity, state identification and transition recovery as sliding-window parameters
and state durations change. We instead isolate aggregation of shared local scores
and measure its paired gain over raw windows on changing-edge temporal AUROC.
The resulting decomposition relates assignment discrimination to the direction of
state-map contrasts. This post-estimation intervention differs from jointly fitting
assignments and graphical models, or coupling local graph estimates through a
temporal penalty; results for one intervention do not establish performance for
the others.
```

## Contribution boundaries and placement

1. **Mathematical statements independent of STCG.** The finite-array same-edge/cross-edge comparison split and static occupancy argument concern binary time-indexed labels and scores. Directedness, Lasso, Gaussianity and the gate are not required. The static optimum specializes classical ranking theory, already cited; the contribution is its diagnostic use in dynamic recovery, not a new general ranking theorem.
2. **A precisely delimited hard-emission identity.** The factorisation requires two true states, a common hard predicted state sequence and two fixed emitted maps, evaluated on changing edges. It does not automatically cover diffuse soft emission, arbitrary numbers of states, or jointly changing local estimators. It is an exact relation on the evaluated data, not a causal allocation of error between independently manipulable factors. It needs truth and is not a deployable tuning threshold.
3. **Conditional empirical evidence.** A/B test a specified local linear scoring/aggregation framework. OLS checks one scoring variation; random phase and irregular schedules check specified coverage variations. These strengthen that evidence without constituting a reproduction or adverse finding about the four fMRI methods. Adding these citations repairs motivation; it does not itself broaden the empirical algorithm coverage.
4. **Gate as an implementation case.** Move the inertia bound/prototype's threshold pathology out of the central contribution sentence. Retain native versus forced outputs and execution frequency as operational diagnostics, with the bound/prototype details in the supplement. Do not imply that Allen, Abrol, Shakil or Chang use this gate. A suitable contribution emphasis is: “A two-state identity explains how state discrimination and graph-contrast direction jointly determine changing-edge temporal AUROC; controlled paired comparisons identify conditions in which aggregation improves or degrades it.”

Suggested narrative priority: established aggregation workflow → distinct evaluation targets → exact diagnosis → controlled conditional evidence. Prototype gate behaviour is a concrete illustration after this argument is established.
