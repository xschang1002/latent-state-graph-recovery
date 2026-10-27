# Author-prepared response to hypothetical review concerns

This local response addresses author-supplied hypothetical comments used to plan
the revision. It does not represent a received external review, editorial
decision or journal correspondence. Private source wording is not reproduced.
The accompanying evidence index supplies exact numerical and protocol locations.

## Contribution and theoretical scope

The paper is framed as an evaluation and controlled aggregation study. The
occupancy ranking argument and positive-affine AUROC invariance are applications
of elementary ranking principles; the two-state formula and comparison-pair
decomposition are diagnostic identities. The revision uses those identities to
design discriminating controls, without presenting them as deep new general
ranking theory or a truth-free selection rule. The title and main terminology
now distinguish temporal state discrimination from boundary localisation.

## Ordinary shrinkage and residual retention

Supplement S17 compares all fixed positive lambdas 0.25, 0.5, 0.75 and 1, plus
the hard endpoint, on unchanged A/B scores and completed C scores. In Study A
Lasso, shrinkage at lambda 0.25 gives pooled AUROC 0.804/0.843 at D=300/600 while
retaining raw temporal state AUROC 0.707/0.767. Residual-hard at that same weight
has temporal-minus-raw differences +0.010/+0.057. These are post hoc controls,
not new primary confirmations, and every other weight is reported. Actual
float64 checks of ordinary shrinkage preserve edge rankings/ties at all tested
positive weights.
The OLS auxiliary controls are in Table S24.

## Gate rejection and raw fallback

Table S25 reports gate counts and all-repeat raw-fallback effects using each
repeat's own raw baseline. At B's D=600, gains are approximately +0.013, +0.013
and +0.012 for fixed, random and irregular schedules. The original native
constant fallback is retained as a comparator. In these B cells raw fallback avoids the
observed constant-rejection loss; it cannot fix an accepted poor candidate, and C retains
negative raw-fallback cells. The revision does not claim universal gate repair.

## Generator and window portability

Study C (S18/Table S28) adds 30 separate repeats paired across ten fitted cells.
Both states exclude self support and have matched nonself support marginals;
shared edges retain coefficients, with a shared recorded stability rescaling.
This bundles several generator changes rather than identifying their individual
effects. Five window/stride settings give 35, 23, 17, 45 and 12 windows. The
default 300/150 forced-minus-raw contrast is approximately -0.047/+0.037 at
D=300/600; at D=600 with stride 300 it is -0.014 with an interval crossing zero.
These results broaden the controlled setting while retaining the two-state
linear model and original local scorer. General nonlinear, unknown-state-count
and broader algorithmic generalisation remain unresolved.

## Direct temporal recovery and stable edges

S17 separately defines boundary-slot AP/F1 at window-centre resolution and
stable-edge false variation, with persistent/absent edges separated. Table S26
reports all eighteen Lasso cells, and Table S27 the stable-edge split. Under the
shared raw-derived threshold, B's D=600 forced output lowers boundary-slot F1
in all three schedules despite improving temporal state AUROC. Boundary AP has
an explicit zero-change abstention convention and handles tied strengths as
groups; missing coverage is retained. These diagnostics do not supply a
sample-level boundary-distance guarantee, and the revised claims respect this.

## Industrial evidence and reproducibility

ESS remains static expert-graph alignment rather than validation of dynamic
truth. The new analyses retain every planned output, source/input/output hashes,
pointwise paired-repeat intervals and complete weight sets. Read-only checks,
display rebuilding and fresh fitting are separate commands. Two transparent
QA-only amendments followed initial outputs: undefined NaN values were removed
from an affine-event list, and null error fields were normalised for C direct
metric serialization. Parameters, samples and numerical findings were unchanged.
Public repository deposition remains pending; no public identifier is invented.
