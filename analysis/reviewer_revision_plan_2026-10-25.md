# Reviewer-directed revision plan — 25 October 2026

This local plan responds to the author's supplied hypothetical review. It is
an internally recorded post-review extension, not external preregistration.
The original Studies A/B, their inferential families and the ESS analysis are
retained unchanged. Existing uncommitted work is preserved; no commit, deposit
or journal submission is authorised by this local revision.

## Questions and fixed comparisons

1. Can a non-clustering operation alter pooled ranking without discarding
   within-edge temporal discrimination? For all retained A/B fits, compare
   `lambda*raw + (1-lambda)*window_mean` at lambda = .25, .5, .75, 1.
   Prove positive edgewise-affine rank invariance and inspect actual floating
   point order/tie changes, rather than forcing numerical agreement.
2. Does retaining raw residuals alter the hard-aggregation tradeoff? Compare
   `lambda*raw + (1-lambda)*forced_hard` at the same values, with lambda = 0
   explicitly identified as the existing hard endpoint. Report every value;
   there is no post hoc selection of a best lambda.
3. How much native degradation comes from the fallback? Keep the old gate
   decision fixed, emit hard on acceptance and raw on rejection, and compare
   with raw, forced hard and the original constant-maximum fallback. Recompute
   paired effects from repeat-level arrays, including accepted-only denominators.
4. Do the results transfer to a symmetric generator and targeted window
   changes? Add Study C below, separately from the original asymmetric study.
5. Does within-edge state discrimination agree with direct temporal recovery?
   Add explicitly defined boundary/interval and stable-edge false-change
   diagnostics. Freeze their separate post hoc metric protocol before reading
   their A/B/C results. They do not replace the original AUROC target.

The score-control module, direct-localisation module and Study C each retain
their exact numeric protocol, source hashes, input hashes, output manifest,
failure/undefined counts and replay entry point. The metric protocol will state
any calibration labels, oracle budgets or threshold choices explicitly.

## Study C: bounded robustness extension

Thirty independent mechanism/initial-history/innovation repeats use seeds
70000–70029, with shared stochastic inputs within each repeat. Observed length
is 3600 and burn-in is 100; N=8, lag order 3, Gaussian innovation SD=.10 and
the existing local Lasso scorer are retained. Known state count remains two.
Fixed-phase schedules use D=300 and D=600. Window size/stride pairs are
(200,100), (300,150), (400,200), (300,75), (300,300), producing 35, 23, 17,
45 and 12 windows. All ten cells are paired within repeat: 300 planned fits,
with no result-dependent seed addition, replacement, window selection or stop.

Each lagged nonself A edge has probability p=.18. Conditional B retention is
o=.5 and a position absent from A is added with probability
q=p(1-o)/(1-p), giving equal Bernoulli marginals in both states. Both diagonal
supports are zero. Shared edges retain exactly the same signed coefficient;
unique edges use the same sign/magnitude distribution in either state. Empty
graphs are allowed rather than conditionally resampled. Thus support sampling
is exchangeable under swapping state names, without claiming equal realised
edge counts or overlap in every repeat.

For initial companion radii rho_A,rho_B, use a common
a=min(1,.90/max(rho_A,rho_B)) and multiply each lag-l coefficient block in
both states by a^l. Retain initial/final tensors, radii and a. Numerically
check the similarity-based identity rho_final=a*rho_initial and the final
radius bound, without claiming stability under arbitrary switching products.
Common lagwise scaling preserves equality of shared coefficients. This new
construction changes self support, marginal density and shared coefficients
together; it does not separately identify each generator nuisance effect.

The Study C configuration and implementation are hashed before the first run;
actual internal clock records bracket freeze and launch. The endpoint is the
fixed 300-case plan, including retained numerical failures. Full arrays permit
post-fit replay of raw, constant, hard, oracle and the new score/fallback
controls without additional scorer fits.

## Analysis and interpretation

The repeat is the statistical unit. Report condition/method means and paired
differences with pointwise 95% percentile-bootstrap intervals using 10,000
resamples and fixed statistical seed(s) recorded in each module. Separate A
Lasso, auxiliary A OLS, B schedules and C windows. New controls and robustness
estimates are descriptive post-review evidence; they do not acquire the original
primary Holm labels. Retain failed fits and undefined metric targets, with
coverage rather than substitution by chance performance.

The core AUROC is described as changing-edge temporal state discrimination.
Boundary or interval recovery is a distinct target. The two-state product is
a diagnostic identity; it does not become a predictive model-selection rule.
The score-preservation and fallback controls provide actionable design checks
without a universal aggregation threshold. ESS remains a compact example of
static-target validation limits.

## Deliverables and closure

Update the main article, separate supplement, all dependent summaries/captions,
submission text, figures and tables from completed outputs. Provide an issue-to-
change response and exact evidence pointers. Add regression tests for the new
invariance, fallback, generation and localisation logic, run affected/full
checks, compile and visually inspect both PDFs, and create an updated local
archive. Test read-only verification and display rebuilding in two fresh
extractions, retaining the original experiment/source hashes. Archive testing
and author review are distinct from independent scientific replication.
