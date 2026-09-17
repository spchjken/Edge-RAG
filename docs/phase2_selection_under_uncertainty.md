# Phase 2 — Query-Conditioned Expansion-Term Selection Under Uncertainty

> Status: research design and debate record, 2026-09-17. This note covers the phase after static
> vocabulary-pool construction and before final lexical weighting. It does not describe implemented
> behavior or freeze a final method.

CRVE means **Context-Reranked Vocabulary Expansion**, the current working name for the proposed
method.

Related documents:

- [Capacity-bounded corpus-informed QE plan](corpus_informed_query_expansion_plan.md)
- [CRVE design refinement notes](crve_design_refinement_notes.md)
- [Phase-2 implementation plan](phase2_selection_implementation_plan.md)

## 1. Objective

Given a query \(q\) and a static vocabulary pool \(\mathcal{P}\), produce:

1. a high-recall candidate set \(C_1(q)\);
2. a smaller, safer candidate set \(C_2(q)\);
3. a final expansion set \(S(q)\) and, later, weights for its terms;
4. an abstention decision when the evidence is insufficient.

The final expanded query is executed once with lexical retrieval. The selector must therefore make a
decision before the true retrieval utility of a candidate is known.

The goals are joint, but must be measured separately:

- preserve or improve nDCG@10 and MRR@10;
- improve Recall@100/200/500/1000 when sparse retrieval is a candidate generator;
- limit harmful expansions and baseline regressions;
- keep preparation, memory, posting-list work and query latency bounded.

This is not a claim of universal expansion. A valid outcome may be “use the original query.”

## 2. The core distinction: uncertain top-k versus hidden expansion utility

There are three different quantities.

### 2.1 Lexical contribution

For term \(t\), document \(d\), and proposed weight \(\mu\), let

\[
L(d,t,\mu)
\]

denote the additional BM25/DPH score contribution. With an inverted index, this can be computed or
bounded from postings, term frequency, document length, and IDF. It is an observable lexical quantity.

### 2.2 Semantic proxy evidence

Let

\[
X(q,t)
\]

denote pre-retrieval features such as anchor similarity, whole-query similarity, context similarity,
document frequency (DF), collection frequency (CF), inverse document frequency (IDF), document support,
ambiguity, and estimated posting cost. These are evidence about a term, not retrieval utility.

### 2.3 Counterfactual retrieval utility

Let \(M\) be an evaluation metric, such as nDCG@10 or Recall@\(K\), and let \(R\) be the retrieval
procedure. Let \(K\) be the requested retrieval depth. For action \(a=(t,\mu)\), define:

\[
Y_M(q,a)=M(R(q\cup a))-M(R(q)),
\]

where \(R\) is the retrieval procedure. \(Y_M\) requires executing the expanded retrieval and
evaluating relevance. It is not available at query time.

For a term set \(S\), use:

\[
Y_M(q,S)=M(R(q\cup S))-M(R(q)).
\]

In general:

\[
Y_M(q,S)\ne\sum_{t\in S}Y_M(q,t).
\]

Terms can be redundant, complementary, or jointly harmful. This is why the problem is not an ordinary
top-k query over independently scored terms.

## 3. Relation to uncertain top-k research

### 3.1 What transfers

Probabilistic top-k research provides useful semantics for uncertainty. U-Topk aggregates the
probability that tuples appear in top-k across possible worlds; U-kRanks chooses the most probable tuple
at each rank. Other work uses probability thresholds, expected rank, quantile rank, and probabilistic
tail bounds for pruning.

- [Efficient Processing of Top-k Queries in Uncertain Databases](https://users.cs.utah.edu/~lifeifei/papers/utopk.pdf)
- [Ranking Queries on Uncertain Data: A Probabilistic Threshold Approach](https://scholars.duke.edu/publication/1530940)
- [Semantics of Ranking Queries for Probabilistic Data and Expected Ranks](https://users.cs.utah.edu/~lifeifei/papers/expectedrank.pdf)

Transferable ideas:

- use a probability/risk threshold instead of forcing a fixed number of terms;
- represent uncertainty at the set level, not only per term;
- progressively prune candidates with valid bounds;
- allow an empty result when no candidate clears the threshold.

### 3.2 What does not transfer automatically

Uncertain-database methods assume a known or estimated distribution over scores, tuple existence, or
possible worlds. We do not observe a distribution over \(Y_M(q,a)\) at query time. A BGE cosine or a
context score is not automatically \(\Pr(Y_M>0)\).

Therefore:

- raw cosine thresholds are not confidence guarantees;
- independent term probabilities cannot be multiplied without a dependence model;
- a top-k theorem over lexical scores cannot prove nDCG or Recall improvement;
- any probability of helpfulness must be calibrated from held-out retrieval-utility labels.

## 4. Related problem families and usable ideas

| Family | Relevant idea | Limitation for this project |
|---|---|---|
| Best/top-k identification | Fixed-budget exploration, elimination, stopping rules | Standard methods observe reward samples from an arm |
| Structured BAI (structured best-arm identification) | Shared noisy micro-observations can update several actions | Requires a calibrated mapping from evidence to action value |
| Contextual BAI (contextual best-arm identification) | Action value changes with query context | Does not remove proxy bias |
| Noisy-evaluation bandits | Explicitly model noisy or biased evaluations of true reward | Guarantees depend on the observation-bias model |
| Feasibility-constrained BAI | Test performance and safety separately; eliminate by either | Safety evidence still requires calibration |
| Conformal risk control | Calibrated acceptance and abstention | Gives marginal guarantees under exchangeability, not per-query certainty |
| Combinatorial pure exploration | Select a good subset under a budget | Usually assumes additive or structured subset utility |
| Active pairwise ranking | Use relative comparisons when absolute scores are unavailable | Comparisons still require an evaluation source |

Primary references:

- [Best Arm Identification: Fixed Budget and Fixed Confidence](https://proceedings.neurips.cc/paper_files/paper/2012/file/8b0d268963dd0cfb808aac48a549829f-Paper.pdf)
- [Structured Best Arm Identification with Fixed Confidence](https://proceedings.mlr.press/v76/huang17a.html)
- [The Role of Contextual Information in Best Arm Identification](https://www.jmlr.org/beta/papers/v27/22-0358.html)
- [Top K Ranking for Multi-Armed Bandit with Noisy Evaluations](https://proceedings.mlr.press/v151/garcelon22b.html)
- [Constrained Best Arm Identification with Tests for Feasibility](https://ojs.aaai.org/index.php/AAAI/article/view/39063)
- [Conformal Risk Control](https://arxiv.org/abs/2208.02814)
- [Nearly Optimal Sampling Algorithms for Combinatorial Pure Exploration](https://proceedings.mlr.press/v65/chen17a.html)
- [Active Learning for Top-K Rank Aggregation from Noisy Comparisons](https://proceedings.mlr.press/v70/mohajer17a.html)

The most accurate working description is:

> Fixed-budget, contextual, structured candidate-set selection under latent counterfactual retrieval utility.

“Uncertain top-k” is a useful analogy, not the complete problem definition.

## 5. Formal selection targets

Define tolerance \(\epsilon>0\) to separate meaningful changes from noise. Use \(\Pr(\cdot)\) for
probability; it is distinct from the vocabulary pool \(\mathcal{P}\).

For a single action \(a\):

\[
H_M(q,a)=1\{Y_M(q,a)>\epsilon\},
\]

\[
G_M(q,a)=1\{Y_M(q,a)<-\epsilon\}.
\]

The positive and harmful sets are different for different metrics. In particular, define separately:

- \(H_{\mathrm{nDCG}}\): terms improving nDCG@10;
- \(H_{\mathrm{Recall},K}\): terms improving Recall@\(K\).

A term can belong to one set and not the other. The first gate should normally target candidate coverage:

\[
\operatorname{Coverage}(C_1)=\Pr(C_1(q)\cap H_M(q)\ne\varnothing).
\]

Later gates should target precision and risk:

\[
\operatorname{HarmRate}(S)=\Pr(Y_M(q,S)<-\epsilon).
\]

For a set, also report expected negative change and lower-tail harm, such as the mean of the worst 5%
of observed \(Y_M(q,S)\) values. Do not replace set-level evaluation with the average of independent
term labels.

## 6. Proposed selection cascade

### Gate 1 — cheap, high-recall proposal

Purpose: retain useful terms, not decide final safety.

Candidate proposal channels to compare:

1. BGE whole-query-to-term similarity;
2. BGE anchor-to-term similarity;
3. query-to-context-centroid retrieval;
4. lexical/morphological/co-occurrence proposals;
5. a union or learned/calibrated hybrid.

Gate 1 should use a high-recall threshold and a hard cap. A fixed top-k is a baseline, not a guarantee
that the k-th candidate is useful. An adaptive threshold may retain fewer terms on confident queries and
more terms on ambiguous queries, but it must have a hard memory and latency cap.

Output: \(C_1(q)\), usually larger than the final expansion set.

### Gate 2 — shared context evidence

Purpose: reduce semantic ambiguity and estimate whether the candidate's corpus usage supports the query.

The sidecar should store canonical context chunks once and a term-to-context-ID map. If many terms
occur in the same window, one embedding is reused rather than recomputed for each term. Track:

- unique context embeddings \(N_{\mathrm{ctx}}\);
- term-context references \(N_{\mathrm{ref}}\);
- reuse factor \(N_{\mathrm{ref}}/N_{\mathrm{ctx}}\);
- duplicate and near-duplicate rates.

For term \(t\), let \(S_t\) be its retained contexts and \(s_j(q,t)\) the normalized query-context
compatibility of context \(j\). Compare:

\[
C_{\max}(q,t)=\max_j s_j(q,t),
\]

\[
C_{\mathrm{top2}}(q,t)=\text{mean of the two largest }s_j(q,t),
\]

\[
C_{\mathrm{support}}(q,t)=\text{compatibility weighted by context-group support}.
\]

Context-group support is the fraction of retained contexts represented by the group containing the
compatible context.

Interpretations:

- maximum: one compatible usage is enough;
- top-2/top-k mean: compatible usage should repeat;
- representative mean: dominant corpus usage should be compatible;
- support-aware score: rare singleton matches are penalized;
- opportunity-plus-risk: retain both a compatible minority sense and evidence of ambiguity.

Do not apply a sigmoid to raw cosine and call the result a probability. The threshold and temperature
must be calibrated on development data.

Use adaptive context budgets, for example 2 -> 4 -> 8 -> 20, only for candidates whose evidence remains
near the decision boundary. A candidate clearly above or below the boundary should stop early.

Maintain two roles when the capacity permits:

- a representative reservoir for dominant-use and ambiguity estimates;
- a diversity reservoir for rare but query-compatible senses.

### Gate 3 — lexical influence, risk, and abstention

Purpose: prevent a context-compatible term from corrupting the lexical candidate funnel.

#### 6.3.1 Lexical influence bound

Use postings and corpus statistics to estimate or bound the maximum additional lexical contribution of
term \(t\). If a valid bound proves that the term cannot affect the target depth, mark it dormant and
remove it cheaply.

This can prove limited lexical influence. It cannot prove relevance benefit. A term that can change the
ranking is merely active, not helpful.

#### 6.3.2 Calibrated risk

From held-out oracle data, estimate:

\[
p_{\mathrm{help}}(q,t)=\Pr(Y_M(q,(t,\mu))>\epsilon\mid X(q,t)),
\]

\[
p_{\mathrm{harm}}(q,t)=\Pr(Y_M(q,(t,\mu))<-\epsilon\mid X(q,t)).
\]

Use calibrated intervals or risk thresholds, not raw model scores. A term that is promising but risky
may be retained with lower weight or a constrained lexical form; it need not be discarded immediately.

#### 6.3.3 Abstention

The selector must be able to return:

- no expansion;
- fewer than the nominal number of terms;
- only low-risk terms;
- an original-query branch plus a limited expansion branch.

Baseline preservation is a first-class outcome, not a failure case.

## 7. Selection and weighting are separate decisions

Selection asks:

> Is this term sufficiently likely to provide useful reach without unacceptable risk?

Weighting asks:

> Given that the term is admitted, how much lexical influence should it receive?

Do not assume that the score best for selection is the score best for weighting. This distinction is
also emphasized in [Robertson's term-selection analysis](https://doi.org/10.1108/eb026866).

Recommended initial protocol:

1. select terms using opportunity, risk, influence, redundancy, and cost;
2. calibrate weights only for selected terms;
3. include \(\mu\) in the offline action label even if query-time implementation is staged later;
4. allow \(\mu=0\) when uncertainty is high;
5. do not set expansion weight equal to anchor weight by default.

Possible later weighting inputs:

- calibrated expected benefit;
- calibrated harm probability or lower-tail risk;
- context support and ambiguity;
- term specificity and posting cost;
- query specificity and candidate depth;
- redundancy and new-reach estimates.

Equal weighting is an ablation, not a theoretical consequence of high selection precision.

## 8. Set selection, redundancy, and interaction

Independent top terms may retrieve the same documents. A lower-ranked term may provide more useful new
reach. For an admitted set \(E\), use a portfolio objective such as:

\[
\operatorname{Score}(t\mid E)=
\operatorname{Opportunity}(t)
-\eta_{\mathrm{red}}\operatorname{Redundancy}(t,E)
 +\eta_{\mathrm{reach}}\operatorname{NewReach}(t,E)
-\eta_{\mathrm{cost}}\operatorname{PostingCost}(t).
\]

Here, \(\eta_{\mathrm{red}}\), \(\eta_{\mathrm{reach}}\), and \(\eta_{\mathrm{cost}}\) are nonnegative
development coefficients; they are hypotheses and require calibration. The set \(E\) contains terms
already admitted.

Measure:

- pairwise synergy and antagonism;
- overlap of sampled document IDs or posting sketches;
- displacement of relevant documents at the target depth;
- whether one anchor consumes the entire expansion budget;
- whether the best one-term action remains best after adding another term.

If interactions are large, independent per-term probabilities are not adequate. Use one-term-at-a-time
selection, a set-aware model, or a bounded combinatorial selector. Do not silently sum term utilities.

## 9. Injection alternatives

Selection alone cannot prevent lexical damage. Compare:

### A. Weighted interpolation

\[
Q'=(1-\lambda)Q+\lambda E.
\]

Original terms receive the dominant mass; \(\lambda=0\) is allowed.

### B. Separate retrieval branches

\[
R_0=\operatorname{BM25}(Q),\qquad R_E=\operatorname{BM25}(E).
\]

Reserve a declared portion of the candidate budget for \(R_E\), then deduplicate or fuse. This may
require extra lexical work, so it must be measured.

### C. Sense-constrained lexical emission

For high-risk terms compare:

1. weighted unigram;
2. corpus-supported phrase or compound;
3. proximity to an original query anchor;
4. bare unigram only in a restricted expansion branch.

The context should influence not only whether a term is selected but also how it enters the lexical
query.

## 10. What this explains about V8

The V8 failure is consistent with a selector that treats a context-free semantic score as if it were a
reliable estimate of retrieval utility:

- isolated-word BGE similarity is a weak proposal signal for many domains;
- fixed top-k retrieval from that signal does not express uncertainty or abstention;
- a compatible term can still have a harmful ambiguous posting list;
- nDCG and deep Recall have different useful-term sets;
- term interactions and displacement are not visible in an isolated similarity score;
- IDF and similarity selection scores should not automatically become lexical weights.

The conclusion is not that the static pool is useless. It is that pool availability, semantic
compatibility, lexical influence, and retrieval benefit are separate stages.

## 11. Experiment plan

All experiments must use held-out queries or datasets for final claims. Utility labels are development
signals and must not be used to tune the final test set.

### E1 — Gate-1 proposal recall

Compare BGE term, anchor, context-centroid, lexical, and hybrid proposals on the same pool.

Report:

- \(C_1\) size;
- helpful-term coverage for nDCG and Recall separately;
- candidate preparation and query cost;
- false-negative examples.

### E2 — Context aggregation

On an identical candidate set, compare maximum, top-2, top-k mean, representative support, and
opportunity-minus-risk scores.

Report candidate precision, harmful-term rate, rare-sense false rejection, and correlation with offline
utility.

### E3 — Negative evidence

Test whether representative incompatibility reduces harmful acceptance without removing useful minority
senses.

### E4 — Lexical influence bound

Test dormant-term pruning using posting statistics and score bounds.

Report bound validity, pruning rate, postings touched, and whether any potentially helpful term is
incorrectly removed.

### E5 — Adaptive context budget

Compare fixed 1/3/5/10/20 samples with adaptive 2 -> 4 -> 8 -> 20 sampling.

Report utility-risk-cost curves and unique embeddings, not only average cosine.

### E6 — Calibration

Train a simple held-out model for helpfulness and harm. Evaluate Brier score, reliability diagrams,
expected calibration error, and cross-dataset calibration drift.

### E7 — Abstention

Compare forced expansion with thresholded selective expansion. Report abstention, harmful acceptance,
baseline-preservation rate, nDCG change, and Recall change.

### E8 — Set interaction

Compare independent top terms, greedy new-reach selection, one-term actions, and bounded term pairs.

### E9 — Weighting

Only after E1-E8 show useful selection precision, sweep weights for accepted terms. Report both positive
gain and lower-tail harm.

### E10 — End-to-end cascade

Evaluate the selected cascade against the untouched BM25/DPH baseline, V7/V8 controls, and the relevant
QE baselines. Include preparation, memory, latency, postings, nDCG, and Recall.

## 12. Required metrics

### Selection metrics

- Gate-1 helpful-term coverage;
- Gate-2 accepted-term precision;
- harmful accepted-term rate;
- query-level set coverage: \(C(q)\cap H_M(q)\ne\varnothing\);
- dormant-term pruning precision;
- abstention rate;
- candidate and emitted-term counts.

### Retrieval metrics

- nDCG@10;
- MRR@10;
- Recall@100, @200, @500, @1000;
- per-query paired deltas;
- tied, improved, and degraded query counts;
- relevant-document displacement from the candidate funnel.

### Resource metrics

- unique contexts and term-context references;
- context reuse factor;
- sidecar size;
- preparation time, including every corpus pass and encoding;
- p50/p95 query latency;
- postings touched;
- peak host RAM and VRAM.

## 13. Guarantee boundaries

The following claims are allowed only with their assumptions stated:

1. A posting-list bound may guarantee limited lexical influence, not positive relevance.
2. A calibrated risk model can estimate harmful-selection probability; it is not a proof for an unseen
   query domain.
3. Conformal risk control can provide a marginal risk guarantee under exchangeability; it does not give
   a per-query guarantee.
4. Probabilistic top-k semantics can describe a constructed uncertainty model; they do not create a
   true utility distribution automatically.
5. Any confidence level must be reported as calibrated, heuristic, or unverified.

## 14. Decision rules

1. If Gate 1 cannot retain useful terms at acceptable cost, revisit pool construction or proposal
   generation before developing CRVE further.
2. If context reranking improves term-level precision but not end-to-end retrieval, treat it as a
   diagnostic selector rather than a successful QE method.
3. If selective expansion cannot beat baseline-preserving abstention, retire universal expansion.
4. If set interactions are large, do not use independent term probabilities.
5. Begin weighting work only after selection has measurable held-out precision and coverage.
6. Keep nDCG and Recall conclusions separate; neither is a substitute for the other.

## 15. Open debate questions

1. Is Gate 1 optimized for nDCG, Recall, or a declared two-objective coverage target?
2. Should the first implementation select one term, a small portfolio, or term-weight actions directly?
3. Is a lexical-influence bound affordable without introducing an unwanted first retrieval pass?
4. What context budget keeps preparation compatible with the edge-device objective?
5. Should high-risk terms be emitted as phrases/proximity constraints rather than unigrams?
6. What level of held-out calibration is enough to permit a nonzero expansion rate?
7. Which failure is acceptable: missed opportunity, abstention, or small ranking harm?

The immediate next experiment should answer E1: whether a better Gate-1 proposal can retain useful terms
that isolated-word BGE misses. CRVE should not be treated as the first gate; it is a precision and risk
stage after high-recall proposal generation.
