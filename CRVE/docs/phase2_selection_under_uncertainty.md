# Phase 2 — Query-Conditioned Expansion-Term Selection Under Uncertainty

> Status: research design and debate record, 2026-09-17, with Gate 1 outcome update on 2026-09-25. This note covers the phase after static
> vocabulary-pool construction and before final lexical weighting. It does not describe implemented
> behavior or freeze a final method. [ARCHITECTURE.md](ARCHITECTURE.md) owns current stage boundaries.
> The frozen Gate 1 pool uses DF >= 2, CF >= 3 and size `min(10,000, |V_eligible|)`;
> its deployable candidate cap is 200.

The current Gate 1 proposer study is concluded as exploratory research without a frozen-protocol
pass. Run 2 and Round 4 provide enough retained opportunity to proceed to Gate 2 research. The
[results and handoff](GATE1_RESULTS_AND_HANDOFF.md) distinguish the unmet original floors and failed
Checkpoint B from the later development milestones, which Extended and Lexical2 meet at L=500 in
the committed budget curves. Larger caps remain exploratory; final policy and budget are open.
The design alternatives below remain hypotheses unless the results record identifies them as tested.

CRVE means **Context-Reranked Vocabulary Expansion**, the current working name for the proposed
method.

Related documents:

- [Capacity-bounded corpus-informed QE plan](corpus_informed_query_expansion_plan.md)
- [CRVE design refinement notes](crve_design_refinement_notes.md)
- [Gate 1 results and handoff to Gate 2](GATE1_RESULTS_AND_HANDOFF.md)
- Former Phase-2 implementation plan (not present in the active CRVE checkout); use the
  [research roadmap](CRVE_RESEARCH_ROADMAP.md) and [architecture](ARCHITECTURE.md).

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

Use two different thresholds:

- \(\tau\): a small numerical tolerance that distinguishes a measured change from floating-point or
  ranking-tie noise;
- \(\delta>\tau\): a practical-effect threshold that defines a materially useful term.

These thresholds must not be conflated. The existing oracle value \(\tau=10^{-5}\) detects a positive
change; it does not establish that the change is operationally meaningful. Use \(\Pr(\cdot)\) for
probability; it is distinct from the vocabulary pool \(\mathcal{P}\).

For a single action \(a\):

\[
H_M(q,a;\delta)=1\{Y_M(q,a)\ge\delta\},
\]

\[
G_M(q,a;\epsilon)=1\{Y_M(q,a)<-\epsilon\}.
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

### 5.1 Safe oracle labels and material-gain thresholds

Because selection and weighting are separate phases, a Gate-1 oracle label means that a term has at
least one useful tested weight. It does not prescribe the deployed weight. For ranking-oriented
evaluation, define the safe individual-term opportunity:

\[
g^{\mathrm{rank}}_{q,t}
=
\max_{\mu:\,\Delta R@1000(q,t,\mu)\ge-\tau}
\Delta\mathrm{nDCG}@10(q,t,\mu).
\]

The materially helpful term set at threshold \(\delta\) is:

\[
H_q^{\mathrm{rank}}(\delta)
=
\{t\in\mathcal{P}:g^{\mathrm{rank}}_{q,t}\ge\delta\}.
\]

Do not report only one post-hoc threshold. The initial sensitivity grid is:

\[
\delta_{\mathrm{nDCG}}\in\{10^{-5},0.001,0.005,0.01,0.02\}.
\]

The primary practical threshold must be frozen on development data before final evaluation. A
provisional primary value is \(\delta=0.005\), subject to development-set validation. Interpret the
grid as usefulness tiers rather than treating every positive delta as equally valuable:

| Opportunity tier | Safe individual-term \(\Delta\mathrm{nDCG}@10\) | Interpretation |
|---|---:|---|
| No positive opportunity | \(g\le\tau\) | no measurable safe improvement |
| Marginal | \(\tau<g<0.001\) | positive but practically negligible |
| Small | \(0.001\le g<0.005\) | weak opportunity |
| Material | \(0.005\le g<0.01\) | meaningful opportunity |
| Strong | \(g\ge0.01\) | high-value opportunity |

Run 2 froze delta=0.005, rho=0.90, tau=1e-5 and epsilon=0.001, with weights
{0.05, 0.10, 0.30, 0.50, 1.00}. Its measured helpful sets and ceilings use the evaluated reference
universe \(\mathcal R_q\), rather than the full pool \(\mathcal P\) in the general formulas here.
Use ReferenceBOR and reference addressability for those results. The implemented material near-best
set is \(\{t\in\mathcal R_q:g^{\mathrm{rank}}_{q,t}\ge\max(0.005,0.90\hat g_q^*)\}\), where
\(\hat g_q^*\) is the best safe individual gain in \(\mathcal R_q\).

Harm is a separate axis. A term can have a helpful weight and a harmful weight, so a term-level maximum
cannot create a mutually exclusive helpful/dormant/harmful partition. Classify those three outcomes
only for a fixed action \((t,\mu)\) or a frozen weighting policy: helpful when gain is at least
\(\delta\), harmful when it is below \(-\epsilon\), and neutral or marginal otherwise. Across weights,
report both best safe opportunity and worst-case or deployed-action harm.

Recall changes are discrete and depend on the number of relevant documents. Therefore, do not impose
one universal decimal \(\Delta R@K\) threshold. Define a recall-helpful term through document counts:

\[
H^{\mathrm{rec}}_{q,K}
=
\left\{t:\exists\mu,\;
\Delta\#\mathrm{RelevantDocs}@K(q,t,\mu)\ge1
\land
\Delta\mathrm{nDCG}@10(q,t,\mu)\ge-\epsilon
\right\}.
\]

Report stronger recall tiers separately, such as net recovery of at least two relevant documents or a
declared relative-recall increase. This avoids treating the same decimal recall change as equivalent
for queries with one and hundreds of relevant documents.

### 5.2 Gate-1 recall, precision, and utility retention

Let \(C_{1,L}(q)\) be the top \(L\) terms surviving Gate 1. For either a ranking-helpful or
recall-helpful oracle set \(H_q\), report:

\[
\operatorname{TermRecall}@L(q)
=
\frac{|C_{1,L}(q)\cap H_q|}{|H_q|},
\]

\[
\operatorname{TermPrecision}@L(q)
=
\frac{|C_{1,L}(q)\cap H_q|}{|C_{1,L}(q)|},
\]

and query-level opportunity coverage:

\[
\operatorname{Hit}@L(q)
=
1\{C_{1,L}(q)\cap H_q\ne\varnothing\}.
\]

Recall asks how many known useful terms survived. Precision asks what fraction of all survivors are
known to be useful. Gate 1 prioritizes recall, but precision still controls Gate-2 cost and the number
of harmful opportunities passed downstream. Compare methods across a candidate-budget curve, initially
\(L\in\{10,20,50,100,200,500\}\), rather than at one favorable cutoff.

If \(|H_q|=0\), TermRecall and Hit are `NA`, not zero: Gate 1 cannot recover an opportunity that does
not exist. Report the addressable-query rate separately and macro-average recall over addressable
queries, with explicit numerators and denominators. If the full-pool best gain is zero, BOR is likewise
`NA`. TermPrecision remains defined when \(|C_{1,L}(q)|>0\), including on unaddressable queries.

Also report best-opportunity retention:

\[
\operatorname{BOR}_M@L(q)
=
\frac{\max_{t\in C_{1,L}(q)}g^M_{q,t,+}}
{\max_{t\in\mathcal{P}}g^M_{q,t,+}},
\]

where \(x_+=\max(x,0)\). Compute this separately for nDCG@10, Recall@100, and Recall@1000. A
threshold-light secondary diagnostic is individual utility retention:

\[
\operatorname{UtilityRecall}_M@L(q)
=
\frac{\sum_{t\in C_{1,L}(q)}g^M_{q,t,+}}
{\sum_{t\in\mathcal{P}}g^M_{q,t,+}}.
\]

This last quantity is not a candidate-set utility: it double-counts redundant terms and sums
counterfactual gains measured independently against the same baseline. It may be used only as a quick
diagnostic with the current per-term audit.

### 5.3 Redundancy-aware gold-document opportunity coverage

Raw term recall can reward many terms that all recover the same relevant document. For cutoff \(K\),
let:

\[
A_{q,t,K}
=
\{d\in G_q:\operatorname{rank}_0(d)>K
\land\exists\mu,\operatorname{rank}_{t,\mu}(d)\le K\},
\]

where \(G_q\) is the judged relevant-document set. Define:

\[
\operatorname{DocOpportunityRecall}@L,K(q)
=
\frac{\left|\bigcup_{t\in C_{1,L}(q)}A_{q,t,K}\right|}
{\left|\bigcup_{t\in\mathcal{P}}A_{q,t,K}\right|}.
\]

The union counts two terms recovering the same gold document once, while rewarding terms that recover
different gold documents. Also report absolute missed-gold recovery by replacing the denominator with
\(|G_q\setminus B_q^K|\), where \(B_q^K\) is the baseline top-\(K\) relevant-document set.

For a graded, rank-sensitive opportunity measure, let \(v_{q,t}(d)\) be the maximum positive DCG
improvement contributed to relevant document \(d\) over safe tested weights, and define:

\[
F_q(C)=\sum_{d\in G_q}\max_{t\in C}v_{q,t}(d).
\]

Then report \(F_q(C_{1,L})/F_q(\mathcal{P})\). The per-document maximum prevents repeated credit for
the same document. This remains an opportunity-coverage diagnostic, not the exact nDCG of injecting
all terms together; exact set utility still requires executing the combined expanded query.

The Phase-1 `pool_candidate_audit.parquet` contains aggregate per-term nDCG and Recall deltas but not
gold-document identities or rank transitions. This is the Phase-1 audit limitation. The completed
Gate 1 Run 2 additionally records cutoff entries and labels every term in its evaluated reference
universe across the five frozen weights; the coverage artifact reports zero missing action triples.
Its cutoff artifact supports raw and safe DocOpportunityRecall relative to that reference universe.
Neither audit measures the retrieval utility of jointly injecting a multi-term set.

The Phase-1 audit evaluates terms extracted from judged relevant documents rather than every pool
term. Run 2 supplements that evidence with all operational channel top-500 proposals. An output
outside the evaluated universe remains unlabeled and requires separate evidence before assigning it
an exact helpful, dormant or harmful status. Report a
mutually exclusive survivor decomposition only under a fixed term-weight action or frozen weighting
policy:

\[
1=P_{\mathrm{helpful}}+P_{\mathrm{neutral/marginal}}+P_{\mathrm{harmful}}.
\]

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

The sidecar should store small canonical context chunks once and a term-to-context-ID map. Chunk
segmentation is independent of the target term; do not save overlapping term-centered windows for
nearby terms. If many terms occur in the same chunk, one embedding is reused. Track:

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

Sweep 5, 10, 15, 20, and 30 retained canonical chunks per term. At each capacity, retain all distinct
eligible chunks when fewer exist; otherwise use a declared representative/diversity policy. At query
time, fetch and score all retained evidence for a candidate in one batch. Do not introduce iterative 8 -> 16 -> 30 scoring: the
embeddings are already local, so repeated rounds add control-flow and aggregation overhead without
avoiding the embedding work. Deduplicate context IDs across candidate terms before scoring so a shared
context is evaluated once.

Maintain two roles when the capacity permits:

- a representative reservoir for dominant-use and ambiguity estimates;
- a diversity reservoir for rare but query-compatible senses.

#### 6.2.1 Optional statistical truncation evidence (deferred Gate 2+ work)

The stored sample may later support an additional optimistic pruning signal. Let

\[
M_F(q,t)=\max_{c\in\mathcal C_t}F(q,c)
\]

be the unknown best compatibility over all corpus contexts for term \(t\). When every distinct context
is stored, \(M_F\) is observed exactly. When only 30 of more than 30 contexts are stored, the observed
sample maximum is a lower bound, not an upper bound. A future method may estimate an optimistic upper
value \(U_F^{(1-\alpha)}(q,t)\) at a declared level such as 95% or 97.5%, under explicit sampling,
distributional, or held-out calibration assumptions.

This estimate can be used like a probabilistic top-k pruning bound:

\[
U_F^{(1-\alpha)}(q,t)<\tau_F
\quad\Longrightarrow\quad
\text{discard }t,
\]

or, for a fixed candidate budget, discard \(t\) when its optimistic value is below the current
competitive boundary. This only states that the term is unlikely to contain a sufficiently compatible
context under the declared model. It does not prove that the term cannot improve retrieval.

Illustrative examples:

- If 30 scores are uniformly low, with observed maximum 0.29 and an estimated 97.5% optimistic value
  of 0.34, a declared compatibility threshold of 0.55 would prune the term.
- If the observed maximum is 0.71 and the upper estimate is higher, the statistical rule cannot prune
  it even when the other contexts are weak; support and ambiguity remain separate evidence.
- A very frequent term may receive a high maximum estimate simply because it has many opportunities
  for an extreme context. Its high DF simultaneously lowers IDF and lexical discrimination. Carry
  this lexical information to the later influence/weighting decision rather than treating a high
  estimated semantic maximum as sufficient evidence of usefulness.

The initial study should remain low-dimensional: compare observed maximum, top-2 mean, and at most one
declared upper-estimation rule. Do not begin with a high-dimensional learned utility predictor over
maximum, mean, variance, DF, CF, IDF, occurrence count, and lexical bounds; these variables are strongly
related and invite overfitting on the limited oracle data. Judge the optional bound by its
pruning-versus-material-helpful-recall curve. If its conservative upper values saturate or it removes
too few candidates to reduce later work, retire it rather than retaining it for theoretical appearance.

### Gate 2 lexical checks and Gate 3 abstention

The lexical-influence and calibrated-risk checks below are candidate Gate 2 rejection signals;
Gate 3 owns the separate, still-open weighting choice and abstention decision. These checks do not
constitute an implemented or selected Gate 3 method.

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

Possible study protocol, not a selected weighting policy:

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

The initial proposer study is completed through Run 2 and Round 4 offline analysis. BGE whole-query
and anchor proposals, lexical co-occurrence, sparse context profiles, acronym rescue and fusions were
compared on the same pool. Context-centroid and other additional proposals remain deferred hypotheses.
The [results and handoff](GATE1_RESULTS_AND_HANDOFF.md) closes this study without a frozen-protocol
pass; meeting the old floors is not a prerequisite for beginning E2/E3/E5 Gate 2 research.

For any later proposal study, report:

- \(C_1\) size and results over the declared \(L\) budget curve;
- ranking-safe and recall-safe TermRecall and TermPrecision separately;
- any-positive, material, and strong usefulness-threshold sensitivity;
- query-level Hit@\(L\) and best-opportunity retention;
- individual utility retention as a secondary, explicitly redundancy-blind diagnostic;
- distinct-gold DocOpportunityRecall from the recorded cutoff entries;
- helpful, dormant, and harmful survivor rates when full candidate labels are available;
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

### E5 — Fixed context-capacity sweep

Compare 5/10/15/20/30 retained canonical chunks per term, fetching and scoring all retained evidence
in one batch at query time. The former adaptive `2 -> 4 -> 8 -> 20` proposal is retired.

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

- Gate-1 TermRecall and TermPrecision across candidate budgets and usefulness thresholds;
- Gate-1 query-level Hit and best-opportunity retention;
- Gate-1 distinct-gold DocOpportunityRecall and absolute missed-gold recovery;
- opportunity-tier distribution: no positive opportunity, marginal, small, material, and strong;
- fixed-action or frozen-policy helpful, neutral/marginal, and harmful survivor rates;
- individual utility retention, labeled as redundancy-blind;
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

1. Use retained opportunity and measured downstream cost to decide whether proposal work must reopen.
   The current Gate 1 study advances to Gate 2 research despite unmet frozen criteria; it does not
   require further budget increases to earn a pass. Revisit proposal if its coverage or cost actually
   prevents a useful downstream result.
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

The immediate next study concerns Gate 2: can context evidence improve precision and reject harmful
actions while retaining the opportunities already present in Gate 1 candidates at an affordable cost?
Reuse the existing candidate and action evidence for development, measure additional opportunity loss,
and preserve abstention. CRVE names the full proposed selection-and-weighting cascade; Gate 2 owns
context verification within it.
