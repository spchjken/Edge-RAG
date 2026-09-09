# CRVE Design Refinement Notes

> Discussion record, 2026-09-09. This note preserves a design discussion for
> a later proposed-method session. It does not describe implemented behavior,
> replace the active architecture, or freeze the research plan. See
> [the capacity-bounded corpus-informed QE plan](corpus_informed_query_expansion_plan.md)
> for the current broader proposal.

## 1. Why this discussion exists

The archived V7 family established a useful starting point: select a static,
corpus-derived vocabulary, compare vocabulary terms with query anchors using
a frozen encoder, and inject a few selected terms into lexical retrieval. The
1,000-term salience pool was competitive with a much larger 50,000-term
bailout store, suggesting that an edge-oriented method should put a hard bound
on its static semantic sidecar rather than scale it with corpus size.

The current research direction retains that static, pre-retrieval property but
changes the representation. Rather than treating a vocabulary term as a single
meaning, it stores a small bounded memory of how that term is used in the
target corpus. At query time, the method proposes terms from the static
vocabulary and reranks them with query-to-usage-context evidence. The working
name is **Context-Reranked Vocabulary Expansion (CRVE)**.

This is not pseudo-relevance feedback: it performs no query-dependent first
retrieval over corpus documents before expansion. Its corpus representation is
prepared before queries arrive.

## 2. The central problem: compatible usage is not safe lexical injection

A sample can show that one use of a candidate term matches a query while the
bare term still matches many incompatible documents.

```text
Query:     JVM memory management
Candidate: garbage
Sample:    "The garbage collector reclaims unused heap objects."
```

The sample validates a programming sense of `garbage`. Adding the unqualified
term `garbage` to BM25 nevertheless adds score to documents about household
waste. Therefore:

\[
\exists c \in S_t: c \text{ matches } Q
\]

does not imply that `t` is globally safe or useful across its full posting
list.

This is not a reason to abandon context-based expansion. It identifies the
actual problem: preserve the extra semantic reach of a context-supported term
while controlling its ambiguous lexical footprint.

## 3. The objective remains joint ranking quality and candidate coverage

CRVE should not be reframed as a recall-only technique. Standalone ranking
quality, especially nDCG@10, remains important. At the same time, sparse
retrieval can be the candidate generator for a downstream pipeline, so deeper
candidate recall also matters.

For a fixed cutoff \(K\), a candidate term is useful only when it improves the
retrieval funnel rather than merely matching more documents:

\[
\Delta R@K(t,Q) = R@K(Q+t)-R@K(Q) > 0.
\]

Recall is not monotonic under expansion. Noisy postings can displace relevant
documents below the same cutoff. Conversely, a term that slightly harms
standalone nDCG may introduce otherwise missed relevant documents that a
downstream reranker can recover. Both outcomes must be measured; neither
metric can substitute for the other.

The desired result is a Pareto improvement, or a clearly declared trade-off:

- preserve or improve nDCG@10 and MRR@10;
- improve Recall@100/200/500/1000 where candidate depth is operationally
  meaningful;
- account for postings touched, retrieval latency, sidecar size, and the cost
  of any downstream stage.

A strong deployment claim is not merely "higher Recall@1000." It is, for
example, reaching a baseline's recall at a materially smaller candidate depth,
or improving recall at the same depth without meaningful nDCG or latency loss.

## 4. Separate opportunity, reliability, and lexical impact

The context memory should not produce one opaque similarity score. It should
separate three questions.

| Quantity | Question answered | Main failure if used alone |
|---|---|---|
| Term prior | Is the vocabulary term semantically plausible for the query? | Context-free embeddings remain polysemous and favor generic neighbors. |
| Contextual opportunity | Does at least one corpus usage support the query-relevant sense? | A single accidental match can dominate. |
| Representative reliability / risk | How much observed corpus usage appears compatible or incompatible? | Can suppress rare but valuable technical senses. |

Let \(s_0(t,Q)\) be the initial term-to-query or term-to-anchor score, and
let \(s_j(t,Q)\) be similarity between the query and sample \(j\) for term
\(t\). The first score is a **prior**, not a score to discard and not a
guarantee of safety.

### 4.1 Preserve the initial term score, but do not let it dominate

Start with a normalized fusion:

\[
S(t,Q) = \alpha z(s_0(t,Q)) + (1-\alpha) z(C(t,Q)).
\]

This may regularize context reranking: the term must be both semantically
plausible and supported by a compatible corpus usage. It becomes harmful if
\(\alpha\) is so large that the method effectively returns to term-only QE.

Test three bounded alternatives:

1. **Normalized linear fusion:** a transparent baseline.
2. **Rank fusion:** combine term-only and context-only ranks, avoiding a
   brittle assumption that their cosine-score scales are comparable.
3. **Contextual revision:** use the term score as a prior and let positive or
   negative contextual evidence revise it:

   \[
   S(t,Q)=s_0(t,Q)+\lambda\Delta_{\text{context}}(t,Q).
   \]

The third formulation is the clearest conceptual account of CRVE.

### 4.2 Aggregation functions express different hypotheses

| Aggregation | Meaning | Likely behavior |
|---|---|---|
| \(\max_j s_j\) | One useful sense exists. | Retains rare senses; vulnerable to accidental evidence. |
| Mean of top-2 or small top-\(k\) | Useful evidence is repeated. | Safer for shallow ranking; can penalize rare senses. |
| Mean over representative samples | A random observed usage is compatible. | Measures broad reliability; misses useful minorities. |
| Sigmoid support | Soft fraction of samples above a compatibility threshold. | Can express calibrated support; depends on threshold and temperature. |

For sigmoid support, use:

\[
p_j = \sigma\left(\frac{\hat{s}_j-\tau}{T}\right),
\qquad
C_{\text{support}}(t,Q)=\frac{1}{R}\sum_{j=1}^{R}p_j.
\]

Here \(\hat{s}_j\) should ideally be query-normalized, for example relative
to a fixed null distribution of unrelated contexts. Applying a sigmoid to raw
cosines does not itself produce a calibrated probability. \(\tau\) and \(T\)
must be fixed or selected only with declared development data.

### 4.3 Penalize low compatibility as distributional risk, not one bad sample

Do not punish the single lowest-scoring context. Broad terms almost always
have some unrelated use. Instead retain both opportunity and risk:

\[
O(t,Q)=\operatorname{TopMean}_k(p_1,\ldots,p_R),
\]

\[
A(t,Q)=1-\frac{1}{R}\sum_{j=1}^{R}p_j,
\]

\[
S(t,Q)=\alpha s_0+\beta O-\gamma A.
\]

`garbage` can then have high opportunity and high risk: it should not
necessarily be discarded, but should receive a lower lexical weight, a safer
query representation, or a separate expansion channel. The strength of the
risk penalty can be different for shallow nDCG-oriented retrieval and
deep-recall-oriented candidate generation.

## 5. Use two context memories, not one overloaded sample set

An occurrence-uniform reservoir is useful for estimating dominant usage and
ambiguity. A diversity-oriented sample set is useful for discovering a rare
query-compatible sense. Neither alone supports both inferences.

Under a fixed per-term cap, retain two roles:

- **Representative reservoir:** approximately occurrence- or document-weighted
  samples. Use it to estimate compatible usage mass and ambiguity risk.
- **Diversity reservoir:** semantically or lexically diverse samples. Use it
  to discover compatible minority senses and measure opportunity.

For an initial 20-context budget, a 12 representative / 8 diverse split is a
reasonable hypothesis, not an established default. It should be compared with
single-reservoir alternatives. Diversity sampling must not be used to infer
corpus sense frequencies; representative sampling must not be assumed to find
rare senses reliably.

This makes a defensible division of labor:

\[
O(t,Q)=\operatorname{TopMean}_k(\text{diverse samples}),
\qquad
A(t,Q)=1-\operatorname{MeanCompatibility}(\text{representative samples}).
\]

## 6. Select a useful term set, not independent individually good terms

The top individual candidates may be redundant:

```text
JVM, Java virtual machine, virtual machine, Java VM, JVM runtime
```

They can all access nearly the same documents, while a lower-ranked term such
as `heap allocation` may reach a distinct relevant region. Select expansions
as a bounded portfolio:

\[
t^*=\arg\max_t\left[S(t,Q)-\eta\operatorname{Redundancy}(t,E)
+\rho\operatorname{NewReach}(t,E)-\mu\operatorname{PostingCost}(t)\right],
\]

where \(E\) contains already admitted terms.

Candidate signals include:

- semantic redundancy among terms or their contexts;
- overlap of sampled document identifiers or compact posting-list sketches;
- estimated lexical reach beyond the original query and current expansion set;
- document frequency or decoded-postings cost;
- query-aspect coverage, so one easy anchor does not absorb the full budget.

Embedding-space diversity is only a proxy. Two terms can be semantic neighbors
but have usefully different postings; a semantic outlier can be an OCR artifact
rather than genuine novel reach.

## 7. Control how a selected term enters lexical retrieval

Term selection alone cannot fully protect ranking. The injection mechanism
should reserve a role for the original query.

### 7.1 Weighted interpolation

Use a standard weighted lexical query in which original terms receive larger
weights than expansions:

\[
Q'=(1-\lambda)Q+\lambda E.
\]

\(\lambda\) can be query-adaptive and should permit zero expansion. This is
the lowest-cost safety control and the natural first integration baseline.

### 7.2 Original and expansion branches

Retrieve from the original query and the expansion terms separately, then fuse
or reserve capacity for each branch:

\[
R_0=\operatorname{BM25}(Q),
\qquad R_e=\operatorname{BM25}(E).
\]

The candidate set can, for example, reserve most capacity for \(R_0\) and a
smaller declared fraction for \(R_e\), deduplicate, and rerank if a downstream
stage exists. This explicitly limits expansion's ability to destroy original
top-ranked results while giving it a route to add new documents. It may require
a second lexical retrieval, so its real latency must be measured rather than
assumed negligible.

### 7.3 Preserve the intended sense in the query representation

For high-risk terms, emitting an unqualified unigram is needlessly lossy.
Compare increasingly constrained lexical forms:

1. weighted unigram;
2. corpus-supported compound or phrase, such as `garbage collector`;
3. proximity condition, such as `garbage` near `heap` or `JVM`;
4. bare ambiguous unigram only in the expansion branch.

This is the most direct response to the central ambiguity problem: use context
not just to select a term, but to choose a lexical representation that retains
some of its intended sense.

## 8. Query-adaptive abstention and budget allocation

Specific lexical queries often need no help; vague or vocabulary-mismatched
queries may benefit more. The system must permit:

- no expansion;
- fewer than the nominal term count;
- lower weights or constrained representations for risky terms;
- different risk tolerance at different candidate depths.

Expansion weight can depend on candidate confidence, opportunity, ambiguity,
query specificity, redundancy, estimated novel reach, and posting cost. A
query-adaptive gate is more credible than forced expansion with 3--10 terms.

## 9. Focused experimental path

Do not create a large unstructured factorial sweep. Hold candidate generation
fixed when comparing rerankers and add one mechanism at a time:

| Step | Question | Variant |
|---|---|---|
| 1 | Does corpus context help term selection? | term-only vs context-only max vs top-2 |
| 2 | Does retaining the term prior help? | term-only/context fusion or rank fusion |
| 3 | Does negative evidence control ambiguity? | opportunity-only vs opportunity minus representative risk |
| 4 | Does set selection add retrieval value? | independent top terms vs redundancy/new-reach portfolio |
| 5 | Does controlled injection preserve ranking? | weighted interpolation vs two-branch fusion/reserved capacity |
| 6 | Does preserving sense matter? | unigram vs phrase/proximity for high-risk terms |

For every step, report nDCG@10, MRR@10, Recall@100/200/500/1000, p50/p95
latency, postings touched, emitted term count, and query-level transitions:

- improves both nDCG and recall;
- improves recall but harms nDCG;
- improves nDCG but not recall;
- harms both.

Additionally report term-level utility on development data, harmful-term rate,
candidate-reranker precision, and the score/risk distributions of admitted
terms. The final evaluation protocol must preserve the repository's canonical
metrics and fixed retrieval-depth contract.

## 10. Working research question

The resulting question is stronger and more precise than "can contexts rerank
vocabulary terms?":

> How can evidence that a term has a query-compatible corpus usage be converted
> into safe and useful lexical expansion over a finite inverted-index candidate
> budget?

The answer may be a lightweight scoring method, a safer lexical emission
mechanism, or a combination. The novelty should rest on converting static
usage-context evidence into controlled lexical reach—not on claiming that term
embedding, context similarity, or query expansion individually are new.
