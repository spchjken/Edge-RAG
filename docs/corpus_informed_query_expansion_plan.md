# Capacity-Bounded Corpus-Informed Query Expansion — Research Plan

> Status: revised research proposal, 2026-09-09. Formerly
> `expansion_platform_plan.md` and then `corpus_informed_query_expansion_plan.md`.
> This document also subsumes the scope of the former documentation-revision
> implementation plan.
> Pipeline V2 V7 is an archived implementation and experimental control, not an active experiment.
> This document proposes future research; it does not describe an implemented system or supersede
> [the current architecture](ARCHITECTURE.md).

## 1. Research objective and deployment setting

Investigate whether a compact, reusable representation of how selected corpus terms are used can
improve BM25 query expansion without task-specific training, a first-pass document retrieval, or a
dense document index. The primary proposed method is a capacity-bounded term-context memory:

`stream corpus -> select vocabulary -> store representative term contexts -> generate and rerank expansion terms -> one BM25 retrieval`

The target is not universal superiority over learned dense or sparse retrieval. The research question
is whether corpus-informed lexical expansion offers a useful effectiveness/resource trade-off when
corpora must become searchable quickly under a 15 GiB host-memory ceiling.

Report two deployment settings separately:

- **Rapidly onboarded or ephemeral collections:** measure total time to usable retrieval, including
  every corpus pass, vocabulary construction and context encoding.
- **Persistent collections:** additionally measure amortized preparation, representation size,
  startup, updates and query latency.

The method uses a pretrained encoder, so the correct claim is **no additional task-specific training**,
not “untrained” or “learning-free.” Any use of relevance labels for selection or tuning must be
restricted to declared development data.

## 2. Positioning and novelty boundary

Query expansion can use three evidence sources:

| Evidence source | Query-time behavior | Representative family |
|---|---|---|
| Query or external knowledge | Propose text without consulting the target corpus | WordNet, LLM synonyms, Query2doc |
| Query-dependent corpus documents | Retrieve documents and extract feedback | RM3, Bo1, CEQE |
| Reusable corpus representation | Select from a structure built before the query | Global vocabulary or term-context memory |

The individual ingredients are established: global corpus analysis, embedding-based vocabulary QE,
whole-query similarity, multi-term agreement, contextual word representations, two-stage candidate
ranking and query-model interpolation. In particular:

- [Qiu and Frei](https://doi.org/10.1145/160688.160713) expanded toward a corpus-derived query
  concept rather than independent query terms.
- [Zamani and Croft](https://doi.org/10.1145/2970398.2970405) evaluated average-query-vector
  expansion and multiplicative compatibility with all query terms.
- [CEQE](https://arxiv.org/abs/2103.05256) scores contextualized term occurrences from
  pseudo-relevant documents.
- [Query2doc](https://aclanthology.org/2023.emnlp-main.585/) and related methods use generated
  pseudo-documents; corpus-steered variants such as
  [CSQE](https://aclanthology.org/2024.eacl-short.34/) use initially retrieved documents.

The proposed contribution must therefore be stated narrowly:

> A capacity-bounded, target-corpus term-context memory constructed during streaming preparation and
> used to rerank embedding-generated lexical expansion candidates, without task-specific training,
> first-pass document retrieval, or document-level neural indexing.

This is a plausible method-and-systems contribution, not a proven priority claim. A wider literature
audit is required before using “first.” The paper must not claim the first contextual QE method, the
first two-stage QE method, or a new embedding model.

## 3. Archived V7 evidence and lessons

V7 uses analyzed query anchors, POS weights, frozen BGE-small term embeddings, a 1,000-term salience
pool, a 50,000-term bailout store, IDF damping and weighted lexical retrieval. Its results are
development evidence for the new design, not ongoing experiments or directly reusable final numbers.

### 3.1 Large vocabulary tail

The archived [bailout grid](../results/legacy/v7_legacy/v7_calibration/v7_bailout_grid_summary.md)
compared the 1,000-term pool with gated access to a 50,000-term store across ten document-level
benchmarks:

| Condition | Strict@10 | DocRec@10 | Mean latency |
|---|---:|---:|---:|
| 1k pool, bailout off | 62.34% | 48.83% | 11.39 ms |
| Selected bailout, similarity 0.80 and IDF 3.0 | 62.94% | 49.35% | 28.55 ms |

The mechanism-specific gain was about 0.6 percentage points while latency was about 2.5 times higher.
This supports diminishing value in the tail and argues against another 50k semantic sidecar. It does
not prove that every 10k or 15k pool is optimal: bailout used particular gates and allocation rules.

### 3.2 Salience versus coverage

The archived [pool-selection study](../results/legacy/v7_legacy/v7_calibration/coverage_vs_salience_pool_summary.md)
found at pool size 1,000:

| Selection | Strict@10 | DocRec@10 | Strict@50 | Mean latency |
|---|---:|---:|---:|---:|
| Semantic coverage/FPS | 62.73% | 49.54% | 74.79% | 8.10 ms |
| Salience | 63.16% | 49.80% | 74.54% | 10.12 ms |

Salience was slightly stronger for direct expansion at rank 10; coverage was competitive, slightly
stronger at Strict@50 and faster. The new cascade changes the first stage from final term selection to
high-recall candidate generation, so coverage deserves reconsideration. The identical reported
“In-Pool Hit” value across all strategies and sizes is not discriminative evidence and must be audited
before reuse.

### 3.3 Scaling lesson

Archived [large-corpus results](../results/legacy/v7_legacy/v7_large_scale/v7_streaming_large_summary.md)
show severe V7 latency on multi-million-document collections. Profiling identified corpus-sized score
accumulation in the legacy streaming scorer as a likely contributor, but expansion fan-out and postings
touched also matter. A small expansion weight still triggers a posting list. The new method must bound
both semantic computation and final lexical work.

## 4. Primary proposed method

Call the working method **Context-Reranked Vocabulary Expansion (CRVE)** until naming is revisited.

### 4.1 Index-time phase

1. Stream the corpus into the standard lexical index and collect term statistics.
2. Construct an eligible vocabulary and select at most 10,000–15,000 terms.
3. Collect bounded, document-diverse usage contexts for each selected term.
4. Select up to 20 representative samples or prototypes per term.
5. Encode selected terms and contexts once with a frozen encoder and store a versioned sidecar.

### 4.2 Query-time phase

1. Analyze the query and encode its anchors and/or whole-query representation once.
2. Search the capped term matrix and retain only the top candidate set, initially 50–100 terms.
3. Gather precomputed contexts for those candidates; do not load or encode raw sample text.
4. Rerank candidates using term similarity and query-to-context compatibility.
5. Admit only 3–10 final expansion terms under weight and postings-cost budgets.
6. Retrieve once with standard weighted BM25.

This is a reranking cascade *inside QE*. It is not PRF because it does not retrieve query-dependent
documents before expansion.

## 5. Capacity-bounded vocabulary construction

Set a hard semantic vocabulary ceiling rather than a corpus-size-dependent fraction:

`V_cap in {1k, 5k, 10k, 15k}`

The final value must be selected on development data using a preregistered plateau rule. If 10k is
within the declared tolerance of 15k, prefer 10k.

Before ranking, enforce eligibility conditions that can provide reliable lexical output and context:

- exact compatibility with the retrieval analyzer and a canonical surface form;
- minimum distinct-document support and enough valid occurrences for context collection;
- removal of stopwords, malformed tokens, boilerplate and obvious extraction artifacts;
- declared handling for compounds, acronyms, entities and versioned identifiers;
- an optional maximum-DF filter or explicit posting-cost penalty for extremely common terms.

Compare three pool constructors:

1. **Salience:** the frozen V7 control, based on IDF and document frequency.
2. **Coverage:** semantic coverage/FPS after eligibility filtering.
3. **Hybrid weighted coverage:** retain a salient core, then cover the residual semantic space while
   weighting represented terms by corpus importance.

Pure coverage may waste slots on embedding outliers, while pure salience may omit domain-specific
vocabulary. The hybrid is a hypothesis, not the assumed winner. Pool construction and query-time
candidate scoring are separate ablations.

## 6. Collecting representative term contexts

### 6.1 What a sample represents

For a selected term `t`, a raw sample is a short occurrence window centered on an exact analyzed-term
match, initially up to 50 tokens before and 50 tokens after the target. Preserve sentence and document
boundaries; never cross documents. Compare fixed windows with sentence-bounded and +/-16 or +/-32
token alternatives because a 101-token passage may dilute the target term.

A BGE embedding of the whole window is a **context-window** or **usage-context embedding**. It is not
technically a contextualized term embedding: the latter is the hidden state of the target token after a
Transformer processes its context. Use precise terminology.

The primary representation should make the target explicit in natural language, for example:

`Target term: garbage. Usage context: The garbage collector reclaims unused JVM heap objects.`

Compare this with an unmarked window. Do not introduce special target tokens unless the frozen encoder
is known to understand them.

### 6.2 Deterministic collection

Vocabulary selection depends on complete corpus statistics, so the simplest exact design uses two
streaming passes:

1. **Pass 1:** build lexical statistics and select the capped vocabulary.
2. **Pass 2:** collect contexts only for selected terms.

The second pass is memory-safe but not free; include it in time to usable retrieval. A later one-pass
approximation may maintain reservoirs for provisional terms, but it must be compared for missed terms,
memory and preparation time rather than assumed equivalent.

During context collection:

- accept at most one occurrence of a term per document;
- normalize and hash windows to reject exact or near duplicates;
- reject windows with insufficient natural-language content or excessive markup;
- use a fixed seed and deterministic reservoir sampling;
- record occurrence count, distinct-document count and rejection reasons;
- never retain an unbounded list of occurrences.

### 6.3 Choosing up to 20 samples

Use “up to 20,” not exactly 20. Rare or semantically consistent terms may require only one to three
samples; polysemous frequent terms may justify more.

Compare progressively more expensive selectors:

1. **First valid occurrences:** diagnostic lower bound with ordering bias.
2. **Document-diverse reservoir:** primary low-preparation-cost baseline, capped directly at 20.
3. **Cheap diverse reservoir:** maintain or collect a slightly larger bounded pool and select contexts
   using lexical/Jaccard or hashed-vector diversity before neural encoding.
4. **Semantic representative selection:** encode a bounded temporary reservoir, cluster it and retain
   medoids or use a facility-location objective. This is conditional on its preparation cost.

For temporary occurrence set R_t and retained samples S_t, a representative objective is:

`argmax_{|S_t| <= 20} sum_{c in R_t} max_{s in S_t} cosine(E(c), E(s))`

Store each retained sample's represented cluster fraction pi_tj. Pure farthest-point selection is a
diversity baseline, not automatically the best selector: it can overrepresent rare noise and malformed
outliers.

Evaluate sample capacities `R in {1, 3, 5, 10, 20}`. Report the realized mean and distribution because
many terms will not use the maximum.

## 7. Measuring sample and reranker quality

“Sample quality” has three distinct levels. Do not collapse them into one cosine score.

### 7.1 Corpus-representation quality

Measure how faithfully retained samples cover held-out occurrences of the same term:

`Coverage(t) = mean_{h in H_t} max_{s in S_t} cosine(E(h), E(s))`

Also report distinct-document coverage, duplicate rate, mean pairwise similarity, uncovered-context
distance, cluster support, sample count and preparation cost. These are representation diagnostics,
not relevance guarantees. If the same encoder selects and evaluates samples, disclose the circularity
and include lexical or human-audit checks on a declared subset.

### 7.2 Query-conditioned context compatibility

For query Q and samples S_t, start with:

`C_max(t,Q) = max_{s in S_t} cosine(E(Q), E(s))`

This performs latent sense selection: one compatible corpus usage can validate a polysemous term. It is
also vulnerable to one accidentally similar context. Compare:

`C_top2(t,Q) = mean of the two largest query-sample similarities`

`C_support(t,Q) = max_j [cosine(E(Q), E(s_tj)) + lambda * log(pi_tj)]`

The support-aware score penalizes a context representing a single anomalous occurrence. Define behavior
for terms with only one valid sample. Softmax/log-sum-exp aggregation is optional because it introduces
a temperature parameter.

### 7.3 Actual expansion-term utility

A topically relevant window does not prove that its target term improves retrieval. On development
queries, issue each candidate separately at a fixed declared weight and measure:

`U(t,Q) = Metric(Q + {t}) - Metric(Q)`

Label terms as helpful, neutral or harmful under a predeclared tolerance. This follows the intrinsic
term evaluation used by CEQE and enables candidate-ranking precision@5/10, utility-nDCG, harmful-term
rate and correlation with measured utility. Do not use these labels to train or tune on the final test
queries.

An optional blinded human audit should judge whether the selected sample expresses the candidate's
query-compatible sense and whether the candidate is a reasonable lexical expansion. Record agreement
and adjudication rules.

## 8. Candidate reranking

Let G(t,Q) be the first-stage term score and C(t,Q) a context score. BGE query-to-window similarity is
the mandatory default because it reuses the query representation and precomputed context vectors. It
measures topical compatibility, not expansion utility, and is not assumed optimal.

Compare:

1. `G(t,Q)` only: simple dense-vocabulary QE control.
2. `C(t,Q)` only: determines whether context can rank the fixed candidate set.
3. `alpha * norm(G) + (1-alpha) * norm(C)`: joint ranking.
4. `G * gate(C)`: context validation rather than unrestricted rescoring.

Scores from different sources require documented normalization; mapping to [0,1] does not create a
calibrated probability. The primary design should retain term evidence because a candidate can occur
incidentally in a query-relevant passage.

Accuracy-oriented conditional references may include:

- target-token contextual embeddings extracted from the same window;
- a cross-encoder over query, candidate and sample;
- an LLM judgment on a small declared subset.

These are upper-bound or diagnostic comparisons, not mandatory edge configurations. A raw token hidden
state is not automatically compatible with a pooled BGE query vector, and a cross-encoder invocation
per candidate can invalidate the latency objective.

## 9. Weight, term-count and posting-cost budgets

Reranking many candidates in a small dense matrix is cheap; executing many additional BM25 posting
lists is not. Initially retain 50–100 candidates for context reranking but emit only 3–10 terms.

Let original anchor weights be w_a >= 0 and A(Q) = sum_a w_a. Define a global expansion budget in
query-weight units:

`B(Q) = mu * f(Q) * min(A(Q), A_cap)`

Allocate nonnegative anchor budgets b_a whose sum is at most B(Q). For normalized candidate allocation
p(t|a):

`delta_w(t|a) = b_a * p(t|a) * min(1, IDF(a)/IDF(t))`

Under identical positive IDF values in allocation and retrieval and sum_t p(t|a) <= 1:

`sum_t delta_w(t|a) <= b_a`

`sum_t delta_w(t|a) * IDF(t) <= b_a * IDF(a)`

These bound query coefficients, not actual per-document BM25 scores. They do not establish semantic
correctness, ranking safety or information-theoretic optimality.

Add an independent computational constraint:

`sum_{t in expansion(Q)} DF(t) <= B_postings`

Compare term-count-only, weight-only and posting-aware selection. All candidate channels, collisions,
aliases and decomposed phrases must share declared budgets.

## 10. Resource model and latency gate

For V selected terms, R samples, dimension d and b bytes per coordinate, context-vector storage is:

`M_context = V * R * d * b`

At V=15,000, d=384 and FP16:

| Samples per term | Maximum context-vector storage |
|---:|---:|
| 5 | 57.6 MB |
| 10 | 115.2 MB |
| 20 | 230.4 MB |

The 15k term matrix adds about 11.5 MB. Metadata, raw audit samples, allocator state and runtime
overheads must be measured separately. The neural sidecar is capacity-bounded even as document count
grows; lexical indexing and corpus scanning are not.

With 100 retained candidates and 20 contexts each, query-time context scoring uses only 2,000 vectors,
or 768,000 coordinate multiplications plus reductions, after one query encoding. It should be batched
over a contiguous tensor. Never encode individual sample text or invoke a cross-encoder in the
mandatory query path.

Predeclare an incremental p95 context-reranking latency ceiling on the target hardware, initially 2 ms
on a warm GPU, plus an end-to-end tolerance relative to simple vocabulary QE. Synchronize CUDA before
timing, report warm and declared cold conditions, and profile encoding, term search, gather, aggregation,
allocation and BM25 separately. Treat the 2 ms value as an acceptance target, not a performance claim.

## 11. Evaluation design

### 11.1 Baselines and references

Mandatory local controls:

- `BM25_Default`;
- `BM25_RM3_Terrier_Default` and `BM25_Bo1_Terrier_Default`;
- `DPH`, `DPH_RM3_Terrier_Default` and `DPH_Bo1_Terrier_Default`;
- simple frozen-BGE vocabulary QE at the same vocabulary, candidate, final-term and weight budgets.

Conditional references:

- local-model LLM synonym or Query2doc-style lexical expansion with generation latency;
- a CEQE-like contextual PRF reference on selected feasible corpora;
- published or locally feasible dense and learned-sparse systems, with training and index costs;
- HyDE only as a cross-paradigm reference unless its dense index and generation path are measured;
- frozen archived V7 only where its historical analyzer and protocol can be reproduced and clearly
  separated from the standard-Terrier comparison.

Do not restore analyzed BM25 or unified custom RM3 merely as future baseline requirements. Any proposed
method using a custom analyzer needs its own matched BM25 attribution control.

### 11.2 Factorial ablations

Change one factor at a time before joint confirmation:

- vocabulary size: 1k, 5k, 10k, 15k;
- pool selection: salience, coverage, hybrid;
- samples per term: 1, 3, 5, 10, 20;
- sample selection: first, reservoir, cheap diversity, semantic representative;
- context: unmarked, target-marked, sentence-bounded and fixed-window variants;
- aggregation: max, top-2 mean, support-aware max;
- reranking: term-only, context-only, joint score, context gate;
- final expansion count and posting budget.

Reuse identical candidate sets when comparing rerankers. Otherwise candidate recall and ranking quality
are confounded.

### 11.3 Metrics and corpus-type claims

Follow [the canonical metric contract](EVALUATION_METRICS.md), including both declared standard and
exponential-gain nDCG fields where required, rather than redefining gains inside this plan. Preserve
retrieval depth 1,000 and control memory by query chunking, not candidate truncation.

Report per-dataset and macro effectiveness, paired per-query uncertainty, term-utility ranking,
harmful-term rate, expansion count, unused weight, postings touched, construction time, each corpus
pass, cache size, startup, process/system memory, GPU peak memory, throughput and component p50/p95.

Universal improvement is not required. “Works on some corpora” is publishable only if the collection
class and mechanism are declared and tested rather than selected after seeing results. Candidate
properties include:

- high query-document vocabulary mismatch;
- technical or domain-specific terminology;
- polysemous terms with repeated stable corpus usages;
- sufficient occurrence support for representative contexts;
- concentrated rather than highly heterogeneous domain language.

Also predeclare likely weak settings: exact-entity queries, very small corpora, insufficient term
contexts and collections where baseline BM25 already has high lexical overlap.

Use development and held-out evaluation partitions. Existing V7 calibration results are development
evidence because labels influenced previous choices.

## 12. Implementation and decision gates

| Stage | Work | Required result before proceeding |
|---|---|---|
| 0. Freeze contracts | Pin datasets, analyzer, metrics, hardware, simple BGE-Vocab-QE and current Terrier baselines | Reproducible controls and resolved configurations |
| 1. Vocabulary study | Audit salience/coverage evidence and compare 1k–15k pools | Candidate coverage/resource plateau supports a capped pool |
| 2. Context collector | Implement deterministic two-pass collection, filters, reservoirs and cache identity | Repeatable samples, bounded memory and measured preparation time |
| 3. Minimal reranker | Target-marked windows, frozen embeddings, max and top-2 scoring over a fixed candidate set | Score parity, component timing and term-utility improvement over term-only ranking |
| 4. Sample-quality study | Capacity, selection, coverage, support-aware and window ablations | Held-out evidence justifies complexity beyond direct reservoir sampling |
| 5. Retrieval integration | Global weight, final-term and posting-cost budgets with standard weighted BM25 | Bounds hold and latency remains within predeclared tolerance |
| 6. Corpus-type evaluation | Run the frozen method across the intended suite and query/corpus buckets | Reproducible benefit or cost-equivalent quality in a defined setting |
| 7. Conditional extensions | LLM proposals validated by the same context memory, CEQE-like reference or richer lexical units | Added contribution justifies preparation and query cost |

Every implementation must use deterministic seeds, a versioned cache identity, invariant checks and
timestamped results. Any new Pipeline V2 variant requires its co-located `pathway_*.md`. Synchronize
`ARCHITECTURE.md`, the results-to-scripts mapping and the manuscript evidence map only after empirical
implementation and verification.

## 13. Later extensions, not primary scope

Phrase/entity inventories, term-association graphs, topic summaries, compressed passage representatives,
relation/event inventories and static validation of LLM proposals remain legitimate follow-up ideas.
They are not parallel mandatory implementations. Promote one only after the bounded term-context method
passes its decision gates and the extension addresses a measured failure mode.
