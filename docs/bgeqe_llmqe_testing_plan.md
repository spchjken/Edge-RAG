# BGE Vocabulary QE and LLM Keyword QE Testing Plan

> Status: revised implementation plan, 2026-09-11. This plan replaces the completed,
> now-duplicative `docs/implementation_plan.md`. It defines two sparse lexical
> query-expansion baselines and their evaluation protocol. It does not define
> the proposed Context-Reranked Vocabulary Expansion (CRVE) method; see
> [the corpus-informed QE research plan](corpus_informed_query_expansion_plan.md)
> for that work.

## 1. Objective and comparison boundary

Add two query-expansion (QE) baselines to the standard Terrier/PyTerrier
harness:

| ID | Name | Evidence available at expansion time | Purpose |
|---|---|---|---|
| `BGE_Vocab_QE` | Frozen-encoder vocabulary QE | A static corpus vocabulary and frozen `BAAI/bge-small-en-v1.5` embeddings | Practical full-suite analogue of pre-retrieval word-embedding QE. |
| `LLM_Q2E_ZS` | Local-LLM zero-shot keyword QE | Query text and a declared local LLM; no retrieved documents | Local-model adaptation of the published Q2E/ZS keyword baseline. |

Neither baseline is a dense document retriever, pseudo-relevance-feedback
(PRF) method, Query2doc-style LLM pseudo-document expansion method, or CRVE
variant. Both use **one standard Terrier BM25 lexical retrieval** after query
rewriting. Both retain the original query, but they follow their own
reference-aligned term selection and weighting rules rather than sharing a
CRVE-derived expansion budget. The existing `BGE_Small_Dense` row is a dense
bi-encoder baseline and must not be relabelled or reused as `BGE_Vocab_QE`.

The experiment asks three narrow questions:

1. Does a frozen encoder plus a bounded target-corpus vocabulary improve over
   `BM25_Default` without document-level dense retrieval or PRF?
2. Does query-only zero-shot keyword generation help sparse retrieval when
   implemented as closely as practical to published Q2E/ZS?
3. Does a later context-aware method improve over the simple BGE vocabulary
   control, rather than only over unexpanded BM25?

### 1.1 What “default” means in this plan

These are baselines, not additional proposed methods. “Default” therefore
means: use a recognizable published formulation, retain the standard Terrier
retrieval path, fix unavoidable implementation choices before effectiveness
evaluation, and do not select a baseline variant because it wins on benchmark
qrels. CRVE-specific salience ranking, context samples, risk penalties,
bailout, anchor priors, and mass-preserving weighting are excluded.

There is no published BGE vocabulary-QE default. The closest pre-retrieval
formulation is Roy et al. (2016): take nearest neighbours of each query term,
form their union, rank candidates by mean similarity to all query terms, and
interpolate the selected expansion distribution with the original query. Kuzi
et al. (2016) instead used Word2Vec CBOW trained over the target search corpus.
This plan preserves the former term-level mechanics but substitutes frozen
BGE-small to avoid training a separate Word2Vec model for every one of the 25
corpora. It is therefore a **modern practical analogue**, not a reproduction of
Kuzi, Roy, or Diaz.

For LLMQE, the reference is Jagerman et al. (2023) Q2E/ZS: the zero-shot prompt
asks for a keyword list, and the generated output is appended to five copies of
the original query before Terrier BM25. This plan preserves those mechanics
while substituting a declared local model for the paper's Flan models.

Primary references, checked 2026-09-11:

- Roy et al., [Using Word Embeddings for Automatic Query Expansion](https://arxiv.org/abs/1606.07608).
- Kuzi et al., [Query Expansion Using Word Embeddings](https://doi.org/10.1145/2983323.2983876).
- Diaz et al., [Query Expansion with Locally-Trained Word Embeddings](https://aclanthology.org/P16-1035/).
- Jagerman et al., [Query Expansion by Prompting Large Language Models](https://arxiv.org/abs/2305.03653).
- BAAI, [`bge-small-en-v1.5` model card](https://huggingface.co/BAAI/bge-small-en-v1.5).
- PyTerrier, [Terrier retrieval and `query_toks` documentation](https://pyterrier.readthedocs.io/en/latest/terrier-retrieval.html).
- Ollama, [`generate` API documentation](https://docs.ollama.com/api/generate).
- Ollama, [thinking-mode documentation](https://docs.ollama.com/capabilities/thinking).

Diaz et al. is included as evidence about global versus topic-specific term
representations, not as a protocol for this static baseline: its local models
are trained from topically retrieved documents and therefore do not satisfy the
zero-first-pass boundary used here.

The prior papers tune or vary expansion counts and interpolation weights; they
do not establish a universal BGEQE term count. Consequently, every numeric
choice below is marked either **reference-derived** or **project-fixed**. A
project-fixed value is a reproducibility decision, not a claim of universal
literature default.

## 2. Non-negotiable evaluation contract

- Use the cached standard Terrier default index already managed by
  `PyTerrierIndexManager`; do not build a custom-analyzer index for these
  baselines.
- Reuse the exact `BM25_Default` retriever. Record its effective query-time
  parameters and verify the expected Terrier values \(k_1=1.2\), \(b=0.75\),
  and \(k_3=8.0\). If the installed Terrier version differs, resolve the
  baseline-wide configuration explicitly rather than changing QE alone.
- Preserve the standard Terrier tokenization, stopword removal, stemming, and
  full candidate depth of 1,000. Query chunks remain `chunk_size=200`. The raw
  query is retained separately for BGE or LLM input; retrieval uses the
  analyzed `query_toks` representation defined below.
- For BRIGHT, rewrite before the sole retrieval, request the existing padded
  depth `min(1000 + max_excluded, 3000)`, where `max_excluded` is the largest
  number of prohibited document identifiers attached to any query in that
  dataset. Then apply the existing post-retrieval exclusion filter and retain
  1,000 eligible documents. There is no feedback pass, therefore no PRF-style
  pre-filter stage.
- Keep original query terms in every rewritten query. If expansion produces no
  admissible term, the result must be exactly the `BM25_Default` retrieval for
  that query under the same index and exclusion conditions.
- Use the canonical primary metrics: linear `ndcg_10`, `ndcg_50`, `ndcg_100`,
  `map_100`, `mrr_10`, precision, Recall@10--1000, Strict@K, completeness, and
  `oracle_ndcg_10_from_1000`, meaning oracle nDCG@10 after reranking the
  retrieved candidate pool of exactly 1,000 documents. Keep `exp_ndcg_10` as
  the declared supplemental metric.
- Do not overwrite `results/pyterrier_baselines/pyterrier_baselines_results.csv`.
  QE results require additional provenance and telemetry, so write a separate,
  timestamped raw result directory and a dedicated append-only QE summary.
- Do not use benchmark relevance judgments (qrels), rankings, retrieved document
  text, or test labels to choose terms, adjust weights, or recover malformed
  LLM output.

## 3. Definitions and notation

The following terms and symbols have one meaning throughout this plan:

| Term or symbol | Definition |
|---|---|
| \(Q\) | One raw user query before sanitization or Terrier analysis. |
| \(T_Q\) | The weighted dictionary of original query terms after the standard Terrier tokenizer, stopword filter, and stemmer. Repeated original terms retain their Terrier query-term frequency. |
| \(\mathcal{A}(Q)\) | The ordered multiset of non-stopword natural query-token surfaces that Terrier maps to terms in \(T_Q\). These surfaces, rather than Porter stems, are embedded as BGEQE query terms. |
| \(\mathcal{A}_u(Q)\) | One deterministic natural-surface representative for each distinct Terrier term in \(T_Q\): choose the most frequent surface for that term within the query and break ties by first occurrence. |
| \(\operatorname{qtf}(a,Q)\) | Terrier query-term frequency of the retrieval term reached by natural query surface \(a\). |
| \(e(x)\) | The L2-normalized BGE embedding of natural surface \(x\), produced through the instruction-free encoding path fixed in Section 5.2. |
| \(t\) | One candidate expansion term after Terrier analysis. |
| \(C(Q)\) | Union of the \(L\) nearest vocabulary neighbours retrieved for every surface in \(\mathcal{A}_u(Q)\). |
| \(E(Q)\) | The top \(k_e\) validated BGEQE terms after ranking \(C(Q)\). |
| \(|E(Q)|\) | Number of terms in \(E(Q)\). Vertical bars around a set mean its number of elements. |
| \(L\) | Per-query-term neighbour depth used to form \(C(Q)\); project-fixed to 50. |
| \(k_e\) | Final BGEQE expansion count; project-fixed to 10. Prior embedding-QE papers tune this value and provide no universal default. |
| \(\alpha\) | Original-query interpolation share for BGEQE; reference-anchored and fixed to 0.60. The expansion share is \(1-\alpha=0.40\). |
| \(G(Q)\) | Raw plaintext generated by the local LLM for the Q2E/ZS prompt. |
| \(r\) | Number of copies of the original query used by Q2E/ZS; reference-derived and fixed to 5. |
| \(w_{\text{orig}}(t)\) | Weight for term \(t\) in the `query_toks` representation obtained from Terrier's own analysis of the original query. It is zero when \(t\) is not an original query term. |
| \(\delta w(t)\) | The nonnegative weight added by query expansion for term \(t\). The symbol \(\delta\) means an added change, not a probability. |
| \(w_{\text{final}}(t)\) | The final analyzed query-term frequency or weight sent to Terrier. Its construction is baseline-specific in Sections 5 and 6. |
| \(A(Q)\) | Original query mass, defined as \(\sum_t w_{\text{orig}}(t)\) over \(T_Q\). Here "mass" means only the sum of query-term weights. |
| DF | Document frequency: number of indexed documents containing a term. |
| collection frequency | Total number of occurrences of a term across the indexed collection. |
| IDF | Inverse document frequency obtained from the Terrier index. The implementation must record the exact Terrier value or formula used. |
| \(K\) | Retrieval cutoff. `Recall@K`, for example, measures relevant-document recall within the first \(K\) results. |
| qrels | Benchmark relevance judgments mapping each query to judged documents and relevance grades. |
| BRIGHT | The reasoning-retrieval benchmark in which each query can declare prohibited source documents that must be excluded from retrieval results. |
| sidecar | A bounded cache stored alongside, but not inside, the Terrier index; for BGEQE it contains vocabulary metadata and embeddings. |
| cache identity | A hash over every input that can change a cache's meaning, including dataset/index fingerprint, model revision, configuration, prompt where applicable, and seed. |

Abbreviations and implementation terms:

| Term | Definition |
|---|---|
| QE | Query expansion: adding weighted lexical terms to the original query before retrieval. |
| BM25 | The standard probabilistic lexical retrieval model used by the Terrier baseline index. |
| PRF | Pseudo-relevance feedback: query expansion based on documents returned by an initial retrieval. Neither baseline in this plan uses PRF. |
| BGE-small | The frozen 384-dimensional encoder `BAAI/bge-small-en-v1.5`. "Frozen" means its parameters are not trained or updated in this experiment. |
| BGEQE | `BGE_Vocab_QE`, the frozen-BGE term-neighbour baseline defined in Section 5. |
| LLM | Large language model. |
| LLMQE | `LLM_Q2E_ZS`, the query-only zero-shot keyword baseline defined in Section 6. |
| CRVE | Context-Reranked Vocabulary Expansion, the proposed-method direction described in the separate corpus-informed QE plan. |
| `query_toks` | PyTerrier input column containing a dictionary from already analyzed index term to numeric query weight. The ordinary `query` string is ignored when this column is used. |
| `qid` | String identifier of one benchmark query. |
| FP16 | 16-bit floating-point storage for cached vocabulary vectors. |
| L2 normalization | Division of a vector by its Euclidean length so that the resulting vector has length one. |
| cosine score | Similarity between two L2-normalized vectors, computed here by their dot product. |
| p50, p90, p95, p99 | The 50th, 90th, 95th, and 99th percentiles of a measured latency distribution. |
| manifest | Machine-readable record of the exact data, model, configuration, environment, and cache identity used for an artifact. |
| model digest | Content-derived identifier reported by the model backend for the exact installed model artifact. |
| quantization | Reduced-precision representation used to store or execute an LLM, such as a named 4-bit format. |
| \(s(t,Q)\) | Query-term-frequency-weighted mean cosine between candidate surface \(t\) and the distinct natural query-token representatives in \(\mathcal{A}_u(Q)\), defined in Section 5. |
| \(u(t,Q)\) | Nonnegative BGEQE weighting score \(\max(s(t,Q),0)\). |
| \(B_{\alpha}(Q)\) | BGEQE expansion mass \(((1-\alpha)/\alpha)A(Q)\). This fixes the ratio of analyzed query-weight mass before retrieval; it does not guarantee the same ratio of final BM25 score contribution. |

Metric names not expanded here follow
[the canonical metric definitions](EVALUATION_METRICS.md).

## 4. Shared retrieval and validation contract

The baselines share the index, Terrier analysis, lexical safety checks,
retrieval depth, exclusions, metrics, and provenance. They deliberately do not
share candidate generation or weighting: forcing a common CRVE-style expansion
budget would make neither baseline recognizable as its literature reference.

### 4.1 Candidate representation

Every candidate has both a display form and a validated Terrier retrieval form:

```text
display_surface: natural corpus surface form used for BGE encoding and logs
retrieval_term:  form accepted by the default Terrier query pipeline and present in its lexicon
```

This distinction is mandatory. Encoding Porter-style stems such as `comput` or
injecting a natural surface form that does not resolve to an index term would
make the baseline unreliable.

### 4.2 Analyzer-parity gate

Before any BGE or LLM corpus-scale run, implement and pass a deterministic
surface-to-index mapping and query-weight parity gate. It must:

1. derive candidate DF, collection frequency, and IDF directly from the cached
   default Terrier lexicon;
2. confirm that the emitted retrieval term is in the loaded index lexicon;
3. verify on a declared sample that the rewritten Terrier query reaches the
   intended lexicon entry;
4. reject terms that cannot be represented safely.

The fixture portion of this gate must include inflections, stopwords,
punctuation, repeated query terms, and at least one rejected term. The original
query weights \(w_{\text{orig}}(t)\) must come from Terrier's own query-analysis
path, including its query-term-frequency behavior; they must not be
reconstructed by a separate Python stemmer. Zero-expansion `query_toks`
retrieval must match `BM25_Default` on the same fixture index in document order
and score, within the documented floating-point tolerance. A failure blocks all
full-dataset evaluation.

For BGEQE only, the lexicon does not reliably preserve the natural surfaces
that should be embedded. Extract an oversized bounded set of 20,000 lexicon
terms in descending collection-frequency order, with lexical tie-breaking.
Then make one streaming corpus pass to retain the most frequent natural surface
that Terrier maps to each candidate. This pass recovers surfaces; it must not
recount the full corpus vocabulary. It may still read every raw document, so
its time and I/O are a real cost of BGEQE. Record elapsed time, bytes read, peak
process memory, and retained candidate keys. The collector must use
`BenchmarkLoader.stream_corpus()` or an equivalent iterator and must never
materialize raw documents or unbounded occurrence lists. Future index builds
may collect this mapping during the existing indexing stream, but existing
cached indices must not be rebuilt only to avoid reporting the recovery pass.

Do not silently use `EdgeRAGAnalyzer` or claim analyzer parity without a test.
Do not embed Porter-style stems as if they were natural words unless a separately
named diagnostic baseline explicitly tests that shortcut.

### 4.3 Query construction and freeze policy

Represent final queries with PyTerrier's pre-tokenized `query_toks` column, a
dictionary from analyzed index term to numeric query-term weight. Do not inject
hand-built `term^weight` strings into TerrierQL. Every natural expansion surface
or generated string must pass through Terrier's configured tokenizer, stopword
filter, and stemmer. Terms absent from the loaded lexicon contribute nothing and
are logged; parser punctuation is never copied into `query_toks`.

The two primary baselines have separate, explicit query-construction formulas:
BGEQE uses similarity-weighted interpolation in Section 5, while Q2E/ZS uses
the reference-derived fivefold original-query repetition in Section 6. The
unit-tested fallback for either method must return the original `query_toks`
unchanged whenever no usable expansion remains, giving exact `BM25_Default`
retrieval for that query.

All primary parameters are fixed in this document before the pilot. Gates B--D
may identify implementation defects, infeasible resource use, or a missing
model, but benchmark effectiveness must not select a query form, term count,
prompt, weight, or filter. Any later sweep is a separately named ablation and
must not replace the primary baseline in the full-suite comparison.

## 5. `BGE_Vocab_QE`: simple static dense-vocabulary expansion

### 5.1 Definition

`BGE_Vocab_QE` performs no document retrieval before expansion:

```text
stream/build vocabulary sidecar once
    -> encode the capped vocabulary once with frozen BGE-small
    -> encode the natural surfaces of analyzed query terms
    -> retrieve neighbours per query term
    -> rank their union by mean query-term similarity
    -> validate/admit up to ten terms
    -> one weighted Terrier BM25 retrieval
```

This fixed term-neighbour formulation follows the pre-retrieval structure of
Roy et al. rather than choosing among whole-query and maximum-anchor variants
on the benchmark. For each query surface \(a\in\mathcal{A}_u(Q)\), retrieve its
\(L=50\) nearest vocabulary surfaces. After excluding original query terms,
their union is \(C(Q)\). Score each candidate by its query-term-frequency-
weighted mean similarity:

\[
s(t,Q)=
\frac{
  \sum_{a\in\mathcal{A}_u(Q)}
  \operatorname{qtf}(a,Q)\cos(e(a),e(t))
}{
  \sum_{a\in\mathcal{A}_u(Q)}\operatorname{qtf}(a,Q)
},
\]

Rank \(C(Q)\) by descending \(s(t,Q)\), break ties lexically by
`retrieval_term`, and retain the first \(k_e=10\) valid terms. Ten terms is a
project-fixed bounded choice aligned with the existing Terrier QE term count;
it is not attributed to Kuzi or Roy, whose studies vary or tune expansion
counts.

This is intentionally a pragmatic control, not a claim that BGE is naturally
calibrated for word-to-word similarity. BGE-small is trained for retrieval and
single-word inputs are a task adaptation. A weak result can therefore reject
this frozen-BGE control, but cannot by itself reject corpus-trained embedding
QE or word-embedding QE in general.

Whole-query-to-term, maximum-anchor, sigmoid-transformed, salience-reranked,
and IDF-damped forms are optional diagnostics or proposed-method components.
They must not be used to select or replace the primary baseline after viewing
qrels.

### 5.2 Static vocabulary sidecar

Create a versioned cache under:

```text
data/cache/qe_vocab/{dataset}/{cache_identity}/
```

It stores, at minimum:

- ordered `display_surface` and `retrieval_term` arrays;
- document frequency and collection frequency;
- normalized FP16 BGE vectors and their embedding dimension;
- a machine-readable manifest with dataset identity/fingerprint, index path,
  index fingerprint, vocabulary-selection rule, eligibility filters, cap,
  encoder model/revision, encoder instruction policy, dtype, seed, source
  corpus-pass count, wall-clock preparation time, and creation timestamp.

Read DF and collection frequency directly from the cached Terrier lexicon.
After the eligibility checks below, retain at most 15,000 terms in descending
collection-frequency order with lexical tie-breaking:

- Terrier lexicon membership;
- collection frequency at least 3;
- a recovered natural surface mapping back to exactly one retained Terrier
  term;
- no stopword-only, punctuation-only, or unsupported parser form.

The 15,000-term ceiling is a declared edge-resource adaptation. Kuzi and Roy
trained over the collection vocabulary; this plan must not describe the cap as
their default. Collection frequency is used as the neutral deterministic cap,
not Pipeline V2's `IDF * log(1 + DF)` salience. Write the realized vocabulary,
ordering rule, filters, and excluded counts to the manifest. Do not choose terms
using queries, qrels, or retrieval outcomes.

Encode every natural display surface exactly once. Encode both query-token
surfaces and vocabulary surfaces with the same instruction-free BGE `encode`
path, L2-normalize them, and use dot products as cosine scores. BGE's model card
reserves its query instruction for short-query-to-passage retrieval; this
baseline instead performs symmetric term-to-term comparison. Record the exact
library and model revision. Do not tune the instruction policy on benchmark
effectiveness.

### 5.3 Query-time transform

Implement a small registered PyTerrier transformer, conceptually:

```text
raw query and Terrier analysis
  -> aligned natural query-token surfaces
  -> batch BGE term embeddings
  -> top-50 neighbours per distinct query term
  -> union and mean-similarity ranking
  -> Terrier-analysis, lexicon, duplicate, and syntax validation
  -> retain the admitted top ten
  -> similarity-normalized interpolation weights in query_toks
  -> existing BM25 retriever and exclusion filter
```

The transformer operates on a query DataFrame and returns the same `qid` with
the rewritten `query_toks` plus telemetry columns. Preserve a separate raw-query
column and align its natural token surfaces to Terrier's analyzed terms. Do not
embed Porter stems or the entire raw query for the primary baseline. Batch all
distinct query-token surfaces for each query chunk. Term vectors remain
resident in a contiguous matrix; no corpus text or document embeddings are
loaded at query time.

Let \(u(t,Q)=\max(s(t,Q),0)\). For nonempty \(E(Q)\) with positive total \(u\),
preserve Terrier's original weights and allocate an expansion mass equivalent
to a pre-retrieval query-weight interpolation share \(1-\alpha\):

\[
B_{\alpha}(Q)=\frac{1-\alpha}{\alpha}A(Q),
\qquad
\delta w(t)=B_{\alpha}(Q)
\frac{u(t,Q)}{\sum_{x\in E(Q)}u(x,Q)},
\]

\[
w_{\text{final}}(t)=w_{\text{orig}}(t)+\delta w(t),
\qquad \alpha=0.60.
\]

Original query terms are excluded from \(E(Q)\), so the usual expansion term
has \(w_{\text{orig}}(t)=0\). If no valid candidate has positive \(u\), return
the unchanged original `query_toks`. The fixed \(\alpha=0.60\) is anchored in
the interpolation range reported by prior word-embedding QE, but its use with
BGE and BM25 is an explicit adaptation rather than an exact reproduction.
Because Terrier BM25 applies its query-term-frequency factor (including
\(k_3\)) after these weights are supplied, \(\alpha\) does not assert that 60%
of the final document score comes from original terms.

Register the transformer explicitly in `PyTerrierBaselineHarness` rather than
relying on the current incidental `self.pipelines` fallback. The registration
contract should make the required resources, cache identity, query-rewrite
function, and pipeline label visible to the harness.

### 5.4 Required BGEQE telemetry

For each dataset and pipeline run, record:

- vocabulary cap and realized eligible count;
- sidecar storage, total preparation time, lexicon-extraction time,
  surface-recovery time and bytes read, corpus-pass count, and cache hit/miss;
- query encoding, matrix search, validation/allocation, and BM25 timings;
- candidate count, admitted-term mean/distribution, expansion mass, mean DF and
  total selected DF as a postings-cost proxy;
- terms rejected for duplication, lexicon mismatch, parser safety, or filters;
- deterministic seed and cache/model identity.

Persist a bounded per-query trace containing raw query ID, selected retrieval
terms, display surfaces, scores, weights, rejections, and rewrite time. Do not
store full document rankings here; the harness's compressed candidate-run cache
remains the ranking artifact.

### 5.5 Optional Word2Vec fidelity check

Do not replace the full 25-dataset BGEQE baseline with per-corpus Word2Vec
training. If schedule permits, `Word2Vec_CBOW_QE_SpotCheck` may be run on the
predeclared `scifact`, `fiqa`, and `bright_stackoverflow` datasets to test
whether the frozen-BGE analogue changes the conclusion. It uses corpus-trained
CBOW term vectors over the same eligible 15,000-term vocabulary and uses the
same candidate union, mean-similarity ranking, \(L\), \(k_e\), and \(\alpha\)
as BGEQE, so the encoder is the intended changed factor. Its training settings,
corpus passes, wall time, peak memory, seed, and worker count must be declared
before execution and recorded.

This optional row is a fidelity diagnostic, not a completion requirement and
not a full Kuzi reproduction. The full-suite decision remains BGEQE because a
separate corpus-trained model for every dataset materially changes preparation
cost and failure exposure under the stated edge/streaming objective.

## 6. `LLM_Q2E_ZS`: query-only zero-shot keyword baseline

### 6.1 Definition and fairness boundary

`LLM_Q2E_ZS` generates keyword text from the query alone and follows the
published Q2E/ZS query-construction pattern:

```text
raw query -> exact zero-shot Q2E prompt -> local LLM keyword text
         -> concatenate five original-query copies and generated text
         -> Terrier analysis of the full concatenated text
         -> one Terrier BM25 retrieval
```

It must not receive corpus documents, top-ranked documents, qrels, examples
from benchmark queries, or a corpus vocabulary prompt. Intersecting generated
outputs with the index lexicon is a target-corpus membership check, but it does
not use retrieved documents or alter generation. Because an out-of-lexicon term
would produce no postings anyway, this validation makes the no-op visible
rather than semantically selecting a replacement term.

Generated keyword phrases are treated as ordinary bags of Terrier-analyzed
tokens; no phrase, Boolean, or proximity operator is created. HyDE, Query2doc,
Q2D, CoT, few-shot prompting, and PRF-augmented prompts are separate methods
and are out of scope. A strict five-single-word synonym generator may be tested
later as `LLM_Synonym_QE_Restricted`, but it is not the primary literature-like
baseline and is not required by this plan.

### 6.2 Fixed local model, availability check, and prompt

The initial intended profile is `qwen3.5-4b` in `configs/models.yaml`, but a
configuration entry does not prove that the model is installed. Before any
generation, query the configured backend's model-list endpoint, require an
exact tag match, and record the resolved model digest. If the intended model is
unavailable, stop before generation and require an explicit configuration
change. Never silently fall back to another model.

Record the resolved backend, tag, digest or model-file hash, endpoint, hardware,
context window, quantization where reported, and model load condition.

Use deterministic decoding: temperature 0, a fixed seed where the backend
supports it, a project-fixed maximum of 64 generated tokens, no tools, and
non-streaming output. The source paper does not establish these decoding values;
they are reproducibility and resource controls. Do not request JSON or another
schema because doing so changes the published prompt and output distribution.
For a backend that exposes reasoning separately, disable thinking when
supported; otherwise store the thinking field for provenance but never append
it to the retrieval query. Only the final response text is \(G(Q)\). Record the
effective setting because current Ollama thinking-capable models can enable it
by default.
The exact versioned prompt is:

```text
Write a list of keywords for the following query: {query}
```

This is the Q2E/ZS prompt reported by Jagerman et al. It must not be adjusted
using pilot effectiveness. Its content and hash are part of every cache and
result identity.

If Terrier analysis of \(G(Q)\) contains at least one usable lexicon term,
construct the reference-aligned raw expanded query:

\[
Q'=\operatorname{Concat}(\underbrace{Q,\ldots,Q}_{r\text{ copies}},G(Q)),
\qquad r=5.
\]

Pass \(Q'\) through Terrier's standard query-analysis path and use the resulting
`query_toks`; do not manually approximate query-term frequency or BM25's query
weighting. Preserve repeated generated terms and generated terms that repeat
original terms because raw concatenation also preserves them. Terms absent from
the corpus lexicon are logged and naturally contribute no postings. If no usable
generated term remains, return the unchanged original `query_toks` rather than
a fivefold copy, so the documented fallback is exactly `BM25_Default`.

### 6.3 Generation cache and failure behavior

Normalize only inconsequential surrounding whitespace, hash the exact query
text, and deduplicate identical raw queries globally before generation.
Generate once per distinct query and cache the raw response separately from the
corpus-specific validated term list:

```text
data/cache/qe_llm/raw/{model_identity}/{prompt_hash}/{query_hash}.json
data/cache/qe_llm/validated/{dataset}/{index_identity}/{generation_identity}.jsonl
```

Each raw record contains the request parameters, response text, parse result,
generator timing, backend/model identity, and failure state. Here `parse result`
means the deterministic Terrier token-analysis result, not JSON validation.
Each validated record additionally contains analyzed term counts, admitted
lexicon terms, rejected tokens, and reasons.

Empty output or output containing no usable lexicon term produces an empty
expansion and is logged. There is no hidden repair prompt or second generation.
Transient backend failures stop the worker so the normal resume mechanism can
retry deterministically from the cache boundary. This separates model-quality
failures from infrastructure failures and prevents unreported extra calls.

Before the full generation sweep, compute and record:

\[
T_{\text{projected}}=N_{\text{uncached}}
\times\overline{T}_{\text{pilot}},
\]

where \(N_{\text{uncached}}\) is the number of distinct queries without a valid
raw cache record and \(\overline{T}_{\text{pilot}}\) is mean generation time in
the frozen pilot. This is an execution estimate, not a result metric.

### 6.4 LLMQE timing report

Report three different numbers, never one conflated latency:

1. **Generation latency:** warm p50/p90/p99 and mean from the local LLM call;
   report cold model-load/startup separately.
2. **Expanded-query retrieval latency:** BGE-free query rewrite plus Terrier
   retrieval from cached validated terms.
3. **End-to-end latency:** sequential generation plus expanded-query retrieval
   under a declared warm or cold serving condition.

Also report nonempty-output rate, empty-expansion rate, raw/generated/admitted
term counts, repeated-term rate, lexicon-rejection rate, and the fraction of
queries whose validated expansion is identical to unexpanded BM25.

## 7. Harness and runner changes

### 7.1 Files to add or modify during implementation

| Path | Planned change |
|---|---|
| `src/evaluation/pyterrier_harness.py` | Add explicit custom-pipeline registration, static-vocabulary QE resources, query-rewrite telemetry collection, and resource fields for QE sidecars. Preserve existing six Terrier, dense BGE, and SPLADE behavior. |
| `src/evaluation/pyterrier_qe.py` | Add isolated, testable components for analyzer-parity mapping, lexicon-statistics extraction, bounded surface recovery, vocabulary-sidecar construction/loading, BGE term-neighbour rewriting, Q2E/ZS plaintext analysis/cache handling, and baseline-specific `query_toks` construction. Do not import Pipeline V2. |
| `scripts/run_pyterrier_qe_baselines.py` | Add a sequential, resumable CLI runner for `BGE_Vocab_QE` and `LLM_Q2E_ZS`; the currently documented PyTerrier runner is not present in this checkout, so this script becomes the explicit QE entry point. |
| `configs/pyterrier_qe.yaml` | Add all fixed QE parameters: vocabulary cap and eligibility, neighbour depth, expansion count, interpolation share, model identity, exact prompt, generation-token limit, repetition count, timing protocol, seed, cache locations, and run-selection options. |
| `tests/test_pyterrier_qe.py` | Add deterministic unit/integration tests described below. |
| `results/pyterrier_baselines/` | Add timestamped QE raw outputs and a dedicated append-only `pyterrier_qe_results.csv`; leave the existing baseline CSV intact. |
| `scripts/results_scripts_mapping.md` | Add the new output-to-runner mapping if that mapping is restored or present at implementation time. |

`src/evaluation/pyterrier_qe.py` is baseline code: it must remain isolated from
`src/pipeline_v2/` and `src/legacy_pipeline/`. It may reuse only general
evaluation/loading utilities and the standard PyTerrier index.

### 7.2 Result schema

Use the existing canonical effectiveness fields and append QE-specific columns
in the dedicated QE CSV:

```text
dataset, pipeline, run_id, config_hash, seed,
ndcg_10, exp_ndcg_10, ndcg_50, ndcg_100, map_100, mrr_10,
p_10, p_100, recall_10, recall_50, recall_100, recall_200, recall_500,
recall_1000, strict_10, strict_50, strict_100, strict_1000,
completeness_100, completeness_500, completeness_1000,
oracle_ndcg_10_from_1000,
retrieval_api_p50_ms, retrieval_api_p90_ms, retrieval_api_p99_ms,
batch_throughput_qps, harness_per_query_ms,
qe_prepare_s, qe_sidecar_mb, qe_cache_hit, qe_vocab_cap, qe_vocab_realized,
qe_lexicon_extract_s, qe_surface_recovery_s, qe_surface_recovery_bytes,
qe_corpus_passes,
qe_candidates_mean, qe_terms_mean, qe_expansion_mass_mean, qe_selected_df_mean,
qe_selected_df_p95, qe_rewrite_p50_ms, qe_rewrite_p95_ms,
llm_model, llm_model_digest, llm_prompt_hash, llm_unique_queries,
llm_raw_cache_hit_rate, llm_projected_generation_s,
llm_cold_load_ms, llm_generation_mean_ms, llm_generation_p50_ms,
llm_generation_p90_ms, llm_generation_p99_ms,
llm_end_to_end_p50_ms, llm_end_to_end_p90_ms, llm_end_to_end_p99_ms,
llm_nonempty_output_rate, llm_empty_rate, llm_repeated_term_rate,
llm_lexicon_reject_rate
```

Fields not applicable to a pipeline are empty, not zero. Store the full
environment and configuration manifest alongside every timestamped result file.

### 7.3 CLI and execution isolation

The runner must accept at least:

```text
--datasets ... | --all-datasets
--pipelines BGE_Vocab_QE,LLM_Q2E_ZS
--seed 42
--chunk-size 200
--config configs/pyterrier_qe.yaml
--resume
--results-dir results/pyterrier_baselines/<timestamp>
--llm-model-profile <explicit-installed-profile>
--preflight-only
```

Run one dataset in an isolated subprocess and release Python, JVM, and GPU
resources before the next dataset. Never run this workload concurrently with
the currently active dense/SPLADE job. The runner must resume only completed,
config-identical dataset/pipeline records and must never overwrite a previous
result artifact.

## 8. Tests and gates before full evaluation

### 8.1 Required automated tests

1. Candidate mapping rejects a surface form that has no verified Terrier
   lexicon representation and accepts a known valid form.
2. Vocabulary DF and collection frequency are read from the fixture Terrier
   lexicon without recounting the raw corpus; the 15,000-term cap is ordered by
   collection frequency with deterministic lexical ties.
3. With zero admitted expansions, the `query_toks` pipeline produces the same
   ranked documents and scores as `BM25_Default` on a fixture index.
4. Fixed BGE vectors/query inputs produce identical top candidates, weights,
   and rewritten queries across runs with the same seed.
5. For every nonempty BGEQE expansion, weights are nonnegative, expansion
   weights sum to \(B_{\alpha}(Q)\) within tolerance, similarity proportions
   are preserved, and all original-query terms and weights are preserved.
6. BGE query encoding is batched per query chunk; no raw corpus text is loaded
   on the query path.
7. Q2E/ZS uses the exact versioned plaintext prompt; mocked responses verify
   construction and Terrier analysis of the fivefold concatenated query,
   preservation of repeated generated terms, lexicon no-ops, cache identity,
   global query deduplication, and exact-BM25 empty-expansion fallback.
8. BRIGHT exclusion filtering still returns up to 1,000 eligible candidates and
   never exposes excluded documents after either QE rewrite.
9. A resumed run skips only records with matching dataset, pipeline, config
   hash, model identity, and seed.
10. The LLM preflight rejects a missing tag, records the resolved model digest,
   and never silently substitutes another configured model.

### 8.2 Smoke and pilot gates

| Gate | Dataset scope | Required evidence |
|---|---|---|
| A: lexical parity | fixture plus one small corpus | Verified surface/index mapping, exact `query_toks` construction, and no-query-expansion parity with BM25. |
| B: BGEQE smoke | `scifact` and one BRIGHT corpus | The fixed term-neighbour baseline builds within the memory cap, is deterministic, matches the declared formula, and preserves full exclusion semantics. |
| C: LLMQE prompt/cache smoke | same two corpora | Exact Q2E/ZS prompt, plaintext output analysis, exact model identity, recorded generation timings, and no undocumented retry path. |
| D: fixed-config pilot | `scifact`, `fiqa`, `quora`, `trec_covid`, `bright_stackoverflow` | Exercise the already-fixed configuration, estimate full-generation time, and inspect failure/resource telemetry. Do not choose variants or change parameters using effectiveness. Mark these datasets as pilot-observed in later reporting. |
| E: full sweep | all 25 datasets, sequentially | Complete raw/summary artifacts, environment manifests, and no breach of 15 GiB host-RAM limit. |

Baseline sanity is established by analyzer parity, deterministic candidate and
weight calculations, exact fallback behavior, and resource/provenance checks;
it is not defined as being within an arbitrary effectiveness margin of BM25. If
a correctly implemented fixed baseline performs poorly, report that result and
diagnose it without tuning the primary baseline on test qrels.

The five large collections that currently lack dense BGE/SPLADE rows are not
automatically exempt from these sparse-QE baselines. Exempt a dataset only with
a logged, reproducible feasibility failure after the streaming/index-reuse path
has been attempted within the configured memory limit.

## 9. Interpretation rules

- Compare `BGE_Vocab_QE` directly with `BM25_Default` and the fixed classical
  baselines. Describe it as frozen-BGE term-neighbour vocabulary QE, not as a
  Kuzi/Word2Vec reproduction and not as dense document retrieval.
- Compare `LLM_Q2E_ZS` with `BM25_Default` and `BGE_Vocab_QE`, while making its
  local-model substitution, generation latency, and model dependency visible.
- A negative `LLM_Q2E_ZS` result does not establish that strict synonym
  generation, few-shot Q2E, CoT, Query2doc, HyDE, or PRF-assisted LLM expansion
  is ineffective.
- A result that improves Recall@K but reduces nDCG@10 is a trade-off, not an
  unconditional gain. Report its query-level frequency and cost.
- If an LLM output is filtered away because it is absent from the corpus index,
  report the event rather than interpreting it as a semantic failure.
- Do not claim a later context-aware method is effective merely because it
  exceeds BM25; compare it with this frozen BGE vocabulary QE control. Exact
  causal attribution may additionally use a separately named matched-budget
  ablation, but the literature-like primary baseline must not be rewritten to
  match CRVE.
- `oracle_ndcg_10_from_1000` means that an oracle reorders the retrieved top
  1,000 candidates and then measures nDCG@10. It is not an oracle over the full
  corpus and must not be interpreted as a different retrieval cutoff.
- LLM cache replay is the reproducibility unit. Temperature zero does not claim
  bit-identical regeneration across different hardware, backends, or model
  builds; those environment fields remain part of the manifest.

## 10. Completion criteria

The baseline implementation is complete only when `BGE_Vocab_QE` and
`LLM_Q2E_ZS` are reproducible from versioned caches, all required tests pass,
the fixed-config pilot is recorded before the full sweep, and full results
include effectiveness, candidate-funnel, resource, generation, and provenance
fields. The optional Word2Vec spot-check and restricted-synonym diagnostic are
not completion requirements. Update the architecture, metric documentation,
and results-to-scripts mapping only after the implementation has empirical
evidence to support those changes.
