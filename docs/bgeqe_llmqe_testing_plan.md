# BGE Vocabulary Query Expansion (BGEQE) and LLM Synonym Query Expansion (LLMQE) Testing Plan

> Status: implementation plan, 2026-09-09. This plan replaces the completed,
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
| `BGE_Vocab_QE` | Simple BGE vocabulary QE | A static corpus vocabulary and frozen `BAAI/bge-small-en-v1.5` embeddings | Direct control for term-only static corpus-informed QE. |
| `LLM_Synonym_QE` | Local large-language-model (LLM) synonym QE | Query text and a declared local LLM; no retrieved documents | Query-only external-knowledge QE reference. |

Neither baseline is a dense document retriever, pseudo-relevance-feedback
(PRF) method, Query2doc-style LLM pseudo-document expansion method, or CRVE
variant. Both use **one standard Terrier BM25 lexical retrieval** after query
rewriting. Both retain the original query and use the same final expansion
budget. The existing `BGE_Small_Dense` row is a dense bi-encoder baseline and
must not be relabelled or reused as `BGE_Vocab_QE`.

The experiment asks three narrow questions:

1. Does a frozen encoder plus a bounded target-corpus vocabulary improve over
   `BM25_Default` without document-level dense retrieval or PRF?
2. Does query-only synonym generation help sparse retrieval under the same
   lexical expansion budget?
3. Does a later context-aware method improve over the simple BGE vocabulary
   control, rather than only over unexpanded BM25?

## 2. Non-negotiable evaluation contract

- Use the cached standard Terrier default index already managed by
  `PyTerrierIndexManager`; do not build a custom-analyzer index for these
  baselines.
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
  oracle-nDCG@10. Keep `exp_ndcg_10` as the declared supplemental metric.
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
| \(t\) | One candidate expansion term after Terrier analysis. |
| \(E(Q)\) | The ordered set of validated expansion terms admitted for query \(Q\); its size is at most five. |
| \(|E(Q)|\) | Number of terms in \(E(Q)\). Vertical bars around a set mean its number of elements. |
| \(w_{\text{orig}}(t)\) | Terrier's original-query weight for term \(t\). It is zero when \(t\) is not an original query term. |
| \(\delta w(t)\) | The nonnegative weight added by query expansion for term \(t\). The symbol \(\delta\) means an added change, not a probability. |
| \(w_{\text{final}}(t)\) | The weight sent to retrieval: \(w_{\text{orig}}(t)+\delta w(t)\). Original terms are excluded from \(E(Q)\), so an admitted expansion normally has \(w_{\text{orig}}(t)=0\). |
| \(A(Q)\) | Original query mass, defined as \(\sum_t w_{\text{orig}}(t)\) over \(T_Q\). Here "mass" means only the sum of query-term weights. |
| \(\mu\) | Expansion-budget multiplier, initially 0.25. |
| \(A_{\max}\) | Cap applied to original query mass when calculating the expansion budget, initially 5.0. |
| \(B(Q)\) | Maximum total expansion weight for query \(Q\), defined below. |
| \(w_{\max}\) | Maximum added weight for any one expansion term, initially 0.25. |
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
| BGEQE | `BGE_Vocab_QE`, the static vocabulary expansion baseline defined in Section 5. |
| LLM | Large language model. |
| LLMQE | `LLM_Synonym_QE`, the query-only synonym baseline defined in Section 6. |
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

Metric names not expanded here follow
[the canonical metric definitions](EVALUATION_METRICS.md).

## 4. Shared lexical-expansion contract

The two baselines must differ only in how they propose candidates. They share
the lexical validation, admission, weighting, retrieval, and evaluation path.

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

Before the full experiment, implement and test a deterministic surface-to-index
mapping path. It must:

1. derive candidate DF, collection frequency, IDF, and salience statistics
   directly from the cached default Terrier lexicon;
2. confirm that the emitted retrieval term is in the loaded index lexicon;
3. verify on a declared sample that the rewritten Terrier query reaches the
   intended lexicon entry;
4. reject terms that cannot be represented safely.

The lexicon contains analyzed index terms but does not reliably preserve the
natural surface forms that BGE should encode. First extract an oversized,
bounded candidate set from the lexicon, initially the top 20,000 terms. Then
make one streaming corpus pass to retain the most frequent natural surface form
that Terrier maps to each candidate. This pass recovers surfaces; it must not
recount the full corpus vocabulary. Record its elapsed time and bytes read as
preparation cost. Future index builds may collect this mapping during the
existing indexing stream, but existing cached indices must not be rebuilt only
to avoid reporting the recovery pass.

Do not silently use `EdgeRAGAnalyzer` or claim analyzer parity without a test.
Do not embed Porter-style stems as if they were natural words unless a separately
named diagnostic baseline explicitly tests that shortcut.

### 4.3 Shared admission and weighting

Initial frozen configuration, subject only to a declared development-pilot
decision before the full suite:

| Parameter | Initial value | Rationale |
|---|---:|---|
| vocabulary cap | 10,000 eligible terms | Bounded static sidecar; directly comparable with later CRVE work. |
| final expansion cap | 5 terms | Small, interpretable lexical intervention. |
| expansion-budget multiplier \(\mu\) | 0.25 | Controls the total added query weight. |
| capped original mass \(A_{\max}\) | 5.0 | Prevents paragraph-length queries from creating a large expansion budget. |
| per-expansion weight cap \(w_{\max}\) | 0.25 | Prevents one expansion from outweighing an ordinary original term of weight 1.0. |
| term allocation | uniform across admitted terms, subject to both caps | Makes BGEQE and LLMQE comparable even though only BGE supplies cosine scores. |
| duplicate/original-term policy | remove | Do not spend expansion mass on original query terms. |
| maximum DF / postings rule | record first; gate only after development evidence | Avoid adding an arbitrary common-term filter before measuring its effect. |

Define the total expansion budget as:

\[
B(Q)=\mu\min(A(Q),A_{\max}).
\]

The function \(\min(x,y)\) returns the smaller of \(x\) and \(y\). Thus,
queries with original mass above 5.0 do not receive a larger expansion budget.

For a nonempty admitted set \(E(Q)\), assign every expansion the same added
weight:

\[
\delta w(t)=\min\left(\frac{B(Q)}{|E(Q)|},w_{\max}\right)
\quad\text{for }t\in E(Q).
\]

Do not redistribute weight left unused by the per-term cap. The resulting
query weight is:

\[
w_{\text{final}}(t)=w_{\text{orig}}(t)+\delta w(t).
\]

BGE cosine similarity and LLM response order decide candidate rank only; they
do not produce primary-run term weights. Score-weighted BGEQE and rank-decayed
LLMQE, where weight decreases with the LLM response position, are optional,
separately named ablations.

The implementation must represent the final query with PyTerrier's
pre-tokenized `query_toks` column, a dictionary from analyzed index term to
numeric weight. Do not manually inject `term^weight` strings into TerrierQL.
Analyze each natural expansion surface exactly once with Terrier's configured
tokenizer, stopword filter, and stemmer; confirm the resulting term in the
index lexicon; then merge it into `T_Q`. Reject and log candidates containing
unsupported syntax instead of silently changing their text.

The unit-tested query-construction function must preserve the unexpanded
`query_toks` dictionary when no validated additions remain.

The initial full benchmark uses exactly one frozen configuration. Any sweep of
vocabulary size, final-term count, mass, candidate count, or term weighting is
development work and must be reported separately from held-out results.

## 5. `BGE_Vocab_QE`: simple static dense-vocabulary expansion

### 5.1 Definition

`BGE_Vocab_QE` performs no document retrieval before expansion:

```text
stream/build vocabulary sidecar once
    -> encode the capped vocabulary once with frozen BGE-small
    -> encode query once
    -> cosine-score query against static vocabulary matrix
    -> validate/admit up to five terms
    -> one weighted Terrier BM25 retrieval
```

The primary version uses **whole-query-to-term** similarity. This is the
simplest interpretable dense-vocabulary QE control. It is deliberately not
V7-style per-anchor expansion, anchor weighting, bailout, or context reranking.
An anchor-max form, which scores a candidate by its largest similarity to any
analyzed content term in the query, may be a later diagnostic ablation. It must
not replace the primary control or be presented as the same method.

### 5.2 Static vocabulary sidecar

Create a versioned cache under:

```text
data/cache/qe_vocab/{dataset}/{cache_identity}/
```

It stores, at minimum:

- ordered `display_surface` and `retrieval_term` arrays;
- document frequency, IDF, and vocabulary-selection score;
- normalized FP16 BGE vectors and their embedding dimension;
- a machine-readable manifest with dataset identity/fingerprint, index path,
  index fingerprint, vocabulary-selection rule, eligibility filters, cap,
  encoder model/revision, encoder instruction policy, dtype, seed, source
  corpus-pass count, wall-clock preparation time, and creation timestamp.

Read DF and collection frequency directly from the cached Terrier lexicon.
Define static salience for an index term \(t\) as:

\[
\operatorname{Salience}(t)=\operatorname{IDF}(t)
\times\ln(1+\operatorname{DF}(t)).
\]

Here \(\ln\) is the natural logarithm. Salience is used only to select the
static vocabulary; it is not a query-dependent relevance score.

Select the oversized surface-recovery set and final 10,000-term vocabulary by
this declared rule after eligibility filtering. The exact IDF source, rule, and
filters must be written into the manifest. Do not choose terms using query
labels or final retrieval outcomes.

Encode each natural display surface exactly once. Use the BGE query encoding
instruction for the raw user query and the passage/document encoding path for
the vocabulary surfaces, L2-normalize both, and use dot products as cosine
scores. Record the exact library and model revision. A pilot may verify the
instruction choice, but it must be frozen before the full sweep.

### 5.3 Query-time transform

Implement a small registered PyTerrier transformer, conceptually:

```text
raw query
  -> BGE query embedding
  -> top-50 static vocabulary candidates
  -> Terrier-analysis, lexicon, duplicate, and syntax validation
  -> retain the admitted top five in cosine-score order
  -> uniform bounded weights in query_toks
  -> existing BM25 retriever and exclusion filter
```

The transformer operates on a query DataFrame and returns the same `qid` with
the rewritten `query_toks` plus telemetry columns. Preserve a separate raw-query
column because BGE must encode the original natural query, not Terrier stems or
the punctuation-stripped retrieval string. It must batch BGE query encoding
for each input chunk rather than encode one query at a time. Term vectors remain
resident in a contiguous matrix; no corpus text or document embeddings are
loaded at query time.

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
- candidate count, admitted-term mean/distribution, unused mass, mean DF and
  total selected DF as a postings-cost proxy;
- terms rejected for duplication, lexicon mismatch, parser safety, or filters;
- deterministic seed and cache/model identity.

Persist a bounded per-query trace containing raw query ID, selected retrieval
terms, display surfaces, scores, weights, rejections, and rewrite time. Do not
store full document rankings here; the harness's compressed candidate-run cache
remains the ranking artifact.

## 6. `LLM_Synonym_QE`: query-only lexical synonym baseline

### 6.1 Definition and fairness boundary

`LLM_Synonym_QE` generates lexical alternatives from the query alone, then
applies the same lexical admission and weighting contract as `BGE_Vocab_QE`:

```text
raw query -> local LLM generates short alternatives -> parse/filter/admit
         -> one weighted Terrier BM25 retrieval
```

It must not receive corpus documents, top-ranked documents, qrels, examples
from benchmark queries, or a corpus vocabulary prompt. Intersecting generated
outputs with the index lexicon is permitted only as output validation; it is not
treated as corpus evidence.

The primary baseline emits up to five **single lexical terms**, not a
pseudo-document or free-form rewrite. Multiword phrases, HyDE, Query2doc, and
LLM-generated Boolean/proximity queries are separate methods and are out of
scope for this baseline.

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
supports it, fixed token limit, no tools, and non-streaming output. For Ollama,
pass the following JSON schema through the API's `format` field and validate the
returned object locally:

```json
{
  "type": "object",
  "properties": {
    "terms": {
      "type": "array",
      "items": {"type": "string"},
      "maxItems": 5
    }
  },
  "required": ["terms"],
  "additionalProperties": false
}
```

Schema-constrained decoding enforces the output structure; it does not establish
that generated terms are relevant or valid index terms. The prompt must be
versioned and stored with the cache. Initial prompt contract:

```text
Return strict JSON only: {"terms": ["..."]}.
Given the search query, provide at most five short English single-word lexical
alternatives that preserve its information need. Do not repeat query words.
Do not explain, answer the query, add punctuation syntax, or invent details.
Query: {query}
```

The prompt may be adjusted only during the declared pilot. Once frozen, its
content and hash are part of every cache and result identity.

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
generator timing, backend/model identity, and failure state. Each validated
record additionally contains admitted/rejected terms and reasons.

Output that fails local schema validation or contains no usable term produces
an empty expansion and is logged;
there is no hidden repair prompt. Transient backend failures stop the worker so
the normal resume mechanism can retry deterministically from the cache boundary.
This separates model-quality failures from infrastructure failures and prevents
unreported extra generation calls.

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

Also report JSON validity rate, empty-expansion rate, raw/generated/admitted
term counts, duplicate and lexicon-rejection rates, and the fraction of queries
whose validated expansion is identical to unexpanded BM25.

## 7. Harness and runner changes

### 7.1 Files to add or modify during implementation

| Path | Planned change |
|---|---|
| `src/evaluation/pyterrier_harness.py` | Add explicit custom-pipeline registration, static-vocabulary QE resources, query-rewrite telemetry collection, and resource fields for QE sidecars. Preserve existing six Terrier, dense BGE, and SPLADE behavior. |
| `src/evaluation/pyterrier_qe.py` | Add isolated, testable components for analyzer-parity mapping, lexicon-statistics extraction, bounded surface recovery, vocabulary-sidecar construction/loading, BGE vocabulary rewriting, LLM schema/cache validation, and shared `query_toks` construction. Do not import Pipeline V2. |
| `scripts/run_pyterrier_qe_baselines.py` | Add a sequential, resumable CLI runner for `BGE_Vocab_QE` and `LLM_Synonym_QE`; the currently documented PyTerrier runner is not present in this checkout, so this script becomes the explicit QE entry point. |
| `configs/pyterrier_qe.yaml` | Add all frozen QE parameters: cap, eligibility, model, prompt version, output cap, mass, timing protocol, seed, cache locations, and run-selection options. |
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
completeness_100, completeness_500, completeness_1000, oracle_ndcg_10,
retrieval_api_p50_ms, retrieval_api_p90_ms, retrieval_api_p99_ms,
batch_throughput_qps, harness_per_query_ms,
qe_prepare_s, qe_sidecar_mb, qe_cache_hit, qe_vocab_cap, qe_vocab_realized,
qe_lexicon_extract_s, qe_surface_recovery_s, qe_surface_recovery_bytes,
qe_corpus_passes,
qe_candidates_mean, qe_terms_mean, qe_unused_mass_mean, qe_selected_df_mean,
qe_selected_df_p95, qe_rewrite_p50_ms, qe_rewrite_p95_ms,
llm_model, llm_model_digest, llm_prompt_hash, llm_unique_queries,
llm_raw_cache_hit_rate, llm_projected_generation_s,
llm_generation_p50_ms, llm_generation_p90_ms,
llm_generation_p99_ms, llm_end_to_end_p50_ms, llm_json_valid_rate,
llm_empty_rate, llm_lexicon_reject_rate
```

Fields not applicable to a pipeline are empty, not zero. Store the full
environment and configuration manifest alongside every timestamped result file.

### 7.3 CLI and execution isolation

The runner must accept at least:

```text
--datasets ... | --all-datasets
--pipelines BGE_Vocab_QE,LLM_Synonym_QE
--seed 42
--chunk-size 200
--config configs/pyterrier_qe.yaml
--resume
--results-dir results/pyterrier_baselines/<timestamp>
--llm-model-profile qwen3.5-4b
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
2. Vocabulary DF, collection frequency, IDF, and salience are read from the
   fixture Terrier lexicon without recounting the raw corpus.
3. With zero admitted expansions, the `query_toks` pipeline produces the same
   ranked documents and scores as `BM25_Default` on a fixture index.
4. Fixed BGE vectors/query inputs produce identical top candidates, weights,
   and rewritten queries across runs with the same seed.
5. For every query, expansion weights are nonnegative, their sum is at most
   \(B(Q)\), each is at most \(w_{\max}\), and all original-query terms and
   weights are preserved.
6. BGE query encoding is batched per query chunk; no raw corpus text is loaded
   on the query path.
7. LLM schema validation, duplicate removal, lexicon filtering, cache identity,
   global query deduplication, and empty-expansion fallback behave
   deterministically using mocked responses.
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
| B: BGEQE smoke | `scifact` and one BRIGHT corpus | Sidecar builds within memory cap, deterministic results, full exclusion semantics. |
| C: LLMQE prompt/cache smoke | same two corpora | Schema-constrained output, local validation, exact model identity, recorded generation timings, and no undocumented retry path. |
| D: pilot | `scifact`, `fiqa`, `quora`, `trec_covid`, `bright_stackoverflow` | Freeze model, prompt, caps, uniform weighting, admissibility rules, and projected full-generation time without using held-out results. |
| E: full sweep | all 25 datasets, sequentially | Complete raw/summary artifacts, environment manifests, and no breach of 15 GiB host-RAM limit. |

The five large collections that currently lack dense BGE/SPLADE rows are not
automatically exempt from these sparse-QE baselines. Exempt a dataset only with
a logged, reproducible feasibility failure after the streaming/index-reuse path
has been attempted within the configured memory limit.

## 9. Interpretation rules

- Compare `BGE_Vocab_QE` directly with `BM25_Default` and the fixed classical
  baselines, but do not imply that it replaces dense BGE retrieval.
- Compare `LLM_Synonym_QE` with `BM25_Default` and `BGE_Vocab_QE`, while making
  its generation latency and model dependency visible.
- A result that improves Recall@K but reduces nDCG@10 is a trade-off, not an
  unconditional gain. Report its query-level frequency and cost.
- If an LLM output is filtered away because it is absent from the corpus index,
  report the event rather than interpreting it as a semantic failure.
- Do not claim a later context-aware method is effective merely because it
  exceeds BM25; it must also be compared with this frozen BGE vocabulary QE
  control under matching vocabulary, final-term, per-term-weight, and total
  expansion-budget rules.

## 10. Completion criteria

The baseline implementation is complete only when both pipeline labels are
reproducible from versioned caches, all required tests pass, pilot decisions are
recorded before the full sweep, and full results include effectiveness,
candidate-funnel, resource, generation, and provenance fields. Update the
architecture, metric documentation, and results-to-scripts mapping only after
the implementation has empirical evidence to support those changes.
