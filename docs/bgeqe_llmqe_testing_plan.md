# BGEQE and LLMQE Testing Plan for the PyTerrier Baseline Harness

> Status: implementation plan, 2026-09-09. This plan replaces the completed,
> now-duplicative `docs/implementation_plan.md`. It defines two sparse lexical
> query-expansion baselines and their evaluation protocol. It does not define
> the proposed CRVE method; see
> [the corpus-informed QE research plan](corpus_informed_query_expansion_plan.md)
> for that work.

## 1. Objective and comparison boundary

Add two query-expansion baselines to the standard Terrier/PyTerrier harness:

| ID | Name | Evidence available at expansion time | Purpose |
|---|---|---|---|
| `BGE_Vocab_QE` | Simple BGE vocabulary QE | A static corpus vocabulary and frozen BGE-small embeddings | Direct control for term-only static corpus-informed QE. |
| `LLM_Synonym_QE` | Local LLM synonym QE | Query text and a declared local LLM; no retrieved documents | Query-only external-knowledge QE reference. |

Neither baseline is a dense document retriever, PRF method, Query2doc method,
or CRVE variant. Both use **one standard Terrier BM25 retrieval** after query
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
- Preserve the standard raw-query sanitization and the full candidate depth of
  1,000. Query chunks remain `chunk_size=200`.
- For BRIGHT, rewrite before the sole retrieval, request the existing padded
  depth `min(1000 + max_excluded, 3000)`, then apply the existing post-retrieval
  exclusion filter and retain 1,000 eligible documents. There is no feedback
  pass, therefore no PRF-style pre-filter stage.
- Keep original query terms in every rewritten query. If expansion produces no
  admissible term, the result must be exactly the `BM25_Default` retrieval for
  that query under the same index and exclusion conditions.
- Use the canonical primary metrics: linear `ndcg_10`, `ndcg_50`, `ndcg_100`,
  `map_100`, `mrr_10`, precision, Recall@10--1000, Strict@K, completeness, and
  oracle-nDCG@10. Keep `exp_ndcg_10` as the declared supplemental metric.
- Do not overwrite `results/pyterrier_baselines/pyterrier_baselines_results.csv`.
  QE results require additional provenance and telemetry, so write a separate,
  timestamped raw result directory and a dedicated append-only QE summary.
- Do not use qrels, rankings, retrieved document text, or test labels to choose
  terms, adjust weights, or recover malformed LLM output.

## 3. Shared lexical-expansion contract

The two baselines must differ only in how they propose candidates. They share
the lexical validation, admission, weighting, retrieval, and evaluation path.

### 3.1 Candidate representation

Every candidate has both a display form and a validated Terrier retrieval form:

```text
display_surface: natural corpus surface form used for BGE encoding and logs
retrieval_term:  form accepted by the default Terrier query pipeline and present in its lexicon
```

This distinction is mandatory. Encoding Porter-style stems such as `comput` or
injecting a natural surface form that does not resolve to an index term would
make the baseline unreliable.

### 3.2 Analyzer-parity gate

Before the full experiment, implement and test a deterministic surface-to-index
mapping path. It must:

1. derive candidate statistics from the default Terrier index or a streamed
   corpus pass;
2. confirm that the emitted retrieval term is in the loaded index lexicon;
3. verify on a declared sample that the rewritten Terrier query reaches the
   intended lexicon entry;
4. reject terms that cannot be represented safely.

The implementation may use a streaming sidecar to retain the most frequent
natural surface form for each verified index term. If it needs an additional
corpus pass, record that pass and its elapsed time as preparation cost. It must
not silently use `EdgeRAGAnalyzer` or claim analyzer parity without a test.

### 3.3 Shared admission and weighting

Initial frozen configuration, subject only to a declared development-pilot
decision before the full suite:

| Parameter | Initial value | Rationale |
|---|---:|---|
| vocabulary cap | 10,000 eligible terms | Bounded static sidecar; directly comparable with later CRVE work. |
| candidate proposal count | 50 | Leaves room for validity filtering without expensive lexical fan-out. |
| final expansion cap | 5 terms | Small, interpretable lexical intervention. |
| expansion mass \(\mu\) | 0.25 of original query mass | Original query remains dominant. |
| term allocation | normalized nonnegative candidate score within \(\mu\) | Prevents accidental score inflation from term count. |
| duplicate/original-term policy | remove | Do not spend expansion mass on original query terms. |
| maximum DF / postings rule | record first; gate only after development evidence | Avoid adding an arbitrary common-term filter before measuring its effect. |

The implementation must render the weighted Terrier query through a single,
unit-tested query-construction function. It must escape or reject parser syntax
and preserve the unexpanded query when no validated additions remain.

The initial full benchmark uses exactly one frozen configuration. Any sweep of
vocabulary size, final-term count, mass, candidate count, or term weighting is
development work and must be reported separately from held-out results.

## 4. `BGE_Vocab_QE`: simple static dense-vocabulary expansion

### 4.1 Definition

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
An anchor-max form may be a later diagnostic ablation, but must not replace the
primary control or be presented as the same method.

### 4.2 Static vocabulary sidecar

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

Select the 10,000 terms by the declared static salience rule after eligibility
filtering. The exact rule and all filters must be written into the manifest.
Do not choose terms using query labels or final retrieval outcomes.

Encode each natural display surface exactly once. Use the BGE query encoding
instruction for the raw user query and the passage/document encoding path for
the vocabulary surfaces, L2-normalize both, and use dot products as cosine
scores. Record the exact library and model revision. A pilot may verify the
instruction choice, but it must be frozen before the full sweep.

### 4.3 Query-time transform

Implement a small registered PyTerrier transformer, conceptually:

```text
raw query
  -> BGE query embedding
  -> top-50 static vocabulary candidates
  -> lexicon/duplicate/parser validation
  -> normalized score allocation across the admitted top five
  -> weighted TerrierQL query
  -> existing BM25 retriever and exclusion filter
```

The transformer operates on a query DataFrame and returns the same `qid` with
the rewritten query plus telemetry columns. It must batch BGE query encoding
for each input chunk rather than encode one query at a time. Term vectors remain
resident in a contiguous matrix; no corpus text or document embeddings are
loaded at query time.

Register the transformer explicitly in `PyTerrierBaselineHarness` rather than
relying on the current incidental `self.pipelines` fallback. The registration
contract should make the required resources, cache identity, query-rewrite
function, and pipeline label visible to the harness.

### 4.4 Required BGEQE telemetry

For each dataset and pipeline run, record:

- vocabulary cap and realized eligible count;
- sidecar storage, preparation time, corpus-pass count, cache hit/miss;
- query encoding, matrix search, validation/allocation, and BM25 timings;
- candidate count, admitted-term mean/distribution, unused mass, mean DF and
  total selected DF as a postings-cost proxy;
- terms rejected for duplication, lexicon mismatch, parser safety, or filters;
- deterministic seed and cache/model identity.

Persist a bounded per-query trace containing raw query ID, selected retrieval
terms, display surfaces, scores, weights, rejections, and rewrite time. Do not
store full document rankings here; the harness's compressed candidate-run cache
remains the ranking artifact.

## 5. `LLM_Synonym_QE`: query-only lexical synonym baseline

### 5.1 Definition and fairness boundary

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

### 5.2 Fixed local model and prompt

Use the existing local model profile `qwen3.5-4b` in `configs/models.yaml`
unless a documented availability check requires a different configured profile.
Record the resolved backend, tag, model-file digest where applicable, endpoint,
hardware, context window, and model load condition.

Use deterministic decoding: temperature 0, a fixed seed where the backend
supports it, fixed token limit, and no tools. The prompt must be versioned and
stored with the cache. Initial prompt contract:

```text
Return strict JSON only: {"terms": ["..."]}.
Given the search query, provide at most five short English single-word lexical
alternatives that preserve its information need. Do not repeat query words.
Do not explain, answer the query, add punctuation syntax, or invent details.
Query: {query}
```

The prompt may be adjusted only during the declared pilot. Once frozen, its
content and hash are part of every cache and result identity.

### 5.3 Generation cache and failure behavior

Generate once per query and cache the raw response separately from the
corpus-specific validated term list:

```text
data/cache/qe_llm/raw/{model_identity}/{prompt_hash}/{query_hash}.json
data/cache/qe_llm/validated/{dataset}/{index_identity}/{generation_identity}.jsonl
```

Each raw record contains the request parameters, response text, parse result,
generator timing, backend/model identity, and failure state. Each validated
record additionally contains admitted/rejected terms and reasons.

Malformed JSON or unusable output produces an empty expansion and is logged;
there is no hidden repair prompt. Transient backend failures stop the worker so
the normal resume mechanism can retry deterministically from the cache boundary.
This separates model-quality failures from infrastructure failures and prevents
unreported extra generation calls.

### 5.4 LLMQE timing report

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

## 6. Harness and runner changes

### 6.1 Files to add or modify during implementation

| Path | Planned change |
|---|---|
| `src/evaluation/pyterrier_harness.py` | Add explicit custom-pipeline registration, static-vocabulary QE resources, query-rewrite telemetry collection, and resource fields for QE sidecars. Preserve existing six Terrier, dense BGE, and SPLADE behavior. |
| `src/evaluation/pyterrier_qe.py` | Add isolated, testable components for analyzer-parity mapping, vocabulary-sidecar construction/loading, BGE vocabulary rewriting, LLM cache validation, and shared weighted-query construction. Do not import Pipeline V2. |
| `scripts/run_pyterrier_qe_baselines.py` | Add a sequential, resumable CLI runner for `BGE_Vocab_QE` and `LLM_Synonym_QE`; the currently documented PyTerrier runner is not present in this checkout, so this script becomes the explicit QE entry point. |
| `configs/pyterrier_qe.yaml` | Add all frozen QE parameters: cap, eligibility, model, prompt version, output cap, mass, timing protocol, seed, cache locations, and run-selection options. |
| `tests/test_pyterrier_qe.py` | Add deterministic unit/integration tests described below. |
| `results/pyterrier_baselines/` | Add timestamped QE raw outputs and a dedicated append-only `pyterrier_qe_results.csv`; leave the existing baseline CSV intact. |
| `scripts/results_scripts_mapping.md` | Add the new output-to-runner mapping if that mapping is restored or present at implementation time. |

`src/evaluation/pyterrier_qe.py` is baseline code: it must remain isolated from
`src/pipeline_v2/` and `src/legacy_pipeline/`. It may reuse only general
evaluation/loading utilities and the standard PyTerrier index.

### 6.2 Result schema

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
qe_candidates_mean, qe_terms_mean, qe_unused_mass_mean, qe_selected_df_mean,
qe_selected_df_p95, qe_rewrite_p50_ms, qe_rewrite_p95_ms,
llm_model, llm_prompt_hash, llm_generation_p50_ms, llm_generation_p90_ms,
llm_generation_p99_ms, llm_end_to_end_p50_ms, llm_json_valid_rate,
llm_empty_rate, llm_lexicon_reject_rate
```

Fields not applicable to a pipeline are empty, not zero. Store the full
environment and configuration manifest alongside every timestamped result file.

### 6.3 CLI and execution isolation

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
```

Run one dataset in an isolated subprocess and release Python, JVM, and GPU
resources before the next dataset. Never run this workload concurrently with
the currently active dense/SPLADE job. The runner must resume only completed,
config-identical dataset/pipeline records and must never overwrite a previous
result artifact.

## 7. Tests and gates before full evaluation

### 7.1 Required automated tests

1. Candidate mapping rejects a surface form that has no verified Terrier
   lexicon representation and accepts a known valid form.
2. With zero admitted expansions, the rewritten pipeline produces the same
   ranked documents and scores as `BM25_Default` on a fixture index.
3. Fixed BGE vectors/query inputs produce identical top candidates, weights,
   and rewritten queries across runs with the same seed.
4. Expansion weights are nonnegative, sum to at most \(\mu\), and preserve all
   original-query terms.
5. BGE query encoding is batched per query chunk; no raw corpus text is loaded
   on the query path.
6. LLM JSON parsing, duplicate removal, lexicon filtering, cache identity, and
   empty-expansion fallback behave deterministically using mocked responses.
7. BRIGHT exclusion filtering still returns up to 1,000 eligible candidates and
   never exposes excluded documents after either QE rewrite.
8. A resumed run skips only records with matching dataset, pipeline, config
   hash, model identity, and seed.

### 7.2 Smoke and pilot gates

| Gate | Dataset scope | Required evidence |
|---|---|---|
| A: lexical parity | fixture plus one small corpus | Verified surface/index mapping and no-query-expansion parity with BM25. |
| B: BGEQE smoke | `scifact` and one BRIGHT corpus | Sidecar builds within memory cap, deterministic results, full exclusion semantics. |
| C: LLMQE prompt/cache smoke | same two corpora | Strict JSON behavior, recorded generation timings, no undocumented retry path. |
| D: pilot | `scifact`, `fiqa`, `quora`, `trec_covid`, `bright_stackoverflow` | Freeze model, prompt, caps, weighting, and any admissibility rule without using held-out results. |
| E: full sweep | all 25 datasets, sequentially | Complete raw/summary artifacts, environment manifests, and no breach of 15 GiB host-RAM limit. |

The five large collections that currently lack dense BGE/SPLADE rows are not
automatically exempt from these sparse-QE baselines. Exempt a dataset only with
a logged, reproducible feasibility failure after the streaming/index-reuse path
has been attempted within the configured memory limit.

## 8. Interpretation rules

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
  control under matching vocabulary, candidate, term-count, and mass budgets.

## 9. Completion criteria

The baseline implementation is complete only when both pipeline labels are
reproducible from versioned caches, all required tests pass, pilot decisions are
recorded before the full sweep, and full results include effectiveness,
candidate-funnel, resource, generation, and provenance fields. Update the
architecture, metric documentation, and results-to-scripts mapping only after
the implementation has empirical evidence to support those changes.
