# PyTerrier Baseline Harness Review — Decisions and Validation Gates

> **Scope:** Baseline-only decisions for the PyTerrier evaluation harness. This document reviews the recorded first run in [results/pyterrier_baselines/pyterrier_baselines_results.csv](../results/pyterrier_baselines/pyterrier_baselines_results.csv), the current harness, and the next baseline suite. It does not select or evaluate a proposed Edge-RAG method.
>
> **Status:** Decisions are split into supported operational choices and validation gates. A result may be used as a comparison baseline only after its retrieval semantics, metrics, and timing boundaries are verified.

This document complements [the PyTerrier adoption plan](pyterrier_adoption_plan.md) and [the evaluation-metrics contract](EVALUATION_METRICS.md).

---

## 1. Findings from the recorded baseline run

The first CSV contains 25 datasets, not 24. On nDCG@10, BM25_Analyzed is lower than BM25_Default on 20 datasets, higher on four, and nearly tied on NQ.

| Finding | Evidence | Decision relevance |
|---|---|---|
| Default BM25 is the stronger general lexical control in this run | BM25_Analyzed loses on 20/25 datasets by nDCG@10 | Use default Terrier BM25 as the primary lexical baseline |
| Analyzer effects are not isolated | The analyzed branch changes tokenization, stopwords, stemming, irregular forms, technical-token treatment, and query encoding together | Do not claim that a particular component caused the gap |
| PRF effectiveness is dataset-dependent | Native RM3 improves some datasets and harms others; custom RM3 variants can diverge sharply from native RM3 | Preserve native RM3 and validate custom RM3 before interpreting it |
| Reported latency is harness time | The timer wraps query preparation, retrieval, row iteration, exclusion filtering, and metric calculation | Do not call it retrieval latency or derive tail latency from it |
| Cached TTI of zero is cache reuse, not zero construction cost | Cached index loading writes zero construction time in the results | Separate build cost, cache-load time, and query latency |

Examples of analyzer exceptions matter: analyzed BM25 improves nDCG@10 on Quora (0.7676 → 0.7847) and Touché (0.2938 → 0.3573). The practical choice below is not a claim that standard processing is uniformly superior.

---

## 2. Decision 1 — Primary lexical baseline and analyzer policy

### Decision

Use Terrier's standard English pipeline and BM25_Default as the **primary lexical baseline** for the next suite. Retain the current custom analyzed index only as a named diagnostic control until the analysis work is complete.

This is a practical baseline decision based on the recorded general-benchmark result. It does not authorize removing EdgeRAGAnalyzer, compound handling, or analyzed-index support from the production pipeline.

### What the data do and do not show

The current comparison does not identify the cause of the performance gap. It jointly changes:

- tokenization and treatment of punctuation/technical compounds;
- stopword policy;
- KStem versus Terrier's default stemmer;
- WordNet irregular-form overrides;
- indexing through WhitespaceTokeniser with empty Terrier term pipelines;
- custom MatchOp query encoding.

Therefore the following claims are unsupported by the CSV alone:

- that SMART stopwords or Porter stemming caused the difference;
- that technical-compound preservation has no value;
- that a technical-domain effect was merely a shared-tokenizer effect.

### Required control

Whenever a system uses a non-default lexical pipeline, include its matched lexical control:

system − BM25(same corpus, tokenizer, stemmer, stopwords, query encoding, and scorer)

This isolates any higher-level retrieval change from lexical processing. It does not require the custom analyzer to remain the primary baseline.

### Optional targeted evidence

A small one-factor analysis may test compound preservation versus splitting while holding stopwords, stemming, corpus, query encoding, and scorer fixed. It should be reported as analyzer evidence, not used to retrofit a causal explanation into the original run.

### Cache identity

Index cache keys must retain every input that can change retrieval behavior: corpus identity, dataset version, Terrier/PyTerrier version, indexing properties, tokenizer, stopword policy, stemmer, and pre-processing source. Removing analyzer-hash cache keying is appropriate only when an equivalent full configuration identity replaces it.

---

## 3. Decision 2 — RM3 and PRF validity before roster expansion

### Decision

Keep BM25_RM3_Terrier_Default as the established RM3 baseline. Treat BM25_RM3_Unified_Default and BM25_RM3_Unified_Analyzed as **custom diagnostic variants** until they pass retrieval-semantics validation.

A custom implementation may remain useful, especially for a controlled formulation across two indices. It must not be described as interchangeable with native RM3 merely because it uses an RM3-style interpolation.

### Validation gates

Before reporting a custom RM3 result as a baseline:

1. **Feedback-document text:** confirm that stored text metadata is not truncated for feedback use, or make the truncation explicit and apply it consistently.
2. **Term-space parity:** derive feedback terms from the same representation used by the index. Raw-text splitting, punctuation stripping, and a different stopword list can create terms absent from the lexicon. A missing lexicon entry must not silently receive collection frequency 1 as if it were a valid indexed term.
3. **Second-pass query parsing:** verify that weighted MatchOp expressions are parsed and scored as intended in both default and analyzed branches.
4. **Exclusion policy:** apply BRIGHT excluded-document filtering before feedback selection as well as before final evaluation. Excluded documents must not influence PRF.
5. **Depth after exclusions:** retrieve enough documents to preserve the requested eligible depth after exclusions; otherwise state the actual depth.
6. **Native agreement test:** on a pinned small corpus and fixed parameters, compare first-pass ranks, feedback terms, second-pass query terms, and final rankings against native RM3. Differences should be explained by declared formulation choices.

The large gap on Quora between native RM3 (nDCG@10 0.7508) and unified-default RM3 (0.5774) makes this validation a prerequisite, not a polish item.

### PRF baselines after validation

| Baseline | Role |
|---|---|
| BM25_Default | Primary lexical control |
| Native Terrier RM3 | Established relevance-model PRF control |
| BM25 + Bo1 | A second classical feedback family with distinct term selection |
| Validated Unified RM3 | Optional controlled implementation, explicitly named by its formulation |
| DPH | Optional DFR lexical-scoring control |

Bo1 broadens the feedback comparison; it does not exhaust the PRF design space. Report fixed feedback depth, feedback term count, interpolation weight, retrieval depth, and all tuning choices for every PRF condition.

---

## 4. Decision 3 — metric contract

### 4.1 Benchmark-compatible primary metrics

Use the benchmark's official or standard evaluation definition as the primary reported metric. For BEIR-style results, this means the standard nDCG cut convention with qrel grades as provided. Do not label an exponential-gain variant as BEIR parity on graded qrels.

If exponential gains are useful for a project-specific analysis, report them separately with an unambiguous name such as Exp-nDCG@10, the gain mapping, and a statement that they are not the official BEIR headline metric.

Keep:

- nDCG@10 and nDCG@100;
- Recall@10, @100, @500, and @1000;
- MRR@10 where suitable for the task;
- MAP@100 where the benchmark and judgments make it meaningful;
- P@10 and P@100, labeled as judged precision;
- hit rate / Strict@10 as a supplemental binary-success metric.

Do not use one metric as a substitute for another. Strict@K measures whether any judged relevant item was found; it does not measure precision or all-relevant-document recall.

### 4.2 Candidate-funnel diagnostics

Add the following after the retrieval output and qrel handling are validated:

| Diagnostic | Definition and reporting rule |
|---|---|
| Recall@100, @500, @1000 | Fraction of all judged relevant documents retrieved by the stated depth |
| All-relevant recall / completeness@K | Fraction of queries for which every judged relevant document is present by K; do not equate it with answerability unless the dataset defines all gold evidence as necessary |
| Oracle nDCG@10 from K | nDCG@10 of an ideal reranking of the retrieved K documents, normalized by the full-qrel ideal ranking |
| Strict@100, @1000 | Binary hit diagnostics, secondary to Recall@K |
| P@100 | Judged precision; annotate incomplete-qrel limitations |
| K90/K99 | Smallest depth reaching 90%/99% of all judged-relevant recall; record >1000 when unreachable rather than censoring it |

For multi-document or multi-hop data, flat qrels do not always say that every gold item is required to answer. Completeness is therefore a retrieval diagnostic, not proof of downstream failure.

### 4.3 Deferred metrics

Defer score margins and confidence calibration until a routing or confidence model exists. Defer expansion-specific telemetry from the baseline harness. QPS belongs only to a declared multi-query serving experiment.

Rank summaries such as median first-gold rank may be computed on demand from per-query runs. They should not replace the recall curve.

---

## 5. Decision 4 — efficiency measurement

### Decision

Efficiency is a first-class baseline outcome, reported alongside effectiveness. Do not use the current avg_latency_ms column as retrieval latency, latency percentiles, or an efficiency-ratio denominator.

The current value measures average end-to-end harness work per query, including preparation and evaluation. It remains a useful coarse run-cost field if renamed clearly.

### Required measurements

For every local baseline condition, record separately:

| Measurement | Boundary |
|---|---|
| Index build time | Raw corpus input to usable index; report even when a later run reuses the cache |
| Cache-load time | Existing index to ready retriever |
| Retrieval latency distribution | Per-query retrieval work only; include both passes for PRF |
| Evaluation overhead | Run construction, qrel conversion, result iteration, exclusion filtering, and metric calculation |
| Disk footprint | All index/cache files associated with the condition |
| Host-memory peak | Process and, when available, system memory during build and retrieval |
| GPU-memory peak | Relevant neural baselines, measured separately |
| Throughput | Only in a declared multi-query serving setup |

Report P50/P95/P99 only from individual-query samples, with query count and warm/cold condition. For 49–50-query datasets, P99 is near the maximum and should be labeled as an unstable estimate.

Keep construction, cache reuse, and warm query latency separate. A cached result with zero build time means “not measured in that invocation,” not “free to construct.”

### Efficiency comparisons

Use tables or quality-versus-latency/memory plots at explicit resource limits as the primary comparison. Recall@1000 / ms and Recall@1000 / MB may be supplemental descriptive ratios, but they are not an efficiency frontier and can reward an inadequate low-recall system.

For dense and learned-sparse baselines, report model loading, document encoding, index construction, search backend, precision, and approximation/pruning settings separately from warm query latency.

---

## 6. Decision 5 — baseline roster

### Core local suite

| Baseline | Status | Required specification |
|---|---|---|
| BM25_Default | Required | Terrier/PyTerrier version, index configuration, query parser and retrieval depth |
| Native Terrier RM3 | Required | Feedback documents/terms, interpolation, first/final-pass depth |
| BM25 + Bo1 | Required after RM3 validation | Feedback and query-expansion configuration |
| Dense BGE-small | Required when hardware permits | Exact checkpoint, document construction/truncation, embedding precision, search index and search parameters |
| SPLADE | Required when hardware permits | Exact checkpoint, document construction/truncation, index format, pruning and retrieval parameters |
| DPH | Useful extension | Terrier weighting-model configuration |

Dense and SPLADE baseline installation is not enough to define a condition. The model checkpoint, document construction, exact/approximate search policy, and resource measurement boundary are part of the baseline.

### Deferred reference systems

ColBERT, PLAID, and PLAID-PRF may be cited or reported as external published references. Keep published values separate from locally reproduced results: do not put them in the same aggregate table, compute differences, or create a gap-closed figure unless corpus, qrels, splits, metrics, and resource protocol are aligned.

---

## 7. Execution order

| Order | Work | Gate |
|---|---|---|
| 1 | Freeze corpus/query/qrel manifests; record software and index configuration identities | A baseline run can be reproduced from its artifact metadata |
| 2 | Correct metric definitions and emit per-query run records at depth 1,000 | Official-style metrics agree with a reference evaluator on representative datasets |
| 3 | Separate timing boundaries and record disk/host/GPU memory | Reported latency has a clear per-query retrieval boundary |
| 4 | Validate native and custom RM3 semantics, including BRIGHT exclusions | Custom RM3 is either validated and retained or labeled diagnostic |
| 5 | Add Bo1 and DPH | Classical lexical/PRF suite is complete enough for its purpose |
| 6 | Add BGE-small and SPLADE with pinned configurations | Modern local comparisons are reproducible and resource-accounted |
| 7 | Review analyzer controls | Choose whether to keep, retire, or further study the custom lexical pipeline |

This order prevents expensive new baselines from inheriting ambiguous metrics, timing, or PRF semantics.

---

## 8. Decision summary

| Question | Decision |
|---|---|
| Primary lexical baseline | Use standard Terrier BM25 because it wins on most datasets in the current run |
| Cause of analyzed-BM25 regressions | Unresolved; do not attribute it to compounds, stopwords, or stemming without controlled evidence |
| Custom analyzer | Retain temporarily as a diagnostic control; do not remove production code based on baseline CSV alone |
| RM3 control | Native Terrier RM3 is primary; custom unified RM3 requires validation |
| Metrics | Use benchmark-compatible nDCG as primary; add deep recall and candidate-pool diagnostics with precise definitions |
| Efficiency | Measure per-query retrieval latency and resources separately from harness evaluation; ratios are supplemental |
| Next baselines | Bo1, BGE-small, SPLADE, then DPH; defer ColBERT/PLAID local runs |

**Guiding principle:** baseline results must be reproducible, semantically comparable, and measured at clear effectiveness and resource boundaries before they are used to judge any retrieval method.
