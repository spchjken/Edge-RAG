# PyTerrier Classical Sparse Baselines — Reproducibility and Reporting Plan

> Status: retrospective specification, 2026-09-11. The six baseline runs are
> already complete in
> [`results/pyterrier_baselines/pyterrier_baselines_results.csv`](../results/pyterrier_baselines/pyterrier_baselines_results.csv).
> This document reconstructs the executed protocol from the retained harness,
> result artifact, canonical metric contract, and Git history. It is not a
> claim that the experiment was preregistered, and it does not alter existing
> scores.

## 1. Objective and comparison boundary

The classical suite provides two unexpanded sparse retrieval controls and four
pseudo-relevance-feedback (PRF) controls:

| Stored pipeline ID | Recommended paper label | First pass | Feedback | Second pass |
|---|---|---|---|---|
| `BM25_Default` | BM25 | — | — | BM25 |
| `BM25_Bo1_Terrier_Default` | BM25+Bo1 (10d/10t) | BM25 | Bo1 | BM25 |
| `BM25_RM3_Terrier_Default` | BM25+RM3 (10d/10t, \(\lambda=0.5\)) | BM25 | RM3 | BM25 |
| `DPH` | DPH | — | — | DPH |
| `DPH_Bo1_Terrier_Default` | DPH+Bo1 (10d/10t) | DPH | Bo1 | DPH |
| `DPH_RM3_Terrier_Default` | DPH+RM3 (10d/10t, \(\lambda=0.5\)) | DPH | RM3 | DPH |

Here `10d/10t` means ten pseudo-relevant feedback documents and ten feedback
terms. “BM25+PRF” or “DPH+PRF” is too ambiguous for a table or method section:
both Bo1 and RM3 are PRF methods, but they construct and weight expansion terms
differently. Always name the feedback model.

These baselines isolate three factors:

1. BM25 versus the parameter-free DPH document-scoring model;
2. no feedback versus corpus-local PRF;
3. Bo1 DFR expansion versus RM3 relevance-model expansion under the same
   first-pass and second-pass ranker.

They do not test a custom analyzer, the removed Unified RM3 implementation,
Pipeline V2, V7, neural sparse retrieval, dense retrieval, query-only LLM
expansion, or the proposed context-reranked vocabulary method.

### 1.1 What “default” means—and does not mean

`BM25_Default` and `DPH` invoke Terrier by weighting-model name without
per-dataset effectiveness tuning. They use the standard Terrier index and term
pipeline constructed by `IterDictIndexer`.

The suffix `_Terrier_Default` in the four stored PRF identifiers is historical
and must not be interpreted literally. The executed harness explicitly sets:

```text
fb_docs  = 10
fb_terms = 10
fb_lambda = 0.5  # RM3 only
```

Current PyTerrier documentation instead lists three feedback documents as the
default for both rewriters and \(0.6\) as RM3's default interpolation value.
Therefore, preserve the stored IDs in code and artifact joins, but do not call
their parameterization “Terrier default” in a paper or thesis. These are
fixed project configurations implemented with Terrier's native rewriters.

No retained tuning sweep establishes that 10 documents, 10 terms, or
\(\lambda=0.5\) is optimal. The completed experiment should be described as a
fixed zero-shot comparison, not a tuned PRF comparison.

### 1.2 Primary references

References checked 2026-09-11:

- Robertson et al., [Okapi at TREC-3](https://trec.nist.gov/pubs/trec3/papers/city.ps.gz), for the BM25 family.
- Amati and van Rijsbergen, [Probabilistic Models of Information Retrieval Based on Measuring the Divergence from Randomness](https://doi.org/10.1145/582415.582416), for the DFR framework and Bo1.
- Amati, [Frequentist and Bayesian Approach to Information Retrieval](https://lintool.github.io/robust04-analysis-papers/Amati2006_Chapter_FrequentistAndBayesianApproach.pdf), for hypergeometric DFR models including DPH.
- Abdul-Jaleel et al., [UMass at TREC 2004: Novelty and HARD](https://trec.nist.gov/pubs/trec13/papers/umass.novelty.hard.pdf), for RM3.
- PyTerrier, [Terrier retrieval](https://pyterrier.readthedocs.io/en/stable/terrier/retrieval.html) and [query rewriting and expansion](https://pyterrier.readthedocs.io/en/stable/terrier/rewrite.html), for the implementation used here.
- PyTerrier, [Robust04 experiment example](https://pyterrier.readthedocs.io/en/stable/experiments/Robust04.html), for the conventional `retrieve >> rewrite >> retrieve` comparison over BM25 and DPH.
- Thakur et al., [BEIR](https://arxiv.org/abs/2104.08663), and Su et al., [BRIGHT](https://arxiv.org/abs/2407.12883), for the benchmark families.

The paper should cite both the foundational method and the actual software
implementation. A generic citation to “PyTerrier” does not replace the BM25,
DFR/Bo1, DPH, or RM3 method citations.

## 2. Retained evidence and validation status

The current checkout contains:

| Evidence | Status | Meaning |
|---|---|---|
| `src/evaluation/pyterrier_harness.py` | **Pass** | The six pipeline definitions, indexing path, metrics, exclusions, and timing code are inspectable. |
| `pyterrier_baselines_results.csv` | **Pass** | Exactly 25 rows exist for each of the six pipeline IDs: 150 unique dataset/pipeline records with no duplicates. |
| `pyterrier_baselines_summary.md` | **Pass** | Human-readable rendering of the retained aggregate results. |
| Git history | **Pass** | The classical rows trace to commits `f1158a974` and `b7812d152`; the latter changed only the exponential-gain constant name in the classical harness path. |
| Standalone run supervisor/CLI | **Not verified** | `scripts/run_pyterrier_baselines.py` is referenced by documentation but is absent from the current checkout. |
| Exact package/JVM/environment manifest | **Not verified** | Dependencies are not version-pinned and the CSV has no environment or config hash. Current installed versions would not prove the historical run versions. |
| Persisted top-1,000 run files | **Not verified** | No classical candidate Parquet files are retained in the current checkout. |
| Per-query exclusion and feedback traces | **Not verified** | The CSV contains aggregate metrics but no per-query feedback depth or eligible-candidate count. |

Consequently, the effectiveness matrix is complete as an aggregate artifact,
but exact environment-level reproduction is incomplete. Do not use phrases
such as “bitwise reproducible” or “fully reproducible from the repository”
until the missing runner and versioned manifests are restored or the suite is
rerun under a frozen environment.

## 3. Definitions and notation

| Term or symbol | Definition |
|---|---|
| \(Q\) | One sanitized query submitted to Terrier. |
| \(D\) | One indexed corpus document. |
| \(t\) | A query or candidate expansion term after Terrier's term pipeline. |
| \(tf(t,D)\) | Frequency of term \(t\) in document \(D\). |
| \(qtf(t,Q)\) | Frequency or weight of term \(t\) in query \(Q\). |
| \(|D|\) | Length of document \(D\) in indexed terms. |
| \(\overline{|D|}\) | Mean indexed document length in the collection. |
| \(N\) | Number of indexed documents. |
| \(df(t)\) | Number of indexed documents containing \(t\). |
| \(F(Q)\) | Ordered first-pass pseudo-relevant document set for query \(Q\). |
| \(f_d\) | Number of feedback documents, fixed to 10. |
| \(f_t\) | Maximum number of feedback terms, fixed to 10. |
| \(\lambda\) | RM3 interpolation control supplied to PyTerrier, fixed to 0.5. Its directional interpretation must follow the resolved implementation version. |
| \(K_1\) | Number of first-pass candidates requested before BRIGHT filtering. |
| \(K_2\) | Number of final-pass candidates requested before BRIGHT filtering. |
| \(X_q\) | Set of document identifiers prohibited for BRIGHT query \(q\). |
| \(M_X\) | Maximum \(|X_q|\) among queries in the evaluated dataset. |
| PRF | Pseudo-relevance feedback: treating top-ranked first-pass documents as feedback without consulting qrels. |
| Bo1 | Bose–Einstein 1 DFR query-expansion model implemented by Terrier. |
| RM3 | Relevance Model 3 query expansion, which interpolates an estimated feedback model with the original query. |
| DPH | A parameter-free hypergeometric Divergence From Randomness document-weighting model. |

Metric definitions follow
[`docs/EVALUATION_METRICS.md`](EVALUATION_METRICS.md); this document specifies
which pipelines produced the candidate runs to which those metrics were
applied.

## 4. Shared data, index, and query contract

### 4.1 Dataset scope

All six classical pipelines have completed rows for the same 25 dataset IDs:
13 BEIR collections and 12 BRIGHT domains. The authoritative counts for this
specific execution are the `num_docs` and `num_queries` columns in the result
CSV, not rounded catalog values copied from a benchmark overview.

The loader preserves document IDs as strings and provides document text through
`BenchmarkLoader.stream_corpus()`. Qrels are converted to integer-valued
`ir_measures.Qrel` records. Positive judgments are those with relevance greater
than zero for Strict, completeness, and oracle diagnostics.

### 4.2 One shared Terrier index

All six methods use the same disk-backed index at:

```text
data/cache/terrier_indices/{dataset}_default/
```

The executed indexing path is:

```text
BenchmarkLoader.stream_corpus(dataset)
  -> {docno, text}
  -> pt.IterDictIndexer
  -> standard Terrier term pipeline and disk index
```

The retained harness configures:

- `meta={"docno": 512, "text": 4096}`;
- `max.term.length=512`;
- `indexing.max.memory=1073741824` bytes;
- JVM memory request of 3,072 MiB;
- streamed ingestion when an in-memory corpus iterable is not explicitly
  supplied.

The `text` MetaIndex bound is not a document-length truncation statement. The
indexing stream supplies the full loader text to Terrier; metadata storage and
indexed postings are distinct structures.

No baseline uses `EdgeRAGAnalyzer`, KStem, the Pipeline V2 vocabulary, or a
separate analyzed index. Standard Terrier processing is a shared control, so
differences inside this six-system matrix arise from the weighting model and/or
feedback stage rather than different tokenization.

### 4.3 Query sanitization

Before Terrier analysis, the harness applies this project-specific rule:

```text
replace every character outside [A-Za-z0-9 whitespace] with a space
split on whitespace
join the remaining tokens with single spaces
use "a" if no token remains
```

Terrier then applies its configured query term pipeline. This pre-sanitizer is
shared by all six systems, but it is not a Terrier or BEIR default. It removes
punctuation and non-ASCII letters, so the paper should report it and avoid
claiming raw-query identity with an external BM25 implementation.

### 4.4 Candidate depth and query chunking

The final target depth is 1,000 eligible documents per query. Queries are
evaluated in chunks of 200 to bound Python/JNI memory; candidate lists are not
intentionally truncated to 200. Retrieval and PRF remain independent across
queries, so chunking changes execution batching rather than the mathematical
definition of a system.

## 5. First- and second-pass document weighting

### 5.1 `BM25_Default`

The harness constructs:

```python
pt.terrier.Retriever(index, wmodel="BM25", num_results=...)
```

Conceptually, Terrier BM25 scores a document by summing an inverse-document-
frequency component and saturated document/query term-frequency components:

\[
\operatorname{BM25}(D,Q)=
\sum_{t\in Q}
\operatorname{IDF}(t)
\frac{(k_1+1)tf(t,D)}
{tf(t,D)+k_1\left(1-b+b\frac{|D|}{\overline{|D|}}\right)}
\frac{(k_3+1)qtf(t,Q)}{k_3+qtf(t,Q)}.
\]

No BM25 controls are set in the retained harness. The run therefore inherited
the installed Terrier defaults. Current PyTerrier documentation reports
\(k_1=1.2\) and \(b=0.75\), but the historical package and JVM versions were
not retained, so exact run-time values—including \(k_3\)—are **Not verified**
from the aggregate CSV alone. For a future rerun, serialize the effective
retriever controls rather than assuming them from current documentation.

### 5.2 `DPH`

The harness constructs the identical retrieval path with `wmodel="DPH"`.
DPH belongs to the Divergence From Randomness family and is used here as a
parameter-free lexical alternative to BM25. “Parameter-free” refers to the
document-weighting formulation; it does not mean that tokenization, indexing,
retrieval depth, or evaluation have no configuration.

The implementation delegates exact DPH scoring to Terrier. Do not copy an
equation from another library and imply numerical identity; cite the DPH/DFR
method and the Terrier implementation.

## 6. Native PRF baselines

### 6.1 Common two-pass structure

Every PRF system uses the same ranker before and after rewriting:

```text
ranker pass 1
  -> keep ten eligible feedback documents
  -> native Terrier Bo1 or RM3 rewriter
  -> same ranker pass 2
  -> keep up to 1,000 eligible final documents
```

Thus `BM25+RM3` means BM25 supplies the feedback set and scores the expanded
query; `DPH+RM3` means DPH performs both roles. There is no BM25-to-DPH or
DPH-to-BM25 mixed pipeline.

The PRF rewriters use only the first-pass ranking, its scores where required,
the shared index, and collection statistics. They never consult qrels.

### 6.2 RM3

The exact constructor is:

```python
pt.rewrite.RM3(index, fb_terms=10, fb_docs=10, fb_lambda=0.5)
```

RM3 estimates a relevance language model from \(F(Q)\), selects feedback
terms, and combines that model with the original query according to the native
rewriter. Because PyTerrier's documentation describes `fb_lambda` only as its
RM3 interpolation control and implementation conventions can differ, thesis
prose should report the numeric value without asserting which side receives
\(\lambda\) unless the historical implementation version is recovered.

### 6.3 Bo1

The exact constructor is:

```python
pt.rewrite.Bo1QueryExpansion(index, fb_terms=10, fb_docs=10)
```

Bo1 is Terrier's Bose–Einstein DFR expansion model. It ranks terms by how much
their observed occurrence in the pseudo-relevant set diverges from an expected
random occurrence under collection statistics, then rewrites the query with at
most ten feedback terms. Exact term scoring and query serialization are
delegated to Terrier's native implementation.

### 6.4 Executed non-BRIGHT pipelines

For datasets without prohibited-document lists, the definitions reduce to:

```text
BM25       = BM25[1000]
BM25+Bo1   = BM25[10] >> Bo1(10 docs, 10 terms) >> BM25[1000]
BM25+RM3   = BM25[10] >> RM3(10 docs, 10 terms, lambda 0.5) >> BM25[1000]

DPH        = DPH[1000]
DPH+Bo1    = DPH[10] >> Bo1(10 docs, 10 terms) >> DPH[1000]
DPH+RM3    = DPH[10] >> RM3(10 docs, 10 terms, lambda 0.5) >> DPH[1000]
```

Square brackets denote requested retrieval depth, not a parameter of the
weighting model.

## 7. BRIGHT prohibited-document handling

BRIGHT queries can declare prohibited source documents. Filtering only after
the second pass would allow PRF to learn terms from forbidden documents, so the
harness filters both the feedback pool and the final ranking.

### 7.1 Unexpanded systems

For BM25 and DPH:

\[
K_2=\min(1000+M_X,3000).
\]

The system retrieves \(K_2\), removes every document in \(X_q\), recomputes
contiguous zero-based ranks, and retains at most 1,000 documents per query.

### 7.2 PRF systems

For Bo1 and RM3:

\[
K_1=\min(\max(100,10+M_X),300),
\qquad
K_2=\min(1000+M_X,3000).
\]

The actual pipeline is:

```text
ranker[K1]
  >> remove prohibited documents and retain at most 10
  >> native PRF rewriter
  >> same ranker[K2]
  >> remove prohibited documents and retain at most 1000
```

The first filter prevents prohibited documents from entering the feedback set.
The second prevents them from entering evaluation or a downstream candidate
pool.

### 7.3 What the padding formulas do not guarantee

The caps at 300 and 3,000 mean the formulas do not mathematically guarantee ten
eligible feedback documents or 1,000 eligible final candidates for every
possible exclusion distribution. They provide bounded padding. A guarantee
requires a retained per-query check of post-filter counts.

The aggregate CSV contains no `feedback_docs_realized`,
`eligible_candidates_realized`, or exclusion-violation fields, and candidate
runs are not retained in the current checkout. Therefore:

- absence of prohibited documents in every feedback set: **Not verified** from
  retained artifacts, although the inspected code performs the correct
  pre-feedback filter;
- full eligible depth 1,000 for every BRIGHT query: **Not verified**;
- no prohibited document in every final ranking: **Not verified** from raw
  runs, although the code filters again and performs a defensive evaluation-
  loop check.

Do not turn the padding strategy into an empirical “100% guaranteed” claim in
the thesis without regenerating or recovering per-query evidence.

## 8. Evaluation, timing, and resource contract

### 8.1 Effectiveness metrics

The retained harness computes, per query and then macro-averages:

- primary linear-gain `ndcg_10`, `ndcg_50`, and `ndcg_100`;
- `map_100` and `mrr_10`;
- `p_10` and `p_100`;
- Recall@10, 50, 100, 200, 500, and 1,000;
- Strict@10, 50, 100, and 1,000;
- Completeness@100, 500, and 1,000;
- linear-gain `oracle_ndcg_10` over the retrieved top-1,000 pool;
- supplemental `exp_ndcg_10` with gains `{1:1, 2:3, 3:7, 4:15}`.

For clarity in prose, rename the display label `oracle_ndcg_10` to
**Oracle-nDCG@10(1000)**. The stored CSV column remains unchanged.

The canonical definitions and the distinction between primary linear gains and
supplemental exponential gains are owned by
[`docs/EVALUATION_METRICS.md`](EVALUATION_METRICS.md).

### 8.2 Retained exponential-gain anomaly

Four cells in the completed CSV require correction or explicit omission:

| Dataset | Pipeline | `ndcg_10` | stored `exp_ndcg_10` |
|---|---|---:|---:|
| `trec_covid` | `DPH_Bo1_Terrier_Default` | 0.6345 | 0.0000 |
| `trec_covid` | `DPH_RM3_Terrier_Default` | 0.6192 | 0.0000 |
| `climate_fever` | `DPH_Bo1_Terrier_Default` | 0.1925 | 0.0000 |
| `climate_fever` | `DPH_RM3_Terrier_Default` | 0.1855 | 0.0000 |

These zeros are inconsistent with the surrounding nonzero retrieval metrics
and, for binary relevance, exponential and linear gains must coincide. Treat
the four supplemental values as **Fail / invalid pending recomputation**. They
do not invalidate the primary linear metrics, but they must not enter a paper
table, macro average, or claim.

### 8.3 Latency fields

The CSV contains three different performance views:

1. `retrieval_api_p50_ms`, `p90`, `p99`, and `mean`: one-query calls through
   Python DataFrame creation, PyTerrier, JNI, JVM matching, PRF where applicable,
   and result conversion. The classical protocol samples at most 1,000 queries
   with seed 42 after up to 30 warm-up calls.
2. `batch_throughput_qps`: total evaluated queries divided by time spent in
   chunked `transformer.transform()` calls. This excludes metric calculation
   and serialization.
3. `harness_per_query_ms`: whole evaluation wall time divided by query count.
   It includes metrics, Python processing, optional Parquet work, and the
   additional single-query latency experiment. It is not retrieval latency.

The isolated API metric is a realistic Python-facing service measurement, not
pure Java matching-loop time. The JNI/DataFrame overhead is part of the number
and must not be subtracted speculatively. Compare pipelines under this same API
contract; do not compare these values directly with in-process Lucene or C++
timings that use another boundary.

### 8.4 Resource fields

- `index_disk_mb` is the shared Terrier index directory size.
- `index_build_s=0` means the evaluated process loaded a cached index; it does
  not mean the index had zero construction cost.
- `cache_load_s` is measured index-open time for that process.
- `host_ram_peak_mb` is misnamed in the retained implementation: it is a single
  process-RSS observation after evaluation, not a sampled maximum over the run.
  Report it as **post-evaluation process RSS** or omit it from peak-memory
  claims.

The experiment provides retrieval-effectiveness and serving-latency evidence,
but not a reliable historical peak-RAM measurement or complete time-to-index
comparison.

## 9. Retrospective empirical snapshot

The following descriptive macro averages are recomputed over the 25 retained
rows per pipeline. They are useful for orientation, not a substitute for the
per-dataset table or paired statistical testing.

| Paper label | Macro nDCG@10 | Macro MRR@10 | Macro Recall@1000 |
|---|---:|---:|---:|
| BM25 | 0.2602 | 0.3192 | 0.6984 |
| BM25+Bo1 (10d/10t) | 0.2567 | 0.3060 | 0.7185 |
| BM25+RM3 (10d/10t, \(\lambda=0.5\)) | 0.2509 | 0.3028 | 0.6686 |
| DPH | 0.2865 | 0.3563 | 0.7060 |
| DPH+Bo1 (10d/10t) | 0.2713 | 0.3273 | 0.7312 |
| DPH+RM3 (10d/10t, \(\lambda=0.5\)) | 0.2828 | 0.3418 | 0.6921 |

Paired dataset-level nDCG@10 wins/ties/losses against the corresponding
unexpanded ranker are:

| Comparison | Wins / ties / losses | Mean nDCG@10 change |
|---|---:|---:|
| BM25+Bo1 versus BM25 | 14 / 1 / 10 | -0.0035 |
| BM25+RM3 versus BM25 | 5 / 0 / 20 | -0.0093 |
| DPH+Bo1 versus DPH | 8 / 0 / 17 | -0.0152 |
| DPH+RM3 versus DPH | 10 / 1 / 14 | -0.0037 |

This supports a restrained interpretation:

- DPH is a strong untuned lexical control in this suite.
- Fixed PRF is collection-dependent rather than uniformly beneficial.
- Bo1 raises macro Recall@1000 for both base rankers while reducing macro
  nDCG@10, showing a candidate-coverage versus early-ranking trade-off.
- Aggregate means do not establish statistical significance or identify a
  causal corpus property.

Before publication, recompute this table with a versioned analysis script and
add paired per-query uncertainty or significance tests. Do not select a
different baseline configuration after observing the 25 test collections and
then report it as untuned.

## 10. Paper and thesis reporting contract

### 10.1 Recommended table names

Use compact display names while preserving exact IDs in an appendix or
artifact manifest:

| Display name | Exact artifact ID |
|---|---|
| BM25 | `BM25_Default` |
| BM25+Bo1 (10d/10t) | `BM25_Bo1_Terrier_Default` |
| BM25+RM3 (10d/10t, \(\lambda=.5\)) | `BM25_RM3_Terrier_Default` |
| DPH | `DPH` |
| DPH+Bo1 (10d/10t) | `DPH_Bo1_Terrier_Default` |
| DPH+RM3 (10d/10t, \(\lambda=.5\)) | `DPH_RM3_Terrier_Default` |

Do not silently rename the files or CSV keys: traceability is more important
than cosmetically fixing historical identifiers.

### 10.2 Methods paragraph template

The following text can be adapted after inserting verified software versions
and hardware details:

> We indexed every collection once using Terrier's standard term pipeline and
> evaluated two classical sparse rankers, BM25 and the parameter-free DPH
> model. For each ranker we additionally evaluated native Terrier Bo1 and RM3
> pseudo-relevance feedback. Both methods used the top 10 eligible first-pass
> documents and added at most 10 feedback terms; RM3 used an interpolation
> control of 0.5. The same weighting model was used before and after feedback.
> We retrieved up to 1,000 final candidates and evaluated queries in chunks of
> 200. For BRIGHT, prohibited source documents were removed before feedback and
> again before evaluation. We report linear-gain nDCG@10 as the primary ranking
> metric together with MRR@10 and candidate-funnel recall through depth 1,000.

Do not include “default RM3,” “default Bo1,” “official BEIR BM25 score,” or
“guaranteed 1,000 BRIGHT candidates” unless the corresponding claim is narrowed
and supported by retained evidence.

### 10.3 Interpretation rules

- Compare each PRF method first with its matched unexpanded ranker.
- Compare BM25 and DPH separately from the incremental effect of feedback.
- Report both nDCG@10 and Recall@K; a recall gain with an nDCG loss is a
  trade-off, not an unconditional improvement.
- Do not treat the best of six systems on each test dataset as one deployable
  oracle system.
- Do not average invalid `exp_ndcg_10` cells as zeros.
- Use equal-dataset macro averages only when explicitly labelled. A
  query-weighted average answers a different question and would be dominated by
  Quora, HotpotQA, and FEVER.
- Separate BEIR and BRIGHT group summaries in addition to the 25-dataset macro;
  their task construction and prohibited-document semantics differ.
- The completed fixed PRF settings are baselines, not evidence that Bo1 or RM3
  cannot perform better after legitimate development-set tuning.
- These results concern initial sparse retrieval. Recall@1000 is directly
  relevant to a downstream reranking/RAG candidate pool, while nDCG@10 remains
  important for early precision and shallow consumers.

## 11. Required work before claiming full reproducibility

The existing effectiveness runs need not be discarded, but the following
items are required before a “fully reproducible baseline suite” claim:

| Gate | Required work | Pass condition |
|---|---|---|
| A: environment | Freeze Python, PyTerrier, Terrier, Java, `ir_measures`, OS, CPU, and memory configuration | Machine-readable manifest stored with results. |
| B: runner | Restore or replace the missing sequential CLI supervisor | One command identifies datasets, pipelines, seed, chunk size, cache policy, and output directory. |
| C: effective controls | Serialize each constructed retriever and rewriter configuration | BM25 parameters and PRF controls are read from the running objects, not inferred from documentation. |
| D: analyzer parity | Persist index properties and a small analyzed query/document fixture | Default term pipeline and sanitizer behavior are reproducible. |
| E: BRIGHT exclusions | Record requested/realized feedback and final depths plus violation counts per query | Zero prohibited feedback/final documents and declared depth shortfalls. |
| F: metric repair | Recompute the four invalid exponential-gain cells from retained or regenerated runs | Correct values and provenance recorded without overwriting the old CSV. |
| G: raw runs | Stream top-1,000 runs to compressed TREC or Parquet artifacts without retaining every dataset ranking in RAM | Every aggregate score maps to a checksummed run file. |
| H: resources | Sample RSS over time and separate index build, cache load, batch retrieval, API latency, evaluation, and serialization | Labels match what timers and monitors actually measure. |
| I: statistics | Add a versioned analysis script for macro summaries and paired query-level uncertainty | Every manuscript table is regenerated from declared artifacts. |

Future corrected results must be written to a timestamped directory and a new
summary artifact. Preserve the completed CSV as historical evidence; never
silently replace or edit individual cells in place.

## 12. Completion criteria for this baseline record

For the current retrospective record:

- the six implemented pipeline definitions are documented: **Pass**;
- the 25-by-6 aggregate matrix is present and unique: **Pass**;
- primary metric meanings and top-1,000 oracle depth are documented: **Pass**;
- historical environment, runner, raw runs, and per-query BRIGHT depth evidence:
  **Not verified**;
- four supplemental exponential-gain cells: **Fail pending recomputation**.

The primary classical baseline evaluation can be cited with these limitations.
Full reproduction, strong resource claims, and the four affected supplemental
scores remain future verification work.
