# Edge-RAG Evaluation Metrics & Benchmarking Protocol

This document defines the formal evaluation metrics, per-query telemetry signals, and benchmarking protocols used to evaluate the **Edge-RAG Retriever** (`src/pipeline_v2/`) against baseline models (Lucene BM25, Dense BGE, SPLADE-v3). All metrics adhere to the reproducibility standards defined in `.agents/rules/02-reproducibility.md`.

---

## 1. Overview & Evaluation Protocols

The evaluation framework evaluates retrieval systems under the **ephemeral edge constraint** (zero offline indexing setup, novel document ingestion at runtime, and strict consumer hardware budgets).

### Measurement & Reproducibility Protocol
1. **Deterministic Execution:** Every benchmark run accepts `--seed` and locks Python `random`, `numpy`, and `torch` RNG seeds.
2. **GPU Synchronization:** Wall-clock timers for GPU operations (e.g., dense matrix embedding and probing) MUST call `torch.cuda.synchronize()` immediately prior to starting and stopping the timer.
3. **Memory Isolation:** 
   - `torch.cuda.reset_peak_memory_stats()` is called at the beginning of each run.
   - `torch.cuda.empty_cache()` is called between benchmark iterations.
   - Host RAM is monitored via `psutil.Process().memory_info().rss`.
4. **Subprocess Execution Isolation:** Each `(dataset, model)` evaluation should execute inside a dedicated, isolated subprocess worker (`subprocess.run`) to prevent cross-model process-state accumulation. The supervisor used for the completed classical PyTerrier artifact is not retained in the current checkout, so historical compliance is **Not verified**; process isolation also cannot literally guarantee zero CUDA fragmentation.
5. **Direct Raw Streaming (`BenchmarkLoader`):** Documents and queries are streamed directly from official raw archives (`corpus.jsonl`, `qrels/test.tsv`, parquets) using [`BenchmarkLoader`](file:///home/donghv/Projects/Edge-RAG/src/evaluation/benchmark_loader.py), ensuring uniform text concatenation (`f"{title} {text}".strip()`) and zero loss of graded relevance.
6. **Warm-Boot Assumption:** Model loading time from disk into GPU memory is excluded from per-query retrieval latency.

---

## 2. Docs & Chunks Level Retrieval Metrics

Evaluated at standard retrieval cutoffs: $K \in \{10, 20, 30, 50\}$.

### 2.1 `Strict@K` (Success Rate / Hit Rate@K)
- **Definition:** The percentage of queries for which at least one ground-truth relevant document chunk appears within the Top-$K$ retrieved candidates.
- **Formula:**
  $$\text{Strict@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left(\text{rank}_{\text{first\_gold}}(q) \le K\right) \times 100\%$$
  where $\mathbb{I}(\cdot)$ is the indicator function and $\text{rank}_{\text{first\_gold}}(q)$ is the 1-indexed rank position of the first ground-truth chunk.

---

### 2.2 `ChunkRec@K` (Chunk-Level Recall@K)
- **Definition:** The proportion of all relevant ground-truth document chunks in the corpus that are successfully retrieved within the Top-$K$ candidate list.
- **Formula:**
  $$\text{ChunkRec@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\text{Retrieved@K}(q) \cap \text{Gold}(q)|}{|\text{Gold}(q)|} \times 100\%$$
  where $\text{Gold}(q)$ is the complete set of ground-truth chunks for query $q$, and $\text{Retrieved@K}(q)$ is the set of top $K$ retrieved chunks.

---

### 2.3 `ChunkPrec@K` (Chunk-Level Precision@K)
- **Definition:** The proportion of chunks in the retrieved Top-$K$ candidate list that are ground-truth relevant.
- **Formula:**
  $$\text{ChunkPrec@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\text{Retrieved@K}(q) \cap \text{Gold}(q)|}{K} \times 100\%$$

---

### 2.4 `MRR@K` (Mean Reciprocal Rank@K)
- **Definition:** The average reciprocal rank of the first relevant document chunk across all queries in the evaluation set, cut off at rank $K$.
- **Formula:**
  $$\text{MRR@K} = \frac{1}{|Q|} \sum_{q \in Q} \begin{cases} \frac{1}{\text{rank}_{\text{first\_gold}}(q)} & \text{if } \text{rank}_{\text{first\_gold}}(q) \le K \\ 0 & \text{otherwise} \end{cases}$$

---

### 2.5 `DocRec@K` (Document-Level Recall@K)
- **Definition:** For multi-document benchmark corpora (e.g., `enterpriserag_doc_level`, `liverag_doc_level`), the proportion of queries where the full source document containing the answer is successfully retrieved in the Top-$K$ candidates.
- **Formula:**
  $$\text{DocRec@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\text{RetrievedDocs@K}(q) \cap \text{GoldDocs}(q)|}{|\text{GoldDocs}(q)|} \times 100\%$$

---

### 2.6 `first_gold_rank` (First Gold Rank Position)
- **Definition:** The exact 1-indexed position of the highest-ranked relevant chunk in the retrieved list (e.g., `first_gold_rank = 1` for perfect Top-1 retrieval; `first_gold_rank = 18` indicates query drift).
- **Usage:** Logged in per-query traces to identify failure cases, track rank degradation, and diagnose multi-synonym score hijacking.

---

### 2.7 `nDCG@K` (Normalized Discounted Cumulative Gain@K)
Measures ranking quality by penalizing relevant documents retrieved at lower rank positions, normalized against the Ideal Discounted Cumulative Gain (IDCG). 

Edge-RAG reports both **Official BEIR Linear nDCG** (primary benchmark metric) and **Supplemental Exponential nDCG** (Table 2 comparison metric):

1. **Official BEIR Linear Gain Formulation (`ndcg_10`, `ndcg_50`, `ndcg_100`):**
   Standard BEIR and TREC evaluation (implemented via `ir_measures.nDCG @ K`):
   $$\text{DCG@K} = \sum_{i=1}^{K} \frac{\text{rel}(d_i)}{\log_2(i + 1)}, \quad \text{IDCG@K} = \sum_{j=1}^{\min\left(K, |\text{Gold}^+(q)|\right)} \frac{\text{rel}^*(j)}{\log_2(j + 1)}$$
   where $\text{rel}^*(j)$ represents all positive relevance scores in $\text{Gold}^+(q)$ sorted in descending order.

2. **Supplemental Exponential Gain Formulation (`exp_ndcg_10`):**
   Evaluated using `ir_measures.nDCG(gains=EXP_GAINS) @ 10` with `EXP_GAINS = {1: 1, 2: 3, 3: 7, 4: 15}`:
   $$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{\text{rel}(d_i)} - 1}{\log_2(i + 1)}, \quad \text{IDCG@K} = \sum_{j=1}^{\min\left(K, |\text{Gold}^+(q)|\right)} \frac{2^{\text{rel}^*(j)} - 1}{\log_2(j + 1)}$$
   *Note:* The exponential gain mapping ($2^{\text{rel}} - 1$) is a supplemental convention adopted in literature comparisons (e.g. SPLADE-v3 arXiv:2403.06789 Table 2 and standard TREC Web tracks) to heavily reward high graded relevance. It is **not** official BEIR parity (official BEIR strictly prescribes linear relevance gain $\text{rel}$).

3. **Binary Relevance Invariance:**
   When judgments are binary ($\text{rel} \in \{0, 1\}$), $2^1 - 1 = 1$, and both linear and exponential formulations yield mathematically identical values.

---

### 2.8 Candidate-Funnel Ceiling Diagnostics & Retrieval Ceilings
To evaluate candidate quality prior to reranking or downstream generation, Edge-RAG computes multi-depth funnel diagnostics across candidate depths $K \in \{10, 50, 100, 200, 500, 1000\}$:

1. **`Recall@K` (`recall_10` through `recall_1000`):**
   Proportion of all gold relevant documents retrieved in the top-$K$:
   $$\text{Recall@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\text{Retrieved@K}(q) \cap \text{Gold}(q)|}{|\text{Gold}(q)|}$$

2. **`Completeness@K` (`completeness_100`, `completeness_500`, `completeness_1000`):**
   The percentage of queries where **100% of all relevant documents** are contained within the top-$K$ candidates:
   $$\text{Completeness@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left(\text{Recall@K}(q) = 1.0\right) \times 100\%$$
   *Diagnostic value:* Measures whether downstream listwise rerankers or late context compressors have the full evidence set available.

3. **`Strict@K` (`strict_10`, `strict_50`, `strict_100`, `strict_1000`):**
   Percentage of queries with at least one relevant document retrieved in top-$K$:
   $$\text{Strict@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left(|\text{Retrieved@K}(q) \cap \text{Gold}(q)| \ge 1\right) \times 100\%$$

4. **`Oracle-nDCG@10` (`oracle_ndcg_10`):**
   The theoretical maximum nDCG@10 achievable if an ideal oracle reranker re-sorted the top-$1,000$ retrieved candidates to place all retrieved ground-truth documents at ranks $1 \dots \min(10, |\text{Gold}|)$:
   $$\text{Oracle-nDCG@10}(q) = \frac{\text{IDCG@10}\left(\text{Retrieved@1000}(q) \cap \text{Gold}(q)\right)}{\text{IDCG@10}\left(\text{Gold}(q)\right)}$$

---

### 2.9 Standard IR Measures Engine & Query Chunking Invariance
Baseline retrieval metrics are computed using `ir_measures`:
```python
PRIMARY_MEASURES = [
    nDCG @ 10,
    nDCG @ 50,
    nDCG @ 100,
    AP @ 100,    # MAP@100
    RR @ 10,     # MRR@10
    R @ 10,
    R @ 50,
    R @ 100,
    R @ 200,
    R @ 500,
    R @ 1000,
    P @ 10,
    P @ 100,
    nDCG(gains=EXP_GAINS) @ 10,  # Supplemental exp_ndcg_10
]
```

#### Query Chunking Invariance Property
Because Information Retrieval scoring is statistically independent across queries (all document scores and PRF term feedback models are conditioned exclusively on query $q$ and static corpus index statistics), evaluating queries in bounded chunks ($Q = Q_1 \cup Q_2 \cup \dots$, `chunk_size=200`) and aggregating metric sums produces mathematically and numerically identical results to evaluating all queries in a single unchunked batch:
$$\text{Metric}(Q) \equiv \frac{1}{|Q|} \sum_{c} \sum_{q \in Q_c} \text{Metric}(q)$$
Verified with zero floating-point drift ($\le 10^{-12}$).

---

### 2.10 BRIGHT Pre/Post PRF Exclusion Dynamics & Safety Clamping
In the BRIGHT reasoning benchmark, each query specifies a set of excluded document IDs (`excluded_doc_ids`, up to 11,206 documents in `theoremqa_questions`) representing source documents that must be disqualified to prevent label leakage and trivial keyword matching:

1. **Pass 1 Pre-PRF Feedback Protection:**
   To prevent Pseudo-Relevance Feedback (RM3 / Bo1) from extracting expansion terms from forbidden documents, Pass 1 requests depth:
   $$K_1 = \min\left(\max(100, 10 + \text{max\_ex}), 300\right)$$
   Excluded documents are filtered out immediately, and the remaining ranking is sliced to `.head(10)` before passing to `pt.rewrite.RM3` or `pt.rewrite.Bo1QueryExpansion`.

2. **Pass 2 Candidate Funnel Depth Clamping:**
   To ensure that the candidate ranking preserves full depth ($K = 1,000$) after post-filtering:
   $$K_2 = \min(1000 + \text{max\_ex}, 3000)$$
   Post-exclusion filtering removes disqualified documents and slices to `.head(1000)`. The bounded padding is intended to preserve 1,000 eligible candidates. However, the current aggregate result artifact does not retain per-query requested/realized depths or the raw candidate runs needed to verify a universal guarantee. Treat full-depth preservation as **Not verified** until those traces are regenerated or recovered; see [`docs/pyterrier_classical_baselines_plan.md`](pyterrier_classical_baselines_plan.md).

---

## 3. Query Level Expansion & Telemetry Metrics

### 3.1 Aggregate Query & Expansion Metrics (Corpus Sweep Level)
Aggregated across all queries in a corpus sweep (reported in summary CSVs):

| Metric | Symbol | Description |
| :--- | :---: | :--- |
| **`Avg_Anchors`** | $\bar{N}_A$ | Mean count of extracted aspect anchors per query after heuristic filtering and IDF/centrality ranking. |
| **`Avg_Cands_Tau`** | $\bar{N}_{\ge \tau}$ | Mean number of vocabulary terms in $\mathcal{V}_{\text{clean}}$ satisfying $\text{Dual\_Sim}(A_k, v) \ge \tau_{\text{sim}}$ across all anchors in a query. |
| **`Avg_Synonyms`** | $\bar{N}_{\text{syn}}$ | Mean count of dense synonyms selected and injected into the augmented token query $Q_{\text{aug}}$. |
| **`Starvation_Rate_pct`** | $\%$ | Percentage of aspect anchors that failed to find at least $C_{\text{exp}}$ candidates meeting $\tau_{\text{sim}}$: $\frac{\text{Starved Anchors}}{\text{Total Anchors}} \times 100\%$. |
| **`Avg_Qaug_Len`** | $|Q_{\text{aug}}|$ | Mean length of the augmented token query $Q_{\text{aug}}$ in tokens after integer token repetition weighting. |
| **`Avg_R_Anchor`** | $\bar{R}_{\text{anchor}}$ | Mean token repetition multiplier assigned to aspect anchors across queries. |

---

### 3.2 Atomic Query Trace Telemetry (Per-Query Trace Level)
Logged in JSON trace files (`trace_*.json`) for individual query diagnostics:

```json
{
  "query_id": "q_8968b27d",
  "raw_question": "How many test tasks are included in the EHR-Complex benchmark?",
  "aspects": [
    {
      "aspect_id": "asp_0_EHR",
      "anchor_term": "EHR",
      "is_heuristic_entity": true,
      "anchor_idf": 5.652,
      "repetition": 5,
      "capacity_cap": 4,
      "total_candidates_above_tau": 10,
      "candidates_above_tau": [
        {
          "term": "erbb2 esr1",
          "final_weight": 0.543,
          "similarity": 0.587,
          "idf": 7.933
        },
        {
          "term": "metadata",
          "final_weight": 0.469,
          "similarity": 0.645,
          "idf": 4.248
        }
      ],
      "injected_synonyms": ["erbb2 esr1", "metadata"]
    }
  ],
  "augmented_token_list": ["EHR", "EHR", "EHR", "EHR", "EHR", "erbb2", "esr1", "metadata"],
  "ground_truth_chunk_ids": ["EHR_Complex_block2_chunk1"],
  "retrieved_top10_chunk_ids": ["EHR_Complex_block1_chunk0", "EHR_Complex_block2_chunk1"],
  "metrics": {
    "strict_hit@10": true,
    "chunk_recall@10": 1.0,
    "precision@10": 0.1,
    "first_gold_rank": 2,
    "latency_ms": 14.85
  }
}
```

---

## 4. System Performance & Resource Metrics

### 4.1 Average Retrieval Latency (`Avg_Latency_ms`)
- **Definition:** The total wall-clock retrieval latency per query in milliseconds on CPU:
  $$\text{Latency} = t_{\text{anchor\_extract}} + t_{\text{dual\_sim\_probe}} + t_{\text{qaug\_build}} + t_{\text{lucene\_bm25\_retrieve}}$$
- **Protocol:** Synchronized across CPU/GPU boundaries and reported as the arithmetic mean across the test corpus.

---

### 4.2 Time-to-Index (`TTI` / Setup Latency)
- **Definition:** The total setup wall-clock time (in seconds) required to ingest a novel, unindexed document corpus at runtime before the first query can be served.
- **Includes:**
  1. Corpus chunking and Lucene BM25 inverted index creation (`BM25LuceneIndexer`).
  2. Document frequency extraction and non-negative Lucene IDF table computation (`CorpusIDFRegistry`).
  3. Sublinear salience candidate vocabulary extraction pool (`CorpusVocabBuilder`).
  4. 1-pass GPU batch embedding of vocabulary matrix in CUDA FP16 (`DenseVocabMatrix`).
- **Target:** $<0.3\text{s}$ on standard corpora ($<17,000$ chunks).

---

### 4.3 Peak VRAM Consumption (GPU Memory)
- **Definition:** The maximum GPU memory allocated during indexing and dense vocabulary probing.
- **Formula:**
  $$\text{Peak VRAM} = \max_{t} \text{AllocatedMemory}_{\text{CUDA}}(t)$$
- **Measurement Protocol:**
  - Call `torch.cuda.reset_peak_memory_stats()` at initialization.
  - Call `torch.cuda.max_memory_allocated() / (1024 ** 3)` at completion.
  - Reported as absolute peak in gigabytes (GB).
- **Empirical Baseline Values Across 10 Core Benchmarks:**
  - **Edge-RAG V7:** $\mathbf{0.38\text{ GB}}$ peak VRAM (probing 2,500 vocabulary hubs on CUDA FP16).
  - **Dense BGE-Small:** $\mathbf{1.23\text{ GB}}$ peak VRAM (document and query FAISS vectors).
  - **SPLADE-v3 (DistilBERT):** $\mathbf{11.46\text{ GB}}$ peak VRAM (batched transformer masked LM forward passes).
  - **Lucene BM25 (Standard & Analyzed):** $\mathbf{0.00\text{ GB}}$ (pure CPU indexing).

---

### 4.4 Host RAM Consumption (System Memory)
- **Definition:** Peak resident set size (RSS) of host physical memory utilized by Python process, inverted posting lists, and IDF dictionaries.
- **Measurement Protocol:**
  - Monitored via `psutil.Process().memory_info().rss / (1024 ** 2)`.
  - Reported in megabytes (MB) or gigabytes (GB).

---

### 4.5 PyTerrier Retrieval API Latency & Throughput Diagnostics
Evaluated by `src/evaluation/pyterrier_harness.py` under two distinct operational regimes:

1. **Isolated Single-Query API Latency (`retrieval_api_p50_ms`, `p90`, `p99`, `mean`):**
   - Measures pure single-query interactive service response time:
     `transformer.transform(pd.DataFrame([{"qid": str(q["query_id"]), "query": q["question"]}]))`
   - Samples 50 queries per dataset (with deterministic seed) after a 5-query warmup.
   - Accurately captures interactive tail latency (P90 and P99) under edge-serving conditions.

2. **Batch Retrieval Throughput (`batch_throughput_qps`):**
   - Total queries evaluated divided by pure pipeline execution time across bounded query chunks (`chunk_size=200`):
     $$\text{Throughput} = \frac{|Q|}{t_{\text{total\_retrieval\_s}}}$$
   - Reports queries served per second (QPS).

3. **End-to-End Harness Overhead (`harness_per_query_ms`):**
   - Total wall-clock time per query including chunk batching, ir_measures accumulation, diagnostic filtering, and Parquet serialization.

4. **Resource Footprint Diagnostics:**
   - `index_disk_mb`: Total disk storage size of the cached Terrier index.
   - `host_ram_peak_mb`: Historical field name. In the retained PyTerrier harness this is one process-RSS observation taken after evaluation, not a sampled maximum; report it as post-evaluation RSS and do not use it as a peak-memory claim.
   - `index_build_s`: Wall-clock seconds to build index from streamed raw corpus.
   - `cache_load_s`: Wall-clock seconds to load cached index from disk via `pt.IndexFactory.of()`.

---

## 5. Downstream Generation Metrics (Future Extensions)

Maintained for full end-to-end RAG pipeline evaluations (Cascade Routing $\to$ LLM Reranking $\to$ Late Expansion $\to$ Final Generation):

### 5.1 Time-to-First-Token (TTFT)
- **Definition:** Total wall-clock time from query submission to the moment the generative LLM yields its first output token.
- **Includes:** Retrieval latency, cascade routing triage, listwise LLM snippet evaluation, uncompressed context restoration, and prompt prefill.

### 5.2 Context Compression Ratio ($C_r$)
- **Definition:** Measure of how effectively the pipeline filters the raw input document down to the final generation context:
  $$C_r = \frac{|D_{\text{initial tokens}}|}{|D_{\text{generation tokens}}|}$$
  where $|D_{\text{generation tokens}}| \le N_{\text{max}} \times \text{chunk\_size}$.

### 5.3 Answer Faithfulness (LLM-as-a-Judge)
- **Definition:** Normalized factual correctness score $[0, 1]$ grading whether the LLM-generated answer is factually substantiated by the retrieved context without hallucination.
