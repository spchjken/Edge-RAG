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
4. **Subprocess Execution Isolation:** Each `(dataset, model)` evaluation executes inside a dedicated, isolated subprocess worker (`subprocess.run`) to guarantee 0% CUDA memory fragmentation and eliminate cross-model cache contamination.
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
- **Definition:** Measures ranking quality by penalizing relevant documents retrieved at lower rank positions, normalized against the Ideal Discounted Cumulative Gain (IDCG). Supports both **official BEIR/TREC graded relevance** and binary fallback.
- **Official Graded BEIR/TREC Formula (Exponential Gain):**
  When continuous or multi-level relevance judgments are available in `qrels` ($r(d) \ge 0$, e.g., NFCorpus 0–3, TREC-COVID 0–2):
  $$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{\text{rel}(d_i)} - 1}{\log_2(i + 1)}$$
  $$\text{IDCG@K} = \sum_{j=1}^{\min\left(K, |\text{Gold}^+(q)|\right)} \frac{2^{\text{rel}^*(j)} - 1}{\log_2(j + 1)}$$
  where $\text{rel}^*(j)$ is the $j$-th relevance score of all positive judgments in $\text{Gold}^+(q)$ sorted in **strictly descending order**, ensuring $0.0 \le \text{nDCG@K} \le 1.0$.
  $$\text{nDCG@K} = \frac{1}{|Q|} \sum_{q \in Q} \begin{cases} \frac{\text{DCG@K}(q)}{\text{IDCG@K}(q)} & \text{if } \text{IDCG@K}(q) > 0 \\ 0.0 & \text{otherwise} \end{cases}$$
- **Binary Fallback:**
  When only binary relevance $\text{Gold}(q) \subset \mathcal{D}$ is provided ($\text{rel} \in \{0, 1\}$), $2^1 - 1 = 1$, recovering the linear indicator formulation:
  $$\text{DCG@K} = \sum_{i=1}^{K} \frac{\mathbb{I}\left(d_i \in \text{Gold}(q)\right)}{\log_2(i + 1)}, \quad \text{IDCG@K} = \sum_{j=1}^{\min\left(K, |\text{Gold}(q)|\right)} \frac{1}{\log_2(j + 1)}$$

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
