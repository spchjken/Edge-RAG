---
trigger: always_on
---

# 🏗️ MODULE BOUNDARIES & ARCHITECTURE RULES (Edge-RAG)

## 1. System Documentation Hierarchy
- **Tier 0 (Current System Description — `docs/ARCHITECTURE.md`)**: Canonical system blueprint describing the active High-Speed Anchored Lexical-Semantic Retriever. Must read first to understand system design.
- **Tier 1 (High-Level Rules & Module Boundaries — This File)**: Defines active pipeline boundaries, isolation constraints, hardware caps ($N_{max}$), and configuration contracts.
- **Tier 2 (Decentralized Pathway Specs — `pathway_*.md` in Sub-modules)**: Co-located algorithm specifications (e.g., `src/pipeline_v2/expansion/pathway_bm25_dense_aspect.md`). Any new retrieval variant added to `src/pipeline_v2/` MUST include a co-located `pathway_<name>.md`.

---

## 2. Active Pipeline Architecture (`src/pipeline_v2/`)
`src/pipeline_v2/` is the primary, production-grade Edge-RAG pipeline. It implements an Anchored Lexical-Semantic Retriever with downstream extension modules:

### 2.1 Indexing & Shared IDF (`src/pipeline_v2/indexer/`)
- `corpus_idf_registry.py` — `CorpusIDFRegistry`: Unified non-negative Lucene IDF table ($\ln(1.0 + \frac{N - n + 0.5}{n + 0.5})$) shared across all modules for zero-lag initialization.
- `corpus_vocab_builder.py` — `CorpusVocabBuilder`: Fast sublinear salience vocabulary extractor ($\text{IDF} \times \ln(1 + \text{DF})$) sampling up to 1,000 document chunks for bigrams.
- `dense_vocab_matrix.py` — `DenseVocabMatrix`: Batched GPU embedding matrix using `BAAI/bge-small-en-v1.5` on CUDA FP16 ($<0.3\text{s}$ TTI).
- `bm25_lucene_indexer.py` — `BM25LuceneIndexer`: Inverted posting list retrieval engine wrapping `LuceneBM25Baseline` ($k_1=1.2, b=0.75$).

### 2.2 Query Expansion (`src/pipeline_v2/expansion/`)
- `bm25_dense_aspect_extractor.py` — `BM25DenseAspectExtractor`: Maps natural language queries to grounded aspect groups with weighted keywords and compiles the sparse term weight dictionary $\vec{w}_Q$ for direct vectorized BM25 retrieval.
  - **Regex Heuristic Extraction:** Acronyms (`\b[A-Z]{2,}\b`), hyphenated terms, and exact quotes with entity validation gate ($\text{IDF} \ge 1.0$).
  - **Anchor Selection & Centrality:** Non-entity words ranked by IDF or Query Centrality with stem/semantic deduplication.
  - **Dual BGE Probing:** $\text{Dual\_Sim}(A_k, v) = \beta \cdot \text{CosSim}(A_k, v) + (1 - \beta) \cdot \text{CosSim}(Q_{\text{full}}, v)$ (threshold $\tau_{\text{sim}} = 0.55, \beta = 0.65$) with synonym weight capped at $1.0$.
  - **Active Schemas:** `BM25Dense_AspectInject` (Schema 1), `BM25Dense_FixedRepDynamicCapacity` (Schema 5a), `BM25Dense_DynamicAspectInject` (Schema 5b), `BM25Dense_CentralityFixedRep` (Schema 6a), `BM25Dense_CentralityDynamicInject` (Schema 6b), `BM25Dense_AspectWeighted`, `BM25Dense_AspectFusion`.

### 2.3 Downstream Extensions (Future Work)
- **Cascade Routing (`src/pipeline_v2/routing/`):** `BM25CascadeRouter` — 3-way triage (Bypass / Rerank / Discard) based on normalized BM25 score and Aspect Coverage $\alpha$.
- **Listwise LLM Reranker (`src/pipeline_v2/reranker/`):** `ListwiseLLMRerankerV2` — Single-pass listwise LLM evaluation using ~250-token sentence snippets extracted around anchor hits.
- **Late Context Expansion (`src/pipeline_v2/expansion_late/`):** `LateExpansionV2` — Restores full uncompressed chunk text and enforces hardware VRAM safety budget ($N_{\text{max}} \le 10$).

### 2.4 Orchestration & Configuration
- `orchestrator.py` — `PipelineV2Orchestrator`: End-to-end runner orchestrating Indexer $\to$ Expansion $\to$ Routing $\to$ Reranker $\to$ Late Expansion.
- `configs/pipeline_v2.yaml`: Authoritative single source of truth for all Pipeline V2 hyperparameters.

### 2.5 Legacy Pipeline V1 (`src/legacy_pipeline/`)
- Legacy experimental 5-stage pipeline (`query_expansion/`, `lexical_search/`, `routing/`, `llm_reranker/`, `late_expansion/`).
- Deprecated and maintained for historical baseline comparisons. Isolated from `src/pipeline_v2/`.

---

## 3. Baselines (`src/baselines/`)
- Fully isolated. Each baseline is self-contained.
- May import `torch`, `transformers`, `FlagEmbedding`, `rank_bm25`.
- MUST NOT import from `src/pipeline_v2/` or `src/legacy_pipeline/`.

---

## 4. Evaluation & Testing (`src/evaluation/`, `scripts/`, `tests/`)
- Orchestrates evaluations comparing Pipeline V2 against baselines (BM25, Dense BGE, SPLADE-v3).
- Primary evaluation scripts:
  - `scripts/run_v2_ablation_sweep.py` / `scripts/run_v7_ablation_sweep.py` — Automated multi-corpus evaluation sweeps.
  - `src/evaluation/benchmark_runner.py` — Baseline vs Edge-RAG orchestrator.
  - `src/evaluation/metrics.py` — Retrieval and generation metric evaluators.

---

## 5. Utils (`src/utils/`)
- Shared utilities: `llm_client.py` (OpenAI-compatible wrapper), `helpers.py`.
- `llm_client.py` auto-detects backend (Ollama vs llama-cpp) from `configs/models.yaml`.
- No domain-specific logic in `utils/`.

---

## 6. Configs (`configs/`)
- All hyperparameters in YAML. No hardcoded magic numbers in source code.
- `configs/pipeline_v2.yaml`: Pipeline V2 expansion, indexing, routing, and VRAM parameters.
- `configs/models.yaml`: Per-model backend, endpoint, tags, context windows.
- `configs/thresholds.yaml`: Legacy V1 threshold definitions.
- `configs/hardware_profiles.yaml`: VRAM fractions for simulated hardware profiles.

---

## 7. Vendor (`vendor/`)
- Custom-built external tools (llama.cpp for ZAYA1-8B).
- Built via `scripts/setup_zaya.sh`. Never committed to git.
- Listed in `.gitignore`.

---

## 8. Benchmark Creation (`scripts/benchmark_creation/`)
5-step synthetic dataset generation pipeline:
1. `step1_chunking.py` — Hierarchical parent block + child chunk parsing
2. `step2_seed_generation.py` — LLM-generated queries + golden answers per parent block
3. `step3_query_paraphrasing.py` — Vocabulary gap injection (lexical bias)
4. `step4_global_recall.py` — Offline hybrid retrieval for false negative detection
5. `step5_oracle_filtering.py` — LLM-as-a-Judge binary relevance annotation

---

## 9. Data (`data/`)
- `raw/` — Source documents (arXiv papers, enterprise datasets, etc.)
- `processed/` — Chunked documents, generated query datasets
- `cache/` — Embedding caches, IDF dictionaries
- `models/` — Downloaded model files (GGUF, BGE, etc.)
- `tmp_test_ai/` — Staging area for benchmark test data

---

## 10. Ignored Paths & Search Guidance
- `tests/` and `scripts/` are listed in `.gitignore`.
- **Search Rule**: When searching for test scripts or evaluation harnesses, use standard path searches or `grep_search`/`find` directly rather than relying solely on git-index tools.