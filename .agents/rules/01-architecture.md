---
trigger: always_on
---

# 🏗️ MODULE BOUNDARIES & ARCHITECTURE RULES (CRVE)

## 1. System Documentation Hierarchy
- **Canonical Reconciliation**: Each architectural decision has one canonical owner. Derivative documents may summarize or link to that owner but must not silently diverge. When Tier 0, Tier 1, or Tier 2 sources conflict, the higher tier governs; reconcile the lower-tier source and all affected references before dependent implementation proceeds.
- **Tier 0 (Current System Description — `CRVE/docs/ARCHITECTURE.md`)**: Canonical system blueprint describing the active CRVE 1st-Stage Retrieval architecture. Must read first to understand system design.
- **Tier 1 (High-Level Rules & Module Boundaries — This File)**: Defines active module boundaries, component isolation constraints, hardware caps, and configuration contracts.
- **Tier 2 (Decentralized Pathway Specs — `pathway_*.md` in Sub-modules)**: Co-located algorithm specifications (e.g., `CRVE/src/crve/selection/pathway_gate1_selection.md`). Any new retrieval variant added to `CRVE/src/crve/` MUST include a co-located `pathway_<name>.md`.

---

## 2. Active Retrieval Architecture (`CRVE/src/crve/`)
`CRVE/src/crve/` is the sole active retrieval package. It implements Context-Reranked Vocabulary Expansion for 1st-stage retrieval:

### 2.1 Indexing & Shared IDF (`CRVE/src/crve/indexer/`)
- `analyzer.py` — `EdgeRAGAnalyzer`: Krovetz stemmer with WordNet irregular suppletion overrides (*went $\to$ go*, *children $\to$ child*) and technical compound protection (*e.g.* `qwen2.5-7b`, `fp16`, `nav2_bringup`). Self-contained canonical tokenization pattern.
- `corpus_idf_registry.py` — `CorpusIDFRegistry`: Unified non-negative Lucene IDF table ($\ln(1.0 + \frac{N - n + 0.5}{n + 0.5})$) and pre-indexed compound `boundary_prefix_map` for $O(1)$ query-time bailout lookups.
- `corpus_vocab_builder.py` — `CorpusVocabBuilder`: Fast sublinear salience vocabulary extractor ($\text{IDF} \times \ln(1 + \text{DF})$) with canonical surface-form mapping (*e.g.*, stem `robot` $\to$ surface `robotics`).
- `dense_vocab_matrix.py` — `DenseVocabMatrix`: Batched GPU embedding matrix using `BAAI/bge-small-en-v1.5` on CUDA FP16 with Farthest-Point Sampling (FPS) for semantic coverage hubs ($<0.3\text{s}$ TTI).

### 2.2 Gate 1 Selection Under Uncertainty (`CRVE/src/crve/selection/`)
- `gate1_proposers.py` — Multi-channel candidate proposers:
  - `WholeQueryBGEProposer`: Global query-to-pool cosine matching ($S_{\text{WQ}}$).
  - `AnchorBGEProposer`: Fine-grained anchor-to-term specificity matching ($S_{\text{ABGE}}$).
- `PPMISidecarProposer`: Bounded, prebuilt co-occurrence evidence for the frozen Gate 1 core RRF; `LexicalPPMIProposer` computes from live PyTerrier postings as a higher-latency diagnostic or generic-orchestrator variant.
  - `RRFHybridProposer`: Reciprocal Rank Fusion ($k=60$) over top candidate lists with unique refill.
- `gate1_metrics.py` — Mathematical evaluation of selection under uncertainty: Recall@K, NDCG@K, ReferenceBOR, NearBestHit, TermRecall, TermPrecision, and transition dynamics.
- `pathway_gate1_selection.md` — Authoritative co-located Tier 2 specification for Gate 1 selection.

### 2.3 Orchestration & Configuration
- `orchestrator.py` — `CRVEOrchestrator`: End-to-end 1st-stage runner connecting Indexer $\to$ Vocab Builder $\to$ Dense Matrix $\to$ Gate 1 Proposers $\to$ PyTerrier Retrieval.
- `CRVE/configs/crve.yaml`: Generic orchestrator runtime defaults. A frozen experiment config and its `for_review/` run manifest own that experiment's tested hyperparameters and results; do not overwrite their provenance with generic defaults. `CRVE/docs/ARCHITECTURE.md` owns stage boundaries and implementation status.

---

## 3. Evaluation & Baselines (`CRVE/src/evaluation/`)
All baseline models and evaluation harnesses live in `CRVE/src/evaluation/`:

### 3.1 Baselines Namespace (`CRVE/src/evaluation/baselines/`)
Contains all 8 standardized retrieval baselines:
- **Classical PyTerrier Baselines:**
  - `pyterrier_harness.py`: Disk-backed Terrier inverted indexing and evaluation harness across 25 BEIR & BRIGHT datasets (`BM25_Default`, `BM25_RM3_Terrier_Default`, `BM25_Bo1_Terrier_Default`, `DPH`, `DPH_Bo1_Terrier_Default`, `DPH_RM3_Terrier_Default`).
  - `pyterrier_qe.py`: Sparse lexical query expansion baselines (`BGE_Vocab_QE`, `LLM_Q2E_ZS`) and `TerrierQueryAnalyzer`.
- **Neural Baselines:**
  - `dense_rag.py`: `DenseRAGBaseline` (BAAI/bge-small-en-v1.5 on CUDA FP16).
  - `splade.py`: `SPLADEBaseline` and `SparseInvertedIndex` (naver/splade-v3-distilbert).

### 3.2 Evaluation Infrastructure (`CRVE/src/evaluation/`)
- `benchmark_loader.py` — `BenchmarkLoader`: Streaming ingestion for all BEIR and BRIGHT datasets without loading multi-million raw corpora into memory.
- `pool_generators.py` — Formal candidate pool policies (Salience, Specificity, Hybrid, Stratified, CELF Coverage) for pool oracle isolation.
- `metrics.py` — Parity-verified IR metrics calculation (nDCG@K, MRR@K, R@K, P@K).

---

## 4. Execution Invariants
- **Local Virtual Environment:** Always run with `.venv/bin/python3`.
- **PYTHONPATH Invariant:** Always set `PYTHONPATH=CRVE` for all scripts and tests.
- **Hardware Profile & Budgets:** Tested under consumer edge constraints (WSL2 Linux, 15 GiB RAM ceiling, NVIDIA GPU). All code must be strictly memory-safe and respect memory caps.
- **Query Chunking:** Never truncate candidate document rankings; cap RAM and JNI overhead by chunking queries (`chunk_size=200`).
