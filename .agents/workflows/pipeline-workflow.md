---
description: Edge-RAG pipeline implementation workflow.
---

# Edge-RAG Pipeline Workflow

## Pre-Work
1. Read all rules in `.agents/rules/`.
2. Read `docs/ARCHITECTURE.md` (canonical Edge-RAG V2 architecture).
3. Read the relevant `pathway_*.md` for the module being modified (`src/pipeline_v2/expansion/pathway_*.md`).
4. Read `configs/pipeline_v2.yaml` for authoritative active hyperparameters.

## Module Map

### 1. Active Pipeline V2 (`src/pipeline_v2/`) — Primary
Streamlined, low-latency Anchored Lexical-Semantic Retriever:
- **`indexer/`** — `BM25LuceneIndexer`, `CorpusVocabBuilder` (sublinear salience), `DenseVocabMatrix` (CUDA FP16 batch embedding), and shared `CorpusIDFRegistry`.
- **`expansion/`** — `BM25DenseAspectExtractor` (Schemas 1, 5a, 5b, 6a, 6b, AspectWeighted, AspectFusion), regex heuristics, Dual BGE probing, and token repetition $Q_{\text{aug}}$.
- **`routing/`** — `BM25CascadeRouter` (3-way triage via normalized BM25 score & aspect coverage $\alpha$) [Future Extension].
- **`reranker/`** — `ListwiseLLMRerankerV2` (IDF-filtered sentence snippet extraction + single-pass listwise LLM) [Future Extension].
- **`expansion_late/`** — `LateExpansionV2` (uncompressed text restoration & $N_{\text{max}} \le 10$ VRAM budget) [Future Extension].
- **`orchestrator.py`** — `PipelineV2Orchestrator` end-to-end runner.
- **Config:** `configs/pipeline_v2.yaml`.

### 2. Legacy Pipeline V1 (`src/legacy_pipeline/`) — Deprecated
Maintained for historical baseline comparisons and isolated from `src/pipeline_v2/`:
- `query_expansion/`, `lexical_search/`, `routing/`, `llm_reranker/`, `late_expansion/`.

## Code Standards
- Read all hyperparameters from `configs/pipeline_v2.yaml`, never hardcode magic numbers.
- Type hints on all public functions.
- Docstrings referencing design equations.
- Each sub-module MUST have a co-located `pathway_*.md` design document.

## Post-Work
1. Update `docs/ARCHITECTURE.md` if architecture changed.
2. Update the relevant `pathway_*.md` if module behavior changed.
