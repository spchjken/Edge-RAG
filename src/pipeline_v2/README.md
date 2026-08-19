# Pipeline V2: High-Speed Aspect-Grouped BM25-Dense RAG (`src/pipeline_v2/`)

## Architecture Contract

Pipeline V2 is a streamlined, low-latency, 5-phase Extractive-Compression RAG pipeline optimized for ephemeral and edge deployments (<0.3s index setup TTI, <8ms expansion latency).

```
Phase 1: Indexing & Shared IDF (`indexer/`)
   └── Lucene BM25 inverted indexer + O(N) Counter + IDF Median filtering + Shared CorpusIDFRegistry

Phase 2: Aspect-Grouped Query Expansion (`expansion/`)
   └── BM25DenseAspectExtractor (Schemas 1-4: AspectInject, AspectWeighted, LocalCascade, AspectFusion)

Phase 3: BM25-Driven Cascade Router (`routing/`)
   └── 3-Way Triage (Bypass / Rerank / Discard) based on normalized BM25 score & Aspect Coverage alpha

Phase 4: Single-Pass Listwise LLM Reranker (`reranker/`)
   └── Evaluates candidates using IDF-filtered sentence snippets (~250 tokens per chunk)

Phase 5: Late Expansion & VRAM Safety (`expansion_late/`)
   └── Restores uncompressed chunk texts, enforces N_max <= 10 VRAM cap, executes final generation
```

## Module Boundaries & Rules
- `src/pipeline_v2/` is completely isolated from `src/legacy_pipeline/`.
- All hyperparameters are loaded from `configs/pipeline_v2.yaml`.
- All modules utilize the shared `CorpusIDFRegistry` instance to prevent scoring mismatches.
