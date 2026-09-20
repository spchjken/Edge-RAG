# CRVE: Context-Reranked Vocabulary Expansion (`src/crve/`)

## Architecture Contract

CRVE is a high-speed, 1st-stage lexical-semantic retriever designed for domain-specific retrieval under edge constraints. It couples corpus-grounded dense vocabulary probing with uncertainty-aware candidate selection (Gate 1) and PyTerrier-native inverted indexing.

```
1. Indexer (`indexer/`)
   ├── analyzer.py            # EdgeRAGAnalyzer (Krovetz stemmer, WordNet overrides, compound protection)
   ├── corpus_idf_registry.py # CorpusIDFRegistry (Lucene non-negative IDF)
   ├── corpus_vocab_builder.py# CorpusVocabBuilder (Sublinear salience scoring IDF * ln(1 + DF))
   └── dense_vocab_matrix.py  # DenseVocabMatrix (BGE-small-en-v1.5 CUDA FP16 embedding matrix)

2. Gate 1 Selection Under Uncertainty (`selection/`)
   ├── gate1_proposers.py     # Proposal channels: WholeQueryBGE, AnchorBGE, LexicalPPMI, RRFHybrid
   ├── gate1_metrics.py       # Telemetry: Recall@K, NDCG@K, Fidelity, Candidate Transitions
   └── pathway_gate1_selection.md # Tier 2 Gate 1 specification

3. Orchestrator (`orchestrator.py`)
   └── CRVEOrchestrator       # End-to-end 1st-stage retrieval coordinator
```

## Module Boundaries & Rules
- `src/crve/` focuses strictly on 1st-stage retrieval. Downstream E2E modules (routing, reranking, late expansion) are decoupled and removed.
- All hyperparameters are loaded from `configs/crve.yaml`.
- All indexing and evaluation use PyTerrier as the primary benchmarking platform (`evaluation/baselines/`).
