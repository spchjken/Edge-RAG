# CRVE: Context-Reranked Vocabulary Expansion (1st-Stage Retrieval)

This repository contains the official implementation and evaluation platform for **CRVE** (Context-Reranked Vocabulary Expansion), a high-speed, 1st-stage lexical-semantic retriever designed for domain-specific retrieval under edge constraints.

CRVE solves the vocabulary mismatch problem in lexical retrieval (BM25, DPH) without incurring the latency, index explosion, or memory overhead of heavy neural bi-encoders or generative LLM query expanders. It couples **Corpus-Grounded Dense Vocabulary Probing** with **Uncertainty-Aware Gate 1 Candidate Selection** and **PyTerrier-Native Inverted Indexing**.

---

## ⚡ Active Workspace & Sole Active Codebase: `CRVE/`

All active development, benchmarks, and tests reside exclusively inside the **[`CRVE/`](file:///home/donghv/Projects/Edge-RAG/CRVE)** directory. Downstream experimental modules (cascade routing, listwise LLM reranking, late expansion) have been decoupled, and obsolete custom BM25 indexers have been replaced by standard PyTerrier infrastructure.

```
CRVE/
├── configs/
│   ├── crve.yaml                   # Single source of truth for CRVE hyperparameters
│   ├── hardware_profiles.yaml      # Hardware memory and device budgets
│   └── pyterrier_qe.yaml           # PyTerrier QE grid search and parameter specs
├── docs/
│   ├── ARCHITECTURE.md             # Canonical CRVE 1st-stage retrieval specification
│   ├── DATASET_PREP.md             # Dataset download & preprocessing guide
│   ├── EVALUATION_METRICS.md       # Metric definitions and parity verification
│   └── phase2_selection_under_uncertainty.md # Selection under uncertainty foundation
├── scripts/
│   ├── run_pyterrier_baselines.py  # 6 classical baselines runner across 25 datasets
│   ├── run_pyterrier_qe_baselines.py # PyTerrier QE baselines runner
│   ├── run_gate1_oracle_evaluation.py # Gate 1 selection empirical runner
│   ├── run_pool_oracle_isolation.py   # Pool oracle isolation experiment
│   ├── compile_gate1_research_tables.py # Gate 1 research table compiler
│   ├── compile_pool_oracle_tables.py   # Pool oracle table compiler
│   └── results_scripts_mapping.md  # Mapping linking result files to scripts
├── src/
│   ├── crve/
│   │   ├── indexer/
│   │   │   ├── analyzer.py         # EdgeRAGAnalyzer (KStem + WordNet overrides)
│   │   │   ├── corpus_idf_registry.py # CorpusIDFRegistry (Lucene IDF)
│   │   │   ├── corpus_vocab_builder.py# CorpusVocabBuilder (Sublinear salience pool)
│   │   │   └── dense_vocab_matrix.py  # DenseVocabMatrix (BGE-small FP16)
│   │   ├── selection/
│   │   │   ├── gate1_proposers.py  # Gate 1 candidate proposers (WQ, ABGE, PPMI, RRF)
│   │   │   ├── gate1_metrics.py    # Gate 1 evaluation metrics & telemetry
│   │   │   └── pathway_gate1_selection.md # Tier 2 Gate 1 specification
│   │   └── orchestrator.py         # CRVEOrchestrator (End-to-end 1st-stage runner)
│   ├── evaluation/
│   │   ├── baselines/
│   │   │   ├── pyterrier_harness.py# PyTerrier baseline harness & index manager
│   │   │   ├── pyterrier_qe.py     # PyTerrier QE operator & BGE sidecar
│   │   │   ├── dense_rag.py        # Dense BGE-small-en-v1.5 baseline
│   │   │   └── splade.py           # SPLADE-v3 baseline
│   │   ├── benchmark_loader.py     # BEIR / BRIGHT streaming data loader
│   │   ├── pool_generators.py      # Candidate pool generation utilities
│   │   └── metrics.py              # Parity-verified IR metrics calculation
│   └── utils/
│       └── helpers.py              # Shared utilities
└── tests/                          # Automated pytest regression suite
```

---

## 🛠️ Quick Start

### 1. Environment Setup
Create a Python virtual environment (Python 3.11+) and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Execution Rule: Always Set `PYTHONPATH=CRVE`
All scripts and tests must be executed with `PYTHONPATH=CRVE`:
```bash
# Run pytest verification suite
PYTHONPATH=CRVE .venv/bin/python3 -m pytest CRVE/tests/test_gate1_selection.py CRVE/tests/test_pool_compiler.py -v

# Run 6 classical PyTerrier baselines on SciFact
PYTHONPATH=CRVE .venv/bin/python3 -u CRVE/scripts/run_pyterrier_baselines.py --datasets scifact

# Run Gate 1 candidate selection evaluation
PYTHONPATH=CRVE .venv/bin/python3 -u CRVE/scripts/run_gate1_oracle_evaluation.py --mode dev
```

### 3. Basic CRVE Orchestrator Usage
```python
import os
import sys

# Ensure CRVE is on PYTHONPATH
sys.path.insert(0, os.path.abspath("CRVE"))

from crve.orchestrator import CRVEOrchestrator

# Sample corpus
corpus = [
    "Large language models benefit from context-reranked vocabulary expansion in retrieval.",
    "BM25 provides exact keyword matching but suffers from the vocabulary mismatch problem.",
    "Dense retrieval encodes semantic representations using bi-encoder neural architectures."
]

# Initialize CRVE 1st-stage orchestrator
orchestrator = CRVEOrchestrator(corpus=corpus)
orchestrator.build_vocabulary_and_matrix(corpus=corpus)

# Formulate expanded query for PyTerrier retrieval
query = "vocabulary gap in lexical retrieval"
print("Base query:", query)
```

---

## 🔬 Reproducibility & Governance

- **Operating Instructions**: [`AGENTS.md`](file:///home/donghv/Projects/Edge-RAG/AGENTS.md)
- **Canonical Architecture**: [`CRVE/docs/ARCHITECTURE.md`](file:///home/donghv/Projects/Edge-RAG/CRVE/docs/ARCHITECTURE.md)
- **Evaluation Metrics & Parity**: [`CRVE/docs/EVALUATION_METRICS.md`](file:///home/donghv/Projects/Edge-RAG/CRVE/docs/EVALUATION_METRICS.md)
- **Selection Under Uncertainty**: [`CRVE/docs/phase2_selection_under_uncertainty.md`](file:///home/donghv/Projects/Edge-RAG/CRVE/docs/phase2_selection_under_uncertainty.md)
- **Results-to-Scripts Mapping**: [`CRVE/scripts/results_scripts_mapping.md`](file:///home/donghv/Projects/Edge-RAG/CRVE/scripts/results_scripts_mapping.md)