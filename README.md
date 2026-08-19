# Edge-RAG: High-Speed Anchored Lexical-Semantic Retriever for Edge Devices

This repository contains the implementation and evaluation infrastructure for **Edge-RAG**, a high-speed, near-zero VRAM Anchored Lexical-Semantic Retriever designed for resource-constrained edge computing environments.

Edge-RAG resolves the "ephemeral constraint"—processing novel, unindexed documents at runtime without pre-computed vector databases—by coupling **Corpus-Grounded Dense Vocabulary Probing** with **Lucene BM25 Inverted Posting List Traversal**. It achieves high recall and exact-match precision in **$<15\text{ms}$ CPU retrieval latency**, **$<0.3\text{s}$ index setup**, and **$0.09\text{ GB}$ VRAM**.

---

## 🚀 Key Features

- **High-Speed Anchored Lexical-Semantic Retrieval (`src/pipeline_v2/`)**: Maps user queries into grounded aspect groups using heuristic extraction and IDF/centrality anchor ranking, probed against corpus vocabulary via Dual BGE similarity.
- **Zero-Lag Indexing & Shared IDF**: Non-negative Lucene IDF registry ($\ln(1 + \frac{N - n + 0.5}{n + 0.5})$) shared across indexing, vocabulary extraction, and query expansion for zero initialization lag.
- **Sublinear Salience Candidate Pool**: Rapidly extracts the top 1,000 domain unigrams and bigrams from raw text ($\text{IDF} \times \ln(1 + \text{DF})$) in $<0.05\text{s}$.
- **Sub-15ms CPU Posting List Scoring**: Standard Lucene BM25 scoring ($k_1=1.2, b=0.75$) evaluated over augmented token queries ($Q_{\text{aug}}$) using token repetition weighting.
- **Near-Zero VRAM Consumption**: Uses a lightweight local embedding matrix (BGE-Small in CUDA FP16, $0.09\text{ GB}$ VRAM) for 1-pass batch matrix probing, completely eliminating generative query expansion overhead.

---

## 📁 Repository Structure

```
Edge-RAG/
├── docs/                             # Architecture blueprint & theoretical foundations
│   ├── ARCHITECTURE.md               # Canonical Edge-RAG V2 Retriever blueprint
│   ├── EVALUATION_METRICS.md         # Metric definitions & measurement protocols
│   └── DATASET_PREP.md               # Dataset download & preprocessing guide
├── configs/                          # Authoritative configuration files
│   ├── pipeline_v2.yaml              # Pipeline V2 expansion & indexer hyperparameters
│   ├── models.yaml                   # Model endpoints & context configurations
│   └── hardware_profiles.yaml        # Simulated VRAM constraints
├── src/
│   ├── pipeline_v2/                  # Production Edge-RAG V2 Pipeline
│   │   ├── indexer/                  # Lucene inverted indexer, Vocab Builder, Dense Matrix, IDF Registry
│   │   ├── expansion/                # BM25DenseAspectExtractor & active expansion schemas
│   │   ├── routing/                  # Cascade Router (Downstream extension)
│   │   ├── reranker/                 # Listwise LLM Reranker (Downstream extension)
│   │   ├── expansion_late/           # Late Expansion & VRAM safety (Downstream extension)
│   │   └── orchestrator.py           # End-to-end PipelineV2Orchestrator runner
│   ├── legacy_pipeline/              # Deprecated Legacy V1 5-stage pipeline (baseline comparison)
│   ├── baselines/                    # Isolated baselines (Lucene BM25, Dense BGE, SPLADE-v3)
│   ├── evaluation/                   # Metrics evaluators and benchmark runners
│   └── utils/                        # OpenAI-compatible API client & helpers
├── scripts/                          # Evaluation sweeps, ablation benchmarks, and dataset converters
├── tests/                            # Pytest test suite
├── data/                             # Raw and processed benchmark datasets
└── results/                          # Benchmark outputs and sweep telemetry logs
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

### 2. Basic Pipeline V2 Usage
```python
from src.pipeline_v2.orchestrator import PipelineV2Orchestrator

# Sample ephemeral document corpus
corpus = [
    "Large Language Models (LLMs) suffer from Key-Value (KV) cache memory overflow during listwise retrieval.",
    "Edge-RAG combines fast Lucene BM25 inverted indexing with dense vocabulary query expansion on edge devices.",
    "BGE-small provides compact 384-dimensional embeddings with near-zero VRAM consumption."
]

# Initialize indexer & shared vocabulary matrix (<0.3s setup)
orchestrator = PipelineV2Orchestrator(corpus=corpus)

# Execute high-speed retrieval (<15ms on CPU)
result = orchestrator.run("How does Edge-RAG optimize KV-cache memory on edge devices?")

print("Augmented Tokens:", result["aspect_payload"]["augmented_token_list"])
print("Retrieved Chunks:", result["aspect_payload"]["aspects"])
```

### 3. Running Multi-Corpus Evaluation Sweeps
To run automated evaluation sweeps comparing Pipeline V2 schemas against baselines (Lucene BM25, Dense BGE, SPLADE-v3):
```bash
python scripts/run_v2_ablation_sweep.py
```

---

## 🔬 Reproducibility & Architecture References

- **Canonical Architecture**: [`docs/ARCHITECTURE.md`](file:///home/donghv/Projects/Edge-RAG/docs/ARCHITECTURE.md)
- **Module Rules & Boundaries**: [`.agents/rules/01-architecture.md`](file:///home/donghv/Projects/Edge-RAG/.agents/rules/01-architecture.md)
- **Evaluation Metrics**: [`docs/EVALUATION_METRICS.md`](file:///home/donghv/Projects/Edge-RAG/docs/EVALUATION_METRICS.md)
- **Theoretical Foundations**:
  - [Query Expansion Weighting & IT-MPE Theorem](file:///home/donghv/Projects/Edge-RAG/docs/theoretical_foundations_query_expansion_weighting.md)
  - [Saliency-Proportional Expansion Capacity](file:///home/donghv/Projects/Edge-RAG/docs/theoretical_foundations_expansion_capacity.md)
