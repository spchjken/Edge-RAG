# Baseline Comparison Report — liverag_corpus_stress_250_full

- **Queries:** 895 | **Corpus chunks:** 1042 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| BM25 (Okapi) | 48.3% | 48.7% | 49.3% | 49.9% |
| BM25+ | 48.0% | 48.7% | 49.3% | 49.9% |
| BM25L | 44.4% | 46.4% | 47.0% | 48.0% |
| BM25 (Lucene) | 47.8% | 48.8% | 49.4% | 50.1% |
| SPLADE-v3 (DistilBERT) | 50.1% | 50.6% | 51.1% | 51.2% |
| Dense (bge-small-en-v1.5) | 50.1% | 50.7% | 50.9% | 51.1% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| BM25 (Okapi) | 53.9% | 56.7% | 58.0% | 60.0% |
| BM25+ | 53.8% | 57.1% | 58.9% | 60.2% |
| BM25L | 44.3% | 49.6% | 51.5% | 54.3% |
| BM25 (Lucene) | 53.4% | 56.9% | 58.9% | 60.2% |
| SPLADE-v3 (DistilBERT) | 62.8% | 65.4% | 66.6% | 68.1% |
| Dense (bge-small-en-v1.5) | 63.4% | 66.3% | 67.5% | 69.2% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| BM25 (Okapi) | 9.4% | 4.9% | 3.4% | 2.1% |
| BM25+ | 9.3% | 5.0% | 3.4% | 2.1% |
| BM25L | 7.7% | 4.3% | 3.0% | 1.9% |
| BM25 (Lucene) | 9.3% | 4.9% | 3.4% | 2.1% |
| SPLADE-v3 (DistilBERT) | 10.9% | 5.7% | 3.9% | 2.4% |
| Dense (bge-small-en-v1.5) | 11.0% | 5.8% | 3.9% | 2.4% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| BM25 (Okapi) | 0.00 | 0.15 | 0.0022 | 0.96 | 0.00 |
| BM25+ | 0.00 | 0.16 | 0.0023 | 0.98 | 0.00 |
| BM25L | 0.00 | 0.15 | 0.0023 | 0.98 | 0.00 |
| BM25 (Lucene) | 0.00 | 0.12 | 0.0025 | 0.99 | 0.00 |
| SPLADE-v3 (DistilBERT) | 7.30 | 10.60 | 0.0070 | 1.85 | 4.04 |
| Dense (bge-small-en-v1.5) | 2.25 | 1.76 | 0.0122 | 2.45 | 1.55 |