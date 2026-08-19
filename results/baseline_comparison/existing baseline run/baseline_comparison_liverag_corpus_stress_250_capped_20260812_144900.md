# Baseline Comparison Report — liverag_corpus_stress_250_capped

- **Queries:** 196 | **Corpus chunks:** 1042 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| BM25 (Okapi) | 94.9% | 95.9% | 96.4% | 96.9% |
| BM25+ | 94.9% | 95.4% | 96.4% | 96.9% |
| BM25L | 90.3% | 93.4% | 93.9% | 95.9% |
| BM25 (Lucene) | 94.9% | 95.9% | 96.4% | 96.9% |
| SPLADE-v3 (DistilBERT) | 98.5% | 99.0% | 99.5% | 99.5% |
| Dense (bge-small-en-v1.5) | 97.4% | 98.5% | 99.0% | 99.0% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| BM25 (Okapi) | 77.5% | 80.9% | 82.3% | 83.7% |
| BM25+ | 76.7% | 81.7% | 83.1% | 84.1% |
| BM25L | 66.3% | 72.7% | 75.3% | 78.5% |
| BM25 (Lucene) | 76.1% | 81.7% | 83.1% | 84.1% |
| SPLADE-v3 (DistilBERT) | 87.8% | 91.6% | 92.6% | 94.8% |
| Dense (bge-small-en-v1.5) | 88.6% | 92.4% | 93.8% | 96.2% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| BM25 (Okapi) | 19.7% | 10.3% | 7.0% | 4.3% |
| BM25+ | 19.5% | 10.4% | 7.0% | 4.3% |
| BM25L | 16.8% | 9.2% | 6.4% | 4.0% |
| BM25 (Lucene) | 19.3% | 10.4% | 7.0% | 4.3% |
| SPLADE-v3 (DistilBERT) | 22.3% | 11.6% | 7.8% | 4.8% |
| Dense (bge-small-en-v1.5) | 22.5% | 11.7% | 7.9% | 4.9% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| BM25 (Okapi) | 0.00 | 0.15 | 0.0022 | 0.95 | 0.00 |
| BM25+ | 0.00 | 0.16 | 0.0022 | 0.98 | 0.00 |
| BM25L | 0.00 | 0.15 | 0.0023 | 0.98 | 0.00 |
| BM25 (Lucene) | 0.00 | 0.13 | 0.0026 | 0.98 | 0.00 |
| SPLADE-v3 (DistilBERT) | 8.42 | 10.48 | 0.0072 | 1.84 | 4.04 |
| Dense (bge-small-en-v1.5) | 2.22 | 1.70 | 0.0122 | 2.44 | 1.55 |