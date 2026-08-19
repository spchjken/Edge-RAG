# Baseline Comparison Report — enterpriserag_corpus_stress_500_capped

- **Queries:** 215 | **Corpus chunks:** 2467 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| BM25 (Okapi) | 78.6% | 79.5% | 80.9% | 82.3% |
| BM25+ | 82.8% | 83.3% | 83.7% | 84.2% |
| BM25L | 68.4% | 75.8% | 77.7% | 78.1% |
| BM25 (Lucene) | 82.3% | 83.3% | 83.7% | 84.2% |
| SPLADE-v3 (DistilBERT) | 81.9% | 83.7% | 83.7% | 84.7% |
| Dense (bge-small-en-v1.5) | 76.7% | 80.5% | 82.3% | 83.7% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| BM25 (Okapi) | 47.6% | 55.4% | 60.8% | 65.0% |
| BM25+ | 52.5% | 63.2% | 69.5% | 73.9% |
| BM25L | 37.8% | 47.6% | 52.4% | 59.0% |
| BM25 (Lucene) | 51.5% | 62.1% | 68.5% | 73.5% |
| SPLADE-v3 (DistilBERT) | 53.8% | 64.0% | 69.5% | 75.5% |
| Dense (bge-small-en-v1.5) | 43.6% | 54.2% | 60.1% | 66.7% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| BM25 (Okapi) | 19.3% | 11.3% | 8.2% | 5.3% |
| BM25+ | 21.3% | 12.8% | 9.4% | 6.0% |
| BM25L | 15.3% | 9.7% | 7.1% | 4.8% |
| BM25 (Lucene) | 20.9% | 12.6% | 9.3% | 6.0% |
| SPLADE-v3 (DistilBERT) | 21.9% | 13.0% | 9.4% | 6.1% |
| Dense (bge-small-en-v1.5) | 17.7% | 11.0% | 8.1% | 5.4% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| BM25 (Okapi) | 0.00 | 0.29 | 0.0098 | 1.01 | 0.00 |
| BM25+ | 0.00 | 0.32 | 0.0111 | 1.05 | 0.00 |
| BM25L | 0.00 | 0.31 | 0.0106 | 1.06 | 0.00 |
| BM25 (Lucene) | 0.00 | 0.24 | 0.0127 | 1.06 | 0.00 |
| SPLADE-v3 (DistilBERT) | 7.32 | 24.25 | 0.0092 | 1.90 | 4.05 |
| Dense (bge-small-en-v1.5) | 2.17 | 3.70 | 0.0150 | 2.50 | 1.57 |