# Baseline Comparison Report — enterpriserag_corpus_stress_500_full

- **Queries:** 500 | **Corpus chunks:** 2467 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| BM25 (Okapi) | 82.0% | 85.4% | 87.2% | 88.2% |
| BM25+ | 86.8% | 89.6% | 90.6% | 91.2% |
| BM25L | 70.0% | 78.2% | 81.0% | 83.8% |
| BM25 (Lucene) | 86.8% | 89.8% | 90.6% | 91.0% |
| SPLADE-v3 (DistilBERT) | 88.8% | 90.4% | 91.0% | 92.2% |
| Dense (bge-small-en-v1.5) | 80.6% | 84.4% | 86.8% | 89.2% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| BM25 (Okapi) | 48.8% | 56.4% | 62.0% | 66.7% |
| BM25+ | 53.9% | 64.0% | 69.4% | 74.3% |
| BM25L | 39.0% | 48.6% | 53.5% | 59.9% |
| BM25 (Lucene) | 53.3% | 63.1% | 68.8% | 73.9% |
| SPLADE-v3 (DistilBERT) | 56.4% | 66.4% | 71.5% | 77.6% |
| Dense (bge-small-en-v1.5) | 46.3% | 56.4% | 61.7% | 68.9% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| BM25 (Okapi) | 16.7% | 9.6% | 7.1% | 4.6% |
| BM25+ | 18.4% | 10.9% | 7.9% | 5.1% |
| BM25L | 13.3% | 8.3% | 6.1% | 4.1% |
| BM25 (Lucene) | 18.2% | 10.8% | 7.8% | 5.0% |
| SPLADE-v3 (DistilBERT) | 19.2% | 11.3% | 8.1% | 5.3% |
| Dense (bge-small-en-v1.5) | 15.8% | 9.6% | 7.0% | 4.7% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| BM25 (Okapi) | 0.00 | 0.30 | 0.0116 | 1.01 | 0.00 |
| BM25+ | 0.00 | 0.31 | 0.0111 | 1.05 | 0.00 |
| BM25L | 0.00 | 0.30 | 0.0111 | 1.06 | 0.00 |
| BM25 (Lucene) | 0.00 | 0.24 | 0.0133 | 1.06 | 0.00 |
| SPLADE-v3 (DistilBERT) | 7.40 | 24.33 | 0.0088 | 1.89 | 4.05 |
| Dense (bge-small-en-v1.5) | 2.25 | 3.62 | 0.0139 | 2.49 | 1.57 |