# Baseline Comparison Report — enterpriserag_corpus_stress_250_full

- **Queries:** 500 | **Corpus chunks:** 2044 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| BM25 (Okapi) | 83.2% | 86.6% | 87.4% | 89.2% |
| BM25+ | 87.4% | 90.0% | 90.6% | 91.2% |
| BM25L | 71.4% | 78.6% | 81.6% | 85.0% |
| BM25 (Lucene) | 87.4% | 90.0% | 90.6% | 91.0% |
| SPLADE-v3 (DistilBERT) | 89.0% | 91.0% | 91.4% | 92.6% |
| Dense (bge-small-en-v1.5) | 81.2% | 85.2% | 87.8% | 89.2% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| BM25 (Okapi) | 49.7% | 57.8% | 63.0% | 67.7% |
| BM25+ | 54.7% | 64.7% | 69.8% | 75.3% |
| BM25L | 39.5% | 49.1% | 54.1% | 60.3% |
| BM25 (Lucene) | 54.4% | 64.1% | 69.2% | 74.8% |
| SPLADE-v3 (DistilBERT) | 57.1% | 68.0% | 72.6% | 79.3% |
| Dense (bge-small-en-v1.5) | 47.3% | 57.4% | 63.3% | 70.0% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| BM25 (Okapi) | 17.0% | 9.9% | 7.2% | 4.6% |
| BM25+ | 18.7% | 11.0% | 7.9% | 5.1% |
| BM25L | 13.5% | 8.4% | 6.2% | 4.1% |
| BM25 (Lucene) | 18.6% | 10.9% | 7.9% | 5.1% |
| SPLADE-v3 (DistilBERT) | 19.5% | 11.6% | 8.3% | 5.4% |
| Dense (bge-small-en-v1.5) | 16.2% | 9.8% | 7.2% | 4.8% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| BM25 (Okapi) | 0.00 | 0.25 | 0.0085 | 0.99 | 0.00 |
| BM25+ | 0.00 | 0.27 | 0.0086 | 1.03 | 0.00 |
| BM25L | 0.00 | 0.24 | 0.0090 | 1.03 | 0.00 |
| BM25 (Lucene) | 0.00 | 0.22 | 0.0108 | 1.04 | 0.00 |
| SPLADE-v3 (DistilBERT) | 7.41 | 20.18 | 0.0080 | 1.86 | 4.05 |
| Dense (bge-small-en-v1.5) | 2.34 | 3.11 | 0.0133 | 2.47 | 1.56 |