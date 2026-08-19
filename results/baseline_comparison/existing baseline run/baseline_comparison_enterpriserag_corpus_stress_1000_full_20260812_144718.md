# Baseline Comparison Report — enterpriserag_corpus_stress_1000_full

- **Queries:** 500 | **Corpus chunks:** 3291 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| BM25 (Okapi) | 80.0% | 84.6% | 86.6% | 87.6% |
| BM25+ | 85.6% | 88.6% | 90.2% | 90.8% |
| BM25L | 68.2% | 75.8% | 80.4% | 82.2% |
| BM25 (Lucene) | 85.6% | 89.0% | 90.2% | 90.6% |
| SPLADE-v3 (DistilBERT) | 88.0% | 89.4% | 90.6% | 91.2% |
| Dense (bge-small-en-v1.5) | 79.6% | 83.0% | 85.4% | 88.4% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| BM25 (Okapi) | 47.5% | 54.8% | 60.2% | 64.7% |
| BM25+ | 53.0% | 62.5% | 68.1% | 73.2% |
| BM25L | 38.1% | 47.5% | 53.0% | 58.6% |
| BM25 (Lucene) | 52.3% | 62.4% | 67.7% | 72.6% |
| SPLADE-v3 (DistilBERT) | 55.4% | 64.4% | 69.6% | 75.6% |
| Dense (bge-small-en-v1.5) | 44.8% | 54.5% | 59.9% | 66.6% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| BM25 (Okapi) | 16.2% | 9.4% | 6.8% | 4.4% |
| BM25+ | 18.1% | 10.7% | 7.7% | 5.0% |
| BM25L | 13.0% | 8.1% | 6.0% | 4.0% |
| BM25 (Lucene) | 17.9% | 10.7% | 7.7% | 5.0% |
| SPLADE-v3 (DistilBERT) | 18.9% | 11.0% | 7.9% | 5.2% |
| Dense (bge-small-en-v1.5) | 15.3% | 9.3% | 6.8% | 4.5% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| BM25 (Okapi) | 0.00 | 0.41 | 0.0147 | 1.04 | 0.00 |
| BM25+ | 0.00 | 0.40 | 0.0146 | 1.10 | 0.00 |
| BM25L | 0.00 | 0.40 | 0.0152 | 1.10 | 0.00 |
| BM25 (Lucene) | 0.00 | 0.34 | 0.0186 | 1.10 | 0.00 |
| SPLADE-v3 (DistilBERT) | 7.33 | 32.61 | 0.0109 | 1.90 | 4.07 |
| Dense (bge-small-en-v1.5) | 2.23 | 4.70 | 0.0154 | 2.53 | 1.57 |