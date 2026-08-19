# Baseline Comparison Report — enterpriserag_corpus_stress_100_full

- **Queries:** 500 | **Corpus chunks:** 1783 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| BM25 (Okapi) | 83.2% | 87.2% | 88.4% | 89.8% |
| BM25+ | 87.8% | 90.2% | 90.6% | 91.2% |
| BM25L | 72.4% | 79.8% | 82.2% | 85.0% |
| BM25 (Lucene) | 87.8% | 90.4% | 90.6% | 91.2% |
| SPLADE-v3 (DistilBERT) | 89.2% | 91.2% | 91.8% | 92.8% |
| Dense (bge-small-en-v1.5) | 82.0% | 86.4% | 88.4% | 89.8% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| BM25 (Okapi) | 50.1% | 58.9% | 64.1% | 69.1% |
| BM25+ | 55.4% | 65.1% | 70.8% | 75.6% |
| BM25L | 40.1% | 49.6% | 54.6% | 60.6% |
| BM25 (Lucene) | 55.1% | 64.6% | 70.1% | 75.4% |
| SPLADE-v3 (DistilBERT) | 57.6% | 68.7% | 74.0% | 80.0% |
| Dense (bge-small-en-v1.5) | 48.1% | 58.5% | 64.6% | 71.2% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| BM25 (Okapi) | 17.1% | 10.1% | 7.3% | 4.7% |
| BM25+ | 18.9% | 11.1% | 8.1% | 5.2% |
| BM25L | 13.7% | 8.5% | 6.2% | 4.1% |
| BM25 (Lucene) | 18.8% | 11.0% | 8.0% | 5.1% |
| SPLADE-v3 (DistilBERT) | 19.7% | 11.7% | 8.4% | 5.5% |
| Dense (bge-small-en-v1.5) | 16.4% | 10.0% | 7.3% | 4.9% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| BM25 (Okapi) | 0.00 | 0.21 | 0.0079 | 0.98 | 0.00 |
| BM25+ | 0.00 | 0.23 | 0.0076 | 1.01 | 0.00 |
| BM25L | 0.00 | 0.21 | 0.0086 | 1.01 | 0.00 |
| BM25 (Lucene) | 0.00 | 0.18 | 0.0101 | 1.02 | 0.00 |
| SPLADE-v3 (DistilBERT) | 7.38 | 17.96 | 0.0078 | 1.86 | 4.04 |
| Dense (bge-small-en-v1.5) | 2.18 | 2.65 | 0.0132 | 2.47 | 1.56 |