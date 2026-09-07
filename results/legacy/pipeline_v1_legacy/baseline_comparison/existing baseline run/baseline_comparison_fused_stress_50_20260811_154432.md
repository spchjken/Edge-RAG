# Baseline Comparison Report — fused_stress_50

- **Queries:** 378 | **Corpus chunks:** 1537 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| BM25 (Okapi) | 82.0% | 84.4% | 85.2% | 86.8% |
| BM25+ | 84.9% | 87.6% | 88.1% | 88.4% |
| BM25L | 61.4% | 73.3% | 76.5% | 79.6% |
| BM25 (Lucene) | 84.7% | 87.6% | 88.1% | 88.6% |
| SPLADE-v3 (DistilBERT) | 82.3% | 85.4% | 86.2% | 87.0% |
| Dense (bge-small-en-v1.5) | 70.4% | 77.8% | 79.4% | 82.0% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| BM25 (Okapi) | 82.9% | 89.0% | 90.7% | 93.1% |
| BM25+ | 85.8% | 93.5% | 94.9% | 95.9% |
| BM25L | 58.3% | 72.4% | 78.0% | 83.3% |
| BM25 (Lucene) | 85.4% | 93.3% | 94.7% | 95.9% |
| SPLADE-v3 (DistilBERT) | 80.9% | 90.4% | 92.5% | 94.5% |
| Dense (bge-small-en-v1.5) | 67.5% | 78.9% | 83.1% | 87.0% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| BM25 (Okapi) | 11.1% | 6.0% | 4.1% | 2.5% |
| BM25+ | 11.5% | 6.3% | 4.3% | 2.6% |
| BM25L | 7.8% | 4.9% | 3.5% | 2.2% |
| BM25 (Lucene) | 11.5% | 6.3% | 4.2% | 2.6% |
| SPLADE-v3 (DistilBERT) | 10.9% | 6.1% | 4.1% | 2.5% |
| Dense (bge-small-en-v1.5) | 9.1% | 5.3% | 3.7% | 2.3% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| BM25 (Okapi) | 0.00 | 0.21 | 0.0036 | 0.97 | 0.00 |
| BM25+ | 0.00 | 0.22 | 0.0037 | 1.00 | 0.00 |
| BM25L | 0.00 | 0.18 | 0.0037 | 1.00 | 0.00 |
| BM25 (Lucene) | 0.00 | 0.16 | 0.0040 | 1.00 | 0.00 |
| SPLADE-v3 (DistilBERT) | 7.52 | 15.49 | 0.0072 | 1.88 | 4.04 |
| Dense (bge-small-en-v1.5) | 2.31 | 2.30 | 0.0128 | 2.51 | 1.55 |