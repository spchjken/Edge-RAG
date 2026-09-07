# Baseline Comparison Report — fused_stress_500

- **Queries:** 1084 | **Corpus chunks:** 17241 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| BM25 (Okapi) | 89.7% | 93.0% | 93.7% | 95.1% |
| BM25+ | 92.0% | 94.6% | 95.6% | 96.4% |
| BM25L | 57.6% | 68.1% | 74.1% | 80.4% |
| BM25 (Lucene) | 92.3% | 94.7% | 95.6% | 96.4% |
| SPLADE-v3 (DistilBERT) | 89.3% | 92.2% | 93.5% | 94.9% |
| Dense (bge-small-en-v1.5) | 74.0% | 80.6% | 82.0% | 84.4% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| BM25 (Okapi) | 76.3% | 83.1% | 85.4% | 87.9% |
| BM25+ | 78.0% | 85.7% | 88.1% | 90.5% |
| BM25L | 43.1% | 55.8% | 62.0% | 68.9% |
| BM25 (Lucene) | 78.3% | 85.7% | 87.9% | 90.6% |
| SPLADE-v3 (DistilBERT) | 71.0% | 79.5% | 83.1% | 85.8% |
| Dense (bge-small-en-v1.5) | 56.3% | 66.9% | 70.4% | 74.6% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| BM25 (Okapi) | 14.4% | 7.8% | 5.4% | 3.3% |
| BM25+ | 14.7% | 8.1% | 5.5% | 3.4% |
| BM25L | 8.1% | 5.3% | 3.9% | 2.6% |
| BM25 (Lucene) | 14.8% | 8.1% | 5.5% | 3.4% |
| SPLADE-v3 (DistilBERT) | 13.4% | 7.5% | 5.2% | 3.2% |
| Dense (bge-small-en-v1.5) | 10.6% | 6.3% | 4.4% | 2.8% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| BM25 (Okapi) | 0.00 | 2.34 | 0.0685 | 1.71 | 0.00 |
| BM25+ | 0.00 | 2.56 | 0.0693 | 1.94 | 0.00 |
| BM25L | 0.00 | 2.47 | 0.0705 | 1.95 | 0.00 |
| BM25 (Lucene) | 0.00 | 1.98 | 0.0727 | 1.97 | 0.00 |
| SPLADE-v3 (DistilBERT) | 7.11 | 174.03 | 0.0309 | 2.12 | 4.23 |
| Dense (bge-small-en-v1.5) | 2.23 | 22.02 | 0.0315 | 2.94 | 1.68 |