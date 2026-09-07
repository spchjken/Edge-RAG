# Baseline Comparison Report — liverag_corpus_stress_full_full

- **Queries:** 895 | **Corpus chunks:** 2102 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| BM25 (Okapi) | 93.9% | 95.3% | 95.5% | 96.1% |
| BM25+ | 93.4% | 95.0% | 95.6% | 96.4% |
| BM25L | 83.6% | 87.4% | 89.3% | 92.1% |
| BM25 (Lucene) | 93.2% | 95.1% | 95.4% | 96.3% |
| SPLADE-v3 (DistilBERT) | 97.2% | 97.7% | 98.3% | 99.2% |
| Dense (bge-small-en-v1.5) | 97.2% | 98.7% | 98.8% | 99.1% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| BM25 (Okapi) | 71.0% | 75.3% | 77.3% | 79.4% |
| BM25+ | 71.9% | 75.8% | 77.9% | 80.7% |
| BM25L | 57.5% | 64.5% | 67.9% | 71.4% |
| BM25 (Lucene) | 71.2% | 75.7% | 77.6% | 80.5% |
| SPLADE-v3 (DistilBERT) | 84.3% | 87.8% | 90.4% | 92.4% |
| Dense (bge-small-en-v1.5) | 84.8% | 89.1% | 91.5% | 93.7% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| BM25 (Okapi) | 18.0% | 9.5% | 6.5% | 4.0% |
| BM25+ | 18.2% | 9.6% | 6.6% | 4.1% |
| BM25L | 14.6% | 8.2% | 5.7% | 3.6% |
| BM25 (Lucene) | 18.0% | 9.6% | 6.6% | 4.1% |
| SPLADE-v3 (DistilBERT) | 21.4% | 11.1% | 7.6% | 4.7% |
| Dense (bge-small-en-v1.5) | 21.5% | 11.3% | 7.7% | 4.7% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| BM25 (Okapi) | 0.00 | 0.30 | 0.0051 | 1.02 | 0.00 |
| BM25+ | 0.00 | 0.34 | 0.0050 | 1.06 | 0.00 |
| BM25L | 0.00 | 0.31 | 0.0052 | 1.06 | 0.00 |
| BM25 (Lucene) | 0.00 | 0.27 | 0.0059 | 1.07 | 0.00 |
| SPLADE-v3 (DistilBERT) | 7.32 | 21.16 | 0.0074 | 1.86 | 4.05 |
| Dense (bge-small-en-v1.5) | 2.22 | 3.15 | 0.0134 | 2.47 | 1.56 |