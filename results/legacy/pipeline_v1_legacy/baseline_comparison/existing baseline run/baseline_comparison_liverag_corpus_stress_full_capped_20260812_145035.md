# Baseline Comparison Report — liverag_corpus_stress_full_capped

- **Queries:** 196 | **Corpus chunks:** 2102 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| BM25 (Okapi) | 94.4% | 94.9% | 95.4% | 95.9% |
| BM25+ | 93.9% | 94.9% | 94.9% | 96.4% |
| BM25L | 87.2% | 91.3% | 91.8% | 93.9% |
| BM25 (Lucene) | 93.4% | 94.9% | 95.4% | 96.4% |
| SPLADE-v3 (DistilBERT) | 96.9% | 98.0% | 98.5% | 99.5% |
| Dense (bge-small-en-v1.5) | 96.9% | 98.0% | 98.0% | 99.0% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| BM25 (Okapi) | 74.1% | 78.7% | 80.7% | 81.7% |
| BM25+ | 74.3% | 77.9% | 80.9% | 82.7% |
| BM25L | 61.4% | 68.9% | 72.1% | 74.7% |
| BM25 (Lucene) | 73.5% | 77.7% | 80.7% | 82.9% |
| SPLADE-v3 (DistilBERT) | 84.3% | 88.4% | 90.6% | 92.8% |
| Dense (bge-small-en-v1.5) | 84.3% | 88.4% | 91.2% | 93.6% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| BM25 (Okapi) | 18.8% | 10.0% | 6.8% | 4.2% |
| BM25+ | 18.9% | 9.9% | 6.9% | 4.2% |
| BM25L | 15.6% | 8.8% | 6.1% | 3.8% |
| BM25 (Lucene) | 18.7% | 9.9% | 6.8% | 4.2% |
| SPLADE-v3 (DistilBERT) | 21.4% | 11.2% | 7.7% | 4.7% |
| Dense (bge-small-en-v1.5) | 21.4% | 11.2% | 7.7% | 4.8% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| BM25 (Okapi) | 0.00 | 0.31 | 0.0050 | 1.01 | 0.00 |
| BM25+ | 0.00 | 0.33 | 0.0051 | 1.06 | 0.00 |
| BM25L | 0.00 | 0.30 | 0.0052 | 1.06 | 0.00 |
| BM25 (Lucene) | 0.00 | 0.27 | 0.0064 | 1.06 | 0.00 |
| SPLADE-v3 (DistilBERT) | 7.28 | 21.11 | 0.0078 | 1.86 | 4.05 |
| Dense (bge-small-en-v1.5) | 2.23 | 3.17 | 0.0134 | 2.47 | 1.56 |