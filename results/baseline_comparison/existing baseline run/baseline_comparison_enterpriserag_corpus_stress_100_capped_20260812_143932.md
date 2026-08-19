# Baseline Comparison Report — enterpriserag_corpus_stress_100_capped

- **Queries:** 215 | **Corpus chunks:** 1783 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| BM25 (Okapi) | 78.6% | 81.4% | 82.3% | 83.3% |
| BM25+ | 83.3% | 83.3% | 83.7% | 84.2% |
| BM25L | 69.3% | 75.8% | 77.7% | 78.6% |
| BM25 (Lucene) | 83.3% | 83.3% | 83.7% | 84.2% |
| SPLADE-v3 (DistilBERT) | 81.9% | 83.7% | 84.2% | 84.7% |
| Dense (bge-small-en-v1.5) | 78.1% | 81.9% | 82.8% | 83.7% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| BM25 (Okapi) | 48.5% | 57.3% | 62.8% | 67.4% |
| BM25+ | 54.0% | 64.4% | 70.3% | 74.5% |
| BM25L | 38.2% | 48.3% | 53.4% | 58.7% |
| BM25 (Lucene) | 53.4% | 63.6% | 69.7% | 74.4% |
| SPLADE-v3 (DistilBERT) | 55.4% | 66.1% | 72.4% | 78.1% |
| Dense (bge-small-en-v1.5) | 45.2% | 55.9% | 62.5% | 69.3% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| BM25 (Okapi) | 19.7% | 11.7% | 8.5% | 5.5% |
| BM25+ | 22.0% | 13.1% | 9.5% | 6.1% |
| BM25L | 15.5% | 9.8% | 7.2% | 4.8% |
| BM25 (Lucene) | 21.7% | 12.9% | 9.4% | 6.0% |
| SPLADE-v3 (DistilBERT) | 22.5% | 13.4% | 9.8% | 6.4% |
| Dense (bge-small-en-v1.5) | 18.4% | 11.4% | 8.5% | 5.6% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| BM25 (Okapi) | 0.00 | 0.22 | 0.0077 | 0.97 | 0.00 |
| BM25+ | 0.00 | 0.24 | 0.0086 | 1.01 | 0.00 |
| BM25L | 0.00 | 0.21 | 0.0080 | 1.01 | 0.00 |
| BM25 (Lucene) | 0.00 | 0.17 | 0.0093 | 1.01 | 0.00 |
| SPLADE-v3 (DistilBERT) | 8.99 | 18.12 | 0.0088 | 1.85 | 4.04 |
| Dense (bge-small-en-v1.5) | 2.33 | 2.67 | 0.0133 | 2.46 | 1.56 |