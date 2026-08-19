# Baseline Comparison Report — enterpriserag_corpus_stress_1000_capped

- **Queries:** 215 | **Corpus chunks:** 3291 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| BM25 (Okapi) | 78.6% | 79.5% | 80.9% | 81.9% |
| BM25+ | 81.9% | 83.3% | 83.3% | 83.7% |
| BM25L | 67.0% | 74.0% | 77.2% | 77.7% |
| BM25 (Lucene) | 81.4% | 83.3% | 83.3% | 83.7% |
| SPLADE-v3 (DistilBERT) | 81.4% | 81.9% | 83.7% | 83.7% |
| Dense (bge-small-en-v1.5) | 76.7% | 78.6% | 81.4% | 83.3% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| BM25 (Okapi) | 46.6% | 53.9% | 59.3% | 63.4% |
| BM25+ | 51.5% | 61.7% | 67.5% | 72.7% |
| BM25L | 37.1% | 46.5% | 51.7% | 57.7% |
| BM25 (Lucene) | 50.6% | 61.3% | 67.2% | 72.2% |
| SPLADE-v3 (DistilBERT) | 53.2% | 62.1% | 67.5% | 73.3% |
| Dense (bge-small-en-v1.5) | 42.2% | 52.9% | 57.9% | 64.6% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| BM25 (Okapi) | 18.9% | 11.0% | 8.0% | 5.2% |
| BM25+ | 20.9% | 12.5% | 9.1% | 5.9% |
| BM25L | 15.1% | 9.4% | 7.0% | 4.7% |
| BM25 (Lucene) | 20.6% | 12.5% | 9.1% | 5.9% |
| SPLADE-v3 (DistilBERT) | 21.6% | 12.6% | 9.1% | 6.0% |
| Dense (bge-small-en-v1.5) | 17.2% | 10.7% | 7.8% | 5.3% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| BM25 (Okapi) | 0.00 | 0.39 | 0.0135 | 1.04 | 0.00 |
| BM25+ | 0.00 | 0.40 | 0.0152 | 1.10 | 0.00 |
| BM25L | 0.00 | 0.40 | 0.0150 | 1.10 | 0.00 |
| BM25 (Lucene) | 0.00 | 0.34 | 0.0180 | 1.10 | 0.00 |
| SPLADE-v3 (DistilBERT) | 7.17 | 32.45 | 0.0108 | 1.89 | 4.07 |
| Dense (bge-small-en-v1.5) | 2.25 | 4.64 | 0.0149 | 2.52 | 1.57 |