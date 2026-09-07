# Baseline Comparison Report — enterpriserag_corpus_stress_250_capped

- **Queries:** 215 | **Corpus chunks:** 2044 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| BM25 (Okapi) | 79.1% | 80.5% | 81.4% | 82.8% |
| BM25+ | 82.8% | 83.3% | 83.7% | 84.2% |
| BM25L | 69.3% | 75.3% | 77.7% | 78.6% |
| BM25 (Lucene) | 82.8% | 83.3% | 83.7% | 84.2% |
| SPLADE-v3 (DistilBERT) | 81.9% | 83.7% | 84.2% | 84.7% |
| Dense (bge-small-en-v1.5) | 77.2% | 81.4% | 82.3% | 83.7% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| BM25 (Okapi) | 48.3% | 56.4% | 61.9% | 65.8% |
| BM25+ | 53.3% | 64.0% | 69.5% | 74.3% |
| BM25L | 38.2% | 48.1% | 53.0% | 59.2% |
| BM25 (Lucene) | 52.7% | 63.4% | 68.8% | 74.0% |
| SPLADE-v3 (DistilBERT) | 54.7% | 65.4% | 70.8% | 77.2% |
| Dense (bge-small-en-v1.5) | 44.5% | 55.1% | 61.4% | 68.2% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| BM25 (Okapi) | 19.6% | 11.5% | 8.4% | 5.3% |
| BM25+ | 21.7% | 13.0% | 9.4% | 6.0% |
| BM25L | 15.5% | 9.8% | 7.2% | 4.8% |
| BM25 (Lucene) | 21.4% | 12.9% | 9.3% | 6.0% |
| SPLADE-v3 (DistilBERT) | 22.2% | 13.3% | 9.6% | 6.3% |
| Dense (bge-small-en-v1.5) | 18.1% | 11.2% | 8.3% | 5.5% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| BM25 (Okapi) | 0.00 | 0.25 | 0.0085 | 0.99 | 0.00 |
| BM25+ | 0.00 | 0.27 | 0.0082 | 1.03 | 0.00 |
| BM25L | 0.00 | 0.24 | 0.0091 | 1.03 | 0.00 |
| BM25 (Lucene) | 0.00 | 0.22 | 0.0105 | 1.04 | 0.00 |
| SPLADE-v3 (DistilBERT) | 7.75 | 20.25 | 0.0084 | 1.86 | 4.05 |
| Dense (bge-small-en-v1.5) | 2.34 | 3.11 | 0.0134 | 2.47 | 1.56 |