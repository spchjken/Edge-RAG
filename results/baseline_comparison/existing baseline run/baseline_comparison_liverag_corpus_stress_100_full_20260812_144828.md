# Baseline Comparison Report — liverag_corpus_stress_100_full

- **Queries:** 895 | **Corpus chunks:** 709 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| BM25 (Okapi) | 33.2% | 33.5% | 34.3% | 34.4% |
| BM25+ | 33.4% | 33.6% | 34.2% | 34.3% |
| BM25L | 31.1% | 32.0% | 32.5% | 33.3% |
| BM25 (Lucene) | 33.4% | 33.6% | 34.2% | 34.3% |
| SPLADE-v3 (DistilBERT) | 34.4% | 34.9% | 35.1% | 35.2% |
| Dense (bge-small-en-v1.5) | 34.5% | 34.9% | 34.9% | 34.9% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| BM25 (Okapi) | 44.5% | 46.2% | 47.9% | 48.7% |
| BM25+ | 44.5% | 46.8% | 47.8% | 48.7% |
| BM25L | 37.4% | 41.1% | 42.8% | 44.9% |
| BM25 (Lucene) | 44.2% | 46.8% | 47.8% | 48.6% |
| SPLADE-v3 (DistilBERT) | 51.2% | 52.7% | 53.5% | 54.6% |
| Dense (bge-small-en-v1.5) | 51.1% | 53.5% | 54.3% | 55.5% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| BM25 (Okapi) | 6.7% | 3.5% | 2.4% | 1.5% |
| BM25+ | 6.7% | 3.5% | 2.4% | 1.5% |
| BM25L | 5.6% | 3.1% | 2.1% | 1.3% |
| BM25 (Lucene) | 6.6% | 3.5% | 2.4% | 1.5% |
| SPLADE-v3 (DistilBERT) | 7.7% | 4.0% | 2.7% | 1.6% |
| Dense (bge-small-en-v1.5) | 7.7% | 4.0% | 2.7% | 1.7% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| BM25 (Okapi) | 0.00 | 0.10 | 0.0013 | 0.94 | 0.00 |
| BM25+ | 0.00 | 0.11 | 0.0014 | 0.96 | 0.00 |
| BM25L | 0.00 | 0.10 | 0.0015 | 0.96 | 0.00 |
| BM25 (Lucene) | 0.00 | 0.08 | 0.0016 | 0.96 | 0.00 |
| SPLADE-v3 (DistilBERT) | 7.41 | 7.14 | 0.0064 | 1.84 | 4.03 |
| Dense (bge-small-en-v1.5) | 2.23 | 1.23 | 0.0119 | 2.43 | 1.49 |