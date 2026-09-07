# Baseline Comparison Report — liverag_corpus_stress_100_capped

- **Queries:** 196 | **Corpus chunks:** 709 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| BM25 (Okapi) | 95.4% | 96.4% | 97.4% | 98.0% |
| BM25+ | 95.9% | 96.4% | 96.9% | 97.4% |
| BM25L | 91.3% | 93.4% | 95.4% | 96.9% |
| BM25 (Lucene) | 95.9% | 96.4% | 96.9% | 97.4% |
| SPLADE-v3 (DistilBERT) | 98.5% | 99.5% | 99.5% | 100.0% |
| Dense (bge-small-en-v1.5) | 97.4% | 99.0% | 99.0% | 99.0% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| BM25 (Okapi) | 79.3% | 82.3% | 83.9% | 85.3% |
| BM25+ | 79.3% | 83.1% | 83.7% | 85.1% |
| BM25L | 68.5% | 74.5% | 77.7% | 80.9% |
| BM25 (Lucene) | 78.7% | 82.9% | 83.7% | 84.9% |
| SPLADE-v3 (DistilBERT) | 89.8% | 92.6% | 93.6% | 95.8% |
| Dense (bge-small-en-v1.5) | 89.8% | 93.8% | 95.4% | 97.4% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| BM25 (Okapi) | 20.2% | 10.5% | 7.1% | 4.3% |
| BM25+ | 20.2% | 10.6% | 7.1% | 4.3% |
| BM25L | 17.4% | 9.5% | 6.6% | 4.1% |
| BM25 (Lucene) | 20.0% | 10.5% | 7.1% | 4.3% |
| SPLADE-v3 (DistilBERT) | 22.8% | 11.8% | 7.9% | 4.9% |
| Dense (bge-small-en-v1.5) | 22.8% | 11.9% | 8.1% | 4.9% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| BM25 (Okapi) | 0.00 | 0.11 | 0.0015 | 0.94 | 0.00 |
| BM25+ | 0.00 | 0.11 | 0.0015 | 0.95 | 0.00 |
| BM25L | 0.00 | 0.11 | 0.0014 | 0.96 | 0.00 |
| BM25 (Lucene) | 0.00 | 0.08 | 0.0016 | 0.96 | 0.00 |
| SPLADE-v3 (DistilBERT) | 7.33 | 7.15 | 0.0069 | 1.84 | 4.03 |
| Dense (bge-small-en-v1.5) | 2.35 | 1.23 | 0.0119 | 2.43 | 1.49 |