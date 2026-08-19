# Baseline Comparison Report — fused_stress_200

- **Queries:** 569 | **Corpus chunks:** 6950 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| BM25 (Okapi) | 87.3% | 91.7% | 92.3% | 92.6% |
| BM25+ | 90.3% | 93.3% | 94.4% | 95.4% |
| BM25L | 58.9% | 70.7% | 76.1% | 81.4% |
| BM25 (Lucene) | 90.7% | 93.3% | 94.2% | 95.3% |
| SPLADE-v3 (DistilBERT) | 90.5% | 93.3% | 94.0% | 95.1% |
| Dense (bge-small-en-v1.5) | 76.4% | 81.5% | 83.1% | 86.3% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| BM25 (Okapi) | 74.8% | 82.3% | 84.4% | 86.8% |
| BM25+ | 76.8% | 84.8% | 87.3% | 90.2% |
| BM25L | 46.9% | 58.9% | 64.9% | 71.2% |
| BM25 (Lucene) | 77.0% | 84.8% | 87.1% | 90.0% |
| SPLADE-v3 (DistilBERT) | 71.4% | 80.8% | 84.7% | 87.6% |
| Dense (bge-small-en-v1.5) | 56.3% | 67.0% | 70.6% | 75.4% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| BM25 (Okapi) | 14.4% | 7.9% | 5.4% | 3.3% |
| BM25+ | 14.8% | 8.2% | 5.6% | 3.5% |
| BM25L | 9.1% | 5.7% | 4.2% | 2.7% |
| BM25 (Lucene) | 14.9% | 8.2% | 5.6% | 3.5% |
| SPLADE-v3 (DistilBERT) | 13.8% | 7.8% | 5.4% | 3.4% |
| Dense (bge-small-en-v1.5) | 10.9% | 6.5% | 4.5% | 2.9% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| BM25 (Okapi) | 0.00 | 0.92 | 0.0196 | 1.22 | 0.00 |
| BM25+ | 0.00 | 1.00 | 0.0195 | 1.32 | 0.00 |
| BM25L | 0.00 | 0.85 | 0.0207 | 1.33 | 0.00 |
| BM25 (Lucene) | 0.00 | 0.77 | 0.0223 | 1.33 | 0.00 |
| SPLADE-v3 (DistilBERT) | 7.62 | 70.76 | 0.0154 | 1.98 | 4.10 |
| Dense (bge-small-en-v1.5) | 2.28 | 8.90 | 0.0194 | 2.67 | 1.59 |