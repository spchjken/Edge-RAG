# Baseline Comparison Report — fused_stress_500

- **Queries:** 1084 | **Corpus chunks:** 17241 | **K:** 10


### Retrieval Quality Metrics

| System | Pre-Rerank Recall | Pre-Rerank Hit Rate | Chunk Recall | Chunk Precision | Strict Recall@10 | Ext Recall@10 | Micro Rerank Recall | Macro Rerank Recall | Avg Retrieved Chunks |
|--------|-------------------|---------------------|--------------|-----------------|------------------|---------------|---------------------|---------------------|----------------------|
| BM25 (Okapi) | 76.3% | 89.7% | 76.3% | 14.4% | 89.7% | 92.7% | — | — | 10.00 |
| BM25+ | 78.0% | 92.0% | 78.0% | 14.7% | 92.0% | 94.3% | — | — | 10.00 |
| BM25L | 43.1% | 57.6% | 43.1% | 8.1% | 57.6% | 68.6% | — | — | 10.00 |
| BM25 (Lucene) | 78.3% | 92.3% | 78.3% | 14.8% | 92.3% | 94.6% | — | — | 10.00 |
| SPLADE-v3 (DistilBERT) | 71.0% | 89.3% | 71.0% | 13.4% | 89.3% | 93.8% | — | — | 10.00 |
| Dense (bge-small-en-v1.5) | 56.3% | 74.0% | 56.3% | 10.6% | 74.0% | 82.7% | — | — | 10.00 |

### Timing & Speed Metrics

| System | Index Build TTI (s) | Avg Total (s/query) |
|--------|---------------------|---------------------|
| BM25 (Okapi) | 2.476 | 0.070 |
| BM25+ | 2.563 | 0.069 |
| BM25L | 2.925 | 0.072 |
| BM25 (Lucene) | 1.975 | 0.073 |
| SPLADE-v3 (DistilBERT) | 180.532 | 0.030 |
| Dense (bge-small-en-v1.5) | 21.587 | 0.031 |

### Memory & Hardware Metrics

| System | Python RAM (GB) | Peak VRAM (GB) |
|--------|-----------------|----------------|
| BM25 (Okapi) | 1.71 | 0.00 |
| BM25+ | 1.94 | 0.00 |
| BM25L | 1.95 | 0.00 |
| BM25 (Lucene) | 1.96 | 0.00 |
| SPLADE-v3 (DistilBERT) | 2.38 | 4.23 |
| Dense (bge-small-en-v1.5) | 3.01 | 1.68 |