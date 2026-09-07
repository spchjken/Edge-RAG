# Baseline Comparison Report — fused_stress_50

- **Queries:** 378 | **Corpus chunks:** 1537 | **K:** 10


### Retrieval Quality Metrics

| System | Pre-Rerank Recall | Pre-Rerank Hit Rate | Chunk Recall | Chunk Precision | Strict Recall@10 | Ext Recall@10 | Micro Rerank Recall | Macro Rerank Recall | Avg Retrieved Chunks |
|--------|-------------------|---------------------|--------------|-----------------|------------------|---------------|---------------------|---------------------|----------------------|
| BM25 (Okapi) | 82.9% | 82.0% | 82.9% | 11.1% | 82.0% | 90.5% | — | — | 10.00 |
| BM25+ | 85.8% | 84.9% | 85.8% | 11.5% | 84.9% | 93.1% | — | — | 10.00 |
| BM25L | 58.3% | 61.4% | 58.3% | 7.8% | 61.4% | 74.6% | — | — | 10.00 |
| BM25 (Lucene) | 85.4% | 84.7% | 85.4% | 11.5% | 84.7% | 92.9% | — | — | 10.00 |
| SPLADE-v3 (DistilBERT) | 80.9% | 82.3% | 80.9% | 10.9% | 82.3% | 91.3% | — | — | 10.00 |
| Dense (bge-small-en-v1.5) | 67.5% | 70.4% | 67.5% | 9.1% | 70.4% | 83.3% | — | — | 10.00 |

### Timing & Speed Metrics

| System | Index Build TTI (s) | Avg Total (s/query) |
|--------|---------------------|---------------------|
| BM25 (Okapi) | 0.209 | 0.003 |
| BM25+ | 0.213 | 0.003 |
| BM25L | 0.184 | 0.003 |
| BM25 (Lucene) | 0.159 | 0.004 |
| SPLADE-v3 (DistilBERT) | 23.250 | 0.008 |
| Dense (bge-small-en-v1.5) | 2.348 | 0.013 |

### Memory & Hardware Metrics

| System | Python RAM (GB) | Peak VRAM (GB) |
|--------|-----------------|----------------|
| BM25 (Okapi) | 0.97 | 0.00 |
| BM25+ | 1.00 | 0.00 |
| BM25L | 1.00 | 0.00 |
| BM25 (Lucene) | 1.01 | 0.00 |
| SPLADE-v3 (DistilBERT) | 1.90 | 4.04 |
| Dense (bge-small-en-v1.5) | 2.52 | 1.55 |