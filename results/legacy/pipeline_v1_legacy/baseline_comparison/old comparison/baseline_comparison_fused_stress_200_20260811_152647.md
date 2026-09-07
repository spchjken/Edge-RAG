# Baseline Comparison Report — fused_stress_200

- **Queries:** 569 | **Corpus chunks:** 6950 | **K:** 10


### Retrieval Quality Metrics

| System | Pre-Rerank Recall | Pre-Rerank Hit Rate | Chunk Recall | Chunk Precision | Strict Recall@10 | Ext Recall@10 | Micro Rerank Recall | Macro Rerank Recall | Avg Retrieved Chunks |
|--------|-------------------|---------------------|--------------|-----------------|------------------|---------------|---------------------|---------------------|----------------------|
| BM25 (Okapi) | 74.8% | 87.3% | 74.8% | 14.4% | 87.3% | 90.3% | — | — | 10.00 |
| BM25+ | 76.8% | 90.3% | 76.8% | 14.8% | 90.3% | 92.8% | — | — | 10.00 |
| BM25L | 46.9% | 58.9% | 46.9% | 9.1% | 58.9% | 69.9% | — | — | 10.00 |
| BM25 (Lucene) | 77.0% | 90.7% | 77.0% | 14.9% | 90.7% | 93.1% | — | — | 10.00 |
| SPLADE-v3 (DistilBERT) | 71.4% | 90.5% | 71.4% | 13.8% | 90.5% | 94.7% | — | — | 10.00 |
| Dense (bge-small-en-v1.5) | 56.3% | 76.4% | 56.3% | 10.9% | 76.4% | 84.9% | — | — | 10.00 |

### Timing & Speed Metrics

| System | Index Build TTI (s) | Avg Total (s/query) |
|--------|---------------------|---------------------|
| BM25 (Okapi) | 0.930 | 0.020 |
| BM25+ | 1.002 | 0.020 |
| BM25L | 0.846 | 0.022 |
| BM25 (Lucene) | 0.762 | 0.023 |
| SPLADE-v3 (DistilBERT) | 77.557 | 0.015 |
| Dense (bge-small-en-v1.5) | 8.906 | 0.019 |

### Memory & Hardware Metrics

| System | Python RAM (GB) | Peak VRAM (GB) |
|--------|-----------------|----------------|
| BM25 (Okapi) | 1.22 | 0.00 |
| BM25+ | 1.32 | 0.00 |
| BM25L | 1.33 | 0.00 |
| BM25 (Lucene) | 1.33 | 0.00 |
| SPLADE-v3 (DistilBERT) | 2.06 | 4.10 |
| Dense (bge-small-en-v1.5) | 2.69 | 1.59 |