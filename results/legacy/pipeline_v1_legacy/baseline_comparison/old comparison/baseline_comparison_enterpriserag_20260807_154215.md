# Baseline Comparison Report — enterpriserag

- **Queries:** 100 | **Corpus chunks:** 331 | **K:** 10


### Retrieval Quality Metrics

| System | Pre-Rerank Recall | Pre-Rerank Hit Rate | Chunk Recall | Chunk Precision | Strict Recall@10 | Ext Recall@10 | Micro Rerank Recall | Macro Rerank Recall | Avg Retrieved Chunks |
|--------|-------------------|---------------------|--------------|-----------------|------------------|---------------|---------------------|---------------------|----------------------|
| BM25 | 70.8% | 94.0% | 70.8% | 16.7% | 94.0% | 94.0% | — | — | 10.00 |
| Dense (bge-small-en-v1.5) | 72.9% | 99.0% | 72.9% | 17.2% | 99.0% | 99.0% | — | — | 10.00 |
| Sweep2_K_15_Gamma_0.5 | 57.6% | 88.0% | 47.9% | 36.0% | 85.0% | 85.0% | 83.1% | 89.0% | 3.14 |

### Timing & Speed Metrics

| System | Index Build TTI (s) | Avg Retrieval (s) | Avg Total (s) |
|--------|---------------------|-------------------|---------------|
| BM25 | 0.050 | 0.001 | 0.140 |
| Dense (bge-small-en-v1.5) | 1.061 | 0.012 | 2.273 |
| Sweep2_K_15_Gamma_0.5 | 0.268 | 1.629 | 2.396 |

### Memory & Hardware Metrics

| System | Python RAM (GB) | Peak VRAM (GB) |
|--------|-----------------|----------------|
| BM25 | 0.91 | 0.00 |
| Dense (bge-small-en-v1.5) | 2.00 | 1.29 |
| Sweep2_K_15_Gamma_0.5 | 1.59 | 2.79 |