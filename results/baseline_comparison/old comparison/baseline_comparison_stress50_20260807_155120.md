# Baseline Comparison Report — stress50

- **Queries:** 378 | **Corpus chunks:** 1537 | **K:** 10


### Retrieval Quality Metrics

| System | Pre-Rerank Recall | Pre-Rerank Hit Rate | Chunk Recall | Chunk Precision | Strict Recall@10 | Ext Recall@10 | Micro Rerank Recall | Macro Rerank Recall | Avg Retrieved Chunks |
|--------|-------------------|---------------------|--------------|-----------------|------------------|---------------|---------------------|---------------------|----------------------|
| BM25 | 82.9% | 82.0% | 82.9% | 11.1% | 82.0% | 90.5% | — | — | 10.00 |
| Dense (bge-small-en-v1.5) | 67.5% | 70.4% | 67.5% | 9.1% | 70.4% | 83.3% | — | — | 10.00 |
| Dense (bge-large-en-v1.5) | 72.6% | 74.6% | 72.6% | 9.8% | 74.6% | 85.4% | — | — | 10.00 |
| Sweep1_Lmbda_0.5_Gamma_0.0 | 79.3% | 78.3% | 65.2% | 18.8% | 70.9% | 85.2% | 82.1% | 84.9% | 4.66 |

### Timing & Speed Metrics

| System | Index Build TTI (s) | Avg Retrieval (s) | Avg Total (s) |
|--------|---------------------|-------------------|---------------|
| BM25 | 0.186 | 0.004 | 1.596 |
| Dense (bge-small-en-v1.5) | 2.434 | 0.013 | 7.381 |
| Dense (bge-large-en-v1.5) | 16.320 | 0.026 | 26.028 |
| Sweep1_Lmbda_0.5_Gamma_0.0 | 0.472 | 2.462 | 3.203 |

### Memory & Hardware Metrics

| System | Python RAM (GB) | Peak VRAM (GB) |
|--------|-----------------|----------------|
| BM25 | 0.97 | 0.00 |
| Dense (bge-small-en-v1.5) | 2.14 | 1.29 |
| Dense (bge-large-en-v1.5) | 2.70 | 3.72 |
| Sweep1_Lmbda_0.5_Gamma_0.0 | 1.50 | 2.79 |