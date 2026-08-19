# Baseline Comparison Report — enterpriserag

- **Queries:** 100 | **Corpus chunks:** 331 | **K:** 10


### Retrieval Quality Metrics

| System | Pre-Rerank Recall | Pre-Rerank Hit Rate | Chunk Recall | Chunk Precision | Strict Recall@10 | Ext Recall@10 | Micro Rerank Recall | Macro Rerank Recall | Avg Retrieved Chunks |
|--------|-------------------|---------------------|--------------|-----------------|------------------|---------------|---------------------|---------------------|----------------------|
| BM25 | 70.8% | 94.0% | 70.8% | 16.7% | 94.0% | 94.0% | — | — | 10.00 |
| Dense (bge-small-en-v1.5) | 72.9% | 99.0% | 72.9% | 17.2% | 99.0% | 99.0% | — | — | 10.00 |
| Dense (bge-large-en-v1.5) | 75.4% | 100.0% | 75.4% | 17.8% | 100.0% | 100.0% | — | — | 10.00 |
| Sweep1_Lmbda_0.5_Gamma_0.0 | 58.5% | 88.0% | 46.2% | 34.7% | 83.0% | 83.0% | 79.0% | 85.8% | 3.14 |

### Timing & Speed Metrics

| System | Index Build TTI (s) | Avg Retrieval (s) | Avg Total (s) |
|--------|---------------------|-------------------|---------------|
| BM25 | 0.050 | 0.001 | 0.139 |
| Dense (bge-small-en-v1.5) | 1.008 | 0.012 | 2.205 |
| Dense (bge-large-en-v1.5) | 5.975 | 0.021 | 8.113 |
| Sweep1_Lmbda_0.5_Gamma_0.0 | 0.268 | 2.086 | 2.862 |

### Memory & Hardware Metrics

| System | Python RAM (GB) | Peak VRAM (GB) |
|--------|-----------------|----------------|
| BM25 | 0.91 | 0.00 |
| Dense (bge-small-en-v1.5) | 2.00 | 1.29 |
| Dense (bge-large-en-v1.5) | 2.72 | 3.72 |
| Sweep1_Lmbda_0.5_Gamma_0.0 | 1.62 | 2.79 |