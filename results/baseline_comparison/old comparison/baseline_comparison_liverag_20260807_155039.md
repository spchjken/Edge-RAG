# Baseline Comparison Report — liverag

- **Queries:** 150 | **Corpus chunks:** 368 | **K:** 10


### Retrieval Quality Metrics

| System | Pre-Rerank Recall | Pre-Rerank Hit Rate | Chunk Recall | Chunk Precision | Strict Recall@10 | Ext Recall@10 | Micro Rerank Recall | Macro Rerank Recall | Avg Retrieved Chunks |
|--------|-------------------|---------------------|--------------|-----------------|------------------|---------------|---------------------|---------------------|----------------------|
| BM25 | 79.0% | 96.7% | 79.0% | 19.5% | 96.7% | 96.7% | — | — | 10.00 |
| Dense (bge-small-en-v1.5) | 91.4% | 98.7% | 91.4% | 22.6% | 98.7% | 98.7% | — | — | 10.00 |
| Dense (bge-large-en-v1.5) | 94.3% | 98.7% | 94.3% | 23.3% | 98.7% | 98.7% | — | — | 10.00 |
| Sweep1_Lmbda_0.5_Gamma_0.0 | 80.9% | 95.3% | 53.9% | 74.6% | 89.3% | 89.3% | 66.3% | 75.4% | 1.79 |

### Timing & Speed Metrics

| System | Index Build TTI (s) | Avg Retrieval (s) | Avg Total (s) |
|--------|---------------------|-------------------|---------------|
| BM25 | 0.058 | 0.001 | 0.146 |
| Dense (bge-small-en-v1.5) | 0.887 | 0.012 | 2.654 |
| Dense (bge-large-en-v1.5) | 5.918 | 0.022 | 9.249 |
| Sweep1_Lmbda_0.5_Gamma_0.0 | 0.653 | 1.446 | 2.142 |

### Memory & Hardware Metrics

| System | Python RAM (GB) | Peak VRAM (GB) |
|--------|-----------------|----------------|
| BM25 | 0.91 | 0.00 |
| Dense (bge-small-en-v1.5) | 2.01 | 1.22 |
| Dense (bge-large-en-v1.5) | 2.59 | 3.66 |
| Sweep1_Lmbda_0.5_Gamma_0.0 | 1.73 | 2.79 |