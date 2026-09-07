# Baseline Comparison Report — enterpriserag

- **Queries:** 100 | **Corpus chunks:** 331 | **K:** 10 | **Mode:** retriever + listwise rerank (gemma4-e2b)


### Retrieval Quality Metrics

| System | Pre-Rerank Recall | Pre-Rerank Hit Rate | Chunk Recall | Chunk Precision | Strict Recall@10 | Ext Recall@10 | Micro Rerank Recall | Macro Rerank Recall | Avg Retrieved Chunks |
|--------|-------------------|---------------------|--------------|-----------------|------------------|---------------|---------------------|---------------------|----------------------|
| BM25 + Rerank | 70.8% | 94.0% | 52.1% | 36.9% | 96.0% | 96.0% | — | — | 3.33 |
| Dense (bge-small-en-v1.5) + Rerank | 72.9% | 99.0% | 53.4% | 32.7% | 97.0% | 97.0% | — | — | 3.85 |
| Dense (bge-large-en-v1.5) + Rerank | 75.4% | 100.0% | 55.5% | 33.6% | 98.0% | 98.0% | — | — | 3.90 |
| Sweep1_Lmbda_0.5_Gamma_0.0 | 58.5% | 88.0% | 46.2% | 34.7% | 83.0% | 83.0% | 79.0% | 85.8% | 3.14 |

### Timing & Speed Metrics

| System | Index Build TTI (s) | Avg Total (s/query) |
|--------|---------------------|---------------------|
| BM25 + Rerank | 0.049 | 2.776 |
| Dense (bge-small-en-v1.5) + Rerank | 1.012 | 2.745 |
| Dense (bge-large-en-v1.5) + Rerank | 5.452 | 2.756 |
| Sweep1_Lmbda_0.5_Gamma_0.0 | 0.268 | 2.862 |

### Memory & Hardware Metrics

| System | Python RAM (GB) | Peak VRAM (GB) |
|--------|-----------------|----------------|
| BM25 + Rerank | 0.91 | 2.79 |
| Dense (bge-small-en-v1.5) + Rerank | 1.67 | 4.08 |
| Dense (bge-large-en-v1.5) + Rerank | 2.31 | 6.51 |
| Sweep1_Lmbda_0.5_Gamma_0.0 | 1.62 | 2.79 |