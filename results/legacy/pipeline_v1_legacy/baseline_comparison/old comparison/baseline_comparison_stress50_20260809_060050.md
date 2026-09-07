# Baseline Comparison Report — stress50

- **Queries:** 378 | **Corpus chunks:** 1537 | **K:** 10 | **Mode:** retriever + listwise rerank (gemma4-e2b)


### Retrieval Quality Metrics

| System | Pre-Rerank Recall | Pre-Rerank Hit Rate | Chunk Recall | Chunk Precision | Strict Recall@10 | Ext Recall@10 | Micro Rerank Recall | Macro Rerank Recall | Avg Retrieved Chunks |
|--------|-------------------|---------------------|--------------|-----------------|------------------|---------------|---------------------|---------------------|----------------------|
| BM25 + Rerank | 82.9% | 82.0% | 62.2% | 17.5% | 69.8% | 84.9% | — | — | 4.77 |
| Dense (bge-small-en-v1.5) + Rerank | 67.5% | 70.4% | 58.3% | 17.1% | 66.1% | 81.2% | — | — | 4.59 |
| Dense (bge-large-en-v1.5) + Rerank | 72.6% | 74.6% | 59.4% | 16.2% | 67.7% | 83.3% | — | — | 4.92 |
| Sweep1_Lmbda_0.5_Gamma_0.0 | 79.3% | 78.3% | 65.2% | 18.8% | 70.9% | 85.2% | 82.1% | 84.9% | 4.66 |

### Timing & Speed Metrics

| System | Index Build TTI (s) | Avg Total (s/query) |
|--------|---------------------|---------------------|
| BM25 + Rerank | 0.201 | 2.843 |
| Dense (bge-small-en-v1.5) + Rerank | 2.209 | 2.888 |
| Dense (bge-large-en-v1.5) + Rerank | 15.017 | 3.015 |
| Sweep1_Lmbda_0.5_Gamma_0.0 | 0.472 | 3.203 |

### Memory & Hardware Metrics

| System | Python RAM (GB) | Peak VRAM (GB) |
|--------|-----------------|----------------|
| BM25 + Rerank | 0.97 | 2.79 |
| Dense (bge-small-en-v1.5) + Rerank | 2.12 | 4.08 |
| Dense (bge-large-en-v1.5) + Rerank | 2.35 | 6.51 |
| Sweep1_Lmbda_0.5_Gamma_0.0 | 1.50 | 2.79 |