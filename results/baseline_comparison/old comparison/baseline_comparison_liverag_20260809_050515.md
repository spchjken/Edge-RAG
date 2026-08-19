# Baseline Comparison Report — liverag

- **Queries:** 150 | **Corpus chunks:** 368 | **K:** 10 | **Mode:** retriever + listwise rerank (gemma4-e2b)


### Retrieval Quality Metrics

| System | Pre-Rerank Recall | Pre-Rerank Hit Rate | Chunk Recall | Chunk Precision | Strict Recall@10 | Ext Recall@10 | Micro Rerank Recall | Macro Rerank Recall | Avg Retrieved Chunks |
|--------|-------------------|---------------------|--------------|-----------------|------------------|---------------|---------------------|---------------------|----------------------|
| BM25 + Rerank | 79.0% | 96.7% | 54.4% | 81.1% | 96.7% | 96.7% | — | — | 1.66 |
| Dense (bge-small-en-v1.5) + Rerank | 91.4% | 98.7% | 54.4% | 79.8% | 98.0% | 98.0% | — | — | 1.69 |
| Dense (bge-large-en-v1.5) + Rerank | 94.3% | 98.7% | 53.6% | 77.1% | 97.3% | 97.3% | — | — | 1.72 |
| Sweep1_Lmbda_0.5_Gamma_0.0 | 80.9% | 95.3% | 53.9% | 74.6% | 89.3% | 89.3% | 66.3% | 75.4% | 1.79 |

### Timing & Speed Metrics

| System | Index Build TTI (s) | Avg Total (s/query) |
|--------|---------------------|---------------------|
| BM25 + Rerank | 0.053 | 2.575 |
| Dense (bge-small-en-v1.5) + Rerank | 0.903 | 2.430 |
| Dense (bge-large-en-v1.5) + Rerank | 5.556 | 2.407 |
| Sweep1_Lmbda_0.5_Gamma_0.0 | 0.653 | 2.142 |

### Memory & Hardware Metrics

| System | Python RAM (GB) | Peak VRAM (GB) |
|--------|-----------------|----------------|
| BM25 + Rerank | 0.91 | 2.79 |
| Dense (bge-small-en-v1.5) + Rerank | 1.99 | 4.01 |
| Dense (bge-large-en-v1.5) + Rerank | 2.25 | 6.45 |
| Sweep1_Lmbda_0.5_Gamma_0.0 | 1.73 | 2.79 |