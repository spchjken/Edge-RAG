# Baseline Comparison Report — liverag_pool20

- **Queries:** 150 | **Corpus chunks:** 368 | **K:** 20


### Retrieval Quality Metrics

| System | Pre-Rerank Recall | Pre-Rerank Hit Rate | Chunk Recall | Chunk Precision | Strict Recall@10 | Ext Recall@10 | Micro Rerank Recall | Macro Rerank Recall | Avg Retrieved Chunks |
|--------|-------------------|---------------------|--------------|-----------------|------------------|---------------|---------------------|---------------------|----------------------|
| BM25 | 82.7% | 98.7% | 82.7% | 10.2% | 98.7% | 98.7% | — | — | 20.00 |
| Dense (bge-small-en-v1.5) | 96.0% | 99.3% | 96.0% | 11.9% | 99.3% | 99.3% | — | — | 20.00 |
| Dense (bge-large-en-v1.5) | 97.3% | 100.0% | 97.3% | 12.0% | 100.0% | 100.0% | — | — | 20.00 |

### Timing & Speed Metrics

| System | Index Build TTI (s) | Avg Retrieval (s) | Avg Total (s) |
|--------|---------------------|-------------------|---------------|
| BM25 | 0.059 | 0.001 | 0.149 |
| Dense (bge-small-en-v1.5) | 0.901 | 0.012 | 2.692 |
| Dense (bge-large-en-v1.5) | 6.140 | 0.023 | 9.519 |

### Memory & Hardware Metrics

| System | Python RAM (GB) | Peak VRAM (GB) |
|--------|-----------------|----------------|
| BM25 | 0.91 | 0.00 |
| Dense (bge-small-en-v1.5) | 2.01 | 1.22 |
| Dense (bge-large-en-v1.5) | 2.58 | 3.66 |