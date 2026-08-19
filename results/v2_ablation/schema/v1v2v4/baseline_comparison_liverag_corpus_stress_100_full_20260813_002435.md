# Baseline Comparison Report — liverag_corpus_stress_100_full

- **Queries:** 895 | **Corpus chunks:** 709 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 33.9% | 34.3% | 34.3% | 34.3% |
| Pipeline V2 (AspectWeighted) | 33.9% | 34.1% | 34.1% | 34.3% |
| Pipeline V2 (AspectFusion) | 33.9% | 34.3% | 34.3% | 34.3% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 45.9% | 48.0% | 48.6% | 49.3% |
| Pipeline V2 (AspectWeighted) | 45.9% | 47.1% | 48.2% | 49.3% |
| Pipeline V2 (AspectFusion) | 45.9% | 48.0% | 48.6% | 49.3% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 6.9% | 3.6% | 2.4% | 1.5% |
| Pipeline V2 (AspectWeighted) | 6.9% | 3.5% | 2.4% | 1.5% |
| Pipeline V2 (AspectFusion) | 6.9% | 3.6% | 2.4% | 1.5% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 2.75 | 0.39 | 0.0009 | 1.91 | 0.09 |
| Pipeline V2 (AspectWeighted) | 2.75 | 0.37 | 0.0006 | 1.96 | 0.09 |
| Pipeline V2 (AspectFusion) | 2.75 | 0.33 | 0.0152 | 1.96 | 0.10 |