# Baseline Comparison Report — liverag_corpus_stress_250_full

- **Queries:** 895 | **Corpus chunks:** 1042 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 48.4% | 49.4% | 49.4% | 49.9% |
| Pipeline V2 (AspectWeighted) | 48.2% | 48.9% | 49.4% | 49.9% |
| Pipeline V2 (AspectFusion) | 48.4% | 49.4% | 49.4% | 49.9% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 56.1% | 59.0% | 59.7% | 61.4% |
| Pipeline V2 (AspectWeighted) | 55.9% | 58.2% | 59.6% | 61.2% |
| Pipeline V2 (AspectFusion) | 56.1% | 59.0% | 59.7% | 61.4% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 9.7% | 5.1% | 3.5% | 2.1% |
| Pipeline V2 (AspectWeighted) | 9.7% | 5.1% | 3.4% | 2.1% |
| Pipeline V2 (AspectFusion) | 9.7% | 5.1% | 3.5% | 2.1% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 2.74 | 0.56 | 0.0014 | 1.93 | 0.09 |
| Pipeline V2 (AspectWeighted) | 2.74 | 0.54 | 0.0010 | 1.99 | 0.09 |
| Pipeline V2 (AspectFusion) | 2.74 | 0.50 | 0.0160 | 1.99 | 0.10 |