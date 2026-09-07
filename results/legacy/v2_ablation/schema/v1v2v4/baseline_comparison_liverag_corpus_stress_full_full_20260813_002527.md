# Baseline Comparison Report — liverag_corpus_stress_full_full

- **Queries:** 895 | **Corpus chunks:** 2102 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 93.9% | 95.4% | 96.0% | 96.1% |
| Pipeline V2 (AspectWeighted) | 93.2% | 94.7% | 95.6% | 96.0% |
| Pipeline V2 (AspectFusion) | 93.9% | 95.4% | 96.0% | 96.1% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 74.6% | 79.3% | 80.5% | 82.0% |
| Pipeline V2 (AspectWeighted) | 74.1% | 78.4% | 80.0% | 81.5% |
| Pipeline V2 (AspectFusion) | 74.6% | 79.3% | 80.5% | 82.0% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 18.9% | 10.0% | 6.8% | 4.2% |
| Pipeline V2 (AspectWeighted) | 18.8% | 9.9% | 6.8% | 4.1% |
| Pipeline V2 (AspectFusion) | 18.9% | 10.0% | 6.8% | 4.2% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 2.71 | 0.73 | 0.0037 | 1.99 | 0.09 |
| Pipeline V2 (AspectWeighted) | 2.71 | 0.71 | 0.0024 | 2.11 | 0.09 |
| Pipeline V2 (AspectFusion) | 2.71 | 0.82 | 0.0185 | 2.11 | 0.10 |