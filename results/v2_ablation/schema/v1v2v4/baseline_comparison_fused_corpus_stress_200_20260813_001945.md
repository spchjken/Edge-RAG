# Baseline Comparison Report — fused_corpus_stress_200

- **Queries:** 569 | **Corpus chunks:** 6950 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 90.2% | 93.5% | 94.6% | 94.7% |
| Pipeline V2 (AspectWeighted) | 88.4% | 91.6% | 93.7% | 94.6% |
| Pipeline V2 (AspectFusion) | 90.2% | 93.5% | 94.6% | 94.7% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 76.6% | 85.5% | 88.4% | 90.4% |
| Pipeline V2 (AspectWeighted) | 73.1% | 84.4% | 87.4% | 89.5% |
| Pipeline V2 (AspectFusion) | 76.6% | 85.5% | 88.4% | 90.4% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 14.8% | 8.2% | 5.7% | 3.5% |
| Pipeline V2 (AspectWeighted) | 14.1% | 8.1% | 5.6% | 3.5% |
| Pipeline V2 (AspectFusion) | 14.8% | 8.2% | 5.7% | 3.5% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 2.81 | 1.43 | 0.0177 | 2.20 | 0.09 |
| Pipeline V2 (AspectWeighted) | 2.81 | 1.25 | 0.0103 | 2.47 | 0.09 |
| Pipeline V2 (AspectFusion) | 2.81 | 1.07 | 0.0337 | 2.47 | 0.10 |