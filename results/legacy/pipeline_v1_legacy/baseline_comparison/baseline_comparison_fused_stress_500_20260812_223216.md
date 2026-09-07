# Baseline Comparison Report — fused_stress_500

- **Queries:** 1084 | **Corpus chunks:** 17241 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 90.8% | 93.8% | 94.9% | 96.6% |
| Pipeline V2 (AspectWeighted) | 88.5% | 92.2% | 93.7% | 95.2% |
| Pipeline V2 (AspectFusion) | 90.7% | 93.6% | 94.6% | 96.3% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 76.2% | 85.5% | 88.1% | 90.8% |
| Pipeline V2 (AspectWeighted) | 73.1% | 83.3% | 86.2% | 89.0% |
| Pipeline V2 (AspectFusion) | 75.9% | 85.3% | 87.8% | 90.6% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 14.3% | 8.0% | 5.5% | 3.4% |
| Pipeline V2 (AspectWeighted) | 13.8% | 7.8% | 5.4% | 3.4% |
| Pipeline V2 (AspectFusion) | 14.3% | 8.0% | 5.5% | 3.4% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 0.00 | 11.36 | 0.1213 | 2.57 | 0.28 |
| Pipeline V2 (AspectWeighted) | 0.00 | 10.70 | 0.1022 | 3.73 | 0.35 |
| Pipeline V2 (AspectFusion) | 0.00 | 10.71 | 0.1549 | 3.79 | 0.35 |