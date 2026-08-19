# Baseline Comparison Report — fused_stress_500

- **Queries:** 1084 | **Corpus chunks:** 17241 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 91.1% | 94.2% | 95.3% | 96.7% |
| Pipeline V2 (AspectWeighted) | 88.9% | 92.6% | 94.3% | 95.7% |
| Pipeline V2 (AspectFusion) | 91.1% | 94.2% | 95.3% | 96.7% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 77.1% | 85.8% | 88.5% | 91.1% |
| Pipeline V2 (AspectWeighted) | 73.9% | 83.6% | 87.1% | 89.9% |
| Pipeline V2 (AspectFusion) | 77.1% | 85.8% | 88.5% | 91.1% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 14.5% | 8.1% | 5.6% | 3.4% |
| Pipeline V2 (AspectWeighted) | 13.9% | 7.9% | 5.5% | 3.4% |
| Pipeline V2 (AspectFusion) | 14.5% | 8.1% | 5.6% | 3.4% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 2.90 | 2.93 | 0.0471 | 2.70 | 0.09 |
| Pipeline V2 (AspectWeighted) | 2.90 | 2.72 | 0.0299 | 3.38 | 0.09 |
| Pipeline V2 (AspectFusion) | 2.90 | 2.29 | 0.0661 | 3.39 | 0.10 |