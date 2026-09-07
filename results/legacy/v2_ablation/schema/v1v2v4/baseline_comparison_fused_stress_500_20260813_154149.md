# Baseline Comparison Report — fused_stress_500

- **Queries:** 1084 | **Corpus chunks:** 17241 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 91.1% | 94.2% | 95.3% | 96.7% |
| Pipeline V2 (AspectWeighted) | 88.9% | 92.6% | 94.3% | 95.7% |
| Pipeline V2 (AspectFusion) | 91.1% | 94.2% | 95.3% | 96.7% |
| Pipeline V2 (DynamicAspectInject) | 90.3% | 93.7% | 94.6% | 96.2% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 77.1% | 85.8% | 88.5% | 91.1% |
| Pipeline V2 (AspectWeighted) | 73.9% | 83.6% | 87.1% | 89.9% |
| Pipeline V2 (AspectFusion) | 77.1% | 85.8% | 88.5% | 91.1% |
| Pipeline V2 (DynamicAspectInject) | 76.3% | 85.6% | 88.0% | 90.7% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 14.5% | 8.1% | 5.6% | 3.4% |
| Pipeline V2 (AspectWeighted) | 13.9% | 7.9% | 5.5% | 3.4% |
| Pipeline V2 (AspectFusion) | 14.5% | 8.1% | 5.6% | 3.4% |
| Pipeline V2 (DynamicAspectInject) | 14.4% | 8.1% | 5.5% | 3.4% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 3.10 | 2.80 | 0.0454 | 2.69 | 0.09 |
| Pipeline V2 (AspectWeighted) | 3.10 | 2.62 | 0.0280 | 3.37 | 0.09 |
| Pipeline V2 (AspectFusion) | 3.10 | 2.23 | 0.0667 | 3.39 | 0.10 |
| Pipeline V2 (DynamicAspectInject) | 3.10 | 2.77 | 0.0417 | 3.55 | 0.09 |