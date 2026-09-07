# Baseline Comparison Report — fused_stress_50

- **Queries:** 378 | **Corpus chunks:** 1537 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 85.2% | 87.8% | 88.1% | 88.6% |
| Pipeline V2 (AspectWeighted) | 84.4% | 87.8% | 88.4% | 88.6% |
| Pipeline V2 (AspectFusion) | 85.2% | 87.8% | 88.1% | 88.6% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 85.6% | 93.7% | 95.3% | 96.9% |
| Pipeline V2 (AspectWeighted) | 83.9% | 92.1% | 94.3% | 96.7% |
| Pipeline V2 (AspectFusion) | 85.6% | 93.7% | 95.3% | 96.9% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 11.5% | 6.3% | 4.3% | 2.6% |
| Pipeline V2 (AspectWeighted) | 11.3% | 6.2% | 4.2% | 2.6% |
| Pipeline V2 (AspectFusion) | 11.5% | 6.3% | 4.3% | 2.6% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 0.00 | 3.30 | 0.0032 | 1.08 | 0.00 |
| Pipeline V2 (AspectWeighted) | 0.00 | 1.07 | 0.0019 | 1.16 | 0.00 |
| Pipeline V2 (AspectFusion) | 0.00 | 0.99 | 0.0229 | 1.16 | 0.10 |