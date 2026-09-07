# Baseline Comparison Report — fused_stress_50

- **Queries:** 378 | **Corpus chunks:** 1537 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 84.7% | 87.8% | 88.1% | 88.4% |
| Pipeline V2 (AspectWeighted) | 83.1% | 87.0% | 88.1% | 88.6% |
| Pipeline V2 (AspectFusion) | 83.6% | 87.3% | 88.1% | 88.4% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 85.2% | 93.9% | 95.3% | 96.5% |
| Pipeline V2 (AspectWeighted) | 82.7% | 91.3% | 92.7% | 95.3% |
| Pipeline V2 (AspectFusion) | 82.5% | 92.9% | 94.9% | 96.5% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 11.5% | 6.3% | 4.3% | 2.6% |
| Pipeline V2 (AspectWeighted) | 11.1% | 6.1% | 4.2% | 2.6% |
| Pipeline V2 (AspectFusion) | 11.1% | 6.2% | 4.3% | 2.6% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 0.00 | 4.21 | 0.0372 | 1.99 | 0.20 |
| Pipeline V2 (AspectWeighted) | 0.00 | 2.88 | 0.0345 | 2.25 | 0.26 |
| Pipeline V2 (AspectFusion) | 0.00 | 2.69 | 0.0542 | 2.32 | 0.26 |