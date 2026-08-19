# Baseline Comparison Report — fused_stress_200

- **Queries:** 569 | **Corpus chunks:** 6950 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 89.8% | 93.1% | 94.6% | 94.9% |
| Pipeline V2 (AspectWeighted) | 87.2% | 91.7% | 93.1% | 94.6% |
| Pipeline V2 (AspectFusion) | 89.8% | 93.1% | 94.6% | 94.9% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 76.4% | 85.4% | 88.2% | 90.4% |
| Pipeline V2 (AspectWeighted) | 72.4% | 84.0% | 87.0% | 89.3% |
| Pipeline V2 (AspectFusion) | 76.4% | 85.4% | 88.2% | 90.4% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 14.7% | 8.2% | 5.7% | 3.5% |
| Pipeline V2 (AspectWeighted) | 14.0% | 8.1% | 5.6% | 3.4% |
| Pipeline V2 (AspectFusion) | 14.7% | 8.2% | 5.7% | 3.5% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 0.00 | 6.97 | 0.0171 | 1.35 | 0.00 |
| Pipeline V2 (AspectWeighted) | 0.00 | 5.15 | 0.0099 | 1.69 | 0.00 |
| Pipeline V2 (AspectFusion) | 0.00 | 4.73 | 0.0363 | 1.69 | 0.10 |