# Baseline Comparison Report — fused_stress_200

- **Queries:** 569 | **Corpus chunks:** 6950 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 89.8% | 93.1% | 94.4% | 94.7% |
| Pipeline V2 (AspectWeighted) | 87.0% | 90.9% | 92.4% | 94.2% |
| Pipeline V2 (AspectFusion) | 89.5% | 93.0% | 94.4% | 94.7% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 76.0% | 85.2% | 88.1% | 90.2% |
| Pipeline V2 (AspectWeighted) | 71.4% | 82.7% | 85.8% | 88.9% |
| Pipeline V2 (AspectFusion) | 75.8% | 85.1% | 88.1% | 90.3% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 14.7% | 8.2% | 5.7% | 3.5% |
| Pipeline V2 (AspectWeighted) | 13.8% | 8.0% | 5.5% | 3.4% |
| Pipeline V2 (AspectFusion) | 14.6% | 8.2% | 5.7% | 3.5% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 0.00 | 6.35 | 0.0628 | 2.29 | 0.21 |
| Pipeline V2 (AspectWeighted) | 0.00 | 5.33 | 0.0557 | 2.82 | 0.27 |
| Pipeline V2 (AspectFusion) | 0.00 | 5.02 | 0.0827 | 2.88 | 0.27 |