# Baseline Comparison Report — fused_stress_500

- **Queries:** 1084 | **Corpus chunks:** 17241 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 91.1% | 94.0% | 95.0% | 96.7% |
| Pipeline V2 (AspectWeighted) | 88.7% | 92.3% | 93.9% | 95.5% |
| Pipeline V2 (AspectFusion) | 91.1% | 94.0% | 95.0% | 96.7% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 76.8% | 85.8% | 88.3% | 91.1% |
| Pipeline V2 (AspectWeighted) | 73.2% | 83.4% | 86.7% | 89.6% |
| Pipeline V2 (AspectFusion) | 76.8% | 85.8% | 88.3% | 91.1% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 14.5% | 8.1% | 5.5% | 3.4% |
| Pipeline V2 (AspectWeighted) | 13.8% | 7.9% | 5.4% | 3.4% |
| Pipeline V2 (AspectFusion) | 14.5% | 8.1% | 5.5% | 3.4% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 0.00 | 15.28 | 0.0474 | 1.88 | 0.00 |
| Pipeline V2 (AspectWeighted) | 0.00 | 12.96 | 0.0293 | 2.61 | 0.00 |
| Pipeline V2 (AspectFusion) | 0.00 | 12.52 | 0.0755 | 2.62 | 0.10 |