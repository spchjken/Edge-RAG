# Baseline Comparison Report — fused_corpus_stress_50

- **Queries:** 378 | **Corpus chunks:** 1537 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 85.7% | 87.6% | 87.8% | 88.9% |
| Pipeline V2 (AspectWeighted) | 84.9% | 88.1% | 88.4% | 88.4% |
| Pipeline V2 (AspectFusion) | 85.7% | 87.6% | 87.8% | 88.9% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 86.4% | 93.7% | 94.7% | 97.0% |
| Pipeline V2 (AspectWeighted) | 84.6% | 93.1% | 94.7% | 96.5% |
| Pipeline V2 (AspectFusion) | 86.4% | 93.7% | 94.7% | 97.0% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 11.6% | 6.3% | 4.2% | 2.6% |
| Pipeline V2 (AspectWeighted) | 11.4% | 6.3% | 4.2% | 2.6% |
| Pipeline V2 (AspectFusion) | 11.6% | 6.3% | 4.2% | 2.6% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 2.95 | 0.52 | 0.0032 | 1.94 | 0.09 |
| Pipeline V2 (AspectWeighted) | 2.95 | 0.51 | 0.0020 | 2.00 | 0.09 |
| Pipeline V2 (AspectFusion) | 2.95 | 0.46 | 0.0187 | 2.00 | 0.10 |