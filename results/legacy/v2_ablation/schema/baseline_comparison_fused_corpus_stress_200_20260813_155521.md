# Baseline Comparison Report — fused_corpus_stress_200

- **Queries:** 569 | **Corpus chunks:** 6950 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 90.2% | 93.5% | 94.6% | 94.7% |
| Pipeline V2 (DynamicAspectInject) | 89.6% | 92.8% | 94.0% | 95.3% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 76.6% | 85.5% | 88.4% | 90.4% |
| Pipeline V2 (DynamicAspectInject) | 76.6% | 85.6% | 87.9% | 90.2% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 14.8% | 8.2% | 5.7% | 3.5% |
| Pipeline V2 (DynamicAspectInject) | 14.8% | 8.3% | 5.6% | 3.5% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 2.73 | 1.24 | 0.0167 | 2.20 | 0.09 |
| Pipeline V2 (DynamicAspectInject) | 2.73 | 1.22 | 0.0146 | 2.47 | 0.09 |