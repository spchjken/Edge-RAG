# Baseline Comparison Report — enterpriserag_corpus_stress_500_full

- **Queries:** 500 | **Corpus chunks:** 2467 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 84.2% | 86.4% | 87.6% | 89.0% |
| Pipeline V2 (DynamicAspectInject) | 83.8% | 85.8% | 87.6% | 89.8% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 54.8% | 64.4% | 69.8% | 74.8% |
| Pipeline V2 (DynamicAspectInject) | 52.7% | 62.0% | 67.9% | 73.7% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 18.7% | 11.0% | 7.9% | 5.1% |
| Pipeline V2 (DynamicAspectInject) | 18.0% | 10.6% | 7.7% | 5.0% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 2.65 | 0.67 | 0.0094 | 1.99 | 0.09 |
| Pipeline V2 (DynamicAspectInject) | 2.65 | 0.63 | 0.0087 | 2.09 | 0.09 |