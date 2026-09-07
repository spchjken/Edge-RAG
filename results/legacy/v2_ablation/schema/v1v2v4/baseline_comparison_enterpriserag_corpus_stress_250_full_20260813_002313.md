# Baseline Comparison Report — enterpriserag_corpus_stress_250_full

- **Queries:** 500 | **Corpus chunks:** 2044 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 85.0% | 87.2% | 88.6% | 89.6% |
| Pipeline V2 (AspectWeighted) | 82.2% | 85.4% | 87.8% | 89.2% |
| Pipeline V2 (AspectFusion) | 85.0% | 87.2% | 88.6% | 89.6% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 55.1% | 65.0% | 70.5% | 76.1% |
| Pipeline V2 (AspectWeighted) | 50.6% | 61.5% | 67.6% | 73.7% |
| Pipeline V2 (AspectFusion) | 55.1% | 65.0% | 70.5% | 76.1% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 18.8% | 11.1% | 8.0% | 5.2% |
| Pipeline V2 (AspectWeighted) | 17.3% | 10.5% | 7.7% | 5.0% |
| Pipeline V2 (AspectFusion) | 18.8% | 11.1% | 8.0% | 5.2% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 2.80 | 0.62 | 0.0079 | 1.97 | 0.09 |
| Pipeline V2 (AspectWeighted) | 2.80 | 0.60 | 0.0042 | 2.06 | 0.09 |
| Pipeline V2 (AspectFusion) | 2.80 | 0.70 | 0.0245 | 2.06 | 0.10 |