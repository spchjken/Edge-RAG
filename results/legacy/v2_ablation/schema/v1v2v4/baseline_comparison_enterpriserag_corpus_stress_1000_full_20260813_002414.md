# Baseline Comparison Report — enterpriserag_corpus_stress_1000_full

- **Queries:** 500 | **Corpus chunks:** 3291 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 83.4% | 86.4% | 87.2% | 88.8% |
| Pipeline V2 (AspectWeighted) | 78.0% | 83.2% | 85.8% | 87.2% |
| Pipeline V2 (AspectFusion) | 83.4% | 86.4% | 87.2% | 88.8% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 52.8% | 63.3% | 68.0% | 73.6% |
| Pipeline V2 (AspectWeighted) | 47.7% | 58.1% | 64.6% | 70.6% |
| Pipeline V2 (AspectFusion) | 52.8% | 63.3% | 68.0% | 73.6% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 18.0% | 10.8% | 7.7% | 5.0% |
| Pipeline V2 (AspectWeighted) | 16.3% | 9.9% | 7.3% | 4.8% |
| Pipeline V2 (AspectFusion) | 18.0% | 10.8% | 7.7% | 5.0% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 2.75 | 0.78 | 0.0137 | 2.02 | 0.09 |
| Pipeline V2 (AspectWeighted) | 2.75 | 0.91 | 0.0069 | 2.16 | 0.09 |
| Pipeline V2 (AspectFusion) | 2.75 | 0.68 | 0.0305 | 2.16 | 0.10 |