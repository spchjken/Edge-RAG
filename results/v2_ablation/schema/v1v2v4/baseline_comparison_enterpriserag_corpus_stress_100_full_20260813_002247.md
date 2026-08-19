# Baseline Comparison Report — enterpriserag_corpus_stress_100_full

- **Queries:** 500 | **Corpus chunks:** 1783 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 85.6% | 88.2% | 89.2% | 90.8% |
| Pipeline V2 (AspectWeighted) | 82.8% | 86.0% | 87.8% | 89.2% |
| Pipeline V2 (AspectFusion) | 85.6% | 88.2% | 89.2% | 90.8% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 55.7% | 66.1% | 71.4% | 77.4% |
| Pipeline V2 (AspectWeighted) | 51.7% | 62.7% | 68.4% | 75.0% |
| Pipeline V2 (AspectFusion) | 55.7% | 66.1% | 71.4% | 77.4% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 19.0% | 11.3% | 8.1% | 5.3% |
| Pipeline V2 (AspectWeighted) | 17.6% | 10.7% | 7.8% | 5.1% |
| Pipeline V2 (AspectFusion) | 19.0% | 11.3% | 8.1% | 5.3% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 2.76 | 0.52 | 0.0067 | 1.95 | 0.09 |
| Pipeline V2 (AspectWeighted) | 2.76 | 0.53 | 0.0035 | 2.03 | 0.09 |
| Pipeline V2 (AspectFusion) | 2.76 | 0.65 | 0.0230 | 2.03 | 0.10 |