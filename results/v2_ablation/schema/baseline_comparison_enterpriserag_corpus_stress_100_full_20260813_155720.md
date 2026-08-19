# Baseline Comparison Report — enterpriserag_corpus_stress_100_full

- **Queries:** 500 | **Corpus chunks:** 1783 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 85.6% | 88.2% | 89.2% | 90.8% |
| Pipeline V2 (DynamicAspectInject) | 85.2% | 87.8% | 89.8% | 90.8% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 55.7% | 66.1% | 71.4% | 77.4% |
| Pipeline V2 (DynamicAspectInject) | 54.2% | 64.3% | 70.2% | 75.9% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 19.0% | 11.3% | 8.1% | 5.3% |
| Pipeline V2 (DynamicAspectInject) | 18.5% | 11.0% | 8.0% | 5.2% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 2.60 | 0.56 | 0.0065 | 1.95 | 0.09 |
| Pipeline V2 (DynamicAspectInject) | 2.60 | 0.54 | 0.0060 | 2.03 | 0.09 |