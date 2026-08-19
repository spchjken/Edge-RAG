# Baseline Comparison Report — liverag_corpus_stress_full_full

- **Queries:** 895 | **Corpus chunks:** 2102 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 93.9% | 95.4% | 96.0% | 96.1% |
| Pipeline V2 (DynamicAspectInject) | 93.7% | 95.2% | 96.0% | 96.2% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 74.6% | 79.3% | 80.5% | 82.0% |
| Pipeline V2 (DynamicAspectInject) | 74.6% | 79.2% | 80.7% | 82.1% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 18.9% | 10.0% | 6.8% | 4.2% |
| Pipeline V2 (DynamicAspectInject) | 18.9% | 10.0% | 6.8% | 4.2% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 2.68 | 0.73 | 0.0037 | 2.00 | 0.09 |
| Pipeline V2 (DynamicAspectInject) | 2.68 | 0.69 | 0.0038 | 2.11 | 0.09 |