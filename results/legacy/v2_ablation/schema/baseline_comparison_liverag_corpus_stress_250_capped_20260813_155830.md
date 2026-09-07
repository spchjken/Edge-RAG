# Baseline Comparison Report — liverag_corpus_stress_250_capped

- **Queries:** 196 | **Corpus chunks:** 1042 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 94.9% | 96.4% | 96.4% | 96.9% |
| Pipeline V2 (DynamicAspectInject) | 94.9% | 95.9% | 95.9% | 96.9% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 79.3% | 83.3% | 84.3% | 85.9% |
| Pipeline V2 (DynamicAspectInject) | 79.5% | 83.3% | 84.1% | 85.9% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 20.2% | 10.6% | 7.1% | 4.4% |
| Pipeline V2 (DynamicAspectInject) | 20.2% | 10.6% | 7.1% | 4.4% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 2.66 | 0.55 | 0.0015 | 1.93 | 0.09 |
| Pipeline V2 (DynamicAspectInject) | 2.66 | 0.52 | 0.0016 | 1.98 | 0.09 |