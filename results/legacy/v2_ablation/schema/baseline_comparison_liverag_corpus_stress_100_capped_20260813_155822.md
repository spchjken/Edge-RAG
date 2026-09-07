# Baseline Comparison Report — liverag_corpus_stress_100_capped

- **Queries:** 196 | **Corpus chunks:** 709 | **K Values:** [10, 20, 30, 50]


### 1. Strict Query Recall @ K (Hit Rate)

| System | Strict Recall@10 | Strict Recall@20 | Strict Recall@30 | Strict Recall@50 |
|--------|------------------|------------------|------------------|------------------|
| Pipeline V2 (AspectInject) | 97.4% | 98.0% | 98.0% | 98.0% |
| Pipeline V2 (DynamicAspectInject) | 96.9% | 97.4% | 98.0% | 98.0% |

### 2. Chunk Recall @ K

| System | Chunk Recall@10 | Chunk Recall@20 | Chunk Recall@30 | Chunk Recall@50 |
|--------|-----------------|-----------------|-----------------|-----------------|
| Pipeline V2 (AspectInject) | 81.1% | 84.5% | 85.3% | 86.1% |
| Pipeline V2 (DynamicAspectInject) | 80.9% | 84.1% | 85.1% | 86.1% |

### 3. Chunk Precision @ K

| System | Precision@10 | Precision@20 | Precision@30 | Precision@50 |
|--------|--------------|--------------|--------------|--------------|
| Pipeline V2 (AspectInject) | 20.6% | 10.7% | 7.2% | 4.4% |
| Pipeline V2 (DynamicAspectInject) | 20.6% | 10.7% | 7.2% | 4.4% |

### 4. Timing, Latency & Hardware Metrics

| System | Model Load (s) | Index Build TTI (s) | Avg Latency (s/query) | Python RAM (GB) | Peak VRAM (GB) |
|--------|----------------|---------------------|-----------------------|-----------------|----------------|
| Pipeline V2 (AspectInject) | 2.61 | 0.39 | 0.0012 | 1.91 | 0.09 |
| Pipeline V2 (DynamicAspectInject) | 2.61 | 0.36 | 0.0010 | 1.95 | 0.09 |