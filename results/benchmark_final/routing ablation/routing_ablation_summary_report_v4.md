# Cascade Routing Threshold Sensitivity Study Report (Dense Vocabulary V4)
**Target Model:** `gemma4-e2b`  
**Scope:** All 9 Fintech Benchmark Corpora (`corpus_single_1..5`, `corpus_multi_1..3`, `corpus_stress_1`)  
**Tested Extractor:** `Dense Vocabulary V4`  
**Threshold Grid Matrix (9):** $\tau_{bypass} \in \{0.65, 0.75, 0.85\} \times \tau_{discard} \in \{0.05, 0.10, 0.15\}$  

## Overall Aggregated Sensitivity Summary (Dense Vocabulary V4 Across All 9 Corpora)
| Combination | Mean Pre-Rerank Recall | Mean Strict Recall | Mean Ext Recall | Mean Precision | Mean Bypass Hits | Mean Avg Latency | Mean Index Overhead |
|-------------|------------------------|--------------------|-----------------|----------------|------------------|------------------|---------------------|
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard010 | 88.8% | 76.0% | 88.1% | 19.3% | 1.1 | 2.39s | 0.2148s |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard010 | 88.8% | 75.8% | 88.0% | 19.4% | 0.0 | 2.39s | 0.2148s |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard010 | 88.8% | 76.4% | 87.6% | 19.3% | 4.8 | 2.40s | 0.2148s |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard005 | 89.6% | 74.7% | 87.3% | 18.2% | 0.0 | 2.64s | 0.2148s |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard005 | 89.6% | 75.1% | 87.3% | 18.4% | 1.1 | 2.67s | 0.2148s |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard005 | 89.6% | 75.4% | 86.9% | 18.1% | 4.8 | 3.06s | 0.2148s |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard015 | 81.1% | 72.7% | 85.0% | 22.4% | 1.1 | 2.26s | 0.2148s |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard015 | 81.1% | 72.4% | 84.9% | 22.6% | 0.0 | 2.26s | 0.2148s |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard015 | 81.1% | 71.5% | 84.5% | 22.5% | 4.8 | 2.30s | 0.2148s |

==========================================================================================

### Benchmark Corpus: `corpus_multi_1`
- **Result File:** `pipeline_run_gemma4-e2b_routing_v4_corpus_multi_1_20260802_102520.json`
- **Ground Truth:** `data/benchmarks/fintech/corpus_multi_1/final_benchmark_corpus_multi_1.json`

| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|-------------|----------------|------------|--------------|
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard010 | 78.5% | 63.2% | 86.2% | 15.5% | 10 | 57 | 2.43s | 0.2555s | 2.79 GB | 1.72 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard005 | 80.2% | 65.5% | 86.2% | 15.2% | 10 | 59 | 3.18s | 0.2555s | 2.79 GB | 1.82 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard010 | 78.5% | 60.9% | 85.1% | 15.7% | 3 | 62 | 2.39s | 0.2555s | 2.79 GB | 1.69 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard005 | 80.2% | 62.1% | 85.1% | 15.8% | 0 | 66 | 2.65s | 0.2555s | 2.79 GB | 1.68 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard005 | 80.2% | 64.4% | 85.1% | 15.6% | 3 | 65 | 2.70s | 0.2555s | 2.79 GB | 1.71 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard010 | 78.5% | 58.6% | 83.9% | 15.7% | 0 | 63 | 2.39s | 0.2555s | 2.79 GB | 1.66 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard015 | 73.3% | 65.5% | 82.8% | 19.2% | 3 | 66 | 2.29s | 0.2555s | 2.79 GB | 1.69 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard015 | 73.3% | 63.2% | 82.8% | 17.5% | 10 | 57 | 2.37s | 0.2555s | 2.79 GB | 1.71 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard015 | 73.3% | 63.2% | 81.6% | 19.1% | 0 | 67 | 2.30s | 0.2555s | 2.79 GB | 1.66 GB |


### Benchmark Corpus: `corpus_multi_2`
- **Result File:** `pipeline_run_gemma4-e2b_routing_v4_corpus_multi_2_20260802_105805.json`
- **Ground Truth:** `data/benchmarks/fintech/corpus_multi_2/final_benchmark_corpus_multi_2.json`

| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|-------------|----------------|------------|--------------|
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard005 | 84.0% | 75.0% | 91.3% | 17.7% | 0 | 80 | 2.60s | 0.2067s | 2.79 GB | 1.83 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard005 | 84.0% | 75.0% | 91.3% | 17.7% | 0 | 80 | 2.63s | 0.2067s | 2.79 GB | 1.83 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard005 | 84.0% | 75.0% | 91.3% | 18.0% | 8 | 76 | 3.10s | 0.2067s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard010 | 83.2% | 77.2% | 90.2% | 18.8% | 0 | 80 | 2.35s | 0.2067s | 2.79 GB | 1.83 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard010 | 83.2% | 76.1% | 89.1% | 18.6% | 0 | 80 | 2.35s | 0.2067s | 2.79 GB | 1.83 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard010 | 83.2% | 76.1% | 89.1% | 18.9% | 8 | 75 | 2.35s | 0.2067s | 2.79 GB | 1.83 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard015 | 76.8% | 72.8% | 85.9% | 19.9% | 8 | 69 | 2.40s | 0.2067s | 2.79 GB | 1.83 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard015 | 76.8% | 70.7% | 84.8% | 18.9% | 0 | 72 | 2.29s | 0.2067s | 2.79 GB | 1.83 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard015 | 76.8% | 70.7% | 84.8% | 18.8% | 0 | 72 | 2.29s | 0.2067s | 2.79 GB | 1.83 GB |


### Benchmark Corpus: `corpus_multi_3`
- **Result File:** `pipeline_run_gemma4-e2b_routing_v4_corpus_multi_3_20260802_113244.json`
- **Ground Truth:** `data/benchmarks/fintech/corpus_multi_3/final_benchmark_corpus_multi_3.json`

| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|-------------|----------------|------------|--------------|
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard005 | 83.5% | 74.0% | 88.5% | 16.7% | 0 | 80 | 2.65s | 0.1899s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard005 | 83.5% | 75.0% | 88.5% | 16.7% | 2 | 79 | 2.68s | 0.1899s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard005 | 83.5% | 75.0% | 88.5% | 16.6% | 3 | 78 | 3.19s | 0.1899s | 2.79 GB | 1.85 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard010 | 83.5% | 75.0% | 87.5% | 17.7% | 2 | 78 | 2.33s | 0.1899s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard010 | 83.5% | 74.0% | 87.5% | 17.8% | 0 | 79 | 2.33s | 0.1899s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard010 | 83.5% | 75.0% | 87.5% | 17.6% | 3 | 77 | 2.33s | 0.1899s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard015 | 75.2% | 67.7% | 80.2% | 18.3% | 2 | 71 | 2.37s | 0.1899s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard015 | 75.2% | 66.7% | 80.2% | 18.3% | 0 | 72 | 2.37s | 0.1899s | 2.79 GB | 1.85 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard015 | 75.2% | 66.7% | 79.2% | 18.1% | 3 | 69 | 2.38s | 0.1899s | 2.79 GB | 1.84 GB |


### Benchmark Corpus: `corpus_single_1`
- **Result File:** `pipeline_run_gemma4-e2b_routing_v4_corpus_single_1_20260802_113814.json`
- **Ground Truth:** `data/benchmarks/fintech/corpus_single_1/final_benchmark_corpus_single_1.json`

| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|-------------|----------------|------------|--------------|
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard015 | 87.5% | 68.8% | 93.8% | 21.1% | 5 | 11 | 2.10s | 0.2121s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard015 | 87.5% | 68.8% | 93.8% | 18.9% | 2 | 12 | 2.11s | 0.2121s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard015 | 87.5% | 68.8% | 93.8% | 18.7% | 0 | 14 | 2.12s | 0.2121s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard010 | 100.0% | 81.2% | 87.5% | 22.0% | 5 | 13 | 2.20s | 0.2121s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard010 | 100.0% | 81.2% | 87.5% | 20.3% | 2 | 14 | 2.23s | 0.2121s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard010 | 100.0% | 81.2% | 87.5% | 20.0% | 0 | 16 | 2.23s | 0.2121s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard005 | 100.0% | 75.0% | 87.5% | 19.2% | 0 | 15 | 2.43s | 0.2121s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard005 | 100.0% | 75.0% | 87.5% | 19.5% | 2 | 13 | 2.45s | 0.2121s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard005 | 100.0% | 81.2% | 87.5% | 22.0% | 5 | 13 | 2.70s | 0.2121s | 2.79 GB | 1.84 GB |


### Benchmark Corpus: `corpus_single_2`
- **Result File:** `pipeline_run_gemma4-e2b_routing_v4_corpus_single_2_20260802_114508.json`
- **Ground Truth:** `data/benchmarks/fintech/corpus_single_2/final_benchmark_corpus_single_2.json`

| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|-------------|----------------|------------|--------------|
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard010 | 95.7% | 75.0% | 85.0% | 23.3% | 0 | 20 | 2.24s | 0.1948s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard010 | 95.7% | 75.0% | 85.0% | 23.3% | 0 | 20 | 2.25s | 0.1948s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard010 | 95.7% | 75.0% | 85.0% | 23.3% | 0 | 20 | 2.25s | 0.1948s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard005 | 95.7% | 70.0% | 80.0% | 20.9% | 0 | 19 | 2.51s | 0.1948s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard005 | 95.7% | 70.0% | 80.0% | 20.9% | 0 | 19 | 2.51s | 0.1948s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard005 | 95.7% | 65.0% | 80.0% | 18.9% | 0 | 17 | 3.15s | 0.1948s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard015 | 82.6% | 65.0% | 75.0% | 24.6% | 0 | 17 | 1.98s | 0.1948s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard015 | 82.6% | 65.0% | 75.0% | 24.6% | 0 | 17 | 1.99s | 0.1948s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard015 | 82.6% | 65.0% | 75.0% | 25.0% | 0 | 17 | 2.07s | 0.1948s | 2.79 GB | 1.84 GB |


### Benchmark Corpus: `corpus_single_3`
- **Result File:** `pipeline_run_gemma4-e2b_routing_v4_corpus_single_3_20260802_115053.json`
- **Ground Truth:** `data/benchmarks/fintech/corpus_single_3/final_benchmark_corpus_single_3.json`

| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|-------------|----------------|------------|--------------|
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard015 | 87.0% | 81.2% | 87.5% | 25.4% | 1 | 14 | 2.13s | 0.1936s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard015 | 87.0% | 81.2% | 87.5% | 26.8% | 0 | 15 | 2.13s | 0.1936s | 2.79 GB | 1.85 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard015 | 87.0% | 81.2% | 87.5% | 28.1% | 2 | 14 | 2.18s | 0.1936s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard010 | 87.0% | 81.2% | 87.5% | 23.4% | 0 | 15 | 2.25s | 0.1936s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard010 | 87.0% | 81.2% | 87.5% | 22.7% | 1 | 14 | 2.27s | 0.1936s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard010 | 87.0% | 81.2% | 87.5% | 23.2% | 2 | 14 | 2.28s | 0.1936s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard005 | 91.3% | 81.2% | 87.5% | 20.9% | 0 | 14 | 2.55s | 0.1936s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard005 | 91.3% | 81.2% | 87.5% | 22.4% | 1 | 14 | 2.65s | 0.1936s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard005 | 91.3% | 81.2% | 87.5% | 20.8% | 2 | 13 | 3.12s | 0.1936s | 2.79 GB | 1.84 GB |


### Benchmark Corpus: `corpus_single_4`
- **Result File:** `pipeline_run_gemma4-e2b_routing_v4_corpus_single_4_20260802_115757.json`
- **Ground Truth:** `data/benchmarks/fintech/corpus_single_4/final_benchmark_corpus_single_4.json`

| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|-------------|----------------|------------|--------------|
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard015 | 84.0% | 80.0% | 95.0% | 26.6% | 0 | 17 | 2.01s | 0.1927s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard015 | 84.0% | 80.0% | 95.0% | 26.2% | 1 | 16 | 2.02s | 0.1927s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard010 | 92.0% | 75.0% | 95.0% | 18.0% | 0 | 16 | 2.27s | 0.1927s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard010 | 92.0% | 75.0% | 95.0% | 18.0% | 1 | 15 | 2.29s | 0.1927s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard015 | 84.0% | 75.0% | 90.0% | 23.2% | 1 | 15 | 2.06s | 0.1927s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard010 | 92.0% | 75.0% | 90.0% | 16.5% | 1 | 15 | 2.32s | 0.1927s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard005 | 92.0% | 75.0% | 90.0% | 16.2% | 0 | 16 | 2.61s | 0.1927s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard005 | 92.0% | 75.0% | 90.0% | 16.0% | 1 | 15 | 2.69s | 0.1927s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard005 | 92.0% | 75.0% | 85.0% | 15.7% | 1 | 15 | 3.08s | 0.1927s | 2.79 GB | 1.84 GB |


### Benchmark Corpus: `corpus_single_5`
- **Result File:** `pipeline_run_gemma4-e2b_routing_v4_corpus_single_5_20260802_120330.json`
- **Ground Truth:** `data/benchmarks/fintech/corpus_single_5/final_benchmark_corpus_single_5.json`

| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|-------------|----------------|------------|--------------|
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard010 | 100.0% | 87.5% | 93.8% | 20.5% | 0 | 15 | 2.24s | 0.1985s | 2.79 GB | 1.85 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard010 | 100.0% | 87.5% | 93.8% | 20.5% | 0 | 15 | 2.26s | 0.1985s | 2.79 GB | 1.85 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard010 | 100.0% | 87.5% | 93.8% | 20.0% | 0 | 15 | 2.26s | 0.1985s | 2.79 GB | 1.85 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard005 | 100.0% | 87.5% | 93.8% | 20.0% | 0 | 15 | 2.46s | 0.1985s | 2.79 GB | 1.85 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard005 | 100.0% | 87.5% | 93.8% | 19.7% | 0 | 15 | 2.47s | 0.1985s | 2.79 GB | 1.85 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard005 | 100.0% | 87.5% | 93.8% | 19.5% | 0 | 15 | 2.80s | 0.1985s | 2.79 GB | 1.84 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard015 | 90.5% | 87.5% | 87.5% | 32.0% | 0 | 16 | 2.08s | 0.1985s | 2.79 GB | 1.85 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard015 | 90.5% | 87.5% | 87.5% | 32.0% | 0 | 16 | 2.08s | 0.1985s | 2.79 GB | 1.85 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard015 | 90.5% | 81.2% | 87.5% | 31.2% | 0 | 15 | 2.14s | 0.1985s | 2.79 GB | 1.85 GB |


### Benchmark Corpus: `corpus_stress_1`
- **Result File:** `pipeline_run_gemma4-e2b_routing_v4_corpus_stress_1_20260802_140639.json`
- **Ground Truth:** `data/benchmarks/fintech/corpus_stress_1/final_benchmark_corpus_stress_1.json`

| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|-------------|----------------|------------|--------------|
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard010 | 79.8% | 72.9% | 82.0% | 17.1% | 14 | 229 | 3.17s | 0.2890s | 2.79 GB | 1.88 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard010 | 79.8% | 72.2% | 82.0% | 17.4% | 1 | 240 | 3.18s | 0.2890s | 2.79 GB | 1.88 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard010 | 79.8% | 72.2% | 82.0% | 17.5% | 0 | 241 | 3.19s | 0.2890s | 2.79 GB | 1.88 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard005 | 79.8% | 73.2% | 82.0% | 16.7% | 14 | 227 | 3.23s | 0.2890s | 2.79 GB | 1.87 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard005 | 79.8% | 72.5% | 82.0% | 16.9% | 1 | 238 | 3.23s | 0.2890s | 2.79 GB | 1.88 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard005 | 79.8% | 72.5% | 82.0% | 17.1% | 0 | 239 | 3.24s | 0.2890s | 2.79 GB | 1.88 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass065_Discard015 | 72.7% | 69.7% | 78.5% | 18.3% | 14 | 217 | 3.02s | 0.2890s | 2.79 GB | 1.88 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass075_Discard015 | 72.7% | 68.3% | 78.5% | 18.5% | 1 | 226 | 3.03s | 0.2890s | 2.79 GB | 1.88 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise_Bypass085_Discard015 | 72.7% | 68.3% | 78.5% | 18.6% | 0 | 227 | 3.04s | 0.2890s | 2.79 GB | 1.88 GB |

