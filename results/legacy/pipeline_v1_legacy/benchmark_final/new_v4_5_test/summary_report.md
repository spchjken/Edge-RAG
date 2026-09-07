# Benchmark Evaluation: gemma4-e2b Corpus Multi 1
**Benchmark Dataset:** `data/benchmarks/fintech/corpus_multi_1/final_benchmark_corpus_multi_1.json`
**Results Log:** `results/benchmark_final/pipeline_run_gemma4-e2b_fintech_corpus_multi_1_20260731_145422.json`

| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM | System RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|-------------|----------------|------------|--------------|------------|
| AspectOnly_Dense_Vocab_V4_Cascade_Pointwise | 78.5% | 67.8% | 86.2% | 18.5% | 7 | 69 | 12.15s | 0.2594s | 2.79 GB | 1.57 GB | 12.64 GB |
| AspectOnly_Dense_Vocab_V5_Cascade_Pointwise | 79.3% | 69.0% | 85.1% | 19.5% | 6 | 72 | 11.66s | 0.0750s | 2.79 GB | 1.56 GB | 12.47 GB |
| AspectOnly_Dense_Vocab_V3_Cascade_Listwise | 80.2% | 60.9% | 83.9% | 11.5% | 7 | 53 | 17.42s | 14.7477s | 2.79 GB | 1.86 GB | 9.79 GB |
| AspectOnly_Dense_Vocab_V3_Cascade_Pointwise | 80.2% | 70.1% | 83.9% | 19.5% | 7 | 71 | 26.48s | 14.7477s | 2.79 GB | 1.64 GB | 13.35 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise | 78.5% | 60.9% | 82.8% | 11.8% | 7 | 56 | 2.95s | 0.2594s | 2.79 GB | 1.61 GB | 13.21 GB |
| AspectOnly_Dense_Vocab_V5_Cascade_Listwise | 79.3% | 62.1% | 81.6% | 13.0% | 6 | 61 | 2.68s | 0.0750s | 2.79 GB | 1.57 GB | 12.77 GB |
| AspectOnly_Dense_Vocab_V4_Adaptive_Listwise | 67.2% | 60.9% | 80.5% | 13.6% | 27 | 37 | 2.31s | 0.2594s | 2.79 GB | 1.58 GB | 13.19 GB |
| AspectOnly_Dense_Vocab_V5_Adaptive_Listwise | 67.2% | 63.2% | 77.0% | 15.2% | 24 | 44 | 2.06s | 0.0750s | 2.79 GB | 1.57 GB | 13.01 GB |
| AspectOnly_Dense_Vocab_V3_Adaptive_Listwise | 69.0% | 52.9% | 72.4% | 13.6% | 25 | 34 | 16.68s | 14.7477s | 2.79 GB | 1.74 GB | 13.14 GB |

====================================================================================================

# Benchmark Evaluation: gemma4-e2b Corpus Multi 2
**Benchmark Dataset:** `data/benchmarks/fintech/corpus_multi_2/final_benchmark_corpus_multi_2.json`
**Results Log:** `results/benchmark_final/pipeline_run_gemma4-e2b_fintech_corpus_multi_2_20260731_160722.json`

| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM | System RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|-------------|----------------|------------|--------------|------------|
| AspectOnly_Dense_Vocab_V5_Cascade_Listwise | 82.4% | 76.1% | 90.2% | 13.2% | 3 | 75 | 2.75s | 0.0902s | 2.79 GB | 1.80 GB | 12.17 GB |
| AspectOnly_Dense_Vocab_V5_Cascade_Pointwise | 82.4% | 75.0% | 87.0% | 19.3% | 3 | 83 | 12.01s | 0.0902s | 2.79 GB | 1.80 GB | 12.07 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Pointwise | 83.2% | 76.1% | 87.0% | 19.3% | 2 | 86 | 12.53s | 0.4870s | 2.79 GB | 1.80 GB | 12.05 GB |
| AspectOnly_Dense_Vocab_V3_Cascade_Pointwise | 83.2% | 76.1% | 85.9% | 19.9% | 0 | 88 | 26.78s | 14.8797s | 2.79 GB | 1.86 GB | 12.08 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise | 83.2% | 71.7% | 84.8% | 12.8% | 2 | 72 | 3.19s | 0.4870s | 2.79 GB | 1.86 GB | 12.26 GB |
| AspectOnly_Dense_Vocab_V3_Adaptive_Listwise | 68.8% | 65.2% | 80.4% | 16.1% | 14 | 53 | 16.97s | 14.8797s | 2.79 GB | 1.87 GB | 12.78 GB |
| AspectOnly_Dense_Vocab_V3_Cascade_Listwise | 83.2% | 63.0% | 79.3% | 12.4% | 0 | 67 | 17.57s | 14.8797s | 2.79 GB | 1.89 GB | 12.61 GB |
| AspectOnly_Dense_Vocab_V4_Adaptive_Listwise | 67.2% | 64.1% | 78.3% | 14.9% | 16 | 52 | 2.64s | 0.4870s | 2.79 GB | 1.81 GB | 12.72 GB |
| AspectOnly_Dense_Vocab_V5_Adaptive_Listwise | 68.0% | 60.9% | 76.1% | 15.0% | 14 | 52 | 2.18s | 0.0902s | 2.79 GB | 1.80 GB | 12.52 GB |

====================================================================================================

# Benchmark Evaluation: gemma4-e2b Corpus Multi 3
**Benchmark Dataset:** `data/benchmarks/fintech/corpus_multi_3/final_benchmark_corpus_multi_3.json`
**Results Log:** `results/benchmark_final/pipeline_run_gemma4-e2b_fintech_corpus_multi_3_20260731_172634.json`

| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM | System RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|-------------|----------------|------------|--------------|------------|
| AspectOnly_Dense_Vocab_V5_Cascade_Pointwise | 82.6% | 75.0% | 86.5% | 17.5% | 1 | 82 | 12.35s | 0.0894s | 2.79 GB | 1.73 GB | 11.96 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Pointwise | 83.5% | 76.0% | 86.5% | 17.4% | 2 | 83 | 12.57s | 0.2289s | 2.79 GB | 1.75 GB | 11.98 GB |
| AspectOnly_Dense_Vocab_V3_Cascade_Listwise | 80.2% | 67.7% | 84.4% | 13.2% | 0 | 74 | 18.13s | 15.2507s | 2.79 GB | 1.86 GB | 12.37 GB |
| AspectOnly_Dense_Vocab_V3_Cascade_Pointwise | 80.2% | 74.0% | 84.4% | 17.1% | 0 | 81 | 27.87s | 15.2507s | 2.79 GB | 1.76 GB | 12.05 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise | 83.5% | 67.7% | 83.3% | 12.9% | 2 | 72 | 3.05s | 0.2289s | 2.79 GB | 1.76 GB | 12.07 GB |
| AspectOnly_Dense_Vocab_V5_Cascade_Listwise | 82.6% | 67.7% | 82.3% | 13.3% | 1 | 74 | 2.88s | 0.0894s | 2.79 GB | 1.75 GB | 12.02 GB |
| AspectOnly_Dense_Vocab_V5_Adaptive_Listwise | 64.5% | 61.5% | 76.0% | 9.8% | 18 | 49 | 2.37s | 0.0894s | 2.79 GB | 1.74 GB | 12.75 GB |
| AspectOnly_Dense_Vocab_V4_Adaptive_Listwise | 64.5% | 62.5% | 74.0% | 10.0% | 24 | 44 | 2.53s | 0.2289s | 2.79 GB | 1.75 GB | 12.87 GB |
| AspectOnly_Dense_Vocab_V3_Adaptive_Listwise | 59.5% | 51.0% | 69.8% | 8.5% | 17 | 38 | 17.50s | 15.2507s | 2.79 GB | 1.78 GB | 12.80 GB |

====================================================================================================

# Benchmark Evaluation: gemma4-e2b Corpus Single 1
**Benchmark Dataset:** `data/benchmarks/fintech/corpus_single_1/final_benchmark_corpus_single_1.json`
**Results Log:** `results/benchmark_final/pipeline_run_gemma4-e2b_fintech_corpus_single_1_20260731_173633.json`

| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM | System RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|-------------|----------------|------------|--------------|------------|
| AspectOnly_Dense_Vocab_V5_Cascade_Pointwise | 100.0% | 87.5% | 93.8% | 18.7% | 2 | 21 | 8.31s | 0.0507s | 2.79 GB | 1.84 GB | 11.98 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Pointwise | 100.0% | 87.5% | 93.8% | 18.5% | 2 | 21 | 8.77s | 0.2220s | 2.79 GB | 1.84 GB | 11.98 GB |
| AspectOnly_Dense_Vocab_V3_Cascade_Pointwise | 100.0% | 87.5% | 93.8% | 18.0% | 0 | 23 | 22.03s | 13.3393s | 2.79 GB | 1.84 GB | 11.98 GB |
| AspectOnly_Dense_Vocab_V5_Adaptive_Listwise | 83.3% | 68.8% | 87.5% | 16.7% | 10 | 8 | 1.56s | 0.0507s | 2.79 GB | 1.84 GB | 11.98 GB |
| AspectOnly_Dense_Vocab_V4_Adaptive_Listwise | 83.3% | 68.8% | 87.5% | 16.8% | 10 | 8 | 1.94s | 0.2220s | 2.79 GB | 1.84 GB | 11.98 GB |
| AspectOnly_Dense_Vocab_V5_Cascade_Listwise | 100.0% | 81.2% | 87.5% | 16.7% | 2 | 13 | 2.05s | 0.0507s | 2.79 GB | 1.84 GB | 11.98 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise | 100.0% | 75.0% | 81.2% | 14.9% | 2 | 12 | 2.58s | 0.2220s | 2.79 GB | 1.84 GB | 11.98 GB |
| AspectOnly_Dense_Vocab_V3_Cascade_Listwise | 100.0% | 75.0% | 81.2% | 15.6% | 0 | 14 | 15.80s | 13.3393s | 2.79 GB | 1.84 GB | 11.97 GB |
| AspectOnly_Dense_Vocab_V3_Adaptive_Listwise | 87.5% | 56.2% | 68.8% | 17.3% | 8 | 9 | 14.96s | 13.3393s | 2.79 GB | 1.84 GB | 11.97 GB |

====================================================================================================

# Benchmark Evaluation: gemma4-e2b Corpus Single 2
**Benchmark Dataset:** `data/benchmarks/fintech/corpus_single_2/final_benchmark_corpus_single_2.json`
**Results Log:** `results/benchmark_final/pipeline_run_gemma4-e2b_fintech_corpus_single_2_20260731_175114.json`

| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM | System RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|-------------|----------------|------------|--------------|------------|
| AspectOnly_Dense_Vocab_V5_Cascade_Listwise | 95.7% | 70.0% | 75.0% | 16.7% | 0 | 18 | 2.35s | 0.0581s | 2.79 GB | 1.85 GB | 12.15 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise | 95.7% | 65.0% | 75.0% | 14.7% | 0 | 17 | 2.82s | 0.2031s | 2.79 GB | 1.85 GB | 12.05 GB |
| AspectOnly_Dense_Vocab_V5_Cascade_Pointwise | 95.7% | 70.0% | 75.0% | 17.4% | 0 | 19 | 10.80s | 0.0581s | 2.79 GB | 1.85 GB | 12.07 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Pointwise | 95.7% | 75.0% | 75.0% | 17.7% | 0 | 20 | 11.08s | 0.2031s | 2.79 GB | 1.85 GB | 12.16 GB |
| AspectOnly_Dense_Vocab_V3_Cascade_Listwise | 95.7% | 70.0% | 75.0% | 15.3% | 1 | 17 | 16.30s | 13.6736s | 2.79 GB | 1.85 GB | 12.03 GB |
| AspectOnly_Dense_Vocab_V3_Cascade_Pointwise | 95.7% | 65.0% | 75.0% | 16.1% | 1 | 17 | 24.35s | 13.6736s | 2.79 GB | 1.85 GB | 12.05 GB |
| AspectOnly_Dense_Vocab_V5_Adaptive_Listwise | 73.9% | 60.0% | 70.0% | 16.3% | 4 | 12 | 1.76s | 0.0581s | 2.79 GB | 1.85 GB | 12.11 GB |
| AspectOnly_Dense_Vocab_V4_Adaptive_Listwise | 78.3% | 60.0% | 65.0% | 17.0% | 4 | 12 | 2.13s | 0.2031s | 2.79 GB | 1.85 GB | 12.11 GB |
| AspectOnly_Dense_Vocab_V3_Adaptive_Listwise | 69.6% | 55.0% | 65.0% | 15.6% | 4 | 11 | 15.54s | 13.6736s | 2.79 GB | 1.85 GB | 12.05 GB |

====================================================================================================

# Benchmark Evaluation: gemma4-e2b Corpus Single 3
**Benchmark Dataset:** `data/benchmarks/fintech/corpus_single_3/final_benchmark_corpus_single_3.json`
**Results Log:** `results/benchmark_final/pipeline_run_gemma4-e2b_fintech_corpus_single_3_20260731_180307.json`

| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM | System RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|-------------|----------------|------------|--------------|------------|
| AspectOnly_Dense_Vocab_V5_Cascade_Pointwise | 87.0% | 81.2% | 81.2% | 28.4% | 1 | 18 | 10.16s | 0.0545s | 2.79 GB | 1.86 GB | 12.18 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Pointwise | 87.0% | 81.2% | 81.2% | 26.8% | 1 | 18 | 10.53s | 0.2026s | 2.79 GB | 1.86 GB | 12.26 GB |
| AspectOnly_Dense_Vocab_V3_Cascade_Pointwise | 91.3% | 81.2% | 81.2% | 27.5% | 1 | 18 | 24.46s | 13.3496s | 2.79 GB | 1.86 GB | 12.10 GB |
| AspectOnly_Dense_Vocab_V3_Adaptive_Listwise | 87.0% | 62.5% | 75.0% | 11.4% | 10 | 3 | 15.53s | 13.3496s | 2.79 GB | 1.86 GB | 12.13 GB |
| AspectOnly_Dense_Vocab_V3_Cascade_Listwise | 91.3% | 56.2% | 75.0% | 11.4% | 1 | 9 | 16.20s | 13.3496s | 2.79 GB | 1.86 GB | 12.10 GB |
| AspectOnly_Dense_Vocab_V5_Adaptive_Listwise | 82.6% | 62.5% | 62.5% | 12.5% | 9 | 4 | 1.82s | 0.0545s | 2.79 GB | 1.86 GB | 12.23 GB |
| AspectOnly_Dense_Vocab_V4_Adaptive_Listwise | 82.6% | 62.5% | 62.5% | 11.9% | 9 | 3 | 2.34s | 0.2026s | 2.79 GB | 1.86 GB | 12.17 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise | 87.0% | 56.2% | 62.5% | 14.7% | 1 | 10 | 2.93s | 0.2026s | 2.79 GB | 1.86 GB | 12.09 GB |
| AspectOnly_Dense_Vocab_V5_Cascade_Listwise | 87.0% | 56.2% | 56.2% | 13.8% | 1 | 10 | 2.19s | 0.0545s | 2.79 GB | 1.86 GB | 12.26 GB |

====================================================================================================

# Benchmark Evaluation: gemma4-e2b Corpus Single 4
**Benchmark Dataset:** `data/benchmarks/fintech/corpus_single_4/final_benchmark_corpus_single_4.json`
**Results Log:** `results/benchmark_final/pipeline_run_gemma4-e2b_fintech_corpus_single_4_20260731_181732.json`

| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM | System RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|-------------|----------------|------------|--------------|------------|
| AspectOnly_Dense_Vocab_V5_Cascade_Listwise | 92.0% | 80.0% | 95.0% | 12.8% | 1 | 15 | 2.11s | 0.0518s | 2.79 GB | 1.85 GB | 12.25 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise | 92.0% | 80.0% | 95.0% | 12.6% | 1 | 15 | 2.82s | 0.2051s | 2.79 GB | 1.86 GB | 12.12 GB |
| AspectOnly_Dense_Vocab_V5_Cascade_Pointwise | 92.0% | 80.0% | 90.0% | 20.9% | 1 | 18 | 10.24s | 0.0518s | 2.79 GB | 1.85 GB | 12.10 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Pointwise | 92.0% | 80.0% | 90.0% | 20.9% | 1 | 18 | 10.64s | 0.2051s | 2.79 GB | 1.85 GB | 12.25 GB |
| AspectOnly_Dense_Vocab_V3_Cascade_Pointwise | 92.0% | 75.0% | 90.0% | 18.0% | 1 | 17 | 23.75s | 12.9275s | 2.79 GB | 1.86 GB | 12.17 GB |
| AspectOnly_Dense_Vocab_V3_Cascade_Listwise | 92.0% | 70.0% | 85.0% | 11.8% | 1 | 14 | 15.57s | 12.9275s | 2.79 GB | 1.85 GB | 12.19 GB |
| AspectOnly_Dense_Vocab_V5_Adaptive_Listwise | 76.0% | 65.0% | 80.0% | 15.5% | 5 | 11 | 1.68s | 0.0518s | 2.79 GB | 1.85 GB | 12.20 GB |
| AspectOnly_Dense_Vocab_V4_Adaptive_Listwise | 76.0% | 65.0% | 80.0% | 15.1% | 5 | 11 | 2.30s | 0.2051s | 2.79 GB | 1.86 GB | 12.16 GB |
| AspectOnly_Dense_Vocab_V3_Adaptive_Listwise | 68.0% | 60.0% | 75.0% | 14.7% | 5 | 9 | 14.88s | 12.9275s | 2.79 GB | 1.85 GB | 12.20 GB |

====================================================================================================

# Benchmark Evaluation: gemma4-e2b Corpus Single 5
**Benchmark Dataset:** `data/benchmarks/fintech/corpus_single_5/final_benchmark_corpus_single_5.json`
**Results Log:** `results/benchmark_final/pipeline_run_gemma4-e2b_fintech_corpus_single_5_20260731_182904.json`

| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM | System RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|-------------|----------------|------------|--------------|------------|
| AspectOnly_Dense_Vocab_V3_Cascade_Listwise | 100.0% | 81.2% | 93.8% | 16.7% | 0 | 16 | 15.97s | 13.4629s | 2.79 GB | 1.85 GB | 12.13 GB |
| AspectOnly_Dense_Vocab_V3_Cascade_Pointwise | 100.0% | 93.8% | 93.8% | 18.6% | 0 | 18 | 24.22s | 13.4629s | 2.79 GB | 1.86 GB | 12.16 GB |
| AspectOnly_Dense_Vocab_V5_Cascade_Listwise | 100.0% | 81.2% | 87.5% | 16.1% | 0 | 14 | 2.11s | 0.0569s | 2.79 GB | 1.86 GB | 12.13 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise | 100.0% | 81.2% | 87.5% | 15.4% | 0 | 14 | 2.58s | 0.1991s | 2.79 GB | 1.86 GB | 12.15 GB |
| AspectOnly_Dense_Vocab_V5_Cascade_Pointwise | 100.0% | 81.2% | 87.5% | 17.2% | 0 | 16 | 10.36s | 0.0569s | 2.79 GB | 1.86 GB | 12.13 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Pointwise | 100.0% | 81.2% | 87.5% | 16.7% | 0 | 16 | 10.76s | 0.1991s | 2.79 GB | 1.86 GB | 12.14 GB |
| AspectOnly_Dense_Vocab_V3_Adaptive_Listwise | 76.2% | 81.2% | 87.5% | 21.4% | 1 | 14 | 15.47s | 13.4629s | 2.79 GB | 1.86 GB | 12.16 GB |
| AspectOnly_Dense_Vocab_V4_Adaptive_Listwise | 81.0% | 81.2% | 81.2% | 27.3% | 0 | 15 | 1.92s | 0.1991s | 2.79 GB | 1.86 GB | 12.14 GB |
| AspectOnly_Dense_Vocab_V5_Adaptive_Listwise | 81.0% | 75.0% | 75.0% | 25.9% | 0 | 14 | 1.58s | 0.0569s | 2.79 GB | 1.86 GB | 12.13 GB |

====================================================================================================

# Benchmark Evaluation: gemma4-e2b Corpus Stress 1
**Benchmark Dataset:** `data/benchmarks/fintech/corpus_stress_1/final_benchmark_corpus_stress_1.json`
**Results Log:** `results/benchmark_final/pipeline_run_gemma4-e2b_fintech_corpus_stress_1_20260731_222122.json`

| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Bypass Hits | Rerank Rescues | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM | System RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|-------------|----------------|------------|--------------|------------|
| AspectOnly_Dense_Vocab_V5_Cascade_Listwise | 79.0% | 63.4% | 79.9% | 12.3% | 6 | 204 | 2.89s | 0.1427s | 2.79 GB | 1.57 GB | 12.43 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise | 79.8% | 66.5% | 78.9% | 12.4% | 10 | 209 | 3.09s | 0.2721s | 2.79 GB | 1.65 GB | 12.56 GB |
| AspectOnly_Dense_Vocab_V3_Cascade_Listwise | 77.9% | 63.7% | 77.8% | 12.3% | 5 | 199 | 21.29s | 18.5926s | 2.79 GB | 1.89 GB | 12.74 GB |
| AspectOnly_Dense_Vocab_V5_Cascade_Pointwise | 79.0% | 68.7% | 75.4% | 18.3% | 6 | 224 | 12.64s | 0.1427s | 2.79 GB | 1.54 GB | 12.25 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Pointwise | 79.8% | 67.6% | 75.0% | 17.5% | 10 | 218 | 13.19s | 0.2721s | 2.79 GB | 1.63 GB | 12.38 GB |
| AspectOnly_Dense_Vocab_V3_Cascade_Pointwise | 77.9% | 67.6% | 75.0% | 18.8% | 5 | 221 | 30.78s | 18.5926s | 2.79 GB | 1.77 GB | 12.56 GB |
| AspectOnly_Dense_Vocab_V4_Adaptive_Listwise | 61.5% | 56.7% | 67.6% | 5.5% | 42 | 141 | 2.45s | 0.2721s | 2.79 GB | 1.63 GB | 12.38 GB |
| AspectOnly_Dense_Vocab_V5_Adaptive_Listwise | 60.9% | 55.3% | 65.8% | 5.4% | 43 | 135 | 2.27s | 0.1427s | 2.79 GB | 1.55 GB | 12.30 GB |
| AspectOnly_Dense_Vocab_V3_Adaptive_Listwise | 57.6% | 55.3% | 65.1% | 5.7% | 44 | 136 | 20.63s | 18.5926s | 2.79 GB | 1.84 GB | 12.76 GB |

====================================================================================================

# Overall Aggregated Performance Summary (Across All Corpora)
| Combination | Pre-Rerank Recall | Strict Recall | Ext Recall | Precision | Avg Latency | Index Overhead | Model VRAM | Pipeline RAM |
|-------------|-------------------|---------------|------------|-----------|-------------|----------------|------------|--------------|
| AspectOnly_Dense_Vocab_V3_Cascade_Pointwise | 88.9% | 76.7% | 84.8% | 19.3% | 25.64s | 14.4693s | 2.79 GB | 1.81 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Pointwise | 88.8% | 77.0% | 84.7% | 19.3% | 11.36s | 0.2533s | 2.79 GB | 1.78 GB |
| AspectOnly_Dense_Vocab_V5_Cascade_Pointwise | 88.7% | 76.4% | 84.6% | 19.7% | 10.95s | 0.0744s | 2.79 GB | 1.77 GB |
| AspectOnly_Dense_Vocab_V3_Cascade_Listwise | 88.9% | 67.5% | 81.7% | 13.3% | 17.14s | 14.4693s | 2.79 GB | 1.86 GB |
| AspectOnly_Dense_Vocab_V5_Cascade_Listwise | 88.7% | 70.9% | 81.7% | 14.2% | 2.44s | 0.0744s | 2.79 GB | 1.77 GB |
| AspectOnly_Dense_Vocab_V4_Cascade_Listwise | 88.8% | 69.4% | 81.2% | 13.6% | 2.89s | 0.2533s | 2.79 GB | 1.79 GB |
| AspectOnly_Dense_Vocab_V4_Adaptive_Listwise | 73.5% | 64.6% | 75.2% | 14.7% | 2.29s | 0.2533s | 2.79 GB | 1.78 GB |
| AspectOnly_Dense_Vocab_V5_Adaptive_Listwise | 73.0% | 63.6% | 74.4% | 14.7% | 1.92s | 0.0744s | 2.79 GB | 1.77 GB |
| AspectOnly_Dense_Vocab_V3_Adaptive_Listwise | 71.5% | 59.9% | 73.2% | 13.8% | 16.46s | 14.4693s | 2.79 GB | 1.83 GB |