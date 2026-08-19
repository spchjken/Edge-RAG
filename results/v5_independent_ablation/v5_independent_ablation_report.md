# Dense Vocabulary V5 Independent Ablation Summary Report

This report summarizes the **1D Independent Sweeps** for V5 Query Expansion across benchmarks.

## Benchmark Corpus: EnterpriseRAG (Corporate Docs)
- **Log File:** `pipeline_run_enterpriserag_20260805_182059.json`

### Retrieval Quality Metrics

| Combination | N_vocab | IDF Filter | Pre-Rerank Recall | Pre-Rerank Hit Rate | Chunk Recall | Chunk Precision | Strict Recall@10 | Ext Recall@10 | Micro Rerank Recall | Macro Rerank Recall |
|-------------|---------|------------|-------------------|---------------------|--------------|-----------------|------------------|---------------|---------------------|---------------------|
| V5_Nvocab_100 | 100 | Median | 54.2% | 77.0% | 43.6% | 27.2% | 74.0% | 74.0% | 80.2% | 86.5% |
| V5_Nvocab_250 | 250 | Median | 61.0% | 83.0% | 46.6% | 29.3% | 81.0% | 81.0% | 76.4% | 86.0% |
| V5_Nvocab_500 | 500 | Median | 57.2% | 79.0% | 45.3% | 30.4% | 76.0% | 76.0% | 79.1% | 86.1% |
| V5_Nvocab_1000 | 1000 | Median | 60.2% | 80.0% | 43.6% | 33.6% | 77.0% | 77.0% | 72.5% | 82.1% |
| V5_Nvocab_1500 | 1500 | Median | 60.2% | 84.0% | 44.1% | 33.0% | 77.0% | 77.0% | 73.2% | 81.0% |
| V5_Nvocab_2000 | 2000 | Median | 59.7% | 85.0% | 42.8% | 32.4% | 78.0% | 78.0% | 71.4% | 80.3% |
| V5_IDFFilter_None | 1000 | None | 62.7% | 84.0% | 48.3% | 30.6% | 79.0% | 79.0% | 76.9% | 83.8% |
| V5_IDFFilter_Top-Quartile | 1000 | Top-Quartile | 58.9% | 83.0% | 43.6% | 33.4% | 77.0% | 77.0% | 74.1% | 82.6% |

### Timing & Speed Metrics

| Combination | Prebuild (s) | Per-Doc (ms) | Avg QE (s) | Avg Search (s) | Avg Route (s) | Avg Rerank (s) | Avg Total (s) |
|-------------|--------------|--------------|------------|----------------|---------------|----------------|---------------|
| V5_Nvocab_100 | 0.5447 | 1.65 | 0.6884 | 0.0104 | 0.0005 | 2.4291 | 3.1287 |
| V5_Nvocab_250 | 0.0946 | 0.29 | 0.6803 | 0.0096 | 0.0004 | 2.1227 | 2.8132 |
| V5_Nvocab_500 | 0.0913 | 0.28 | 0.6831 | 0.0096 | 0.0003 | 1.9147 | 2.6079 |
| V5_Nvocab_1000 | 0.1069 | 0.32 | 0.6855 | 0.0092 | 0.0002 | 1.6005 | 2.2957 |
| V5_Nvocab_1500 | 0.1223 | 0.37 | 0.6919 | 0.0095 | 0.0003 | 1.6958 | 2.3978 |
| V5_Nvocab_2000 | 0.1395 | 0.42 | 0.6967 | 0.0094 | 0.0003 | 1.5031 | 2.2098 |
| V5_IDFFilter_None | 0.1048 | 0.32 | 0.6872 | 0.0095 | 0.0003 | 1.9614 | 2.6587 |
| V5_IDFFilter_Top-Quartile | 0.1080 | 0.33 | 0.6859 | 0.0096 | 0.0003 | 1.7663 | 2.4624 |

### Memory & Hardware Metrics

| Combination | Python RAM (GB) | Peak VRAM (GB) |
|-------------|-----------------|----------------|
| V5_Nvocab_100 | 1.84 | 0.10 |
| V5_Nvocab_250 | 1.73 | 0.10 |
| V5_Nvocab_500 | 1.74 | 0.10 |
| V5_Nvocab_1000 | 1.74 | 0.10 |
| V5_Nvocab_1500 | 1.74 | 0.10 |
| V5_Nvocab_2000 | 1.74 | 0.10 |
| V5_IDFFilter_None | 1.74 | 0.10 |
| V5_IDFFilter_Top-Quartile | 1.74 | 0.10 |

### Router & Triage Metrics

| Combination | Bypass Chunks | Bypass Rate | Rerank Queue | Discard Count | Discard Rate | Compression Ratio |
|-------------|---------------|-------------|--------------|---------------|--------------|-------------------|
| V5_Nvocab_100 | 2 | 0.0% | 2000 | 31098 | 94.0% | 0.3665 |
| V5_Nvocab_250 | 0 | 0.0% | 1977 | 31123 | 94.0% | 0.2680 |
| V5_Nvocab_500 | 1 | 0.0% | 1783 | 31316 | 94.6% | 0.2342 |
| V5_Nvocab_1000 | 0 | 0.0% | 1447 | 31653 | 95.6% | 0.2186 |
| V5_Nvocab_1500 | 0 | 0.0% | 1535 | 31565 | 95.4% | 0.2438 |
| V5_Nvocab_2000 | 1 | 0.0% | 1456 | 31643 | 95.6% | 0.2225 |
| V5_IDFFilter_None | 1 | 0.0% | 1848 | 31251 | 94.4% | 0.2363 |
| V5_IDFFilter_Top-Quartile | 0 | 0.0% | 1545 | 31555 | 95.3% | 0.2508 |

### QE Ablation Diagnostics (Query Expansion Ablation Only)

| Combination | Effective Vocab Size | Avg Aspects/Query | Macro Query Recall |
|-------------|----------------------|-------------------|--------------------|
| V5_Nvocab_100 | 100 | 10.0 | 56.6% |
| V5_Nvocab_250 | 250 | 10.0 | 59.2% |
| V5_Nvocab_500 | 500 | 10.0 | 56.3% |
| V5_Nvocab_1000 | 1000 | 10.0 | 56.8% |
| V5_Nvocab_1500 | 1500 | 10.0 | 58.8% |
| V5_Nvocab_2000 | 2000 | 10.0 | 58.2% |
| V5_IDFFilter_None | 1000 | 10.0 | 60.7% |
| V5_IDFFilter_Top-Quartile | 881 | 10.0 | 58.8% |

====================================================================================================

## Benchmark Corpus: LiveRAG (Streaming News)
- **Log File:** `pipeline_run_liverag_20260805_174453.json`

### Retrieval Quality Metrics

| Combination | N_vocab | IDF Filter | Pre-Rerank Recall | Pre-Rerank Hit Rate | Chunk Recall | Chunk Precision | Strict Recall@10 | Ext Recall@10 | Micro Rerank Recall | Macro Rerank Recall |
|-------------|---------|------------|-------------------|---------------------|--------------|-----------------|------------------|---------------|---------------------|---------------------|
| V5_Nvocab_100 | 100 | Median | 69.0% | 88.7% | 51.5% | 72.9% | 85.3% | 85.3% | 74.2% | 83.2% |
| V5_Nvocab_250 | 250 | Median | 66.3% | 86.0% | 50.9% | 77.1% | 82.7% | 82.7% | 76.7% | 85.3% |
| V5_Nvocab_500 | 500 | Median | 68.2% | 86.0% | 49.3% | 80.3% | 80.7% | 80.7% | 72.3% | 80.1% |
| V5_Nvocab_1000 | 1000 | Median | 70.6% | 88.7% | 48.5% | 72.0% | 84.7% | 84.7% | 68.7% | 78.8% |
| V5_Nvocab_1500 | 1500 | Median | 72.2% | 90.0% | 46.6% | 69.8% | 85.3% | 85.3% | 64.3% | 74.7% |
| V5_Nvocab_2000 | 2000 | Median | 68.7% | 90.0% | 47.4% | 72.4% | 82.0% | 82.0% | 68.5% | 76.0% |
| V5_IDFFilter_None | 1000 | None | 68.5% | 86.0% | 49.3% | 80.6% | 81.3% | 81.3% | 71.9% | 80.5% |
| V5_IDFFilter_Top-Quartile | 1000 | Top-Quartile | 69.5% | 88.0% | 45.8% | 70.5% | 81.3% | 81.3% | 65.9% | 75.3% |

### Timing & Speed Metrics

| Combination | Prebuild (s) | Per-Doc (ms) | Avg QE (s) | Avg Search (s) | Avg Route (s) | Avg Rerank (s) | Avg Total (s) |
|-------------|--------------|--------------|------------|----------------|---------------|----------------|---------------|
| V5_Nvocab_100 | 0.5924 | 1.61 | 0.6691 | 0.0132 | 0.0004 | 2.5178 | 3.2009 |
| V5_Nvocab_250 | 0.1286 | 0.35 | 0.6444 | 0.0122 | 0.0002 | 1.6337 | 2.2906 |
| V5_Nvocab_500 | 0.1225 | 0.33 | 0.6169 | 0.0113 | 0.0002 | 1.3765 | 2.0052 |
| V5_Nvocab_1000 | 0.1506 | 0.41 | 0.6217 | 0.0110 | 0.0002 | 1.3354 | 1.9684 |
| V5_Nvocab_1500 | 0.1538 | 0.42 | 0.6250 | 0.0107 | 0.0001 | 1.2306 | 1.8667 |
| V5_Nvocab_2000 | 0.1738 | 0.47 | 0.6309 | 0.0108 | 0.0001 | 1.1827 | 1.8248 |
| V5_IDFFilter_None | 0.1433 | 0.39 | 0.6255 | 0.0113 | 0.0002 | 1.4454 | 2.0826 |
| V5_IDFFilter_Top-Quartile | 0.1424 | 0.39 | 0.6201 | 0.0108 | 0.0001 | 1.2346 | 1.8659 |

### Memory & Hardware Metrics

| Combination | Python RAM (GB) | Peak VRAM (GB) |
|-------------|-----------------|----------------|
| V5_Nvocab_100 | 1.84 | 0.10 |
| V5_Nvocab_250 | 1.85 | 0.10 |
| V5_Nvocab_500 | 1.85 | 0.10 |
| V5_Nvocab_1000 | 1.86 | 0.10 |
| V5_Nvocab_1500 | 1.86 | 0.10 |
| V5_Nvocab_2000 | 1.86 | 0.10 |
| V5_IDFFilter_None | 1.86 | 0.10 |
| V5_IDFFilter_Top-Quartile | 1.79 | 0.10 |

### Router & Triage Metrics

| Combination | Bypass Chunks | Bypass Rate | Rerank Queue | Discard Count | Discard Rate | Compression Ratio |
|-------------|---------------|-------------|--------------|---------------|--------------|-------------------|
| V5_Nvocab_100 | 4 | 0.0% | 2787 | 52409 | 94.9% | 0.4421 |
| V5_Nvocab_250 | 1 | 0.0% | 1879 | 53320 | 96.6% | 0.3153 |
| V5_Nvocab_500 | 0 | 0.0% | 1623 | 53577 | 97.1% | 0.2990 |
| V5_Nvocab_1000 | 0 | 0.0% | 1596 | 53604 | 97.1% | 0.2627 |
| V5_Nvocab_1500 | 2 | 0.0% | 1462 | 53736 | 97.3% | 0.2416 |
| V5_Nvocab_2000 | 4 | 0.0% | 1451 | 53745 | 97.4% | 0.2162 |
| V5_IDFFilter_None | 1 | 0.0% | 1753 | 53446 | 96.8% | 0.3084 |
| V5_IDFFilter_Top-Quartile | 0 | 0.0% | 1448 | 53752 | 97.4% | 0.2497 |

### QE Ablation Diagnostics (Query Expansion Ablation Only)

| Combination | Effective Vocab Size | Avg Aspects/Query | Macro Query Recall |
|-------------|----------------------|-------------------|--------------------|
| V5_Nvocab_100 | 100 | 10.0 | 62.6% |
| V5_Nvocab_250 | 250 | 10.0 | 61.8% |
| V5_Nvocab_500 | 500 | 10.0 | 57.3% |
| V5_Nvocab_1000 | 1000 | 10.0 | 58.3% |
| V5_Nvocab_1500 | 1500 | 10.0 | 55.4% |
| V5_Nvocab_2000 | 2000 | 10.0 | 57.5% |
| V5_IDFFilter_None | 1000 | 10.0 | 57.7% |
| V5_IDFFilter_Top-Quartile | 909 | 10.0 | 54.7% |

====================================================================================================

## Benchmark Corpus: Fused 50-Paper Corpus Stress 50
- **Log File:** `pipeline_run_stress50_20260805_170016.json`

### Retrieval Quality Metrics

| Combination | N_vocab | IDF Filter | Pre-Rerank Recall | Pre-Rerank Hit Rate | Chunk Recall | Chunk Precision | Strict Recall@10 | Ext Recall@10 | Micro Rerank Recall | Macro Rerank Recall |
|-------------|---------|------------|-------------------|---------------------|--------------|-----------------|------------------|---------------|---------------------|---------------------|
| V5_Nvocab_100 | 100 | Median | 70.5% | 72.0% | 59.8% | 17.8% | 66.7% | 77.8% | 84.2% | 87.1% |
| V5_Nvocab_250 | 250 | Median | 74.6% | 74.3% | 63.0% | 18.1% | 68.8% | 80.4% | 84.3% | 87.2% |
| V5_Nvocab_500 | 500 | Median | 76.8% | 76.7% | 62.4% | 18.0% | 70.4% | 82.0% | 81.1% | 85.2% |
| V5_Nvocab_1000 | 1000 | Median | 76.8% | 76.2% | 62.4% | 18.3% | 68.3% | 82.0% | 81.0% | 83.9% |
| V5_Nvocab_1500 | 1500 | Median | 77.4% | 77.2% | 63.0% | 18.8% | 68.8% | 83.3% | 81.2% | 83.3% |
| V5_Nvocab_2000 | 2000 | Median | 77.2% | 77.0% | 61.2% | 18.3% | 67.5% | 82.0% | 79.0% | 81.7% |
| V5_IDFFilter_None | 1000 | None | 76.4% | 76.7% | 64.8% | 18.2% | 72.0% | 82.5% | 84.5% | 87.7% |
| V5_IDFFilter_Top-Quartile | 1000 | Top-Quartile | 77.0% | 76.5% | 62.6% | 18.5% | 70.6% | 83.6% | 81.1% | 85.4% |

### Timing & Speed Metrics

| Combination | Prebuild (s) | Per-Doc (ms) | Avg QE (s) | Avg Search (s) | Avg Route (s) | Avg Rerank (s) | Avg Total (s) |
|-------------|--------------|--------------|------------|----------------|---------------|----------------|---------------|
| V5_Nvocab_100 | 0.4677 | 0.30 | 0.7697 | 0.0500 | 0.0019 | 3.1069 | 3.9288 |
| V5_Nvocab_250 | 0.3639 | 0.24 | 0.7173 | 0.0468 | 0.0012 | 2.7565 | 3.5222 |
| V5_Nvocab_500 | 0.3413 | 0.22 | 0.7172 | 0.0468 | 0.0012 | 2.5967 | 3.3622 |
| V5_Nvocab_1000 | 0.3754 | 0.24 | 0.7208 | 0.0463 | 0.0009 | 2.4611 | 3.2295 |
| V5_Nvocab_1500 | 0.3546 | 0.23 | 0.7221 | 0.0459 | 0.0010 | 2.4087 | 3.1781 |
| V5_Nvocab_2000 | 0.3647 | 0.24 | 0.7209 | 0.0448 | 0.0009 | 2.3173 | 3.0842 |
| V5_IDFFilter_None | 0.3106 | 0.20 | 0.7085 | 0.0457 | 0.0010 | 2.5677 | 3.3232 |
| V5_IDFFilter_Top-Quartile | 0.3149 | 0.20 | 0.7020 | 0.0453 | 0.0009 | 2.3588 | 3.1073 |

### Memory & Hardware Metrics

| Combination | Python RAM (GB) | Peak VRAM (GB) |
|-------------|-----------------|----------------|
| V5_Nvocab_100 | 1.89 | 0.10 |
| V5_Nvocab_250 | 1.64 | 0.10 |
| V5_Nvocab_500 | 1.64 | 0.10 |
| V5_Nvocab_1000 | 1.64 | 0.10 |
| V5_Nvocab_1500 | 1.65 | 0.10 |
| V5_Nvocab_2000 | 1.64 | 0.10 |
| V5_IDFFilter_None | 1.64 | 0.10 |
| V5_IDFFilter_Top-Quartile | 1.59 | 0.10 |

### Router & Triage Metrics

| Combination | Bypass Chunks | Bypass Rate | Rerank Queue | Discard Count | Discard Rate | Compression Ratio |
|-------------|---------------|-------------|--------------|---------------|--------------|-------------------|
| V5_Nvocab_100 | 43 | 0.0% | 7560 | 573383 | 98.7% | 0.4926 |
| V5_Nvocab_250 | 13 | 0.0% | 7531 | 573442 | 98.7% | 0.3904 |
| V5_Nvocab_500 | 10 | 0.0% | 7498 | 573478 | 98.7% | 0.3400 |
| V5_Nvocab_1000 | 13 | 0.0% | 7354 | 573619 | 98.7% | 0.3005 |
| V5_Nvocab_1500 | 11 | 0.0% | 7148 | 573827 | 98.8% | 0.3053 |
| V5_Nvocab_2000 | 16 | 0.0% | 7034 | 573936 | 98.8% | 0.2903 |
| V5_IDFFilter_None | 17 | 0.0% | 7480 | 573489 | 98.7% | 0.3437 |
| V5_IDFFilter_Top-Quartile | 6 | 0.0% | 7149 | 573831 | 98.8% | 0.2950 |

### QE Ablation Diagnostics (Query Expansion Ablation Only)

| Combination | Effective Vocab Size | Avg Aspects/Query | Macro Query Recall |
|-------------|----------------------|-------------------|--------------------|
| V5_Nvocab_100 | 100 | 10.0 | 57.9% |
| V5_Nvocab_250 | 250 | 10.0 | 61.0% |
| V5_Nvocab_500 | 500 | 10.0 | 61.5% |
| V5_Nvocab_1000 | 1000 | 10.0 | 60.4% |
| V5_Nvocab_1500 | 1500 | 10.0 | 61.0% |
| V5_Nvocab_2000 | 2000 | 10.0 | 59.4% |
| V5_IDFFilter_None | 1000 | 10.0 | 62.6% |
| V5_IDFFilter_Top-Quartile | 778 | 10.0 | 62.0% |

====================================================================================================
