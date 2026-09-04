# ⚖️ Side-by-Side Benchmark Summary: V7 vs Baselines

- **Evaluated Date:** 2026-09-04 22:38:32
- **Evaluated Benchmarks:** 10 Corpora (Subprocess-Isolated)

## 🏆 Global Macro-Averaged Results

| Model / Architecture | Strict@10 | DocRec@10 | Strict@50 | DocRec@50 | MRR@10 | nDCG@10 | Latency (Mean) | Setup TTI | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 (Analyzed Lucene, kstem)** | 61.17% | 48.21% | 73.63% | 60.15% | 0.4667 | **0.4331** | 0.96 ms | 10.52 s | 0.00 GB |
| **BM25 (Standard Lucene, unstemmed)** | 61.34% | 47.66% | 72.40% | 59.54% | 0.4621 | **0.4279** | 1.53 ms | 5.45 s | 0.00 GB |
| **Dense (bge-small-en-v1.5)** | 66.26% | 50.39% | 77.34% | 63.10% | 0.4827 | **0.4470** | 53.05 ms | 22.72 s | 1.23 GB |
| **Edge-RAG V7 (GPU-Sparse Bailout)** | 63.05% | 49.54% | 75.53% | 61.89% | 0.4728 | **0.4413** | 14.41 ms | 16.69 s | 0.38 GB |
| **SPLADE-v3 (DistilBERT)** | 67.38% | 51.42% | 78.56% | 64.23% | 0.5133 | **0.4667** | 9.38 ms | 157.26 s | 11.46 GB |

## 📊 Per-Dataset Comparison (Strict@10)

| dataset                        |   BM25 (Analyzed Lucene, kstem) |   BM25 (Standard Lucene, unstemmed) |   Dense (bge-small-en-v1.5) |   Edge-RAG V7 (GPU-Sparse Bailout) |   SPLADE-v3 (DistilBERT) |
|:-------------------------------|--------------------------------:|------------------------------------:|----------------------------:|-----------------------------------:|-------------------------:|
| beir_fiqa_doc_level            |                           48.15 |                               48.77 |                       67.44 |                              47.84 |                    60.8  |
| beir_nfcorpus_doc_level        |                           68.11 |                               68.73 |                       69.04 |                              69.97 |                    72.76 |
| beir_scifact_doc_level         |                           80.33 |                               80.67 |                       84.67 |                              82.33 |                    82.33 |
| bright_economics_doc_level     |                           23.3  |                               28.16 |                       37.86 |                              27.18 |                    31.07 |
| bright_robotics_doc_level      |                           17.82 |                               18.81 |                       30.69 |                              25.74 |                    35.64 |
| bright_stackoverflow_doc_level |                           29.06 |                               35.9  |                       26.5  |                              38.46 |                    31.62 |
| enterpriserag_doc_level        |                           84.89 |                               86.17 |                       62.98 |                              84.89 |                    78.3  |
| financebench_doc_level         |                           63.33 |                               50    |                       90.67 |                              56.67 |                    87.33 |
| liverag_doc_level              |                           97.54 |                               96.76 |                       96.76 |                              98.1  |                    95.75 |
| multihop_rag_doc_level         |                           99.16 |                               99.47 |                       95.96 |                              99.33 |                    98.18 |


## 📊 Per-Dataset Comparison (DocRec@10)

| dataset                        |   BM25 (Analyzed Lucene, kstem) |   BM25 (Standard Lucene, unstemmed) |   Dense (bge-small-en-v1.5) |   Edge-RAG V7 (GPU-Sparse Bailout) |   SPLADE-v3 (DistilBERT) |
|:-------------------------------|--------------------------------:|------------------------------------:|----------------------------:|-----------------------------------:|-------------------------:|
| beir_fiqa_doc_level            |                           30.89 |                               30.17 |                       46.57 |                              30.95 |                    40.52 |
| beir_nfcorpus_doc_level        |                           14.11 |                               14.89 |                       16.04 |                              14.9  |                    16.74 |
| beir_scifact_doc_level         |                           79    |                               78.76 |                       83.62 |                              80.61 |                    80.89 |
| bright_economics_doc_level     |                           10.89 |                               13.03 |                       16.74 |                              14.07 |                    15.06 |
| bright_robotics_doc_level      |                            8.07 |                                9.76 |                       15.33 |                              13.86 |                    18.5  |
| bright_stackoverflow_doc_level |                           16.02 |                               17.65 |                       12.63 |                              21.82 |                    15.67 |
| enterpriserag_doc_level        |                           81.99 |                               83.34 |                       57.54 |                              81.75 |                    72.84 |
| financebench_doc_level         |                           59    |                               48.11 |                       86.22 |                              55    |                    82.33 |
| liverag_doc_level              |                           97.21 |                               96.03 |                       96.2  |                              97.77 |                    95.2  |
| multihop_rag_doc_level         |                           84.95 |                               84.89 |                       73.03 |                              84.7  |                    76.42 |


## 📊 Per-Dataset Comparison (Strict@50)

| dataset                        |   BM25 (Analyzed Lucene, kstem) |   BM25 (Standard Lucene, unstemmed) |   Dense (bge-small-en-v1.5) |   Edge-RAG V7 (GPU-Sparse Bailout) |   SPLADE-v3 (DistilBERT) |
|:-------------------------------|--------------------------------:|------------------------------------:|----------------------------:|-----------------------------------:|-------------------------:|
| beir_fiqa_doc_level            |                           65.9  |                               64.04 |                       79.78 |                              67.59 |                    76.39 |
| beir_nfcorpus_doc_level        |                           74.92 |                               75.54 |                       79.57 |                              78.02 |                    78.64 |
| beir_scifact_doc_level         |                           88    |                               88    |                       93    |                              89    |                    88.67 |
| bright_economics_doc_level     |                           45.63 |                               48.54 |                       57.28 |                              45.63 |                    54.37 |
| bright_robotics_doc_level      |                           36.63 |                               32.67 |                       46.53 |                              47.52 |                    53.47 |
| bright_stackoverflow_doc_level |                           48.72 |                               51.28 |                       47.86 |                              52.14 |                    52.14 |
| enterpriserag_doc_level        |                           90.43 |                               90.43 |                       74.47 |                              90.43 |                    87.02 |
| financebench_doc_level         |                           87.33 |                               75.33 |                       96.67 |                              86    |                    96.67 |
| liverag_doc_level              |                           98.99 |                               98.21 |                       98.55 |                              99.22 |                    98.55 |
| multihop_rag_doc_level         |                           99.73 |                               99.91 |                       99.65 |                              99.78 |                    99.73 |


## 📊 Per-Dataset Comparison (DocRec@50)

| dataset                        |   BM25 (Analyzed Lucene, kstem) |   BM25 (Standard Lucene, unstemmed) |   Dense (bge-small-en-v1.5) |   Edge-RAG V7 (GPU-Sparse Bailout) |   SPLADE-v3 (DistilBERT) |
|:-------------------------------|--------------------------------:|------------------------------------:|----------------------------:|-----------------------------------:|-------------------------:|
| beir_fiqa_doc_level            |                           46.21 |                               44.59 |                       62.17 |                              47.72 |                    56.53 |
| beir_nfcorpus_doc_level        |                           20.16 |                               21.07 |                       25.46 |                              22.2  |                    23.71 |
| beir_scifact_doc_level         |                           87.02 |                               87.04 |                       92.5  |                              88.42 |                    88.2  |
| bright_economics_doc_level     |                           25.01 |                               30.32 |                       38.82 |                              25.51 |                    34.96 |
| bright_robotics_doc_level      |                           20.81 |                               21.49 |                       24.65 |                              27.92 |                    33.53 |
| bright_stackoverflow_doc_level |                           31.62 |                               32.69 |                       32.83 |                              37.37 |                    33.48 |
| enterpriserag_doc_level        |                           88.97 |                               88.76 |                       69.01 |                              89.04 |                    83.27 |
| financebench_doc_level         |                           84.78 |                               73.33 |                       95.44 |                              83.67 |                    96.11 |
| liverag_doc_level              |                           98.83 |                               97.93 |                       98.44 |                              99.11 |                    98.21 |
| multihop_rag_doc_level         |                           98.07 |                               98.22 |                       91.67 |                              97.9  |                    94.26 |


## 📊 Per-Dataset Comparison (nDCG@10)

| dataset                        |   BM25 (Analyzed Lucene, kstem) |   BM25 (Standard Lucene, unstemmed) |   Dense (bge-small-en-v1.5) |   Edge-RAG V7 (GPU-Sparse Bailout) |   SPLADE-v3 (DistilBERT) |
|:-------------------------------|--------------------------------:|------------------------------------:|----------------------------:|-----------------------------------:|-------------------------:|
| beir_fiqa_doc_level            |                          0.2463 |                              0.2377 |                      0.4032 |                             0.244  |                   0.3441 |
| beir_nfcorpus_doc_level        |                          0.3123 |                              0.3074 |                      0.3445 |                             0.3249 |                   0.3571 |
| beir_scifact_doc_level         |                          0.6637 |                              0.6634 |                      0.7124 |                             0.6691 |                   0.6921 |
| bright_economics_doc_level     |                          0.1103 |                              0.1113 |                      0.1451 |                             0.1205 |                   0.142  |
| bright_robotics_doc_level      |                          0.0537 |                              0.0684 |                      0.1219 |                             0.0893 |                   0.1533 |
| bright_stackoverflow_doc_level |                          0.1406 |                              0.1513 |                      0.0867 |                             0.1722 |                   0.1161 |
| enterpriserag_doc_level        |                          0.726  |                              0.7446 |                      0.4756 |                             0.7309 |                   0.6575 |
| financebench_doc_level         |                          0.4011 |                              0.335  |                      0.6531 |                             0.3827 |                   0.6391 |
| liverag_doc_level              |                          0.9428 |                              0.9284 |                      0.9118 |                             0.9473 |                   0.9156 |
| multihop_rag_doc_level         |                          0.7338 |                              0.7312 |                      0.6153 |                             0.7325 |                   0.6502 |

