# ⚖️ Side-by-Side Benchmark Summary: V7 vs Baselines

- **Evaluated Date:** 2026-09-03 05:31:31
- **Evaluated Benchmarks:** 10 Corpora (Subprocess-Isolated)

## 🏆 Global Macro-Averaged Results

| Model / Architecture | Strict@10 | DocRec@10 | Strict@50 | DocRec@50 | MRR@10 | nDCG@10 | Latency (Mean) | Setup TTI | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 (Analyzed Baseline)** | 56.82% | 44.40% | 69.04% | 55.54% | 0.4370 | **0.4037** | 0.86 ms | 9.41 s | 0.00 GB |
| **BM25 (Blank Baseline)** | 49.47% | 37.52% | 59.38% | 47.08% | 0.3672 | **0.3372** | 1.25 ms | 5.01 s | 0.00 GB |
| **Dense (bge-small-en-v1.5)** | 61.85% | 46.77% | 73.69% | 59.49% | 0.4513 | **0.4180** | 49.27 ms | 20.97 s | 1.27 GB |
| **Edge-RAG V7 (GPU-Sparse Bailout)** | 59.84% | 46.73% | 71.23% | 57.64% | 0.4476 | **0.4175** | 13.38 ms | 15.74 s | 0.38 GB |
| **SPLADE-v3 (DistilBERT)** | 63.46% | 48.29% | 74.54% | 60.24% | 0.4762 | **0.4365** | 42.22 ms | 177.46 s | 4.20 GB |

## 📊 Per-Dataset Comparison (Strict@10)

| dataset                        |   BM25 (Analyzed Baseline) |   BM25 (Blank Baseline) |   Dense (bge-small-en-v1.5) |   Edge-RAG V7 (GPU-Sparse Bailout) |   SPLADE-v3 (DistilBERT) |
|:-------------------------------|---------------------------:|------------------------:|----------------------------:|-----------------------------------:|-------------------------:|
| beir_fiqa_doc_level            |                      47.4  |                   33.4  |                       66.4  |                              46.6  |                    60.2  |
| beir_nfcorpus_doc_level        |                      68.11 |                   63.47 |                       69.35 |                              69.97 |                    72.76 |
| beir_scifact_doc_level         |                      80.33 |                   69.67 |                       85    |                              82.33 |                    82.33 |
| bright_economics_doc_level     |                      23.3  |                   24.27 |                       35.92 |                              27.18 |                    31.07 |
| bright_robotics_doc_level      |                      17.82 |                    9.9  |                       30.69 |                              25.74 |                    32.67 |
| bright_stackoverflow_doc_level |                      29.06 |                   23.08 |                       26.5  |                              38.46 |                    33.33 |
| enterpriserag_doc_level        |                      79.8  |                   70.6  |                       59.2  |                              79.8  |                    73.6  |
| financebench_doc_level         |                      50.67 |                   32.67 |                       75.33 |                              56    |                    78.67 |
| liverag_doc_level              |                      97.54 |                   95.08 |                       96.76 |                              98.1  |                    95.75 |
| multihop_rag_doc_level         |                      74.19 |                   72.58 |                       73.39 |                              74.19 |                    74.19 |


## 📊 Per-Dataset Comparison (DocRec@10)

| dataset                        |   BM25 (Analyzed Baseline) |   BM25 (Blank Baseline) |   Dense (bge-small-en-v1.5) |   Edge-RAG V7 (GPU-Sparse Bailout) |   SPLADE-v3 (DistilBERT) |
|:-------------------------------|---------------------------:|------------------------:|----------------------------:|-----------------------------------:|-------------------------:|
| beir_fiqa_doc_level            |                      30.34 |                   20.37 |                       45.51 |                              29.88 |                    39.82 |
| beir_nfcorpus_doc_level        |                      14.11 |                   12.69 |                       16.2  |                              14.94 |                    16.74 |
| beir_scifact_doc_level         |                      79    |                   68.37 |                       83.96 |                              80.61 |                    80.89 |
| bright_economics_doc_level     |                      10.89 |                    7.34 |                       15.98 |                              14.09 |                    15.14 |
| bright_robotics_doc_level      |                       8.07 |                    4.31 |                       15.33 |                              14.36 |                    18.25 |
| bright_stackoverflow_doc_level |                      16.02 |                   11.46 |                       12.63 |                              21.82 |                    17    |
| enterpriserag_doc_level        |                      77.07 |                   66.94 |                       54.09 |                              76.85 |                    68.47 |
| financebench_doc_level         |                      47.44 |                   32    |                       72    |                              53.11 |                    72.78 |
| liverag_doc_level              |                      97.21 |                   94.08 |                       96.2  |                              97.77 |                    95.2  |
| multihop_rag_doc_level         |                      63.88 |                   57.63 |                       55.78 |                              63.84 |                    58.6  |


## 📊 Per-Dataset Comparison (Strict@50)

| dataset                        |   BM25 (Analyzed Baseline) |   BM25 (Blank Baseline) |   Dense (bge-small-en-v1.5) |   Edge-RAG V7 (GPU-Sparse Bailout) |   SPLADE-v3 (DistilBERT) |
|:-------------------------------|---------------------------:|------------------------:|----------------------------:|-----------------------------------:|-------------------------:|
| beir_fiqa_doc_level            |                      64    |                   51.8  |                       79.8  |                              66.4  |                    76    |
| beir_nfcorpus_doc_level        |                      74.92 |                   72.14 |                       79.57 |                              78.02 |                    78.64 |
| beir_scifact_doc_level         |                      88    |                   79    |                       93    |                              89    |                    88.67 |
| bright_economics_doc_level     |                      45.63 |                   36.89 |                       57.28 |                              45.63 |                    53.4  |
| bright_robotics_doc_level      |                      36.63 |                   16.83 |                       46.53 |                              47.52 |                    52.48 |
| bright_stackoverflow_doc_level |                      48.72 |                   37.61 |                       47.86 |                              52.14 |                    50.43 |
| enterpriserag_doc_level        |                      85    |                   79.4  |                       70    |                              85    |                    81.8  |
| financebench_doc_level         |                      72.67 |                   49.33 |                       89.33 |                              74    |                    90    |
| liverag_doc_level              |                      98.99 |                   96.98 |                       98.55 |                              99.22 |                    98.55 |
| multihop_rag_doc_level         |                      75.81 |                   73.79 |                       75    |                              75.4  |                    75.4  |


## 📊 Per-Dataset Comparison (DocRec@50)

| dataset                        |   BM25 (Analyzed Baseline) |   BM25 (Blank Baseline) |   Dense (bge-small-en-v1.5) |   Edge-RAG V7 (GPU-Sparse Bailout) |   SPLADE-v3 (DistilBERT) |
|:-------------------------------|---------------------------:|------------------------:|----------------------------:|-----------------------------------:|-------------------------:|
| beir_fiqa_doc_level            |                      45.28 |                   33.85 |                       62.05 |                              47.05 |                    56.45 |
| beir_nfcorpus_doc_level        |                      20.16 |                   18.26 |                       25.48 |                              22.14 |                    23.71 |
| beir_scifact_doc_level         |                      87.02 |                   77.75 |                       92.5  |                              88.42 |                    88.2  |
| bright_economics_doc_level     |                      25.01 |                   19.66 |                       38.82 |                              25.51 |                    34.44 |
| bright_robotics_doc_level      |                      20.81 |                    8.97 |                       24.65 |                              27.92 |                    33.68 |
| bright_stackoverflow_doc_level |                      31.62 |                   21.59 |                       32.83 |                              37.23 |                    31.37 |
| enterpriserag_doc_level        |                      83.64 |                   76.25 |                       64.87 |                              83.7  |                    78.27 |
| financebench_doc_level         |                      68.22 |                   47.89 |                       85.22 |                              71    |                    85.67 |
| liverag_doc_level              |                      98.83 |                   96.59 |                       98.44 |                              99.11 |                    98.21 |
| multihop_rag_doc_level         |                      74.83 |                   69.96 |                       70.06 |                              74.29 |                    72.38 |


## 📊 Per-Dataset Comparison (nDCG@10)

| dataset                        |   BM25 (Analyzed Baseline) |   BM25 (Blank Baseline) |   Dense (bge-small-en-v1.5) |   Edge-RAG V7 (GPU-Sparse Bailout) |   SPLADE-v3 (DistilBERT) |
|:-------------------------------|---------------------------:|------------------------:|----------------------------:|-----------------------------------:|-------------------------:|
| beir_fiqa_doc_level            |                     0.2366 |                  0.1587 |                      0.3888 |                             0.2322 |                   0.3348 |
| beir_nfcorpus_doc_level        |                     0.3093 |                  0.2706 |                      0.3431 |                             0.3221 |                   0.3539 |
| beir_scifact_doc_level         |                     0.6637 |                  0.5723 |                      0.7137 |                             0.6691 |                   0.6923 |
| bright_economics_doc_level     |                     0.1103 |                  0.0704 |                      0.1417 |                             0.1209 |                   0.143  |
| bright_robotics_doc_level      |                     0.0537 |                  0.0295 |                      0.1213 |                             0.0919 |                   0.1518 |
| bright_stackoverflow_doc_level |                     0.1414 |                  0.0963 |                      0.0868 |                             0.1688 |                   0.114  |
| enterpriserag_doc_level        |                     0.6824 |                  0.5807 |                      0.4471 |                             0.687  |                   0.6173 |
| financebench_doc_level         |                     0.3427 |                  0.2281 |                      0.555  |                             0.3744 |                   0.5551 |
| liverag_doc_level              |                     0.9428 |                  0.8883 |                      0.9118 |                             0.9473 |                   0.9156 |
| multihop_rag_doc_level         |                     0.554  |                  0.477  |                      0.4711 |                             0.5614 |                   0.4868 |

