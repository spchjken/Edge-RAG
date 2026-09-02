# ⚖️ Side-by-Side Benchmark Summary: V7 vs Baselines

- **Evaluated Date:** 2026-09-02 23:22:04
- **Evaluated Benchmarks:** 10 Corpora

## 🏆 Global Macro-Averaged Results

| Model / Architecture | Strict@10 | DocRec@10 | Strict@50 | DocRec@50 | MRR@10 | Latency (Mean) | Setup TTI | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 (Analyzed Baseline)** | 59.80% | 47.02% | 71.95% | 58.41% | 0.4620 | 1.22 ms | 10.12 s | 0.00 GB |
| **BM25 (Blank Baseline)** | 52.34% | 39.87% | 62.29% | 49.84% | 0.3888 | 1.64 ms | 5.23 s | 0.00 GB |
| **Dense (bge-small-en-v1.5)** | 64.62% | 48.92% | 76.59% | 62.19% | 0.4718 | 53.38 ms | 23.01 s | 2.07 GB |
| **Edge-RAG V7 (GPU-Sparse Bailout)** | 62.82% | 49.35% | 74.18% | 60.54% | 0.4731 | 15.61 ms | 14.56 s | 1.06 GB |
| **SPLADE-v3 (DistilBERT)** | 66.40% | 50.68% | 77.47% | 63.04% | 0.4985 | 46.50 ms | 197.84 s | 5.00 GB |

## 📊 Per-Dataset Comparison (Strict@10)

| dataset                        |   BM25 (Analyzed Baseline) |   BM25 (Blank Baseline) |   Dense (bge-small-en-v1.5) |   Edge-RAG V7 (GPU-Sparse Bailout) |   SPLADE-v3 (DistilBERT) |
|:-------------------------------|---------------------------:|------------------------:|----------------------------:|-----------------------------------:|-------------------------:|
| beir_fiqa_doc_level            |                      47.4  |                   33.4  |                       66.4  |                              46.6  |                    60.2  |
| beir_nfcorpus_doc_level        |                      68.11 |                   63.47 |                       69.35 |                              69.97 |                    72.76 |
| beir_scifact_doc_level         |                      80.33 |                   69.67 |                       85    |                              82.33 |                    82.33 |
| bright_economics_doc_level     |                      23.3  |                   24.27 |                       35.92 |                              27.18 |                    31.07 |
| bright_robotics_doc_level      |                      17.82 |                    9.9  |                       30.69 |                              25.74 |                    32.67 |
| bright_stackoverflow_doc_level |                      29.06 |                   23.08 |                       26.5  |                              38.46 |                    33.33 |
| enterpriserag_doc_level        |                      84.89 |                   75.11 |                       62.98 |                              84.89 |                    78.3  |
| financebench_doc_level         |                      50.67 |                   32.67 |                       75.33 |                              56    |                    78.67 |
| liverag_doc_level              |                      97.54 |                   95.08 |                       96.76 |                              98.1  |                    95.75 |
| multihop_rag_doc_level         |                      98.92 |                   96.77 |                       97.31 |                              98.92 |                    98.92 |


## 📊 Per-Dataset Comparison (DocRec@10)

| dataset                        |   BM25 (Analyzed Baseline) |   BM25 (Blank Baseline) |   Dense (bge-small-en-v1.5) |   Edge-RAG V7 (GPU-Sparse Bailout) |   SPLADE-v3 (DistilBERT) |
|:-------------------------------|---------------------------:|------------------------:|----------------------------:|-----------------------------------:|-------------------------:|
| beir_fiqa_doc_level            |                      30.34 |                   20.37 |                       45.51 |                              29.88 |                    39.82 |
| beir_nfcorpus_doc_level        |                      14.11 |                   12.69 |                       16.2  |                              14.94 |                    16.74 |
| beir_scifact_doc_level         |                      79    |                   68.37 |                       83.96 |                              80.61 |                    80.89 |
| bright_economics_doc_level     |                      10.89 |                    7.34 |                       15.98 |                              14.09 |                    15.14 |
| bright_robotics_doc_level      |                       8.07 |                    4.31 |                       15.33 |                              14.36 |                    18.25 |
| bright_stackoverflow_doc_level |                      16.02 |                   11.46 |                       12.63 |                              21.82 |                    17    |
| enterpriserag_doc_level        |                      81.99 |                   71.21 |                       57.54 |                              81.75 |                    72.84 |
| financebench_doc_level         |                      47.44 |                   32    |                       72    |                              53.11 |                    72.78 |
| liverag_doc_level              |                      97.21 |                   94.08 |                       96.2  |                              97.77 |                    95.2  |
| multihop_rag_doc_level         |                      85.17 |                   76.84 |                       73.84 |                              85.13 |                    78.14 |


## 📊 Per-Dataset Comparison (Strict@50)

| dataset                        |   BM25 (Analyzed Baseline) |   BM25 (Blank Baseline) |   Dense (bge-small-en-v1.5) |   Edge-RAG V7 (GPU-Sparse Bailout) |   SPLADE-v3 (DistilBERT) |
|:-------------------------------|---------------------------:|------------------------:|----------------------------:|-----------------------------------:|-------------------------:|
| beir_fiqa_doc_level            |                      64    |                   51.8  |                       79.8  |                              66.4  |                    76    |
| beir_nfcorpus_doc_level        |                      74.92 |                   72.14 |                       79.57 |                              78.02 |                    78.64 |
| beir_scifact_doc_level         |                      88    |                   79    |                       93    |                              89    |                    88.67 |
| bright_economics_doc_level     |                      45.63 |                   36.89 |                       57.28 |                              45.63 |                    53.4  |
| bright_robotics_doc_level      |                      36.63 |                   16.83 |                       46.53 |                              47.52 |                    52.48 |
| bright_stackoverflow_doc_level |                      48.72 |                   37.61 |                       47.86 |                              52.14 |                    50.43 |
| enterpriserag_doc_level        |                      90.43 |                   84.47 |                       74.47 |                              90.43 |                    87.02 |
| financebench_doc_level         |                      72.67 |                   49.33 |                       89.33 |                              74    |                    90    |
| liverag_doc_level              |                      98.99 |                   96.98 |                       98.55 |                              99.22 |                    98.55 |
| multihop_rag_doc_level         |                      99.46 |                   97.85 |                       99.46 |                              99.46 |                    99.46 |


## 📊 Per-Dataset Comparison (DocRec@50)

| dataset                        |   BM25 (Analyzed Baseline) |   BM25 (Blank Baseline) |   Dense (bge-small-en-v1.5) |   Edge-RAG V7 (GPU-Sparse Bailout) |   SPLADE-v3 (DistilBERT) |
|:-------------------------------|---------------------------:|------------------------:|----------------------------:|-----------------------------------:|-------------------------:|
| beir_fiqa_doc_level            |                      45.28 |                   33.85 |                       62.05 |                              47.05 |                    56.45 |
| beir_nfcorpus_doc_level        |                      20.16 |                   18.26 |                       25.48 |                              22.14 |                    23.71 |
| beir_scifact_doc_level         |                      87.02 |                   77.75 |                       92.5  |                              88.42 |                    88.2  |
| bright_economics_doc_level     |                      25.01 |                   19.66 |                       38.82 |                              25.51 |                    34.44 |
| bright_robotics_doc_level      |                      20.81 |                    8.97 |                       24.65 |                              27.92 |                    33.68 |
| bright_stackoverflow_doc_level |                      31.62 |                   21.59 |                       32.83 |                              37.23 |                    31.37 |
| enterpriserag_doc_level        |                      88.97 |                   81.12 |                       69.01 |                              89.04 |                    83.27 |
| financebench_doc_level         |                      68.22 |                   47.89 |                       85.22 |                              71    |                    85.67 |
| liverag_doc_level              |                      98.83 |                   96.59 |                       98.44 |                              99.11 |                    98.21 |
| multihop_rag_doc_level         |                      98.16 |                   92.74 |                       92.88 |                              97.98 |                    95.43 |

