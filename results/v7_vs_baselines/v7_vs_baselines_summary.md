# ⚖️ Side-by-Side Benchmark Summary: V7 vs Baselines

- **Evaluated Benchmarks:** 10 Corpora

## 🏆 Global Macro-Averaged Results

| Model / Architecture | Strict@10 | DocRec@10 | Strict@50 | DocRec@50 | MRR@10 | Latency (Mean) | Setup TTI | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense (bge-large-en-v1.5)** | 70.49% | 54.64% | 81.63% | 67.56% | 0.5299 | 115.06 ms | 108.45 s | 3.76 GB |
| **SPLADE-v3 (DistilBERT)** | 67.85% | 52.38% | 78.53% | 64.40% | 0.5140 | 37.73 ms | 146.92 s | 4.18 GB |
| **Dense (bge-small-en-v1.5)** | 66.73% | 51.09% | 78.50% | 64.38% | 0.4936 | 47.62 ms | 16.23 s | 1.32 GB |
| **Edge-RAG V7 Calibrated (Frozen)** | 62.80% | 49.56% | 74.87% | 61.33% | 0.4816 | 46.67 ms | 14.62 s | 0.36 GB |
| **BM25 (Analyzed Parity Baseline)** | 60.74% | 48.05% | 72.77% | 59.31% | 0.4768 | 0.83 ms | 4.75 s | 0.00 GB |
| **Edge-RAG V7 Default (p=1.00)** | 59.41% | 47.80% | 72.14% | 59.79% | 0.4481 | 68.90 ms | 14.76 s | 0.36 GB |

## 📊 Per-Dataset Comparison (Strict@10)

| dataset                        |   BM25 (Analyzed Parity Baseline) |   Dense (bge-large-en-v1.5) |   Dense (bge-small-en-v1.5) |   Edge-RAG V7 Calibrated (Frozen) |   Edge-RAG V7 Default (p=1.00) |   SPLADE-v3 (DistilBERT) |
|:-------------------------------|----------------------------------:|----------------------------:|----------------------------:|----------------------------------:|-------------------------------:|-------------------------:|
| beir_fiqa_doc_level            |                             47.4  |                       72.4  |                       66.4  |                             45.4  |                          45.2  |                    60.2  |
| beir_nfcorpus_doc_level        |                             68.11 |                       76.16 |                       69.35 |                             69.35 |                          69.97 |                    72.76 |
| beir_scifact_doc_level         |                             80.33 |                       88    |                       85    |                             81    |                          79.67 |                    82.33 |
| bright_economics_doc_level     |                             23.3  |                       33.98 |                       35.92 |                             24.27 |                          20.39 |                    31.07 |
| bright_robotics_doc_level      |                             17.82 |                       33.66 |                       30.69 |                             23.76 |                          14.85 |                    32.67 |
| bright_stackoverflow_doc_level |                             29.06 |                       31.62 |                       26.5  |                             39.32 |                          18.8  |                    33.33 |
| enterpriserag_doc_level        |                             94.26 |                       88.3  |                       84.04 |                             93.62 |                          94.47 |                    92.77 |
| financebench_doc_level         |                             50.67 |                       86    |                       75.33 |                             54.67 |                          54    |                    78.67 |
| liverag_doc_level              |                             97.54 |                       97.43 |                       96.76 |                             97.65 |                          97.88 |                    95.75 |
| multihop_rag_doc_level         |                             98.92 |                       97.31 |                       97.31 |                             98.92 |                          98.92 |                    98.92 |


## 📊 Per-Dataset Comparison (DocRec@10)

| dataset                        |   BM25 (Analyzed Parity Baseline) |   Dense (bge-large-en-v1.5) |   Dense (bge-small-en-v1.5) |   Edge-RAG V7 Calibrated (Frozen) |   Edge-RAG V7 Default (p=1.00) |   SPLADE-v3 (DistilBERT) |
|:-------------------------------|----------------------------------:|----------------------------:|----------------------------:|----------------------------------:|-------------------------------:|-------------------------:|
| beir_fiqa_doc_level            |                             30.34 |                       51.42 |                       45.51 |                             28.69 |                          28.46 |                    39.82 |
| beir_nfcorpus_doc_level        |                             14.11 |                       19.29 |                       16.2  |                             14.49 |                          15.07 |                    16.74 |
| beir_scifact_doc_level         |                             79    |                       87.32 |                       83.96 |                             79.29 |                          77.69 |                    80.89 |
| bright_economics_doc_level     |                             10.89 |                       16.71 |                       15.98 |                             12.58 |                          11.84 |                    15.14 |
| bright_robotics_doc_level      |                              8.07 |                       15.4  |                       15.33 |                             12.55 |                           9.35 |                    18.25 |
| bright_stackoverflow_doc_level |                             16.02 |                       16.08 |                       12.63 |                             21.39 |                           9.69 |                    17    |
| enterpriserag_doc_level        |                             92.28 |                       84.09 |                       79.2  |                             91.83 |                          92.47 |                    89.79 |
| financebench_doc_level         |                             47.44 |                       83    |                       72    |                             52.44 |                          51.44 |                    72.78 |
| liverag_doc_level              |                             97.21 |                       96.87 |                       96.2  |                             97.26 |                          97.49 |                    95.2  |
| multihop_rag_doc_level         |                             85.17 |                       76.25 |                       73.84 |                             85.04 |                          84.5  |                    78.14 |

