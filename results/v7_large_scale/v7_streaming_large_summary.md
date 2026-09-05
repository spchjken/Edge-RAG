# 🐘 Large-Scale 5M+ BEIR Benchmark Summary (Streaming IO)

- **Evaluated Date:** 2026-09-05 04:49:46
- **Evaluated Massive Datasets:** beir_quora
- **Methodology:** 16-Bucket Radix Partitioned Memory-Mapped Streaming Inverted Index.

## 📊 Retrieval Quality & Efficiency Comparison

| Dataset | Docs Count | Queries | Model | Strict@10 | DocRec@10 | MRR@10 | nDCG@10 | Latency (Mean) | Peak RSS | Peak VRAM |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `beir_arguana` | 8,674 | 1406 | **BM25 (Analyzed Lucene, kstem)** | 75.39% | 75.39% | 0.2387 | **0.3625** | 2.91 ms | 885.8 MB | 0.00 GB |
| `beir_arguana` | 8,674 | 1406 | **Edge-RAG V7 (GPU-Sparse Bailout)** | 66.50% | 66.50% | 0.2036 | **0.3139** | 38.11 ms | 2055.4 MB | 0.36 GB |
| `beir_scidocs` | 25,657 | 1000 | **BM25 (Analyzed Lucene, kstem)** | 49.00% | 16.07% | 0.2745 | **0.1554** | 0.67 ms | 952.6 MB | 0.00 GB |
| `beir_scidocs` | 25,657 | 1000 | **Edge-RAG V7 (GPU-Sparse Bailout)** | 49.00% | 16.18% | 0.2790 | **0.1574** | 31.64 ms | 2101.1 MB | 0.39 GB |
| `beir_trec_covid` | 171,331 | 50 | **BM25 (Analyzed Lucene, kstem)** | 96.00% | 1.47% | 0.8307 | **0.5379** | 2.76 ms | 1100.3 MB | 0.00 GB |
| `beir_trec_covid` | 171,331 | 50 | **Edge-RAG V7 (GPU-Sparse Bailout)** | 100.00% | 1.46% | 0.8327 | **0.5419** | 73.64 ms | 2226.8 MB | 0.39 GB |
| `beir_webis_touche2020` | 382,545 | 49 | **BM25 (Analyzed Lucene, kstem)** | 93.88% | 20.99% | 0.6467 | **0.3468** | 4.17 ms | 1382.4 MB | 0.00 GB |
| `beir_webis_touche2020` | 382,545 | 49 | **Edge-RAG V7 (GPU-Sparse Bailout)** | 95.92% | 21.56% | 0.7095 | **0.3724** | 113.57 ms | 2396.3 MB | 0.38 GB |
| `beir_quora` | 522,931 | 10000 | **BM25 (Analyzed Lucene, kstem)** | 92.82% | 88.32% | 0.7797 | **0.7862** | 5.25 ms | 971.8 MB | 0.00 GB |
| `beir_quora` | 522,931 | 10000 | **Edge-RAG V7 (GPU-Sparse Bailout)** | 92.35% | 87.95% | 0.7721 | **0.7799** | 156.39 ms | 2161.4 MB | 0.38 GB |

## 📑 Reference Published Literature Baselines (Official Table 2, arXiv:2403.06789)

| Dataset | SPLADE-v3-DistilBERT (†) nDCG@10 | BGE-small-en-v1.5 (†) nDCG@10 | Relevance Type |
| :--- | :---: | :---: | :---: |
| `beir_dbpedia_entity` | **0.4260** | **0.3800** | Binary (BEIR official) |
| `beir_climate_fever` | **0.2280** | **0.2050** | Binary (BEIR official) |
| `beir_nq` | **0.5490** | **0.5280** | Binary (BEIR official) |
| `beir_hotpotqa` | **0.6780** | **0.6550** | Binary (BEIR official) |
| `beir_fever` | **0.7960** | **0.7480** | Binary (BEIR official) |

> *Note: (†) Baselines marked with dagger are cited directly from Table 2 of Lassance et al. (2024) [arXiv:2403.06789] and Xiao et al. (2023). Local neural encoding over 5.4M documents requires ~12–25 hours per corpus on edge hardware and is unviable for ephemeral edge retrieval.*
