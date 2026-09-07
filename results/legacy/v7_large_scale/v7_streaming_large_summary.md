# 🐘 Large-Scale 5M+ BEIR Benchmark Summary (Streaming IO)

- **Evaluated Date:** 2026-09-06 14:49:01
- **Evaluated Massive Datasets:** beir_dbpedia_entity, beir_climate_fever, beir_nq, beir_hotpotqa, beir_fever
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
| `beir_dbpedia_entity` | 4,635,922 | 400 | **BM25 (Analyzed Lucene, kstem)** | 79.50% | 19.47% | 0.5697 | **0.2850** | 45.30 ms | 2867.5 MB | 0.00 GB |
| `beir_dbpedia_entity` | 4,635,922 | 400 | **Edge-RAG V7 (GPU-Sparse Bailout)** | 79.50% | 19.04% | 0.5741 | **0.2793** | 6013.51 ms | 3819.8 MB | 0.37 GB |
| `beir_climate_fever` | 5,416,593 | 1535 | **BM25 (Analyzed Lucene, kstem)** | 34.72% | 16.30% | 0.1802 | **0.1303** | 84.89 ms | 3503.4 MB | 0.00 GB |
| `beir_climate_fever` | 5,416,593 | 1535 | **Edge-RAG V7 (GPU-Sparse Bailout)** | 37.33% | 17.89% | 0.1960 | **0.1429** | 8112.08 ms | 4616.6 MB | 0.37 GB |
| `beir_nq` | 2,681,468 | 3452 | **BM25 (Analyzed Lucene, kstem)** | 47.51% | 43.95% | 0.2433 | **0.2815** | 27.63 ms | 2059.3 MB | 0.00 GB |
| `beir_nq` | 2,681,468 | 3452 | **Edge-RAG V7 (GPU-Sparse Bailout)** | 47.57% | 44.14% | 0.2435 | **0.2818** | 1191.29 ms | 3234.3 MB | 0.36 GB |
| `beir_hotpotqa` | 5,233,329 | 7405 | **BM25 (Analyzed Lucene, kstem)** | 87.85% | 60.79% | 0.7442 | **0.5788** | 79.23 ms | 2909.1 MB | 0.00 GB |
| `beir_hotpotqa` | 5,233,329 | 7405 | **Edge-RAG V7 (GPU-Sparse Bailout)** | 88.24% | 61.29% | 0.7407 | **0.5791** | 7861.18 ms | 3879.3 MB | 0.38 GB |
| `beir_fever` | 5,416,568 | 6666 | **BM25 (Analyzed Lucene, kstem)** | 70.67% | 67.11% | 0.4658 | **0.5044** | 50.88 ms | 3687.1 MB | 0.00 GB |
| `beir_fever` | 5,416,568 | 6666 | **Edge-RAG V7 (GPU-Sparse Bailout)** | 70.43% | 66.87% | 0.4620 | **0.5009** | 6941.14 ms | 4780.8 MB | 0.37 GB |

## 📑 Reference Published Literature Baselines (Official Table 2, arXiv:2403.06789)

| Dataset | SPLADE-v3-DistilBERT (†) nDCG@10 | BGE-small-en-v1.5 (†) nDCG@10 | Relevance Type |
| :--- | :---: | :---: | :---: |
| `beir_dbpedia_entity` | **0.4260** | **0.3800** | Binary (BEIR official) |
| `beir_climate_fever` | **0.2280** | **0.2050** | Binary (BEIR official) |
| `beir_nq` | **0.5490** | **0.5280** | Binary (BEIR official) |
| `beir_hotpotqa` | **0.6780** | **0.6550** | Binary (BEIR official) |
| `beir_fever` | **0.7960** | **0.7480** | Binary (BEIR official) |

> *Note: (†) Baselines marked with dagger are cited directly from Table 2 of Lassance et al. (2024) [arXiv:2403.06789] and Xiao et al. (2023). Local neural encoding over 5.4M documents requires ~12–25 hours per corpus on edge hardware and is unviable for ephemeral edge retrieval.*
