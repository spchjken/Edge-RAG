# 🔬 Coverage (FPS) vs Salience vs IDF Pool Strategy Empirical Report

- **Scope:** 10 Document-Level Benchmarks (3,237 queries, 29,133 total evaluations)
- **Bailout Status:** Permanently Disabled (Zero prefix dilution, pure core V7)

---

## 🏆 1. Macro Retrieval Performance Comparison

| Strategy | Pool Size N | **Strict@10** | **DocRec@10** | Strict@50 | DocRec@50 | MRR@10 | In-Pool Hit % | Total TTI (s) | Total Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`COVERAGE (FPS)`** | **500** | **62.57%** | **49.50%** | 74.82% | 61.03% | **0.4801** | **`71.8%`** 🚀 | 8.70s | **`5.91 ms`** ⚡ |
| **`COVERAGE (FPS)`** | **1000** | **62.74%** | **49.57%** | 74.79% | 61.02% | **0.4813** | **`71.8%`** 🚀 | 8.61s | **`8.04 ms`** ⚡ |
| **`COVERAGE (FPS)`** | **2500** | **62.93%** | **49.65%** | 74.79% | 61.04% | **0.4814** | **`71.8%`** 🚀 | 8.90s | **`15.03 ms`** |
| **`SALIENCE`** | **500** | **62.86%** | **49.73%** | 74.83% | 61.04% | **0.4802** | 6.3% | 5.90s | **8.84 ms** |
| **`SALIENCE`** | **1000** | **63.16%** | **49.80%** | 74.54% | 60.90% | **0.4813** | 13.4% | 5.94s | **11.77 ms** |
| **`SALIENCE`** | **2500** | **62.91%** | **49.74%** | 74.60% | 60.96% | **0.4800** | 31.3% | 6.00s | **20.26 ms** |
| **`IDF`** | **500** | **62.64%** | **49.54%** | 74.69% | 61.07% | **0.4799** | 0.1% | 5.92s | **8.52 ms** |
| **`IDF`** | **1000** | **62.73%** | **49.59%** | 74.72% | 61.08% | **0.4805** | 0.2% | 5.93s | **11.04 ms** |
| **`IDF`** | **2500** | **62.73%** | **49.62%** | 74.69% | 61.07% | **0.4805** | 0.4% | 6.00s | **18.70 ms** |

---

## ⏱️ 2. Dedicated TTI Indexing Breakdown (Seconds)

$$\text{Total TTI} = t_{\text{bm25\_idx}} + t_{\text{vocab\_salience}} + t_{\text{matrix\_fps}}$$

| Strategy | Pool Size N | BM25 Indexing ($t_{\text{bm25}}$) | Vocab Filtering ($t_{\text{vocab}}$) | Dense Matrix / FPS ($t_{\text{matrix}}$) | **Total TTI ($t_{\text{tti}}$)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`COVERAGE (FPS)`** | 500 | 5.64 s | 0.05 s | 3.01 s | **`8.70 s`** |
| **`COVERAGE (FPS)`** | 1000 | 5.55 s | 0.04 s | 3.02 s | **`8.61 s`** |
| **`COVERAGE (FPS)`** | 2500 | 5.56 s | 0.04 s | 3.30 s | **`8.90 s`** |
| **`SALIENCE`** | 500 | 5.57 s | 0.05 s | 0.28 s | **`5.90 s`** ⚡ |
| **`SALIENCE`** | 1000 | 5.57 s | 0.04 s | 0.33 s | **`5.94 s`** ⚡ |
| **`SALIENCE`** | 2500 | 5.58 s | 0.04 s | 0.38 s | **`6.00 s`** ⚡ |
| **`IDF`** | 500 | 5.57 s | 0.05 s | 0.30 s | **`5.92 s`** ⚡ |
| **`IDF`** | 1000 | 5.56 s | 0.04 s | 0.33 s | **`5.93 s`** ⚡ |
| **`IDF`** | 2500 | 5.57 s | 0.04 s | 0.39 s | **`6.00 s`** ⚡ |

---

## ⚡ 3. Query-Time Sub-Timer Latency Breakdown (Milliseconds)

| Strategy | Pool N | Anchor Total ($t_{\text{anchor}}$) | Probing GEMM ($t_{\text{prob}}$) | IT-MPE ($t_{\text{itmpe}}$) | BM25 ($t_{\text{bm25}}$) | **Total Query Latency** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`COVERAGE (FPS)`** | 500 | **`3.23 ms`** ⚡ | 0.12 ms | 0.45 ms | 2.07 ms | **`5.91 ms`** ⚡ |
| **`COVERAGE (FPS)`** | 1000 | **`3.26 ms`** ⚡ | 0.11 ms | 0.81 ms | 3.81 ms | **`8.04 ms`** ⚡ |
| **`COVERAGE (FPS)`** | 2500 | **`3.37 ms`** ⚡ | 0.13 ms | 1.98 ms | 9.48 ms | **`15.03 ms`** |
| **`SALIENCE`** | 500 | 4.91 ms | 0.11 ms | 0.51 ms | 3.27 ms | **8.84 ms** |
| **`SALIENCE`** | 1000 | 4.87 ms | 0.11 ms | 0.89 ms | 5.84 ms | **11.77 ms** |
| **`SALIENCE`** | 2500 | 4.64 ms | 0.13 ms | 2.08 ms | 13.35 ms | **20.26 ms** |
| **`IDF`** | 500 | 5.04 ms | 0.11 ms | 0.55 ms | 2.78 ms | **8.52 ms** |
| **`IDF`** | 1000 | 5.03 ms | 0.12 ms | 0.97 ms | 4.87 ms | **11.04 ms** |
| **`IDF`** | 2500 | 5.18 ms | 0.13 ms | 2.24 ms | 11.08 ms | **18.70 ms** |

---

## 📊 4. Per-Dataset Breakdown: TTI, Accuracy & Latency (Coverage vs Salience at N=1000)

| Dataset | Coverage TTI | Salience TTI | Coverage Strict@10 | Salience Strict@10 | Delta (pp) | Coverage In-Pool % | Coverage Latency | Salience Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `enterpriserag_doc_level` | 4.17 s | 2.57 s | **93.83%** | 93.62% | `+0.21 pp` | **42.8%** | **`9.22 ms`** | 10.22 ms |
| `liverag_doc_level` | 3.73 s | 2.13 s | **97.65%** | 97.77% | `-0.12 pp` | **56.7%** | **`5.55 ms`** | 8.56 ms |
| `beir_scifact_doc_level` | 3.50 s | 2.34 s | **81.00%** | 81.00% | `+0.00 pp` | **76.6%** | **`5.68 ms`** | 8.34 ms |
| `beir_nfcorpus_doc_level` | 2.55 s | 1.60 s | **69.35%** | 68.42% | `+0.93 pp` | **83.3%** | **`2.89 ms`** | 5.76 ms |
| `beir_fiqa_doc_level` | 15.82 s | 11.98 s | **46.76%** | 46.76% | `+0.00 pp` | **77.4%** | **`4.17 ms`** | 10.05 ms |
| `multihop_rag_doc_level` | 3.02 s | 2.09 s | **99.38%** | 99.29% | `+0.09 pp` | **51.7%** | **`8.60 ms`** | 10.80 ms |
| `financebench_doc_level` | 1.64 s | 1.24 s | **54.00%** | 54.67% | `-0.67 pp` | **58.0%** | **`7.71 ms`** | 9.21 ms |
| `bright_economics_doc_level` | 10.01 s | 5.30 s | **23.30%** | 27.18% | `-3.88 pp` | **96.4%** | **`8.14 ms`** | 15.28 ms |
| `bright_stackoverflow_doc_level` | 28.31 s | 21.91 s | **39.32%** | 40.17% | `-0.85 pp` | **83.4%** | **`13.83 ms`** | 19.60 ms |
| `bright_robotics_doc_level` | 8.79 s | 4.15 s | **22.77%** | 22.77% | `+0.00 pp` | **91.3%** | **`14.60 ms`** | 19.92 ms |
