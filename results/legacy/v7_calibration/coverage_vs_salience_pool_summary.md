# 🔬 Coverage (FPS) vs Salience vs IDF Pool Strategy Empirical Report

- **Generated:** 2026-09-02 04:05:41
- **Scope:** 10 Document-Level Benchmarks (3,237 queries, 29,133 total evaluations)
- **Bailout Status:** Permanently Disabled (Zero prefix dilution, pure core V7)

---

## 🏆 1. Macro Retrieval Performance Comparison

| Strategy | Pool Size N | **Strict@10** | **DocRec@10** | Strict@50 | DocRec@50 | MRR@10 | In-Pool Hit % | Total TTI (s) | Total Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`COVERAGE (FPS)`** | 500 | **62.55%** | **49.46%** | 74.74% | 60.96% | **0.4806** | **71.8%** | 7.60s | **5.99 ms** ⚡ |
| **`COVERAGE (FPS)`** | 1000 | **62.73%** | **49.54%** | 74.79% | 61.01% | **0.4813** | **71.8%** | 7.40s | **8.10 ms** ⚡ |
| **`COVERAGE (FPS)`** | 2500 | **62.73%** | **49.55%** | 74.91% | 61.07% | **0.4808** | **71.8%** | 7.68s | **15.09 ms** |
| **`SALIENCE`** | 500 | **62.84%** | **49.71%** | 74.83% | 61.06% | **0.4797** | **71.8%** | 7.23s | **7.13 ms** ⚡ |
| **`SALIENCE`** | **1000** | **`63.16%`** 👑 | **`49.80%`** 👑 | 74.54% | 60.90% | **`0.4812`** | **`71.8%`** | 7.28s | **`10.12 ms`** ⚡ |
| **`SALIENCE`** | 2500 | **62.91%** | **49.74%** | 74.64% | 60.97% | **0.4799** | **71.8%** | 7.25s | **19.07 ms** |
| **`IDF`** | 500 | **62.71%** | **49.61%** | 74.69% | 61.08% | **0.4800** | **71.8%** | 7.26s | **6.67 ms** ⚡ |
| **`IDF`** | 1000 | **62.73%** | **49.59%** | 74.72% | 61.08% | **0.4805** | **71.8%** | 7.27s | **9.25 ms** |
| **`IDF`** | 2500 | **62.73%** | **49.62%** | 74.69% | 61.06% | **0.4802** | **71.8%** | 7.30s | **16.85 ms** |

---

## ⏱️ 2. Dedicated TTI Indexing Breakdown (Seconds)

$$\text{Total TTI} = t_{\text{bm25\_idx}} + t_{\text{vocab\_salience}} + t_{\text{matrix\_fps}}$$

| Strategy | Pool Size N | BM25 Indexing ($t_{\text{bm25}}$) | Vocab Filtering ($t_{\text{vocab}}$) | Dense Matrix / FPS ($t_{\text{matrix}}$) | **Total TTI ($t_{\text{tti}}$)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`COVERAGE (FPS)`** | 500 | 5.44 s | 0.05 s | 2.11 s | **7.60 s** |
| **`COVERAGE (FPS)`** | 1000 | 5.41 s | 0.05 s | 1.95 s | **7.40 s** |
| **`COVERAGE (FPS)`** | 2500 | 5.43 s | 0.04 s | 2.21 s | **7.68 s** |
| **`SALIENCE`** | 500 | 5.43 s | 0.05 s | 1.76 s | **7.23 s** |
| **`SALIENCE`** | 1000 | 5.43 s | 0.05 s | 1.80 s | **7.28 s** |
| **`SALIENCE`** | 2500 | 5.43 s | 0.04 s | 1.78 s | **7.25 s** |
| **`IDF`** | 500 | 5.43 s | 0.05 s | 1.79 s | **7.26 s** |
| **`IDF`** | 1000 | 5.43 s | 0.04 s | 1.80 s | **7.27 s** |
| **`IDF`** | 2500 | 5.43 s | 0.04 s | 1.83 s | **7.30 s** |

---

## ⚡ 3. Query-Time Sub-Timer Latency Breakdown (Milliseconds)

| Strategy | Pool N | Anchor Total ($t_{\text{anchor}}$) | Probing GEMM ($t_{\text{prob}}$) | IT-MPE ($t_{\text{itmpe}}$) | BM25 ($t_{\text{bm25}}$) | **Total Latency** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`COVERAGE`** | 500 | 3.25 ms | 0.12 ms | 0.43 ms | 2.12 ms | **5.99 ms** ⚡ |
| **`COVERAGE`** | 1000 | 3.27 ms | 0.11 ms | 0.78 ms | 3.86 ms | **8.10 ms** ⚡ |
| **`COVERAGE`** | 2500 | 3.37 ms | 0.13 ms | 1.96 ms | 9.54 ms | **15.09 ms** |
| **`SALIENCE`** | 500 | 3.24 ms | 0.11 ms | 0.48 ms | 3.24 ms | **7.13 ms** ⚡ |
| **`SALIENCE`** | **1000** | **3.27 ms** ⚡ | **0.11 ms** | **0.86 ms** | **5.80 ms** | **`10.12 ms`** ⚡ |
| **`SALIENCE`** | 2500 | 3.41 ms | 0.13 ms | 2.05 ms | 13.39 ms | **19.07 ms** |
| **`IDF`** | 500 | 3.22 ms | 0.11 ms | 0.51 ms | 2.77 ms | **6.67 ms** ⚡ |
| **`IDF`** | 1000 | 3.26 ms | 0.11 ms | 0.93 ms | 4.87 ms | **9.25 ms** ⚡ |
| **`IDF`** | 2500 | 3.35 ms | 0.12 ms | 2.18 ms | 11.10 ms | **16.85 ms** |

---

## 📊 4. Per-Dataset Breakdown (Coverage vs Salience at N=1000)

| Dataset | Coverage Strict@10 | Salience Strict@10 | Delta (pp) | Coverage In-Pool % | Salience In-Pool % | Coverage Latency | Salience Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `enterpriserag_doc_level` | **93.83%** | **93.62%** | +0.21 pp | 42.8% | 42.8% | 9.30 ms | 9.53 ms |
| `liverag_doc_level` | **97.65%** | **97.77%** | -0.12 pp | 56.7% | 56.7% | 5.58 ms | 7.66 ms |
| `beir_scifact_doc_level` | **81.00%** | **81.00%** | +0.00 pp | 76.6% | 76.6% | 5.51 ms | 7.22 ms |
| `beir_nfcorpus_doc_level` | **69.35%** | **68.42%** | +0.93 pp | 83.3% | 83.3% | 2.90 ms | 4.87 ms |
| `beir_fiqa_doc_level` | **46.76%** | **46.76%** | +0.00 pp | 77.4% | 77.4% | 4.43 ms | 8.88 ms |
| `multihop_rag_doc_level` | **99.38%** | **99.29%** | +0.09 pp | 51.7% | 51.7% | 8.62 ms | 9.92 ms |
| `financebench_doc_level` | **54.00%** | **54.67%** | -0.67 pp | 58.0% | 58.0% | 7.80 ms | 8.45 ms |
| `bright_economics_doc_level` | **24.27%** | **27.18%** | -2.91 pp | 96.4% | 96.4% | 8.40 ms | 11.49 ms |
| `bright_stackoverflow_doc_level` | **39.32%** | **40.17%** | -0.85 pp | 83.4% | 83.4% | 14.00 ms | 16.50 ms |
| `bright_robotics_doc_level` | **21.78%** | **22.77%** | -0.99 pp | 91.3% | 91.3% | 14.45 ms | 16.63 ms |
