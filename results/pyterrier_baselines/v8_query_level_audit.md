# Provenance & Query-Level Retrieval Audit: Edge-RAG V8 vs BM25 Baseline

## 1. Scientific Provenance & Runtime Environment

- **Git Commit SHA:** `572bd9e47ea36730007b58950e48530c9b153176`
- **Host OS / Runtime:** WSL2 Linux (Ubuntu 24.04), Python 3.12 (`.venv/bin/python3`)
- **Information Retrieval Engine:** PyTerrier 5.11 (Terrier core, JVM heap cap: 4096 MB)
- **Sparse Index Properties:** Single-field unstemmed/Porter default index (`data/cache/terrier_indices/*_default/`)
- **Dense Embedding Sidecar:** BAAI/bge-small-en-v1.5 (CUDA FP16, batch GEMM)
- **Candidate Retrieval Funnel:** Bounded heap depth $K=1,000$
- **Query-Level Tie Tolerance:** $\epsilon = 0.001$ ($|\Delta nDCG@10| \le 0.001$ classified as exact tie)
- **Raw Run Artifacts:** Cached depth-1,000 parquets under `data/cache/runs/{dataset}_{pipeline}.parquet`

---

## 2. Master Query Distribution Table

| Dataset | Tier | Queries | BM25 nDCG@10 | V8 nDCG@10 | $\Delta$ nDCG@10 | Gold in Top-10 % | Ties % (Count) | Gains % (Count) | Drops % (Count) | Dominant Outcome |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `nfcorpus` | < 100k docs | 323 | 0.3282 | 0.3222 | -0.0060 | 70.0% | 83.0% (268) | 5.3% (17) | 11.8% (38) | Drops > Gains (▼) |
| `scifact` | < 100k docs | 300 | 0.2149 | 0.6817 | +0.4668 | 26.7% | 38.7% (116) | 60.0% (180) | 1.3% (4) | Gains > Drops (▲) |
| `arguana` | < 100k docs | 1406 | 0.3662 | 0.3569 | -0.0092 | 76.3% | 77.3% (1087) | 9.2% (129) | 13.5% (190) | Drops > Gains (▼) |
| `bright_pony` | < 100k docs | 112 | 0.0252 | 0.0231 | -0.0020 | 21.4% | 91.1% (102) | 3.6% (4) | 5.4% (6) | Drops > Gains (▼) |
| `bright_theoremqa_theorems` | < 100k docs | 76 | 0.0192 | 0.0161 | -0.0032 | 6.6% | 96.1% (73) | 1.3% (1) | 2.6% (2) | Drops > Gains (▼) |
| `scidocs` | < 100k docs | 1000 | 0.1582 | 0.1506 | -0.0076 | 49.8% | 85.7% (857) | 5.3% (53) | 9.0% (90) | Drops > Gains (▼) |
| `bright_economics` | < 100k docs | 103 | 0.1177 | 0.1114 | -0.0063 | 26.2% | 87.4% (90) | 5.8% (6) | 6.8% (7) | Drops > Gains (▼) |
| `bright_psychology` | < 100k docs | 101 | 0.0926 | 0.0938 | +0.0013 | 21.8% | 89.1% (90) | 6.9% (7) | 4.0% (4) | Gains > Drops (▲) |
| `bright_biology` | < 100k docs | 103 | 0.0912 | 0.0860 | -0.0052 | 28.2% | 90.3% (93) | 3.9% (4) | 5.8% (6) | Drops > Gains (▼) |
| `fiqa` | < 100k docs | 648 | 0.2526 | 0.2446 | -0.0081 | 49.1% | 90.0% (583) | 3.5% (23) | 6.5% (42) | Drops > Gains (▼) |
| `bright_sustainable_living` | < 100k docs | 108 | 0.0981 | 0.0895 | -0.0086 | 30.6% | 84.3% (91) | 5.6% (6) | 10.2% (11) | Drops > Gains (▼) |
| `bright_robotics` | < 100k docs | 101 | 0.0996 | 0.0985 | -0.0012 | 28.7% | 89.1% (90) | 5.9% (6) | 5.0% (5) | Gains > Drops (▲) |
| `bright_stackoverflow` | >= 100k docs | 117 | 0.1561 | 0.1515 | -0.0046 | 32.5% | 87.2% (102) | 4.3% (5) | 8.5% (10) | Drops > Gains (▼) |
| `bright_earth_science` | >= 100k docs | 116 | 0.1201 | 0.1183 | -0.0018 | 31.9% | 81.9% (95) | 9.5% (11) | 8.6% (10) | Gains > Drops (▲) |
| `bright_aops` | >= 100k docs | 111 | 0.0604 | 0.0610 | +0.0007 | 20.7% | 99.1% (110) | 0.9% (1) | 0.0% (0) | Gains > Drops (▲) |
| `bright_theoremqa_questions` | >= 100k docs | 194 | 0.0700 | 0.0731 | +0.0031 | 11.3% | 96.4% (187) | 2.1% (4) | 1.5% (3) | Gains > Drops (▲) |
| `webis_touche2020` | >= 100k docs | 49 | 0.2979 | 0.2781 | -0.0198 | 91.8% | 71.4% (35) | 6.1% (3) | 22.4% (11) | Drops > Gains (▼) |
| `trec_covid` | >= 100k docs | 50 | 0.6295 | 0.5981 | -0.0314 | 100.0% | 66.0% (33) | 6.0% (3) | 28.0% (14) | Drops > Gains (▼) |
| `bright_leetcode` | >= 100k docs | 142 | 0.2496 | 0.2370 | -0.0126 | 41.5% | 88.7% (126) | 2.8% (4) | 8.5% (12) | Drops > Gains (▼) |
| `quora` | >= 100k docs | 10000 | 0.7676 | 0.7450 | -0.0226 | 91.1% | 94.1% (9411) | 1.0% (103) | 4.9% (486) | Drops > Gains (▼) |

### Small Benchmarks (< 100k docs, N=12) Averages:
- **Mean Ties Rate:** 83.5%
- **Mean Gains Rate:** 9.7%
- **Mean Drops Rate:** 6.8%
