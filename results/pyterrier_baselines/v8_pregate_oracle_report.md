# Comprehensive 4-Stage Candidate-Oracle Diagnostic Report (20 Corpora)

## 1. Executive Summary & Macro Decomposition

- **Evaluated Corpora:** 20 benchmarks (12 Small, 7 Medium, 1 Sentinel)
- **Total Sampled Queries:** 1000
- **Stage 1 (Full-Pool Availability Ceiling):** **25.0%** of queries have useful unigrams in the 15k pool
- **Stage 2 (BGE Proposal Addressability):** Top-20 = **13.5%** [95% Bootstrap CI: 5.4% – 23.6%] | Top-100 = **13.7%**
- **Abstaining Oracle Mean $\Delta nDCG@10$:** +0.0080 (All $\ge 0.0$)
- **Forced Expansion Mean $\Delta nDCG@10$:** +0.0080

---

## 2. Master 4-Stage Decomposition Table

| Dataset | Tier | Queries | Stage 1: Pool Avail % | Stage 2: BGE Safe@20 % [Wilson 95%] | Stage 2: BGE Safe@100 % | Stage 3: Gate Recall % | Stage 3: Harm Rej % | Stage 3: Acc Prec % | Stage 4: V8 Safe % |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `nfcorpus` | Small (< 100k) | 50 | 82.0% | 48.0% [34.8%, 61.5%] | 48.0% | 43.1% | 54.5% | 81.5% | 28.0% |
| `scifact` | Small (< 100k) | 50 | 18.0% | 8.0% [3.2%, 18.8%] | 8.0% | 50.0% | 66.7% | 50.0% | 2.0% |
| `arguana` | Small (< 100k) | 50 | 14.0% | 0.0% [0.0%, 7.1%] | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `bright_pony` | Small (< 100k) | 50 | 56.0% | 18.0% [9.8%, 30.8%] | 18.0% | 14.3% | 0.0% | 100.0% | 4.0% |
| `bright_theoremqa_theorems` | Small (< 100k) | 50 | 16.0% | 0.0% [0.0%, 7.1%] | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% |
| `scidocs` | Small (< 100k) | 50 | 18.0% | 8.0% [3.2%, 18.8%] | 8.0% | 0.0% | 100.0% | 0.0% | 2.0% |
| `bright_economics` | Small (< 100k) | 50 | 6.0% | 4.0% [1.1%, 13.5%] | 4.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `bright_psychology` | Small (< 100k) | 50 | 16.0% | 6.0% [2.1%, 16.2%] | 6.0% | 0.0% | 100.0% | 0.0% | 0.0% |
| `bright_biology` | Small (< 100k) | 50 | 12.0% | 4.0% [1.1%, 13.5%] | 4.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `fiqa` | Small (< 100k) | 50 | 44.0% | 12.0% [5.6%, 23.8%] | 12.0% | 0.0% | 71.4% | 0.0% | 0.0% |
| `bright_sustainable_living` | Small (< 100k) | 50 | 16.0% | 2.0% [0.4%, 10.5%] | 2.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `bright_robotics` | Small (< 100k) | 50 | 14.0% | 4.0% [1.1%, 13.5%] | 4.0% | 0.0% | 0.0% | 0.0% | 2.0% |
| `bright_stackoverflow` | Medium (100k-500k) | 50 | 8.0% | 2.0% [0.4%, 10.5%] | 2.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `bright_earth_science` | Medium (100k-500k) | 50 | 14.0% | 6.0% [2.1%, 16.2%] | 6.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `bright_aops` | Medium (100k-500k) | 50 | 4.0% | 0.0% [0.0%, 7.1%] | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `bright_theoremqa_questions` | Medium (100k-500k) | 50 | 10.0% | 0.0% [0.0%, 7.1%] | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `bright_leetcode` | Medium (100k-500k) | 50 | 4.0% | 0.0% [0.0%, 7.1%] | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `trec_covid` | Medium (100k-500k) | 50 | 60.0% | 80.0% [67.0%, 88.8%] | 80.0% | 31.2% | 77.8% | 71.4% | 34.0% |
| `webis_touche2020` | Medium (100k-500k) | 49 | 69.4% | 61.2% [47.2%, 73.6%] | 65.3% | 19.6% | 64.5% | 45.0% | 16.3% |
| `quora` | Sentinel (> 500k) | 50 | 18.0% | 6.0% [2.1%, 16.2%] | 6.0% | 50.0% | 50.0% | 50.0% | 2.0% |

### Small (< 100k) Tier Averages (N=12):
- **Stage 1 Pool Availability:** 26.0%
- **Stage 2 BGE Safe@20:** 9.5%
- **Stage 2 BGE Safe@100:** 9.5%

### Medium (100k-500k) Tier Averages (N=7):
- **Stage 1 Pool Availability:** 24.2%
- **Stage 2 BGE Safe@20:** 21.3%
- **Stage 2 BGE Safe@100:** 21.9%

---

## 3. DPH Confirmatory Transfer Check

| Dataset | Queries | DPH Baseline nDCG@10 | DPH Oracle nDCG@10 | DPH Oracle $\Delta$ |
|:---|---:|---:|---:|---:|
| `nfcorpus` | 50 | 0.0210 | 0.2746 | +0.2536 |
| `trec_covid` | 50 | 0.3073 | 0.6272 | +0.3199 |
| `bright_stackoverflow` | 50 | 0.0671 | 0.1569 | +0.0899 |
| `scifact` | 50 | 0.0598 | 0.7222 | +0.6624 |
