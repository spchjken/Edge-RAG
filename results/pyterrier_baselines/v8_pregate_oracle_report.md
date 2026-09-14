# Candidate-Oracle Viability Gate Report (12 Small Benchmarks < 100k Docs)

## 1. Executive Summary & Decision Verdict

- **Overall Safely Addressable Query Rate:** **6.3%**
- **Decision Verdict (§12.4):** **GATE MARGINAL (5% - 10%)**
- **Architectural Consequence:** Consider cheap query-type routing only. Do not build universal context sidecar.

---

## 2. 12-Benchmark Candidate-Oracle Summary Table

| Dataset | Sampled Queries | Safely Addressable Count | Safely Addressable % | Mean Oracle $\Delta$ nDCG@10 | Gate False Rejections | Gate False Acceptances |
|:---|---:|---:|---:|---:|---:|---:|
| `nfcorpus` | 50 | 10 | 20.0% | +0.0048 | 23 | 54 |
| `scifact` | 50 | 4 | 8.0% | -0.0025 | 11 | 33 |
| `arguana` | 50 | 3 | 6.0% | -0.0164 | 0 | 61 |
| `bright_pony` | 50 | 2 | 4.0% | +0.0013 | 0 | 6 |
| `bright_theoremqa_theorems` | 50 | 1 | 2.0% | +0.0011 | 0 | 3 |
| `scidocs` | 50 | 5 | 10.0% | +0.0006 | 15 | 30 |
| `bright_economics` | 50 | 1 | 2.0% | -0.0113 | 0 | 18 |
| `bright_psychology` | 50 | 3 | 6.0% | +0.0020 | 0 | 15 |
| `bright_biology` | 50 | 2 | 4.0% | -0.0106 | 0 | 21 |
| `fiqa` | 50 | 3 | 6.0% | -0.0063 | 9 | 31 |
| `bright_sustainable_living` | 50 | 2 | 4.0% | -0.0098 | 0 | 48 |
| `bright_robotics` | 50 | 2 | 4.0% | -0.0058 | 0 | 23 |

**Macro Average Safely Addressable Rate across 12 Benchmarks:** **6.3%**
