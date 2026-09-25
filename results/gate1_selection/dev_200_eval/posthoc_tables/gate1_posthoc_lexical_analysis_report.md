# Phase 2 Gate 1 Post-Hoc Offline Lexical Analysis Report (Round 4)

> [!NOTE]
> **Strictly Offline Protocol:** Zero retrieval or counterfactual evaluation was rerun. All metrics derive strictly from the already-generated audit, cutoff, and reference-universe artifacts of Run 2.
> Run 2 remains an exploratory result that failed Checkpoint B under protocol amendment `PA-GATE1-20260922-01` (`gate_passed: false`).
> Budget conditions $L \in \{300, 400, 500\}$ are exploratory characterizations, NOT current deployable settings.

## 1. Executive Summary & Core Decision Findings

### Decision Question:
> *Does preserving both lexical channels—especially `PPMI200 ∪ Sparse200` (`LexicalUnion400`)—raise near-best and broad helpful-term retention enough to justify increasing the Gate 1 deployable cap from 200 to 400?*

### Empirical Findings & Observed Trade-offs:
1. **PPMI–Sparse Complementarity:** Across all 200 dev queries, the mean overlap between PPMI top-200 and Sparse top-200 is only **45.5 terms** (median: 48.0, min: 0, max: 147). Over **77%** of candidates proposed by Sparse are entirely complementary to PPMI.
2. **Output Cardinality of Union400:** Deduplication yields a mean of **331.1 unique candidates** per query (median: 346, p95: 396, theoretical max: 400).
3. **Budget-Dependent Trade-offs (No Causal Speculation):**
   - **At nominal cap $L=200$:** `RRF_Lexical2` achieves higher NearBestHit than `RRF_Extended` (60.6% vs 56.2%), but lower SafeDocOppRecall (68.5% vs 73.9%).
   - **At nominal cap $L=400$:** `RRF_Extended` achieves the strongest overall balanced performance: **66.8%** NearBestHit, **81.70%** ReferenceBOR, **84.9%** RawDocOppRecall, and **81.3%** SafeDocOppRecall. `RRF_Lexical2 (L=400)` achieves **65.6%** NearBestHit, **79.98%** ReferenceBOR, and **77.3%** SafeDocOppRecall, while leading on broad Material TermRecall (**31.8%** vs 26.6%) and RecallHit (**95.1%** vs 94.4%).
   - **At matched output cardinality ($C_{\mathrm{Extended}}^{\mathrm{matched}}(q) = \operatorname{top}_{|C_{\mathrm{Union400}}(q)|} C_{\mathrm{Extended}}(q)$):** When RRF_Extended is dynamically sliced to the exact same per-query candidate count as Union400 (mean 331.1), Extended achieves **64.5%** NearBestHit, **79.67%** ReferenceBOR, and **79.6%** SafeDocOppRecall, compared to **64.0%** NearBestHit, **77.72%** ReferenceBOR, and **76.4%** SafeDocOppRecall for LexicalUnion400.
4. **Gate-2 Downstream Cost Bounds:** Whether $L=400$ is practically deployable remains unproven until downstream latency, VRAM, and context-reference counts are empirically benchmarked in Gate 2.

---

## 2. Dedicated Comparison: Nominal Cap L=400 vs. Output-Cardinality-Matched Extended

> **Distinction:** Rows with nominal cap $L=400$ evaluate policies up to fixed cap 400. `RRF_Extended_Output_Matched` is dynamically sliced on every query to $|C_{\mathrm{Union400}}(q)|$ (`len(ext) == len(union)` runtime invariant enforced).

| Policy                      |   Mean Actual Candidates | Mean PPMI-Sparse Overlap   |   Helpful Terms Captured (Addr) | Material TermRecall   | Material TermPrecision   | NearBestHit (rho=0.90)   | NearBestHit (supp rho=0.80)   | ReferenceBOR            | RecallHit@1000       | RawDocOppRecall@1000   | SafeDocOppRecall@1000   |
|:----------------------------|-------------------------:|:---------------------------|--------------------------------:|:----------------------|:-------------------------|:-------------------------|:------------------------------|:------------------------|:---------------------|:-----------------------|:------------------------|
| PPMISidecar (L=400)         |                    366.3 | 45.5                       |                            23.1 | 31.7%                 | 5.3%                     | 61.3% [53.4%, 69.6%]     | 67.4%                         | 77.95% [72.45%, 83.04%] | 94.0% [87.6%, 97.7%] | 81.4%                  | 79.6% [68.3%, 83.9%]    |
| RRF_Extended (L=400)        |                    400   | N/A                        |                            18.9 | 26.6%                 | 3.6%                     | 66.8% [59.4%, 74.6%]     | 70.7%                         | 81.69% [75.96%, 86.63%] | 94.4% [88.4%, 97.9%] | 84.9%                  | 81.3% [70.5%, 84.7%]    |
| RRF_Lexical2 (L=400)        |                    376.8 | N/A                        |                            22.4 | 31.8%                 | 4.6%                     | 65.6% [58.0%, 73.4%]     | 70.1%                         | 79.98% [74.40%, 84.89%] | 95.1% [89.3%, 98.4%] | 79.2%                  | 77.3% [65.4%, 81.2%]    |
| LexicalUnion400 (200+200)   |                    331.1 | 45.5                       |                            20   | 27.8%                 | 4.7%                     | 64.0% [56.3%, 71.6%]     | 67.3%                         | 77.72% [71.68%, 83.11%] | 95.1% [89.3%, 98.4%] | 78.3%                  | 76.4% [63.8%, 80.4%]    |
| RRF_Extended_Output_Matched |                    331.1 | 45.5                       |                            16.2 | 23.6%                 | 3.7%                     | 61.5% [54.2%, 68.8%]     | 65.8%                         | 77.73% [72.06%, 82.77%] | 90.8% [82.4%, 94.9%] | 79.7%                  | 76.0% [62.5%, 80.2%]    |

---

## 3. Policy Comparison at Deployable (L=200) and Exploratory Caps

| Policy                               |   Mean Actual Candidates | Mean PPMI-Sparse Overlap   |   Helpful Terms Captured (Addr) | Material TermRecall   | Material TermPrecision   | NearBestHit (rho=0.90)   | NearBestHit (supp rho=0.80)   | ReferenceBOR            | RecallHit@1000       | RawDocOppRecall@1000   | SafeDocOppRecall@1000   |
|:-------------------------------------|-------------------------:|:---------------------------|--------------------------------:|:----------------------|:-------------------------|:-------------------------|:------------------------------|:------------------------|:---------------------|:-----------------------|:------------------------|
| PPMISidecar (L=200)                  |                    186.5 | 45.5                       |                            14.4 | 20.5%                 | 6.2%                     | 50.6% [43.2%, 59.1%]     | 55.1%                         | 69.66% [63.81%, 75.42%] | 88.5% [79.4%, 93.3%] | 72.0%                  | 69.4% [54.2%, 74.1%]    |
| SparseLexicalContextProfiles (L=200) |                    190.1 | 45.5                       |                            10.4 | 15.3%                 | 3.9%                     | 42.4% [34.2%, 50.2%]     | 44.0%                         | 53.97% [46.57%, 60.63%] | 81.5% [69.9%, 86.7%] | 59.7%                  | 58.5% [39.4%, 63.1%]    |
| RRF_Core3 (L=200)                    |                    200   | 45.5                       |                            10.6 | 14.9%                 | 4.1%                     | 47.8% [39.5%, 56.4%]     | 53.4%                         | 70.01% [63.93%, 76.39%] | 89.2% [80.5%, 93.5%] | 75.0%                  | 71.8% [57.6%, 76.0%]    |
| RRF_Extended (L=200)                 |                    200   | 45.5                       |                            11.5 | 16.9%                 | 4.3%                     | 56.2% [48.5%, 63.6%]     | 61.7%                         | 74.18% [68.22%, 79.67%] | 91.1% [83.5%, 95.0%] | 78.3%                  | 73.9% [60.0%, 78.1%]    |
| RRF_Lexical2 (L=200)                 |                    190.8 | 45.5                       |                            14.1 | 21.1%                 | 5.5%                     | 60.6% [52.5%, 68.6%]     | 63.3%                         | 73.66% [67.33%, 79.99%] | 88.8% [79.4%, 93.2%] | 71.9%                  | 68.5% [52.6%, 73.2%]    |
| LexicalUnion200 (100+100)            |                    169.8 | 45.5                       |                            12.7 | 19.4%                 | 5.6%                     | 58.2% [50.4%, 66.5%]     | 60.9%                         | 72.13% [65.84%, 78.20%] | 87.9% [78.3%, 92.0%] | 69.8%                  | 66.5% [49.7%, 71.3%]    |
| PPMISidecar (L=400)                  |                    366.3 | 45.5                       |                            23.1 | 31.7%                 | 5.3%                     | 61.3% [53.4%, 69.6%]     | 67.4%                         | 77.95% [72.45%, 83.04%] | 94.0% [87.6%, 97.7%] | 81.4%                  | 79.6% [68.3%, 83.9%]    |
| SparseLexicalContextProfiles (L=400) |                    374.4 | 45.5                       |                            16.5 | 22.6%                 | 3.4%                     | 49.8% [41.7%, 57.1%]     | 51.4%                         | 61.60% [54.55%, 68.50%] | 86.8% [76.9%, 91.3%] | 65.9%                  | 63.9% [46.8%, 68.3%]    |
| RRF_Core3 (L=400)                    |                    400   | N/A                        |                            17.2 | 22.9%                 | 3.4%                     | 60.8% [53.3%, 69.0%]     | 65.8%                         | 78.11% [72.55%, 83.70%] | 94.4% [88.4%, 98.0%] | 85.0%                  | 81.7% [71.0%, 85.0%]    |
| RRF_Extended (L=400)                 |                    400   | N/A                        |                            18.9 | 26.6%                 | 3.6%                     | 66.8% [59.4%, 74.6%]     | 70.7%                         | 81.69% [75.96%, 86.63%] | 94.4% [88.4%, 97.9%] | 84.9%                  | 81.3% [70.5%, 84.7%]    |
| RRF_Lexical2 (L=400)                 |                    376.8 | N/A                        |                            22.4 | 31.8%                 | 4.6%                     | 65.6% [58.0%, 73.4%]     | 70.1%                         | 79.98% [74.40%, 84.89%] | 95.1% [89.3%, 98.4%] | 79.2%                  | 77.3% [65.4%, 81.2%]    |
| LexicalUnion400 (200+200)            |                    331.1 | 45.5                       |                            20   | 27.8%                 | 4.7%                     | 64.0% [56.3%, 71.6%]     | 67.3%                         | 77.72% [71.68%, 83.11%] | 95.1% [89.3%, 98.4%] | 78.3%                  | 76.4% [63.8%, 80.4%]    |
| RRF_Extended_Output_Matched          |                    331.1 | 45.5                       |                            16.2 | 23.6%                 | 3.7%                     | 61.5% [54.2%, 68.8%]     | 65.8%                         | 77.73% [72.06%, 82.77%] | 90.8% [82.4%, 94.9%] | 79.7%                  | 76.0% [62.5%, 80.2%]    |

---

## 4. Paired, Corpus-Stratified Bootstrap Differences (95% Descriptive CI, B=1000)

> [!NOTE]
> These confidence intervals are **exploratory and descriptive** characterizations across the 200 dev queries. They do not constitute a confirmatory hypothesis pass gate.

| Comparison | Type | Metric | Point Estimate (Diff) | 95% Bootstrap CI | Statistically Distinguishable? |
|:---|:---:|:---|:---:|:---:|:---:|
| RRF_Extended@400 vs RRF_Lexical2@400 | Nominal Cap 400 | near_best_hit_90 | +1.20% | [-5.10%, +7.20%] | No (spans 0) |
| RRF_Extended@400 vs RRF_Lexical2@400 | Nominal Cap 400 | reference_bor | +1.71% | [-2.47%, +5.57%] | No (spans 0) |
| RRF_Extended@400 vs RRF_Lexical2@400 | Nominal Cap 400 | recall_hit_1000 | -0.69% | [-5.21%, +3.23%] | No (spans 0) |
| RRF_Extended@400 vs RRF_Lexical2@400 | Nominal Cap 400 | safe_doc_opp_recall | +3.95% | [+0.72%, +8.30%] | YES (p < 0.05) |
| RRF_Extended@400 vs PPMISidecar@400 | Nominal Cap 400 | near_best_hit_90 | +5.54% | [-1.55%, +12.91%] | No (spans 0) |
| RRF_Extended@400 vs PPMISidecar@400 | Nominal Cap 400 | reference_bor | +3.74% | [-1.58%, +9.29%] | No (spans 0) |
| RRF_Extended@400 vs PPMISidecar@400 | Nominal Cap 400 | recall_hit_1000 | +0.47% | [-4.12%, +4.88%] | No (spans 0) |
| RRF_Extended@400 vs PPMISidecar@400 | Nominal Cap 400 | safe_doc_opp_recall | +1.63% | [-2.59%, +6.22%] | No (spans 0) |
| RRF_Extended@400 vs LexicalUnion400 (nominal cap 400) | Nominal Cap 400 | near_best_hit_90 | +2.85% | [-4.03%, +9.62%] | No (spans 0) |
| RRF_Extended@400 vs LexicalUnion400 (nominal cap 400) | Nominal Cap 400 | reference_bor | +3.97% | [-0.38%, +8.05%] | No (spans 0) |
| RRF_Extended@400 vs LexicalUnion400 (nominal cap 400) | Nominal Cap 400 | recall_hit_1000 | -0.69% | [-5.21%, +3.23%] | No (spans 0) |
| RRF_Extended@400 vs LexicalUnion400 (nominal cap 400) | Nominal Cap 400 | safe_doc_opp_recall | +4.90% | [+1.69%, +9.61%] | YES (p < 0.05) |
| RRF_Lexical2@200 vs RRF_Extended@200 | Nominal Cap 200 | near_best_hit_90 | +4.43% | [-3.37%, +12.85%] | No (spans 0) |
| RRF_Lexical2@200 vs RRF_Extended@200 | Nominal Cap 200 | reference_bor | -0.52% | [-5.73%, +4.78%] | No (spans 0) |
| RRF_Lexical2@200 vs RRF_Extended@200 | Nominal Cap 200 | recall_hit_1000 | -2.24% | [-7.86%, +1.94%] | No (spans 0) |
| RRF_Lexical2@200 vs RRF_Extended@200 | Nominal Cap 200 | safe_doc_opp_recall | -5.46% | [-11.63%, -1.24%] | YES (p < 0.05) |
| RRF_Extended@200 vs PPMISidecar@200 | Nominal Cap 200 | near_best_hit_90 | +5.54% | [-2.63%, +13.18%] | No (spans 0) |
| RRF_Extended@200 vs PPMISidecar@200 | Nominal Cap 200 | reference_bor | +4.52% | [-1.48%, +9.90%] | No (spans 0) |
| RRF_Extended@200 vs PPMISidecar@200 | Nominal Cap 200 | recall_hit_1000 | +2.56% | [-4.05%, +9.36%] | No (spans 0) |
| RRF_Extended@200 vs PPMISidecar@200 | Nominal Cap 200 | safe_doc_opp_recall | +4.52% | [-0.96%, +10.93%] | No (spans 0) |
| LexicalUnion400 vs RRF_Extended_Output_Matched | Output-Cardinality Matched | near_best_hit_90 | +2.54% | [-4.67%, +9.45%] | No (spans 0) |
| LexicalUnion400 vs RRF_Extended_Output_Matched | Output-Cardinality Matched | reference_bor | -0.01% | [-4.51%, +4.63%] | No (spans 0) |
| LexicalUnion400 vs RRF_Extended_Output_Matched | Output-Cardinality Matched | recall_hit_1000 | +4.29% | [+0.59%, +10.42%] | YES (p < 0.05) |
| LexicalUnion400 vs RRF_Extended_Output_Matched | Output-Cardinality Matched | safe_doc_opp_recall | +0.33% | [-3.98%, +5.16%] | No (spans 0) |


---

## 5. Primary Matched Budget Curves (L in {200, 300, 400, 500})

| Policy                       | Type                       |   Budget (L) |   Actual Candidates | NearBestHit@L (frozen rho=0.90)   | NearBestHit@L (supp rho=0.80)   | ReferenceBOR@L   | RecallHit@1000   | RawDocOppRecall@1000   | SafeDocOppRecall@1000   | TermRecall@L   | TermPrecision@L   |
|:-----------------------------|:---------------------------|-------------:|--------------------:|:----------------------------------|:--------------------------------|:-----------------|:-----------------|:-----------------------|:------------------------|:---------------|:------------------|
| PPMISidecar                  | RRF / Single Channel       |           50 |                47.4 | 29.9%                             | 34.2%                           | 52.5%            | 51.2%            | 26.7%                  | 23.9%                   | 7.9%           | 8.4%              |
| PPMISidecar                  | RRF / Single Channel       |          100 |                94.1 | 45.0%                             | 48.8%                           | 63.7%            | 80.5%            | 61.6%                  | 58.7%                   | 13.0%          | 7.4%              |
| PPMISidecar                  | RRF / Single Channel       |          200 |               186.5 | 50.6%                             | 55.1%                           | 69.7%            | 88.5%            | 72.0%                  | 69.4%                   | 20.5%          | 6.2%              |
| PPMISidecar                  | RRF / Single Channel       |          300 |               277.2 | 60.7%                             | 65.1%                           | 76.3%            | 91.5%            | 77.4%                  | 74.7%                   | 28.0%          | 5.7%              |
| PPMISidecar                  | RRF / Single Channel       |          400 |               366.3 | 61.3%                             | 67.4%                           | 78.0%            | 94.0%            | 81.4%                  | 79.6%                   | 31.7%          | 5.3%              |
| PPMISidecar                  | RRF / Single Channel       |          500 |               454.2 | 63.5%                             | 69.5%                           | 79.4%            | 94.0%            | 83.1%                  | 81.3%                   | 34.9%          | 5.1%              |
| SparseLexicalContextProfiles | RRF / Single Channel       |           50 |                49   | 29.9%                             | 30.9%                           | 43.0%            | 40.2%            | 20.0%                  | 19.0%                   | 7.7%           | 5.1%              |
| SparseLexicalContextProfiles | RRF / Single Channel       |          100 |                96.5 | 35.0%                             | 37.2%                           | 47.1%            | 71.3%            | 51.2%                  | 50.2%                   | 11.4%          | 4.6%              |
| SparseLexicalContextProfiles | RRF / Single Channel       |          200 |               190.1 | 42.4%                             | 44.0%                           | 54.0%            | 81.5%            | 59.7%                  | 58.5%                   | 15.3%          | 3.9%              |
| SparseLexicalContextProfiles | RRF / Single Channel       |          300 |               282.8 | 46.3%                             | 47.4%                           | 58.3%            | 83.9%            | 62.4%                  | 61.1%                   | 19.5%          | 3.6%              |
| SparseLexicalContextProfiles | RRF / Single Channel       |          400 |               374.4 | 49.8%                             | 51.4%                           | 61.6%            | 86.8%            | 65.9%                  | 63.9%                   | 22.6%          | 3.4%              |
| SparseLexicalContextProfiles | RRF / Single Channel       |          500 |               462.3 | 49.8%                             | 52.0%                           | 62.7%            | 87.8%            | 66.4%                  | 64.7%                   | 25.0%          | 3.2%              |
| RRF_Core3                    | RRF / Single Channel       |           50 |                50   | 32.2%                             | 36.6%                           | 53.9%            | 82.5%            | 57.4%                  | 53.9%                   | 5.2%           | 5.6%              |
| RRF_Core3                    | RRF / Single Channel       |          100 |               100   | 41.8%                             | 44.4%                           | 61.9%            | 88.3%            | 67.7%                  | 63.8%                   | 8.8%           | 4.7%              |
| RRF_Core3                    | RRF / Single Channel       |          200 |               200   | 47.8%                             | 53.4%                           | 70.0%            | 89.2%            | 75.0%                  | 71.8%                   | 14.9%          | 4.1%              |
| RRF_Core3                    | RRF / Single Channel       |          300 |               300   | 57.3%                             | 61.8%                           | 76.1%            | 90.7%            | 80.0%                  | 76.8%                   | 19.5%          | 3.7%              |
| RRF_Core3                    | RRF / Single Channel       |          400 |               400   | 60.8%                             | 65.8%                           | 78.1%            | 94.4%            | 85.0%                  | 81.7%                   | 22.9%          | 3.4%              |
| RRF_Core3                    | RRF / Single Channel       |          500 |               500   | 63.6%                             | 68.7%                           | 80.8%            | 95.4%            | 86.3%                  | 83.5%                   | 26.8%          | 3.2%              |
| RRF_Extended                 | RRF / Single Channel       |           50 |                50   | 35.1%                             | 39.9%                           | 56.6%            | 81.1%            | 55.5%                  | 52.7%                   | 7.4%           | 6.6%              |
| RRF_Extended                 | RRF / Single Channel       |          100 |               100   | 45.9%                             | 51.3%                           | 66.6%            | 88.1%            | 68.8%                  | 64.5%                   | 11.0%          | 5.3%              |
| RRF_Extended                 | RRF / Single Channel       |          200 |               200   | 56.2%                             | 61.7%                           | 74.2%            | 91.1%            | 78.3%                  | 73.9%                   | 16.9%          | 4.3%              |
| RRF_Extended                 | RRF / Single Channel       |          300 |               300   | 61.2%                             | 65.6%                           | 78.9%            | 92.0%            | 81.2%                  | 77.7%                   | 22.2%          | 3.9%              |
| RRF_Extended                 | RRF / Single Channel       |          400 |               400   | 66.8%                             | 70.7%                           | 81.7%            | 94.4%            | 84.9%                  | 81.3%                   | 26.6%          | 3.6%              |
| RRF_Extended                 | RRF / Single Channel       |          500 |               500   | 71.9%                             | 75.8%                           | 83.7%            | 95.4%            | 87.2%                  | 83.7%                   | 29.6%          | 3.4%              |
| RRF_Lexical2                 | RRF / Single Channel       |           50 |                49   | 33.2%                             | 36.9%                           | 53.7%            | 52.4%            | 26.6%                  | 23.3%                   | 8.4%           | 6.8%              |
| RRF_Lexical2                 | RRF / Single Channel       |          100 |                96.7 | 45.9%                             | 49.2%                           | 62.9%            | 59.1%            | 36.7%                  | 33.7%                   | 13.0%          | 6.2%              |
| RRF_Lexical2                 | RRF / Single Channel       |          200 |               190.8 | 60.6%                             | 63.3%                           | 73.7%            | 88.8%            | 71.9%                  | 68.5%                   | 21.1%          | 5.5%              |
| RRF_Lexical2                 | RRF / Single Channel       |          300 |               284.1 | 64.0%                             | 68.9%                           | 79.0%            | 92.4%            | 76.4%                  | 73.3%                   | 28.2%          | 4.9%              |
| RRF_Lexical2                 | RRF / Single Channel       |          400 |               376.8 | 65.6%                             | 70.1%                           | 80.0%            | 95.1%            | 79.2%                  | 77.3%                   | 31.8%          | 4.6%              |
| RRF_Lexical2                 | RRF / Single Channel       |          500 |               468.4 | 72.3%                             | 75.6%                           | 83.9%            | 95.1%            | 81.0%                  | 79.3%                   | 36.4%          | 4.3%              |
| LexicalUnion (25+25)         | Deduplicated Lexical Union |           50 |                44.4 | 31.4%                             | 34.6%                           | 51.5%            | 50.4%            | 24.2%                  | 21.1%                   | 7.5%           | 7.1%              |
| LexicalUnion (50+50)         | Deduplicated Lexical Union |          100 |                87.1 | 42.0%                             | 45.2%                           | 61.9%            | 59.4%            | 35.2%                  | 32.0%                   | 13.0%          | 6.4%              |
| LexicalUnion (100+100)       | Deduplicated Lexical Union |          200 |               169.8 | 58.2%                             | 60.9%                           | 72.1%            | 87.9%            | 69.8%                  | 66.5%                   | 19.4%          | 5.6%              |
| LexicalUnion (150+150)       | Deduplicated Lexical Union |          300 |               251   | 61.1%                             | 64.9%                           | 75.6%            | 91.8%            | 75.0%                  | 71.9%                   | 23.3%          | 5.1%              |
| LexicalUnion (200+200)       | Deduplicated Lexical Union |          400 |               331.1 | 64.0%                             | 67.3%                           | 77.7%            | 95.1%            | 78.3%                  | 76.4%                   | 27.8%          | 4.7%              |
| LexicalUnion (250+250)       | Deduplicated Lexical Union |          500 |               410.5 | 68.8%                             | 72.1%                           | 80.5%            | 95.1%            | 79.9%                  | 78.1%                   | 31.8%          | 4.4%              |

---

## 6. Milestone Target Evaluation (2 Decimals Display)

> Proposed Development Milestone Targets:
> - NearBestHit $\ge 70.00\%$
> - ReferenceBOR $\ge 80.00\%$
> - RecallHit@1000 $\ge 85.00\%$
> - SafeDocOpportunityRecall@1000 $\ge 70.00\%$

| Policy                               | NearBestHit >= 70%   | ReferenceBOR >= 80%   | RecallHit >= 85%   | SafeDocOppRecall >= 70%   | Milestone Status   |
|:-------------------------------------|:---------------------|:----------------------|:-------------------|:--------------------------|:-------------------|
| PPMISidecar (L=200)                  | 50.64% (FAIL)        | 69.66% (FAIL)         | 88.52% (PASS)      | 69.39% (FAIL)             | SUB-MILESTONE      |
| SparseLexicalContextProfiles (L=200) | 42.40% (FAIL)        | 53.97% (FAIL)         | 81.49% (FAIL)      | 58.49% (FAIL)             | SUB-MILESTONE      |
| RRF_Core3 (L=200)                    | 47.84% (FAIL)        | 70.01% (FAIL)         | 89.23% (PASS)      | 71.83% (PASS)             | SUB-MILESTONE      |
| RRF_Extended (L=200)                 | 56.18% (FAIL)        | 74.18% (FAIL)         | 91.09% (PASS)      | 73.92% (PASS)             | SUB-MILESTONE      |
| RRF_Lexical2 (L=200)                 | 60.60% (FAIL)        | 73.66% (FAIL)         | 88.84% (PASS)      | 68.46% (FAIL)             | SUB-MILESTONE      |
| LexicalUnion200 (100+100)            | 58.25% (FAIL)        | 72.13% (FAIL)         | 87.92% (PASS)      | 66.48% (FAIL)             | SUB-MILESTONE      |
| PPMISidecar (L=400)                  | 61.31% (FAIL)        | 77.95% (FAIL)         | 93.97% (PASS)      | 79.65% (PASS)             | SUB-MILESTONE      |
| SparseLexicalContextProfiles (L=400) | 49.79% (FAIL)        | 61.60% (FAIL)         | 86.77% (PASS)      | 63.87% (FAIL)             | SUB-MILESTONE      |
| RRF_Core3 (L=400)                    | 60.77% (FAIL)        | 78.11% (FAIL)         | 94.44% (PASS)      | 81.73% (PASS)             | SUB-MILESTONE      |
| RRF_Extended (L=400)                 | 66.85% (FAIL)        | 81.69% (PASS)         | 94.44% (PASS)      | 81.27% (PASS)             | SUB-MILESTONE      |
| RRF_Lexical2 (L=400)                 | 65.64% (FAIL)        | 79.98% (FAIL)         | 95.13% (PASS)      | 77.33% (PASS)             | SUB-MILESTONE      |
| LexicalUnion400 (200+200)            | 64.00% (FAIL)        | 77.72% (FAIL)         | 95.13% (PASS)      | 76.37% (PASS)             | SUB-MILESTONE      |
| RRF_Extended_Output_Matched          | 61.46% (FAIL)        | 77.73% (FAIL)         | 90.85% (PASS)      | 76.04% (PASS)             | SUB-MILESTONE      |

---

## 7. PPMI Candidate Emission & Exact Micro-Precision Arithmetic

### Exact Query-Level Algebraic Identity:
$$\operatorname{mean}(|C_q \cap H_q|) = \operatorname{mean}(|C_q|) \cdot \mathrm{Precision}_{\mathrm{micro}}$$

- **Total Dev Queries Evaluated ($N=200$):** 200
- **Queries Emitting Full 200 Terms:** 182 / 200 (91.0%)
- **Queries Emitting < 200 Terms:** 18 (queries with rare words or sparse co-occurrence)
- **Candidate Count Distribution:** Mean = `186.47`, Median = `200.0`, Min = `0`, Max = `200`, p95 = `200.0`

### Micro-Precision Verification by Denominator:
1. **Unconditioned ($N=200$ Queries):**
   - Mean candidates: `186.47`
   - Candidate-weighted micro precision: `5.86%`
   - Unconditioned mean helpful terms captured: `10.94`
   - Algebraic Identity: `186.47 × 0.0586 = 10.93` (exact match).

2. **Addressable-Conditioned ($g^* \ge 0.005$, $N=134$ Queries):**
   - Mean candidates: `183.72`
   - Addressable micro precision: `8.88%`
   - Addressable mean helpful terms captured: `16.32`
   - Algebraic Identity: `183.72 × 0.0888 = 16.32` (exact match).

---

## 8. Protocol Ambiguity Documentation: Recorded vs. Enforced PPMI Fidelity

> [!IMPORTANT]
> **Protocol Inconsistency Audit:**
> In the frozen config (`CRVE/configs/gate1_phase2_1a.yaml` Section 4), the protocol declared:
> - `ppmi_fidelity.min_recall_at_500: 0.90`
> - `ppmi_fidelity.min_rbo: 0.85`
> 
> However, the executed Checkpoint-B loss gate runner (`run_gate1_oracle_evaluation.py`) gated strictly on operational counterfactual loss limits ($\Delta\text{nDCG}@10 \text{ Loss} \le 0.02$ and $\text{DocOppRecall}@1000 \text{ Loss} \le 0.02$) and did not enforce the PPMI Recall@500/RBO thresholds.
> 
> During probe evaluation, PPMI Recall@500 was recorded at `0.8316` on BRIGHT-AoPS and `0.7854` on TREC-COVID. These measurements were recorded for diagnostic fidelity but were not enforced as halting gates by the Checkpoint-B runner.
> 
> This does not invalidate the exploratory characterization—the master audit Parquets accurately reflect the frozen $M=600$ sidecar—but confirms that Run 2 is exploratory. In any future confirmatory protocol, these fidelity thresholds must either be formally designated as diagnostic or integrated directly into the automated pre-flight gate.

---

## 9. Master Parquet Audit Provenance

| Artifact | Rows | SHA-256 Hash |
|:---|:---:|:---|
| [`gate1_candidate_audit.parquet`](results/gate1_selection/dev_200_eval/gate1_candidate_audit.parquet) | 2074475 | `006a39c65862eca4f34eb5f0b921cd5cf98216eba80f6a5fd0faebf0f65d5490` |
| [`gate1_cutoff_entries.parquet`](results/gate1_selection/dev_200_eval/gate1_cutoff_entries.parquet) | 952204 | `da7c53b195fdd9e2b485968afa40d54bcc841ae029f2f8921bb180281bf3109d` |
| [`reference_universe.parquet`](results/gate1_selection/dev_200_eval/reference_universe.parquet) | 414895 | `68865c34e2bfe5b3e3da095445c65fd92af2d8791c682f0ee7709002bcbd6962` |
| [`run_manifest.json`](results/gate1_selection/dev_200_eval/run_manifest.json) | N/A | `21098e3d40dd5b93f57106896964490639068c67960e4ae75856c99c798d4739` |

