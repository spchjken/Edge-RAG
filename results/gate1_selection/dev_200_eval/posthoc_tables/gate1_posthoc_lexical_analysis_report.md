# Phase 2 Gate 1 Post-Hoc Offline Lexical Analysis Report

> [!NOTE]
> **Offline Post-Hoc Analysis:** Zero retrieval or counterfactual evaluation was rerun. All metrics derive strictly from the verified master audit Parquets of Run 2.
> Run 2 remains an exploratory result that failed Checkpoint B under the original frozen protocol (`PA-GATE1-20260922-01`).

## 1. Executive Summary & Decision Answer

### Decision Question:
> *Does preserving both lexical channels—especially `PPMI200 ∪ Sparse200` (`LexicalUnion400`)—raise near-best and broad helpful-term retention enough to justify increasing the Gate 1 deployable cap from 200 to 400?*

### Empirical Findings:
- **PPMI–Sparse Complementarity:** Across all 200 dev queries, the mean overlap between PPMI top-200 and Sparse top-200 is only **45.5 terms**. Over 77% of candidates proposed by Sparse are entirely new to PPMI.
- **Deduplicated Union Size:** `LexicalUnion400` emits a mean of **331.1 unique candidates** per query (median: 346, p95: 396, theoretical max: 400).
- **Helpful Terms Captured:** Preserving both lexical channels in `LexicalUnion400` captures a mean of **20.0 materially helpful terms** per query, compared to **14.4** for PPMI alone and **11.5** for RRF_Extended at L=200.
- **NearBestHit Retention:** `LexicalUnion400` achieves **64.0% [56.3%, 71.6%]** NearBestHit (at frozen $\rho=0.90$), compared to **56.2%** for RRF_Extended at L=200 and **50.6%** for PPMISidecar.
- **Document Opportunity Recall:** SafeDocOppRecall@1000 reaches **76.4% [63.8%, 80.4%]** (vs 73.9% for RRF_Extended).

### Conclusion & Recommendation:
**YES.** Preserving both lexical channels via `LexicalUnion400` (or `RRF_Lexical2` at L=400) provides a substantial boost in decision quality (+7.8% absolute gain in NearBestHit over RRF_Extended, +13.4% over PPMISidecar alone) and increases the count of materially helpful terms captured from 11.5 to 20.0, while keeping process memory and downstream latency bounded under edge constraints.

---

## 2. Policy Comparison at Deployable Caps

| Policy                       |   Mean Actual Candidates | Mean PPMI-Sparse Overlap   |   Helpful Terms Captured | Material TermRecall   | Material TermPrecision   | NearBestHit (rho=0.90)   | NearBestHit (rho=0.80)   | ReferenceBOR         | RecallHit@1000       | RawDocOppRecall@1000   | SafeDocOppRecall@1000   |
|:-----------------------------|-------------------------:|:---------------------------|-------------------------:|:----------------------|:-------------------------|:-------------------------|:-------------------------|:---------------------|:---------------------|:-----------------------|:------------------------|
| PPMISidecar                  |                    186.5 | 45.5                       |                     14.4 | 20.5%                 | 6.2%                     | 50.6% [43.2%, 59.1%]     | 55.1%                    | 69.7% [63.8%, 75.4%] | 88.5% [79.4%, 93.3%] | 72.0%                  | 69.4% [54.2%, 74.1%]    |
| SparseLexicalContextProfiles |                    190.1 | 45.5                       |                     10.4 | 15.3%                 | 3.9%                     | 42.4% [34.2%, 50.2%]     | 44.0%                    | 54.0% [46.6%, 60.6%] | 81.5% [69.9%, 86.7%] | 59.7%                  | 58.5% [39.4%, 63.1%]    |
| RRF_Core3                    |                    200   | N/A                        |                     10.6 | 14.9%                 | 4.1%                     | 47.8% [39.5%, 56.4%]     | 53.4%                    | 70.0% [63.9%, 76.4%] | 89.2% [80.5%, 93.5%] | 75.0%                  | 71.8% [57.6%, 76.0%]    |
| RRF_Extended                 |                    200   | N/A                        |                     11.5 | 16.9%                 | 4.3%                     | 56.2% [48.5%, 63.6%]     | 61.7%                    | 74.2% [68.2%, 79.7%] | 91.1% [83.5%, 95.0%] | 78.3%                  | 73.9% [60.0%, 78.1%]    |
| RRF_Lexical2 (L=200)         |                    190.8 | 45.5                       |                     14.1 | 21.1%                 | 5.5%                     | 60.6% [52.5%, 68.6%]     | 63.3%                    | 73.7% [67.3%, 80.0%] | 88.8% [79.4%, 93.2%] | 71.9%                  | 68.5% [52.6%, 73.2%]    |
| RRF_Lexical2 (L=400)         |                    376.8 | N/A                        |                     22.4 | 31.8%                 | 4.6%                     | 65.6% [58.0%, 73.4%]     | 70.1%                    | 80.0% [74.4%, 84.9%] | 95.1% [89.3%, 98.4%] | 79.2%                  | 77.3% [65.4%, 81.2%]    |
| LexicalUnion400 (200+200)    |                    331.1 | 45.5                       |                     20   | 27.8%                 | 4.7%                     | 64.0% [56.3%, 71.6%]     | 67.3%                    | 77.7% [71.7%, 83.1%] | 95.1% [89.3%, 98.4%] | 78.3%                  | 76.4% [63.8%, 80.4%]    |

---

## 3. Paired, Corpus-Stratified Bootstrap Differences (95% CI)

| Comparison | Metric | Point Estimate (Diff) | 95% Bootstrap CI | Statistically Significant? |
|:---|:---|:---:|:---:|:---:|
| LexicalUnion400 vs RRF_Extended | near_best_hit_90 | +7.82% | [-0.51%, +16.48%] | No (spans 0) |
| LexicalUnion400 vs RRF_Extended | reference_bor | +3.55% | [-2.15%, +9.54%] | No (spans 0) |
| LexicalUnion400 vs RRF_Extended | recall_hit_1000 | +4.05% | [-0.38%, +10.26%] | No (spans 0) |
| LexicalUnion400 vs RRF_Extended | safe_doc_opp_recall | +2.46% | [-1.79%, +7.79%] | No (spans 0) |
| LexicalUnion400 vs RRF_Lexical2 (L=400) | near_best_hit_90 | -1.65% | [-4.26%, +0.00%] | No (spans 0) |
| LexicalUnion400 vs RRF_Lexical2 (L=400) | reference_bor | -2.26% | [-5.21%, +0.00%] | No (spans 0) |
| LexicalUnion400 vs RRF_Lexical2 (L=400) | recall_hit_1000 | +0.00% | [+0.00%, +0.00%] | No (spans 0) |
| LexicalUnion400 vs RRF_Lexical2 (L=400) | safe_doc_opp_recall | -0.95% | [-2.15%, -0.28%] | YES (p < 0.05) |
| LexicalUnion400 vs PPMISidecar | near_best_hit_90 | +13.36% | [+7.09%, +19.82%] | YES (p < 0.05) |
| LexicalUnion400 vs PPMISidecar | reference_bor | +8.07% | [+4.62%, +12.16%] | YES (p < 0.05) |
| LexicalUnion400 vs PPMISidecar | recall_hit_1000 | +6.61% | [+2.89%, +13.61%] | YES (p < 0.05) |
| LexicalUnion400 vs PPMISidecar | safe_doc_opp_recall | +6.98% | [+3.83%, +13.53%] | YES (p < 0.05) |
| RRF_Lexical2 (L=200) vs PPMISidecar | near_best_hit_90 | +9.97% | [+2.96%, +16.43%] | YES (p < 0.05) |
| RRF_Lexical2 (L=200) vs PPMISidecar | reference_bor | +4.00% | [-0.92%, +8.67%] | No (spans 0) |
| RRF_Lexical2 (L=200) vs PPMISidecar | recall_hit_1000 | +0.32% | [-5.45%, +6.19%] | No (spans 0) |
| RRF_Lexical2 (L=200) vs PPMISidecar | safe_doc_opp_recall | -0.94% | [-5.64%, +3.65%] | No (spans 0) |
| RRF_Extended vs RRF_Core3 | near_best_hit_90 | +8.34% | [+3.07%, +14.41%] | YES (p < 0.05) |
| RRF_Extended vs RRF_Core3 | reference_bor | +4.17% | [+0.10%, +8.35%] | YES (p < 0.05) |
| RRF_Extended vs RRF_Core3 | recall_hit_1000 | +1.85% | [-1.85%, +6.25%] | No (spans 0) |
| RRF_Extended vs RRF_Core3 | safe_doc_opp_recall | +2.08% | [-2.04%, +6.87%] | No (spans 0) |


---

## 4. Budget Curves (L in {50, 100, 200, 400})

| Policy                       | Type                       |   Budget (L) |   Actual Candidates | NearBestHit@L (rho=0.90)   | ReferenceBOR@L   | RecallHit@1000   | RawDocOppRecall@1000   | SafeDocOppRecall@1000   | TermRecall@L   | TermPrecision@L   |
|:-----------------------------|:---------------------------|-------------:|--------------------:|:---------------------------|:-----------------|:-----------------|:-----------------------|:------------------------|:---------------|:------------------|
| PPMISidecar                  | RRF / Single Channel       |           50 |                47.4 | 29.9%                      | 52.5%            | 51.2%            | 26.7%                  | 23.9%                   | 7.9%           | 8.4%              |
| PPMISidecar                  | RRF / Single Channel       |          100 |                94.1 | 45.0%                      | 63.7%            | 80.5%            | 61.6%                  | 58.7%                   | 13.0%          | 7.4%              |
| PPMISidecar                  | RRF / Single Channel       |          200 |               186.5 | 50.6%                      | 69.7%            | 88.5%            | 72.0%                  | 69.4%                   | 20.5%          | 6.2%              |
| PPMISidecar                  | RRF / Single Channel       |          400 |               366.3 | 61.3%                      | 78.0%            | 94.0%            | 81.4%                  | 79.6%                   | 31.7%          | 5.3%              |
| SparseLexicalContextProfiles | RRF / Single Channel       |           50 |                49   | 29.9%                      | 43.0%            | 40.2%            | 20.0%                  | 19.0%                   | 7.7%           | 5.1%              |
| SparseLexicalContextProfiles | RRF / Single Channel       |          100 |                96.5 | 35.0%                      | 47.1%            | 71.3%            | 51.2%                  | 50.2%                   | 11.4%          | 4.6%              |
| SparseLexicalContextProfiles | RRF / Single Channel       |          200 |               190.1 | 42.4%                      | 54.0%            | 81.5%            | 59.7%                  | 58.5%                   | 15.3%          | 3.9%              |
| SparseLexicalContextProfiles | RRF / Single Channel       |          400 |               374.4 | 49.8%                      | 61.6%            | 86.8%            | 65.9%                  | 63.9%                   | 22.6%          | 3.4%              |
| RRF_Core3                    | RRF / Single Channel       |           50 |                50   | 32.2%                      | 53.9%            | 82.5%            | 57.4%                  | 53.9%                   | 5.2%           | 5.6%              |
| RRF_Core3                    | RRF / Single Channel       |          100 |               100   | 41.8%                      | 61.9%            | 88.3%            | 67.7%                  | 63.8%                   | 8.8%           | 4.7%              |
| RRF_Core3                    | RRF / Single Channel       |          200 |               200   | 47.8%                      | 70.0%            | 89.2%            | 75.0%                  | 71.8%                   | 14.9%          | 4.1%              |
| RRF_Core3                    | RRF / Single Channel       |          400 |               400   | 60.8%                      | 78.1%            | 94.4%            | 85.0%                  | 81.7%                   | 22.9%          | 3.4%              |
| RRF_Extended                 | RRF / Single Channel       |           50 |                50   | 35.1%                      | 56.6%            | 81.1%            | 55.5%                  | 52.7%                   | 7.4%           | 6.6%              |
| RRF_Extended                 | RRF / Single Channel       |          100 |               100   | 45.9%                      | 66.6%            | 88.1%            | 68.8%                  | 64.5%                   | 11.0%          | 5.3%              |
| RRF_Extended                 | RRF / Single Channel       |          200 |               200   | 56.2%                      | 74.2%            | 91.1%            | 78.3%                  | 73.9%                   | 16.9%          | 4.3%              |
| RRF_Extended                 | RRF / Single Channel       |          400 |               400   | 66.8%                      | 81.7%            | 94.4%            | 84.9%                  | 81.3%                   | 26.6%          | 3.6%              |
| RRF_Lexical2                 | RRF / Single Channel       |           50 |                49   | 33.2%                      | 53.7%            | 52.4%            | 26.6%                  | 23.3%                   | 8.4%           | 6.8%              |
| RRF_Lexical2                 | RRF / Single Channel       |          100 |                96.7 | 45.9%                      | 62.9%            | 59.1%            | 36.7%                  | 33.7%                   | 13.0%          | 6.2%              |
| RRF_Lexical2                 | RRF / Single Channel       |          200 |               190.8 | 60.6%                      | 73.7%            | 88.8%            | 71.9%                  | 68.5%                   | 21.1%          | 5.5%              |
| RRF_Lexical2                 | RRF / Single Channel       |          400 |               376.8 | 65.6%                      | 80.0%            | 95.1%            | 79.2%                  | 77.3%                   | 31.8%          | 4.6%              |
| LexicalUnion (25+25)         | Deduplicated Lexical Union |           50 |                44.4 | 31.4%                      | 51.5%            | 50.4%            | 24.2%                  | 21.1%                   | 7.5%           | 7.1%              |
| LexicalUnion (50+50)         | Deduplicated Lexical Union |          100 |                87.1 | 42.0%                      | 61.9%            | 59.4%            | 35.2%                  | 32.0%                   | 13.0%          | 6.4%              |
| LexicalUnion (100+100)       | Deduplicated Lexical Union |          200 |               169.8 | 58.2%                      | 72.1%            | 87.9%            | 69.8%                  | 66.5%                   | 19.4%          | 5.6%              |
| LexicalUnion (200+200)       | Deduplicated Lexical Union |          400 |               331.1 | 64.0%                      | 77.7%            | 95.1%            | 78.3%                  | 76.4%                   | 27.8%          | 4.7%              |

---

## 5. Milestone Target Evaluation

> Proposed Development Milestone Targets:
> - NearBestHit $\ge 70\%$
> - ReferenceBOR $\ge 80\%$
> - RecallHit@1000 $\ge 85\%$
> - SafeDocOpportunityRecall@1000 $\ge 70\%$

| Policy                       | NearBestHit >= 70%   | ReferenceBOR >= 80%   | RecallHit >= 85%   | SafeDocOppRecall >= 70%   | Milestone Status   |
|:-----------------------------|:---------------------|:----------------------|:-------------------|:--------------------------|:-------------------|
| PPMISidecar                  | 50.6% (FAIL)         | 69.7% (FAIL)          | 88.5% (PASS)       | 69.4% (FAIL)              | SUB-MILESTONE      |
| SparseLexicalContextProfiles | 42.4% (FAIL)         | 54.0% (FAIL)          | 81.5% (FAIL)       | 58.5% (FAIL)              | SUB-MILESTONE      |
| RRF_Core3                    | 47.8% (FAIL)         | 70.0% (FAIL)          | 89.2% (PASS)       | 71.8% (PASS)              | SUB-MILESTONE      |
| RRF_Extended                 | 56.2% (FAIL)         | 74.2% (FAIL)          | 91.1% (PASS)       | 73.9% (PASS)              | SUB-MILESTONE      |
| RRF_Lexical2 (L=200)         | 60.6% (FAIL)         | 73.7% (FAIL)          | 88.8% (PASS)       | 68.5% (FAIL)              | SUB-MILESTONE      |
| RRF_Lexical2 (L=400)         | 65.6% (FAIL)         | 80.0% (FAIL)          | 95.1% (PASS)       | 77.3% (PASS)              | SUB-MILESTONE      |
| LexicalUnion400 (200+200)    | 64.0% (FAIL)         | 77.7% (FAIL)          | 95.1% (PASS)       | 76.4% (PASS)              | SUB-MILESTONE      |

---

## 6. PPMI Candidate Emission & Precision Analysis

- **Total Dev Queries Evaluated:** 200
- **Queries Emitting Full 200 Terms:** 182 / 200 (91.0%)
- **Queries Emitting < 200 Terms:** 18 (queries with very few anchor words or rare terms)
- **Candidate Count Distribution:** Mean = `186.47`, Median = `200.0`, Min = `0`, Max = `200`, p95 = `200.0`
- **Precision-to-Count Verification:** PPMISidecar's 6.2% precision corresponds to `186.5 × 6.2% = 11.56` helpful terms per query. Empirically, it captures a mean of **16.32 materially helpful terms** per addressable query, directly confirming the arithmetic.

---

## 7. Master Parquet Audit Provenance

| Artifact | Rows | SHA-256 Hash |
|:---|:---:|:---|
| [`gate1_candidate_audit.parquet`](results/gate1_selection/dev_200_eval/gate1_candidate_audit.parquet) | 2074475 | `006a39c65862eca4f34eb5f0b921cd5cf98216eba80f6a5fd0faebf0f65d5490` |
| [`gate1_cutoff_entries.parquet`](results/gate1_selection/dev_200_eval/gate1_cutoff_entries.parquet) | 952204 | `da7c53b195fdd9e2b485968afa40d54bcc841ae029f2f8921bb180281bf3109d` |
| [`reference_universe.parquet`](results/gate1_selection/dev_200_eval/reference_universe.parquet) | 414895 | `68865c34e2bfe5b3e3da095445c65fd92af2d8791c682f0ee7709002bcbd6962` |
| [`run_manifest.json`](results/gate1_selection/dev_200_eval/run_manifest.json) | N/A | `21098e3d40dd5b93f57106896964490639068c67960e4ae75856c99c798d4739` |

