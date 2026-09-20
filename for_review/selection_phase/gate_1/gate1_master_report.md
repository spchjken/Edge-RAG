# Phase 2 Gate 1 Candidate Selection: Comprehensive Empirical Report

**Master Audit Parquet:** [`/home/donghv/Projects/Edge-RAG/results/gate1_selection/combined_run/gate1_candidate_audit.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/combined_run/gate1_candidate_audit.parquet)  
**Sparse Transitions Parquet:** [`/home/donghv/Projects/Edge-RAG/results/gate1_selection/dev_run/gate1_sparse_transitions.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/dev_run/gate1_sparse_transitions.parquet)  
**Generated At:** 2026-09-19 21:55:10 UTC  

---

## 1. Executive Summary & Winning Configuration

- **Winning Deployable Configuration:** `RRF_Core_L200`
- **Winning Proposer:** `rrf_core`
- **Winning Candidate Budget:** `200`
- **Macro ReferenceBOR:** `71.4%`
- **Macro NearBestHit:** `48.6%`
- **Macro RecallHit@1000:** `90.7%`
- **Centroid Trigger Decision:** `TRIGGER_CENTROIDS` (Triggered: True)
- **Frozen Configuration Hash:** `e17a5fb877b81abc7c0126a4cb522325e1179d35f738452bfb4f02ccbdb97c2c`

---

## 2. Table 1 (Q1): Reference Opportunity Ceiling vs Baseline

| Dataset               | Partition   |   Queries |   Baseline nDCG@10 |   Baseline R@1000 | Ceiling Delta-nDCG@10    |   Ceiling Net Rel Docs | Materially Addressable % (g* >= 0.005)   | Recall Addressable % (r* >= 1)   |
|:----------------------|:------------|----------:|-------------------:|------------------:|:-------------------------|-----------------------:|:-----------------------------------------|:---------------------------------|
| arguana               | Extension   |        50 |             0.3548 |            1      | +0.0924 [0.0579, 0.1305] |                   0    | 54.0%                                    | 0.0%                             |
| bright_aops           | Dev         |        50 |             0.0372 |            0.4589 | +0.1001 [0.0608, 0.1556] |                   0.6  | 42.0%                                    | 54.0%                            |
| bright_stackoverflow  | Extension   |        50 |             0.1011 |            0.627  | +0.0752 [0.0366, 0.1262] |                   0.46 | 42.0%                                    | 18.0%                            |
| fiqa                  | Extension   |        50 |             0.2209 |            0.7788 | +0.3952 [0.3297, 0.4607] |                   0.56 | 92.0%                                    | 42.0%                            |
| nfcorpus              | Dev         |        44 |             0.2977 |            0.3552 | +0.4479 [0.3569, 0.5377] |                  12.59 | 84.1%                                    | 88.6%                            |
| scidocs               | Extension   |        50 |             0.1855 |            0.593  | +0.2645 [0.2278, 0.3043] |                   1.3  | 92.0%                                    | 78.0%                            |
| scifact               | Dev         |        50 |             0.6444 |            0.98   | +0.2703 [0.1811, 0.3673] |                   0.02 | 48.0%                                    | 2.0%                             |
| trec_covid            | Dev         |        50 |             0.6055 |            0.4486 | +0.2804 [0.2308, 0.3312] |                  32.72 | 98.0%                                    | 100.0%                           |
| Macro Dev (4 Corpora) | Macro       |       194 |             0.3962 |            0.5607 | +0.2747                  |                  11.48 | 68.0%                                    | 61.2%                            |
| Macro Ext (4 Corpora) | Macro       |       200 |             0.2156 |            0.7497 | +0.2068                  |                   0.58 | 70.0%                                    | 34.5%                            |
| Macro All (8 Corpora) | Macro       |       394 |             0.3059 |            0.6552 | +0.2408                  |                   6.03 | 69.0%                                    | 47.8%                            |

---

## 3. Table 2 (Q2): Proposal Channel Comparison at Deployable Cap ($L=200$)

| Channel           | Partition               |   Budget (L) | TermRecall@L   | TermPrecision@L   | NearBestHit@L   | ReferenceBOR@L   | RecallHit@L,1000   |
|:------------------|:------------------------|-------------:|:---------------|:------------------|:----------------|:-----------------|:-------------------|
| WholeQueryBGE     | Development (4 Corpora) |          200 | 4.5%           | 3.8%              | 35.1%           | 63.5%            | 88.9%              |
| WholeQueryBGE     | Extension (4 Corpora)   |          200 | 8.3%           | 1.1%              | 32.9%           | 55.2%            | 84.1%              |
| AnchorBGEFiltered | Development (4 Corpora) |          200 | 3.1%           | 3.4%              | 28.2%           | 54.0%            | 87.2%              |
| AnchorBGEFiltered | Extension (4 Corpora)   |          200 | 3.6%           | 0.7%              | 22.1%           | 37.1%            | 69.6%              |
| AnchorBGEAll      | Development (4 Corpora) |          200 | 3.3%           | 3.4%              | 31.3%           | 55.1%            | 89.7%              |
| AnchorBGEAll      | Extension (4 Corpora)   |          200 | 4.2%           | 0.8%              | 22.9%           | 38.5%            | 73.9%              |
| LexicalPPMI       | Development (4 Corpora) |          200 | 10.5%          | 6.9%              | 50.4%           | 71.8%            | 87.2%              |
| LexicalPPMI       | Extension (4 Corpora)   |          200 | 14.1%          | 2.2%              | 44.3%           | 61.7%            | 81.2%              |

---

## 4. Table 6 (Q6): Candidate Budget Retention Knee Curves

|   Budget (L) | Macro TermRecall   | Macro TermPrecision   | Macro NearBestHit   | Macro ReferenceBOR   | Macro RecallHit@1000   |
|-------------:|:-------------------|:----------------------|:--------------------|:---------------------|:-----------------------|
|           10 | 1.0%               | 12.8%                 | 21.4%               | 41.3%                | 69.2%                  |
|           20 | 1.5%               | 10.6%                 | 26.7%               | 49.3%                | 76.9%                  |
|           50 | 3.1%               | 8.2%                  | 32.1%               | 60.0%                | 82.1%                  |
|          100 | 5.0%               | 6.8%                  | 39.7%               | 66.0%                | 84.6%                  |
|          200 | 8.2%               | 5.7%                  | 48.9%               | 73.3%                | 91.5%                  |

---

## 5. Table 9 (Q9): Centroid Gating Diagnostic

| Diagnostic Metric | Value | Gating Floor | Status |
| :--- | :--- | :--- | :--- |
| Macro Material Ranking TermRecall | 8.2% | $\ge 85\%$ | FAIL |
| Macro Material Ranking NearBestHit | 48.9% | $\ge 90\%$ | FAIL |
| Macro Material Ranking ReferenceBOR | 73.3% | $\ge 92\%$ | FAIL |
| Macro Proposer Recall Coverage | 91.5% | $\ge 80\%$ | PASS |
| **Final Centroid Action** | **TRIGGER_CENTROIDS** | | **Trigger Fired** |

---

## 6. Table 10 (Q10): Multi-Dataset Configuration Evaluation

| config                 | proposer         |   budget |   mean_candidate_count |   macro_term_recall |   macro_near_best_hit |   macro_reference_bor |   macro_recall_hit |   worst_term_recall | passes_gate_a   |
|:-----------------------|:-----------------|---------:|-----------------------:|--------------------:|----------------------:|----------------------:|-------------------:|--------------------:|:----------------|
| WholeQueryBGE_L50      | wq_rank          |       50 |                     50 |           0.0246414 |              0.258911 |              0.481455 |           0.759972 |          0.0112976  | False           |
| WholeQueryBGE_L100     | wq_rank          |      100 |                    100 |           0.0350765 |              0.296706 |              0.542718 |           0.838319 |          0.0203777  | False           |
| WholeQueryBGE_L200     | wq_rank          |      200 |                    200 |           0.0488703 |              0.335942 |              0.602767 |           0.885328 |          0.0356459  | False           |
| AnchorBGEFiltered_L50  | anchor_filt_rank |       50 |                     50 |           0.0114736 |              0.178342 |              0.360434 |           0.736453 |          0.00229903 | False           |
| AnchorBGEFiltered_L100 | anchor_filt_rank |      100 |                    100 |           0.0204047 |              0.23659  |              0.442403 |           0.807692 |          0.0131098  | False           |
| AnchorBGEFiltered_L200 | anchor_filt_rank |      200 |                    200 |           0.0317165 |              0.269115 |              0.497066 |           0.861111 |          0.0184008  | False           |
| RRF_Core_L50           | rrf_core         |       50 |                     50 |           0.0365511 |              0.31886  |              0.573908 |           0.825499 |          0.0141117  | False           |
| RRF_Core_L100          | rrf_core         |      100 |                    100 |           0.0590427 |              0.397591 |              0.635722 |           0.847578 |          0.0263284  | False           |
| RRF_Core_L200          | rrf_core         |      200 |                    200 |           0.0964423 |              0.48601  |              0.713694 |           0.907407 |          0.0455516  | False           |

