# Phase 2 Gate 1 Candidate Selection: Comprehensive Empirical Report

**Master Audit Parquet:** [`/home/donghv/Projects/Edge-RAG/results/gate1_selection/ext_run/gate1_candidate_audit.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/ext_run/gate1_candidate_audit.parquet)  
**Sparse Transitions Parquet:** [`/home/donghv/Projects/Edge-RAG/results/gate1_selection/ext_run/gate1_sparse_transitions.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/ext_run/gate1_sparse_transitions.parquet)  
**Generated At:** 2026-09-19 21:53:12 UTC  

---

## 1. Executive Summary & Winning Configuration

- **Winning Deployable Configuration:** `RRF_Core_L200`
- **Winning Proposer:** `rrf_core`
- **Winning Candidate Budget:** `200`
- **Macro ReferenceBOR:** `63.6%`
- **Macro NearBestHit:** `44.0%`
- **Macro RecallHit@1000:** `78.1%`
- **Centroid Trigger Decision:** `TRIGGER_CENTROIDS` (Triggered: True)
- **Frozen Configuration Hash:** `4df58c42a27f71515dcf9c499b96283b6e494b9294e7cecd53c33d904bc1453f`

---

## 2. Table 1 (Q1): Reference Opportunity Ceiling vs Baseline

| Dataset               | Partition   |   Queries |   Baseline nDCG@10 |   Baseline R@1000 | Ceiling Delta-nDCG@10    |   Ceiling Net Rel Docs | Materially Addressable % (g* >= 0.005)   | Recall Addressable % (r* >= 1)   |
|:----------------------|:------------|----------:|-------------------:|------------------:|:-------------------------|-----------------------:|:-----------------------------------------|:---------------------------------|
| arguana               | Extension   |        50 |             0.3548 |            1      | +0.0924 [0.0579, 0.1305] |                   0    | 54.0%                                    | 0.0%                             |
| bright_stackoverflow  | Extension   |        50 |             0.1011 |            0.627  | +0.0752 [0.0366, 0.1262] |                   0.46 | 42.0%                                    | 18.0%                            |
| fiqa                  | Extension   |        50 |             0.2209 |            0.7788 | +0.3952 [0.3297, 0.4607] |                   0.56 | 92.0%                                    | 42.0%                            |
| scidocs               | Extension   |        50 |             0.1855 |            0.593  | +0.2645 [0.2278, 0.3043] |                   1.3  | 92.0%                                    | 78.0%                            |
| Macro Ext (4 Corpora) | Macro       |       200 |             0.2156 |            0.7497 | +0.2068                  |                   0.58 | 70.0%                                    | 34.5%                            |
| Macro All (8 Corpora) | Macro       |       200 |             0.2156 |            0.7497 | +0.2068                  |                   0.58 | 70.0%                                    | 34.5%                            |

---

## 3. Table 2 (Q2): Proposal Channel Comparison at Deployable Cap ($L=200$)

| Channel           | Partition             |   Budget (L) | TermRecall@L   | TermPrecision@L   | NearBestHit@L   | ReferenceBOR@L   | RecallHit@L,1000   |
|:------------------|:----------------------|-------------:|:---------------|:------------------|:----------------|:-----------------|:-------------------|
| WholeQueryBGE     | Extension (4 Corpora) |          200 | 8.3%           | 1.1%              | 32.9%           | 55.2%            | 84.1%              |
| AnchorBGEFiltered | Extension (4 Corpora) |          200 | 3.6%           | 0.7%              | 22.1%           | 37.1%            | 69.6%              |
| AnchorBGEAll      | Extension (4 Corpora) |          200 | 4.2%           | 0.8%              | 22.9%           | 38.5%            | 73.9%              |
| LexicalPPMI       | Extension (4 Corpora) |          200 | 14.1%          | 2.2%              | 44.3%           | 61.7%            | 81.2%              |

---

## 4. Table 6 (Q6): Candidate Budget Retention Knee Curves

|   Budget (L) | Macro TermRecall   | Macro TermPrecision   | Macro NearBestHit   | Macro ReferenceBOR   | Macro RecallHit@1000   |
|-------------:|:-------------------|:----------------------|:--------------------|:---------------------|:-----------------------|
|           10 | 2.2%               | 5.1%                  | 14.3%               | 25.8%                | 31.9%                  |
|           20 | 3.0%               | 4.0%                  | 17.9%               | 34.5%                | 50.7%                  |
|           50 | 5.4%               | 2.8%                  | 25.0%               | 45.8%                | 65.2%                  |
|          100 | 8.5%               | 2.3%                  | 36.4%               | 55.3%                | 79.7%                  |
|          200 | 11.7%              | 1.8%                  | 44.3%               | 64.4%                | 81.2%                  |

---

## 5. Table 9 (Q9): Centroid Gating Diagnostic

| Diagnostic Metric | Value | Gating Floor | Status |
| :--- | :--- | :--- | :--- |
| Macro Material Ranking TermRecall | 11.7% | $\ge 85\%$ | FAIL |
| Macro Material Ranking NearBestHit | 44.3% | $\ge 90\%$ | FAIL |
| Macro Material Ranking ReferenceBOR | 64.4% | $\ge 92\%$ | FAIL |
| Macro Proposer Recall Coverage | 81.2% | $\ge 80\%$ | PASS |
| **Final Centroid Action** | **TRIGGER_CENTROIDS** | | **Trigger Fired** |

---

## 6. Table 10 (Q10): Multi-Dataset Configuration Evaluation

| config                 | proposer         |   budget |   mean_candidate_count |   macro_term_recall |   macro_near_best_hit |   macro_reference_bor |   macro_recall_hit |   worst_term_recall | passes_gate_a   |
|:-----------------------|:-----------------|---------:|-----------------------:|--------------------:|----------------------:|----------------------:|-------------------:|--------------------:|:----------------|
| WholeQueryBGE_L50      | wq_rank          |       50 |                     50 |           0.0400261 |              0.162497 |              0.357657 |           0.461538 |           0.0214015 | False           |
| WholeQueryBGE_L100     | wq_rank          |      100 |                    100 |           0.0549228 |              0.225529 |              0.427145 |           0.687424 |           0.0396791 | False           |
| WholeQueryBGE_L200     | wq_rank          |      200 |                    200 |           0.0795583 |              0.320595 |              0.52863  |           0.797721 |           0.0477332 | False           |
| AnchorBGEFiltered_L50  | anchor_filt_rank |       50 |                     50 |           0.0184848 |              0.108581 |              0.218478 |           0.412291 |           0.0106886 | False           |
| AnchorBGEFiltered_L100 | anchor_filt_rank |      100 |                    100 |           0.024816  |              0.163964 |              0.279663 |           0.480667 |           0.015264  | False           |
| AnchorBGEFiltered_L200 | anchor_filt_rank |      200 |                    200 |           0.0350624 |              0.202007 |              0.344094 |           0.647945 |           0.0183504 | False           |
| RRF_Core_L50           | rrf_core         |       50 |                     50 |           0.0580071 |              0.249339 |              0.450242 |           0.628816 |           0.0385674 | False           |
| RRF_Core_L100          | rrf_core         |      100 |                    100 |           0.0895626 |              0.372182 |              0.551867 |           0.77208  |           0.0607509 | False           |
| RRF_Core_L200          | rrf_core         |      200 |                    200 |           0.121151  |              0.439614 |              0.635612 |           0.780627 |           0.0951437 | False           |

