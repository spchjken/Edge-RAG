# Phase 2 Gate 1 Candidate Selection: Comprehensive Empirical Report

**Master Audit Parquet:** [`results/gate1_selection/test_probe_ultrafast/gate1_candidate_audit.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/test_probe_ultrafast/gate1_candidate_audit.parquet)  
**Sparse Transitions Parquet:** [`results/gate1_selection/test_probe_ultrafast/gate1_sparse_transitions.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/test_probe_ultrafast/gate1_sparse_transitions.parquet)  
**Generated At:** 2026-09-18 09:39:44 UTC  

---

## 1. Executive Summary & Winning Configuration

- **Winning Deployable Configuration:** `WholeQueryBGE_L50`
- **Winning Proposer:** `wq_rank`
- **Winning Candidate Budget:** `50`
- **Macro ReferenceBOR:** `nan%`
- **Macro NearBestHit:** `nan%`
- **Macro RecallHit@1000:** `nan%`
- **Centroid Trigger Decision:** `RETAIN_CORE_PROPOSERS` (Triggered: False)
- **Frozen Configuration Hash:** `N/A`

---

## 2. Table 1 (Q1): Reference Opportunity Ceiling vs Baseline

| Dataset               | Partition   |   Queries |   Baseline nDCG@10 |   Baseline R@1000 | Ceiling Delta-nDCG@10    |   Ceiling Net Rel Docs | Materially Addressable % (g* >= 0.005)   | Recall Addressable % (r* >= 1)   |
|:----------------------|:------------|----------:|-------------------:|------------------:|:-------------------------|-----------------------:|:-----------------------------------------|:---------------------------------|
| scifact               | Dev         |         1 |                  1 |                 1 | +0.0000 [0.0000, 0.0000] |                      0 | 0.0%                                     | 0.0%                             |
| Macro Dev (4 Corpora) | Macro       |         1 |                  1 |                 1 | +0.0000                  |                      0 | 0.0%                                     | 0.0%                             |
| Macro All (8 Corpora) | Macro       |         1 |                  1 |                 1 | +0.0000                  |                      0 | 0.0%                                     | 0.0%                             |

---

## 3. Table 2 (Q2): Proposal Channel Comparison at Deployable Cap ($L=200$)

| Channel           | Partition               |   Budget (L) | TermRecall@L   | TermPrecision@L   | NearBestHit@L   | ReferenceBOR@L   | RecallHit@L,1000   |
|:------------------|:------------------------|-------------:|:---------------|:------------------|:----------------|:-----------------|:-------------------|
| WholeQueryBGE     | Development (4 Corpora) |          200 | nan%           | 0.0%              | nan%            | nan%             | nan%               |
| AnchorBGEFiltered | Development (4 Corpora) |          200 | nan%           | 0.0%              | nan%            | nan%             | nan%               |
| AnchorBGEAll      | Development (4 Corpora) |          200 | nan%           | 0.0%              | nan%            | nan%             | nan%               |
| LexicalPPMI       | Development (4 Corpora) |          200 | nan%           | 0.0%              | nan%            | nan%             | nan%               |

---

## 4. Table 6 (Q6): Candidate Budget Retention Knee Curves

|   Budget (L) | Macro TermRecall   | Macro TermPrecision   | Macro NearBestHit   | Macro ReferenceBOR   | Macro RecallHit@1000   |
|-------------:|:-------------------|:----------------------|:--------------------|:---------------------|:-----------------------|
|           10 | nan%               | 0.0%                  | nan%                | nan%                 | nan%                   |
|           20 | nan%               | 0.0%                  | nan%                | nan%                 | nan%                   |
|           50 | nan%               | 0.0%                  | nan%                | nan%                 | nan%                   |
|          100 | nan%               | 0.0%                  | nan%                | nan%                 | nan%                   |
|          200 | nan%               | 0.0%                  | nan%                | nan%                 | nan%                   |

---

## 5. Table 9 (Q9): Centroid Gating Diagnostic

| Diagnostic Metric | Value | Gating Floor | Status |
| :--- | :--- | :--- | :--- |
| Macro Material Ranking TermRecall | nan% | $\ge 85\%$ | FAIL |
| Macro Material Ranking NearBestHit | nan% | $\ge 90\%$ | FAIL |
| Macro Material Ranking ReferenceBOR | nan% | $\ge 92\%$ | FAIL |
| Macro Proposer Recall Coverage | nan% | $\ge 80\%$ | FAIL |
| **Final Centroid Action** | **RETAIN_CORE_PROPOSERS** | | **No Trigger Fired** |

---

## 6. Table 10 (Q10): Multi-Dataset Configuration Evaluation

| config                 | proposer         |   budget |   mean_candidate_count |   macro_term_recall |   macro_near_best_hit |   macro_reference_bor |   macro_recall_hit |   worst_term_recall | passes_gate_a   |
|:-----------------------|:-----------------|---------:|-----------------------:|--------------------:|----------------------:|----------------------:|-------------------:|--------------------:|:----------------|
| WholeQueryBGE_L50      | wq_rank          |       50 |                     50 |                 nan |                   nan |                   nan |                nan |                 nan | False           |
| WholeQueryBGE_L100     | wq_rank          |      100 |                    100 |                 nan |                   nan |                   nan |                nan |                 nan | False           |
| WholeQueryBGE_L200     | wq_rank          |      200 |                    200 |                 nan |                   nan |                   nan |                nan |                 nan | False           |
| AnchorBGEFiltered_L50  | anchor_filt_rank |       50 |                     50 |                 nan |                   nan |                   nan |                nan |                 nan | False           |
| AnchorBGEFiltered_L100 | anchor_filt_rank |      100 |                    100 |                 nan |                   nan |                   nan |                nan |                 nan | False           |
| AnchorBGEFiltered_L200 | anchor_filt_rank |      200 |                    200 |                 nan |                   nan |                   nan |                nan |                 nan | False           |
| RRF_Core_L50           | rrf_core         |       50 |                     50 |                 nan |                   nan |                   nan |                nan |                 nan | False           |
| RRF_Core_L100          | rrf_core         |      100 |                    100 |                 nan |                   nan |                   nan |                nan |                 nan | False           |
| RRF_Core_L200          | rrf_core         |      200 |                    200 |                 nan |                   nan |                   nan |                nan |                 nan | False           |

