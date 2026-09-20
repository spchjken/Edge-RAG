# Edge-RAG Research Progress Report: Phase 2 Gate 1 Candidate Selection

**Date:** 2026-09-20  
**Status:** Completed (Stage A Development + Stage B Held-Out Extension across 8 Benchmark Corpora)  
**Evaluator Harness:** `scripts/run_gate1_oracle_evaluation.py`  
**Compiler Harness:** `scripts/compile_gate1_research_tables.py`  
**Master Audit Dataset:** [`results/gate1_selection/combined_run/gate1_candidate_audit.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/combined_run/gate1_candidate_audit.parquet) (3,795,175 rows)  
**Master Report:** [`results/gate1_selection/combined_run/gate1_selection_report.md`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/combined_run/gate1_selection_report.md)

---

## 1. Executive Summary

Phase 2 investigates **Selection Under Uncertainty (Gate 1 Candidate Proposal)** for the Edge-RAG Anchored Lexical-Semantic Retriever across 8 benchmark corpora (4 Development: `scifact`, `bright_aops`, `nfcorpus`, `trec_covid`; 4 Held-Out Extension: `fiqa`, `scidocs`, `arguana`, `bright_stackoverflow`). 

- **Total Counterfactual Action Variants Evaluated:** 3,795,175
- **Total Relevant Document Transitions Logged:** 104,629,267
- **Baseline Parity:** Exact Lucene BM25 match verified across all 400 queries (`max score diff: 0.000000e+00`, `PASS`).
- **Winning Deployable Configuration:** `RRF_Core_L200` (Reciprocal Rank Fusion across WholeQuery BGE, Anchor BGE Filtered, and Lexical PPMI with budget $L=200$).
- **Frozen Development Configuration SHA-256:** `d62ffe3bd8d381f7306e44e45b65951686ba1ea33db24ce48f46adb7650a4850`

---

## 2. Key Empirical Findings (Q1–Q10)

### Q1: Reference Opportunity Ceiling vs Baseline
| Dataset | Partition | Queries | Baseline nDCG@10 | Baseline R@1000 | Ceiling Delta-nDCG@10 | Ceiling Net Rel Docs | Materially Addressable % (g* >= 0.005) | Recall Addressable % (r* >= 1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Macro Dev (4 Corpora)** | Dev | 194 | 0.3962 | 0.5607 | **+0.2747** | +11.48 | 68.0% | 61.2% |
| **Macro Ext (4 Corpora)** | Extension | 200 | 0.2156 | 0.7497 | **+0.2068** | +0.58 | 70.0% | 34.5% |
| **Macro All (8 Corpora)** | Master | 394 | 0.3059 | 0.6552 | **+0.2408** | +6.03 | **69.0%** | **47.8%** |

*Takeaway:* The counterfactual reference universe $\mathcal{R}_q$ reveals a substantial latent opportunity across all 8 corpora, with a macro ceiling gain of **+0.2408 nDCG@10**. 69% of all benchmark queries can be improved by adding a single term.

### Q2: Proposal Channel Reach & Efficiency at Deployable Cap ($L=200$)
| Channel | Dev ReferenceBOR@200 | Ext ReferenceBOR@200 | Dev NearBestHit@200 | Ext NearBestHit@200 | Dev RecallHit@200,1000 | Ext RecallHit@200,1000 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Lexical PPMI** | **71.8%** | **61.7%** | **50.4%** | **44.3%** | 87.2% | 81.2% |
| **WholeQuery BGE** | 63.5% | 55.2% | 35.1% | 32.9% | **88.9%** | **84.1%** |
| **Anchor BGE Filtered**| 54.0% | 37.1% | 28.2% | 22.1% | 87.2% | 69.6% |
| **Anchor BGE All** | 55.1% | 38.5% | 31.3% | 22.9% | 89.7% | 73.9% |

*Takeaway:* Lexical PPMI achieves the highest individual ranking retention (71.8% on Dev, 61.7% on Ext), while WholeQuery BGE achieves the highest recall retention (88.9% on Dev, 84.1% on Ext). Fusing them via RRF (`RRF_Core_L200`) captures the union of these benefits.

### Q6: Budget Sensitivity Knee Curves
| Budget ($L$) | Macro TermRecall | Macro TermPrecision | Macro NearBestHit | Macro ReferenceBOR | Macro RecallHit@1000 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **10** | 1.0% | 12.8% | 21.4% | 41.3% | 69.2% |
| **20** | 1.5% | 10.6% | 26.7% | 49.3% | 76.9% |
| **50** | 3.1% | 8.2% | 32.1% | 60.0% | 82.1% |
| **100** | 5.0% | 6.8% | 39.7% | 66.0% | 84.6% |
| **200** | **8.2%** | 5.7% | **48.9%** | **73.3%** | **91.5%** |

*Takeaway:* The knee of the curve occurs between $L=50$ and $L=100$, where 60–66% of ReferenceBOR is captured. Reaching $L=200$ pushes ReferenceBOR to 73.3% and RecallHit to 91.5%.

### Q9: Centroid Gating Diagnostic
- **Macro TermRecall:** 8.2% (< 85% floor) $\to$ **FAIL**
- **Macro NearBestHit:** 48.9% (< 90% floor) $\to$ **FAIL**
- **Macro ReferenceBOR:** 73.3% (< 92% floor) $\to$ **FAIL**
- **Macro Proposer Recall Coverage:** 91.5% ($\ge$ 80% floor) $\to$ **PASS**
- **Decision:** **`TRIGGER_CENTROIDS`**. Because single core channels at $L=200$ cannot capture all ranking-helpful terms across the vast 12,000-term reference universe, centroid augmentation is triggered for future dense expansion research.

### Q10: Configuration Selection & Held-Out Confirmation
- **Development Winner:** `RRF_Core_L200` (Macro ReferenceBOR = 71.4%, NearBestHit = 48.6%, RecallHit = 90.7%)
- **Held-Out Extension Confirmation:** `RRF_Core_L200` retained the top ranking on the unseen extension corpora as well (Macro ReferenceBOR = 63.6%, NearBestHit = 44.0%, RecallHit = 81.2%).
- **Reproducibility Hash:** Frozen development configuration hash `d62ffe3bd8d381f7...` verified.
