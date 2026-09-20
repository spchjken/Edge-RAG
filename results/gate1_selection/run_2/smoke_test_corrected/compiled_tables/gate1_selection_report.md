# Phase 2 Gate 1 Candidate Selection Report

**Master Audit Parquet:** [`results/gate1_selection/run_2/smoke_test_corrected/gate1_candidate_audit.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/run_2/smoke_test_corrected/gate1_candidate_audit.parquet)  
**Cutoff Entries Parquet:** [`results/gate1_selection/run_2/smoke_test_corrected/gate1_cutoff_entries.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/run_2/smoke_test_corrected/gate1_cutoff_entries.parquet)  
**Generated At:** 2026-09-20 14:29:59 UTC  

---

## 1. Table 1: Reference Opportunity Ceiling vs Baseline

| Dataset                      | Partition   |   Queries |   Baseline nDCG@10 |   Baseline R@1000 | Ceiling Delta-nDCG@10    |   Ceiling Net Rel Docs | Materially Addressable % (g* >= 0.005)   | Recall Addressable % (r* >= 1)   |
|:-----------------------------|:------------|----------:|-------------------:|------------------:|:-------------------------|-----------------------:|:-----------------------------------------|:---------------------------------|
| scifact                      | Dev         |         5 |             0.6003 |                 1 | +0.2997 [0.0288, 0.5706] |                      0 | 60.0%                                    | 0.0%                             |
| Corpus-Macro Dev (4 Corpora) | Macro       |         5 |             0.6003 |                 1 | +0.2997                  |                      0 | 60.0%                                    | 0.0%                             |

---

## 2. Table 2: Channel Comparison at Deployable Cap ($L=200$)

| Channel                      |   Budget (L) | Corpus-Macro TermRecall@L   | Corpus-Macro TermPrecision@L   | Corpus-Macro NearBestHit@L   | Corpus-Macro ReferenceBOR@L   | Corpus-Macro RecallHit@1000   |
|:-----------------------------|-------------:|:----------------------------|:-------------------------------|:-----------------------------|:------------------------------|:------------------------------|
| WholeQueryBGE                |          200 | 8.3%                        | 0.6%                           | 0.0%                         | 40.7%                         | nan%                          |
| AnchorBGEFiltered            |          200 | 6.4%                        | 0.5%                           | 33.3%                        | 34.9%                         | nan%                          |
| AnchorBGEAll                 |          200 | 6.4%                        | 0.5%                           | 33.3%                        | 34.9%                         | nan%                          |
| PPMISidecar                  |          200 | 28.3%                       | 1.6%                           | 100.0%                       | 100.0%                        | nan%                          |
| LivePPMI                     |          200 | 0.0%                        | nan%                           | 0.0%                         | 0.0%                          | nan%                          |
| SparseLexicalContextProfiles |          200 | 35.5%                       | 1.6%                           | 100.0%                       | 100.0%                        | nan%                          |
| AcronymDefinitionRescue      |          200 | 0.0%                        | 0.0%                           | 0.0%                         | 0.0%                          | nan%                          |
| RRF_Core3                    |          200 | 21.7%                       | 1.3%                           | 33.3%                        | 58.0%                         | nan%                          |
| RRF_Extended                 |          200 | 32.7%                       | 1.8%                           | 66.7%                        | 74.1%                         | nan%                          |

---

## 3. Table: 100% Counterfactual Label Coverage Verification

| Dataset   |   Queries |   Unique Candidates |   Evaluated Variants |   Expected Variants (5 weights) | Labeling Coverage %   | Coverage Status   |
|:----------|----------:|--------------------:|---------------------:|--------------------------------:|:----------------------|:------------------|
| scifact   |         5 |                4285 |                37680 |                           21425 | 175.87%               | 100.0% COMPLETE   |
