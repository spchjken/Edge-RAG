# Phase 2 Gate 1 Candidate Selection Report

**Master Audit Parquet:** [`results/gate1_selection/run_2/smoke_test_corrected/gate1_candidate_audit.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/run_2/smoke_test_corrected/gate1_candidate_audit.parquet)  
**Cutoff Entries Parquet:** [`results/gate1_selection/run_2/smoke_test_corrected/gate1_cutoff_entries.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/run_2/smoke_test_corrected/gate1_cutoff_entries.parquet)  
**Generated At:** 2026-09-20 16:31:06 UTC  

---

## 1. Table 1: Reference Opportunity Ceiling vs Baseline

| Dataset                      | Partition   |   Queries |   Baseline nDCG@10 |   Baseline R@1000 | Ceiling Delta-nDCG@10    |   Ceiling Net Rel Docs | Materially Addressable % (g* >= 0.005)   | Recall Addressable % (r* >= 1)   |
|:-----------------------------|:------------|----------:|-------------------:|------------------:|:-------------------------|-----------------------:|:-----------------------------------------|:---------------------------------|
| scifact                      | Dev         |         1 |                  1 |                 1 | +0.0000 [0.0000, 0.0000] |                      0 | 0.0%                                     | 0.0%                             |
| Corpus-Macro Dev (4 Corpora) | Macro       |         1 |                  1 |                 1 | +0.0000                  |                      0 | 0.0%                                     | 0.0%                             |

---

## 2. Table 2: Channel Comparison at Deployable Cap ($L=200$)

| Channel                      |   Budget (L) | Corpus-Macro TermRecall@L   | Corpus-Macro TermPrecision@L   | Corpus-Macro NearBestHit@L   | Corpus-Macro ReferenceBOR@L   | Corpus-Macro RecallHit@1000   | Corpus-Macro RawDocOppRecall@1000   | Corpus-Macro SafeDocOppRecall@1000   |
|:-----------------------------|-------------:|:----------------------------|:-------------------------------|:-----------------------------|:------------------------------|:------------------------------|:------------------------------------|:-------------------------------------|
| WholeQueryBGE                |          200 | nan%                        | 0.0%                           | nan%                         | nan%                          | nan%                          | N/A                                 | N/A                                  |
| AnchorBGEFiltered            |          200 | nan%                        | 0.0%                           | nan%                         | nan%                          | nan%                          | N/A                                 | N/A                                  |
| AnchorBGEAll                 |          200 | nan%                        | 0.0%                           | nan%                         | nan%                          | nan%                          | N/A                                 | N/A                                  |
| PPMISidecar                  |          200 | nan%                        | 0.0%                           | nan%                         | nan%                          | nan%                          | N/A                                 | N/A                                  |
| LivePPMI                     |          200 | nan%                        | nan%                           | nan%                         | nan%                          | nan%                          | N/A                                 | N/A                                  |
| SparseLexicalContextProfiles |          200 | nan%                        | 0.0%                           | nan%                         | nan%                          | nan%                          | N/A                                 | N/A                                  |
| AcronymDefinitionRescue      |          200 | nan%                        | 0.0%                           | nan%                         | nan%                          | nan%                          | N/A                                 | N/A                                  |
| RRF_Core3                    |          200 | nan%                        | 0.0%                           | nan%                         | nan%                          | nan%                          | N/A                                 | N/A                                  |
| RRF_Extended                 |          200 | nan%                        | 0.0%                           | nan%                         | nan%                          | nan%                          | N/A                                 | N/A                                  |

---

## 3. Table: 100% Counterfactual Label Coverage Verification

| Dataset   |   Queries |   Unique Query-Candidate Pairs |   Evaluated Variants (5 weights) |   Expected Variants | Labeling Coverage %   | Coverage Status   |
|:----------|----------:|-------------------------------:|---------------------------------:|--------------------:|:----------------------|:------------------|
| scifact   |         1 |                           1601 |                             8005 |                8005 | 100.00%               | 100.0% COMPLETE   |
