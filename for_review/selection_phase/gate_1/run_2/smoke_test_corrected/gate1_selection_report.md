# Phase 2 Gate 1 Candidate Selection Report

**Master Audit Parquet:** [`results/gate1_selection/run_2/micro_smoke_corrected/gate1_candidate_audit.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/run_2/micro_smoke_corrected/gate1_candidate_audit.parquet)  
**Cutoff Entries Parquet:** [`results/gate1_selection/run_2/micro_smoke_corrected/gate1_cutoff_entries.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/run_2/micro_smoke_corrected/gate1_cutoff_entries.parquet)  
**Reference Universe Parquet:** [`results/gate1_selection/run_2/micro_smoke_corrected/reference_universe.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/run_2/micro_smoke_corrected/reference_universe.parquet)  
**Config Hash:** `2a6b5bed6a5e18acfbd2ac51b4cd46d9833886bf23e9eb7b658e971af6b8e7db`  
**Git Commit:** `0b39cc86dd2b0f19875cb054130ba5996865297c` (dirty: `True`)  
**Peak Process-Tree RSS:** `5.134 GiB`  
**Peak CUDA VRAM:** `218.29 MiB`  
**Generated At:** 2026-09-20 18:26:19 UTC  

---

## 1. Table 1: Reference Opportunity Ceiling vs Baseline

| Dataset                      | Partition   |   Queries |   Baseline nDCG@10 |   Baseline R@1000 | Ceiling Delta-nDCG@10    |   Ceiling Net Rel Docs | Materially Addressable % (g* >= 0.005)   | Recall Addressable % (r* >= 1)   |
|:-----------------------------|:------------|----------:|-------------------:|------------------:|:-------------------------|-----------------------:|:-----------------------------------------|:---------------------------------|
| bright_aops                  | Dev         |         1 |             0      |            1      | +0.0000 [0.0000, 0.0000] |                   0    | 0.0%                                     | 0.0%                             |
| nfcorpus                     | Dev         |         1 |             0      |            1      | +1.0000 [1.0000, 1.0000] |                   0    | 100.0%                                   | 0.0%                             |
| scifact                      | Dev         |         1 |             1      |            1      | +0.0000 [0.0000, 0.0000] |                   0    | 0.0%                                     | 0.0%                             |
| trec_covid                   | Dev         |         1 |             0.4809 |            0.2606 | +0.3153 [0.3153, 0.3153] |                  99    | 100.0%                                   | 100.0%                           |
| Corpus-Macro Dev (4 Corpora) | Macro       |         4 |             0.3702 |            0.8151 | +0.3288                  |                  24.75 | 50.0%                                    | 25.0%                            |

---

## 2. Table 2: Channel Comparison at Deployable Cap ($L=200$)

| Channel                      |   Budget (L) | Corpus-Macro TermRecall@L   | Corpus-Macro TermPrecision@L   | Corpus-Macro NearBestHit@L   | Corpus-Macro ReferenceBOR@L   | Corpus-Macro RecallHit@1000   | Corpus-Macro RawDocOppRecall@1000   | Corpus-Macro SafeDocOppRecall@1000   |
|:-----------------------------|-------------:|:----------------------------|:-------------------------------|:-----------------------------|:------------------------------|:------------------------------|:------------------------------------|:-------------------------------------|
| WholeQueryBGE                |          200 | 8.7%                        | 1.6%                           | 0.0%                         | 71.0%                         | 100.0%                        | 52.2%                               | 51.9%                                |
| AnchorBGEFiltered            |          200 | 3.7%                        | 1.4%                           | 0.0%                         | 38.1%                         | 100.0%                        | 49.6%                               | 24.9%                                |
| AnchorBGEAll                 |          200 | 3.7%                        | 1.4%                           | 0.0%                         | 38.1%                         | 100.0%                        | 49.6%                               | 24.9%                                |
| PPMISidecar                  |          200 | 8.5%                        | 3.1%                           | 50.0%                        | 50.0%                         | 100.0%                        | 68.4%                               | 86.9%                                |
| LivePPMI                     |          200 | 8.5%                        | 3.1%                           | 50.0%                        | 50.0%                         | 100.0%                        | 68.4%                               | 86.9%                                |
| SparseLexicalContextProfiles |          200 | 0.7%                        | 0.2%                           | 0.0%                         | 5.6%                          | 100.0%                        | 12.7%                               | 4.5%                                 |
| AcronymDefinitionRescue      |          200 | 0.3%                        | 1.5%                           | 0.0%                         | 1.8%                          | 100.0%                        | 1.1%                                | 0.7%                                 |
| RRF_Core3                    |          200 | 10.0%                       | 2.9%                           | 50.0%                        | 69.3%                         | 100.0%                        | 70.6%                               | 78.9%                                |
| RRF_Extended                 |          200 | 8.6%                        | 2.4%                           | 50.0%                        | 69.3%                         | 100.0%                        | 69.3%                               | 78.2%                                |

---

## 3. Table: 100% Counterfactual Label Coverage Verification

| Dataset     |   Queries |   Reference Universe Pairs |   Expected Variants |   Evaluated Variants |   Missing Triples |   Extra Triples | Labeling Coverage %   | Coverage Status   |
|:------------|----------:|---------------------------:|--------------------:|---------------------:|------------------:|----------------:|:----------------------|:------------------|
| bright_aops |         1 |                       1846 |                9230 |                 9230 |                 0 |               0 | 100.00%               | 100.0% COMPLETE   |
| nfcorpus    |         1 |                       1634 |                8170 |                 8170 |                 0 |               0 | 100.00%               | 100.0% COMPLETE   |
| scifact     |         1 |                       1418 |                7090 |                 7090 |                 0 |               0 | 100.00%               | 100.0% COMPLETE   |
| trec_covid  |         1 |                       3386 |               16930 |                16930 |                 0 |               0 | 100.00%               | 100.0% COMPLETE   |
