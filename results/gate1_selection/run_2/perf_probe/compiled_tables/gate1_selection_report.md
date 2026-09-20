# Phase 2 Gate 1 Candidate Selection Report

**Master Audit Parquet:** [`results/gate1_selection/run_2/perf_probe/gate1_candidate_audit.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/run_2/perf_probe/gate1_candidate_audit.parquet)  
**Cutoff Entries Parquet:** [`results/gate1_selection/run_2/perf_probe/gate1_cutoff_entries.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/run_2/perf_probe/gate1_cutoff_entries.parquet)  
**Generated At:** 2026-09-20 14:11:19 UTC  

---

## 1. Table 1: Reference Opportunity Ceiling vs Baseline

| Dataset                      | Partition   |   Queries |   Baseline nDCG@10 |   Baseline R@1000 | Ceiling Delta-nDCG@10    |   Ceiling Net Rel Docs | Materially Addressable % (g* >= 0.005)   | Recall Addressable % (r* >= 1)   |
|:-----------------------------|:------------|----------:|-------------------:|------------------:|:-------------------------|-----------------------:|:-----------------------------------------|:---------------------------------|
| bright_aops                  | Dev         |        10 |             0.0592 |            0.3643 | +0.0720 [0.0118, 0.1352] |                   0.9  | 40.0%                                    | 80.0%                            |
| nfcorpus                     | Dev         |        10 |             0.2855 |            0.3212 | +0.4136 [0.2376, 0.6128] |                   8.4  | 80.0%                                    | 80.0%                            |
| scifact                      | Dev         |        10 |             0.6526 |            1      | +0.3025 [0.1025, 0.5287] |                   0    | 50.0%                                    | 0.0%                             |
| trec_covid                   | Dev         |        10 |             0.5724 |            0.3692 | +0.2971 [0.2146, 0.3820] |                  31.2  | 100.0%                                   | 100.0%                           |
| Corpus-Macro Dev (4 Corpora) | Macro       |        40 |             0.3924 |            0.5137 | +0.2713                  |                  10.12 | 67.5%                                    | 65.0%                            |

---

## 2. Table 2: Channel Comparison at Deployable Cap ($L=200$)

| Channel                      |   Budget (L) | Corpus-Macro TermRecall@L   | Corpus-Macro TermPrecision@L   | Corpus-Macro NearBestHit@L   | Corpus-Macro ReferenceBOR@L   | Corpus-Macro RecallHit@1000   |
|:-----------------------------|-------------:|:----------------------------|:-------------------------------|:-----------------------------|:------------------------------|:------------------------------|
| WholeQueryBGE                |          200 | 9.7%                        | 2.7%                           | 35.0%                        | 61.7%                         | 83.3%                         |
| AnchorBGEFiltered            |          200 | 11.1%                       | 2.7%                           | 41.2%                        | 66.3%                         | 70.8%                         |
| AnchorBGEAll                 |          200 | 12.3%                       | 2.7%                           | 46.2%                        | 68.4%                         | 79.2%                         |
| PPMISidecar                  |          200 | 20.8%                       | 6.5%                           | 45.6%                        | 60.0%                         | 83.3%                         |
| LivePPMI                     |          200 | 21.8%                       | 6.9%                           | 50.6%                        | 67.4%                         | 79.2%                         |
| SparseLexicalContextProfiles |          200 | 12.5%                       | 4.2%                           | 31.9%                        | 51.5%                         | 75.0%                         |
| AcronymDefinitionRescue      |          200 | 2.8%                        | 7.3%                           | 17.5%                        | 29.4%                         | 58.3%                         |
| RRF_Core3                    |          200 | 17.2%                       | 4.6%                           | 50.6%                        | 67.9%                         | 83.3%                         |
| RRF_Extended                 |          200 | 17.5%                       | 4.7%                           | 53.8%                        | 70.0%                         | 87.5%                         |

---

## 3. Table: 100% Counterfactual Label Coverage Verification

| Dataset     |   Queries |   Unique Candidates |   Evaluated Variants |   Expected Variants (5 weights) | Labeling Coverage %   | Coverage Status   |
|:------------|----------:|--------------------:|---------------------:|--------------------------------:|:----------------------|:------------------|
| bright_aops |        10 |                6210 |                88245 |                           31050 | 284.20%               | 100.0% COMPLETE   |
| nfcorpus    |        10 |                6118 |                79415 |                           30590 | 259.61%               | 100.0% COMPLETE   |
| scifact     |        10 |                4934 |                74065 |                           24670 | 300.22%               | 100.0% COMPLETE   |
| trec_covid  |        10 |                8926 |               162640 |                           44630 | 364.42%               | 100.0% COMPLETE   |
