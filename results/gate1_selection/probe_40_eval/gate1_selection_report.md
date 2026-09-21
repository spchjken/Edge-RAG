# Phase 2 Gate 1 Candidate Selection Report

**Master Audit Parquet:** [`results/gate1_selection/probe_40_eval/gate1_candidate_audit.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/probe_40_eval/gate1_candidate_audit.parquet)  
**Cutoff Entries Parquet:** [`results/gate1_selection/probe_40_eval/gate1_cutoff_entries.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/probe_40_eval/gate1_cutoff_entries.parquet)  
**Reference Universe Parquet:** [`results/gate1_selection/probe_40_eval/reference_universe.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/probe_40_eval/reference_universe.parquet)  
**Config Hash:** `907e37c68546b95917839a4ea069997cca358ba29f13d912342bf8e6da549aa0`  
**Git Commit:** `6f0c3291cf0d018e510543d80950713a0c72eb2f` (dirty: `True`)  
**Peak Process-Tree RSS:** `5.963 GiB`  
**Peak CUDA VRAM:** `192.65 MiB`  
**Generated At:** 2026-09-21 11:43:05 UTC  

---

## 1. Table 1: Reference Opportunity Ceiling vs Baseline

| Dataset                      | Partition   |   Queries |   Baseline nDCG@10 |   Baseline R@1000 | Ceiling Delta-nDCG@10    |   Ceiling Net Rel Docs | Materially Addressable % (g* >= 0.005)   | Recall Addressable % (r* >= 1)   |
|:-----------------------------|:------------|----------:|-------------------:|------------------:|:-------------------------|-----------------------:|:-----------------------------------------|:---------------------------------|
| bright_aops                  | Dev         |        10 |             0.0939 |            0.3643 | +0.1323 [0.0390, 0.2335] |                   1.1  | 50.0%                                    | 80.0%                            |
| nfcorpus                     | Dev         |        10 |             0.2855 |            0.3212 | +0.4136 [0.2376, 0.6128] |                   8.6  | 80.0%                                    | 80.0%                            |
| scifact                      | Dev         |        10 |             0.6526 |            1      | +0.3025 [0.1025, 0.5287] |                   0    | 50.0%                                    | 0.0%                             |
| trec_covid                   | Dev         |        10 |             0.5724 |            0.3692 | +0.2971 [0.2146, 0.3820] |                  31.2  | 100.0%                                   | 100.0%                           |
| Corpus-Macro Dev (4 Corpora) | Macro       |        40 |             0.4011 |            0.5137 | +0.2864                  |                  10.22 | 70.0%                                    | 65.0%                            |

---

## 2. Table 2: Operational Channels Comparison at Deployable Cap ($L=200$)

| Channel                      |   Budget (L) | Corpus-Macro TermRecall@L   | Corpus-Macro TermPrecision@L   | Corpus-Macro NearBestHit@L   | Corpus-Macro ReferenceBOR@L   | Corpus-Macro RecallHit@1000   | Corpus-Macro RawDocOppRecall@1000   | Corpus-Macro SafeDocOppRecall@1000   |
|:-----------------------------|-------------:|:----------------------------|:-------------------------------|:-----------------------------|:------------------------------|:------------------------------|:------------------------------------|:-------------------------------------|
| WholeQueryBGE                |          200 | 12.3%                       | 2.5%                           | 41.2%                        | 63.6%                         | 83.3%                         | 55.4%                               | 50.6%                                |
| AnchorBGEFiltered            |          200 | 8.1%                        | 2.3%                           | 41.2%                        | 62.6%                         | 79.2%                         | 50.6%                               | 44.2%                                |
| AnchorBGEAll                 |          200 | 9.1%                        | 2.4%                           | 46.2%                        | 65.3%                         | 83.3%                         | 54.3%                               | 47.8%                                |
| PPMISidecar                  |          200 | 19.1%                       | 6.6%                           | 44.4%                        | 60.7%                         | 91.7%                         | 63.9%                               | 61.0%                                |
| SparseLexicalContextProfiles |          200 | 13.6%                       | 3.7%                           | 40.0%                        | 50.1%                         | 83.3%                         | 47.0%                               | 45.9%                                |
| AcronymDefinitionRescue      |          200 | 0.7%                        | 7.2%                           | 5.0%                         | 13.3%                         | 54.2%                         | 12.5%                               | 11.4%                                |
| RRF_Core3                    |          200 | 20.0%                       | 4.3%                           | 54.4%                        | 74.1%                         | 87.5%                         | 64.7%                               | 58.7%                                |
| RRF_Extended                 |          200 | 20.4%                       | 4.6%                           | 57.5%                        | 73.3%                         | 87.5%                         | 67.7%                               | 62.2%                                |

---

## 2b. Table 2-Diag: LivePPMI Diagnostic Comparator ($L=200$)

| Channel     |   Budget (L) | Corpus-Macro TermRecall@L   | Corpus-Macro TermPrecision@L   | Corpus-Macro NearBestHit@L   | Corpus-Macro ReferenceBOR@L   | Corpus-Macro RecallHit@1000   | Corpus-Macro RawDocOppRecall@1000   | Corpus-Macro SafeDocOppRecall@1000   |
|:------------|-------------:|:----------------------------|:-------------------------------|:-----------------------------|:------------------------------|:------------------------------|:------------------------------------|:-------------------------------------|
| LivePPMI    |          200 | 19.8%                       | 7.0%                           | 49.4%                        | 63.0%                         | 87.5%                         | 64.2%                               | 61.4%                                |
| PPMISidecar |          200 | 19.0%                       | 6.6%                           | 44.4%                        | 60.7%                         | 91.7%                         | 63.9%                               | 61.0%                                |

---

## 2c. Checkpoint B Operational Loss Gate Evaluation

**Gate Status:** **FAILED**  
- **Corpus-Macro $\Delta$nDCG@10 Loss:** `0.0072` (threshold: `0.02`)  
- **Corpus-Macro RawDocOppRecall@1000 Loss:** `0.0186` (threshold: `0.02`)  

---

## 3. Table: 100% Counterfactual Label Coverage Verification

| Dataset     |   Queries |   Reference Universe Pairs |   Expected Variants |   Evaluated Variants |   Missing Triples |   Extra Triples | Labeling Coverage %   | Coverage Status   |
|:------------|----------:|---------------------------:|--------------------:|---------------------:|------------------:|----------------:|:----------------------|:------------------|
| bright_aops |        10 |                      17400 |               87000 |                87000 |                 0 |               0 | 100.00%               | 100.0% COMPLETE   |
| nfcorpus    |        10 |                      16024 |               80120 |                80120 |                 0 |               0 | 100.00%               | 100.0% COMPLETE   |
| scifact     |        10 |                      14890 |               74450 |                74450 |                 0 |               0 | 100.00%               | 100.0% COMPLETE   |
| trec_covid  |        10 |                      33251 |              166255 |               166255 |                 0 |               0 | 100.00%               | 100.0% COMPLETE   |
