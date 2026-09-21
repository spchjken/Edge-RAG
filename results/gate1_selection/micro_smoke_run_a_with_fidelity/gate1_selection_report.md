# Phase 2 Gate 1 Candidate Selection Report

**Master Audit Parquet:** [`results/gate1_selection/micro_smoke_run_a_with_fidelity/gate1_candidate_audit.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/micro_smoke_run_a_with_fidelity/gate1_candidate_audit.parquet)  
**Cutoff Entries Parquet:** [`results/gate1_selection/micro_smoke_run_a_with_fidelity/gate1_cutoff_entries.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/micro_smoke_run_a_with_fidelity/gate1_cutoff_entries.parquet)  
**Reference Universe Parquet:** [`results/gate1_selection/micro_smoke_run_a_with_fidelity/reference_universe.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/micro_smoke_run_a_with_fidelity/reference_universe.parquet)  
**Config Hash:** `907e37c68546b95917839a4ea069997cca358ba29f13d912342bf8e6da549aa0`  
**Git Commit:** `6f0c3291cf0d018e510543d80950713a0c72eb2f` (dirty: `True`)  
**Peak Process-Tree RSS:** `5.717 GiB`  
**Peak CUDA VRAM:** `193.58 MiB`  
**Generated At:** 2026-09-21 08:28:06 UTC  

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

## 2. Table 2: Operational Channels Comparison at Deployable Cap ($L=200$)

| Channel                      |   Budget (L) | Corpus-Macro TermRecall@L   | Corpus-Macro TermPrecision@L   | Corpus-Macro NearBestHit@L   | Corpus-Macro ReferenceBOR@L   | Corpus-Macro RecallHit@1000   | Corpus-Macro RawDocOppRecall@1000   | Corpus-Macro SafeDocOppRecall@1000   |
|:-----------------------------|-------------:|:----------------------------|:-------------------------------|:-----------------------------|:------------------------------|:------------------------------|:------------------------------------|:-------------------------------------|
| WholeQueryBGE                |          200 | 7.7%                        | 1.2%                           | 0.0%                         | 71.0%                         | 100.0%                        | 48.7%                               | 51.6%                                |
| AnchorBGEFiltered            |          200 | 5.5%                        | 1.2%                           | 0.0%                         | 63.1%                         | 100.0%                        | 40.6%                               | 13.5%                                |
| AnchorBGEAll                 |          200 | 5.5%                        | 1.2%                           | 0.0%                         | 63.1%                         | 100.0%                        | 40.6%                               | 13.5%                                |
| PPMISidecar                  |          200 | 8.4%                        | 3.1%                           | 50.0%                        | 50.0%                         | 100.0%                        | 68.4%                               | 86.9%                                |
| SparseLexicalContextProfiles |          200 | 1.0%                        | 0.4%                           | 0.0%                         | 1.8%                          | 100.0%                        | 16.9%                               | 5.5%                                 |
| AcronymDefinitionRescue      |          200 | 0.3%                        | 1.5%                           | 0.0%                         | 1.8%                          | 100.0%                        | 1.1%                                | 0.7%                                 |
| RRF_Core3                    |          200 | 15.3%                       | 3.2%                           | 50.0%                        | 81.5%                         | 100.0%                        | 68.0%                               | 79.6%                                |
| RRF_Extended                 |          200 | 11.1%                       | 2.5%                           | 50.0%                        | 81.5%                         | 100.0%                        | 65.8%                               | 77.9%                                |

---

## 2b. Table 2-Diag: LivePPMI Diagnostic Comparator ($L=200$)

| Channel     |   Budget (L) | Corpus-Macro TermRecall@L   | Corpus-Macro TermPrecision@L   | Corpus-Macro NearBestHit@L   | Corpus-Macro ReferenceBOR@L   | Corpus-Macro RecallHit@1000   | Corpus-Macro RawDocOppRecall@1000   | Corpus-Macro SafeDocOppRecall@1000   |
|:------------|-------------:|:----------------------------|:-------------------------------|:-----------------------------|:------------------------------|:------------------------------|:------------------------------------|:-------------------------------------|
| LivePPMI    |          200 | 8.4%                        | 3.1%                           | 50.0%                        | 50.0%                         | 100.0%                        | 68.4%                               | 86.9%                                |
| PPMISidecar |          200 | 8.4%                        | 3.1%                           | 50.0%                        | 50.0%                         | 100.0%                        | 68.4%                               | 86.9%                                |

---

## 2c. Checkpoint B Operational Loss Gate Evaluation

**Gate Status:** **PASSED**  
- **Corpus-Macro $\Delta$nDCG@10 Loss:** `0.0000` (threshold: `0.02`)  
- **Corpus-Macro RawDocOppRecall@1000 Loss:** `0.0000` (threshold: `0.02`)  

---

## 3. Table: 100% Counterfactual Label Coverage Verification

| Dataset     |   Queries |   Reference Universe Pairs |   Expected Variants |   Evaluated Variants |   Missing Triples |   Extra Triples | Labeling Coverage %   | Coverage Status   |
|:------------|----------:|---------------------------:|--------------------:|---------------------:|------------------:|----------------:|:----------------------|:------------------|
| bright_aops |         1 |                       1930 |                9650 |                 9650 |                 0 |               0 | 100.00%               | 100.0% COMPLETE   |
| nfcorpus    |         1 |                       1662 |                8310 |                 8310 |                 0 |               0 | 100.00%               | 100.0% COMPLETE   |
| scifact     |         1 |                       1457 |                7285 |                 7285 |                 0 |               0 | 100.00%               | 100.0% COMPLETE   |
| trec_covid  |         1 |                       3389 |               16945 |                16945 |                 0 |               0 | 100.00%               | 100.0% COMPLETE   |
