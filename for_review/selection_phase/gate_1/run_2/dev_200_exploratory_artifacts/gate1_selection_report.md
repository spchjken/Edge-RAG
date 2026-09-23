# Phase 2 Gate 1 Candidate Selection Report

> [!WARNING]
> ### ⚠️ EXPLORATORY PROTOCOL AMENDMENT (`PA-GATE1-20260922-01`)
> **Status:** EXPLORATORY CHARACTERIZATION (Failed Checkpoint B Operational Loss Gate)
> 
> This 200-query run was executed under an explicit exploratory continuation option after Checkpoint B failed during Stage 1 probe evaluation.
> - **Checkpoint B Status:** **FAILED**
> - **Corpus-Macro Losses:** $\Delta$nDCG@10 Loss = `0.0072` (limit $\le 0.02$), DocOppRecall@1000 Loss = `0.0186` (limit $\le 0.02$)
> - **Failed Per-Corpus Limits (TREC-COVID):** $\Delta$nDCG@10 Loss = `0.027` (limit $\le 0.02$), DocOppRecall@1000 Loss = `0.0742` (limit $\le 0.02$)
> - **Protocol Compliance:** This run does **NOT** claim confirmatory compliance with the frozen Phase 2.1a protocol. It serves strictly as an empirical comparison across the full 200-query dev set.

---
**Master Audit Parquet:** [`results/gate1_selection/dev_200_eval/gate1_candidate_audit.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/dev_200_eval/gate1_candidate_audit.parquet)  
**Cutoff Entries Parquet:** [`results/gate1_selection/dev_200_eval/gate1_cutoff_entries.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/dev_200_eval/gate1_cutoff_entries.parquet)  
**Reference Universe Parquet:** [`results/gate1_selection/dev_200_eval/reference_universe.parquet`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/dev_200_eval/reference_universe.parquet)  
**Config Hash:** `907e37c68546b95917839a4ea069997cca358ba29f13d912342bf8e6da549aa0`  
**Git Commit:** `82e7e62d0c735cb8dec35c9b563bd6ad55be8fa6` (dirty: `True`)  
**Peak Process-Tree RSS:** `6.773 GiB`  
**Peak CUDA VRAM:** `196.82 MiB`  
**Generated At:** 2026-09-22 17:04:24 UTC  

---

## 1. Table 1: Reference Opportunity Ceiling vs Baseline

| Dataset                      | Partition   |   Queries |   Baseline nDCG@10 |   Baseline R@1000 | Ceiling Delta-nDCG@10    |   Ceiling Net Rel Docs | Materially Addressable % (g* >= 0.005)   | Recall Addressable % (r* >= 1)   |
|:-----------------------------|:------------|----------:|-------------------:|------------------:|:-------------------------|-----------------------:|:-----------------------------------------|:---------------------------------|
| bright_aops                  | Dev         |        50 |             0.0627 |            0.4798 | +0.1211 [0.0770, 0.1761] |                   0.64 | 44.0%                                    | 54.0%                            |
| nfcorpus                     | Dev         |        50 |             0.3111 |            0.3559 | +0.4214 [0.3308, 0.5016] |                   9.42 | 82.0%                                    | 86.0%                            |
| scifact                      | Dev         |        50 |             0.6444 |            0.98   | +0.2540 [0.1659, 0.3517] |                   0.02 | 44.0%                                    | 2.0%                             |
| trec_covid                   | Dev         |        50 |             0.6055 |            0.4486 | +0.2660 [0.2181, 0.3149] |                  25.04 | 98.0%                                    | 100.0%                           |
| Corpus-Macro Dev (4 Corpora) | Macro       |       200 |             0.4059 |            0.5661 | +0.2656                  |                   8.78 | 67.0%                                    | 60.5%                            |

---

## 2. Table 2: Operational Channels Comparison at Deployable Cap ($L=200$)

| Channel                      |   Budget (L) | Corpus-Macro TermRecall@L   | Corpus-Macro TermPrecision@L   | Corpus-Macro NearBestHit@L   | Corpus-Macro ReferenceBOR@L   | Corpus-Macro RecallHit@1000   | Corpus-Macro RawDocOppRecall@1000   | Corpus-Macro SafeDocOppRecall@1000   |
|:-----------------------------|-------------:|:----------------------------|:-------------------------------|:-----------------------------|:------------------------------|:------------------------------|:------------------------------------|:-------------------------------------|
| WholeQueryBGE                |          200 | 7.9%                        | 2.4%                           | 37.0%                        | 59.8%                         | 86.2%                         | 68.0%                               | 63.8%                                |
| AnchorBGEFiltered            |          200 | 5.9%                        | 2.2%                           | 31.1%                        | 52.7%                         | 85.0%                         | 64.9%                               | 61.7%                                |
| AnchorBGEAll                 |          200 | 6.1%                        | 2.3%                           | 34.5%                        | 53.8%                         | 86.5%                         | 66.4%                               | 63.0%                                |
| PPMISidecar                  |          200 | 20.5%                       | 6.2%                           | 50.6%                        | 69.7%                         | 88.5%                         | 72.0%                               | 69.4%                                |
| SparseLexicalContextProfiles |          200 | 15.3%                       | 3.9%                           | 42.4%                        | 54.0%                         | 81.5%                         | 59.7%                               | 58.5%                                |
| AcronymDefinitionRescue      |          200 | 0.7%                        | 3.4%                           | 4.2%                         | 12.3%                         | 61.7%                         | 34.8%                               | 33.1%                                |
| RRF_Core3                    |          200 | 14.9%                       | 4.1%                           | 47.8%                        | 70.0%                         | 89.2%                         | 75.0%                               | 71.8%                                |
| RRF_Extended                 |          200 | 16.9%                       | 4.3%                           | 56.2%                        | 74.2%                         | 91.1%                         | 78.3%                               | 73.9%                                |


---

## 2c. Checkpoint B Operational Loss Gate Evaluation

**Gate Status:** **FAILED**  
- **Corpus-Macro $\Delta$nDCG@10 Loss:** `0.0072` (threshold: `0.02`)  
- **Corpus-Macro RawDocOppRecall@1000 Loss:** `0.0186` (threshold: `0.02`)  

---

## 3. Table: 100% Counterfactual Label Coverage Verification

| Dataset     |   Queries |   Reference Universe Pairs |   Expected Variants |   Evaluated Variants |   Missing Triples |   Extra Triples | Labeling Coverage %   | Coverage Status   |
|:------------|----------:|---------------------------:|--------------------:|---------------------:|------------------:|----------------:|:----------------------|:------------------|
| bright_aops |        50 |                      84476 |              422380 |               422380 |                 0 |               0 | 100.00%               | 100.0% COMPLETE   |
| nfcorpus    |        50 |                      86555 |              432775 |               432775 |                 0 |               0 | 100.00%               | 100.0% COMPLETE   |
| scifact     |        50 |                      76226 |              381130 |               381130 |                 0 |               0 | 100.00%               | 100.0% COMPLETE   |
| trec_covid  |        50 |                     167638 |              838190 |               838190 |                 0 |               0 | 100.00%               | 100.0% COMPLETE   |
