# Phase 2.1a Gate 1 Candidate Selection Progress Report

**Date:** 2026-09-21  
**Active Codebase:** `CRVE/`  
**Evaluation Scope:** 40-Query Probe (`probe_40_qids`, 10 queries $\times$ 4 dev corpora: `scifact`, `bright_aops`, `nfcorpus`, `trec_covid`)  
**Frozen Configuration:** [`CRVE/configs/gate1_phase2_1a.yaml`](file:///home/donghv/Projects/Edge-RAG/CRVE/configs/gate1_phase2_1a.yaml)  

---

## 1. Executive Summary

We have executed the **40-query probe evaluation** under Phase 2.1a Gate 1 Candidate Selection Under Uncertainty following all Revision Round 3 amendments.
- **Action Coverage:** **100.0% COMPLETE** (407,825 / 407,825 variants evaluated across $\mathcal{R}_q^{\text{diag}} \times W$; 0 missing, 0 extra).
- **Sorted Canonical Invariance:** Asserted and proven exact logical equality of `reference_universe.parquet` with and without `--measure-fidelity`.
- **Macro Operational Loss Gate:**
  - **Corpus-Macro $\Delta$nDCG@10 Loss:** **0.0072** (threshold: $\le 0.02$, **PASSED**)
  - **Corpus-Macro RawDocOppRecall@1000 Loss:** **0.0186** (threshold: $\le 0.02$, **PASSED**)
- **Channel Dominance at $L=200$:**
  - `RRF_Extended` achieves **20.4% TermRecall@200**, **57.5% NearBestHit@200**, **73.3% ReferenceBOR@200**, **87.5% RecallHit@1000**, and **67.7% RawDocOppRecall@1000**.
  - `PPMISidecar` achieves **19.1% TermRecall@200**, **60.7% ReferenceBOR@200**, **91.7% RecallHit@1000** (surpassing LivePPMI's 87.5%), and **63.9% RawDocOppRecall@1000**.
- **Resource Footprint:** Peak RSS was **5.963 GiB** (well below the 12.0 GiB watchdog threshold and 15 GiB WSL2 ceiling); CUDA VRAM peak was **192.65 MiB**.

---

## 2. Table 1: Reference Opportunity Ceiling vs Baseline

| Dataset | Partition | Queries | Baseline nDCG@10 | Baseline R@1000 | Ceiling $\Delta$nDCG@10 [95% CI] | Ceiling Net Rel Docs | Materially Addressable % ($g^* \ge 0.005$) | Recall Addressable % ($r^* \ge 1$) |
|:---|:---|---:|---:|---:|:---|---:|:---|:---|
| `bright_aops` | Dev | 10 | 0.0939 | 0.3643 | +0.1323 [0.0390, 0.2335] | 1.1 | 50.0% | 80.0% |
| `nfcorpus` | Dev | 10 | 0.2855 | 0.3212 | +0.4136 [0.2376, 0.6128] | 8.6 | 80.0% | 80.0% |
| `scifact` | Dev | 10 | 0.6526 | 1.0000 | +0.3025 [0.1025, 0.5287] | 0.0 | 50.0% | 0.0% |
| `trec_covid` | Dev | 10 | 0.5724 | 0.3692 | +0.2971 [0.2146, 0.3820] | 31.2 | 100.0% | 100.0% |
| **Corpus-Macro Dev** | **Macro** | **40** | **0.4011** | **0.5137** | **+0.2864** | **10.22** | **70.0%** | **65.0%** |

---

## 3. Table 2: Operational Channels Comparison ($L=200$)

| Channel | Budget ($L$) | Corpus-Macro TermRecall@$L$ | Corpus-Macro TermPrecision@$L$ | Corpus-Macro NearBestHit@$L$ | Corpus-Macro ReferenceBOR@$L$ | Corpus-Macro RecallHit@1000 | Corpus-Macro RawDocOppRecall@1000 | Corpus-Macro SafeDocOppRecall@1000 |
|:---|---:|:---|:---|:---|:---|:---|:---|:---|
| `WholeQueryBGE` | 200 | 12.3% | 2.5% | 41.2% | 63.6% | 83.3% | 55.4% | 50.6% |
| `AnchorBGEFiltered` | 200 | 8.1% | 2.3% | 41.2% | 62.6% | 79.2% | 50.6% | 44.2% |
| `AnchorBGEAll` | 200 | 9.1% | 2.4% | 46.2% | 65.3% | 83.3% | 54.3% | 47.8% |
| `PPMISidecar` | 200 | 19.1% | 6.6% | 44.4% | 60.7% | **91.7%** | 63.9% | 61.0% |
| `SparseLexicalContextProfiles` | 200 | 13.6% | 3.7% | 40.0% | 50.1% | 83.3% | 47.0% | 45.9% |
| `AcronymDefinitionRescue` | 200 | 0.7% | 7.2% | 5.0% | 13.3% | 54.2% | 12.5% | 11.4% |
| `RRF_Core3` | 200 | 20.0% | 4.3% | 54.4% | **74.1%** | 87.5% | 64.7% | 58.7% |
| `RRF_Extended` | 200 | **20.4%** | 4.6% | **57.5%** | 73.3% | 87.5% | **67.7%** | **62.2%** |

---

## 4. Table 2-Diag: LivePPMI Diagnostic Comparator ($L=200$)

| Channel | Budget ($L$) | Corpus-Macro TermRecall@$L$ | Corpus-Macro TermPrecision@$L$ | Corpus-Macro NearBestHit@$L$ | Corpus-Macro ReferenceBOR@$L$ | Corpus-Macro RecallHit@1000 | Corpus-Macro RawDocOppRecall@1000 | Corpus-Macro SafeDocOppRecall@1000 |
|:---|---:|:---|:---|:---|:---|:---|:---|:---|
| `LivePPMI` | 200 | 19.8% | 7.0% | 49.4% | 63.0% | 87.5% | 64.2% | 61.4% |
| `PPMISidecar` | 200 | 19.0% | 6.6% | 44.4% | 60.7% | **91.7%** | 63.9% | 61.0% |

---

## 5. Frozen Review Package Location

The full review package containing raw Parquet outputs, compiled CSV tables, and diagnostic JSONs is frozen at:
- **Directory:** [`for_review/selection_phase/gate_1/run_2/probe_40_artifacts/`](file:///home/donghv/Projects/Edge-RAG/for_review/selection_phase/gate_1/run_2/probe_40_artifacts/)
- **Parquets:** [`results/gate1_selection/probe_40_eval/`](file:///home/donghv/Projects/Edge-RAG/results/gate1_selection/probe_40_eval/)
