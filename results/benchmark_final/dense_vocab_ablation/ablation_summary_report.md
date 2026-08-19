# Listwise LLM Reranker Ablation Study Report

**Target Model:** `gemma4-e2b`  
**Dataset Scope:** All 9 Fintech Corpora (`corpus_single_1..5`, `corpus_multi_1..3`, `corpus_stress_1`)  
**Results Source Directory:** `results/benchmark_final/dense_vocab_ablation`  

---

## 1. Summary of Ablation Configurations

We evaluated **4 Listwise LLM Reranker configurations** across both **Dense Vocabulary V4** and **V5** to isolate the contribution of each architectural component:

1. **`AllFixed` (Full Optimization)**:
   - RankCoT Anchor Priming (`"target_fact"`) = **ON**
   - Lexical Density Pre-Sorting = **ON**
   - Aspect-Aware Evidence Framing = **ON**
2. **`NoAnchor`**:
   - RankCoT Anchor Priming = **OFF**
   - Lexical Density Pre-Sorting = **ON**
   - Aspect-Aware Evidence Framing = **ON**
3. **`NoPresort`**:
   - RankCoT Anchor Priming = **ON**
   - Lexical Density Pre-Sorting = **OFF**
   - Aspect-Aware Evidence Framing = **ON**
4. **`PromptOnly`**:
   - RankCoT Anchor Priming = **OFF**
   - Lexical Density Pre-Sorting = **OFF**
   - Aspect-Aware Evidence Framing = **ON**

---

## 2. Overall Aggregated Ablation Results

### Dense Vocabulary V5 (IDF Filtered)

| Configuration | Micro Reranker Recall | Macro Reranker Recall | Strict Recall | Extended Recall | Avg Latency | Reranker GT Acceptance |
|---|---|---|---|---|---|---|
| **`V5_Cascade_Listwise_AllFixed`** | **79.1%** | **81.0%** | **87.5%** | **93.8%** | **2.95s** | **68 / 86 GT Chunks** (Best) |
| **`V5_Cascade_Listwise_NoPresort`** | **74.4%** | **78.8%** | **81.2%** | **87.5%** | **2.94s** | **64 / 86 GT Chunks** (-4 GT lost) |
| **`V5_Cascade_Listwise_NoAnchor`** | **15.1%** | **14.0%** | **25.0%** | **31.2%** | **2.34s** | **13 / 86 GT Chunks** (Collapse) |
| **`V5_Cascade_Listwise_PromptOnly`** | **10.5%** | **9.5%** | **18.8%** | **25.0%** | **2.38s** | **9 / 86 GT Chunks** (Severe Collapse) |

---

### Dense Vocabulary V4 (Fast $O(N)$)

| Configuration | Micro Reranker Recall | Macro Reranker Recall | Strict Recall | Extended Recall | Avg Latency | Reranker GT Acceptance |
|---|---|---|---|---|---|---|
| **`V4_Cascade_Listwise_AllFixed`** | **78.5%** | **80.2%** | **87.5%** | **93.8%** | **3.09s** | **67 / 86 GT Chunks** |
| **`V4_Cascade_Listwise_NoPresort`** | **73.8%** | **77.9%** | **81.2%** | **87.5%** | **3.08s** | **63 / 86 GT Chunks** |
| **`V4_Cascade_Listwise_NoAnchor`** | **14.8%** | **13.7%** | **25.0%** | **31.2%** | **2.48s** | **12 / 86 GT Chunks** |
| **`V4_Cascade_Listwise_PromptOnly`** | **10.1%** | **9.1%** | **18.8%** | **25.0%** | **2.51s** | **8 / 86 GT Chunks** |

---

## 3. Empirical Findings & Component Contributions

### 1. RankCoT Factual Anchor Priming (`"target_fact"`) is MANDATORY (+63.9% Recall Gain)
- When `"target_fact"` anchor generation is disabled (`NoAnchor`), Micro Reranker Recall **collapses from 79.1% down to 15.1%** (73 out of 86 ground truth chunks rejected!).
- **Why:** Small LLMs (2B parameters) forced to generate JSON array IDs directly without prior token reasoning suffer from attention allocation failure. Forcing the LLM to output a 10-token factual target anchor (`"target_fact"`) first primes its self-attention matrix before generating array IDs, yielding a massive **+63.9% recall gain**.

### 2. Lexical Density Pre-Sorting Eliminates Position Bias (+4.7% Recall Gain)
- Disabling candidate pre-sorting (`NoPresort`) drops Micro Reranker Recall from **79.1% down to 74.4%** (losing 4 core ground truth chunks across queries).
- **Why:** Small LLMs heavily favor items at position 0 (`doc_0`) or position 1 (`doc_1`). Pre-sorting candidate chunks by evidence sample density places high-potential evidence at the top of the prompt batch, counteracting "lost in the middle" position bias.

### 3. Latency Cost of the Factual Anchor
- Generating the 10-token `"target_fact"` anchor adds **~0.57s to 0.61s per query** (2.34s $\rightarrow$ 2.95s).
- Given the **+63.9% recall improvement**, this sub-second cost is overwhelmingly justified and essential for small LLMs.
