# 🏛️ Edge-RAG Pipeline V7 Architectural Plan: 5-Phase Anchored Lexical-Semantic Retriever

## Executive Summary

Edge-RAG Pipeline V2 (Schemas 1 through 6b) demonstrated that near-zero VRAM ($0.09\text{ GB}$) lexical-semantic hybrid retrieval can match or surpass heavy neural bi-encoders on standard corpora without the latency or memory footprint of SPLADE-v3 ($174\text{s}$ TTI, $4.2\text{ GB}$ VRAM) or Dense BGE-Large ($17.5\text{s}$ TTI, $2.6\text{ GB}$ VRAM).

However, empirical evaluations across diverse benchmarks (`fused_stress_500`, `enterpriserag`, `liverag`, and BEIR datasets) revealed that the legacy architecture suffered from modular coupling, discrete integer repetition artifacts, and query drift on rare technical vocabulary. A separate audit also established two structural weaknesses this plan now addresses head-on:

1. **Morphological blind spot — resolved at the analyzer.** The index is no longer exact-token; V7 adopts an **index-time-stemmed (KStem) index** (already built in `EdgeRAGAnalyzer` / `AnalyzedLuceneBM25`), which conjoins inflectional variants (`price`/`prices`/`pricing`) into one posting list. The Rev 2 two-tier fold-in machinery is **cancelled**; the only residual morphology work is a small **WordNet suppletion override** in the analyzer — fully specified in [`morphology_expansion_strategy.md`](morphology_expansion_strategy.md) (Rev 3).
2. **Baseline parity achieved.** The [`lucene_bm25_parity_plan.md`](lucene_bm25_parity_plan.md) Stages 1–2 are complete: `AnalyzedLuceneBM25` now has the analyzer chain, posting-list index, and weighted retrieval. Empirically it **beats every v1/v5/v6 schema** on the doc-level benchmarks, so it is now the primary control V7 must beat.

This document restructures the **Edge-RAG V7 Retriever (`BM25Dense_V7`)** into **5 clean, decoupled architectural phases** across two distinct execution lifecycles: **Index-Time (Offline / Startup)** and **Query-Time (Online Execution)**, and pins down where each of the above concerns lands.

---

## 1. End-to-End 5-Phase Architecture & Data Flow

```mermaid
graph TD
    classDef offline fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px,color:#000000;
    classDef query fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000000;
    classDef scoring fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000000;

    subgraph Offline ["Lifecycle A: Offline / Startup Phase (<0.3s)"]
        P1["Phase 1: Corpus Grounding & Index Construction<br/>(Inverted Index, Shared IDF, Analyzed Vocab (KStem),<br/>Dense Vocab Matrix V)"]:::offline
    end

    subgraph Online ["Lifecycle B: Online Query Execution (<15ms on CPU)"]
        P2["Phase 2: Query Dissection & Anchor Formulation<br/>(Heuristics, POS Priors, Centrality, Anchor Selection p,<br/>Stem-Based Dedup)"]:::query
        P3["Phase 3: Dense Semantic Probing & Adaptive Gating<br/>(1-Pass Batch Tensor Probing, Dual-Sim,<br/>Adaptive Gate τ_sim)"]:::query
        P4["Phase 4: IT-MPE Mass Allocation & Vector Compilation<br/>(Cross-Root Synonyms, Budget μ,<br/>Sparse Vector w_Q)"]:::query
        P5["Phase 5: Vectorized Inverted Posting Retrieval<br/>(Posting-List Traversal, Weighted Lucene BM25, Min-Heap Top-K)"]:::scoring
    end

    P1 -.-> P2
    P1 -.-> P3
    P1 -.-> P4
    P1 -.-> P5

    P2 --> P3 --> P4 --> P5
```

---

## 2. Detailed Phase Specifications & Upgrades

### Phase 1: Corpus Grounding & Index Construction (Index-Time / Startup)
* **Module Owner:** `src/pipeline_v2/indexer/` ([`bm25_lucene_indexer.py`](src/pipeline_v2/indexer/bm25_lucene_indexer.py), [`corpus_idf_registry.py`](src/pipeline_v2/indexer/corpus_idf_registry.py), [`corpus_vocab_builder.py`](src/pipeline_v2/indexer/corpus_vocab_builder.py), [`dense_vocab_matrix.py`](src/pipeline_v2/indexer/dense_vocab_matrix.py))
* **Role:** Establishes static corpus statistics, inverted posting lists, shared Lucene IDF tables, and the analyzed (stemmed) dense vocabulary embedding matrix before query arrival.
* **Baseline Heritage:** Built on the fast, lightweight **V5 Indexer Foundation (`BM25Dense_V5b`)**, operating in $<0.3\text{s}$ TTI and $<0.09\text{GB}$ VRAM.
* **Canonical Baseline Reference:** [`docs/ARCHITECTURE.md` (§2)](docs/ARCHITECTURE.md) and [`pathway_bm25_dense_aspect.md` (§2.1, §4)](src/pipeline_v2/expansion/pathway_bm25_dense_aspect.md)

> **Lucene-Parity Note (cross-ref [`lucene_bm25_parity_plan.md`](lucene_bm25_parity_plan.md)):** the analyzer chain (canonical tokenization, possessive stripping, stopword removal, case folding, and **KStem**) is **already implemented** in `EdgeRAGAnalyzer` and proven in `AnalyzedLuceneBM25`. Phase 1 adopts this **index-time-stemmed** index wholesale — stemming is **no longer query-side**. Technical-token protection is preserved by the analyzer's exemption set, not by keeping stemming out of the index.

#### 1. Finalized Upgrades for Phase 1
* **Upgrade 1.1 — Canonical Tokenization (`EdgeRAGTokenizer`):**
  - **Role:** the tokenization stage of `EdgeRAGAnalyzer` — not the sole source of truth (that is the analyzer, Upgrade 1.7).
  - **Tokenization Pattern (`r'\b[a-z0-9]+(?:[-._][a-z0-9]+)+\b|\b[a-z0-9]{2,}\b'`):**
    - Splits on whitespace and standard delimiter punctuation (commas, parentheses, brackets, slashes).
    - Preserves alphanumeric technical compounds and version identifiers intact (e.g., `"qwen2.5-7b"`, `"gpt-4"`, `"llama-3.1"`, `"fp16"`, `"zero-shot"`).
    - Retains vital 2-letter technical acronyms (`"ai"`, `"ml"`, `"db"`, `"kv"`, `"ip"`).
  - **Bug Fix:** Eliminates cross-module tokenization boundary mismatches (e.g. indexing `"qwen-2.5"` in BM25 but querying `"qwen"` + `"2.5"`), guaranteeing 1:1 key parity across inverted index postings, vocabulary matrix, and query parsing.
* **Upgrade 1.2 — Scaled Vocabulary Pool (Static, Index-Time):**
  - Scale candidate vocabulary capacity from $1,000 \to N_{\text{vocab}} = \min(2,500,\ \#\text{distinct stems passing } DF \ge 2)$ in `CorpusVocabBuilder` using sublinear salience ranking ($\text{IDF} \times \ln(1 + \text{DF})$ for $\text{Doc\_Freq} \ge 2$), providing broad domain coverage at negligible VRAM ($<0.12\text{GB}$).
  - **Post-stemming recalibration:** $2,500$ was sized pre-stemming, when inflectional clones wasted slots. With a stem-diverse pool, re-validate $N_{\text{vocab}} \in \{1000, 1500, 2500\}$ empirically — $1,000$ stems may already cover what $2,500$ raw terms did.
* **Upgrade 1.3 — Normalized Vocab Matrix Preparation (Index-Time):**
  - Pre-allocate and $L_2$-normalize the analyzed vocabulary tensor $\mathbf{V} \in \mathbb{R}^{N_{\text{vocab}} \times 384}$ as a contiguous CUDA FP16 tensor in `DenseVocabMatrix`. (The query-time GEMM $\mathbf{E}_A \cdot \mathbf{V}^\top$ is a Phase 3 operation, already specified there.)
  - **Embedding-form note:** $\mathbf{V}$ embeds **surface forms** (BGE is surface-trained), while postings are keyed by **stems**; the analyzer bridges surface ↔ stem (Upgrade 1.7).
* **Upgrade 1.7 — Analyzer-Parity Wiring (the single source of truth):**
  - `EdgeRAGAnalyzer` (tokenizer → possessive → stopword → exemption → `stemmer_override` → KStem) is the sole token-producing source of truth.
  - **1.7a (index-side, Phase 1):** route `CorpusVocabBuilder`, `CorpusIDFRegistry`, and the posting index through `EdgeRAGAnalyzer`, so postings, IDF tables, and the vocab pool are keyed by analyzed stems.
  - **1.7b (query-side, Phase 2):** route `BM25DenseAspectExtractor`'s query analysis through the same analyzer, so query anchors are analyzed stems matching the index keys.
* **Upgrade 1.8 — WordNet Suppletion Override (new):**
  - Add a `stemmer_override` stage to `EdgeRAGAnalyzer` (before KStem) using WordNet `wn_s.pl` / `wn_v.pl` to map suppletive forms (`went → go`, `bought → buy`, `better → good`); exempt technical tokens. This is the only residual morphology work.

> **Retired / relocated from Phase 1** (kept for history; not active):
> - **1.4 In-Context Sample Embedding** — deferred to V7b/V8.
> - **1.5 Stem-Diversified Vocabulary Pool** — cancelled (index-time KStem conjoins inflections before the pool is built).
> - **1.6 `MorphologicalStemRegistry`** — cancelled (redundant once postings are keyed by stem).
> - **Query-Time Anchor Bailout (formerly part of 1.2)** — relocated to Phase 2 (needs the finished anchor list).

#### 2. Phase 1 Parameter Defaults & Calibration
| Parameter | Default Value | Mathematical & Operational Rationale |
| :--- | :---: | :--- |
| **$N_{\text{vocab}}$ (Vocab Pool Size)** | **$\min(2,500,\ \#\text{distinct stems})$** | Ceiling, not a guaranteed fill; re-calibrate post-stemming ($\{1000, 1500, 2500\}$). |
| **Analyzer `stemmer`** | **`kstem`** | Conservative IR stemmer applied at index and query time; conjoins inflections without derivational conflation. |
| **Analyzer `use_wordnet_override`** | **`true`** | WordNet `stemmer_override` maps suppletive forms (`went → go`); the only residual morphology stage. |

*Bailout params (`τ_rescue_IDF`, `min_len_rescue`, acronym exception) live in Phase 2, alongside the relocated Anchor Bailout operation.*

---

### Phase 2: Query Dissection & Anchor Formulation (Query Analysis)
* **Module Owner:** `src/pipeline_v2/expansion/` (`bm25_dense_aspect_extractor.py`)
* **Role:** Analyzes user query intent, isolates explicit technical entities, identifies grounded anchors, and computes continuous base weights $w(a)$.
* **Analyzer parity (Upgrade 1.7b) & pipeline order:** the query is analyzed with `EdgeRAGAnalyzer`, so anchors are keyed by analyzed stems. **Order matters** — ops 1–2 (entity extraction, POS) run on **raw text** (pre-lowercase, pre-stemming); the analyzer runs between op 2 and op 3; ops 3–7 operate on analyzed stems.

#### 1. Finalized Operations for Phase 2
  1. **Heuristic Entity Extraction (raw text):** Regex pattern matching for technical acronyms (`\b[A-Z]{2,}\b`), hyphenated/versioned identifiers (`\b[A-Za-z0-9\.]+(?:-[A-Za-z0-9\.]+)+\b`), and quoted phrases (`"..."`). Runs *before* `EdgeRAGAnalyzer` (acronyms require uppercase; the analyzer lowercases).
  2. **Linguistic POS Weighting (raw text):** Assigns grammatical prior weights on **surface tokens** (POS tagging needs unstemmed words):
     $$w_{\text{POS}}(\text{Noun / Entity}) = 1.25, \quad w_{\text{POS}}(\text{Verb}) = 0.85, \quad w_{\text{POS}}(\text{Modifier / Other}) = 0.70$$
     POS labels are carried onto the corresponding analyzed stems via token index.
  3. **Anchor Selection Policy ($p$, analyzed):** After `EdgeRAGAnalyzer`, select the top $N_{\text{anchors}} = \max(2, \lceil p \cdot |Q_{\text{clean}}| \rceil)$ distinct analyzed tokens, ranked by **corpus IDF** (Schemas 1–5) or **query centrality** (Schema 6). $|Q_{\text{clean}}|$ = distinct analyzed tokens after stopword removal.
  4. **Query Centrality Formulation:** Computes graph density in BGE embedding space:
     $$\text{Centrality}(a) = \frac{1}{|Q| - 1} \sum_{t \in Q \setminus \{a\}} \cos(\mathbf{e}_a, \mathbf{e}_t) \quad \text{or} \quad \cos(\mathbf{e}_a, \mathbf{e}_Q)$$
     **Surface-form note:** $\mathbf{e}_a, \mathbf{e}_t$ are embedded from **surface forms** (BGE is surface-trained); the analyzer bridges surface ↔ stem (Phase 1, Upgrade 1.3).
  5. **Continuous Anchor Base Weighting:**
     $$w(a) = w_{\text{POS}}(a) \times \left(1.0 + \gamma \cdot \frac{\text{IDF}(a)}{\max_{t \in Q} \text{IDF}(t)} \cdot \text{Centrality}(a)\right), \quad \gamma = 2.0 \quad \implies \quad w(a) \in [0.70,\ 3.75]$$
  6. **Anchor Deduplication & Entity Validation (semantic, additive):**
     - Replace the destructive prefix/suffix heuristic (`w.startswith(sel) ... → is_dup`) — which falsely conflates `plan`/`planet`, `cost`/`costume`, `organ`/`organization` — with **semantic cosine dedup** ($\text{CosSim} \ge 0.90$). (Stem-equality is now implicit: anchors are already analyzed stems, so identical stems are identical tokens.)
     - **Additive rule:** dedup may only prevent *duplicate anchor slots*; it must **never drop an analyzed token** from the retrieval vector.
  7. **Anchor Bailout (Rare Singleton Rescue, technical-only):** For each surviving anchor $a$ meeting the gate ($\text{IDF}(a) \ge 3.0$ and $\text{length}(a) \ge 3$, or $a \in \text{HeuristicEntities}$), perform an instant $O(1)$ token-boundary prefix/suffix lookup in the **exempt (unstemmed) technical slice** of the posting dictionary (`a-`, `a_`, `a.`, `a[0-9]`, `-a`) to bail out matching $DF=1$ compounds, versions, and typos (e.g., `gpt` $\to$ `gpt-4`, `gpt4`; `qwen` $\to$ `qwen2.5`, `qwen-vl`; `kv` $\to$ `kv-cache`).
     - **Weighting:** a bailed term $a'$ joins $\vec{w}_Q$ with the anchor's weight under score-space IDF damping, $w(a') = w(a) \cdot \min\left(1.0,\ \frac{\text{IDF}(a)}{\text{IDF}(a')}\right)$, **outside** the Phase-4 $\mu(Q)$ synonym budget (bailout is lexical rescue, not semantic expansion).

#### 2. Phase 2 Parameter Defaults & Calibration
| Parameter | Default Value | Mathematical & Operational Rationale |
| :--- | :---: | :--- |
| **$p$ (Anchor Selection Ratio)** | **$0.80$** | Recent sweeps favor $p \in [0.8, 1.0]$ for recall across broad corpora. |
| **$\gamma$ (Centrality Weight)** | **$2.0$** | Scales the IDF × centrality term in the anchor weight. |
| **$w_{\text{POS}}$ (POS priors)** | **Noun 1.25 / Verb 0.85 / Other 0.70** | Grammatical prior: content nouns dominate, modifiers damped. |
| **$\tau_{\text{rescue\_IDF}}$ (Bailout Anchor IDF Gate)** | **$3.0$** | Only anchors in $\le 4.98\%$ of the corpus trigger $DF=1$ bailout. |
| **$\text{min\_len}_{\text{rescue}}$ (Bailout Anchor Length)** | **$3$ chars** | Permits 3-letter technical anchors (`gpt`, `rag`, `sql`, `ehr`, `llm`). |
| **Regex Entity Acronym Exception** | **`[A-Z]{2,}`** | Raw-text (pre-lowercase) acronym detection for entity + bailout gates. |

* **Output Artifacts:** Grounded anchor dictionary $\mathcal{A} = \{(a, w(a))\}$ plus any bailed-out technical compounds.

---

### Phase 3: Dense Semantic Probing & Adaptive Gating (Vocabulary Projection)
* **Module Owner:** `src/pipeline_v2/expansion/` (`bm25_dense_aspect_extractor.py`)
* **Role:** Projects query anchors into the grounded corpus vocabulary space and enforces anchor-calibrated quality filters.
* **Core Operations:**
  1. **1-Pass Batch Tensor Probing:** Stacks anchor embeddings $\mathbf{E}_A \in \mathbb{R}^{|A| \times d}$ and computes full similarity tensor in a single matrix multiplication:
     $$\mathbf{S} = \mathbf{E}_A \cdot \mathbf{V}^\top \in \mathbb{R}^{|A| \times |\mathcal{V}|} \quad (\approx 1.2\text{ms})$$
  2. **Dual-Similarity Synthesis:** Blends anchor-specific and global query context:
     $$\text{Dual\_Sim}(a, v) = \beta \cdot \text{CosSim}(\mathbf{e}_a, \mathbf{e}_v) + (1 - \beta) \cdot \text{CosSim}(\mathbf{e}_Q, \mathbf{e}_v), \quad \beta = 0.65$$
  3. **Adaptive Similarity Quality Gate:** Replaces rigid binary entity freezing with an anchor-importance scaled threshold:
     $$\tau_{\text{sim}}(a) = \tau_{\text{base}} + \Delta\tau \cdot \left(\frac{\text{IDF}(a)}{\text{IDF}_{\max}}\right) \quad \implies \tau_{\text{sim}}(a) \in [0.80, \ 0.90]$$
  4. **Stem-Diversity Gate** *(removed):* No longer needed — the vocabulary pool is built from analyzed (stemmed) tokens, so inflectional clones never enter the pool; probing returns cross-root candidates by construction.
  5. **Candidate Extraction:** Gathers candidate vocabulary terms satisfying $\text{Dual\_Sim}(a, v) \ge \tau_{\text{sim}}(a)$, retaining top $C_{\text{exp}} = 2$ synonyms per anchor.
* **Output Artifacts:** Per-anchor candidate sets $\text{Syn}(a) = \{s_1, \dots, s_K\}$ with corresponding similarity scores, all cross-root.

---

### Phase 4: IT-MPE Mass Allocation & Vector Compilation (Expansion Weighting)
* **Module Owner:** `src/pipeline_v2/expansion/` (`bm25_dense_aspect_extractor.py`)
* **Role:** Assigns continuous float weights to **cross-root semantic synonyms** under the Information-Theoretic Mass-Preserving Expansion (IT-MPE) Theorem, strictly preventing rare-synonym score hijacking.
* **Core Operations:**
  1. **Query-Level Expansion Budget:**
     $$\mu(Q) = 0.35 \times \left(1 - 0.5 \times \frac{\max_{t \in Q} \text{IDF}(t)}{\text{IDF}_{\text{max\_corpus}}}\right) \quad \implies \mu(Q) \in [0.18, \ 0.35]$$
  2. **Temperature-Scaled Softmax Allocation:**
     $$p(s_k \mid a) = \frac{\exp\left(\frac{\text{CosSim}(\mathbf{e}_{s_k}, \mathbf{e}_a)}{\tau}\right)}{\sum_{j=1}^K \exp\left(\frac{\text{CosSim}(\mathbf{e}_{s_j}, \mathbf{e}_a)}{\tau}\right)}, \quad \tau = 0.10$$
  3. **Score-Space IDF Damping Factor:** Prevents rare synonyms ($\text{IDF}(s) \gg \text{IDF}(a)$) from overpowering the anchor:
     $$\text{Damping}(s_k, a) = \min\left(1.0, \ \frac{\text{IDF}(a)}{\text{IDF}(s_k)}\right)$$
  4. **Continuous Expansion Weight:**
     $$w(s_k \mid a) = \mu(Q) \cdot w(a) \cdot \min\left(1.0, \ \frac{\text{IDF}(a)}{\text{IDF}(s_k)}\right) \cdot p(s_k \mid a)$$
  5. **Sparse Vector Synthesis:** Compiles primary anchors $w(a)$ and synonyms $w(s \mid a)$ into a unified sparse float vector $\vec{w}_Q \in \mathbb{R}^{|V|}$, summing weights when two entries collide on the same term (Lucene boost-summing semantics).
* **Mathematical Invariant (single-tier):**
  $$\sum_{s \in S(a)} w(s \mid a) \le \mu(Q) \cdot w(a)$$
  In score space, each synonym inherits $\min(1, \text{IDF}_a/\text{IDF}_s)$ damping, so no single synonym hit can out-score a single anchor hit.
* **Output Artifacts:** Continuous sparse term weight dictionary $\vec{w}_Q = \{t: w_Q(t)\}$.

---

### Phase 5: Vectorized Inverted Posting Retrieval (Scoring Execution)
* **Module Owner:** `src/pipeline_v2/indexer/` (`bm25_lucene_indexer.py`, `posting_index.py`)
* **Role:** Evaluates the compiled sparse vector directly over inverted posting lists with sub-15ms CPU latency.
* **Status:** *(largely implemented already — `InvertedPostingIndex` + `AnalyzedLuceneBM25.retrieve_weighted` exist from parity-plan Stage 2. Remaining work is routing the extractor's $\vec{w}_Q$ through it, not building the index.)*
* **Core Operations:**
  1. **Posting-List Traversal:** Traverse inverted posting lists only for active terms $t \in \vec{w}_Q$ — a true posting-list structure (compact `array`-typed doc-id/TF arrays) replaces the legacy O($|Q| \cdot N$) full scan.
  2. **Single-Pass Weighted Lucene BM25 Scoring:**
     $$\text{Score}(D, Q) = \sum_{t \in \vec{w}_Q} w_Q(t) \cdot \text{IDF}(t) \cdot \frac{\text{TF}(t, D) \cdot (k_1 + 1)}{\text{TF}(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgDL}}\right)}$$
  3. **Top-$K$ Selection:** Accumulates scores into a pre-allocated `float32` buffer and extracts top-$K$ chunks via a min-heap ($<15\text{ms}$ on 17k chunks).
* **Semantics note:** this consumes $\vec{w}_Q$ directly via `retrieve_weighted`; token **repetition is retired** (the legacy repetition-scaling path is removed from the pipeline path, kept only in the frozen baseline). Unseen query terms receive IDF computed with `docFreq = 0`.
* **Output Artifacts:** Ranked list of retrieved candidate chunks `List[Dict[str, Any]]`.

---

## 3. Mapping Current V7 Upgrades to the 5 Phases

| Upgrade / Refinement | Target Phase | Implementation Impact |
| :--- | :--- | :--- |
| **Canonical Tokenization (`EdgeRAGTokenizer`)** | **Phase 1** | Tokenization stage of `EdgeRAGAnalyzer`; eliminates cross-module boundary mismatches. *(done)* |
| **Analyzer Chain (possessive + stopword + case + KStem, index-time)** | **Phase 1** | Adopted from `EdgeRAGAnalyzer`; stemming is index-time, not query-side. *(done, extend with suppletion override)* |
| **Punctuation Tokenization** | **Phase 1** | Canonical `EdgeRAGTokenizer`; splits delimiters, preserves technical compounds. *(done)* |
| **WordNet Suppletion Override (Upgrade 1.8)** | **Phase 1** | Add `stemmer_override` stage to `EdgeRAGAnalyzer` (suppletion → lemma). |
| **Analyzer-Parity Wiring — index-side (Upgrade 1.7a)** | **Phase 1** | Route VocabBuilder / IDF registry / posting index through `EdgeRAGAnalyzer`. |
| **Analyzer-Parity Wiring — query-side (Upgrade 1.7b)** | **Phase 2** | Route `BM25DenseAspectExtractor` query analysis through `EdgeRAGAnalyzer`. |
| **Rare Singleton Rescue / Anchor Bailout** | **Phase 2** | Query-time bailout rescues $DF=1$ technical compounds; pool floor stays $DF \ge 2$. |
| **1-Pass Batch Tensor Probing** | **Phase 3** | Single matrix multiplication $\mathbf{E}_A \cdot \mathbf{V}^\top$ in `DenseVocabMatrix`. |
| **High Anchor Selection Ratio ($p \in [0.8, 1.0]$)** | **Phase 2** | Updates default $p$ parameter in `configs/pipeline_v2.yaml` and anchor extractor. |
| **Continuous POS & Centrality Weighting** | **Phase 2** | Implements continuous $w(a) \in [0.70, 3.75]$ replacing discrete integer repetition. |
| **Semantic Anchor Dedup (additive)** | **Phase 2** | Replaces destructive prefix/suffix heuristic with cosine $\ge 0.90$ dedup (stems already conjoined by the analyzer). |
| **Adaptive Similarity Quality Gate ($\tau_{\text{sim}} \in [0.80, 0.90]$)** | **Phase 3** | Replaces hard binary entity freezing with dynamic similarity cutoff based on anchor IDF. |
| **Dual-Sim Context Synthesis ($\beta = 0.65$)** | **Phase 3** | Combines local anchor embedding with global query embedding. |
| **IT-MPE Continuous Mass Allocation & Score-Space Damping** | **Phase 4** | Enforces $\min(1, \text{IDF}_a/\text{IDF}_s)$ damping and temperature-scaled softmax allocation (single-tier). |
| **Retirement of Discrete Repetition** | **Phase 4 / 5** | Replaces discrete string replication (`augmented_token_list`) with sparse float vector $\vec{w}_Q$. |
| **Posting-List Index (no full scan)** | **Phase 5** | Compact posting arrays; traversal cost ∝ matched postings. *(done)* |
| **Float-Weighted Inverted Posting Accumulator** | **Phase 5** | `retrieve_weighted` consumes $\vec{w}_Q$ directly. *(done)* |

Removed (obsolete under index-time stemming): Stem-Diversified Vocabulary Pool (1.5), `MorphologicalStemRegistry` (1.6), Stem-Diversity Gate, Tier-1 Fold-In + Synonym Closure, Budget Tier-Split ($\eta_{\text{morph}}$).

---

## 4. Evaluation Protocol & Targets

### 4.1 Baselines (frozen controls — integrity rules)

| Baseline condition | Role | Status |
| :--- | :--- | :--- |
| `AnalyzedLuceneBM25` (analyzed-parity: analyzer + posting index) | **New primary control** — strongest lexical baseline, beats v1/v5/v6 on doc-level benchmarks | **Frozen** |
| `LuceneBM25Baseline` (legacy, pure-Python formula port) | Historical control; all legacy targets reference it | **Frozen** |
| `BM25Baseline` (`rank_bm25` pip package) | Secondary control; external-package credibility | **Frozen, optional** |
| `BM25_stemmed` (`rank_bm25` over pre-analyzed text) | Industry-style "stemmed BM25" morphology reference | **New, cheap** |
| v1 / v5 / v6 pipelines | Measurement controls | **Untouched — no backport** |

Rules: add named conditions, never mutate a frozen baseline. Analyzed scores live in a different avgdl/DF space and are reported as separate conditions, never mixed with legacy numbers. **V7 targets are measured against the analyzed-parity baseline, not the legacy one.**

### 4.2 Benchmark Targets

| Benchmark Corpus | Chunks / Docs | Queries | Primary Baseline to Beat | Target V7 Metric |
| :--- | :---: | :---: | :--- | :--- |
| **`fused_stress_500`** | 17,241 / 500 | 1,084 | Analyzed-parity BM25 (re-derive) | **$\ge$ analyzed-parity + margin** |
| **`enterpriserag_doc_level`** | 1,722 full docs | 500 | Analyzed-parity BM25 (re-derive) | **$\ge$ analyzed-parity + margin** |
| **`liverag_doc_level`** | 970 full docs | 895 | Analyzed-parity BM25 (re-derive) | **$\ge$ analyzed-parity + margin** |
| **BEIR Scientific / Financial Sets** | Variable | Variable | Analyzed-parity BM25 / Dense BGE | Close recall gap on hard semantic corpora |

> **Note:** the historical targets in the previous revision (`78.3%`/`83.9%`/`93.9%`) referenced the *legacy* BM25. They must be re-derived against `AnalyzedLuceneBM25` before V7 is held to them — the analyzed baseline is materially stronger, and the p-sweep already shows it beating v1/v5/v6 on fiqa/nfcorpus/scifact/bright_economics.

### 4.3 Morphology Evaluation Gate (cross-ref `morphology_expansion_strategy.md` §7)

1. **Diagnostic census first:** measure the % of gold-miss queries failing *only* on suppletion — inflection is already handled by index-time KStem.
2. **Technical-token regression guard:** `qwen2.5`/`gpt-4`/`fp16`/`zero-shot`/`kv-cache` query set must stay bit-identical (exemption set guarantees; verify anyway).
3. **A/B sweep axes:** suppletion override {on, off} × cross-root probing {on, off}. The `η_morph` / fold-in / pool-stem-dedup axes are removed.
4. **Metric pairing:** report **ChunkRec@10 and Strict@10 together** — expansion buys recall, risks precision; Strict@10 is the guardrail.
5. **Per-stage attribution:** isolate analyzer effect (index-time KStem) vs posting-index effect vs cross-root synonym effect so regressions are traceable.

---

## 5. File Modification Plan

1. **[`src/pipeline_v2/indexer/tokenizer.py`](src/pipeline_v2/indexer/tokenizer.py)** *(already created):* `EdgeRAGTokenizer` — Phase 1 canonical tokenization.
2. **[`src/pipeline_v2/indexer/analyzer.py`](src/pipeline_v2/indexer/analyzer.py)** *(already created — extend):* add the WordNet `stemmer_override` (suppletion) stage before KStem.
3. **[`src/pipeline_v2/indexer/posting_index.py`](src/pipeline_v2/indexer/posting_index.py)** *(already created):* compact posting lists + weighted top-K retrieval (Phase 5).
4. **[`src/pipeline_v2/indexer/corpus_vocab_builder.py`](src/pipeline_v2/indexer/corpus_vocab_builder.py):** analyzer-parity wiring (build pool from analyzed tokens); singleton rescue.
5. **[`src/pipeline_v2/indexer/dense_vocab_matrix.py`](src/pipeline_v2/indexer/dense_vocab_matrix.py):** Phase 1 & 3 batch tensor projection $\mathbf{E}_A \cdot \mathbf{V}^\top$.
6. **[`src/pipeline_v2/expansion/bm25_dense_aspect_extractor.py`](src/pipeline_v2/expansion/bm25_dense_aspect_extractor.py):** Phase 2 anchor weighting + stem dedup; Phase 3 adaptive gating; Phase 4 single-tier IT-MPE compilation.
7. **[`src/pipeline_v2/indexer/bm25_lucene_indexer.py`](src/pipeline_v2/indexer/bm25_lucene_indexer.py):** Phase 5 `retrieve_weighted` over the posting index; `mode: legacy | parity` switch.
8. **[`src/pipeline_v2/indexer/corpus_idf_registry.py`](src/pipeline_v2/indexer/corpus_idf_registry.py):** consume analyzed DF tables (single source of truth).
9. **[`configs/pipeline_v2.yaml`](configs/pipeline_v2.yaml):** authoritative single-source-of-truth configuration for 5-phase parameters (`stemmer: kstem`, `use_wordnet_override: true`, `indexer.mode`). **Drop** `eta_morph`, `k_morph`, `morph_fold_synonyms`.

*Removed from the plan:* `morphological_stem_registry.py` (redundant under index-time stemming).
