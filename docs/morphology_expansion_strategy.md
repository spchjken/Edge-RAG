# 🔬 Morphology Resolution Strategy in Anchored Lexical-Semantic Retrieval

## 1. Executive Summary

In high-speed hybrid retrieval, the classic dilemma is that an **unstemmed exact-token index** cannot match morphological variants, while a **dense probing channel** wastes its small expansion budget on inflectional clones. This revision (**Rev 3**) retires that dilemma by changing the premise: **Edge-RAG now indexes with an index-time-stemmed (KStem) analyzer** instead of an unstemmed exact-token index.

Rev 3 does three things:

1. **Cancels the two-tier fold-in machinery.** Rev 2 introduced Tier-1 morphological fold-in plus a `MorphologicalStemRegistry` to patch the "lexical blind spot" of an *unstemmed* index. With index-time KStem — already implemented in `EdgeRAGAnalyzer` and proven in `AnalyzedLuceneBM25` — `price`/`prices`/`pricing` already share one posting list keyed by the stem, so there is nothing left to fold in. Tier-1 fold-in, the stem-diversity gate, and the stem-diversified vocabulary pool are all **cancelled**.
2. **Reduces morphology to two residual concerns.** (a) **Suppletion** (`went → go`, `bought → buy`, `better → good`), which KStem does not handle — resolved by a small WordNet `stemmer_override` stage in the analyzer, applied identically at index and query time. (b) **Cross-root semantic synonyms** (`price`/`cost`/`valuation`), which no stemmer can bridge — resolved by the (now single-tier) dense probing channel.
3. **Re-scopes the "exact-first" principle.** For ordinary words, inflectional surface form is *normalization noise*, not a relevance signal, so conflation is correct and desirable. Exactness matters only for **technical tokens** (`qwen2.5`, `gpt-4`, `fp16`), and that protection is already guaranteed by the analyzer's **exemption set** — not by a dual-field index.

---

## 2. Problem Restatement

The original strategy (Rev 2) was written against a codebase where `BM25LuceneIndexer` wrapped a naive `lower().split()` index — no analyzer, no stemmer, postings keyed by exact surface token. That status quo is now false: [`lucene_bm25_parity_plan.md`](lucene_bm25_parity_plan.md) Stages 1–2 are complete, and `AnalyzedLuceneBM25` stems at both index and query time.

### 2.1 What index-time KStem already solves

* **The lexical blind spot (Rev 2 "Challenge 1").** A gold document containing `"…pricing…"` used to score zero against the query `"price"`. Under index-time KStem, `pricing` is analyzed to the stem `price` at index time, so it lives in the same posting list as `price` and `prices`; the query `"price"` now matches it. **Resolved.**
* **The inflectional slot-starvation (Rev 2 "Challenge 2").** Dense probing used to return `["prices", "pricing"]` and starve `["cost", "valuation"]` out of the $C_{\text{exp}} = 2$ slots. When the vocabulary pool is itself built from analyzed (stemmed) tokens, those inflectional clones never enter the pool — the top-$C_{\text{exp}}$ candidates are cross-root by construction. **Resolved** (conditional on analyzer-parity wiring, §4.5).

### 2.2 Residual gap 1 — suppletion

KStem is a light inflectional stemmer (`-s/-es/-ed/-ing` plus a dictionary of regular irregulars). It does **not** handle suppletive forms: `went → go`, `bought → buy`, `better → good`. These are the only remaining *morphological* mismatches, and they are rare but real on technical corpora.

### 2.3 Residual gap 2 — cross-root semantic bridging

No stemmer can bridge distinct roots that share meaning (`price`/`cost`/`valuation`, `car`/`automobile`). This is the domain of the dense probing channel, and it is now the **primary value-add** of Edge-RAG over the analyzed baseline.

### 2.4 Codebase status

| Component | Status | State |
| :--- | :---: | :--- |
| `EdgeRAGTokenizer` | ✅ DONE | Canonical tokenizer; technical compounds + 2-letter acronyms preserved. |
| `EdgeRAGAnalyzer` | ✅ DONE (extend) | 5-stage chain: tokenizer → possessive → stopword → exemption → KStem. **Needs the `stemmer_override` suppletion stage added.** |
| `InvertedPostingIndex` | ✅ DONE | Compact posting arrays + vectorized `retrieve_weighted`. |
| `AnalyzedLuceneBM25` | ✅ DONE | Analyzed baseline; beats all v1/v5/v6 schemas on doc-level benchmarks. |
| Suppletion override (WordNet `wn_s.pl` / `wn_v.pl`) | ⏳ PENDING | Map suppletive forms to lemma inside the analyzer. |
| Analyzer-parity wiring (VocabBuilder / IDF registry / extractor) | ⏳ PENDING | Ensure every producer of $\vec{w}_Q$ keys emits analyzed stems (§4.5). |
| `MorphologicalStemRegistry` | ❌ CANCELLED | Redundant under index-time stemming. |
| Tier-1 fold-in + synonym closure | ❌ CANCELLED | Redundant under index-time stemming. |
| Stem-diversity gate | ❌ CANCELLED | Pool is already stem-diverse. |
| Anchor dedup (destructive prefix/suffix heuristic) | ⏳ PENDING (simplify) | Replace with stem-equality (§4.2). |

---

## 3. Comparative Paradigm Analysis

| Retrieval Architecture | Morphology Mechanism | Semantic Expansion Mechanism | Trade-off / Limitation |
| :--- | :--- | :--- | :--- |
| **BM25 + Porter (Elasticsearch/Solr `english` analyzer default)** | Stemming at **both index and query time** via a standard analyzer chain; `keyword_marker` / `stemmer_override` filters exempt protected terms | ❌ None (analyzer-level `synonym` filter optional) | 20+ years in production; raw Porter over-stems (`organization → organ`); use `kstem` or exclusions. |
| **Anserini / Pyserini BM25 (standard BEIR baselines)** | Lucene `EnglishAnalyzer` (Porter) by default, with an optional **Krovetz/KStem** mode | ❌ None | Published "BM25" on BEIR *includes* stemming; KStem is the light-stemming IR alternative. |
| **KStem / Krovetz** | Dictionary-augmented light stemming (`-s/-es/-ed/-ing` + irregular lookup) | ❌ None | Best precision/recall balance for weakly-inflected English; still misses suppletion (`went → go`). |
| **WordNet lemmatizer + exception lists** | POS-aware lemmatization; `wn_s.pl` / `wn_v.pl` map suppletive forms exactly (`went → go`, `better → good`) | ❌ None | Ideal as a small *auxiliary* override dict, not a replacement for a stemmer. |
| **Algolia `ignorePlurals`** | Dictionary + algorithmic hybrid, applied at index and query time | ❌ None | Confirms conflation is the norm for search; exactness reserved for protected tokens. |
| **Dense Bi-Encoders (BGE / Contriever)** | Continuous vector space handles inflections implicitly | Embeds entire sentence into latent space | High latency, query drift, large index footprint (out of budget). |
| **SPLADE-v3 (Sparse Neural MLM)** | WordPiece subwords (`research` + `##ing`) | 30k MLM logits activate 50–150 terms | Correct by construction, but extreme compute (out of budget). |
| **Edge-RAG V7 Rev 3 (This Document)** | **Index-time KStem** (analyzer) + **WordNet suppletion override** + technical-token exemptions | **Single-tier cross-root dense probing** | Inflectional conflation at the analysis layer; only suppletion + cross-root synonyms remain. |

### 3.1 Consensus lessons (re-scoped)

1. **Morphology belongs to the token-analysis layer, decoupled from semantic/synonym expansion.** Rev 2 reached this conclusion and kept re-implementing it at query time; Rev 3 simply *uses* the analyzer that now exists.
2. **Stemming is safe only with a protection list.** `keyword_marker`/`stemmer_override` semantics — technical tokens are exempted, never conflated.
3. **Inflectional conflation is normalization, not ranking.** For ordinary words, surface form (`price` vs `prices`) carries no relevance signal; conflating them is correct. Exactness matters only for technical tokens, and that is handled by the exemption set. (Rev 2's "exact matches must outrank variant matches" is **retired**.)
4. **Rules cannot cover suppletion; a small dictionary can.** `went → go` needs WordNet exception lists, not a better stemmer.

---

## 4. The Rev 3 Design: Single-Tier Morphology

```mermaid
graph TD
    classDef input fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px,color:#000000;
    classDef analyzer fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000000;
    classDef probing fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000000;
    classDef output fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000000;

    Q[Query 'price']:::input --> AN["EdgeRAGAnalyzer (index & query)<br/>Tokenizer → Possessive → Stopword →<br/>Exemption → StemmerOverride → KStem"]:::analyzer
    AN --> A[Anchor 'price' (stemmed)]
    A --> T2["Cross-root dense probing<br/>(no stem-diversity gate needed)"]:::probing
    T2 --> SYN["Top-2 synonyms w(s|a)<br/>('cost', 'valuation')"]:::probing
    A --> VEC["Continuous sparse vector w_Q<br/>(anchor + budgeted synonyms)"]:::output
    SYN --> VEC
```

### 4.0 Design principles

1. **Morphology is an analyzer concern**, applied identically at index and query time.
2. **Protection by construction** — technical tokens are exempt from stemming and override.
3. **Semantic expansion is budgeted** (IT-MPE) so no synonym can out-score its anchor.

### 4.1 Morphology at index time (KStem + suppletion override)

* Index and query both run through `EdgeRAGAnalyzer`:
  `Tokenizer → PossessiveFilter → StopwordFilter → KeywordMarkerExemption → StemmerOverride → KStem`.
* **StemmerOverride** (new, small): a static dict from WordNet `wn_s.pl` / `wn_v.pl` mapping suppletive forms to lemma (`went → go`, `bought → buy`, `better → good`, `best → good`). Applied before KStem; exempt tokens are skipped.
* **Exemption set** (unchanged): tokens matching the technical patterns `[a-z0-9]+(?:[-._][a-z0-9]+)+`, `[A-Z]{2,}`, or containing digits are never stemmed and never overridden.
* **Result:** postings, IDF tables, the vocabulary pool, and query anchors all share one analyzed-token space — the "single source of truth" from V7 Upgrade 1.7 (analyzer parity), built on top of the tokenizer (Upgrade 1.1).

### 4.2 Cross-root semantic probing (the surviving channel)

* Probing is unchanged in mechanism (batch $\mathbf{E}_A \cdot \mathbf{V}^\top$, Dual-Sim with $\beta = 0.65$, adaptive $\tau_{\text{sim}}$), but the **stem-diversity gate is removed** — the pool is already stem-diverse, so the top-$C_{\text{exp}}$ candidates are cross-root by construction.
* **Anchor dedup:** replace the destructive prefix/suffix substring heuristic with **stem-equality** (trivial now that tokens are analyzed) plus the existing cosine check ($\ge 0.90$).

### 4.3 IT-MPE budget (single-tier)

Only cross-root synonyms consume the expansion budget. The Rev 2 `η_morph` / `μ_morph` tier-split is removed:

$$w(s \mid a) = \mu(Q) \cdot w(a) \cdot \min\left(1.0, \frac{\text{IDF}(a)}{\text{IDF}(s)}\right) \cdot p(s \mid a)$$

$$\sum_{s \in S(a)} w(s \mid a) \le \mu(Q) \cdot w(a), \qquad \mu(Q) \in [0.18, 0.35]$$

This is the original Tier-2 formula with the fold-in and synonym-closure terms dropped; the second-order mass term is gone, so $\mu_{\text{eff}} = \mu(Q)$ exactly.

### 4.4 Decision record: no dual-field "exact-first" index

* Rev 2 catalogued a dual-field raw+stemmed index (exact-first ranking) as the "stronger V8 design." **Rev 3 rejects this.** For ordinary words, exact-first ranks on surface-form noise; the correct behavior is conflation. Exactness for technical tokens is already guaranteed by the exemption set.
* The single-field stemmed index is therefore the **final** V7 design, not a compromise.

### 4.5 Analyzer-parity wiring (the real integration work)

A stemmed index only works if every producer of $\vec{w}_Q$ keys emits the same analyzed stems the postings are keyed on. `CorpusVocabBuilder`, `CorpusIDFRegistry`, and `BM25DenseAspectExtractor` must all consume `EdgeRAGAnalyzer`, not raw tokens — otherwise injected synonyms and anchors will miss postings. This is the concrete carry-over of V7 Upgrade 1.7 (analyzer parity), built on top of the tokenizer (Upgrade 1.1).

---

## 5. Worked example

### Query: `"fiscal policy impact on price levels"` — anchor `"price"`

1. **Anchor primary weight:** $w(\text{"price"}) = 4.2$ (high-IDF topic anchor).
2. **Query-level budget:** $\mu(Q) = 0.35$.
3. **Dense probing** (pool is already stem-diverse, so inflections are absent): raw BGE list yields `cost` (sim 0.84) and `valuation` (sim 0.81).
4. **Softmax** ($\tau = 0.10$): $p(\text{cost}) \approx 0.57$, $p(\text{valuation}) \approx 0.43$.
5. **Weights:** $w(\text{cost}) = 0.35 \cdot 4.2 \cdot 1.0 \cdot 0.57 \approx 0.84$; $w(\text{valuation}) \approx 0.63$.
6. **Invariant check:** $0.84 + 0.63 = 1.47 = \mu(Q) \cdot w(a) = 0.35 \cdot 4.2$ ✅
7. **Final vector for `"price"`:** `{"price": 4.2, "cost": 0.84, "valuation": 0.63}`.

### Suppletion trace: `"prices went up"`

* The analyzer overrides `went → go` at query time; at index time, documents containing `went`/`go`/`goes`/`going`/`gone` all land in the `go` posting list. No fold-in machinery is needed — the override is part of the shared analyzer chain.

### Exemption trace: `"qwen2.5 context length"`

* `qwen2.5` matches the technical pattern → never stemmed, never overridden. Behavior is bit-identical to today's pipeline.

---

## 6. Latency guarantees

* The analyzer is a single pass over tokens; the suppletion override is an `O(1)` dict lookup per token.
* Dense probing is unchanged (batch GEMM).
* Rev 2's registry build, registry lookup, and fold-in are gone, so query-time work is strictly *less* than Rev 2.

---

## 7. Evaluation Protocol (Gate Before Adoption)

1. **Diagnostic first — suppletion census:** on `fused_stress_500`, `enterpriserag`, and `liverag`, measure the % of gold-miss queries failing *only* on suppletion. This sizes the remaining morphological ceiling; if negligible, the override is optional.
2. **Technical-token regression guard:** the fixed set `qwen2.5`, `gpt-4`, `fp16`, `zero-shot`, `kv-cache` must stay bit-identical (guaranteed by the exemption set — verify anyway).
3. **A/B sweep:** suppletion override {on, off} × cross-root probing {on, off}; the `η_morph` axis is removed.
4. **Metric pairing:** report ChunkRec@10 **and** Strict@10 together; Strict@10 remains the precision guardrail.
5. **Starvation telemetry:** assert post-probing synonyms are cross-root (trivially true once the pool is stemmed).

---

## 8. Implementation Roadmap & Progress Tracker

* [x] **Step 1: Canonical Tokenizer (`EdgeRAGTokenizer`)** — **DONE**
* [x] **Foundation: Analyzer, Posting Index, Baseline** (`analyzer.py`, `posting_index.py`, `AnalyzedLuceneBM25`) — **DONE**
  - 5-stage analyzer (possessive → stopword → exemption → KStem), vectorized posting index, analyzed baseline beating all v1/v5/v6 schemas.
* [ ] **Step 2 (REVISED): Suppletion override in `EdgeRAGAnalyzer`** — **PENDING**
  - Add a `stemmer_override` stage from WordNet `wn_s.pl` / `wn_v.pl` before KStem; exempt technical tokens.
* [ ] **Step 3 (REVISED): Analyzer-parity wiring** — **PENDING**
  - Route `CorpusVocabBuilder`, `CorpusIDFRegistry`, and `BM25DenseAspectExtractor` through `EdgeRAGAnalyzer` so all $\vec{w}_Q$ keys are analyzed stems.
* [ ] **Step 4 (REVISED): Replace destructive anchor dedup** — **PENDING**
  - Replace the prefix/suffix substring heuristic with stem-equality + cosine check.
* [ ] ~~**Step 5: Tier-1 fold-in / synonym closure**~~ — **CANCELLED** (redundant under index-time stemming)
* [ ] **Step 6: Configuration YAML updates** — **PENDING**
  - Expose `stemmer: kstem`, `use_wordnet_override: true`; **drop** `eta_morph`, `k_morph`, `morph_fold_synonyms`.
* [ ] **Step 7: Evaluation diagnostics & sweeps** — **PENDING**
  - Run suppletion census, technical-token regression checks, and cross-root synonym recovery sweeps on `bright_economics` and `financebench`.

**Explicitly out of scope for V7:** dual-field raw+stemmed "exact-first" index (rejected, §4.4), SPLADE-style subword postings, sentence-contextualized embeddings (V7b/V8).
