# 🧱 LuceneBM25Baseline Completion Plan (Lucene-Parity Modernization)

## 1. Executive Summary

`LuceneBM25Baseline` ([`src/baselines/bm25.py`](src/baselines/bm25.py)) is currently **a Lucene-*formula* port over naive tokens** — Lucene's IDF and `k1/b` defaults, but with plain `lower().split()` tokenization, a full O(|Q|·N) score scan, and no analyzer chain. The morphology strategy (see [morphology_expansion_strategy.md](morphology_expansion_strategy.md), Rev 2) depends on exactly the machinery this baseline lacks (KStem, exemptions, weighted retrieval).

This plan upgrades the baseline to **Lucene-parity behavior while staying pure-Python** (no JVM/PyLucene, no Elasticsearch), in 5 reversible stages, each independently evaluable:

| Stage | Scope | Target Phase (V7 plan) |
| :---: | :--- | :--- |
| 0 | Freeze legacy baseline as eval control; add golden fixtures | — |
| 1 | Analyzer chain: `EdgeRAGTokenizer` + possessive + stopword + KStem + exemptions | Phase 1 |
| 2 | True posting-list index + weighted retrieval + Lucene query semantics | Phase 5 |
| 3 | Morphology fold-in integration (Tier 1/2 of morphology doc) | Phase 2–4 |
| 4 | Optional parity extras: positions/phrase queries, multi-field, explain | Post-V7 |

> **Status:** planning document only — **no code is modified by this plan**; implementation starts after design sign-off.

---

## 2. Gap Inventory (from the Code Audit)

| # | Gap | Severity | Effect | Fixed by |
| :---: | :--- | :---: | :--- | :---: |
| A1 | No punctuation stripping — `"pricing,"` ≠ `"pricing"` | 🔴 High | Vocabulary explosion; DF/IDF split across surface variants; query misses | Stage 1 |
| A2 | No possessive filter — `"model's"` ≠ `"model"` | 🟠 Medium | Missed matches on possessives | Stage 1 |
| A3 | No stopword removal | 🟠 Medium | Inflated `avgdl` distorts length norm for all docs; noise matches; wasted compute | Stage 1 |
| A4 | No stemming (KStem) | 🟠 Medium | Inflectional blind spot (covered in morphology doc) | Stage 1 |
| B5 | Query-term repetition scales score linearly | 🔴 High | Non-Lucene semantics the pipeline currently exploits ("discrete repetition") | Stage 2 |
| B6 | Unseen query terms skipped without IDF computation | 🟡 Low | IDF registry inconsistent for OOV gating | Stage 2 |
| B7 | No overlap discounting for colliding terms | 🟡 Low | Double-counting once fold-in aliases collide | Stage 2 |
| C8 | No inverted index — O(\|Q\|·N) full scan per query | 🟠 Medium | Latency ceiling; blocks Phase 5 design | Stage 2 |
| C9 | No term positions | 🟡 Low | No phrase queries (Phase 2 quoted entities have nothing to match) | Stage 4 |
| C10 | Per-doc `Counter` dicts instead of posting arrays | 🟡 Low | Memory overhead | Stage 2 |
| D | No multi-field / boosts / `explain()` | ⚪ Optional | Not needed for single-field chunk retrieval | Stage 4 |
| E | Naming implies real Lucene | 🟡 Low | Misleading documentation | All stages |

---

## 3. Design Decisions

### 3.1 Stay pure-Python (no PyLucene, no external search engine)

- Consistent with V7 budgets ($<0.09\text{ GB}$ VRAM, $<15\text{ms}$ CPU, $<0.3\text{s}$ TTI).
- PyLucene adds a JVM dependency and Windows packaging pain; OpenSearch/Elasticsearch externalizes the index — both are architectural changes belonging to the §4.4 dual-field decision in the morphology doc (V8 candidate), not this plan.
- Cost: we implement the analyzer chain and posting lists ourselves (~200–300 lines total), mirroring Lucene semantics.

### 3.2 Analyzer chain (index-time and query-time, single source of truth)

```
EdgeRAGTokenizer ──> PossessiveFilter ──> StopwordFilter ──> KStemFilter (with exemptions) ──> Lowercase
```

1. **`EdgeRAGTokenizer`** — the canonical tokenizer from the V7 plan (Upgrade 1.1): `r'\b[a-z0-9]+(?:[-._][a-z0-9]+)+\b|\b[a-z0-9]{2,}\b'`. Strips delimiter punctuation *without* breaking technical compounds (`qwen2.5`, `gpt-4`, `fp16`). Unifies the three divergent tokenizations currently in the repo (indexer `.split()`, IDF-registry translate+split, vocab-builder regex+split).
2. **PossessiveFilter** — strips trailing `'s` (Lucene `EnglishPossessiveFilter` semantics); runs *before* the technical-token exemption so `qwen2.5's` still resolves to `qwen2.5`.
3. **StopwordFilter** — embed Lucene's standard English stop set (~30 terms) as a static constant; no dependency. Configurable via `configs/pipeline_v2.yaml`.
4. **KStemFilter** — `krovetzstemmer` (pure-Python KStem) as primary; NLTK Snowball English as fallback. Raw Porter rejected (see morphology doc §4.1).
5. **Exemption set (keyword_marker semantics)** — tokens matching `[a-z0-9]+(?:[-._][a-z0-9]+)+`, `[A-Z]{2,}`, or containing digits are never stemmed and never stopword-dropped. Built automatically at index time.

### 3.3 Posting-list index (replaces the O(|Q|·N) scan)

```text
term → PostingList { df, total_tf, doc_ids: array('I'), tfs: array('I'), positions?: array('I') }
```

- Compact `array`-typed posting arrays (C10); positions stored only when `store_positions = true` (C9, Stage 4).
- Query-time traversal costs ∝ matched postings, per Lucene; accumulator is a pre-allocated `float32` array; top-K via min-heap (this is precisely the V7 Phase 5 design — "Vectorized Inverted Posting Retrieval").

### 3.4 Scoring semantics (full Lucene parity)

Weighted scoring — the V7 Phase 5 formula, with boosts replacing repetition:

$$\text{Score}(D, Q) = \sum_{t \in \vec{w}_Q} w_Q(t) \cdot \text{IDF}(t) \cdot \frac{\text{TF}(t, D) \cdot (k_1 + 1)}{\text{TF}(t, D) + k_1 \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

- **B5 fix:** `retrieve_weighted(term_weights: Dict[str, float], top_k)` consumes $\vec{w}_Q$ directly; the legacy `retrieve(tokenized_query)` repetition path is retired for the pipeline (call sites migrate in the same change — the "Retirement of Discrete Repetition" from the V7 plan).
- **B6 fix:** OOV query terms get IDF from the formula with `docFreq = 0` (`ln(1 + N + 0.5 / 0.5)`), so the IDF registry stays consistent with gating logic.
- **B7 fix:** when two query entries alias to the same posting term (fold-in collision), weights are summed once — Lucene's boost-summing semantics — instead of double traversal.

### 3.5 Naming

- Legacy class stays `LuceneBM25Baseline`, frozen as the eval control.
- New indexer: `AnalyzedLuceneBM25` (working name) — the docs will describe it as "Lucene-parity BM25 (pure-Python)" to fix the E gap.

---

## 4. Staged Implementation

### Stage 0 — Freeze & Fixtures (no behavior change)
- **Changes:** add golden fixtures: a fixed corpus + query set with hand-computed scores for the legacy baseline; snapshot of current results on `fused_stress_500` (recall@10/strict@10 + latency) stored as regression references.
- **Acceptance:** legacy path reproduces snapshots bit-identically after every later stage.

### Stage 1 — Analyzer Chain (Phase 1)
- **Changes:** new `tokenizer.py` (`EdgeRAGTokenizer`), new `analyzer.py` (chain from §3.2); index-time analysis only; scoring engine still the legacy scan so the stage isolates the analysis effect.
- **Acceptance:**
  - Vocabulary size shrinks; no token containing delimiters survives except protected technical patterns (assert on corpus sample).
  - Technical-token regression guard: queries containing `qwen2.5`, `gpt-4`, `fp16`, `zero-shot` produce identical *matched postings* to Stage 0 (bit-identical on a fixed query set).
  - `avgdl`/DF distributions reported for review; stopword list configurable.

### Stage 2 — Posting Index & Weighted Retrieval (Phase 5)
- **Changes:** new `posting_index.py` (PostingList + InvertedPostingIndex + min-heap top-K); `BM25LuceneIndexer.retrieve_weighted()` consuming $\vec{w}_Q$; OOV-IDF and overlap-summing per §3.4; orchestrator/extractor call sites migrate from token repetition to weights (coupled change — same PR).
- **Acceptance:**
  - **Score parity:** Stage 2 scores equal Stage 1 scores to $10^{-6}$ on all fixtures (only the data structure changed).
  - Latency on 17k chunks: $<15\text{ms}$ (V7 Phase 5 target); memory footprint reported vs legacy.
  - Repetition-based `retrieve()` retained only for the frozen baseline; new pipeline path uses `retrieve_weighted`.

### Stage 3 — Morphology Integration (Phase 2–4)
- **Changes:** `MorphologicalStemRegistry` + Tier-1 fold-in + Tier-2 stem-diversity gate + synonym closure, exactly per the morphology doc (Rev 2, §4).
- **Acceptance:** the morphology doc's §7 evaluation gate (diagnostic census → regression guard → A/B sweep with `morph_fold_synonyms`/`eta_morph` axes).

### Stage 4 — Optional Parity Extras (post-V7, gated)
- Positions + quoted-phrase matching for Phase 2's `"..."` entities; multi-field support; `explain()`-style score breakdown; index serialization. Each is behind its own flag; none is required for V7 targets.

---

## 5. Compatibility & Rollout

| Rule | Decision |
| :--- | :--- |
| Legacy baseline | **Frozen** — `LuceneBM25Baseline` unchanged; remains the eval control (per baseline-integrity decision) |
| v1 / v5 / v6 pipelines | **Untouched** — no backport; they stay measurement controls |
| New mode selection | `configs/pipeline_v2.yaml`: `indexer: {mode: legacy \| parity}`; rollback = one flag flip |
| Analyzer config | `analyzer: {stemmer: kstem \| snowball \| none, use_stopwords: true, store_positions: false, exemptions: auto}` |
| Score comparability | Analyzed scores are **not comparable** to legacy scores (different avgdl/DF space) — evaluated as separate conditions, never mixed |

---

## 6. Evaluation Protocol (per stage)

1. **Regression guards:** Stage 0 snapshots must reproduce bit-identically; technical-token query set must be unchanged through Stages 1–3.
2. **Score-parity test:** Stage 1 → 2 transition must not move any score beyond $10^{-6}$.
3. **Metric pairing:** ChunkRec@10 **and** Strict@10 together on `fused_stress_500` / `enterpriserag` / `liverag` — analyzer changes buy recall and risk precision; Strict@10 is the guardrail.
4. **Isolated attribution:** report each stage's delta separately (analyzer effect vs posting-index effect vs morphology effect) so regressions are attributable.
5. **Latency & memory:** per-stage TTI, per-query latency ($<15\text{ms}$), and index footprint ($<0.09\text{ GB}$ VRAM budget) tracked in the same harness.

---

## 7. File Modification Plan (proposed — not executed)

| File | Change |
| :--- | :--- |
| [`src/baselines/bm25.py`](src/baselines/bm25.py) | **Frozen** (control); optional thin `AnalyzedLuceneBM25` wrapper for parity evals |
| `src/pipeline_v2/indexer/tokenizer.py` *(new)* | `EdgeRAGTokenizer` (V7 Upgrade 1.1) |
| `src/pipeline_v2/indexer/analyzer.py` *(new)* | Possessive + stopword + KStem + exemption chain |
| `src/pipeline_v2/indexer/posting_index.py` *(new)* | `PostingList`, `InvertedPostingIndex`, weighted top-K retrieval |
| [`src/pipeline_v2/indexer/bm25_lucene_indexer.py`](src/pipeline_v2/indexer/bm25_lucene_indexer.py) | Consume analyzer + posting index; add `retrieve_weighted`; `mode` switch |
| [`src/pipeline_v2/indexer/corpus_idf_registry.py`](src/pipeline_v2/indexer/corpus_idf_registry.py) | Consume analyzed `df` tables (single source of truth) |
| `configs/pipeline_v2.yaml` | `indexer.mode`, `analyzer.*` keys |
| `tests/` *(new fixtures)* | Golden scores, regression snapshots, score-parity, technical-token guard |

---

## 8. Risks & Mitigations

| Risk | Mitigation |
| :--- | :--- |
| Stopword removal shifts ranking unpredictably | Lucene's standard stop set as default; `use_stopwords` flag; A/B in Stage 1 |
| `avgdl` change makes cross-version score comparison invalid | Modes evaluated as separate conditions; never mixed (§5) |
| `krovetzstemmer` unavailable in target env | Pure-Python package; NLTK Snowball fallback; `stemmer: none` escape hatch |
| Position storage doubles memory | `store_positions: false` default; enabled only in Stage 4 |
| Call-site coupling (B5 retirement) breaks pipelines mid-migration | Orchestrator + extractor migrate in the same change as Stage 2; legacy `retrieve()` kept for frozen baseline |
| Exemption set misses an edge-case technical token | Exemption patterns are the same regexes as Phase 2 heuristic entities; regression guard asserts bit-identical technical-token results |

---

## 9. Explicitly Out of Scope

- Real PyLucene/JVM integration; Elasticsearch/OpenSearch as external index (V8 dual-field decision, morphology doc §4.4).
- Index serialization / incremental indexing / concurrency.
- Backporting any of this to v1/v5/v6 pipelines.
- SPLADE-style learned expansion and sentence-contextualized embeddings (V7b/V8 per architectural plan).
