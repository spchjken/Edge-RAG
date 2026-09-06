# Edge-RAG Expansion Platform — Research & Refactor Plan

> Status: consolidated plan (draft). Supersedes the "Edge-RAG as a single V7 retriever" framing;
> positions Edge-RAG as a **general platform for zero-shot query expansion over BM25**.
> This document packs up the positioning decision, the simplification target, the expansion-source
> taxonomy, the expansion-mass fix, and the large-corpus latency blocker.

---

## 1. Positioning: V7 is *not* comparable to learned methods

V7 (BM25 + frozen zero-shot expansion) **cannot and should not** be compared with learned methods
(SPLADE, uniCOIL, DeepImpact, SPAR) as a target.

**⇒ The comparison family is "unsupervised / zero-shot expansion over BM25" only.** V7's fair
opponents are methods that also *start from BM25, no fine-tuning, no cross-encoder distillation*.
Learned-sparse models replace BM25's term weighting with a supervised function — a different
category (different training regime, different signal). Any measured gap is therefore confounded by
"supervised vs zero-shot," not by "better expansion."

**⇒ Learned methods appear exactly once, as a context line:** "zero-shot expansion closes X% of the
supervised gap at 0% of the training cost." They are a ruler, never the aim.

---

## 2. Simplify, and the full baseline list

### 2.1 Simplify the method (keep the load-bearing core, cut the knob sprawl)

| Keep (load-bearing) | Cut / shelve (legacy knobs) |
|---|---|
| IT-MPE: `μ` mass ceiling + `min(1, IDF(a)/IDF(s))` damping + normalized allocation | `beta` (query-context blend) |
| One gate threshold `τ` | `delta_tau` (adaptive gate slope) |
| One mass budget `μ` | `gate_variant` (two_gate / soft_reweight) |
| Source-agnostic input `(term, score, provenance)` | `allocation` variants (uniform / softmax) |
| | POS ratios → fold into a single anchor-salience weight |

Target surface: **3 decisions only** — (1) which source + its score, (2) one gate, (3) one mass budget.

### 2.2 Simplify the theory

Reduce to a single claim: *"for any source, any support size K, expansion mass in score space
≤ μ·anchor mass."* Everything else (capacity policy, gating, source choice) is operational, not
theorem. The retired "saliency-proportional optimality" stays dead.

### 2.3 Full baseline list

**Zero-shot / unsupervised (the aim):**

1. BM25 blank (unstemmed) — floor
2. BM25 analyzed (kstem+wordnet) — floor, the actual base
3. **BM25 + RM3** (classic PRF) — the champion to beat
4. BM25 + RM1/RM2 / LCA (PRF variants)
5. BM25 + WordNet (lexical)
6. BM25 + PMI / co-occurrence (statistical)
7. BM25 + dense **anchor-level** cosine (current V7)
8. BM25 + dense **query-level** cosine
9. BM25 + **centroid/cluster** expansion (PLAID-PRF-inspired)
10. BM25 + **LLM-grounded** expansion (GAR/HyDE/Query2Doc, vocab-projected)

**Reference / context lines (NOT the aim):**

- SPLADE-v3, uniCOIL, DeepImpact, SPAR (learned sparse)
- Dense BGE (single-vector)
- ColBERT / PLAID (late interaction)
- BM25 + PLAID (sparse-first + dense rerank)

---

## 3. Expansion source ideas (the `ExpansionSource` platform)

| # | Source | Signal | Cost | Notes |
|---|---|---|---|---|
| 1 | **Classic PRF (RM3)** | retrieval-derived (top-k doc terms) | +1 retrieval round | the champion; corpus-adaptive |
| 2 | **LLM expansion** | generative, vocab-grounded | high (model call) | must project onto corpus vocab → zero-hallucination |
| 3 | **Dense anchor-level cosine** | per-anchor → vocab projection | low (GPU GEMM) | **current V7** |
| 4 | **Dense query-level cosine** | whole-query embedding → vocab | low | retired "Dual-Sim" (`β<1`) revived as a *separate source* |
| 5 | **Dense centroid expansion** | cluster vocab embeddings → prototype terms | low (one k-means) | PLAID-PRF idea, no late-interaction index |
| 6 | **WordNet / lexical** | synsets/hypernyms/aliases | near-zero (lookup) | poor technical coverage |
| 7 | **PMI / co-occurrence** | corpus window statistics | low (precompute) | distributional, domain-aware |
| 8 | **Alias / entity** | acronyms, compounds, KB aliases | low | keep separate budget (exempt from length-damping) |

All sources emit `(term, score, provenance)` → one gate → one IT-MPE allocator. That is the platform.

---

## 4. Expansion-mass problem: length-proportional is backwards

**Problem:** current `μ(Q)` is applied *per anchor* and ships with `η=0`, so total expansion mass
≈ `μ·Σ_a w(a)` — grows with anchor count. Longer (richer) queries get *more* expansion, which is
backwards.

**Fix (change + calibrate):**

1. **Query-level global budget**, not per-anchor: `M(Q) = μ · f(specificity(Q))`, allocated across
   anchors ∝ salience (peripheral anchors starved).
2. **Specificity damping, not length:** activate `μ(Q) = μ_ceil·(1 − η·min(1, max_query_IDF/IDF_max))`
   — the lever is already wired, just shipped at `η=0`.
3. **Calibrate:** sweep `η` (and budget-policy: per-anchor vs global); verify on verbose/long vs
   short/ambiguous query buckets that *long queries expand less per anchor*, short/ambiguous expand more.
4. **Keep alias/entity expansion exempt** — a long precise query about one entity may still want its variants.

---

## 5. Latency regression on large corpora — breakdown & fix

Observed: BM25 ≈ **45–140 ms/query**, Edge-RAG ≈ **6–11 s/query** on 2.7–5.4M-doc corpora (~100×
blowup). NQ (2.68M) is ~1.2 s, so it scales with corpus size — classic disk-I/O pathology after the
streaming (disk-backed) index change.

### 5.1 Step-by-step breakdown (don't fix blind)

1. **Per-phase timing** — `expansion_latency_ms` vs `retrieval_latency_ms` (orchestrator) and
   `timings_ms` (extractor). On `beir_dbpedia_entity`, print each. Expected: `retrieval_latency_ms`
   dominates → posting traversal, not GEMM.
2. **Count the fan-out** — read `total_synonyms_injected` / `final_qvec_len` / `unique_synonyms`.
   Hypothesis: expansion produces 30–80 terms/query, so 30–80 posting lists are loaded (vs BM25's ~5),
   each now streamed from disk.
3. **Per-posting-list cost** — instrument bytes-read and load-time per term. If a few *common* terms
   (posting lists of millions of entries) dominate, that is the smoking gun.

### 5.2 Likely root cause

The streaming index reads posting lists *from disk per query*, and expansion multiplies the number
(and often the commonness) of those lists — fine in RAM, catastrophic from disk.

### 5.3 Fix, in priority order

1. **mmap + page cache, not read-per-query** — memory-map the index so hot posting lists live in OS
   page cache; first query warms, later queries hit RAM.
2. **Hybrid RAM/disk** — keep *hot* (common-term) posting lists in RAM, spill only the long tail to
   disk (mirrors the original in-memory index).
3. **Cap expansion fan-out by posting-list cost, not just count** — fold posting-list length into the
   mass floor / budget, so expansion never adds a term whose posting list costs more than it is worth.
   This ties §4 (budget) and §5 (latency) into one lever.
4. **Re-verify the budget calibration after the fix** — with disk I/O removed, re-measure whether the
   "ultrahigh latency" was purely I/O (then expansion stays ~15 ms) or also exposes a fan-out that
   needs capping regardless.

**Order of operations:** fix I/O (mmap/hybrid) first and re-measure; only then tune fan-out/budget.
Fixing the budget before fixing I/O would hide the real problem.

---

## One-line summary

*Edge-RAG is "BM25 + zero-shot expansion," compared only against other zero-shot/unsupervised
expansion sources (RM3 first), simplified to source→gate→budget, with a specificity-driven (not
length-driven) mass budget, benchmarked per-query-type on both in-domain and BEIR/BRIGHT — and the
immediate blocker is a disk-I/O latency regression on large corpora that must be profiled and fixed
before any of the above is trustworthy.*
