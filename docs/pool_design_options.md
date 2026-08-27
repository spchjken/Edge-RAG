# Design Options — Probe-Pool Selection & Bailout

## Problem

Fill the **probe pool** (the terms embedded in `DenseVocabMatrix` and probed against at query time) under a VRAM / TTI / query-latency budget. Corpus vocabulary = `V` distinct analyzed (**KStem**) stems; pool capacity = `N ≤ V`.

## Corrected premise

> ~~The only index-time signals available are frequency (DF/IDF) or nothing. Semantic similarity does not exist until a query arrives.~~

This was wrong, and it forced a false dichotomy. **Term–term similarity is an offline property** — the moment the stems are embedded, `cos(e_u, e_v)` exists for every pair. Only **query–term** similarity genuinely waits for query time. Moreover, "good expansion candidate" has a query-independent component: a **semantic hub** (a term near many *other* terms) is a good expansion target for *any* future query, because whatever anchor arrives has a better chance of landing near it. So **coverage/centrality in embedding space** is a valid index-time ordering signal — and it is the one this design should use, because **frequency is the wrong feature**: it is a proxy for a semantic property it only weakly correlates with.

## Decision levers (four, not two)

1. **Pool size `N`** — how many stems are embedded and probed (`N ≤ V`).
2. **Selection function** — how the top-`N` are chosen (ordering / coverage).
3. **DF floor** — none / `DF≥2` / … (largely superseded by coverage; see below).
4. **Bailout scope** — which not-in-pool terms get a second chance at query time, and whether they are re-assessed.

These four are *independent*; the old "three approaches" collapsed them into fixed combos and hid the middle options (e.g. full vocab + bailout, no-floor + ordering + no bailout).

---

## Selection functions (corrected)

| Function | What the pool fills with |
| :--- | :--- |
| pure IDF (rarest first) | DF=1 islands dominate. A rare term has ~no embedding neighbors, so it is a **poor probe target** even though "rare terms matter" feels right. The technical-compound-vs-typo split cannot be made on DF alone. |
| salience `IDF × ln(1+DF)` | Peaks at **mid-DF** (~`DF≈10²` for `N_docs≈20k`); ranks **all** rare terms (DF=1,2,3) at the bottom, just above each other. Under tight `N` it keeps mid-DF only — it does *not* "keep DF=2/3 trash" (that was a self-contradiction: DF=2/3 sit at the bottom, barely above DF=1). Frequency-blind to semantics. |
| DF descending | Common terms; loses rare discriminators. *(This is what the current code effectively does — see divergence note below.)* |
| agnostic (random) | Distribution-preserving sample. "No DF bias" ≠ "no signal": uniform random is a representative sample and a necessary control. |
| **coverage / farthest-point (FPS)** | Spreads across embedding space; surfaces **mid-frequency technical hubs** (`embedding`, `retrieval`, `attention`, `latency`, `cache`); deprioritizes DF=1 islands automatically (they have no neighbors to cover). |

## Recommended direction — coverage-based selection

1. **Index phase (offline, cached to disk):**
   1. Take the **full analyzed stem vocab** — **no DF floor**. (The analyzer already applies stopword / digit / length filters; those are lexical, not frequency.)
   2. **Batch-embed every distinct stem once**, each via its **canonical surface form** (BGE is surface-trained; KStem outputs like `relat` embed poorly — the analyzer's surface↔stem bridge picks a real word). Do **not** embed surface-form variants separately: KStem has already conjoined them, and inflectional clones would only form spurious clusters that waste coverage slots.
   3. Compute coverage/hub scores; select top-`N` by **farthest-point sampling** (greedy max-min cosine to the already-selected set) or k-means centroids (`k=N`).
      - **Coverage cost:** FPS is `O(N·V·d)` ≈ **96 GFLOP** at `V≈50k`, `N≈2500` — ~10 ms on GPU, ~1 s on CPU — **cheaper than the embedding pass feeding it** (a BGE-small forward pass over 50k stems is ~3 TFLOP), and it is one-time offline, so it never touches TTI or query latency. **Do not materialize the `V×V` pairwise matrix** (10 GB at `V=50k`): FPS only keeps a length-`V` distance vector, updated with one `V×d` matvec per selection step. k-means is ~`T`× costlier (the iteration factor) and non-deterministic — prefer FPS. Run the selection GEMMs on GPU if available (current `dense_vocab_matrix.py` stores CPU tensors; at `V≈100k` on CPU the FPS creeps into seconds — still fine offline).
   4. **Cache the full stem embedding matrix**, not just the pool: the `N` pool rows become the runtime `DenseVocabMatrix`, and the remaining rows are the assessment store for bailout (see below). One embedding pass, one matrix, three uses.
2. **Budget facts (corrected):** VRAM = `V·384·4 B` (FP32, current code) or `V·384·2 B` (FP16 target). `V≈20k` → 30 / 15 MB; `V≈200k` → 307 / 153 MB. The old "heavy at 200k" was miscalibrated — that is **0.15–0.31 GB**, only "heavy" relative to the self-imposed ~0.09 GB target, not in any absolute sense, and it is a BEIR-fiqa-scale edge case, not the primary corpora (500 / 1,722 / 970 docs → `V` in the low tens of thousands). The binding constraint on `N` is **query-time GEMM** (linear in `N`) and **candidate noise**, not VRAM/TTI.

## Bailout (corrected scope)

- **Bailout is not "DF=1 rescue."** That label is a leftover from the old `DF≥2` floor, where the only discarded terms were singletons. Under coverage selection the not-in-pool set is **everything that lost the coverage ranking** — DF=1 islands, rare technical terms, mid-frequency hubs that didn't make the cut, even common terms. Bailout covers that whole complement.
- **Bailout is candidate generation, not acceptance.** A bailed term (lexical/boundary match against the posting dictionary) is a *candidate* that must still pass the **same `τ_sim` semantic assessment** as a pool term before it becomes a true expansion term. Assessment of a bailed term is a lookup into the cached full-embedding matrix — zero extra inference (no on-demand encode).
- **`DF=0` is impossible.** Every stem is corpus-sourced (pool and posting dictionary), so every candidate has `DF≥1` by construction. The "`DF=0` is useless for retrieval" concern only applies to *external* expansion sources (thesaurus, LLM-generated synonyms), which this design does not have.

Pipeline (query time): anchors → probe against **pool** (semantic candidates) ⊕ **bailout** against not-in-pool posting terms (lexical candidates) → both streams pass the same semantic assessment → Phase 4 weight assignment → Phase 5 weighted BM25.

---

## Open questions (unresolved)

1. **What is `V` (distinct stems) on the target corpora?** One-line census: `len(idf_registry.doc_freqs)` after the 1.7a analyzer-parity wiring. Expect low tens of thousands for the primary corpora; low-100k is only the BEIR-fiqa ceiling. This bounds both the "embed all" cost and the `N` range.
2. **Which selection function wins?** Must be measured, not assumed. Ablate `{pure IDF, salience, coverage/FPS, random}` at fixed analyzed base, `N`, `C_exp`, `τ_sim`. Hybrid collapses to a **lexicographic tiebreak** (coverage first, IDF second) for determinism only — there is no principled signal mix, since coverage already deprioritizes DF=1 islands and precision is handled downstream (`τ_sim`, Phase-4 IDF damping, `C_exp`).
3. **`N`: fixed vs scaled?** Neither is obviously right. The real criterion is **where recall plateaus**, not corpus size. Sweep `N ∈ {500, 1000, 2500, 5000}` and `% ∈ {5%, 10%}` separately. If coverage selection is good, recall should saturate at a *smaller* `N` than frequency selection — a falsifiable consequence of the coverage hypothesis. If `%` is adopted, tie it to the stem census and cap it (`min(10%, 10000)`) to protect the latency budget.
4. **Is the "technical vs not" classification still needed?** Under coverage selection, no — a term earns a slot by covering embedding space, not by document count. The only residual OOV case (anchor with no embedded neighbor) is exactly what bailout's assessment gate handles as a fallback.

---

## Code-vs-doc divergence (must be reconciled)

`corpus_vocab_builder.py` does **not** implement "salience over the `DF≥2` vocab" as this file and `v7_architectural_plan.md` (Upgrade 1.2) claim. It pre-truncates to the **top-2500 by DF descending** (`valid_unigrams[:2500]`) *before* the salience ranking runs, so the salience function only ever sees the most frequent terms — the effective selection is "DF descending," not "`IDF × ln(1+DF)`." It also adds bigrams, applies an upper `15%` DF cap, and defaults `max_vocab_size = 1000` (not 2500). Any ablation must run each selection function **cleanly over the full stem vocab**, not through this pre-truncation.

*(This is a design-options list, not an implementation plan. No code, no docs-outside-this-file, until the approach is chosen.)*
