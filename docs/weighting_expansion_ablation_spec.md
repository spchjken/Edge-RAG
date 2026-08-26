# 🧪 External Weighting & Expansion Ablation Spec

**Question:** does an *external* query-side weight layer `w_Q(t)` ever beat the no-external-weight control (plain analyzed BM25), and how much of the old v1/v5/v6 gain was **weighting** vs **expansion**?

**Status:** specification only — no code is written or run by this document.

---

## 1. Purpose

The analyzed-parity baseline (`AnalyzedLuceneBM25`, KStem-stemmed single-field index) is now the strongest lexical control and beats every v1/v5/v6 schema on the doc-level corpora. Two consequences follow:

1. The old schemas' advantage over *legacy* (unstemmed) BM25 is **confounded**: they changed anchor weighting (repetition counts) and synonym expansion together, so neither can be attributed.
2. The V7 continuous anchor weight `w(a)` (POS × centrality × IDF re-scaling) has **never been tested**, and the whole V7 value-add now rests on **analyzed BM25 + budgeted cross-root synonym expansion** — which has also never been run.

This spec defines an ablation that separates the three levers — **KStem (already in the baseline), external weighting, and expansion** — into attributable deltas.

---

## 2. Instrument

`AnalyzedLuceneBM25.retrieve_weighted(term_weights: Dict[str, float], top_k)` over the analyzed (stemmed) index.

Scoring (per [`theoretical_foundations_anchored_expansion.md`](theoretical_foundations_anchored_expansion.md) Eq. 1–2):

$$\text{Score}(D,Q) = \sum_{t \in Q} \underbrace{w_Q(t)}_{\text{external weight}} \cdot \text{IDF}(t) \cdot \psi(\text{TF}(t,D), |D|)$$

The **only** quantity varied here is `w_Q(t)` (and, for the expansion arms, *which extra terms* are injected). BM25's own `IDF(t)·ψ` is never modified.

**Key simplification:** no anchor selection — every analyzed query token is an anchor (`p = 1.0`, which the p-sweep showed is optimal). This removes the selection/ranking dimension entirely.

---

## 3. Weighting-Only Ladder (W0–W4)

No synonym injection, no BGE probing. Only the anchor weight `w_Q(a)` changes. For each arm, `a` ranges over **all analyzed query tokens**.

| Arm | `w_Q(a)` | Isolates | External deps |
| :---: | :--- | :--- | :--- |
| **W0** (control) | `1.0` | = plain analyzed BM25 | none |
| **W1** | `1 + γ·(IDF(a)/max_IDF(Q))`, `γ = 2.0` | extra IDF re-scaling | none |
| **W2** | `w_POS(a)` ∈ {Noun/Entity `1.25`, Verb `0.85`, Other `0.70`} | POS priors | POS tagger (on raw text, pre-analyzer) |
| **W3** | `1 + γ·Centrality(a)`, `γ = 2.0`, `Centrality ∈ [0,1]` (clamped) | centrality weighting | BGE embeddings of surface forms |
| **W4** | `w_POS(a)·(1 + γ·(IDF(a)/max_IDF)·Centrality(a))` | full V7 op-5 formula | POS + BGE |

Centrality (per V7 plan Phase 2 op 4):
$$\text{Centrality}(a) = \frac{1}{|Q|-1}\sum_{t \in Q \setminus \{a\}} \cos(\mathbf{e}_a, \mathbf{e}_t)$$
with `e_a`, `e_t` embedded from **surface forms** (BGE is surface-trained); the analyzer bridges surface ↔ stem.

---

## 4. Expansion Arms (E1, E2 — attribution)

These inject cross-root synonyms on top of the weight layer, via the dense probing pipeline (Phase 3 → Phase 4). They require `DenseVocabMatrix` + `BM25DenseAspectExtractor`.

| Arm | Anchors | Synonyms | Isolates |
| :---: | :--- | :--- | :--- |
| **E1** | W0 (`w=1.0`) | top `C_exp = 2` cross-root synonyms per anchor, IT-MPE-damped | **expansion effect alone** |
| **E2** | best of W1–W4 (if any wins) | same as E1 | full V7 hypothesis (weighting + expansion) |

Synonym generation (V7 Phase 3): `Dual_Sim(a,v) = β·cos(e_a,e_v) + (1-β)·cos(e_Q,e_v)`, `β = 0.65`, adaptive gate `τ_sim(a) ∈ [0.80, 0.90]`, keep top `C_exp = 2` **cross-root** candidates (pool is already stem-diverse, so no stem-diversity gate).

Synonym weight (theorem Eq. 4, single-tier):
$$w(s \mid a) = \mu(Q)\cdot w(a)\cdot \min\!\left(1.0,\ \frac{\text{IDF}(a)}{\text{IDF}(s)}\right)\cdot p(s \mid a)$$
with `μ(Q) ∈ [0.18, 0.35]` (V7 Phase 4), `p(s|a)` = temperature-scaled softmax (`τ = 0.10`).

> For E1, `w(a) = 1.0`, so `w(s|a) = μ(Q)·min(1, IDF_a/IDF_s)·p(s|a)` — the pure expansion contribution on the analyzed base.

---

## 5. Corpora & Metrics

**Corpora** (doc-level, unchanged from the p-sweep): `beir_fiqa_doc_level`, `beir_nfcorpus_doc_level`, `beir_scifact_doc_level`, `bright_economics_doc_level`, `bright_robotics_doc_level`, plus any remaining doc-level corpora in the full p-sweep run.

**Primary metric pairing (must be reported together):**

- **DocRec@10** — recall guardrail (the value V7 must move)
- **Strict@10** — precision guardrail (expansion buys recall, risks precision)

**Secondary:** Complete@10, DocRec@50, Prec@10, MRR@10, Latency (p50 ms), TTI (s), Peak VRAM (GB).

---

## 6. Reproducibility Requirements

Per `.agents/rules/02-reproducibility.md`:

1. `--seed` argument on every run; lock `random`, `numpy`, `torch` seeds at entry.
2. JSON environment header in every results file: CPU model, GPU name, driver, CUDA, PyTorch, Ollama, Python.
3. VRAM via `torch.cuda.max_memory_allocated()` after `torch.cuda.reset_peak_memory_stats()`; clear GPU cache between arms; record peak per-run, not averaged.
4. Timing: `torch.cuda.synchronize()` before timing GPU ops; report **median over ≥3 runs with std** for latency.
5. Results as timestamped CSV/JSON under `results/weighting_ablation/`; never overwrite prior results.

---

## 7. Interpretation & Decision Rules

1. **Weighting effect = W0 → W1..W4.**
   - If all of W1–W4 are ≤ W0 (within noise): external weighting adds nothing → **adopt `w(a) = 1.0`** (uniform anchors; BM25's own IDF is the sole term weight). The old schemas' gain over legacy BM25 must then be attributed to expansion (and/or KStem), *not* weighting.
   - If some `Wi` exceeds W0 significantly: keep that component and note it as the one justified external weight.

2. **Expansion effect = W0 → E1.**
   - If E1 ≤ W0: budgeted cross-root expansion adds nothing on the analyzed base → **V7 has no demonstrated value-add over analyzed BM25** → re-scope.
   - If E1 > W0: expansion is the value-add, and E1 becomes the new reference.

3. **Weighting-on-top-of-expansion = E1 → E2.**
   - If E2 ≈ E1: the external weighting is redundant even alongside expansion → final design = E1 (uniform anchors + IT-MPE expansion).
   - If E2 > E1: keep the winning weight layer.

4. **Significance:** deltas must clear a noise floor (std over the ≥3 runs), not just raw-point differences.

---

## 8. Output Artifacts

- `results/weighting_ablation/<timestamp>_<arm>.json` — per-arm per-corpus metrics + env header.
- `results/weighting_ablation/weighting_ablation_summary.md` — the W0–W4 / E1–E2 comparison table and the §7 verdicts.

---

## 9. Dependencies & Out of Scope

- **Runs on the core project folder (other PC)** where the analyzed index, doc-level data, and harness live — not in the docs-only checkout.
- **W2** needs a POS tagger (raw text, pre-analyzer); **W3/W4** and **E1/E2** need BGE embeddings + `DenseVocabMatrix`.
- **Out of scope:** anchor selection (p is fixed at 1.0), stemmer variants (fixed KStem), dual-field index, re-tuning of `μ`, `τ_sim`, or `C_exp` (those are separate sweeps).

---

## 10. Related Documents

- [`theoretical_foundations_anchored_expansion.md`](theoretical_foundations_anchored_expansion.md) — IT-MPE theorem (Eq. 1–4).
- [`v7_architectural_plan.md`](v7_architectural_plan.md) — Phase 2 (anchor weighting), Phase 3 (probing), Phase 4 (IT-MPE allocation).
- [`results/p_sweep_ablation/p_sweep_summary.md`](../results/p_sweep_ablation/p_sweep_summary.md) — the W0-equivalent ("Analyzed Parity Baseline") numbers already exist here.
