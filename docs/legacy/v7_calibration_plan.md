# 🎯 V7 Calibration Plan — Runbook

> The architecture is frozen in [`v7_architectural_plan.md`](v7_architectural_plan.md). This document is the **operational runbook** for choosing every free parameter, against the frozen baselines, using the p-sweep harness.

---

## 1. Purpose & Scope

Calibrate the V7 free parameters, phase-by-phase, so that the final frozen configuration beats the analyzed-parity BM25 baseline on recall **without** sacrificing precision or blowing the latency budget. Every sweep uses the same measurement harness (§3) and the same decision rules (§4); nothing is tuned by hand.

**Free parameters to calibrate** (all defaults are the "hold-fixed" neutral values):

| Phase | Parameter | Default | Sweep |
| :--- | :--- | :---: | :--- |
| 1 (pool) | selection | coverage/FPS | `{coverage/FPS, pure IDF, salience, random}` |
| 1 (pool) | `N` (pool size) | 2500 | `{500, 1000, 2500, 5000}` and `{5%, 10%}` of stems |
| 3 (gate) | `Δτ` (signed) | 0 | `{-0.25 … +0.25}` step 0.05 |
| 3 (gate) | `β` (query blend) | 1.0 | `{0, 0.5, 0.65, 1.0}` |
| 3 (gate) | gate variant | adaptive single | `{adaptive single, two-gate, gate + soft reweight}` |
| 4 (budget) | `η` (direction) | 0 | `{-0.5, 0, +0.5}` |
| 4 (budget) | `μ_ceil` (scale) | 0.5 | `{0.25, 0.5, 0.75, 1.0}` |
| 4 (sparsity) | `ε` (mass floor) | 0 | `{0, 0.1%, 0.5%, 1%, 2%, 5%}` of `w(a)` |
| 4 (alloc) | allocation | normalized cosine | `{normalized cosine, softmax(τ=1.0), softmax(τ=0.1), uniform}` |

Fixed (not swept here): `τ_base = 0.55`, `k1 = 1.2`, `b = 0.75`. Constraint: `μ_ceil(1+|η|) ≤ 1`. *(Phase 2 — POS priors and the bailout gate — is already calibrated and out of scope; Phase 5's BM25 is frozen.)*

---

## 2. Success Gates (acceptance criteria)

V7 is **done** when, at the final frozen config:

1. **Precision:** macro-avg `Strict@10` ≥ analyzed-parity BM25, on **≥ 8/10** benchmarks (tolerance ±0.5pp).
2. **Recall:** macro-avg `DocRec@10` strictly **>** analyzed-parity BM25, on **≥ 7/10** benchmarks.
3. **Latency:** p50 retrieval (Phase 5) `≤ 15ms`; p50 total `≤ 50ms`.
4. **No single-benchmark domination:** the win is not driven by one benchmark alone (drop-one-out check — the gates must still hold after removing any single benchmark).

If a config fails any gate, it is disqualified regardless of recall.

---

## 3. Measurement Harness

### 3.1 Benchmarks (fixed, 10)

`beir_fiqa_doc_level`, `beir_nfcorpus_doc_level`, `beir_scifact_doc_level`, `bright_economics_doc_level`, `bright_robotics_doc_level`, `bright_stackoverflow_doc_level`, `enterpriserag_doc_level`, `financebench_doc_level`, `liverag_doc_level`, `multihop_rag_doc_level`.

### 3.2 Metrics (fixed)

Per benchmark, report **Strict@10, Complete@10, DocRec@10, DocRec@50, Prec@10, MRR@10**, plus **p50 latency** (expansion + retrieval split), **TTI**, and **Peak VRAM** (baselines only). Aggregate with **macro-average** across the 10 benchmarks.

### 3.3 Frozen baselines (integrity rules)

`AnalyzedLuceneBM25` is the **primary control** (all gains measured against it, never the legacy baseline). `LuceneBM25Baseline`, `BM25Baseline`, `BM25_stemmed`, and v1/v5/v6 are measurement controls — **untouched, no backport**. Analyzed scores live in a separate avgdl/DF space; never mixed with legacy numbers.

### 3.4 Trace files

Every run emits per-dataset `trace_<dataset>_<model>.json` with per-query: `query_id`, `raw_question`, `aspects`, `augmented_token_list`, `ground_truth_chunk_ids`, `retrieved_top10_chunk_ids`, and `metrics{strict_hit@10, chunk_recall@10, precision@10, first_gold_rank, latency_ms}`. A summary table (the p-sweep format) is generated from the traces.

### 3.5 Run naming

Config id format: `<stage>__<param>=<value>__…`. Every distinct config is a named condition; the decision log (§10) records the metric table and the reason for each keep/reject.

---

## 4. Decision Rules (applied at every stage)

- **Primary objective:** maximize macro-avg **DocRec@10**.
- **Precision guardrail:** `Strict@10` must stay ≥ analyzed-parity (macro, ±0.5pp). Below it → disqualified.
- **Latency guardrail:** p50 retrieval ≤ 15ms. Above it → disqualified unless it is the *only* config reaching the recall target.
- **Tie-break order:** higher `Strict@10` → lower p50 latency → fewer parameters (simpler config).
- **Plateau rule (for magnitude sweeps):** pick the smallest/least-extreme value whose recall is within **0.5pp** of the sweep maximum (avoid over-fitting the exact grid point).
- **Overfitting guard:** report per-benchmark, not just macro; a config must improve on the gate-relevant metric on ≥ 7/10 benchmarks, not via one outlier.

---

## 5. Stage 0 — Stem census (prerequisite)

`len(idf_registry.doc_freqs)` after analyzer-parity wiring = `#distinct stems`. Record per benchmark. This bounds the "embed all" cost, the `N` sweep, and the `%`-scaling range.

---

## 6. Stage 1 — Phase 1 pool (selection × size)

**Hold fixed:** Phases 2–4 at defaults (`Δτ=0, β=1.0, η=0, μ_ceil=0.5, ε=0`, normalized cosine).

**6.1 Selection** — at `N = 2500`, run `{coverage/FPS, pure IDF, salience, random}`. Each runs **cleanly over the full stem vocab** (no `[:2500]`/top-5000 salience pre-truncation). **Pick** the selection maximizing DocRec@10.

*Falsifiable prediction:* coverage > random > salience > pure IDF.

**6.2 Size** — at the winning selection, run `N ∈ {500, 1000, 2500, 5000}` and `{5%, 10%}` of stems. **Pick** the smallest `N` within 0.5pp of the max DocRec@10 (plateau rule).

*Falsifiable prediction:* coverage saturates at a smaller `N` than frequency selection.

---

## 7. Stage 2 — Phase 3 gate (Δτ → β → variant)

**Hold fixed:** Stage-1 winning pool; Phase 4 at defaults.

**7.1 `Δτ` sign (the open question).** At `β = 1.0`, adaptive single gate, run `Δτ ∈ {-0.25, 0, +0.25}`. **Pick** the sign maximizing DocRec@10. Report `starved_aspects` as the diagnostic.

**7.2 `Δτ` magnitude.** At the winning sign, run `Δτ ∈ {-0.25 … +0.25}` step 0.05 (11 values). **Pick** the least-extreme value within 0.5pp of max (plateau rule).

**7.3 `β`.** At the winning `Δτ`, run `β ∈ {0, 0.5, 0.65, 1.0}`. **Pick** the `β` maximizing DocRec@10.

**7.4 Gate variant.** At the winning `(Δτ, β)`, run `{adaptive single, two-gate (anchor→query), gate + soft reweight}`. **Pick** the variant maximizing DocRec@10.

---

## 8. Stage 3 — Phase 4 budget & allocation (η → μ_ceil → ε → allocation)

**Hold fixed:** Stage-1 pool + Stage-2 gate.

**8.1 `η` (direction).** At `μ_ceil = 0.5`, run `η ∈ {-0.5, 0, +0.5}`. **Pick** the direction maximizing DocRec@10. Respect `μ_ceil(1+|η|) ≤ 1` (`η = -0.5 ⇒ μ_ceil ≤ 0.67`).

**8.2 `μ_ceil` (scale).** At the winning `η`, run `μ_ceil ∈ {0.25, 0.5, 0.75, 1.0}`. **Pick** per the plateau rule.

**8.3 `ε` (mass floor — the latency/recall tradeoff).** At the winning `(η, μ_ceil)`, run `ε ∈ {0, 0.1%, 0.5%, 1%, 2%, 5%}` of `w(a)`. **Decision procedure:** sort by `ε` ascending; pick the **largest** `ε` whose p50 retrieval is ≤ 15ms *and* whose DocRec@10 is within 0.5pp of the `ε = 0` maximum. *(This is the mechanism that converts the ~10× `|w_Q|` blow-up back to ~hundreds of terms, per the smoke test, while keeping the recall gain.)*

**8.4 Allocation.** At the winning `(η, μ_ceil, ε)`, run `{normalized cosine, softmax(τ=1.0), softmax(τ=0.1), uniform}`. **Pick** the allocation maximizing DocRec@10 (expected: normalized cosine confirms the theory-§4.2 default).

---

## 9. Stage 4 — Joint confirmation & freeze

1. Run **2–3 joint `η` × `μ_ceil` × `ε`** combos around the winners to check for interaction (the staging assumed independence).
2. If the joint sweep moves a parameter by more than one grid step, promote that interaction to a small local grid.
3. **Freeze** the final config and re-run all 10 benchmarks end-to-end (fresh process, fresh seeds) for the record.

---

## 10. Reporting & Decision Log

For every stage, append to the decision log:

- The config ids tested.
- The macro table (Strict@10, DocRec@10, Prec@10, MRR@10, p50 retrieval/total).
- The per-benchmark table (overfitting guard).
- The keep/reject decision and the rule that decided it.

Final deliverable: the frozen config + the full decision log + the final trace files + a one-paragraph "what we learned" (esp. the `Δτ` sign answer, the `η` direction answer, and the winning `ε`).

---

## Appendix — Parameter reference (defaults)

| Parameter | Default | Meaning |
| :--- | :---: | :--- |
| `selection` | coverage/FPS | pool selection strategy |
| `N` | 2500 | pool size |
| `τ_base` | 0.55 | gate floor (fixed) |
| `Δτ` | 0 | gate adaptivity (signed) |
| `β` | 1.0 | pure anchor-sim (no query blend) |
| `gate variant` | adaptive single | how query context is applied |
| `η` | 0 | budget direction (flat) |
| `μ_ceil` | 0.5 | budget scale (half anchor mass) |
| `ε` | 0 | mass floor (pure gate-only) |
| `allocation` | normalized cosine | synonym mass split |
| `k1`, `b` | 1.2, 0.75 | BM25 (frozen) |
