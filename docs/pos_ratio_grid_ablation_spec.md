# 🧪 POS Weight Ratio Grid Ablation Spec

**Question:** what are the optimal POS prior weights (as *ratios*) for the anchor weighting `w(a) = w_POS(a)`?

**Status:** specification only — no code is written or run by this document.

---

## 1. Purpose

The weighting ablation (`docs/weighting_expansion_ablation_spec.md`) found that, among W0–W4, **only POS priors (W2) added value** (`+0.52` DocRec@10 macro), and that the composite W4 (`POS·(1+γ·IDF·Centrality)`) was *worse* than POS alone. Consequence: Phase 2's anchor weight collapses to

$$w(a) = w_{\text{POS}}(a)$$

with the IDF re-scaling and centrality multipliers **retired**.

The remaining open question is the **numeric values** of the POS priors. The current `{noun 1.25, verb 0.85, modifier 0.70}` are arbitrary; this spec defines a grid to find the optimum.

---

## 2. Scale-Invariance → 2D Design Space

The score is `Score(D) = Σ_t w_Q(t)·IDF(t)·ψ(t)`. Multiplying **all** POS weights by the same `c > 0` scales the whole score by `c` and leaves **top-k ranking unchanged**. Therefore only two **ratios** are meaningful:

- `verb/noun`
- `modifier/noun`

Fix **noun = 1.0** as the reference and grid the two ratios.

### Grid (56 cells — Option B Uniform High-Density)

| Axis | Values | Count |
| :--- | :--- | :---: |
| `verb/noun` | `{0.2, 0.4, 0.6, 0.75, 0.9, 1.0, 1.2}` | 7 |
| `modifier/noun` | `{0.0, 0.2, 0.4, 0.6, 0.75, 0.9, 1.0, 1.2}` | 8 |

- **Total grid points:** $7 \times 8 = \mathbf{56\text{ cells}}$.
- **Coverage:** Includes complete modifier suppression (`modifier = 0.0`), intermediate fine-grained ratios around the historical default (`1.25/0.85/0.70` $\implies$ ratios `0.68/0.56`), uniform anchor control (`1.0/1.0`), and inverted boost (`1.2`).

---

## 3. Instrument & Pipeline

`AnalyzedLuceneBM25.retrieve_weighted(term_weights, top_k)` over the analyzed (KStem-stemmed) index.

- **No anchor selection:** every analyzed query token is an anchor (`p = 1.0`).
- **No expansion:** no synonym injection (expansion/`τ_sim` is Phase 3, out of scope here).
- **Weights:** `w_Q(t) = w_POS(pos_tag(t))` per the grid cell.

### POS tagging & category mapping

Tagging runs on **raw text** (pre-analyzer); labels are carried onto the corresponding analyzed stems via token index.

| Category | Grid weight | Penn Treebank | spaCy UPOS |
| :--- | :--- | :--- | :--- |
| **Noun/Entity** (reference) | `1.0` (fixed) | `NN, NNS, NNP, NNPS` | `NOUN, PROPN` |
| **Verb** | `verb/noun` × 1.0 | `VB, VBD, VBG, VBN, VBP, VBZ` | `VERB` |
| **Modifier/Other** | `modifier/noun` × 1.0 | `JJ, JJR, JJS, RB, RBR, RBS`, + everything else | `ADJ, ADV`, + everything else |

**Technical tokens** (acronyms, compounds, digits — the analyzer's exemption set) are always **Noun/Entity** (`1.0`), regardless of tag. The tagger (and its version) MUST be recorded in the env header for reproducibility.

---

## 4. Trace Files (REQUIRED — per-query)

Every query MUST emit a `trace_*.json` under the same run directory. This is the primary diagnostic — it lets us verify that a miss was (or wasn't) caused by a key noun being mis-tagged and damped.

Schema (adapted from `docs/EVALUATION_METRICS.md` §3.2 for the V7 float-weighting regime — note `anchor_weight` replaces the legacy integer `repetition`, and `term_weight_dict` replaces `augmented_token_list`):

```json
{
  "query_id": "q_8968b27d",
  "raw_question": "How many test tasks are included in the EHR-Complex benchmark?",
  "anchors": [
    {
      "anchor_term": "ehr",               // analyzed (stemmed) form
      "surface_form": "EHR",
      "pos_tag": "NNP",
      "pos_category": "noun",
      "anchor_weight": 1.0,
      "anchor_idf": 5.652,
      "is_heuristic_entity": true
    },
    {
      "anchor_term": "test",
      "surface_form": "test",
      "pos_tag": "NN",
      "pos_category": "noun",
      "anchor_weight": 1.0,
      "anchor_idf": 3.10,
      "is_heuristic_entity": false
    }
  ],
  "term_weight_dict": { "ehr": 1.0, "test": 1.0, "task": 0.7 },
  "ground_truth_chunk_ids": ["EHR_Complex_block2_chunk1"],
  "retrieved_top10_chunk_ids": ["EHR_Complex_block1_chunk0", "EHR_Complex_block2_chunk1"],
  "retrieved_top10_scores": [12.34, 11.98],
  "metrics": {
    "strict_hit@10": true,
    "doc_recall@10": 1.0,
    "precision@10": 0.1,
    "first_gold_rank": 2,
    "latency_ms": 8.2
  }
}
```

**Forward-compatible fields:** expansion arms later add `candidates_above_tau` and `injected_synonyms` to each anchor; in this POS-only ablation they are absent (`null`/omitted).

---

## 5. Corpora & Metrics

**Corpora:** the same 10 doc-level corpora as the weighting ablation (`enterpriserag_doc_level`, `liverag_doc_level`, `beir_scifact_doc_level`, `beir_nfcorpus_doc_level`, `beir_fiqa_doc_level`, `multihop_rag_doc_level`, `financebench_doc_level`, `bright_economics_doc_level`, `bright_stackoverflow_doc_level`, `bright_robotics_doc_level`).

**Primary:** `DocRec@10` and `Strict@10` (recall + precision guardrail).
**Secondary:** `Complete@10`, `DocRec@50`, `Prec@10`, `MRR@10`, latency (p50 ms).

---

## 6. Reproducibility

Per `.agents/rules/02-reproducibility.md` and `docs/EVALUATION_METRICS.md` §1:

1. `--seed` on every run; lock `random`, `numpy`, `torch`.
2. JSON env header (CPU, GPU, driver, CUDA, PyTorch, Python, **POS tagger name + version**).
3. Peak VRAM via `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()`; clear cache between cells.
4. Latency: `torch.cuda.synchronize()` around GPU ops; median over ≥3 runs with std.
5. Results to `results/pos_weight_ablation/` as timestamped files; never overwrite.

---

## 7. Decision Rules

1. **Optimum = highest `DocRec@10`** on the ratio grid, **subject to** `Strict@10` not regressing below W0's value (POS must not buy recall by leaking precision).
2. **Plateau check:** report the full 2D heatmap (not just the argmax) to see whether the optimum is a sharp peak or a flat plateau — a plateau means the exact weights are forgiving; a peak means they matter.
3. **financebench attribution:** report macro mean **with and without `financebench_doc_level`** (it drove W2's entire gain in the previous run). If the optimum disappears without financebench, the POS benefit is corpus-specific and must be stated as such.
4. **Trace spot-check:** for a sample of queries where a high-weight noun anchor changed `first_gold_rank` vs W0, confirm the POS tag was correct (this is the guard against a tagger error masquerading as a weighting win).

---

## 8. Output Artifacts

- `results/pos_weight_ablation/<timestamp>/<cell>.json` — per-cell per-corpus metrics + env header.
- `results/pos_weight_ablation/<timestamp>/trace_<query_id>.json` — per-query traces (§4).
- `results/pos_weight_ablation/pos_ratio_grid_summary.md` — 2D heatmaps (DocRec@10, Strict@10) + top-3 cells + the §7 verdicts.

---

## 9. Out of Scope

- Expansion / `τ_sim` gate (Phase 3).
- Centrality and extra-IDF weighting (retired by the weighting ablation).
- Anchor selection (fixed `p = 1.0`).
- Re-tuning `k1`, `b`, or the stemmer (fixed).
