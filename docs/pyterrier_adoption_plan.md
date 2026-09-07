# PyTerrier Adoption Plan: Decision & Integration Guide

**Scope:** Whether and how to migrate the Edge-RAG baseline/evaluation harness to [PyTerrier](https://github.com/terrier-org/pyterrier), given the custom analyzer, custom baselines, edge-memory metrics, and large-corpus (500k–5M doc) requirements.

**Decision (TL;DR):** **Selectively adopt, do not refactor wholesale.** Take PyTerrier's *evaluation metrics* and use it as the *baseline harness* (Terrier BM25 + dense + SPLADE plugins), keep the custom `EdgeRAGAnalyzer` (it is portable via pre-tokenization), keep the custom v7 pipeline, and keep the edge-memory metrics (TTI/VRAM/RAM) as a custom layer PyTerrier does not provide.

---

## 1. What PyTerrier provides

PyTerrier is a **declarative IR-experiment framework** over Terrier (Java). Pipelines are chained with `>>`, evaluated with `pt.Experiment`, and scored with standard trec_eval semantics.

| Capability | What we get |
|---|---|
| Standard sparse retrievers | Terrier BM25 / DPH / TF-IDF via `pt.BatchRetrieve` |
| Dense retrieval | `pyterrier_dr` (FAISS/HNSW, pluggable HF models) |
| Sparse neural | `pyterrier_splade` (SPLADE checkpoints) |
| LLM reranking | `pyterrier_genrank` |
| Standard datasets | `ir_datasets` / BEIR / TREC / MS MARCO loaders |
| Standard metrics | `ir_measures` / `pytrec_eval` (recall, MRR, nDCG, etc.) |
| Reproducibility | serializable declarative pipelines, shareable with the IR community |
| Large-corpus indexing | Terrier single-pass indexing + bit-compressed postings |

## 2. What our custom setup provides that PyTerrier does not

These are the reasons a *wholesale* migration is wrong for this project:

1. **Edge-memory metrics (TTI / VRAM / RAM).** PyTerrier does not *report* per-component TTI/VRAM/RAM as `pt.Experiment` columns. They are measurable (wall-clock around `indexer.index()`, `torch.cuda` for VRAM, `psutil` RSS for RAM — Terrier's JVM runs in-process via Jnius, so RSS captures it), but we have to measure them ourselves. This is our differentiator (ephemeral edge constraint, 16 GB RAM / 16 GB VRAM budgets).
2. **The v7 pipeline.** Custom query expansion, dense-vocab matrix, aspect extraction — PyTerrier contributes nothing to it; it can only *host* it as a custom transformer.
3. **The custom analyzer.** `EdgeRAGAnalyzer` (KStem + Lucene stopwords + technical-token exemption) is a research contribution, not a detail.
4. **Distributed/streaming analysis** (sharding, IES/Poisson top-k, federated vocabulary). PyTerrier is orthogonal to this.

## 3. What we adopt vs. keep

| Component | Action | Rationale |
|---|---|---|
| Metric computation | **Adopt** `ir_measures` / `pytrec_eval` | drop-in, standard semantics, removes any "did we compute recall right?" doubt |
| Sparse baseline | **Adopt** Terrier BM25 (`pt.BatchRetrieve`) *as an external anchor*, keep `AnalyzedLuceneBM25` as primary control | standard implementation for credibility; same-analyzer control stays |
| Dense baseline | **Adopt** `pyterrier_dr` (bge-small) alongside `DenseRAGBaseline` | standard dense harness; model-flexible |
| SPLADE baseline | **Adopt** `pyterrier_splade` | standard sparse-neural harness |
| v7 | **Keep**, wrap as a custom `TransformerBase` | it is the system under study |
| Analyzer | **Keep** via pre-tokenization (see §4) | no need to switch to Terrier's tokenizer |
| TTI/VRAM/RAM harness | **Keep** as a custom layer | PyTerrier does not provide it |

## 4. Keeping `EdgeRAGAnalyzer` on Terrier (BYO tokenizer)

Terrier's term pipeline is **Tokeniser → Stopwords → Stemmer**, and all three are pluggable. We do **not** need to adopt Terrier's default tokenizer (which would fragment `qwen2.5-7b`, `fp16`, `nav2_bringup` — a real regression on `enterpriserag`/`stackoverflow`).

**Approach (pre-tokenize + pass-through):**

1. Run `EdgeRAGAnalyzer.analyze(doc)` in Python as we already do, producing the token stream (technical-token exemption, KStem, Lucene stopwords already applied).
2. Feed the pre-tokenized text into `pt.IterDictIndexer` with Terrier's term pipeline neutralized (whitespace-only/pass-through tokeniser, `termpipelines=` empty), so Terrier indexes our tokens **verbatim** and does not re-split compounds.
3. Apply the **same** pre-analysis to queries before `pt.BatchRetrieve` so query terms match index terms 1:1 (the same parity rule the code already enforces).

Fallback: write a small Java `Tokeniser` implementing `org.terrier.indexing.tokenisation.Tokeniser` if pass-through config is insufficient.

## 5. Hosting v7 as a custom transformer

**Terminology:** in PyTerrier a "transformer" is a *pipeline stage* (dataflow `input → output`, base class `pt.TransformerBase` with a `transform()` method) — **not** the neural Transformer architecture. Retrievers, rerankers, indexers, and query expanders are all transformers composed with `>>`.

Convention: query transformers take `["qid","query"]`; retrieval transformers return `["qid","docno","score","rank"]`; rerankers modify scores in place.

To host v7: wrap `V7AspectExtractor.extract()` + `InvertedPostingIndex` retrieval in a class implementing `transform()`, so it becomes a drop-in `>>` stage evaluated side-by-side with `pt.BatchRetrieve` and the dense/SPLADE plugins. This is the main integration cost.

## 6. Methodological findings that shape the comparison

These are the *correct* ways to report v7's contribution, independent of which framework hosts it.

1. **The clean control is same-analyzer BM25, not dense.** Because both v7 and `AnalyzedLuceneBM25` share `EdgeRAGAnalyzer`, the only clean measure of v7's *expansion* is:

   ```
   v7 − BM25(analyzed)     # same tokenizer → isolates the expansion
   ```

   Current data (Strict@10): `enterpriserag` **0.0**, `bright_stackoverflow` **+9.4**. These are the numbers to headline as "v7's gain".

2. **The technical-token exemption is a *shared* advantage, not a v7 win.** On `enterpriserag`, `BM25(analyzed) = v7 = 79.8` while `BM25(blank) = 70.6` and `dense = 59.2`. The +20.6 "v7 over dense" is the *shared regex tokenizer* (keeps compounds whole) beating BGE's BPE — it applies identically to analyzed BM25. So the v7-vs-dense margin on technical corpora is largely a **tokenization confound**, not a retrieval-method superiority.

3. **The correct ablation is same-tokenizer, not same-framework.** To test whether the expansion survives a tokenizer change, compare `v7-with-tokenizer-X` vs `BM25-with-tokenizer-X` (same X). That gap — not the v7-vs-dense margin — is the claim.

## 7. Large-corpus fit (16 GB RAM + 16 GB VRAM)

| Component | Cost @ 5M docs | Fits? |
|---|---|---|
| Terrier BM25 (index + retrieve) | few GB RAM, **0 VRAM** (single-pass indexing, bit-compressed postings) | ✅ trivially |
| Full dense retrieval, bge-small (`N × 384 × 4B`) | **7.7 GB** embeddings + FAISS overhead | ✅ but it is the pressure point |
| Full dense retrieval, bge-base/large | 15–20 GB | ❌ over RAM |
| **v7 dense component** (`DenseVocabMatrix`, vocab-capped 50k) | **77 MB** (`50k × 384 × 4B`) | ✅ negligible |
| SPLADE model | ~0.26 GB VRAM + sparse index | ✅ |
| Raw corpus text (if held in Python) | 3–5 GB | ✅ (mind the running total) |

**Key distinction:** *full dense retrieval* embeds every document (`N × dim` → 7.7 GB @ 5M), whereas *v7* embeds only the vocabulary (`≤50k × dim` → 77 MB). This is the "v7 embeds O(vocabulary), dense embeds O(corpus)" property. The 7.7 GB belongs only to the dense *baseline*; v7's dense side is ~100× smaller.

**Conclusion:** Terrier handles 5M-doc BEIR easily on 16 GB RAM (BM25 is a non-issue, 0 VRAM). The only budget pressure is the *dense baseline* — stick to bge-small (7.7 GB), not bge-base/large. The 16 GB VRAM is far more than the neural models need (0.13–0.26 GB); the real constraint is total RAM once dense embeddings + raw corpus + JVM heap are summed.

## 8. Action items (phased)

1. **Phase 0 — Metrics (no refactor).** Swap metric computation to `ir_measures`/`pytrec_eval`; confirm our recall/MRR/nDCG match trec_eval semantics. Cheapest, highest-value.
2. **Phase 1 — Baseline anchor.** Add one Terrier-BM25 row (`pt.BatchRetrieve`) as an external reference, keeping `AnalyzedLuceneBM25` as the primary same-analyzer control. This doubles as a tokenization ablation (Terrier/Porter vs. our KStem analyzer).
3. **Phase 2 — Dense + SPLADE plugins.** Adopt `pyterrier_dr` (bge-small) and `pyterrier_splade` as standard harnesses, cross-checked against our `DenseRAGBaseline`/`SPLADEBaseline`.
4. **Phase 3 — v7 as a transformer.** Wrap v7 in `TransformerBase`, keeping the pre-tokenized `EdgeRAGAnalyzer` pipeline; evaluate with `pt.Experiment` side-by-side with the above.
5. **Phase 4 — Edge-memory layer.** Keep the custom TTI/VRAM/RAM harness as an outer layer around PyTerrier (it is not a PyTerrier feature).

**Do not** replace `AnalyzedLuceneBM25` with Terrier BM25 as the primary control — the same-analyzer comparison is the load-bearing methodology; Terrier BM25 is a supplementary anchor, not a substitute.
