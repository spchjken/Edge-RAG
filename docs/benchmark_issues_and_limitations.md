# 🎯 Benchmark-Specific Issues, Limitations & Dataset Vulnerabilities

**Author:** Edge-RAG Benchmark Audit (revised — code-verified)
**Scope:** Issues, limitations, biases, conversion-code edge cases, and evaluation-protocol discrepancies inherent to the **benchmark datasets themselves** and the **dataset adapter / evaluation code**. (Deployment/retriever-engine implementation details are excluded except where they make the converted corpora un-runnable.)

> **Verification status legend**
> - ✅ **Confirmed** — verified against current source code.
> - ⚠️ **Partially confirmed** — mechanism verified; a dataset-specific example still needs the real downloaded data to confirm.
> - 🔍 **Needs data** — cannot be resolved without the actual `corpus.jsonl` / `qrels`.
>
> **Static-audit note:** `data/` is currently empty in this checkout, so every finding below is verified at **code/logic level**, not against a live downloaded corpus. The two dataset-specific examples marked 🔍 (DBPedia ID format, Quora title) must be re-checked against `data/raw/beir/<dataset>/corpus.jsonl` before relying on them.

---

## 0. Decision-Relevant Summary

These are the issues that change the "should I keep converting / accuracy-vs-efficiency split" decision:

1. **The realism claim is only half true.** "A corpus is many separate documents" is realistic; "5.4M tiny JSON files in one flat directory" is a self-inflicted pathology. The retriever never sees the file structure — the loader just globs and reads every file into RAM (`run_v7_calibration_suite.py::load_doc_corpus`). Realism comes from **document-unit granularity + distractor ratio**, not file count.
2. **Conversion is currently lossy in ways that corrupt *accuracy*** (issues 1.1, 1.2, 1.3). Accuracy must be measured on the raw/official corpus + graded qrels to be comparable to BEIR/SPLADE-v3 Table 2.
3. **Even the scalable methods (BM25/Lucene) cannot load the million-doc converted corpora in 16 GB RAM** because the loader materializes everything (issue 3.1). The bottleneck is the loader, not Lucene.
4. **Dense-BGE / SPLADE baselines are non-viable on million-doc corpora by construction** (issue 3.6) and should not be rewritten; their OOM/timeout is a measured scaling result.

---

## 1. Benchmark Data Conversion & Adapter Code Vulnerabilities

### 1.1 Document ID Sanitization & Overwriting Collision Risk — ⚠️ (mechanism ✅, examples 🔍)
* **Affected code:** `scripts/data_adapters/convert_retriever_doc_level_benchmarks.py::convert_beir` (lines 78, 92)
* **Problem:**
  The filename is derived from the raw `_id` after sanitization:
  ```python
  did = str(rec.get("_id", ""))          # line 78
  safe_fname = re.sub(r'[^\w\-_\.]', '_', did) + ".json"   # line 92
  ```
  `\w` is `[a-zA-Z0-9_]`, so the allowed set is effectively `[a-zA-Z0-9_\-\.]`. **Every other character** (`/ : ? % < > # space` …) collapses to a single `_`.
* **Collision surface (code-verified):**
  - `<a/b>`, `<a:b>`, `<a?b>`, `<a%b>`, `<a b>` all sanitize to `_a_b_.json`.
  - A *legitimate* ID containing a literal underscore (`a_b`) also collides with `a/b` and `a:b`.
  - Therefore two distinct documents silently overwrite each other on disk while `doc_map` (keyed by the raw ID) still counts both → **on-disk document count < in-memory count**.
* **Dataset examples — status:**
  - 🔍 **DBPedia-Entity:** the doc's original claim of RDF URI IDs (`<dbpedia:Category/…>` vs `<dbpedia:Category:…>`) must be confirmed against the downloaded `corpus.jsonl`. If BEIR ships numeric-string `_id`s (as several BEIR datasets do), this example does not apply.
  - 🔍 **FEVER / Climate-FEVER:** the original claim about Wikipedia *titles* with slashes (`180/Movement…`) is **likely incorrect** for the current code — the filename uses `_id` (numeric Wikipedia page id in BEIR FEVER), **not** `title`. Verify `_id` contents before relying on this example.
  - ✅ **General danger is real** for any dataset whose `_id` contains non-`[a-zA-Z0-9_\-\.]` characters.
* **Impact:** silent corpus corruption → false-negative evaluations; on-disk `documents/` set is smaller than `corpus.jsonl`.

---

### 1.2 Loss of Graded Relevance via Hardcoded Binary Filtering — ✅
* **Affected code:** `scripts/data_adapters/convert_retriever_doc_level_benchmarks.py::convert_beir` (line 121)
* **Problem:**
  ```python
  if score >= 1.0 and did in doc_map:
      q_to_gold_docs.setdefault(qid, []).append(did)
  ```
  The numerical judgment is discarded; all `score >= 1.0` docs become equal binary relevance.
* **Affected datasets:** TREC-COVID (3-level), NFCorpus (4-level), Touché-2020 (3-level).
* **Consequences (code-verified):**
  1. True graded nDCG@10 (BEIR exponential gain `2^rel - 1`) becomes impossible → irreconcilable with published baselines (e.g. arXiv:2403.06789 Table 2).
  2. Queries whose only gold docs have `score < 1.0` are **silently dropped** (the loop only iterates keys present in `q_to_gold_docs`).
  3. The `did in doc_map` guard also drops qrel docs absent from the corpus — compounding with 1.1 to produce false negatives.
* **Note:** even if scores were kept, the current evaluation path does not consume graded relevance (see 2.1).

---

### 1.3 Document Title Duplication & Term-Frequency Distortion — ⚠️ (mechanism ✅, Quora example 🔍)
* **Affected code:** `scripts/data_adapters/convert_retriever_doc_level_benchmarks.py::convert_beir` (line 81)
* **Problem:**
  ```python
  full_text = f"{title}\n\n{text}".strip() if title and title != text else text
  ```
  The guard only suppresses concatenation on **exact** equality. When `title` is a meaningful, non-identical copy (different case/whitespace/punctuation), the title is duplicated.
* **Impact:**
  - ✅ **Real for FEVER / NQ / HotpotQA / Climate-FEVER** — Wikipedia titles are meaningful and distinct from body text, so title tokens get ~2× TF, inflating BM25 term weighting on the title words relative to the raw benchmark.
  - 🔍 **Quora** — the original claim depends on the raw corpus having a non-empty `title` that differs from `text`; many BEIR Quora records have an empty/equal title, in which case the guard already skips concatenation. Verify against the downloaded corpus.

---

### 1.4 Single-Directory Inode Exhaustion from Millions of JSON Files — ✅
* **Affected code:** `scripts/data_adapters/convert_retriever_doc_level_benchmarks.py` (lines 92–93)
* **Problem:** every document is written as an individual `.json` file into a single flat `documents/` folder.
* **Scale:** fever ≈ 5.42M files, climate-fever ≈ 5.42M, hotpotqa ≈ 5.23M, dbpedia-entity ≈ 4.64M (from the corpus sizes in `docs/DATASET_PREP.md`).
* **Consequences:**
  - OS ops (`ls`, `find`, `rm`, globbing) freeze / hit E2BIG; inode exhaustion.
  - This is a **deliberate design choice** (the "fragmented files" goal), but it is the *wrong* way to express realism — see §0.1 and issue 3.1.

---

## 2. Evaluation Metric Discrepancies & Gold Representation

### 2.1 Binary nDCG Formula vs. Academic Leaderboard Standards — ✅ (latent)
* **Affected code:** `src/evaluation/metrics.py::calculate_ndcg_at_k` (lines 37–53)
* **Discrepancy:** implements **binary** relevance nDCG (1.0 gain for any gold hit), not the official BEIR graded gain `2^rel - 1`.
* **Additional finding:** the primary benchmark runner `scripts/run_v7_calibration_suite.py::evaluate_v7_config` **does not compute nDCG at all** (it reports strict/complete/doc-rec/precision/MRR only; nDCG is absent from `FIELDNAMES`). So the "±0.015–0.035 vs. paper" variance is currently **latent** — it will only appear once graded nDCG is added. Any BEIR Table-2 comparison requires (a) preserving graded qrels (1.2) and (b) implementing graded nDCG in the harness.

---

### 2.2 Gold Document ID Key Fragmentation across JSON Formats — ✅ (latent risk, partial mitigation)
* **Affected:** query JSON records across benchmark folders.
* **Observed key variants:** `expected_doc_ids` (BEIR/MultiHop/FinanceBench/BRIGHT/EnterpriseRAG/LiveRAG), `ground_truth_child_chunks` (all), `doc_id_source` (all), plus legacy `golden_doc_ids` / `gold_doc_ids` (documented in `DATASET_PREP.md` examples and synthetic pipeline).
* **Current mitigation:** `run_v7_calibration_suite.py::load_doc_queries` falls back `expected_doc_ids → ground_truth_child_chunks` (lines 161–164), so it is safe for the *current* converters but would silently zero-out a `golden_doc_ids`/`gold_doc_ids`-only record.
* **Risk:** any consumer that hardcodes one key without this fallback evaluates against empty ground truth → artificial 0.0%.

---

## 3. Additional Issues Found During Code Audit

### 3.1 Corpus Loader Materializes Everything in RAM (blocks BM25 at scale) — ✅
* **Affected code:** `scripts/run_v7_calibration_suite.py::load_doc_corpus` (lines 109–143)
* **Problem:** `os.listdir` over the `documents/` dir, then `json.load` each file and append the **full text** to `corpus_texts` and a dict to `corpus_docs`. Two parallel in-memory lists hold the entire corpus.
* **Impact:** at 5.4M docs this is tens of GB of host RAM **before Lucene ever indexes anything** → BM25/Lucene, the only scalable retriever in the suite, will OOM in a 16 GB machine purely due to the loader. The fragmentation in 1.4 makes this worse (millions of `json.load` calls + list growth).
* **Fix direction:** stream from a sharded `corpus.jsonl`/`parquet` and hand documents to the indexer incrementally instead of building two full lists.

### 3.2 Integrity Audit Compares Raw IDs to Sanitized Filenames — ✅
* **Affected code:** `scripts/data_adapters/convert_retriever_doc_level_benchmarks.py::validate_all_benchmarks` (lines 689, 697–701)
* **Problem:**
  ```python
  doc_files = set(f.replace(".json", "") for f in os.listdir(docs_dir) if f.endswith(".json"))
  ...
  if did not in doc_files: missing_gold_count += 1
  ```
  `expected_doc_ids` are **raw** IDs; `doc_files` are **sanitized** filenames. For any dataset whose IDs contain `/ : < > ? %`, every gold link is reported missing → **spurious FAILED** even with zero collisions.
* **Impact:** the audit can neither detect real collisions (1.1) nor correctly pass sanitized-ID datasets; it currently masks the 1.1 bug.

### 3.3 No On-Disk Count Parity Check after Conversion — ✅
* **Affected code:** `convert_retriever_doc_level_benchmarks.py::convert_beir`
* **Problem:** the converter prints `len(doc_map)` (in-memory, keyed by raw ID) but never asserts `len(os.listdir(docs_dir)) == len(doc_map)` or checks per-file write success.
* **Impact:** collisions (1.1) overwrite silently while the reported count stays inflated; the mismatch is only discoverable later (and, per 3.2, the audit reports the wrong reason).

### 3.4 `financebench_doc_level` Is Not the Official FinanceBench Corpus — ✅
* **Affected code:** `convert_retriever_doc_level_benchmarks.py::convert_financebench` (line 554, `_build_financebench_distractors`)
* **Problem:** up to 2000 synthetic SEC distractor chunks are injected to avoid a 100%-gold corpus. Design is defensible (true negatives), but it changes the corpus away from the official FinanceBench.
* **Impact:** retrieval accuracy on `financebench_doc_level` **cannot** be compared to any published FinanceBench retrieval baseline. Separate from the table-linearization issue already noted in the master matrix (plain text destroys table cell alignment).

### 3.5 `final_benchmark.json` and `final_benchmark_full.json` Are Identical; Capped File Semantics Confusing — ✅
* **Affected code:** `convert_retriever_doc_level_benchmarks.py::convert_beir` (lines 146–149)
* **Problem:** `final_benchmark.json` and `final_benchmark_full.json` are written with the same content; `final_benchmark_capped.json` is only emitted when the query count exceeds 500.
* **Impact:** "full" is not "more complete than capped"; for ≤500-query datasets there is no capped file. A future consumer can easily pick the wrong file and unknowingly evaluate a subset.

### 3.6 Baseline Scale Limits (dense-BGE & SPLADE) — ✅
* **Affected code:** `src/baselines/dense_rag.py` (line 70), `src/baselines/splade.py` (lines 97–112, 153–190)
* **Dense BGE-small:** `build_index` stores an **fp32 384-dim embedding for every doc** (`np.asarray(self.model.encode_corpus(...))`) plus the full text list. At 5.4M docs ≈ **8.3 GB** of embeddings + text list + query-time full-corpus matmul → borderline/over 16 GB RAM; TTI is hours. VRAM is not the constraint (model is tiny).
* **SPLADE-v3:** `build_index` stores a **Python list of sparse dicts** per doc, and `_compute_scores_manual` performs a **pure-Python O(N) scan per query**. Non-viable (memory *and* minutes-per-query) at millions of docs.
* **Recommendation:** do **not** rewrite these to survive millions of docs — that mutates them away from "off-the-shelf baselines" and breaks comparability. Run them only up to a documented per-method corpus ceiling (or a stratified sample with the same gold docs), and record "did not complete / OOM / TTI > X" beyond that as a measured scaling result.

### 3.7 Environment-Specific Filename Risks (WSL / NTFS) — ⚠️
* **Relevant when** `data/` lives under `/mnt/c` (NTFS) rather than the ext4 WSL root.
* **Risks:** (a) NTFS is case-insensitive → `Foo` vs `foo` collide; (b) Windows reserved device names (`CON`, `PRN`, `NUL`, `COM1`, …) and trailing dots are invalid and can silently fail. The current sanitizer (1.1) handles neither.
* **Impact:** additional silent-overwrite sources on top of 1.1; not an issue on the default ext4 WSL filesystem.

---

## 4. Master Benchmark Vulnerability Matrix

| Benchmark | Inherent problem | Conversion / code flaw | Evaluation risk |
| :--- | :--- | :--- | :--- |
| `beir_dbpedia_entity_doc_level` | Incomplete pooling; millions of unjudged entities | 🔍 URI-vs-numeric ID sanitization collision (1.1); audit false-FAIL (3.2) | Overwritten docs; false negatives |
| `beir_fever_doc_level` | Single-hop over 5.4M Wikipedia | 5.4M flat JSON files (1.4); loader OOM (3.1) | E2BIG; BM25 OOM in 16 GB |
| `beir_climate_fever_doc_level` | Claim verification over Wikipedia | Title-sanitization collision **likely N/A** (filename uses `_id`, 1.1); flat files (1.4) | Same loader issues |
| `beir_hotpotqa_doc_level` | Multi-hop (2 hops) | 5.2M flat files; unstratified 500-query capping | Standard nDCG hides missing hop 2 |
| `beir_trec_covid_doc_level` | 50 queries; 3-level relevance | Binary threshold drops graded levels (1.2) | Binary nDCG mismatch (2.1) |
| `beir_webis_touche2020_doc_level` | 49 queries; 3-level relevance | Binary threshold (1.2) | Binary nDCG mismatch |
| `beir_nfcorpus_doc_level` | 4-level relevance | Binary threshold (1.2) | Binary nDCG mismatch |
| `beir_quora_doc_level` | Duplicate questions from same pool | 🔍 title duplication (1.3) | Self-match inflation if not filtered |
| `bright_*_doc_level` | Near-zero lexical overlap by design | None | Misleading under keyword search |
| `financebench_doc_level` | SEC tables & footnotes | Plain-text linearization destroys tables; distractor injection makes corpus non-official (3.4) | Not peer-comparable |
| `multihop_rag_doc_level` | 2–4 disjoint docs required | None | `DocRec@10` counts partial chains |
| `liverag_doc_level` | Temporal drift | Legacy `doc_id_source` single-key schema | Key-lookup error without fallback |

---

## 5. Recommended Engineering Action Items (prioritized)

**Correctness first (block accuracy):**
1. **Preserve graded relevance** in `convert_beir()`: store `expected_doc_scores: {did: score}` (and keep `expected_doc_ids` for binary consumers) so graded nDCG@10 is computable. (Fixes 1.2, 2.1.)
2. **Collision-safe IDs:** append a deterministic hash (e.g. 8-char MD5) to `safe_fname`, or store docs in a single sharded `corpus.jsonl`/`parquet` keyed by raw ID with a manifest. (Fixes 1.1, 1.4.)
3. **Fix `validate_all_benchmarks`:** compare against the JSON payload `doc_id`, not the sanitized filename; add an on-disk count parity assertion. (Fixes 3.2, 3.3.)

**Scale / efficiency:**
4. **Stream the corpus loader** (`load_doc_corpus`) from `jsonl`/`parquet` instead of materializing two full text lists. (Fixes 3.1 — unblocks BM25 on large corpora within 16 GB.)
5. **Document per-method corpus ceilings** for dense-BGE / SPLADE; record OOM/timeout as scaling results rather than rewriting the baselines. (Fixes 3.6.)

**Methodology / comparability:**
6. **Decouple accuracy from efficiency corpora:** measure accuracy on the raw/official corpus + graded qrels (comparable to BEIR Table 2); measure TTI/latency/VRAM/RAM on a realistic sharded/multi-file layout — same document set, different serialization. (Implements the accuracy-vs-efficiency split.)
7. **Clarify benchmark-file semantics:** make `final_benchmark_full.json` truly full, and emit `final_benchmark_capped.json` deterministically with a documented sampling note. (Fixes 3.5.)
8. **Quora self-match guard:** exclude `doc_id == query_id` in evaluation for symmetric paraphrase datasets.
9. **Decide FinanceBench policy:** either evaluate against the official corpus (drop distractors) for peer comparability, or clearly label the distractor-augmented variant as an Edge-RAG-specific stress set, not "FinanceBench". (Fixes 3.4.)
