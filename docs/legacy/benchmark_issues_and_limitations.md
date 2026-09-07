# 🎯 Benchmark-Specific Issues, Limitations & Dataset Vulnerabilities

**Author:** Edge-RAG Benchmark Audit (revised — code-verified)
**Scope:** Issues, limitations, biases, conversion-code edge cases, and evaluation-protocol discrepancies inherent to the **benchmark datasets themselves** and the **dataset adapter / evaluation code**. (Deployment/retriever-engine implementation details are excluded except where they make the converted corpora un-runnable.)

> **Verification status legend**
> - ✅ **Confirmed** — verified against source code and, where noted, against the live downloaded corpus in `data/raw/beir/`.
> - ⚠️ **Conditional / environment-dependent** — holds only under the stated condition (e.g. NTFS filesystem).
>
> **Audit note:** the dataset-specific claims in §1.1 and §1.3 (DBPedia ID format, FEVER/Climate-FEVER slash IDs, Quora empty title) were resolved by direct inspection of `data/raw/beir/<dataset>/corpus.jsonl` on the full benchmark environment.

---

## 0. Decision-Relevant Summary

These are the issues that change the "should I keep converting / accuracy-vs-efficiency split" decision:

1. **The realism claim is only half true.** "A corpus is many separate documents" is realistic; "5.4M tiny JSON files in one flat directory" is a self-inflicted pathology. The retriever never sees the file structure — the loader just globs and reads every file into RAM (`load_doc_corpus` in both `run_v7_vs_baselines_comparison.py` and `run_v7_calibration_suite.py`). Realism comes from **document-unit granularity + distractor ratio**, not file count.
2. **Conversion is currently lossy in ways that corrupt *accuracy*** (issues 1.1, 1.2, 1.3). Accuracy must be measured on the raw/official corpus + graded qrels to be comparable to BEIR/SPLADE-v3 Table 2.
3. **Even the scalable methods (BM25/Lucene) cannot load the million-doc converted corpora in 16 GB RAM** because the loader materializes everything (issue 3.1). The bottleneck is the loader, not Lucene.
4. **Dense-BGE / SPLADE baselines are non-viable on million-doc corpora by construction** (issue 3.6) and should not be rewritten; their OOM/timeout is a measured scaling result.

---

## 1. Benchmark Data Conversion & Adapter Code Vulnerabilities

### 1.1 Document ID Sanitization & Overwriting Collision Risk — ✅
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
* **Dataset examples — verified against live data:**
  - ✅ **DBPedia-Entity:** `_id` values are **RDF URIs** with angle brackets, colons, and parentheses — e.g. `<dbpedia:Animalia_(book)>`, `<dbpedia:Academy_Award_…>`. These sanitize to forms like `_dbpedia_Animalia__book_.json`, so distinct URIs collapse together after sanitization.
  - ✅ **FEVER / Climate-FEVER:** `_id` is the **Wikipedia title slug**, not a numeric page id — e.g. `180/Movement_for_Democracy_and_Education`. The `/` sanitizes to `_`, so it collides with any `180_Movement_for_Democracy_and_Education`-style ID (silent overwrite).
  - ✅ **Related hard-failure hazard:** any writer that emits `did + ".json"` *without* sanitization raises `FileNotFoundError` when `did` contains `/` (open() targets a non-existent subfolder like `180/`); this was observed live during conversion on the full environment.
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
* **Note:** even if scores were kept, the current harness consumes only binary relevance — the reported `ndcg_10` is binary (see 2.1).

---

### 1.3 Document Title Duplication & Term-Frequency Distortion — ✅
* **Affected code:** `scripts/data_adapters/convert_retriever_doc_level_benchmarks.py::convert_beir` (line 81)
* **Problem:**
  ```python
  full_text = f"{title}\n\n{text}".strip() if title and title != text else text
  ```
  The guard only suppresses concatenation on **exact** equality. When `title` is a meaningful, non-identical copy (different case/whitespace/punctuation), the title is duplicated.
* **Impact:**
  - ✅ **Real for FEVER / NQ / HotpotQA / Climate-FEVER** — Wikipedia titles are meaningful and distinct from body text, so title tokens get ~2× TF, inflating BM25 term weighting on the title words relative to the raw benchmark.
  - ✅ **Quora is exempt (verified):** raw BEIR Quora records have `title: ""` (empty string), so `if title and title != text` is `False` and concatenation is skipped. No TF distortion on Quora — the distortion applies only to Wikipedia-backed corpora (FEVER / NQ / HotpotQA / Climate-FEVER).

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

### 2.1 Binary nDCG Formula vs. Academic Leaderboard Standards — ✅ (active)
* **Affected code:** `src/evaluation/metrics.py::calculate_ndcg_at_k` (lines 37–53)
* **Discrepancy:** implements **binary** relevance nDCG (1.0 gain for any gold hit), not the official BEIR graded gain `2^rel - 1`.
* **Status — active, not latent:** the comparative runner `scripts/run_v7_vs_baselines_comparison.py` **does** compute and report `ndcg_10` (present in `results/v7_vs_baselines/v7_vs_baselines_results.csv`), but it uses the **binary** relevance from `metrics.py` — so the ±0.015–0.035 variance vs. official graded BEIR numbers is already baked into reported results (e.g. `beir_nfcorpus_doc_level`). The older `scripts/run_v7_calibration_suite.py` omits nDCG entirely, which is why a quick grep of that file alone is misleading. Any BEIR Table-2 comparison requires (a) preserving graded qrels (1.2) and (b) a graded-nDCG path in the harness.

---

### 2.2 Gold Document ID Key Fragmentation across JSON Formats — ✅ (latent risk, partial mitigation)
* **Affected:** query JSON records across benchmark folders.
* **Observed key variants:** `expected_doc_ids` (BEIR/MultiHop/FinanceBench/BRIGHT/EnterpriseRAG/LiveRAG), `ground_truth_child_chunks` (all), `doc_id_source` (all), plus legacy `golden_doc_ids` / `gold_doc_ids` (documented in `DATASET_PREP.md` examples and synthetic pipeline).
* **Current mitigation:** `run_v7_calibration_suite.py::load_doc_queries` falls back `expected_doc_ids → ground_truth_child_chunks` (lines 161–164), so it is safe for the *current* converters but would silently zero-out a `golden_doc_ids`/`gold_doc_ids`-only record.
* **Risk:** any consumer that hardcodes one key without this fallback evaluates against empty ground truth → artificial 0.0%.

---

## 3. Additional Issues Found During Code Audit

### 3.1 Corpus Loader Materializes Everything in RAM (blocks BM25 at scale) — ✅
* **Affected code:** `scripts/run_v7_vs_baselines_comparison.py::load_doc_corpus` and `scripts/run_v7_calibration_suite.py::load_doc_corpus` (lines 109–143 in the latter) — both share the same materialize-everything pattern.
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
* **SPLADE-v3:** `build_index` stores a **Python list of sparse dicts** per doc, and `_compute_scores_manual` performs a **pure-Python O(N) scan per query**. Non-viable (memory *and* minutes-per-query) at millions of docs. **Fairness caveat:** published SPLADE results (arXiv:2403.06789) use a C++ inverted-index engine (Pisa / Anserini/Lucene) with WAND/MaxScore posting traversal, so their query latency is comparable to Lucene. Our `src/baselines/splade.py` is a non-inverted Python reference implementation — its latency/TTI figures reflect *that* implementation, not an inverted-index SPLADE deployment.
* **Recommendation:** do **not** rewrite these to survive millions of docs — that mutates them away from "off-the-shelf baselines" and breaks comparability. Run them only up to a documented per-method corpus ceiling (or a stratified sample with the same gold docs), and record "did not complete / OOM / TTI > X" beyond that as a measured scaling result.

### 3.7 Environment-Specific Filename Risks (WSL / NTFS) — ⚠️
* **Relevant when** `data/` lives under `/mnt/c` (NTFS) rather than the ext4 WSL root.
* **Risks:** (a) NTFS is case-insensitive → `Foo` vs `foo` collide; (b) Windows reserved device names (`CON`, `PRN`, `NUL`, `COM1`, …) and trailing dots are invalid and can silently fail. The current sanitizer (1.1) handles neither.
* **Impact:** additional silent-overwrite sources on top of 1.1; not an issue on the default ext4 WSL filesystem.

---

## 4. Master Benchmark Vulnerability Matrix

| Benchmark | Inherent problem | Conversion / code flaw | Evaluation risk |
| :--- | :--- | :--- | :--- |
| `beir_dbpedia_entity_doc_level` | Incomplete pooling; millions of unjudged entities | ✅ RDF-URI `_id` sanitization collision (1.1); audit false-FAIL (3.2) | Overwritten docs; false negatives |
| `beir_fever_doc_level` | Single-hop over 5.4M Wikipedia | ✅ slash `_id` sanitization collision (1.1); 5.4M flat JSON files (1.4); loader OOM (3.1) | E2BIG; overwritten docs; BM25 OOM in 16 GB |
| `beir_climate_fever_doc_level` | Claim verification over Wikipedia | ✅ slash `_id` sanitization collision (1.1); flat files (1.4) | Overwritten docs; same loader issues |
| `beir_hotpotqa_doc_level` | Multi-hop (2 hops) | 5.2M flat files; unstratified 500-query capping | Standard nDCG hides missing hop 2 |
| `beir_trec_covid_doc_level` | 50 queries; 3-level relevance | Binary threshold drops graded levels (1.2) | Binary nDCG mismatch (2.1) |
| `beir_webis_touche2020_doc_level` | 49 queries; 3-level relevance | Binary threshold (1.2) | Binary nDCG mismatch |
| `beir_nfcorpus_doc_level` | 4-level relevance | Binary threshold (1.2) | Binary nDCG mismatch |
| `beir_quora_doc_level` | Duplicate questions from same pool | ✅ title duplication does **not** apply (empty `title`, 1.3) | Self-match inflation if not filtered |
| `bright_*_doc_level` | Near-zero lexical overlap by design | None | Misleading under keyword search |
| `financebench_doc_level` | SEC tables & footnotes | Plain-text linearization destroys tables; distractor injection makes corpus non-official (3.4) | Not peer-comparable |
| `multihop_rag_doc_level` | 2–4 disjoint docs required | None | `DocRec@10` counts partial chains |
| `liverag_doc_level` | Temporal drift | Legacy `doc_id_source` single-key schema | Key-lookup error without fallback |

---

## 5. Recommended Engineering Action Items & Implementation Resolution Status

### ✅ Implemented & Resolved (2026-09-04 Full Re-evaluation)
1. **Preserve Graded Relevance & Official nDCG@10:**
   - Implemented BEIR/TREC exponential gain graded formula $\sum \frac{2^{\text{rel}} - 1}{\log_2(i + 1)}$ with explicit descending IDCG sort in [`src/evaluation/metrics.py`](file:///home/donghv/Projects/Edge-RAG/src/evaluation/metrics.py).
   - Preserved full continuous/graded qrel tables in [`src/evaluation/benchmark_loader.py`](file:///home/donghv/Projects/Edge-RAG/src/evaluation/benchmark_loader.py) and wired directly to the benchmark runner.
2. **Direct Raw Streaming (Accuracy Decoupled from File-Structure Artifacts):**
   - Implemented [`BenchmarkLoader`](file:///home/donghv/Projects/Edge-RAG/src/evaluation/benchmark_loader.py) streaming directly from raw BEIR `corpus.jsonl`, BRIGHT parquets, MultiHop-RAG JSONs, and LiveRAG parquets without flat-file filesystem fragmentation or filename collisions.
   - Standardized document text to BEIR standard: `f"{title} {text}".strip() if title else text`.
3. **Full Test Set Accounting:**
   - Evaluated all un-capped test queries: FiQA (648 queries), MultiHop-RAG (2,255 queries with news evidence), SciFact (300 queries), NFCorpus (323 queries), BRIGHT (Economics 103, StackOverflow 117, Robotics 101), FinanceBench (150 queries), EnterpriseRAG (470 valid queries), LiveRAG (895 queries).
4. **FinanceBench Peer Comparability:**
   - Official SEC filing evidence pages loaded directly from `financebench_train.jsonl` without synthetic distractor chunks.
5. **SPLADE-v3 Standardized Sparse Neural Retrieval:**
   - Implemented in [`src/baselines/splade.py`](file:///home/donghv/Projects/Edge-RAG/src/baselines/splade.py) with canonical HuggingFace encoder (`naver/splade-v3-distilbert`), symmetric document & query attention masking, special token exclusion, batched FP16 inference, and in-memory compact inverted index.
   - Latency dropped from ~500ms sequential loop to **9.38 ms** average query latency with **0.4667** macro nDCG@10.
6. **Lucene BM25 (Unstemmed Standard):**
   - Implemented standard regex word tokenization stripping punctuation without stemming, renaming the baseline to **`BM25 (Standard Lucene, unstemmed)`** for citable clarity.

---

## 6. Remaining Future Enhancements
- **Quora self-match guard:** Exclude `doc_id == query_id` if evaluating symmetrical paraphrase datasets.
- **Shard-level partitioning:** When benchmarking million-document corpora (such as full FEVER 5.4M), use memory-mapped inverted indexes or SQLite/posting shards to avoid 16 GB RAM limits.

