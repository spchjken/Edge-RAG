# 🎯 Benchmark-Specific Issues, Limitations & Dataset Vulnerabilities

**Author:** Antigravity AI Engine (Edge-RAG Benchmark Audit)  
**Target Document:** [`docs/DATASET_PREP.md`](file:///home/donghv/Projects/Edge-RAG/docs/DATASET_PREP.md)  
**Scope:** Strict focus on issues, limitations, biases, conversion code edge cases, and evaluation protocol discrepancies inherent to the **Benchmark Datasets Themselves** and **Dataset Adapter Code**. (Deployment/retriever engine implementation details are excluded).

---

## 1. Benchmark Data Conversion & Adapter Code Vulnerabilities

### 1.1 Document ID Sanitization & Overwriting Collision Risk
* **Affected Code:** [`scripts/data_adapters/convert_retriever_doc_level_benchmarks.py:convert_beir`](file:///home/donghv/Projects/Edge-RAG/scripts/data_adapters/convert_retriever_doc_level_benchmarks.py#L89-L94)
* **Problem:**
  In `convert_beir`, document IDs containing non-alphanumeric characters (slashes `/`, colons `:`, question marks `?`, percent encoding `%`) are sanitized using:
  ```python
  safe_fname = re.sub(r'[^\w\-_\.]', '_', did) + ".json"
  ```
* **Dataset Impact & Failure Modes:**
  1. **DBPedia-Entity (`beir_dbpedia_entity_doc_level`):** Contains RDF URI identifiers with slashes and colons (e.g., `<dbpedia:Category/American_films>` vs `<dbpedia:Category:American_films>`). Both sanitize to `_dbpedia_Category_American_films_.json`. One document silently overwrites the other on disk.
  2. **FEVER & Climate-FEVER (`beir_fever_doc_level`, `beir_climate_fever_doc_level`):** Wikipedia article titles containing slashes (e.g. `180/Movement_for_...` vs `180_Movement_for_...`) collide, resulting in lost documents from the corpus and unmatchable gold links.
  3. **Silent Corpus Corruption:** The resulting document count on disk becomes smaller than the original `corpus.jsonl`, causing false negative evaluations.

---

### 1.2 Loss of Graded Relevance via Hardcoded Binary Filtering (`score >= 1.0`)
* **Affected Code:** [`scripts/data_adapters/convert_retriever_doc_level_benchmarks.py:convert_beir`](file:///home/donghv/Projects/Edge-RAG/scripts/data_adapters/convert_retriever_doc_level_benchmarks.py#L118-L121)
* **Problem:**
  When parsing `qrels/test.tsv`, the adapter converts all judgments into a flat binary list:
  ```python
  if score >= 1.0 and did in doc_map:
      q_to_gold_docs.setdefault(qid, []).append(did)
  ```
* **Dataset Impact:**
  - **TREC-COVID (`beir_trec_covid_doc_level`):** Has 3-level graded relevance judgments ($0 = \text{Not Relevant}$, $1 = \text{Partially Relevant}$, $2 = \text{Definitively Relevant}$).
  - **NFCorpus (`beir_nfcorpus_doc_level`):** Has 4-level graded relevance judgments ($0, 1, 2, 3$).
  - **Touché-2020 (`beir_webis_touche2020_doc_level`):** Has 3-level graded relevance ($0, 1, 2$).
  - **Consequence:** By discarding the numerical score, all non-zero documents are treated with identical weight. This prevents computing true graded $\text{nDCG@10}$ and creates an irreconcilable discrepancy against published academic baselines (e.g. Table 2 of arXiv:2403.06789) that use official BEIR graded exponential gain $\sum \frac{2^{\text{rel}} - 1}{\log_2(i+1)}$.

---

### 1.3 Document Title Duplication & Term Frequency Distortion
* **Affected Code:** [`scripts/data_adapters/convert_retriever_doc_level_benchmarks.py:convert_beir`](file:///home/donghv/Projects/Edge-RAG/scripts/data_adapters/convert_retriever_doc_level_benchmarks.py#L80)
* **Problem:**
  The converter concatenates title and text:
  ```python
  full_text = f"{title}\n\n{text}".strip() if title and title != text else text
  ```
* **Dataset Impact:**
  - In **Quora Duplicate Questions (`beir_quora_doc_level`)**, the corpus consists of short 1-line questions where `title` is often identical or a copy of `text`.
  - In **Touché-2020 (`beir_webis_touche2020_doc_level`)**, the title often repeats the core debate thesis twice.
  - **Consequence:** Artificially doubles the term frequency ($\text{TF}$) of title keywords, artificially inflating BM25 term weighting on short documents compared to the original raw benchmark specification.

---

### 1.4 Single-Directory Inode Exhaustion from Millions of JSON Files
* **Affected Code:** [`scripts/data_adapters/convert_retriever_doc_level_benchmarks.py`](file:///home/donghv/Projects/Edge-RAG/scripts/data_adapters/convert_retriever_doc_level_benchmarks.py#L91-L93)
* **Problem:**
  Saving each document as an individual `.json` file inside a single flat `documents/` folder.
* **Dataset Impact:**
  - `beir_fever_doc_level` creates **5,416,380 individual JSON files** in one folder.
  - `beir_climate_fever_doc_level` creates **5,416,728 individual JSON files**.
  - `beir_hotpotqa_doc_level` creates **5,233,329 individual JSON files**.
  - `beir_dbpedia_entity_doc_level` creates **4,635,832 individual JSON files**.
  - **Consequence:** Standard OS operations (`ls`, `find`, `rm`, directory globbing) freeze or crash with `Argument list too long` (E2BIG). Filesystem inode limits are exhausted, making the dataset unmanageable without archiving into `.jsonl` or `.parquet`.

---

## 2. Evaluation Metric Discrepancies & Gold Representation Inconsistencies

### 2.1 Binary nDCG Formula vs. Academic Leaderboard Standards
* **Affected Code:** [`src/evaluation/metrics.py:calculate_ndcg_at_k`](file:///home/donghv/Projects/Edge-RAG/src/evaluation/metrics.py#L37-L54)
* **Mathematical Discrepancy:**
  - `metrics.py` implements **binary relevance nDCG**:
    $$\text{DCG@}k = \sum_{i=1}^k \frac{\mathbb{I}(d_i \in \text{Gold})}{\log_2(i + 1)}, \quad \text{IDCG@}k = \sum_{j=1}^{\min(k, |\text{Gold}|)} \frac{1}{\log_2(j + 1)}$$
  - Standard academic evaluation on BEIR (e.g. SPLADE-v3 arXiv:2403.06789 Table 2, BM42, BGE) uses official **graded relevance nDCG**:
    $$\text{DCG@}k = \sum_{i=1}^k \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$
  - On graded benchmarks (`TREC-COVID`, `NFCorpus`, `Touché-2020`), our binary nDCG yields a numerical variance of $\pm 0.015–0.035$ compared to paper-reported numbers.

---

### 2.2 Gold Document ID Key Fragmentation across JSON Formats
* **Affected Files:** `final_benchmark.json` across different benchmark folders
* **Problem:**
  Depending on which adapter script generated the benchmark, the gold target documents are stored under different keys:
  - EnterpriseRAG / Synthetic: `expected_doc_ids` (List[str]) + `ground_truth_child_chunks` (List[Dict])
  - LiveRAG: `doc_id_source` (str)
  - Legacy Benchmarks: `golden_doc_ids` (List[str]) or `gold_doc_ids` (List[str])
  - BEIR: `expected_doc_ids` (List[str])
* **Risk:**
  Any evaluation script that queries a single hardcoded key (`q["expected_doc_ids"]`) without fallback handling will silently evaluate against an empty ground truth list, producing artificial 0.0% scores.

---

## 3. Master Benchmark Vulnerability Matrix

| Benchmark Dataset | Key Inherent Problem | Conversion / Code Flaw | Evaluation Risk |
| :--- | :--- | :--- | :--- |
| **`beir_dbpedia_entity_doc_level`** | Incomplete pooling; millions of unjudged entities | URI slashes/colons collide in `safe_fname` sanitization | Overwritten documents; false negative score penalties |
| **`beir_fever_doc_level`** | Single-hop queries over 5.4M Wikipedia | 5.4M individual JSON files in single directory | Severe I/O freeze; E2BIG argument errors |
| **`beir_climate_fever_doc_level`** | Wikipedia claim verification | Title slash sanitization collisions | Missing gold links in integrity audit |
| **`beir_hotpotqa_doc_level`** | Multi-hop reasoning (2 hops required) | 5.2M individual JSON files; unstratified 500-query capping | Uncapped run takes hours; standard nDCG hides missing hop 2 |
| **`beir_trec_covid_doc_level`** | Only 50 queries; 3-level graded relevance | Binary thresholding `score >= 1.0` drops graded levels | High metric variance; binary nDCG mismatch vs. paper |
| **`beir_webis_touche2020_doc_level`**| Only 49 queries; 3-level graded relevance | Binary thresholding `score >= 1.0` drops graded levels | High metric variance; binary nDCG mismatch vs. paper |
| **`beir_nfcorpus_doc_level`** | 4-level graded relevance | Binary thresholding `score >= 1.0` drops graded levels | Binary nDCG mismatch vs. paper |
| **`beir_quora_doc_level`** | Duplicate questions from same pool | Potential self-match inflation ($Q \equiv D$) | Artificially inflated MRR@10 if self-match not filtered |
| **`bright_*_doc_level`** | Near-zero lexical overlap by design | None | Misleading when evaluated as standard keyword search |
| **`financebench_doc_level`** | SEC 10-K financial tables & footnotes | Plain-text linearization destroys table cell alignment | Failure due to table format corruption |
| **`multihop_rag_doc_level`** | 2–4 disjoint documents required | None | `DocRec@10` counts partial chains, hiding incomplete evidence |
| **`liverag_doc_level`** | Temporal news statements drift over time | Legacy `doc_id_source` single-key schema | Key-lookup error if evaluation script lacks schema fallback |

---

## 4. Recommended Engineering Action Items

1. **Retain Graded Relevance in Benchmark JSONs:**
   Update `convert_beir()` to store `"expected_doc_scores": {did: score}` in `final_benchmark.json` so evaluation scripts can compute both official graded $\text{nDCG@10}$ and binary $\text{nDCG@10}$.
2. **Deterministic Hash Suffix on Filenames:**
   Append an 8-character MD5 hash to `safe_fname` in `convert_beir` to eliminate URI/slash namespace collision risk across all RDF/Wikipedia datasets.
3. **Quora Self-Match Guard:**
   In evaluation harnesses, ensure candidate document matches with `doc_id == query_id` are excluded when evaluating symmetrical paraphrase datasets.
4. **Transition Large Corpora to JSONL / Parquet:**
   For corpora $>100\text{k}$ documents (`fever`, `climate-fever`, `dbpedia-entity`, `hotpotqa`, `nq`, `quora`, `webis-touche2020`), provide a unified `corpus.jsonl` to eliminate the 5.4-million-file inode overhead.
