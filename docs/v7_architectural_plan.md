# 🏛️ Edge-RAG Pipeline V7 Architectural Plan: Unified Continuous IT-MPE Expansion, Adaptive High-Gate Filtering, and Fast Inverted Retrieval

## Executive Summary

Edge-RAG Pipeline V2 (Schemas 1 through 6b) proved that near-zero VRAM ($0.09\text{ GB}$) lexical-semantic hybrid retrieval can match or surpass heavy neural bi-encoders without the extreme latency or memory footprint of SPLADE-v3 ($174\text{s}$ TTI, $4.2\text{ GB}$ VRAM) or Dense BGE-Large ($17.5\text{s}$ TTI, $2.6\text{ GB}$ VRAM). However, rigorous cross-corpus diagnostic audits across `enterpriserag`, `liverag`, and `fused_stress_500` exposed five key areas for architectural refinement:

1. **Retirement of Discrete Repetition:**
   Integer token repetition ($R \in [2, 5]$) served as an effective initial discrete heuristic, but suffered from step-function quantization noise. V7 replaces all discrete repetition with a single, unified **Continuous IT-MPE Architecture (`BM25Dense_V7`)** operating on sparse float vectors $\mathbf{w} \in \mathbb{R}^{|V|}$.
2. **Adaptive High-Gate Filtering (Replacing Binary Entity Freezing):**
   Hard binary freezing ($\text{IDF} \ge 4.0 \implies \mu = 0$) prevented high-IDF umbrella entities (`"qwen"`, `"EHR"`, `"RAG"`) from expanding to genuine versioned models (`"qwen3.8"`) or full technical names. V7 keeps full expansion capacity for high-weight anchors ($C_{\text{exp}} \ge 2$) while dynamically scaling the similarity threshold:
   $$\tau_{\text{sim}}(a) = 0.80 + 0.10 \times \left(\frac{\text{IDF}(a)}{\text{IDF}_{\text{max}}}\right) \quad \implies \tau_{\text{sim}} \in [0.80, \ \mathbf{0.90}]$$
3. **IDF-Normalized Weight Damping Factor ($\min\left(1.0, \frac{\text{IDF}(a)}{\text{IDF}(s)}\right)$):**
   In BM25, $\text{Score} = w(t) \cdot \text{IDF}(t) \cdot \psi(\text{TF})$. Unchecked rare synonyms with $\text{IDF}(s) \gg \text{IDF}(a)$ (e.g., `erbb2` vs `EHR`) hijacked document scores. Normalizing synonym weights by $\min\left(1.0, \frac{\text{IDF}(a)}{\text{IDF}(s)}\right)$ mathematically guarantees that $\frac{\text{Score}(\text{expan})}{\text{Score}(\text{anchor})} \le 0.35$ in Lucene BM25 scoring.
4. **Standard Lucene Punctuation/Hyphen Tokenization for Academic Text:**
   Aggressive alphanumeric/digit splitting (`"qwen3.8"` $\to$ `["qwen", "3.8"]`) is rejected to prevent flooding scientific corpora with common number noise (`"3.8"`, `"4"`, `"10"`). We use standard Lucene punctuation splitting (hyphens, slashes, dots), keeping alphanumeric entities intact.
5. **Inclusive Vocabulary Rescue (Index-Time High Recall):**
   During index construction, rare 1-document technical singletons and compound words are preserved in the vocabulary pool, relying on query-time $\tau_{\text{sim}} \ge 0.85$ to filter out non-relevant words with high precision.

---

## 1. End-to-End System Architecture

```text
                      ┌──────────────────────────────────────────────┐
                      │             User Query Intent Q              │
                      └──────────────────────┬───────────────────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
          [Primary Anchor Mass]                         [Latent Expansion Mass]
       Explicit User Prior: P=1.0                      Probabilistic Association: P(s|a)
     w(a) = w_base · w_POS · (1 + γ·IDF)               w(s) = μ(Q) · w(a) · min(1, IDF_a/IDF_s) · Softmax
                     │                                               │
                     └───────────────────────┬───────────────────────┘
                                             │
                                             ▼
                     ┌──────────────────────────────────────────────┐
                     │    Mass Invariant: Σ w(s) ≤ μ(Q) · w(a)      │
                     │         with Hard Ceiling μ(Q) ≤ 0.35        │
                     └──────────────────────┬───────────────────────┘
                                            │
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │    Continuous Sparse Vector w ∈ R^|V|        │
                     │    Scored via Lucene Inverted Posting Index  │
                     │                 (<15ms on CPU)               │
                     └──────────────────────────────────────────────┘
```

---

## 2. Mathematical Formalization of Unified V7 (`BM25Dense_V7`)

Let user query $Q = (a_1, a_2, \dots, a_M)$ after stopword removal.

### 2.1 Layer 1: Continuous Anchor Weighting
For each query anchor $a \in Q$:
$$w(a) = \text{Weight}_{\text{POS}}(a) \times \left(1.0 + \gamma \cdot \frac{\text{IDF}(a)}{\max_{t \in Q} \text{IDF}(t)} \cdot \text{Centrality}(a)\right)$$

Where:
* $\text{Weight}_{\text{POS}}(\text{Noun / Entity}) = 1.25, \quad \text{Weight}_{\text{POS}}(\text{Verb}) = 0.85, \quad \text{Weight}_{\text{POS}}(\text{Adjective / Modifier}) = 0.70$.
* $\text{Centrality}(a) = \frac{1}{|Q|-1} \sum_{t \in Q \setminus \{a\}} \cos(\mathbf{e}_a, \mathbf{e}_t)$ (Graph density in BGE embedding space).
* $\gamma = 2.0 \implies w(a) \in [1.0, \ 3.75]$.

---

### 2.2 Layer 2: Adaptive High-Gate & IDF-Damped Synonym Allocation

1. **Adaptive Similarity Gate:**
   $$\tau_{\text{sim}}(a) = 0.80 + 0.10 \times \left(\frac{\text{IDF}(a)}{\text{IDF}_{\text{max}}}\right) \quad \implies \tau_{\text{sim}} \in [0.80, \ \mathbf{0.90}]$$

2. **Query Expansion Budget Ceiling ($\mu(Q) \le 0.35$):**
   $$\mu(Q) = 0.35 \times \left(1 - 0.5 \times \frac{\max_{t \in Q} \text{IDF}(t)}{\text{IDF}_{\text{max\_corpus}}}\right) \quad \implies \mu(Q) \in [0.18, \ \mathbf{0.35}]$$

3. **Continuous Synonym Weight Formula:**
   For candidate vocabulary terms $s_k \in \mathcal{V}_{\text{clean}}$ satisfying $\text{CosSim}(\mathbf{e}_{s_k}, \mathbf{e}_a) \ge \tau_{\text{sim}}(a)$:
   $$w(s_k \mid a) = \mu(Q) \cdot w(a) \cdot \mathbf{\min\left(1.0, \ \frac{\text{IDF}(a)}{\text{IDF}(s_k)}\right)} \cdot \frac{\exp\left(\frac{\text{CosSim}(\mathbf{e}_{s_k}, \mathbf{e}_a)}{\tau}\right)}{\sum_{j=1}^K \exp\left(\frac{\text{CosSim}(\mathbf{e}_{s_j}, \mathbf{e}_a)}{\tau}\right)}$$

   Where:
   * **IDF Damping Factor:** $\min\left(1.0, \frac{\text{IDF}(a)}{\text{IDF}(s_k)}\right)$ cancels out high-IDF inflation from rare synonyms.
   * **Softmax Temperature:** $\tau = 0.10$ exponentially attenuates lower-similarity candidates.
   * **Capacity:** Top $C_{\text{exp}} = 2$ synonyms per anchor.

---

### 2.3 Layer 3: Fast Inverted Posting Index Scoring (<15ms Latency)

In [`src/pipeline_v2/indexer/bm25_lucene_indexer.py`](file:///home/donghv/Projects/Edge-RAG/src/pipeline_v2/indexer/bm25_lucene_indexer.py), the resulting sparse vector $\mathbf{w} \in \mathbb{R}^{|V|}$ is evaluated directly in a single pass:

$$\text{Score}(D, Q) = \sum_{t \in \mathbf{w}} w(t) \cdot \text{IDF}(t) \cdot \frac{\text{TF}(t, D) \cdot (k_1 + 1)}{\text{TF}(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgDL}}\right)}$$

* Traverses pre-computed inverted posting lists (`term \to List[(chunk_id, tf)]`), achieving **$\le 15\text{ms}$ retrieval latency** on 17,241 chunks on CPU.

---

## 3. Vocabulary Indexing & Inclusive Rescue

During index construction (`CorpusVocabBuilder`):
1. **Punctuation Tokenization:** Lowercase, split on standard punctuation (`-`, `_`, `/`, `.`), preserve alphanumeric strings (`"qwen3.8"`, `"gpt4"`, `"ehr-complex"`).
2. **Inclusive Vocabulary Rescue:** Preserves rare 1-document singletons ($DF=1$) and technical compounds in the vocabulary matrix pool, ensuring specific model versions are not pruned.
3. **1-Pass Batch Tensor Probing:** At query time, all anchor embeddings are stacked into 2D tensor $\mathbf{E}_A \in \mathbb{R}^{|A| \times d}$, computing all vocabulary similarities in a single matrix multiplication:
   $$\mathbf{S} = \mathbf{E}_A \cdot \mathbf{V}^{\top} \in \mathbb{R}^{|A| \times |\mathcal{V}|} \quad (\approx 1.2\text{ms})$$

---

## 4. Experimental Matrix & Evaluation Protocol

| Benchmark Corpus | Chunks / Docs | Queries | Primary Baseline to Beat | Target V7 Metric |
| :--- | :---: | :---: | :--- | :--- |
| **`fused_stress_500`** | 17,241 / 500 | 1,084 | Lucene BM25 (78.3% ChunkRec@10 / 92.3% Strict@10) | **$\ge 79.5\%$ ChunkRec@10** |
| **`enterpriserag_doc_level`** | 1,722 full docs | 500 | Lucene BM25 (83.9% DocRec@10 / 86.8% Strict@10) | **$\ge 84.5\%$ DocRec@10** |
| **`liverag_doc_level`** | 970 full docs | 895 | Lucene BM25 (93.9% DocRec@10 / 94.8% Strict@10) | **$\ge 95.5\%$ DocRec@10** |

---

## 5. Implementation Roadmap & Files to Modify

1. **[`src/pipeline_v2/expansion/bm25_dense_aspect_extractor.py`](file:///home/donghv/Projects/Edge-RAG/src/pipeline_v2/expansion/bm25_dense_aspect_extractor.py):**
   * Implement unified `BM25Dense_V7` with continuous IT-MPE weights, POS priors, adaptive similarity gates ($\tau_{\text{sim}} \in [0.80, 0.90]$), and IDF-normalized synonym damping.
2. **[`src/pipeline_v2/indexer/bm25_lucene_indexer.py`](file:///home/donghv/Projects/Edge-RAG/src/pipeline_v2/indexer/bm25_lucene_indexer.py):**
   * Implement continuous float posting list score accumulator (`retrieve_weighted`) supporting `Dict[str, float]`.
   * Implement 1-pass batch matrix probing in `DenseVocabMatrix`.
3. **[`src/pipeline_v2/indexer/corpus_vocab_builder.py`](file:///home/donghv/Projects/Edge-RAG/src/pipeline_v2/indexer/corpus_vocab_builder.py):**
   * Add inclusive index-time vocabulary rescue for 1-document technical singletons and compound tokens.
4. **[`configs/pipeline_v2.yaml`](file:///home/donghv/Projects/Edge-RAG/configs/pipeline_v2.yaml):**
   * Define authoritative V7 continuous hyperparameters (`mu_max: 0.35`, `temperature: 0.10`, `tau_base: 0.80`, `tau_max: 0.90`, `pos_weights`).
5. **[`scripts/run_v7_ablation_sweep.py`](file:///home/donghv/Projects/Edge-RAG/scripts/run_v7_ablation_sweep.py):**
   * Automated multi-corpus evaluation harness benchmarking V7 against Lucene BM25, SPLADE-v3, and Dense BGE baselines.
6. **[`results/v2_ablation/v7_ablation/v7_sweep_summary.md`](file:///home/donghv/Projects/Edge-RAG/results/v2_ablation/v7_ablation/v7_sweep_summary.md):**
   * Consolidated comparative evaluation report with telemetry dashboards and case studies.
