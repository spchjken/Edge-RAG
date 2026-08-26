# Pathway Specification: BM25DenseAspectExtractor (`pathway_bm25_dense_aspect.md`)

## 1. Overview
The `BM25DenseAspectExtractor` replaces slow LLM query expansion with hybrid BM25 Inverse Document Frequency (IDF) anchor selection + Aspect-Grouped BGE dense vocabulary expansion. It eliminates LLM generation latency while providing zero-hallucination, corpus-grounded aspects.

The pathway supports **7 active expansion schemas**:
1. **`BM25Dense_AspectInject` (Primary Baseline Winner / Schema 1):** Force-injects all user query aspect anchors at uniform weight ($3\times$ repetition) + top $C_{\text{exp}}=2$ Dual BGE synonyms per aspect.
2. **`BM25Dense_FixedRepDynamicCapacity` (Schema 5a):** Fixed anchor repetition ($n_{\text{reps}}=3$ or $4$) + dynamic per-aspect synonym capacity ($C_{\text{exp}} = R_{\text{dynamic\_IDF}} + c$). Isolates dynamic synonym budgeting without altering uniform anchor weighting.
3. **`BM25Dense_DynamicAspectInject` (Schema 5b):** Joint dynamic anchor repetition ($R_{\text{anchor}} \in [r_{\min}, r_{\max}]$) scaled via Max Query IDF + coupled synonym capacity ($C_{\text{exp}} = R_{\text{anchor}} + c$).
4. **`BM25Dense_CentralityFixedRep` (Schema 6a):** Query Centrality Anchor Scoring + Stem/Semantic Deduplication + Fix B Entity Validation + Fixed Repetition ($n_{\text{reps}}$) + Zero-Floor Capacity ($C_{\text{exp}} = \max(0, R + c)$).
5. **`BM25Dense_CentralityDynamicInject` (Schema 6b):** Query Centrality Anchor Scoring + Stem/Semantic Deduplication + Fix B Entity Validation + Max Query IDF Dynamic Repetition + Zero-Floor Capacity ($C_{\text{exp}} = \max(0, R + c)$).
6. **`BM25Dense_AspectWeighted` (Ablation Baseline):** Scales aspect anchor weights dynamically by relative IDF score ($w \in [0.5, 1.0]$) + soft-dampened synonyms.
7. **`BM25Dense_AspectFusion` (HAC Clustering Variant):** Applies Hierarchical Agglomerative Clustering (HAC, distance threshold $d=0.35$) on anchor embeddings + joint BM25 IDF and BGE similarity score fusion.

*(Note: Legacy Schema 3 `BM25Dense_LocalCascade` has been deprecated and removed).*

---

## 2. Core Mathematical Formulation

### 2.1 Sublinear Salience Vocabulary Extraction (`CorpusVocabBuilder`)
To populate the clean 1,000-word candidate vocabulary $\mathcal{V}_{\text{clean}}$ with high-value domain concepts rather than ultra-rare typos or generic stopwords:
1. **Upper Frequency Ceiling:** $\text{Doc\_Freq}(t) \le 0.15 \times N_{\text{docs}}$ (filters out corpus-level generic stopwords like `"data"`, `"system"`, `"model"`).
2. **Lower Frequency Floor:** $\text{Doc\_Freq}(t) \ge 2$ and $\text{IDF}(t) \ge 1.5$.
3. **Sublinear Salience Ranking:**
   $$\text{Salience}(t) = \text{IDF}(t) \times \ln\left(1 + \text{Doc\_Freq}(t)\right)$$
   The top 1,000 highest-salience unigrams and bigrams form $\mathcal{V}_{\text{clean}}$.

### 2.2 Aspect Anchor Selection & Centrality Scoring (Schema 6)
In Schema 6, candidate query words are scored by their **Semantic Centrality** to the full query vector $\mathbf{e}_{Q_{\text{full}}}$:
$$\text{Centrality\_Score}(w) = \text{IDF}(w) \times \text{CosSim}\left(\mathbf{e}_w, \mathbf{e}_{Q_{\text{full}}}\right)$$

- **Stem & Semantic Deduplication:** Discards morphological duplicates (e.g. `upload` vs `uploads` with stem overlap or $\text{CosSim} \ge 0.90$) to rescue unrepresented intent keywords.
- **Fix B (Validated Entity Check):** Acronyms and heuristic expressions receive entity boosts ($W = r_{\max}$) when $\text{IDF}(w) \ge 1.0$ (allowing domain-essential acronyms such as `API`, `USD`, `SDK` up to 36.8% corpus prevalence while filtering universal conversational stopwords).

### 2.3 Dual BGE Semantic Similarity Probing
For each Aspect Anchor $A_k$, cosine similarity is computed against candidate vocabulary terms $v \in \mathcal{V}_{\text{clean}}$ and the full query $Q_{\text{full}}$:
$$\text{Dual\_Sim}(A_k, v) = \beta \cdot \text{CosSim}(A_k, v) + (1 - \beta) \cdot \text{CosSim}(Q_{\text{full}}, v)$$
where default $\beta = 0.65$, with threshold $\tau_{\text{sim}} = 0.55$.

### 2.4 Schema 5a: Fixed Repetition & Dynamic Capacity Capping
* **Anchor Weight:** Fixed at $W_{\text{anchor}} = n_{\text{reps}}$ ($3.0$ or $4.0$) for all anchors.
* **Dynamic Aspect Capacity:**
  $$R_{\text{dynamic\_IDF}}(A_k) = \text{clamp}\left( \text{round}\left(r_{\min} + (r_{\max} - r_{\min}) \times \frac{\text{IDF}(A_k)}{\text{Max\_Query\_IDF}}\right), r_{\min}, r_{\max} \right)$$
  $$C_{\text{exp}}(A_k) = \text{clamp}\left(R_{\text{dynamic\_IDF}}(A_k) + c, 1, 5\right)$$
  where $\text{Max\_Query\_IDF} = \max_{a \in \text{anchors}} \text{IDF}(a)$.

### 2.5 Schema 5b: Dynamic Repetition & Coupled Capacity Capping
* **Dynamic Anchor Weight:**
  $$W_{\text{anchor}}(A_k) = \begin{cases} r_{\max} & \text{if } A_k \text{ is a Heuristic Entity} \\ \text{clamp}\left( \text{round}\left(r_{\min} + (r_{\max} - r_{\min}) \times \frac{\text{IDF}(A_k)}{\text{Max\_Query\_IDF}}\right), r_{\min}, r_{\max} \right) & \text{otherwise} \end{cases}$$
* **Coupled Synonym Capacity:**
  $$C_{\text{exp}}(A_k) = \text{clamp}\left(W_{\text{anchor}}(A_k) + c, 1, 5\right)$$

### 2.6 Schema 6a & 6b: Centrality-Gated & Diversity-Pruned Expansion
* **Schema 6a (`BM25Dense_CentralityFixedRep`):** Fixed $W_{\text{anchor}} = n_{\text{reps}}$ ($3.0$ or $4.0$) + Zero-Floor Capacity $C_{\text{exp}} = \max(0, R_{\text{dyn}} + c)$.
* **Schema 6b (`BM25Dense_CentralityDynamicInject`):** Validated Entity Boost + Dynamic Anchor Weight $W(A_k) \in [r_{\min}, r_{\max}]$ + Zero-Floor Capacity $C_{\text{exp}} = \max(0, W(A_k) + c)$.

---

## 3. Direct Weighted Term Vector Retrieval ($\vec{w}_Q$)

Rather than quantizing weights into repetitive string tokens ($Q_{\text{aug}}$), the extractor compiles a direct sparse weight vector $\vec{w}_Q = \{t: W_Q(t)\}$ evaluated in a single vectorized pass by `BM25LuceneIndexer.retrieve_weighted()`:

$$\text{Score}(D, Q) = \sum_{t \in \text{keys}(\vec{w}_Q)} W_Q(t) \cdot \text{IDF}(t) \cdot \frac{\text{TF}(t, D) \cdot (k_1 + 1)}{\text{TF}(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgDL}}\right)}$$

### Term Weight Assignment & Synonym Capping Rules:
1. **Core Anchors:** Retain full anchor weight multiplier $W_Q(A_k) = W_{\text{anchor}}(A_k) \in [2.0, 5.0]$.
2. **Expansion Synonyms:** Assigned their continuous semantic final weight:
   $$\text{final\_weight}(v) = \text{Dual\_Sim}(A_k, v) \times \left(0.5 + 0.5 \times \frac{\text{IDF}(v)}{\text{IDF}_{\max}}\right)$$
   **Strict Capping Rule:** Across all aspect groups, the accumulated weight of any injected synonym is strictly capped at $W_Q(v) \le 1.0$, guaranteeing that synthetic expansions never outweigh core user query terms.

---

## 4. Single-Pass Indexing & TTI Performance Optimization

During index creation (Time-To-Index, TTI):
1. **Shared Lucene IDF Pass:** `BM25LuceneIndexer` builds the inverted posting lists and extracts document frequencies (`nd`). `CorpusIDFRegistry` reuses `nd` directly (**0.001s** vs 5.0s).
2. **Sublinear Salience Vocab Extraction:** `CorpusVocabBuilder` samples corpus chunks and ranks unigrams/bigrams by $\text{IDF} \times \ln(1 + \text{DF})$ (**0.03s** vs 5.2s).
3. **Warm Batch Embedding:** `DenseVocabMatrix` embeds the 1,000 clean vocabulary terms in 1 GPU batch call on CUDA FP16 (**0.45ms** execution time).

---

## 5. Inputs, Outputs & Telemetry Contract

- **Input:** 
  - `query`: Raw user query string.
  - `idf_registry`: Precomputed `CorpusIDFRegistry`.
  - `vocab_matrix`: Embedded `DenseVocabMatrix`.
  - `schema`: `"BM25Dense_AspectInject"` | `"BM25Dense_FixedRepDynamicCapacity"` | `"BM25Dense_DynamicAspectInject"` | `"BM25Dense_CentralityFixedRep"` | `"BM25Dense_CentralityDynamicInject"`.
- **Output Dict:**
  - `"aspects"`: Array of structured aspect groups with weighted keywords.
  - `"term_weights"`: Sparse mapping of unique query tokens & synonyms to weight multipliers $\{t: W_Q(t)\}$.
  - `"telemetry"`: Diagnostic metadata (`num_anchors`, `total_candidates_above_tau`, `total_synonyms_injected`, `starved_aspects_count`, `avg_r_anchor`, `aspect_traces`).
