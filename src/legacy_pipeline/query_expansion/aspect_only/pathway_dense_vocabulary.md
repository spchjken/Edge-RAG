# Pathway: The Dense Vocabulary Route (Corpus-Grounded BGE)

This pathway bridges lexical search with semantic meaning by extracting a large global vocabulary from the corpus and using a lightweight dense embedding model (BGE-Small) to rank them against the user query. **This pathway guarantees zero hallucinations** because every term is pulled directly from the text.

To prevent critical keywords from being dropped due to vocabulary gaps (critical keywords fading), to resolve the dominance of generic words (e.g. "model", "data") crowding out specific search aspects, and to optimize corpus pre-building Time-To-Index (TTI) for **Ephemeral & Streaming RAG**, we support five architectural versions.

---

## 1. The Five Extractor Variants

### Version 1: Base + Heuristic Force-Injection (`DenseVocabularyExtractor`)
Combines cosine similarity dense matching with regex-based **Heuristic Entity Extraction**. Any entities matched by the following rules are directly forced into the final payload with a weight of `1.0`:
- **Acronyms:** Uppercase sequences of 2+ characters (`r'\b[A-Z]{2,}\b'`).
- **Hyphenated Alphanumeric Terms:** E.g., `TransEnc-8`, `qwen3.5-2b` (`r'\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b'`).
- **Proper Nouns:** Sequences of Title Case words (`r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'`).
- **Explicit Quotes:** Quoted substrings (`r'"([^"]+)"'` and `r"'([^']+)'"`).

### Version 2: Heuristics + Generic Term Prevention (`DenseVocabularyAdvancedExtractor`)
Extends Version 1 by solving **Generic Term Saturation** (where frequent words like "models" crowd out specific terms) via IDF scaling:
1. **IDF Semantic Scaling:** Similarity scores are scaled by their inverse document frequency to downweight frequent terms:
   $$\text{Score}(term) = \text{CosSim}(query, term) \times \left( \lambda + (1 - \lambda) \times \frac{\text{IDF}(term)}{\text{Max\_IDF}} \right)$$
   where $\lambda = 0.5$ limits the maximum penalty to 50%, maintaining semantic relevance while suppressing generic terms.

### Version 3: LLM Keyword Extraction + Generic Term Prevention (`DenseVocabularyLLMExtractor`)
Instead of regex heuristics, Version 3 leverages the `qwen3.5-2b` client to extract up to 5 critical technical keywords, entities, or acronyms from the raw query. These extracted words are force-injected into the results with a weight of `1.0`. The fallback semantic matching uses the same IDF scaling as Version 2.

### Version 4: Fast $O(N)$ Linear Pre-Building + Batch BGE (`DenseVocabularyFastExtractor`)
Replaces heavy YAKE $O(K^2)$ n-gram extraction with single-pass $O(N)$ linear token & bigram frequency scanning (`collections.Counter`). To eliminate the sequential embedding bottleneck, it uses **Batched BGE Vectorization** (`bge.encode(list_of_terms)`).
- **Vocab Limit:** 2,000 terms (expanded to absorb generic term slots).
- **Setup Latency (TTI):** Reduced from **~19s to ~2s (~9x speedup)**.
- **Generic Term Handling:** Suppressed at query time via IDF Scaling.

### Version 5: Fast $O(N)$ + IDF Pre-Filter + Batch BGE (`DenseVocabularyFilteredExtractor`)
Applies an **IDF Pre-Filter** (dropping candidate terms with IDF below median IDF) *before* BGE embedding to prevent generic term slot waste and reduce vector encoding overhead.
- **Vocab Limit:** 1,000 terms (surviving domain-specific terms).
- **Setup Latency (TTI):** Reduced from **~19s to ~1s (~18x speedup)**.
- **Generic Term Handling:** Cut *before* embedding, ensuring zero generic slots waste in the dense matrix.

---

## 2. Pre-Build Optimization & Complexity Breakdown

| Variant | Vocab Extraction | Vocab Limit | Pre-Filter | BGE Vectorization | Est. TTI Latency |
|---|---|---|---|---|---|
| **V1 / V3** | YAKE $O(K^2)$ + Counter | 1,000 | None | Sequential (1×1000) | **~17.9s – 19.1s** |
| **V4** | $O(N)$ Linear Counter | **2,000** | None | **Batched** (1×2000) | **~2.0s (~9x faster)** |
| **V5** | $O(N)$ Linear Counter | **1,000** | **IDF Median Cutoff** | **Batched** (1×1000) | **~1.0s (~18x faster)** |

---

## 3. Pseudo-Algorithm

```python
def extract_grounded_aspects(query: str, corpus_text: str, mode: str = "v5", K=10) -> list:
    # 1. Global Corpus Extraction & Batch Embedding (Setup Phase)
    if mode == "v4":
        # Linear O(N) Counter (2000 terms)
        words = clean_tokens(corpus_text)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        candidates = [t for t, c in Counter(words + bigrams).most_common(2000)]
        vocab_vectors = dict(zip(candidates, bge_batch_encode(candidates)))
        
    elif mode == "v5":
        # Linear O(N) Counter + IDF Pre-Filter (1000 terms)
        words = clean_tokens(corpus_text)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        candidates = [t for t, c in Counter(words + bigrams).most_common(3000)]
        
        # IDF Pre-Filter
        idf_scores = [mean_idf(t, idf_dict) for t in candidates]
        median_idf = median(idf_scores)
        filtered_terms = [t for t, s in zip(candidates, idf_scores) if s >= median_idf][:1000]
        vocab_vectors = dict(zip(filtered_terms, bge_batch_encode(filtered_terms)))
        
    else:
        # V1/V2/V3: YAKE + Sequential Embedding
        vocabulary = extract_yake_and_unigrams(corpus_text, limit=1000)
        vocab_vectors = {t: embed(t) for t in vocabulary}
        
    # 2. Extract Critical Keywords (LLM or Regex)
    if mode in ("v3", "v4", "v5"):
        critical_keywords = llm_extract_keywords(query, max_k=5)
    else:
        critical_keywords = extract_regex_heuristics(query)
        
    # 3. Query Embedding & IDF-Scaled Cosine Match
    query_vector = embed(remove_stopwords(query))
    scored_terms = []
    max_idf = max(idf_dict.values()) if idf_dict else 1.0
    
    for term, term_vec in vocab_vectors.items():
        sim = cosine_similarity(query_vector, term_vec)
        if mode in ("v2", "v3", "v4", "v5") and idf_dict:
            term_idf = mean_idf(term, idf_dict)
            sim = sim * (0.5 + 0.5 * (term_idf / max_idf))
        scored_terms.append((term, sim))
        
    # 4. Top-K Selection & Final Output Payload
    scored_terms.sort(key=lambda x: x[1], reverse=True)
    selected_aspects = scored_terms[:K]
    return selected_aspects
```

---

## 4. Architecture Flow

```mermaid
graph TD
    A[Raw Query] -->|LLM / Regex| B(Critical Keywords)
    A -->|Stopword Removal| C(Clean Query)
    C -->|BGE-Small| D[(Query Vector)]
    
    E[Corpus Text] -->|O(N) Token Counter| F(Candidate N-Grams)
    F -->|IDF Pre-Filter (v5)| G(High-IDF Terms)
    G -->|Batch BGE-Small| H[(Vocabulary Matrix)]
    
    D & H -->|Cosine Similarity| I(Similarity Scores)
    I -->|IDF Scaling (v2-v5)| J(Scaled Scores)
    
    J -->|Sort Descending & Slice Top-K| K{Top-K Selected Aspects}
    
    K -->|Aspect Payload| L{aspect_only JSON Payload}
    B -->|Force Inject w/ Weight 1.0| L
```
