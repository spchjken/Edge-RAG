# Ground Truth Evaluation: Keyword Similarity & Evidence Recall

We evaluated the `pipeline_run_20260713_044533.json` against the ground truth dataset (`cluster_1_queries.json`).

First, we injected the **exact text blocks** from the source papers directly into `cluster_1_queries.json` to act as the raw evidence for string overlap evaluation. We then evaluated two metrics:
1. **Keyword Match Score**: What percentage of the ground truth `oracle_aspect` names were captured by the pipeline's query expander.
2. **Evidence Recall**: What percentage of the queries successfully retrieved a chunk that had >30% text overlap with the exact ground truth evidence.

## Results Table

| Combination | Keyword Match Score | Evidence Recall |
|-------------|---------------------|-----------------|
| SimilarKW_Statistical_IDF_Adaptive | 0.00% | 100.00% |
| SimilarKW_Statistical_IDF_Cascade | 0.00% | 100.00% |
| AspectOnly_YAKE_Cascade | 6.67% | 60.00% |
| AspectOnly_Dense_Vocab_Cascade | 6.67% | 60.00% |
| SimilarKW_Vector_Projection_Cascade | 0.00% | 60.00% |
| AspectOnly_Dense_Vocab_Adaptive | 6.67% | 40.00% |
| SimilarKW_LLM_Query_Expander_Cascade | 0.00% | 40.00% |
| AspectOnly_LLM_Aspect_Adaptive | 11.67% | 20.00% |
| AspectOnly_LLM_Aspect_Cascade | 11.67% | 20.00% |
| AspectOnly_YAKE_Adaptive | 6.67% | 20.00% |
| SimilarKW_LLM_Query_Expander_Adaptive | 0.00% | 20.00% |
| SimilarKW_Vector_Projection_Adaptive | 0.00% | 20.00% |

---

## 1. Keyword Match Analysis

**The LLM Aspect Extractor was the most "accurate", but failed structurally.**
`AspectOnly_LLM_Aspect` scored the highest (11.67%) in generating exact keywords that matched the ground truth aspects. However, an analysis of the JSON reveals that the 2B parameter LLM (`Qwen3.5-2b`) **completely ignored** the prompt constraint to output natural spaces. It still outputted snake_case entities like `deployment_adjusted_rank_1_acceptability_value`. 
Because it forced these variables into a single concatenated string, Lexical Search failed to match the natural English text in the documents, resulting in extremely poor Evidence Recall.

**SimilarKW Methods score 0% but it's a structural artifact.**
`SimilarKW` approaches intentionally explode the query into a massive list of individual related terms rather than high-level "Aspects" (like the Oracle dataset expects). Thus, they didn't match the Oracle strings perfectly, leading to a 0% match score, even though the words themselves were highly effective for retrieval.

---

## 2. Evidence Recall (Snippet Overlap) Analysis

**Statistical IDF Brute-Forced 100% Recall**
`SimilarKW_Statistical_IDF` achieved a perfect 100% evidence recall. How? By brute-forcing the retriever. Because Statistical IDF extracts a huge amount of loose, high-frequency keywords, it pulled almost **100 chunks** per query into the Rerank Queue. With that much noise, it was guaranteed to eventually pull the correct chunk containing the evidence, but as we saw in the previous analysis, this took **11 seconds** of latency.

**Cascade Router Allowed More Hits than Adaptive**
Notice that for almost every extractor type, `Cascade` scored higher recall than `Adaptive`. 
- `AspectOnly_Dense_Vocab_Cascade` hit 60% recall, while `Adaptive` only hit 40%.
- This proves that while `tau_discard=0.15` (Adaptive) successfully protects the system from latency explosions by deleting 90% of retrieved chunks, it is **accidentally throwing away the correct evidence chunks** because their aspect density isn't high enough.

---

## Others

- **"Fading Critical Word"** problem in Dense Vocabulary Extractor. Terms like `TransEnc-8` or `Cleveland Fed` are being dropped or ignored during query expansion because they are not found in the IF extracted keywords list. As a result, the retriever relies on generic words with higher frequency. The Adaptive router correctly sees that generic words have low density and trashes the chunk. 

