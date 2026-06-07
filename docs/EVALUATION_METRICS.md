# Evaluation Metrics & Benchmarking Protocol

This document defines the strict evaluation metrics used to benchmark the **Edge-RAG** pipeline against baseline models (BM25, Dense RAG, and LLMLingua-2). All metrics must adhere to the reproducibility standards defined in `.agents/rules/02-reproducibility.md`.

## 1. Latency Metrics

### Time-to-First-Token (TTFT)
- **Definition:** The total wall-clock time from when the user submits the query to the pipeline until the generation LLM yields its first output token.
- **Includes:** Query Expansion, Lexical Search, Aspect-Weighted Density Routing, LLM Reranking (if bypassed chunks < $N_{max}$), and Prompt encoding.
- **Excludes:** Model loading times (benchmarks assume a "warm boot" server).
- **Measurement Protocol:** 
  - Must call `torch.cuda.synchronize()` immediately before starting the timer and immediately upon receiving the first token stream chunk.
  - Reported as the median over $N \ge 3$ runs, alongside the standard deviation.

## 2. Resource Constraints

### Peak VRAM Consumption
- **Definition:** The maximum GPU memory allocated during the processing of a single query.
- **Measurement Protocol:**
  - Call `torch.cuda.reset_peak_memory_stats()` at the beginning of the pipeline execution.
  - Call `torch.cuda.max_memory_allocated()` at the very end of the pipeline.
  - GPU cache must be cleared (`torch.cuda.empty_cache()`) between benchmark iterations.
  - Reported as absolute Peak VRAM (in GB), not an average.

### Context Compression Ratio ($C_r$)
- **Definition:** The ratio measuring how effectively the pipeline filters the initial document pool down to the final generation context.
- **Formula:** $C_r = \frac{|D_{initial}|}{|D_{final}|}$ where $|D_{initial}|$ is the token count of the raw corpus and $|D_{final}|$ is the token count of the final chunks retrieved for generation ($|D_{final}| \le N_{max}$).

## 3. Retrieval Efficacy

### Precision@K ($P@K$)
- **Definition:** The proportion of chunks routed to the final generation phase that are actually relevant to the query (contain ground-truth evidence).

### Recall@K ($R@K$)
- **Definition:** The proportion of all relevant ground-truth chunks in the corpus that successfully made it through the Dual-Bypass router and LLM Reranker into the final generation context.

## 4. Generation Quality

### Answer Faithfulness (LLM-as-a-Judge)
- **Definition:** A normalized score $[0, 1]$ evaluating whether the final generated answer is factually supported by the retrieved context and correctly answers the user's query without hallucination.
- **Measurement Protocol:** Evaluated using a strong judge model (e.g., GPT-4 or an equivalent oracle) against a standardized grading prompt.
