# Benchmark Generation Pipeline (Edge-RAG)

This document outlines the state-of-the-art methodology for generating synthetic RAG evaluation datasets from raw arXiv papers. This approach guarantees zero token-overlap waste during generation and perfectly handles global collisions via offline brute-force recall.

## 1. Document Parsing & Hierarchical Chunking
Before calling any LLM, the raw `.txt` documents from `data/raw/latest_arxiv` must be split into two parallel representations:

- **Parent Blocks (Non-Overlapping):** Split the document by logical boundaries (e.g., Markdown headings, double newlines) into large, non-overlapping sections (e.g., ~3000 tokens). Assign each block a unique ID (e.g., `doc42_block7`).
- **Child Chunks (Overlapping):** Run the exact same sliding-window chunker used in your RAG pipeline (e.g., 1000 tokens, 100 overlap) over the document. Every child chunk must store a `parent_id` linking it to its parent block.

## 2. Seed Generation
Use a powerful LLM (Deepseek-V4-pro).
- Pass a single **Parent Block** to the LLM.
- Instruct the LLM to generate diverse queries and their corresponding "Golden Answers" based *strictly* on that block.

## 3. Query Paraphrasing (Vocabulary Gap Injection)
To test query expansion properly, we must artificially enforce a vocabulary mismatch (Lexical Bias) between the query and the source block.
- Pass the generated Query from Step 2 back to the powerful LLM (in a separate prompt/pass) and instruct it to rewrite the query using synonymous phrasing.
- **Why a separate pass?** If you ask the LLM to comprehend the text, extract the answer, *and* avoid using the text's vocabulary all in one single prompt, its performance drops and it often accidentally borrows words anyway. By doing it in two steps, you can also save both the "Easy" (raw) and "Hard" (paraphrased) queries in your dataset for better benchmarking!

## 4. Offline Global Recall Pass (The False Negative Fix)
For every generated query, we must find *all* possible valid chunks in the global corpus.
- Set up an offline, heavy-weight retrieval pipeline (e.g., Hybrid BM25 + BGE-Large). 
- Run every generated query against the *entire* index of **Child Chunks**.
- Retrieve a massive number of chunks (e.g., `Top-50` or `Top-100`). At this stage, maximize **Recall**, ignore precision, and ignore latency.

## 5. Oracle Filtering (LLM-as-a-Judge Annotation)
Filter the noisy retrieved chunks to build the final ground truth.
- For each generated query, feed the query, the Golden Answer, and the retrieved child chunks to a powerful offline LLM.
- Ask the LLM a binary question for each chunk: *"Does this chunk contain sufficient evidence to answer the query?"*
- Keep all chunks that the LLM flags as `True`.

## 6. Ground-Truth Finalization
Compile the results into a final JSON dataset. For every query, the `answer_evidence` is no longer a fuzzy text string or section header. It is a strict list of verified `Chunk_IDs`.

### Example Final JSON Structure:
```json
{
  "query_id": "q_8472",
  "raw_question": "How does the Belief-at-Risk measure combine uncertainty and tail risk?",
  "paraphrased_question": "Explain the combination of risk measures in Belief-at-Risk. ",
  "golden_answer": "Belief-at-Risk multiplies the exponential of the normalized posterior entropy by a rolling 95% CVaR...",
  "ground_truth_parent_block_id": "doc12_block14",
  "complexity": "Medium",
  "ground_truth_child_chunk_ids": [
    "doc12_chunk45",
    "doc12_chunk46",
  ],
  "extended_child_chunk_ids": [
    "doc12_chunk12",
    "doc12_chunk43",
    "doc12_chunk44",
    "doc12_chunk47",
    "doc12_chunk48",
    "doc89_chunk10",
    "doc89_chunk11",
    "doc89_chunk13",
    "doc89_chunk14" // Found via the Global Recall Pass!
  ]
}
```
