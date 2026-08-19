# Pathway Specification: ListwiseLLMRerankerV2 (`pathway_listwise_reranker.md`)

## Overview
Evaluates candidate chunks in the Rerank Queue using a single-pass Listwise LLM call.

## Features
- **IDF-Filtered Sentence Snippets:** Extracts ~250-token window snippets around high-IDF query anchors to reduce LLM prompt token load by 75%.
- **LLM Client Wrapper:** Uses `src/utils/llm_client.py` for structured JSON completion.
