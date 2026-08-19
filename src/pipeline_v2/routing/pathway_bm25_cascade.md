# Pathway Specification: BM25CascadeRouter (`pathway_bm25_cascade.md`)

## Overview
Three-way triage router for BM25 candidates based on relative BM25 score and Aspect Coverage ($\alpha$).

## Triage Math
$$\text{Normalized\_BM25}(D) = \frac{S_{\text{BM25}}(D)}{\max_{D' \in \text{Pool}} S_{\text{BM25}}(D')}$$
$$\alpha = \frac{\text{Number of Aspect Groups with } \ge 1 \text{ keyword match in } D}{N_{\text{aspects}}}$$
$$\text{Score}_{\text{route}}(D) = \text{Normalized\_BM25}(D) \times (0.5 + 0.5 \cdot \alpha)$$

## Thresholds
- **Bypass Queue:** $\text{Score}_{\text{route}} \ge 0.75$ (high relevance, skips LLM reranker)
- **Discard:** $\text{Score}_{\text{route}} < 0.15$ (low relevance)
- **Rerank Queue:** $0.15 \le \text{Score}_{\text{route}} < 0.75$ (passed to Listwise LLM Reranker)
