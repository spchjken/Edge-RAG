# Query Expansion Ablation Summary (Fintech Dataset)

Evaluated on 30 complex, multi-hop queries from the fintech ground-truth dataset.

## Empirical Performance
| Method | Avg Latency (s) | Aspect Coverage | Keyword Jaccard | Acronym Retention | Stopword Count |
|--------|-----------------|-----------------|-----------------|-------------------|----------------|
| llm | 2.6152 | 0.00% | 33.85% | 98.33% | 1.53 |
| vector | 0.2953 | 0.00% | 21.48% | 85.00% | 2.00 |
| yake_vector | 0.0088 | 1.11% | 26.86% | 92.22% | 0.03 |
| llm_aspect_only | 1.2409 | 0.00% | 24.75% | 88.89% | 0.40 |
| statistical | 0.0612 | 0.00% | 20.83% | 80.00% | 10.23 |
| yake_statistical | 0.0013 | 1.11% | 26.86% | 92.22% | 0.03 |

## Metric Definitions
- **Aspect Coverage**: Percentage of `detached_aspect` ground truth concepts captured.
- **Keyword Jaccard**: Strict token overlap with ground truth synonyms (lower is normal, high means precise match).
- **Acronym Retention**: Percentage of acronyms in the query that were preserved in the output.
