# PyTerrier Baseline Evaluation Summary

Generated on 2026-09-07 16:34:12

### Cross-Dataset Retrieval Performance (nDCG@10, MRR@10, Recall@10)

| dataset                        | pipeline                  |   ndcg_10 |   mrr_10 |   recall_10 |   strict_10 |      p_10 |   avg_latency_ms |
|:-------------------------------|:--------------------------|----------:|---------:|------------:|------------:|----------:|-----------------:|
| bright_stackoverflow_doc_level | BM25_Default              |  0.156134 | 0.184517 |    0.194518 |    0.307692 | 0.0581197 |          40.4335 |
| bright_stackoverflow_doc_level | BM25_Analyzed             |  0.14372  | 0.186416 |    0.159975 |    0.282051 | 0.0529915 |          44.1343 |
| bright_stackoverflow_doc_level | BM25_RM3_Terrier_Default  |  0.147962 | 0.158544 |    0.186459 |    0.282051 | 0.0632479 |          81.5885 |
| bright_stackoverflow_doc_level | BM25_RM3_Unified_Default  |  0.111435 | 0.125451 |    0.150983 |    0.264957 | 0.0487179 |          99.5524 |
| bright_stackoverflow_doc_level | BM25_RM3_Unified_Analyzed |  0.103843 | 0.128181 |    0.117596 |    0.222222 | 0.0401709 |         104.786  |

### Indexing Efficiency (TTI in Seconds)

| dataset                        | pipeline                  |   default_index_time_s |   analyzed_pretokenize_s |   analyzed_index_s |   analyzed_total_tti_s |
|:-------------------------------|:--------------------------|-----------------------:|-------------------------:|-------------------:|-----------------------:|
| bright_stackoverflow_doc_level | BM25_Default              |                      0 |                        0 |                  0 |                      0 |
| bright_stackoverflow_doc_level | BM25_Analyzed             |                      0 |                        0 |                  0 |                      0 |
| bright_stackoverflow_doc_level | BM25_RM3_Terrier_Default  |                      0 |                        0 |                  0 |                      0 |
| bright_stackoverflow_doc_level | BM25_RM3_Unified_Default  |                      0 |                        0 |                  0 |                      0 |
| bright_stackoverflow_doc_level | BM25_RM3_Unified_Analyzed |                      0 |                        0 |                  0 |                      0 |
