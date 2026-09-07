# PyTerrier Baseline Evaluation Summary

Generated on 2026-09-07 19:24:05

### Cross-Dataset Retrieval Performance (nDCG@10, MRR@10, Recall@10)

| dataset   | pipeline                  |   ndcg_10 |   mrr_10 |   recall_10 |   strict_10 |      p_10 |   avg_latency_ms |
|:----------|:--------------------------|----------:|---------:|------------:|------------:|----------:|-----------------:|
| arguana   | BM25_Default              |  0.367459 | 0.241613 |    0.765882 |    0.765882 | 0.0765882 |         27.6267  |
| arguana   | BM25_Analyzed             |  0.365958 | 0.24056  |    0.763026 |    0.763026 | 0.0763026 |         29.9918  |
| arguana   | BM25_RM3_Terrier_Default  |  0.36448  | 0.241028 |    0.755175 |    0.755175 | 0.0755175 |         59.3384  |
| arguana   | BM25_RM3_Unified_Default  |  0.249003 | 0.155484 |    0.551749 |    0.551749 | 0.0551749 |         70.0887  |
| arguana   | BM25_RM3_Unified_Analyzed |  0.340637 | 0.220023 |    0.72591  |    0.72591  | 0.072591  |         74.1018  |
| fiqa      | BM25_Default              |  0.252634 | 0.3104   |    0.309708 |    0.490741 | 0.0703704 |         22.2547  |
| fiqa      | BM25_Analyzed             |  0.247319 | 0.301615 |    0.311739 |    0.487654 | 0.0692901 |         23.4118  |
| fiqa      | BM25_RM3_Terrier_Default  |  0.243154 | 0.292421 |    0.298587 |    0.462963 | 0.0665123 |         48.14    |
| fiqa      | BM25_RM3_Unified_Default  |  0.199058 | 0.240656 |    0.264231 |    0.432099 | 0.0569444 |         62.4652  |
| fiqa      | BM25_RM3_Unified_Analyzed |  0.231013 | 0.279394 |    0.296236 |    0.462963 | 0.0646605 |         72.2041  |
| nfcorpus  | BM25_Default              |  0.329505 | 0.532077 |    0.153175 |    0.69969  | 0.239319  |          8.73861 |
| nfcorpus  | BM25_Analyzed             |  0.314299 | 0.525683 |    0.141281 |    0.684211 | 0.227245  |          8.00837 |
| nfcorpus  | BM25_RM3_Terrier_Default  |  0.351735 | 0.53586  |    0.174338 |    0.705882 | 0.259133  |         27.3221  |
| nfcorpus  | BM25_RM3_Unified_Default  |  0.335892 | 0.530952 |    0.164497 |    0.696594 | 0.244892  |         42.3933  |
| nfcorpus  | BM25_RM3_Unified_Analyzed |  0.317277 | 0.516901 |    0.151935 |    0.687307 | 0.229102  |         46.2224  |
| scidocs   | BM25_Default              |  0.158157 | 0.276956 |    0.164733 |    0.498    | 0.0813    |         22.3987  |
| scidocs   | BM25_Analyzed             |  0.155768 | 0.275418 |    0.1609   |    0.49     | 0.0793    |         23.1816  |
| scidocs   | BM25_RM3_Terrier_Default  |  0.157234 | 0.269939 |    0.16265  |    0.463    | 0.0801    |         51.0286  |
| scidocs   | BM25_RM3_Unified_Default  |  0.149847 | 0.263308 |    0.157433 |    0.479    | 0.0778    |         66.5427  |
| scidocs   | BM25_RM3_Unified_Analyzed |  0.151554 | 0.267833 |    0.15685  |    0.478    | 0.0773    |         94.6526  |
| scifact   | BM25_Default              |  0.683904 | 0.644094 |    0.826278 |    0.843333 | 0.0916667 |         20.7182  |
| scifact   | BM25_Analyzed             |  0.667464 | 0.629094 |    0.8      |    0.813333 | 0.0873333 |         19.0976  |
| scifact   | BM25_RM3_Terrier_Default  |  0.673053 | 0.633279 |    0.805722 |    0.816667 | 0.0896667 |         43.2143  |
| scifact   | BM25_RM3_Unified_Default  |  0.648071 | 0.609452 |    0.785111 |    0.796667 | 0.0866667 |         57.681   |
| scifact   | BM25_RM3_Unified_Analyzed |  0.650232 | 0.611795 |    0.786667 |    0.803333 | 0.0856667 |         62.0055  |
| quora     | BM25_Default              |  0.631789 | 0.64     |    0.8      |    0.8      | 0.12      |         23.173   |
| quora     | BM25_Analyzed             |  0.677371 | 0.64     |    0.8      |    0.8      | 0.12      |         43.2775  |
| quora     | BM25_RM3_Terrier_Default  |  0.58268  | 0.6      |    0.6      |    0.6      | 0.1       |         53.0013  |
| quora     | BM25_RM3_Unified_Default  |  0.346228 | 0.4      |    0.466667 |    0.6      | 0.06      |         32.0476  |
| quora     | BM25_RM3_Unified_Analyzed |  0.690071 | 0.7      |    0.8      |    0.8      | 0.12      |         60.8367  |

### Indexing Efficiency (TTI in Seconds)

| dataset   | pipeline                  |   default_index_time_s |   analyzed_pretokenize_s |   analyzed_index_s |   analyzed_total_tti_s |
|:----------|:--------------------------|-----------------------:|-------------------------:|-------------------:|-----------------------:|
| arguana   | BM25_Default              |                   0    |                        0 |               0    |                   0    |
| arguana   | BM25_Analyzed             |                   0    |                        0 |               0    |                   0    |
| arguana   | BM25_RM3_Terrier_Default  |                   0    |                        0 |               0    |                   0    |
| arguana   | BM25_RM3_Unified_Default  |                   0    |                        0 |               0    |                   0    |
| arguana   | BM25_RM3_Unified_Analyzed |                   0    |                        0 |               0    |                   0    |
| fiqa      | BM25_Default              |                   5.73 |                        0 |              10.74 |                  10.74 |
| fiqa      | BM25_Analyzed             |                   5.73 |                        0 |              10.74 |                  10.74 |
| fiqa      | BM25_RM3_Terrier_Default  |                   5.73 |                        0 |              10.74 |                  10.74 |
| fiqa      | BM25_RM3_Unified_Default  |                   5.73 |                        0 |              10.74 |                  10.74 |
| fiqa      | BM25_RM3_Unified_Analyzed |                   5.73 |                        0 |              10.74 |                  10.74 |
| nfcorpus  | BM25_Default              |                   0.77 |                        0 |               1.53 |                   1.53 |
| nfcorpus  | BM25_Analyzed             |                   0.77 |                        0 |               1.53 |                   1.53 |
| nfcorpus  | BM25_RM3_Terrier_Default  |                   0.77 |                        0 |               1.53 |                   1.53 |
| nfcorpus  | BM25_RM3_Unified_Default  |                   0.77 |                        0 |               1.53 |                   1.53 |
| nfcorpus  | BM25_RM3_Unified_Analyzed |                   0.77 |                        0 |               1.53 |                   1.53 |
| scidocs   | BM25_Default              |                   5.7  |                        0 |               7.56 |                   7.56 |
| scidocs   | BM25_Analyzed             |                   5.7  |                        0 |               7.56 |                   7.56 |
| scidocs   | BM25_RM3_Terrier_Default  |                   5.7  |                        0 |               7.56 |                   7.56 |
| scidocs   | BM25_RM3_Unified_Default  |                   5.7  |                        0 |               7.56 |                   7.56 |
| scidocs   | BM25_RM3_Unified_Analyzed |                   5.7  |                        0 |               7.56 |                   7.56 |
| scifact   | BM25_Default              |                   2.11 |                        0 |               1.94 |                   1.94 |
| scifact   | BM25_Analyzed             |                   2.11 |                        0 |               1.94 |                   1.94 |
| scifact   | BM25_RM3_Terrier_Default  |                   2.11 |                        0 |               1.94 |                   1.94 |
| scifact   | BM25_RM3_Unified_Default  |                   2.11 |                        0 |               1.94 |                   1.94 |
| scifact   | BM25_RM3_Unified_Analyzed |                   2.11 |                        0 |               1.94 |                   1.94 |
| quora     | BM25_Default              |                  12.76 |                        0 |              15.29 |                  15.29 |
| quora     | BM25_Analyzed             |                  12.76 |                        0 |              15.29 |                  15.29 |
| quora     | BM25_RM3_Terrier_Default  |                  12.76 |                        0 |              15.29 |                  15.29 |
| quora     | BM25_RM3_Unified_Default  |                  12.76 |                        0 |              15.29 |                  15.29 |
| quora     | BM25_RM3_Unified_Analyzed |                  12.76 |                        0 |              15.29 |                  15.29 |
