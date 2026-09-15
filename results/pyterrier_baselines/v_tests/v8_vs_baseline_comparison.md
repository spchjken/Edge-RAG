# Comprehensive Empirical Audit: Edge-RAG V8 vs Baselines (20 Corpora)

## Executive Summary

The table below summarizes the multi-metric performance across all 20 small-to-medium corpora comparing **Edge-RAG V8** against **Standard BM25** and **Standard DPH**:

| Metric | BM25 Base | V8_BM25 | Δ BM25 | DPH Base | V8_DPH | Δ DPH |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| nDCG@10 (Target Ranking Quality) | 0.2342 | 0.2268 | **-0.0074 (3W/17L)** | 0.2549 | 0.2476 | **-0.0073 (1W/16L)** |
| MRR@10 (First Relevant Hit Reciprocal Rank) | 0.2876 | 0.2778 | **-0.0099 (2W/16L)** | 0.3210 | 0.3122 | **-0.0087 (2W/16L)** |
| MAP@100 (Mean Average Precision) | 0.1657 | 0.1604 | **-0.0053 (1W/18L)** | 0.1769 | 0.1724 | **-0.0045 (2W/16L)** |
| Recall@100 (Candidate Funnel for Downstream Reranker) | 0.4238 | 0.4169 | **-0.0069 (1W/18L)** | 0.4472 | 0.4438 | **-0.0034 (6W/13L)** |
| Recall@1000 (Candidate Ceiling) | 0.6748 | 0.6640 | **-0.0108 (1W/16L)** | 0.6801 | 0.6696 | **-0.0104 (2W/17L)** |
| Precision@10 | 0.1049 | 0.1018 | **-0.0031 (4W/16L)** | 0.1205 | 0.1170 | **-0.0035 (1W/15L)** |
| Online Latency P50 (ms) | 45.52 | 31.94 | **-13.59 (17W/3L)** | 45.95 | 33.03 | **-12.92 (17W/3L)** |


---

Results are directly extracted from `results/pyterrier_baselines/pyterrier_v8_results.csv` and `results/pyterrier_baselines/pyterrier_baselines_results.csv`.

## 1. Metric: nDCG@10 (Target Ranking Quality)

### A. BM25 vs V8_BM25
| Dataset | Baseline | V8 | $\Delta$ ndcg_10 | Outcome |
|:---|:---:|:---:|:---:|:---:|
| `scifact` | 0.6839 | 0.6817 | **-0.0022** | LOSS (▼) |
| `nfcorpus` | 0.3282 | 0.3222 | **-0.0060** | LOSS (▼) |
| `arguana` | 0.3662 | 0.3569 | **-0.0093** | LOSS (▼) |
| `bright_pony` | 0.0252 | 0.0231 | **-0.0021** | LOSS (▼) |
| `bright_theoremqa_theorems` | 0.0192 | 0.0161 | **-0.0031** | LOSS (▼) |
| `scidocs` | 0.1582 | 0.1506 | **-0.0076** | LOSS (▼) |
| `bright_economics` | 0.1177 | 0.1114 | **-0.0063** | LOSS (▼) |
| `bright_psychology` | 0.0926 | 0.0938 | **+0.0012** | WIN (▲) |
| `bright_biology` | 0.0912 | 0.0860 | **-0.0052** | LOSS (▼) |
| `fiqa` | 0.2526 | 0.2446 | **-0.0080** | LOSS (▼) |
| `bright_sustainable_living` | 0.0981 | 0.0895 | **-0.0086** | LOSS (▼) |
| `bright_robotics` | 0.0996 | 0.0985 | **-0.0011** | LOSS (▼) |
| `bright_stackoverflow` | 0.1561 | 0.1515 | **-0.0046** | LOSS (▼) |
| `bright_earth_science` | 0.1201 | 0.1183 | **-0.0018** | LOSS (▼) |
| `trec_covid` | 0.6295 | 0.5981 | **-0.0314** | LOSS (▼) |
| `bright_aops` | 0.0604 | 0.0610 | **+0.0006** | WIN (▲) |
| `bright_theoremqa_questions` | 0.0700 | 0.0731 | **+0.0031** | WIN (▲) |
| `webis_touche2020` | 0.2979 | 0.2781 | **-0.0198** | LOSS (▼) |
| `bright_leetcode` | 0.2496 | 0.2370 | **-0.0126** | LOSS (▼) |
| `quora` | 0.7676 | 0.7450 | **-0.0226** | LOSS (▼) |
| **Mean (20 Datasets)** | **0.2342** | **0.2268** | **-0.0074** | **3W / 17L / 0T** |

### B. DPH vs V8_DPH
| Dataset | Baseline | V8 | $\Delta$ ndcg_10 | Outcome |
|:---|:---:|:---:|:---:|:---:|
| `scifact` | 0.6716 | 0.6703 | **-0.0013** | LOSS (▼) |
| `nfcorpus` | 0.3221 | 0.3170 | **-0.0051** | LOSS (▼) |
| `arguana` | 0.3266 | 0.3152 | **-0.0114** | LOSS (▼) |
| `bright_pony` | 0.0639 | 0.0559 | **-0.0080** | LOSS (▼) |
| `bright_theoremqa_theorems` | 0.0248 | 0.0248 | **+0.0000** | TIE (=) |
| `scidocs` | 0.1499 | 0.1445 | **-0.0054** | LOSS (▼) |
| `bright_economics` | 0.1725 | 0.1577 | **-0.0148** | LOSS (▼) |
| `bright_psychology` | 0.1756 | 0.1742 | **-0.0014** | LOSS (▼) |
| `bright_biology` | 0.2523 | 0.2438 | **-0.0085** | LOSS (▼) |
| `fiqa` | 0.2433 | 0.2369 | **-0.0064** | LOSS (▼) |
| `bright_sustainable_living` | 0.1669 | 0.1608 | **-0.0061** | LOSS (▼) |
| `bright_robotics` | 0.1543 | 0.1540 | **-0.0003** | TIE (=) |
| `bright_stackoverflow` | 0.1677 | 0.1609 | **-0.0068** | LOSS (▼) |
| `bright_earth_science` | 0.3539 | 0.3563 | **+0.0024** | WIN (▲) |
| `trec_covid` | 0.6310 | 0.6106 | **-0.0204** | LOSS (▼) |
| `bright_aops` | 0.0536 | 0.0533 | **-0.0003** | TIE (=) |
| `bright_theoremqa_questions` | 0.0656 | 0.0632 | **-0.0024** | LOSS (▼) |
| `webis_touche2020` | 0.4174 | 0.3907 | **-0.0267** | LOSS (▼) |
| `bright_leetcode` | 0.2418 | 0.2344 | **-0.0074** | LOSS (▼) |
| `quora` | 0.4429 | 0.4271 | **-0.0158** | LOSS (▼) |
| **Mean (20 Datasets)** | **0.2549** | **0.2476** | **-0.0073** | **1W / 16L / 3T** |

## 1. Metric: MRR@10 (First Relevant Hit Reciprocal Rank)

### A. BM25 vs V8_BM25
| Dataset | Baseline | V8 | $\Delta$ mrr_10 | Outcome |
|:---|:---:|:---:|:---:|:---:|
| `scifact` | 0.6441 | 0.6446 | **+0.0005** | TIE (=) |
| `nfcorpus` | 0.5321 | 0.5245 | **-0.0076** | LOSS (▼) |
| `arguana` | 0.2408 | 0.2341 | **-0.0067** | LOSS (▼) |
| `bright_pony` | 0.0636 | 0.0570 | **-0.0066** | LOSS (▼) |
| `bright_theoremqa_theorems` | 0.0178 | 0.0145 | **-0.0033** | LOSS (▼) |
| `scidocs` | 0.2770 | 0.2601 | **-0.0169** | LOSS (▼) |
| `bright_economics` | 0.1355 | 0.1282 | **-0.0073** | LOSS (▼) |
| `bright_psychology` | 0.0996 | 0.1003 | **+0.0007** | WIN (▲) |
| `bright_biology` | 0.1294 | 0.1179 | **-0.0115** | LOSS (▼) |
| `fiqa` | 0.3104 | 0.3040 | **-0.0064** | LOSS (▼) |
| `bright_sustainable_living` | 0.1249 | 0.1115 | **-0.0134** | LOSS (▼) |
| `bright_robotics` | 0.1244 | 0.1233 | **-0.0011** | LOSS (▼) |
| `bright_stackoverflow` | 0.1845 | 0.1771 | **-0.0074** | LOSS (▼) |
| `bright_earth_science` | 0.1583 | 0.1472 | **-0.0111** | LOSS (▼) |
| `trec_covid` | 0.8725 | 0.8322 | **-0.0403** | LOSS (▼) |
| `bright_aops` | 0.1245 | 0.1245 | **+0.0000** | TIE (=) |
| `bright_theoremqa_questions` | 0.0804 | 0.0809 | **+0.0005** | WIN (▲) |
| `webis_touche2020` | 0.5713 | 0.5596 | **-0.0117** | LOSS (▼) |
| `bright_leetcode` | 0.3028 | 0.2798 | **-0.0230** | LOSS (▼) |
| `quora` | 0.7584 | 0.7339 | **-0.0245** | LOSS (▼) |
| **Mean (20 Datasets)** | **0.2876** | **0.2778** | **-0.0099** | **2W / 16L / 2T** |

### B. DPH vs V8_DPH
| Dataset | Baseline | V8 | $\Delta$ mrr_10 | Outcome |
|:---|:---:|:---:|:---:|:---:|
| `scifact` | 0.6349 | 0.6359 | **+0.0010** | WIN (▲) |
| `nfcorpus` | 0.5254 | 0.5160 | **-0.0094** | LOSS (▼) |
| `arguana` | 0.2101 | 0.2019 | **-0.0082** | LOSS (▼) |
| `bright_pony` | 0.1452 | 0.1210 | **-0.0242** | LOSS (▼) |
| `bright_theoremqa_theorems` | 0.0307 | 0.0307 | **+0.0000** | TIE (=) |
| `scidocs` | 0.2624 | 0.2507 | **-0.0117** | LOSS (▼) |
| `bright_economics` | 0.2161 | 0.1976 | **-0.0185** | LOSS (▼) |
| `bright_psychology` | 0.2215 | 0.2153 | **-0.0062** | LOSS (▼) |
| `bright_biology` | 0.3531 | 0.3490 | **-0.0041** | LOSS (▼) |
| `fiqa` | 0.3039 | 0.2986 | **-0.0053** | LOSS (▼) |
| `bright_sustainable_living` | 0.2026 | 0.1906 | **-0.0120** | LOSS (▼) |
| `bright_robotics` | 0.1818 | 0.1822 | **+0.0004** | TIE (=) |
| `bright_stackoverflow` | 0.1987 | 0.1863 | **-0.0124** | LOSS (▼) |
| `bright_earth_science` | 0.4804 | 0.4819 | **+0.0015** | WIN (▲) |
| `trec_covid` | 0.8472 | 0.8306 | **-0.0166** | LOSS (▼) |
| `bright_aops` | 0.1082 | 0.1055 | **-0.0027** | LOSS (▼) |
| `bright_theoremqa_questions` | 0.0686 | 0.0668 | **-0.0018** | LOSS (▼) |
| `webis_touche2020` | 0.7204 | 0.6973 | **-0.0231** | LOSS (▼) |
| `bright_leetcode` | 0.2895 | 0.2838 | **-0.0057** | LOSS (▼) |
| `quora` | 0.4190 | 0.4032 | **-0.0158** | LOSS (▼) |
| **Mean (20 Datasets)** | **0.3210** | **0.3122** | **-0.0087** | **2W / 16L / 2T** |

## 1. Metric: MAP@100 (Mean Average Precision)

### A. BM25 vs V8_BM25
| Dataset | Baseline | V8 | $\Delta$ map_100 | Outcome |
|:---|:---:|:---:|:---:|:---:|
| `scifact` | 0.6376 | 0.6369 | **-0.0007** | LOSS (▼) |
| `nfcorpus` | 0.1463 | 0.1447 | **-0.0016** | LOSS (▼) |
| `arguana` | 0.2519 | 0.2458 | **-0.0061** | LOSS (▼) |
| `bright_pony` | 0.0124 | 0.0118 | **-0.0006** | LOSS (▼) |
| `bright_theoremqa_theorems` | 0.0119 | 0.0111 | **-0.0008** | LOSS (▼) |
| `scidocs` | 0.1079 | 0.1025 | **-0.0054** | LOSS (▼) |
| `bright_economics` | 0.0900 | 0.0856 | **-0.0044** | LOSS (▼) |
| `bright_psychology` | 0.0778 | 0.0734 | **-0.0044** | LOSS (▼) |
| `bright_biology` | 0.0683 | 0.0598 | **-0.0085** | LOSS (▼) |
| `fiqa` | 0.2087 | 0.2006 | **-0.0081** | LOSS (▼) |
| `bright_sustainable_living` | 0.0740 | 0.0653 | **-0.0087** | LOSS (▼) |
| `bright_robotics` | 0.0779 | 0.0767 | **-0.0012** | LOSS (▼) |
| `bright_stackoverflow` | 0.1338 | 0.1265 | **-0.0073** | LOSS (▼) |
| `bright_earth_science` | 0.0970 | 0.0935 | **-0.0035** | LOSS (▼) |
| `trec_covid` | 0.0850 | 0.0820 | **-0.0030** | LOSS (▼) |
| `bright_aops` | 0.0412 | 0.0412 | **+0.0000** | TIE (=) |
| `bright_theoremqa_questions` | 0.0607 | 0.0664 | **+0.0057** | WIN (▲) |
| `webis_touche2020` | 0.1931 | 0.1816 | **-0.0115** | LOSS (▼) |
| `bright_leetcode` | 0.2106 | 0.1987 | **-0.0119** | LOSS (▼) |
| `quora` | 0.7270 | 0.7037 | **-0.0233** | LOSS (▼) |
| **Mean (20 Datasets)** | **0.1657** | **0.1604** | **-0.0053** | **1W / 18L / 1T** |

### B. DPH vs V8_DPH
| Dataset | Baseline | V8 | $\Delta$ map_100 | Outcome |
|:---|:---:|:---:|:---:|:---:|
| `scifact` | 0.6299 | 0.6279 | **-0.0020** | LOSS (▼) |
| `nfcorpus` | 0.1435 | 0.1422 | **-0.0013** | LOSS (▼) |
| `arguana` | 0.2229 | 0.2151 | **-0.0078** | LOSS (▼) |
| `bright_pony` | 0.0244 | 0.0229 | **-0.0015** | LOSS (▼) |
| `bright_theoremqa_theorems` | 0.0196 | 0.0196 | **+0.0000** | TIE (=) |
| `scidocs` | 0.1011 | 0.0969 | **-0.0042** | LOSS (▼) |
| `bright_economics` | 0.1313 | 0.1168 | **-0.0145** | LOSS (▼) |
| `bright_psychology` | 0.1320 | 0.1301 | **-0.0019** | LOSS (▼) |
| `bright_biology` | 0.2145 | 0.2046 | **-0.0099** | LOSS (▼) |
| `fiqa` | 0.1985 | 0.1942 | **-0.0043** | LOSS (▼) |
| `bright_sustainable_living` | 0.1384 | 0.1345 | **-0.0039** | LOSS (▼) |
| `bright_robotics` | 0.1217 | 0.1230 | **+0.0013** | WIN (▲) |
| `bright_stackoverflow` | 0.1389 | 0.1321 | **-0.0068** | LOSS (▼) |
| `bright_earth_science` | 0.2865 | 0.2901 | **+0.0036** | WIN (▲) |
| `trec_covid` | 0.0774 | 0.0760 | **-0.0014** | LOSS (▼) |
| `bright_aops` | 0.0364 | 0.0360 | **-0.0004** | TIE (=) |
| `bright_theoremqa_questions` | 0.0595 | 0.0556 | **-0.0039** | LOSS (▼) |
| `webis_touche2020` | 0.2535 | 0.2423 | **-0.0112** | LOSS (▼) |
| `bright_leetcode` | 0.2070 | 0.2010 | **-0.0060** | LOSS (▼) |
| `quora` | 0.4014 | 0.3870 | **-0.0144** | LOSS (▼) |
| **Mean (20 Datasets)** | **0.1769** | **0.1724** | **-0.0045** | **2W / 16L / 2T** |

## 1. Metric: Recall@100 (Candidate Funnel for Downstream Reranker)

### A. BM25 vs V8_BM25
| Dataset | Baseline | V8 | $\Delta$ recall_100 | Outcome |
|:---|:---:|:---:|:---:|:---:|
| `scifact` | 0.9260 | 0.9176 | **-0.0084** | LOSS (▼) |
| `nfcorpus` | 0.2483 | 0.2473 | **-0.0010** | LOSS (▼) |
| `arguana` | 0.9701 | 0.9651 | **-0.0050** | LOSS (▼) |
| `bright_pony` | 0.1439 | 0.1424 | **-0.0015** | LOSS (▼) |
| `bright_theoremqa_theorems` | 0.1140 | 0.1118 | **-0.0022** | LOSS (▼) |
| `scidocs` | 0.3620 | 0.3509 | **-0.0111** | LOSS (▼) |
| `bright_economics` | 0.3885 | 0.3689 | **-0.0196** | LOSS (▼) |
| `bright_psychology` | 0.3575 | 0.3439 | **-0.0136** | LOSS (▼) |
| `bright_biology` | 0.3389 | 0.3260 | **-0.0129** | LOSS (▼) |
| `fiqa` | 0.5592 | 0.5504 | **-0.0088** | LOSS (▼) |
| `bright_sustainable_living` | 0.4080 | 0.4059 | **-0.0021** | LOSS (▼) |
| `bright_robotics` | 0.3813 | 0.3852 | **+0.0039** | WIN (▲) |
| `bright_stackoverflow` | 0.4526 | 0.4443 | **-0.0083** | LOSS (▼) |
| `bright_earth_science` | 0.4050 | 0.3908 | **-0.0142** | LOSS (▼) |
| `trec_covid` | 0.1237 | 0.1201 | **-0.0036** | LOSS (▼) |
| `bright_aops` | 0.1884 | 0.1854 | **-0.0030** | LOSS (▼) |
| `bright_theoremqa_questions` | 0.1600 | 0.1600 | **+0.0000** | TIE (=) |
| `webis_touche2020` | 0.5373 | 0.5216 | **-0.0157** | LOSS (▼) |
| `bright_leetcode` | 0.4444 | 0.4432 | **-0.0012** | LOSS (▼) |
| `quora` | 0.9660 | 0.9571 | **-0.0089** | LOSS (▼) |
| **Mean (20 Datasets)** | **0.4238** | **0.4169** | **-0.0069** | **1W / 18L / 1T** |

### B. DPH vs V8_DPH
| Dataset | Baseline | V8 | $\Delta$ recall_100 | Outcome |
|:---|:---:|:---:|:---:|:---:|
| `scifact` | 0.9159 | 0.9179 | **+0.0020** | WIN (▲) |
| `nfcorpus` | 0.2449 | 0.2441 | **-0.0008** | LOSS (▼) |
| `arguana` | 0.9509 | 0.9438 | **-0.0071** | LOSS (▼) |
| `bright_pony` | 0.1993 | 0.1980 | **-0.0013** | LOSS (▼) |
| `bright_theoremqa_theorems` | 0.0965 | 0.0965 | **+0.0000** | TIE (=) |
| `scidocs` | 0.3475 | 0.3388 | **-0.0087** | LOSS (▼) |
| `bright_economics` | 0.4498 | 0.4553 | **+0.0055** | WIN (▲) |
| `bright_psychology` | 0.4452 | 0.4358 | **-0.0094** | LOSS (▼) |
| `bright_biology` | 0.5172 | 0.5203 | **+0.0031** | WIN (▲) |
| `fiqa` | 0.5393 | 0.5267 | **-0.0126** | LOSS (▼) |
| `bright_sustainable_living` | 0.4593 | 0.4768 | **+0.0175** | WIN (▲) |
| `bright_robotics` | 0.4745 | 0.4803 | **+0.0058** | WIN (▲) |
| `bright_stackoverflow` | 0.4291 | 0.4313 | **+0.0022** | WIN (▲) |
| `bright_earth_science` | 0.6187 | 0.6105 | **-0.0082** | LOSS (▼) |
| `trec_covid` | 0.1136 | 0.1117 | **-0.0019** | LOSS (▼) |
| `bright_aops` | 0.1916 | 0.1845 | **-0.0071** | LOSS (▼) |
| `bright_theoremqa_questions` | 0.1659 | 0.1582 | **-0.0077** | LOSS (▼) |
| `webis_touche2020` | 0.5496 | 0.5391 | **-0.0105** | LOSS (▼) |
| `bright_leetcode` | 0.4549 | 0.4367 | **-0.0182** | LOSS (▼) |
| `quora` | 0.7805 | 0.7690 | **-0.0115** | LOSS (▼) |
| **Mean (20 Datasets)** | **0.4472** | **0.4438** | **-0.0034** | **6W / 13L / 1T** |

## 1. Metric: Recall@1000 (Candidate Ceiling)

### A. BM25 vs V8_BM25
| Dataset | Baseline | V8 | $\Delta$ recall_1000 | Outcome |
|:---|:---:|:---:|:---:|:---:|
| `scifact` | 0.9767 | 0.9727 | **-0.0040** | LOSS (▼) |
| `nfcorpus` | 0.3650 | 0.3581 | **-0.0069** | LOSS (▼) |
| `arguana` | 0.9922 | 0.9915 | **-0.0007** | LOSS (▼) |
| `bright_pony` | 0.7426 | 0.7441 | **+0.0015** | WIN (▲) |
| `bright_theoremqa_theorems` | 0.3289 | 0.3289 | **+0.0000** | TIE (=) |
| `scidocs` | 0.5713 | 0.5628 | **-0.0085** | LOSS (▼) |
| `bright_economics` | 0.6961 | 0.6957 | **-0.0004** | TIE (=) |
| `bright_psychology` | 0.6385 | 0.6326 | **-0.0059** | LOSS (▼) |
| `bright_biology` | 0.6850 | 0.6852 | **+0.0002** | TIE (=) |
| `fiqa` | 0.7742 | 0.7591 | **-0.0151** | LOSS (▼) |
| `bright_sustainable_living` | 0.7489 | 0.7354 | **-0.0135** | LOSS (▼) |
| `bright_robotics` | 0.7051 | 0.6887 | **-0.0164** | LOSS (▼) |
| `bright_stackoverflow` | 0.6781 | 0.6648 | **-0.0133** | LOSS (▼) |
| `bright_earth_science` | 0.6823 | 0.6685 | **-0.0138** | LOSS (▼) |
| `trec_covid` | 0.4593 | 0.4483 | **-0.0110** | LOSS (▼) |
| `bright_aops` | 0.4085 | 0.3719 | **-0.0366** | LOSS (▼) |
| `bright_theoremqa_questions` | 0.3814 | 0.3600 | **-0.0214** | LOSS (▼) |
| `webis_touche2020` | 0.8781 | 0.8536 | **-0.0245** | LOSS (▼) |
| `bright_leetcode` | 0.7905 | 0.7694 | **-0.0211** | LOSS (▼) |
| `quora` | 0.9926 | 0.9890 | **-0.0036** | LOSS (▼) |
| **Mean (20 Datasets)** | **0.6748** | **0.6640** | **-0.0108** | **1W / 16L / 3T** |

### B. DPH vs V8_DPH
| Dataset | Baseline | V8 | $\Delta$ recall_1000 | Outcome |
|:---|:---:|:---:|:---:|:---:|
| `scifact` | 0.9733 | 0.9693 | **-0.0040** | LOSS (▼) |
| `nfcorpus` | 0.3671 | 0.3594 | **-0.0077** | LOSS (▼) |
| `arguana` | 0.9900 | 0.9893 | **-0.0007** | LOSS (▼) |
| `bright_pony` | 0.6766 | 0.6793 | **+0.0027** | WIN (▲) |
| `bright_theoremqa_theorems` | 0.3297 | 0.3242 | **-0.0055** | LOSS (▼) |
| `scidocs` | 0.5649 | 0.5607 | **-0.0042** | LOSS (▼) |
| `bright_economics` | 0.7256 | 0.7220 | **-0.0036** | LOSS (▼) |
| `bright_psychology` | 0.6922 | 0.6961 | **+0.0039** | WIN (▲) |
| `bright_biology` | 0.7758 | 0.7754 | **-0.0004** | TIE (=) |
| `fiqa` | 0.7562 | 0.7431 | **-0.0131** | LOSS (▼) |
| `bright_sustainable_living` | 0.8037 | 0.7892 | **-0.0145** | LOSS (▼) |
| `bright_robotics` | 0.7673 | 0.7623 | **-0.0050** | LOSS (▼) |
| `bright_stackoverflow` | 0.6667 | 0.6556 | **-0.0111** | LOSS (▼) |
| `bright_earth_science` | 0.7812 | 0.7711 | **-0.0101** | LOSS (▼) |
| `trec_covid` | 0.4102 | 0.4018 | **-0.0084** | LOSS (▼) |
| `bright_aops` | 0.3776 | 0.3445 | **-0.0331** | LOSS (▼) |
| `bright_theoremqa_questions` | 0.3698 | 0.3379 | **-0.0319** | LOSS (▼) |
| `webis_touche2020` | 0.8543 | 0.8309 | **-0.0234** | LOSS (▼) |
| `bright_leetcode` | 0.8164 | 0.7858 | **-0.0306** | LOSS (▼) |
| `quora` | 0.9026 | 0.8947 | **-0.0079** | LOSS (▼) |
| **Mean (20 Datasets)** | **0.6801** | **0.6696** | **-0.0104** | **2W / 17L / 1T** |

## 1. Metric: Precision@10

### A. BM25 vs V8_BM25
| Dataset | Baseline | V8 | $\Delta$ p_10 | Outcome |
|:---|:---:|:---:|:---:|:---:|
| `scifact` | 0.0917 | 0.0907 | **-0.0010** | LOSS (▼) |
| `nfcorpus` | 0.2393 | 0.2344 | **-0.0049** | LOSS (▼) |
| `arguana` | 0.0763 | 0.0747 | **-0.0016** | LOSS (▼) |
| `bright_pony` | 0.0259 | 0.0241 | **-0.0018** | LOSS (▼) |
| `bright_theoremqa_theorems` | 0.0066 | 0.0053 | **-0.0013** | LOSS (▼) |
| `scidocs` | 0.0813 | 0.0786 | **-0.0027** | LOSS (▼) |
| `bright_economics` | 0.0563 | 0.0534 | **-0.0029** | LOSS (▼) |
| `bright_psychology` | 0.0426 | 0.0446 | **+0.0020** | WIN (▲) |
| `bright_biology` | 0.0379 | 0.0359 | **-0.0020** | LOSS (▼) |
| `fiqa` | 0.0704 | 0.0681 | **-0.0023** | LOSS (▼) |
| `bright_sustainable_living` | 0.0491 | 0.0463 | **-0.0028** | LOSS (▼) |
| `bright_robotics` | 0.0426 | 0.0416 | **-0.0010** | LOSS (▼) |
| `bright_stackoverflow` | 0.0581 | 0.0590 | **+0.0009** | WIN (▲) |
| `bright_earth_science` | 0.0560 | 0.0578 | **+0.0018** | WIN (▲) |
| `trec_covid` | 0.6640 | 0.6400 | **-0.0240** | LOSS (▼) |
| `bright_aops` | 0.0306 | 0.0315 | **+0.0009** | WIN (▲) |
| `bright_theoremqa_questions` | 0.0155 | 0.0144 | **-0.0011** | LOSS (▼) |
| `webis_touche2020` | 0.2776 | 0.2633 | **-0.0143** | LOSS (▼) |
| `bright_leetcode` | 0.0592 | 0.0585 | **-0.0007** | LOSS (▼) |
| `quora` | 0.1170 | 0.1144 | **-0.0026** | LOSS (▼) |
| **Mean (20 Datasets)** | **0.1049** | **0.1018** | **-0.0031** | **4W / 16L / 0T** |

### B. DPH vs V8_DPH
| Dataset | Baseline | V8 | $\Delta$ p_10 | Outcome |
|:---|:---:|:---:|:---:|:---:|
| `scifact` | 0.0883 | 0.0883 | **+0.0000** | TIE (=) |
| `nfcorpus` | 0.2325 | 0.2288 | **-0.0037** | LOSS (▼) |
| `arguana` | 0.0699 | 0.0678 | **-0.0021** | LOSS (▼) |
| `bright_pony` | 0.0661 | 0.0589 | **-0.0072** | LOSS (▼) |
| `bright_theoremqa_theorems` | 0.0066 | 0.0066 | **+0.0000** | TIE (=) |
| `scidocs` | 0.0780 | 0.0762 | **-0.0018** | LOSS (▼) |
| `bright_economics` | 0.0806 | 0.0777 | **-0.0029** | LOSS (▼) |
| `bright_psychology` | 0.0822 | 0.0832 | **+0.0010** | WIN (▲) |
| `bright_biology` | 0.0835 | 0.0796 | **-0.0039** | LOSS (▼) |
| `fiqa` | 0.0670 | 0.0642 | **-0.0028** | LOSS (▼) |
| `bright_sustainable_living` | 0.0657 | 0.0639 | **-0.0018** | LOSS (▼) |
| `bright_robotics` | 0.0614 | 0.0604 | **-0.0010** | LOSS (▼) |
| `bright_stackoverflow` | 0.0658 | 0.0641 | **-0.0017** | LOSS (▼) |
| `bright_earth_science` | 0.1405 | 0.1397 | **-0.0008** | LOSS (▼) |
| `trec_covid` | 0.6720 | 0.6560 | **-0.0160** | LOSS (▼) |
| `bright_aops` | 0.0306 | 0.0306 | **+0.0000** | TIE (=) |
| `bright_theoremqa_questions` | 0.0139 | 0.0139 | **+0.0000** | TIE (=) |
| `webis_touche2020` | 0.3735 | 0.3531 | **-0.0204** | LOSS (▼) |
| `bright_leetcode` | 0.0570 | 0.0549 | **-0.0021** | LOSS (▼) |
| `quora` | 0.0745 | 0.0723 | **-0.0022** | LOSS (▼) |
| **Mean (20 Datasets)** | **0.1205** | **0.1170** | **-0.0035** | **1W / 15L / 4T** |

## 1. Metric: Online Latency P50 (ms)

### A. BM25 vs V8_BM25
| Dataset | Baseline | V8 | $\Delta$ retrieval_api_p50_ms | Outcome |
|:---|:---:|:---:|:---:|:---:|
| `scifact` | 21.88 | 22.42 | **+0.54** | WIN (▲) |
| `nfcorpus` | 6.51 | 7.46 | **+0.95** | WIN (▲) |
| `arguana` | 27.67 | 30.91 | **+3.24** | WIN (▲) |
| `bright_pony` | 28.71 | 30.25 | **+1.54** | WIN (▲) |
| `bright_theoremqa_theorems` | 13.45 | 16.35 | **+2.90** | WIN (▲) |
| `scidocs` | 23.02 | 23.60 | **+0.58** | WIN (▲) |
| `bright_economics` | 30.87 | 35.20 | **+4.33** | WIN (▲) |
| `bright_psychology` | 31.49 | 35.15 | **+3.66** | WIN (▲) |
| `bright_biology` | 29.63 | 31.77 | **+2.14** | WIN (▲) |
| `fiqa` | 24.15 | 24.85 | **+0.70** | WIN (▲) |
| `bright_sustainable_living` | 30.50 | 33.50 | **+3.00** | WIN (▲) |
| `bright_robotics` | 31.20 | 35.16 | **+3.96** | WIN (▲) |
| `bright_stackoverflow` | 40.40 | 44.55 | **+4.15** | WIN (▲) |
| `bright_earth_science` | 29.74 | 32.24 | **+2.50** | WIN (▲) |
| `trec_covid` | 20.48 | 21.00 | **+0.52** | WIN (▲) |
| `bright_aops` | 314.96 | 45.36 | **-269.60** | LOSS (▼) |
| `bright_theoremqa_questions` | 51.98 | 43.81 | **-8.17** | LOSS (▼) |
| `webis_touche2020` | 16.97 | 17.73 | **+0.76** | WIN (▲) |
| `bright_leetcode` | 123.51 | 93.19 | **-30.32** | LOSS (▼) |
| `quora` | 13.36 | 14.27 | **+0.91** | WIN (▲) |
| **Mean (20 Datasets)** | **45.52** | **31.94** | **-13.59** | **17W / 3L / 0T** |

### B. DPH vs V8_DPH
| Dataset | Baseline | V8 | $\Delta$ retrieval_api_p50_ms | Outcome |
|:---|:---:|:---:|:---:|:---:|
| `scifact` | 21.79 | 22.42 | **+0.63** | WIN (▲) |
| `nfcorpus` | 6.55 | 7.40 | **+0.85** | WIN (▲) |
| `arguana` | 28.06 | 31.50 | **+3.44** | WIN (▲) |
| `bright_pony` | 26.69 | 30.27 | **+3.58** | WIN (▲) |
| `bright_theoremqa_theorems` | 13.84 | 16.97 | **+3.13** | WIN (▲) |
| `scidocs` | 23.08 | 23.72 | **+0.64** | WIN (▲) |
| `bright_economics` | 32.00 | 34.25 | **+2.25** | WIN (▲) |
| `bright_psychology` | 31.81 | 34.30 | **+2.49** | WIN (▲) |
| `bright_biology` | 27.85 | 31.72 | **+3.87** | WIN (▲) |
| `fiqa` | 24.18 | 24.56 | **+0.38** | WIN (▲) |
| `bright_sustainable_living` | 29.29 | 34.09 | **+4.80** | WIN (▲) |
| `bright_robotics` | 31.62 | 35.49 | **+3.87** | WIN (▲) |
| `bright_stackoverflow` | 42.55 | 45.14 | **+2.59** | WIN (▲) |
| `bright_earth_science` | 29.47 | 32.41 | **+2.94** | WIN (▲) |
| `trec_covid` | 21.94 | 36.24 | **+14.30** | WIN (▲) |
| `bright_aops` | 313.51 | 45.48 | **-268.03** | LOSS (▼) |
| `bright_theoremqa_questions` | 53.78 | 44.89 | **-8.89** | LOSS (▼) |
| `webis_touche2020` | 17.81 | 18.31 | **+0.50** | WIN (▲) |
| `bright_leetcode` | 129.03 | 96.49 | **-32.54** | LOSS (▼) |
| `quora` | 14.18 | 14.94 | **+0.76** | WIN (▲) |
| **Mean (20 Datasets)** | **45.95** | **33.03** | **-12.92** | **17W / 3L / 0T** |
