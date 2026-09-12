# PyTerrier Baseline Evaluation Summary (Default Baselines Suite)

Generated on 2026-09-12 15:04:57

### 1. Headline Retrieval Quality (Linear nDCG@10, Supplemental Exp-nDCG, MRR@10, P@10)

| dataset                    | pipeline                 |   ndcg_10 |   exp_ndcg_10 |   ndcg_100 |   map_100 |   mrr_10 |   p_10 |   strict_10 |
|:---------------------------|:-------------------------|----------:|--------------:|-----------:|----------:|---------:|-------:|------------:|
| scifact                    | BM25_Default             |    0.6839 |        0.6839 |     0.7062 |    0.6376 |   0.6441 | 0.0917 |      0.8433 |
| scifact                    | BM25_RM3_Terrier_Default |    0.6731 |        0.6731 |     0.6979 |    0.6313 |   0.6333 | 0.0897 |      0.8167 |
| scifact                    | BM25_Bo1_Terrier_Default |    0.6351 |        0.6351 |     0.6571 |    0.571  |   0.5778 | 0.092  |      0.8533 |
| scifact                    | DPH                      |    0.6716 |        0.6716 |     0.6968 |    0.6299 |   0.6349 | 0.0883 |      0.82   |
| nfcorpus                   | BM25_Default             |    0.3282 |        0.3295 |     0.2765 |    0.1463 |   0.5321 | 0.2393 |      0.6997 |
| nfcorpus                   | BM25_RM3_Terrier_Default |    0.3504 |        0.3517 |     0.3154 |    0.1684 |   0.5359 | 0.2591 |      0.7059 |
| nfcorpus                   | BM25_Bo1_Terrier_Default |    0.3377 |        0.3385 |     0.3083 |    0.1638 |   0.5122 | 0.2548 |      0.6842 |
| nfcorpus                   | DPH                      |    0.3221 |        0.3238 |     0.2729 |    0.1435 |   0.5254 | 0.2325 |      0.6904 |
| fiqa                       | BM25_Default             |    0.2526 |        0.2526 |     0.3175 |    0.2087 |   0.3104 | 0.0704 |      0.4907 |
| fiqa                       | BM25_RM3_Terrier_Default |    0.2432 |        0.2432 |     0.3055 |    0.2031 |   0.2924 | 0.0665 |      0.463  |
| fiqa                       | BM25_Bo1_Terrier_Default |    0.2343 |        0.2343 |     0.3012 |    0.188  |   0.276  | 0.0698 |      0.4815 |
| fiqa                       | DPH                      |    0.2433 |        0.2433 |     0.305  |    0.1985 |   0.3039 | 0.067  |      0.4799 |
| arguana                    | BM25_Default             |    0.3662 |        0.3662 |     0.4136 |    0.2519 |   0.2408 | 0.0763 |      0.7632 |
| arguana                    | BM25_RM3_Terrier_Default |    0.3632 |        0.3632 |     0.413  |    0.252  |   0.2402 | 0.0752 |      0.7525 |
| arguana                    | BM25_Bo1_Terrier_Default |    0.3441 |        0.3441 |     0.3958 |    0.2284 |   0.216  | 0.0759 |      0.7589 |
| arguana                    | DPH                      |    0.3266 |        0.3266 |     0.3831 |    0.2229 |   0.2101 | 0.0699 |      0.6991 |
| scidocs                    | BM25_Default             |    0.1582 |        0.1582 |     0.2247 |    0.1079 |   0.277  | 0.0813 |      0.498  |
| scidocs                    | BM25_RM3_Terrier_Default |    0.1572 |        0.1572 |     0.2275 |    0.1118 |   0.2699 | 0.0801 |      0.463  |
| scidocs                    | BM25_Bo1_Terrier_Default |    0.1619 |        0.1619 |     0.2329 |    0.1125 |   0.2745 | 0.085  |      0.509  |
| scidocs                    | DPH                      |    0.1499 |        0.1499 |     0.214  |    0.1011 |   0.2624 | 0.078  |      0.488  |
| quora                      | BM25_Default             |    0.7676 |        0.7676 |     0.7928 |    0.727  |   0.7584 | 0.117  |      0.9109 |
| quora                      | BM25_RM3_Terrier_Default |    0.7508 |        0.7508 |     0.7784 |    0.7129 |   0.7373 | 0.1151 |      0.8877 |
| quora                      | BM25_Bo1_Terrier_Default |    0.7345 |        0.7345 |     0.7631 |    0.6904 |   0.7182 | 0.115  |      0.8919 |
| quora                      | DPH                      |    0.4429 |        0.4429 |     0.4913 |    0.4014 |   0.419  | 0.0745 |      0.6181 |
| hotpotqa                   | BM25_Default             |    0.5858 |        0.5858 |     0.6252 |    0.5022 |   0.7505 | 0.1232 |      0.886  |
| hotpotqa                   | BM25_RM3_Terrier_Default |    0.5286 |        0.5286 |     0.5686 |    0.4474 |   0.7165 | 0.1071 |      0.8305 |
| hotpotqa                   | BM25_Bo1_Terrier_Default |    0.531  |        0.531  |     0.5743 |    0.4477 |   0.6833 | 0.1143 |      0.8477 |
| hotpotqa                   | DPH                      |    0.6169 |        0.6169 |     0.6539 |    0.5322 |   0.7843 | 0.1294 |      0.9111 |
| trec_covid                 | BM25_Default             |    0.6295 |        0.6073 |     0.483  |    0.085  |   0.8725 | 0.664  |      1      |
| trec_covid                 | BM25_RM3_Terrier_Default |    0.6511 |        0.6255 |     0.4849 |    0.0858 |   0.8467 | 0.706  |      0.96   |
| trec_covid                 | BM25_Bo1_Terrier_Default |    0.6348 |        0.613  |     0.5004 |    0.0904 |   0.8107 | 0.688  |      0.96   |
| trec_covid                 | DPH                      |    0.631  |        0.6087 |     0.4684 |    0.0774 |   0.8472 | 0.672  |      0.98   |
| webis_touche2020           | BM25_Default             |    0.2979 |        0.2938 |     0.4278 |    0.1931 |   0.5713 | 0.2776 |      0.9184 |
| webis_touche2020           | BM25_RM3_Terrier_Default |    0.3388 |        0.3358 |     0.456  |    0.2142 |   0.5966 | 0.3102 |      0.9184 |
| webis_touche2020           | BM25_Bo1_Terrier_Default |    0.3551 |        0.3509 |     0.4594 |    0.2231 |   0.6361 | 0.3265 |      0.9184 |
| webis_touche2020           | DPH                      |    0.4174 |        0.4135 |     0.4987 |    0.2535 |   0.7204 | 0.3735 |      0.9592 |
| dbpedia_entity             | BM25_Default             |    0.3087 |        0.2879 |     0.3659 |    0.2137 |   0.5777 | 0.2735 |      0.8    |
| dbpedia_entity             | BM25_RM3_Terrier_Default |    0.302  |        0.2808 |     0.3596 |    0.2144 |   0.5653 | 0.2703 |      0.77   |
| dbpedia_entity             | BM25_Bo1_Terrier_Default |    0.3058 |        0.2843 |     0.3648 |    0.2126 |   0.5788 | 0.2753 |      0.795  |
| dbpedia_entity             | DPH                      |    0.31   |        0.2908 |     0.3642 |    0.2094 |   0.5747 | 0.2783 |      0.8175 |
| nq                         | BM25_Default             |    0.2814 |        0.2814 |     0.3473 |    0.2383 |   0.2434 | 0.0516 |      0.4754 |
| nq                         | BM25_RM3_Terrier_Default |    0.2818 |        0.2818 |     0.3502 |    0.2387 |   0.2401 | 0.0522 |      0.4751 |
| nq                         | BM25_Bo1_Terrier_Default |    0.2869 |        0.2869 |     0.3546 |    0.2431 |   0.2476 | 0.0529 |      0.4844 |
| nq                         | DPH                      |    0.2846 |        0.2846 |     0.3484 |    0.2404 |   0.2436 | 0.0523 |      0.4797 |
| climate_fever              | BM25_Default             |    0.1386 |        0.1386 |     0.1949 |    0.104  |   0.1887 | 0.0442 |      0.3681 |
| climate_fever              | BM25_RM3_Terrier_Default |    0.1408 |        0.1408 |     0.1959 |    0.1047 |   0.1848 | 0.0468 |      0.3805 |
| climate_fever              | BM25_Bo1_Terrier_Default |    0.1529 |        0.1529 |     0.2148 |    0.1141 |   0.2053 | 0.0501 |      0.4059 |
| climate_fever              | DPH                      |    0.1719 |        0.1719 |     0.2342 |    0.1268 |   0.2288 | 0.0573 |      0.458  |
| fever                      | BM25_Default             |    0.5077 |        0.5077 |     0.5482 |    0.4539 |   0.4682 | 0.0743 |      0.7156 |
| fever                      | BM25_RM3_Terrier_Default |    0.4747 |        0.4747 |     0.5158 |    0.4202 |   0.4331 | 0.0713 |      0.688  |
| fever                      | BM25_Bo1_Terrier_Default |    0.4863 |        0.4863 |     0.5265 |    0.4288 |   0.4428 | 0.0733 |      0.7048 |
| fever                      | DPH                      |    0.6808 |        0.6808 |     0.7036 |    0.6303 |   0.6567 | 0.0903 |      0.8602 |
| bright_biology             | BM25_Default             |    0.0912 |        0.0912 |     0.1524 |    0.0683 |   0.1294 | 0.0379 |      0.2718 |
| bright_biology             | BM25_RM3_Terrier_Default |    0.0844 |        0.0844 |     0.1347 |    0.0635 |   0.1203 | 0.0369 |      0.233  |
| bright_biology             | BM25_Bo1_Terrier_Default |    0.1025 |        0.1025 |     0.1658 |    0.0815 |   0.1354 | 0.0427 |      0.2621 |
| bright_biology             | DPH                      |    0.2523 |        0.2523 |     0.3305 |    0.2145 |   0.3531 | 0.0835 |      0.4757 |
| bright_earth_science       | BM25_Default             |    0.1201 |        0.1201 |     0.1962 |    0.097  |   0.1583 | 0.056  |      0.319  |
| bright_earth_science       | BM25_RM3_Terrier_Default |    0.0928 |        0.0928 |     0.1647 |    0.0875 |   0.1126 | 0.0414 |      0.2155 |
| bright_earth_science       | BM25_Bo1_Terrier_Default |    0.1648 |        0.1648 |     0.2318 |    0.1425 |   0.2255 | 0.0698 |      0.3276 |
| bright_earth_science       | DPH                      |    0.3539 |        0.3539 |     0.4227 |    0.2865 |   0.4804 | 0.1405 |      0.6724 |
| bright_economics           | BM25_Default             |    0.1177 |        0.1177 |     0.18   |    0.09   |   0.1355 | 0.0563 |      0.2621 |
| bright_economics           | BM25_RM3_Terrier_Default |    0.0871 |        0.0871 |     0.1623 |    0.0791 |   0.0999 | 0.0427 |      0.1553 |
| bright_economics           | BM25_Bo1_Terrier_Default |    0.118  |        0.118  |     0.1808 |    0.09   |   0.1172 | 0.067  |      0.2621 |
| bright_economics           | DPH                      |    0.1725 |        0.1725 |     0.2349 |    0.1313 |   0.2161 | 0.0806 |      0.3883 |
| bright_psychology          | BM25_Default             |    0.0926 |        0.0926 |     0.1529 |    0.0778 |   0.0996 | 0.0426 |      0.2178 |
| bright_psychology          | BM25_RM3_Terrier_Default |    0.079  |        0.079  |     0.1271 |    0.0663 |   0.0784 | 0.0396 |      0.198  |
| bright_psychology          | BM25_Bo1_Terrier_Default |    0.1074 |        0.1074 |     0.1615 |    0.0932 |   0.1054 | 0.0515 |      0.2376 |
| bright_psychology          | DPH                      |    0.1756 |        0.1756 |     0.228  |    0.132  |   0.2215 | 0.0822 |      0.3861 |
| bright_robotics            | BM25_Default             |    0.0996 |        0.0996 |     0.1675 |    0.0779 |   0.1244 | 0.0426 |      0.2871 |
| bright_robotics            | BM25_RM3_Terrier_Default |    0.0866 |        0.0866 |     0.1424 |    0.0734 |   0.1081 | 0.0297 |      0.2079 |
| bright_robotics            | BM25_Bo1_Terrier_Default |    0.1035 |        0.1035 |     0.1721 |    0.0813 |   0.1214 | 0.0446 |      0.2871 |
| bright_robotics            | DPH                      |    0.1543 |        0.1543 |     0.2294 |    0.1217 |   0.1818 | 0.0614 |      0.3762 |
| bright_stackoverflow       | BM25_Default             |    0.1561 |        0.1561 |     0.2279 |    0.1338 |   0.1845 | 0.0581 |      0.3077 |
| bright_stackoverflow       | BM25_RM3_Terrier_Default |    0.148  |        0.148  |     0.2095 |    0.1296 |   0.1585 | 0.0632 |      0.2821 |
| bright_stackoverflow       | BM25_Bo1_Terrier_Default |    0.1616 |        0.1616 |     0.2317 |    0.1452 |   0.1704 | 0.0632 |      0.2991 |
| bright_stackoverflow       | DPH                      |    0.1677 |        0.1677 |     0.2289 |    0.1389 |   0.1987 | 0.0658 |      0.3333 |
| bright_sustainable_living  | BM25_Default             |    0.0981 |        0.0981 |     0.171  |    0.074  |   0.1249 | 0.0491 |      0.3056 |
| bright_sustainable_living  | BM25_RM3_Terrier_Default |    0.0864 |        0.0864 |     0.1543 |    0.0683 |   0.1092 | 0.0426 |      0.25   |
| bright_sustainable_living  | BM25_Bo1_Terrier_Default |    0.0981 |        0.0981 |     0.1742 |    0.075  |   0.1273 | 0.05   |      0.3056 |
| bright_sustainable_living  | DPH                      |    0.1669 |        0.1669 |     0.2404 |    0.1384 |   0.2026 | 0.0657 |      0.3889 |
| bright_leetcode            | BM25_Default             |    0.2496 |        0.2496 |     0.283  |    0.2106 |   0.3028 | 0.0592 |      0.4155 |
| bright_leetcode            | BM25_RM3_Terrier_Default |    0.2183 |        0.2183 |     0.2503 |    0.1911 |   0.2758 | 0.0507 |      0.338  |
| bright_leetcode            | BM25_Bo1_Terrier_Default |    0.2143 |        0.2143 |     0.2534 |    0.1831 |   0.2448 | 0.0556 |      0.3803 |
| bright_leetcode            | DPH                      |    0.2418 |        0.2418 |     0.2819 |    0.207  |   0.2895 | 0.057  |      0.4014 |
| bright_pony                | BM25_Default             |    0.0252 |        0.0252 |     0.0836 |    0.0124 |   0.0636 | 0.0259 |      0.2143 |
| bright_pony                | BM25_RM3_Terrier_Default |    0.0193 |        0.0193 |     0.0613 |    0.0102 |   0.042  | 0.0205 |      0.1339 |
| bright_pony                | BM25_Bo1_Terrier_Default |    0.0254 |        0.0254 |     0.0848 |    0.016  |   0.0441 | 0.0277 |      0.1786 |
| bright_pony                | DPH                      |    0.0639 |        0.0639 |     0.126  |    0.0244 |   0.1452 | 0.0661 |      0.4732 |
| bright_aops                | BM25_Default             |    0.0604 |        0.0604 |     0.1011 |    0.0412 |   0.1245 | 0.0306 |      0.2072 |
| bright_aops                | BM25_RM3_Terrier_Default |    0.0476 |        0.0476 |     0.0775 |    0.0328 |   0.1041 | 0.0216 |      0.1622 |
| bright_aops                | BM25_Bo1_Terrier_Default |    0.047  |        0.047  |     0.0823 |    0.0336 |   0.0966 | 0.0234 |      0.1712 |
| bright_aops                | DPH                      |    0.0536 |        0.0536 |     0.0966 |    0.0364 |   0.1082 | 0.0306 |      0.2072 |
| bright_theoremqa_questions | BM25_Default             |    0.07   |        0.07   |     0.0863 |    0.0607 |   0.0804 | 0.0155 |      0.1134 |
| bright_theoremqa_questions | BM25_RM3_Terrier_Default |    0.0595 |        0.0595 |     0.0713 |    0.0582 |   0.0632 | 0.0124 |      0.0722 |
| bright_theoremqa_questions | BM25_Bo1_Terrier_Default |    0.0503 |        0.0503 |     0.0655 |    0.0454 |   0.0558 | 0.0119 |      0.067  |
| bright_theoremqa_questions | DPH                      |    0.0656 |        0.0656 |     0.0852 |    0.0595 |   0.0686 | 0.0139 |      0.1031 |
| bright_theoremqa_theorems  | BM25_Default             |    0.0192 |        0.0192 |     0.0346 |    0.0119 |   0.0178 | 0.0066 |      0.0658 |
| bright_theoremqa_theorems  | BM25_RM3_Terrier_Default |    0.0082 |        0.0082 |     0.0264 |    0.0065 |   0.006  | 0.0026 |      0.0263 |
| bright_theoremqa_theorems  | BM25_Bo1_Terrier_Default |    0.0248 |        0.0248 |     0.0512 |    0.0217 |   0.0261 | 0.0066 |      0.0658 |
| bright_theoremqa_theorems  | DPH                      |    0.0248 |        0.0248 |     0.0388 |    0.0196 |   0.0307 | 0.0066 |      0.0526 |
| scifact                    | DPH_Bo1_Terrier_Default  |    0.6375 |        0.6375 |     0.6647 |    0.5845 |   0.5899 | 0.089  |      0.82   |
| scifact                    | DPH_RM3_Terrier_Default  |    0.658  |        0.658  |     0.6857 |    0.618  |   0.6198 | 0.0873 |      0.7967 |
| nfcorpus                   | DPH_Bo1_Terrier_Default  |    0.3368 |        0.338  |     0.3074 |    0.1618 |   0.5226 | 0.252  |      0.678  |
| nfcorpus                   | DPH_RM3_Terrier_Default  |    0.3452 |        0.3467 |     0.3165 |    0.1662 |   0.5334 | 0.2536 |      0.6966 |
| fiqa                       | DPH_Bo1_Terrier_Default  |    0.2211 |        0.2211 |     0.2879 |    0.1768 |   0.2667 | 0.0648 |      0.4676 |
| fiqa                       | DPH_RM3_Terrier_Default  |    0.2356 |        0.2356 |     0.297  |    0.1942 |   0.2928 | 0.065  |      0.4522 |
| arguana                    | DPH_Bo1_Terrier_Default  |    0.296  |        0.296  |     0.3644 |    0.1966 |   0.1803 | 0.0674 |      0.6743 |
| arguana                    | DPH_RM3_Terrier_Default  |    0.3285 |        0.3285 |     0.3885 |    0.2267 |   0.2127 | 0.0698 |      0.6984 |
| scidocs                    | DPH_Bo1_Terrier_Default  |    0.1522 |        0.1522 |     0.2222 |    0.1049 |   0.255  | 0.0809 |      0.494  |
| scidocs                    | DPH_RM3_Terrier_Default  |    0.1499 |        0.1499 |     0.2204 |    0.1059 |   0.2602 | 0.0767 |      0.463  |
| quora                      | DPH_Bo1_Terrier_Default  |    0.3585 |        0.3585 |     0.4176 |    0.3159 |   0.3263 | 0.0666 |      0.5475 |
| quora                      | DPH_RM3_Terrier_Default  |    0.3933 |        0.3933 |     0.4467 |    0.3544 |   0.3665 | 0.069  |      0.563  |
| hotpotqa                   | DPH_Bo1_Terrier_Default  |    0.5534 |        0.5534 |     0.5941 |    0.4655 |   0.707  | 0.1201 |      0.8774 |
| hotpotqa                   | DPH_RM3_Terrier_Default  |    0.5651 |        0.5651 |     0.6013 |    0.4815 |   0.7579 | 0.1144 |      0.8644 |
| trec_covid                 | DPH_Bo1_Terrier_Default  |    0.6345 |        0      |     0.4884 |    0.0935 |   0.829  | 0.726  |      0.98   |
| trec_covid                 | DPH_RM3_Terrier_Default  |    0.6192 |        0      |     0.4603 |    0.0859 |   0.8214 | 0.7    |      0.98   |
| webis_touche2020           | DPH_Bo1_Terrier_Default  |    0.4457 |        0.4399 |     0.5298 |    0.2815 |   0.758  | 0.3837 |      0.8776 |
| webis_touche2020           | DPH_RM3_Terrier_Default  |    0.458  |        0.4541 |     0.5347 |    0.2773 |   0.7436 | 0.4082 |      0.9796 |
| dbpedia_entity             | DPH_Bo1_Terrier_Default  |    0.3032 |        0.2829 |     0.3604 |    0.2061 |   0.564  | 0.2807 |      0.8    |
| dbpedia_entity             | DPH_RM3_Terrier_Default  |    0.3083 |        0.2883 |     0.3627 |    0.2115 |   0.5585 | 0.2805 |      0.7875 |
| nq                         | DPH_Bo1_Terrier_Default  |    0.2683 |        0.2683 |     0.3383 |    0.2235 |   0.2245 | 0.0514 |      0.471  |
| nq                         | DPH_RM3_Terrier_Default  |    0.2886 |        0.2886 |     0.3554 |    0.2453 |   0.2473 | 0.0529 |      0.4815 |
| climate_fever              | DPH_Bo1_Terrier_Default  |    0.1925 |        0      |     0.2597 |    0.1413 |   0.2557 | 0.065  |      0.5075 |
| climate_fever              | DPH_RM3_Terrier_Default  |    0.1855 |        0      |     0.2483 |    0.1366 |   0.2413 | 0.0633 |      0.4906 |
| fever                      | DPH_Bo1_Terrier_Default  |    0.6427 |        0.6427 |     0.6654 |    0.5842 |   0.6098 | 0.0891 |      0.847  |
| fever                      | DPH_RM3_Terrier_Default  |    0.6571 |        0.6571 |     0.6791 |    0.6052 |   0.6311 | 0.0884 |      0.8402 |
| bright_biology             | DPH_Bo1_Terrier_Default  |    0.2875 |        0.2875 |     0.3615 |    0.2453 |   0.3876 | 0.1068 |      0.466  |
| bright_biology             | DPH_RM3_Terrier_Default  |    0.2922 |        0.2922 |     0.3491 |    0.2553 |   0.352  | 0.1068 |      0.4466 |
| bright_earth_science       | DPH_Bo1_Terrier_Default  |    0.3001 |        0.3001 |     0.3797 |    0.2487 |   0.3602 | 0.1302 |      0.5776 |
| bright_earth_science       | DPH_RM3_Terrier_Default  |    0.3637 |        0.3637 |     0.4203 |    0.3183 |   0.4353 | 0.1509 |      0.569  |
| bright_economics           | DPH_Bo1_Terrier_Default  |    0.1614 |        0.1614 |     0.2376 |    0.1369 |   0.1777 | 0.0854 |      0.3204 |
| bright_economics           | DPH_RM3_Terrier_Default  |    0.1699 |        0.1699 |     0.2262 |    0.1381 |   0.1708 | 0.0845 |      0.2913 |
| bright_psychology          | DPH_Bo1_Terrier_Default  |    0.1849 |        0.1849 |     0.2393 |    0.1526 |   0.199  | 0.0941 |      0.3366 |
| bright_psychology          | DPH_RM3_Terrier_Default  |    0.1821 |        0.1821 |     0.2255 |    0.1478 |   0.2065 | 0.0891 |      0.3267 |
| bright_robotics            | DPH_Bo1_Terrier_Default  |    0.1173 |        0.1173 |     0.2064 |    0.0969 |   0.1385 | 0.0475 |      0.297  |
| bright_robotics            | DPH_RM3_Terrier_Default  |    0.1474 |        0.1474 |     0.2086 |    0.1216 |   0.1691 | 0.0574 |      0.3069 |
| bright_stackoverflow       | DPH_Bo1_Terrier_Default  |    0.1629 |        0.1629 |     0.2338 |    0.147  |   0.1692 | 0.0675 |      0.3077 |
| bright_stackoverflow       | DPH_RM3_Terrier_Default  |    0.1694 |        0.1694 |     0.2269 |    0.1497 |   0.1782 | 0.0718 |      0.2821 |
| bright_sustainable_living  | DPH_Bo1_Terrier_Default  |    0.1519 |        0.1519 |     0.2397 |    0.1245 |   0.1651 | 0.0685 |      0.3796 |
| bright_sustainable_living  | DPH_RM3_Terrier_Default  |    0.1632 |        0.1632 |     0.2369 |    0.1442 |   0.1905 | 0.0639 |      0.3333 |
| bright_leetcode            | DPH_Bo1_Terrier_Default  |    0.2073 |        0.2073 |     0.2503 |    0.1737 |   0.2361 | 0.0549 |      0.3803 |
| bright_leetcode            | DPH_RM3_Terrier_Default  |    0.2285 |        0.2285 |     0.2592 |    0.1979 |   0.2781 | 0.0521 |      0.3662 |
| bright_pony                | DPH_Bo1_Terrier_Default  |    0.0529 |        0.0529 |     0.1315 |    0.0274 |   0.095  | 0.0616 |      0.4196 |
| bright_pony                | DPH_RM3_Terrier_Default  |    0.0478 |        0.0478 |     0.1047 |    0.0224 |   0.105  | 0.0464 |      0.3214 |
| bright_aops                | DPH_Bo1_Terrier_Default  |    0.0418 |        0.0418 |     0.0769 |    0.0275 |   0.0869 | 0.0243 |      0.1802 |
| bright_aops                | DPH_RM3_Terrier_Default  |    0.0389 |        0.0389 |     0.075  |    0.0271 |   0.0986 | 0.0189 |      0.1441 |
| bright_theoremqa_questions | DPH_Bo1_Terrier_Default  |    0.045  |        0.045  |     0.062  |    0.0411 |   0.0435 | 0.0103 |      0.067  |
| bright_theoremqa_questions | DPH_RM3_Terrier_Default  |    0.046  |        0.046  |     0.0573 |    0.0427 |   0.0445 | 0.0098 |      0.067  |
| bright_theoremqa_theorems  | DPH_Bo1_Terrier_Default  |    0.0272 |        0.0272 |     0.055  |    0.0257 |   0.0345 | 0.0066 |      0.0526 |
| bright_theoremqa_theorems  | DPH_RM3_Terrier_Default  |    0.0286 |        0.0286 |     0.0423 |    0.0251 |   0.03   | 0.0079 |      0.0526 |
| scifact                    | BGE_Small_Dense          |    0.7159 |        0.7159 |     0.7446 |    0.6818 |   0.684  | 0.0937 |      0.84   |
| scifact                    | SPLADE_v3_PISA           |    0.687  |        0.687  |     0.7135 |    0.6558 |   0.6652 | 0.0887 |      0.7967 |
| nfcorpus                   | BGE_Small_Dense          |    0.3367 |        0.3382 |     0.3043 |    0.1549 |   0.5295 | 0.2508 |      0.6997 |
| nfcorpus                   | SPLADE_v3_PISA           |    0.347  |        0.3483 |     0.3033 |    0.1611 |   0.5737 | 0.2495 |      0.7152 |
| arguana                    | BGE_Small_Dense          |    0.4262 |        0.4262 |     0.4612 |    0.3032 |   0.2946 | 0.0834 |      0.8336 |
| arguana                    | SPLADE_v3_PISA           |    0.3633 |        0.3633 |     0.4139 |    0.2497 |   0.2377 | 0.0762 |      0.7624 |
| bright_pony                | BGE_Small_Dense          |    0.0236 |        0.0236 |     0.0917 |    0.0134 |   0.048  | 0.0286 |      0.2232 |
| bright_pony                | SPLADE_v3_PISA           |    0.2417 |        0.2417 |     0.2551 |    0.082  |   0.656  | 0.1768 |      0.8839 |
| bright_theoremqa_theorems  | BGE_Small_Dense          |    0.0588 |        0.0588 |     0.0875 |    0.0429 |   0.0605 | 0.0171 |      0.1579 |
| bright_theoremqa_theorems  | SPLADE_v3_PISA           |    0.0301 |        0.0301 |     0.0635 |    0.029  |   0.0278 | 0.0079 |      0.0658 |
| scidocs                    | BGE_Small_Dense          |    0.1941 |        0.1941 |     0.2805 |    0.1358 |   0.3306 | 0.1014 |      0.588  |
| scidocs                    | SPLADE_v3_PISA           |    0.1472 |        0.1472 |     0.216  |    0.1008 |   0.268  | 0.0746 |      0.481  |
| bright_economics           | BGE_Small_Dense          |    0.1225 |        0.1225 |     0.1835 |    0.0949 |   0.1337 | 0.066  |      0.2816 |
| bright_economics           | SPLADE_v3_PISA           |    0.1316 |        0.1316 |     0.1976 |    0.1008 |   0.1662 | 0.0631 |      0.301  |
| bright_psychology          | BGE_Small_Dense          |    0.1356 |        0.1356 |     0.1909 |    0.1049 |   0.1676 | 0.0733 |      0.3168 |
| bright_psychology          | SPLADE_v3_PISA           |    0.1256 |        0.1256 |     0.1768 |    0.093  |   0.1612 | 0.0673 |      0.3069 |
| bright_biology             | BGE_Small_Dense          |    0.0915 |        0.0915 |     0.1576 |    0.0692 |   0.1311 | 0.0447 |      0.2621 |
| bright_biology             | SPLADE_v3_PISA           |    0.1589 |        0.1589 |     0.2401 |    0.124  |   0.2292 | 0.068  |      0.3883 |
| fiqa                       | BGE_Small_Dense          |    0.3713 |        0.3713 |     0.4395 |    0.318  |   0.451  | 0.1012 |      0.6312 |
| fiqa                       | SPLADE_v3_PISA           |    0.3365 |        0.3365 |     0.395  |    0.281  |   0.4116 | 0.0909 |      0.6034 |
| bright_sustainable_living  | BGE_Small_Dense          |    0.1002 |        0.1002 |     0.1788 |    0.0833 |   0.1344 | 0.0426 |      0.25   |
| bright_sustainable_living  | SPLADE_v3_PISA           |    0.1309 |        0.1309 |     0.2239 |    0.1033 |   0.1625 | 0.0583 |      0.3889 |
| bright_robotics            | BGE_Small_Dense          |    0.1077 |        0.1077 |     0.1504 |    0.078  |   0.1486 | 0.0406 |      0.2871 |
| bright_robotics            | SPLADE_v3_PISA           |    0.1226 |        0.1226 |     0.1848 |    0.0953 |   0.1623 | 0.0505 |      0.297  |
| bright_stackoverflow       | BGE_Small_Dense          |    0.0798 |        0.0798 |     0.1551 |    0.0639 |   0.0967 | 0.0376 |      0.2479 |
| bright_stackoverflow       | SPLADE_v3_PISA           |    0.0859 |        0.0859 |     0.1543 |    0.0608 |   0.0984 | 0.0385 |      0.265  |
| bright_earth_science       | BGE_Small_Dense          |    0.2285 |        0.2285 |     0.2843 |    0.1926 |   0.2732 | 0.0784 |      0.4828 |
| bright_earth_science       | SPLADE_v3_PISA           |    0.2108 |        0.2108 |     0.2744 |    0.1708 |   0.2425 | 0.0784 |      0.4828 |
| trec_covid                 | BGE_Small_Dense          |    0.6494 |        0.6309 |     0.491  |    0.0828 |   0.869  | 0.682  |      0.98   |
| trec_covid                 | SPLADE_v3_PISA           |    0.6918 |        0.6675 |     0.5158 |    0.0896 |   0.914  | 0.732  |      1      |
| bright_aops                | BGE_Small_Dense          |    0.055  |        0.055  |     0.1011 |    0.0414 |   0.1023 | 0.0297 |      0.1982 |
| bright_aops                | SPLADE_v3_PISA           |    0.0842 |        0.0842 |     0.1331 |    0.0583 |   0.1474 | 0.0459 |      0.2883 |
| bright_theoremqa_questions | BGE_Small_Dense          |    0.1095 |        0      |     0.1328 |    0.0964 |   0.1221 | 0.0242 |      0.1804 |
| bright_theoremqa_questions | SPLADE_v3_PISA           |    0.0815 |        0      |     0.109  |    0.0715 |   0.1007 | 0.0186 |      0.1495 |
| webis_touche2020           | BGE_Small_Dense          |    0.1783 |        0.1756 |     0.3117 |    0.1235 |   0.3536 | 0.1612 |      0.6735 |
| webis_touche2020           | SPLADE_v3_PISA           |    0.3047 |        0.3019 |     0.4062 |    0.1824 |   0.5535 | 0.2776 |      0.9184 |
| bright_leetcode            | BGE_Small_Dense          |    0.2382 |        0.2382 |     0.2681 |    0.1924 |   0.2882 | 0.0606 |      0.4296 |
| bright_leetcode            | SPLADE_v3_PISA           |    0.2757 |        0.2757 |     0.3168 |    0.2295 |   0.3208 | 0.069  |      0.4859 |
| quora                      | BGE_Small_Dense          |    0.8834 |        0.8834 |     0.8955 |    0.8523 |   0.8759 | 0.1338 |      0.9789 |
| quora                      | SPLADE_v3_PISA           |    0.8172 |        0.8172 |     0.8362 |    0.7774 |   0.8077 | 0.1247 |      0.9492 |
| nq                         | BGE_Small_Dense          |    0.4181 |        0.4181 |     0.4743 |    0.3579 |   0.37   | 0.0735 |      0.6567 |
| nq                         | SPLADE_v3_PISA           |    0.5499 |        0.5499 |     0.5935 |    0.488  |   0.5022 | 0.0884 |      0.7775 |
| dbpedia_entity             | BGE_Small_Dense          |    0.3832 |        0.3696 |     0.4245 |    0.2458 |   0.7301 | 0.2958 |      0.88   |
| dbpedia_entity             | SPLADE_v3_PISA           |    0.4254 |        0.4099 |     0.4782 |    0.2972 |   0.7274 | 0.3497 |      0.8975 |
| hotpotqa                   | BGE_Small_Dense          |    0.654  |        0.654  |     0.6878 |    0.5774 |   0.7958 | 0.1377 |      0.9114 |
| hotpotqa                   | SPLADE_v3_PISA           |    0.6759 |        0.6759 |     0.7057 |    0.5915 |   0.8601 | 0.1379 |      0.9495 |
| climate_fever              | BGE_Small_Dense          |    0.2786 |        0.2786 |     0.351  |    0.2149 |   0.3721 | 0.088  |      0.6352 |
| climate_fever              | SPLADE_v3_PISA           |    0.2091 |        0.2091 |     0.277  |    0.159  |   0.2847 | 0.0655 |      0.5101 |
| fever                      | BGE_Small_Dense          |    0.8194 |        0.8194 |     0.8294 |    0.7765 |   0.8216 | 0.1007 |      0.9598 |
| fever                      | SPLADE_v3_PISA           |    0.7684 |        0.7684 |     0.7858 |    0.7257 |   0.7648 | 0.0964 |      0.9167 |

### 2. Candidate Funnel Ceiling Diagnostics (Recall@K, Completeness@K, Oracle-nDCG@10)

| dataset                    | pipeline                 |   recall_100 |   recall_500 |   recall_1000 |   completeness_100 |   completeness_500 |   completeness_1000 |   oracle_ndcg_10 |
|:---------------------------|:-------------------------|-------------:|-------------:|--------------:|-------------------:|-------------------:|--------------------:|-----------------:|
| scifact                    | BM25_Default             |       0.926  |       0.97   |        0.9767 |             0.92   |             0.97   |              0.9767 |           0.9767 |
| scifact                    | BM25_RM3_Terrier_Default |       0.922  |       0.9633 |        0.985  |             0.92   |             0.96   |              0.9833 |           0.9854 |
| scifact                    | BM25_Bo1_Terrier_Default |       0.9293 |       0.965  |        0.98   |             0.9233 |             0.9633 |              0.98   |           0.98   |
| scifact                    | DPH                      |       0.9159 |       0.9667 |        0.9733 |             0.9067 |             0.9667 |              0.9733 |           0.9733 |
| nfcorpus                   | BM25_Default             |       0.2483 |       0.3305 |        0.365  |             0.0557 |             0.065  |              0.0743 |           0.647  |
| nfcorpus                   | BM25_RM3_Terrier_Default |       0.3114 |       0.4672 |        0.55   |             0.0774 |             0.1115 |              0.1269 |           0.8052 |
| nfcorpus                   | BM25_Bo1_Terrier_Default |       0.3095 |       0.4617 |        0.5514 |             0.0867 |             0.1269 |              0.1455 |           0.7996 |
| nfcorpus                   | DPH                      |       0.2449 |       0.3323 |        0.3671 |             0.0557 |             0.0681 |              0.0743 |           0.6503 |
| fiqa                       | BM25_Default             |       0.5592 |       0.7114 |        0.7742 |             0.3735 |             0.5309 |              0.6049 |           0.8085 |
| fiqa                       | BM25_RM3_Terrier_Default |       0.5377 |       0.7131 |        0.7803 |             0.3673 |             0.5525 |              0.6219 |           0.8127 |
| fiqa                       | BM25_Bo1_Terrier_Default |       0.5674 |       0.7295 |        0.7952 |             0.3935 |             0.5571 |              0.6358 |           0.8262 |
| fiqa                       | DPH                      |       0.5393 |       0.6961 |        0.7562 |             0.3596 |             0.5123 |              0.5787 |           0.7923 |
| arguana                    | BM25_Default             |       0.9701 |       0.99   |        0.9922 |             0.9701 |             0.99   |              0.9922 |           0.9922 |
| arguana                    | BM25_RM3_Terrier_Default |       0.968  |       0.99   |        0.9936 |             0.968  |             0.99   |              0.9936 |           0.9936 |
| arguana                    | BM25_Bo1_Terrier_Default |       0.9808 |       0.9929 |        0.9943 |             0.9808 |             0.9929 |              0.9943 |           0.9943 |
| arguana                    | DPH                      |       0.9509 |       0.9851 |        0.99   |             0.9509 |             0.9851 |              0.99   |           0.99   |
| scidocs                    | BM25_Default             |       0.362  |       0.5106 |        0.5713 |             0.055  |             0.125  |              0.158  |           0.663  |
| scidocs                    | BM25_RM3_Terrier_Default |       0.3675 |       0.518  |        0.5884 |             0.063  |             0.134  |              0.188  |           0.6765 |
| scidocs                    | BM25_Bo1_Terrier_Default |       0.3823 |       0.5357 |        0.6006 |             0.065  |             0.138  |              0.195  |           0.6878 |
| scidocs                    | DPH                      |       0.3475 |       0.4974 |        0.5649 |             0.044  |             0.115  |              0.155  |           0.6579 |
| quora                      | BM25_Default             |       0.966  |       0.9878 |        0.9926 |             0.9424 |             0.9749 |              0.9833 |           0.9942 |
| quora                      | BM25_RM3_Terrier_Default |       0.9594 |       0.9842 |        0.9904 |             0.9376 |             0.9723 |              0.9824 |           0.9918 |
| quora                      | BM25_Bo1_Terrier_Default |       0.963  |       0.9868 |        0.9922 |             0.9398 |             0.9743 |              0.9841 |           0.9937 |
| quora                      | DPH                      |       0.7805 |       0.8696 |        0.9026 |             0.7418 |             0.8397 |              0.8777 |           0.9079 |
| hotpotqa                   | BM25_Default             |       0.7701 |       0.8413 |        0.8689 |             0.5739 |             0.6964 |              0.7458 |           0.8968 |
| hotpotqa                   | BM25_RM3_Terrier_Default |       0.6945 |       0.7874 |        0.8248 |             0.4586 |             0.6032 |              0.6678 |           0.8604 |
| hotpotqa                   | BM25_Bo1_Terrier_Default |       0.7413 |       0.8282 |        0.8599 |             0.5291 |             0.6718 |              0.7291 |           0.8895 |
| hotpotqa                   | DPH                      |       0.7913 |       0.8573 |        0.8824 |             0.6093 |             0.7249 |              0.771  |           0.9076 |
| trec_covid                 | BM25_Default             |       0.1237 |       0.3426 |        0.4593 |             0      |             0      |              0      |           0.9987 |
| trec_covid                 | BM25_RM3_Terrier_Default |       0.1219 |       0.3521 |        0.467  |             0      |             0      |              0      |           0.9916 |
| trec_covid                 | BM25_Bo1_Terrier_Default |       0.1273 |       0.3644 |        0.4789 |             0      |             0      |              0      |           0.9973 |
| trec_covid                 | DPH                      |       0.1136 |       0.3068 |        0.4102 |             0      |             0      |              0      |           1      |
| webis_touche2020           | BM25_Default             |       0.5373 |       0.792  |        0.8781 |             0.0612 |             0.1837 |              0.3673 |           0.995  |
| webis_touche2020           | BM25_RM3_Terrier_Default |       0.5582 |       0.8027 |        0.8909 |             0.0816 |             0.2449 |              0.4082 |           0.9963 |
| webis_touche2020           | BM25_Bo1_Terrier_Default |       0.5497 |       0.7989 |        0.8832 |             0.0612 |             0.2449 |              0.3673 |           0.995  |
| webis_touche2020           | DPH                      |       0.5496 |       0.7779 |        0.8543 |             0.0204 |             0.2041 |              0.2857 |           0.9936 |
| dbpedia_entity             | BM25_Default             |       0.4589 |       0.6215 |        0.6695 |             0.0875 |             0.1775 |              0.2075 |           0.8417 |
| dbpedia_entity             | BM25_RM3_Terrier_Default |       0.4577 |       0.619  |        0.6731 |             0.1    |             0.185  |              0.225  |           0.8397 |
| dbpedia_entity             | BM25_Bo1_Terrier_Default |       0.468  |       0.6314 |        0.6834 |             0.0975 |             0.185  |              0.225  |           0.8435 |
| dbpedia_entity             | DPH                      |       0.4607 |       0.6108 |        0.6663 |             0.1    |             0.175  |              0.2025 |           0.8435 |
| nq                         | BM25_Default             |       0.7387 |       0.8532 |        0.8906 |             0.7112 |             0.836  |              0.8769 |           0.8936 |
| nq                         | BM25_RM3_Terrier_Default |       0.7522 |       0.8713 |        0.9048 |             0.7297 |             0.8581 |              0.894  |           0.9073 |
| nq                         | BM25_Bo1_Terrier_Default |       0.7543 |       0.8745 |        0.909  |             0.7306 |             0.8612 |              0.8986 |           0.9112 |
| nq                         | DPH                      |       0.7328 |       0.8524 |        0.8926 |             0.7065 |             0.8352 |              0.8772 |           0.8959 |
| climate_fever              | BM25_Default             |       0.3796 |       0.5305 |        0.595  |             0.1466 |             0.2417 |              0.3205 |           0.6568 |
| climate_fever              | BM25_RM3_Terrier_Default |       0.3835 |       0.5152 |        0.5789 |             0.157  |             0.2612 |              0.3218 |           0.635  |
| climate_fever              | BM25_Bo1_Terrier_Default |       0.4201 |       0.5711 |        0.6344 |             0.1681 |             0.301  |              0.37   |           0.6913 |
| climate_fever              | DPH                      |       0.4447 |       0.5916 |        0.6513 |             0.1772 |             0.3023 |              0.3726 |           0.7111 |
| fever                      | BM25_Default             |       0.8606 |       0.9236 |        0.9409 |             0.8272 |             0.893  |              0.913  |           0.9475 |
| fever                      | BM25_RM3_Terrier_Default |       0.8367 |       0.9098 |        0.9304 |             0.8036 |             0.8794 |              0.9029 |           0.9367 |
| fever                      | BM25_Bo1_Terrier_Default |       0.8508 |       0.9182 |        0.94   |             0.8174 |             0.8879 |              0.9125 |           0.9463 |
| fever                      | DPH                      |       0.9182 |       0.9476 |        0.9572 |             0.8858 |             0.92   |              0.932  |           0.963  |
| bright_biology             | BM25_Default             |       0.3389 |       0.5567 |        0.685  |             0.1359 |             0.2718 |              0.4369 |           0.7294 |
| bright_biology             | BM25_RM3_Terrier_Default |       0.2921 |       0.5307 |        0.6454 |             0.1262 |             0.2913 |              0.4466 |           0.6898 |
| bright_biology             | BM25_Bo1_Terrier_Default |       0.3601 |       0.5984 |        0.7086 |             0.1553 |             0.3204 |              0.5146 |           0.7453 |
| bright_biology             | DPH                      |       0.5172 |       0.7129 |        0.7758 |             0.2524 |             0.4757 |              0.5728 |           0.8093 |
| bright_earth_science       | BM25_Default             |       0.405  |       0.6006 |        0.6823 |             0.2241 |             0.3793 |              0.4741 |           0.7485 |
| bright_earth_science       | BM25_RM3_Terrier_Default |       0.3572 |       0.5264 |        0.6022 |             0.2328 |             0.3621 |              0.4052 |           0.6664 |
| bright_earth_science       | BM25_Bo1_Terrier_Default |       0.407  |       0.6265 |        0.7073 |             0.25   |             0.4138 |              0.5172 |           0.767  |
| bright_earth_science       | DPH                      |       0.6187 |       0.7309 |        0.7812 |             0.3966 |             0.5172 |              0.5776 |           0.8341 |
| bright_economics           | BM25_Default             |       0.3885 |       0.5992 |        0.6961 |             0.2136 |             0.3689 |              0.466  |           0.7624 |
| bright_economics           | BM25_RM3_Terrier_Default |       0.3964 |       0.5586 |        0.6466 |             0.2621 |             0.3689 |              0.4563 |           0.7117 |
| bright_economics           | BM25_Bo1_Terrier_Default |       0.4053 |       0.6355 |        0.7368 |             0.2427 |             0.4175 |              0.5146 |           0.798  |
| bright_economics           | DPH                      |       0.4498 |       0.6445 |        0.7256 |             0.2621 |             0.4369 |              0.5049 |           0.7945 |
| bright_psychology          | BM25_Default             |       0.3575 |       0.528  |        0.6385 |             0.2673 |             0.3465 |              0.4455 |           0.7086 |
| bright_psychology          | BM25_RM3_Terrier_Default |       0.2845 |       0.4509 |        0.5868 |             0.198  |             0.3069 |              0.4356 |           0.6482 |
| bright_psychology          | BM25_Bo1_Terrier_Default |       0.3435 |       0.5461 |        0.6592 |             0.2376 |             0.3663 |              0.4851 |           0.7248 |
| bright_psychology          | DPH                      |       0.4452 |       0.5941 |        0.6922 |             0.3168 |             0.4059 |              0.4851 |           0.7624 |
| bright_robotics            | BM25_Default             |       0.3813 |       0.5849 |        0.7051 |             0.2574 |             0.3861 |              0.495  |           0.7697 |
| bright_robotics            | BM25_RM3_Terrier_Default |       0.3044 |       0.5448 |        0.6467 |             0.1683 |             0.3861 |              0.4752 |           0.7092 |
| bright_robotics            | BM25_Bo1_Terrier_Default |       0.4027 |       0.6499 |        0.7501 |             0.2772 |             0.4554 |              0.5743 |           0.8029 |
| bright_robotics            | DPH                      |       0.4745 |       0.6746 |        0.7673 |             0.3267 |             0.495  |              0.5644 |           0.8191 |
| bright_stackoverflow       | BM25_Default             |       0.4526 |       0.628  |        0.6781 |             0.3333 |             0.4957 |              0.5385 |           0.7094 |
| bright_stackoverflow       | BM25_RM3_Terrier_Default |       0.3983 |       0.5681 |        0.5944 |             0.2821 |             0.4701 |              0.4872 |           0.6253 |
| bright_stackoverflow       | BM25_Bo1_Terrier_Default |       0.4423 |       0.6382 |        0.6998 |             0.3419 |             0.5299 |              0.5983 |           0.726  |
| bright_stackoverflow       | DPH                      |       0.4291 |       0.615  |        0.6667 |             0.3077 |             0.4615 |              0.5214 |           0.6984 |
| bright_sustainable_living  | BM25_Default             |       0.408  |       0.6359 |        0.7489 |             0.2593 |             0.4352 |              0.5556 |           0.8025 |
| bright_sustainable_living  | BM25_RM3_Terrier_Default |       0.3703 |       0.6405 |        0.7229 |             0.2407 |             0.4815 |              0.5556 |           0.7724 |
| bright_sustainable_living  | BM25_Bo1_Terrier_Default |       0.4034 |       0.6994 |        0.7756 |             0.2407 |             0.5093 |              0.5833 |           0.8228 |
| bright_sustainable_living  | DPH                      |       0.4593 |       0.7093 |        0.8037 |             0.2963 |             0.5278 |              0.6111 |           0.8608 |
| bright_leetcode            | BM25_Default             |       0.4444 |       0.6835 |        0.7905 |             0.3028 |             0.5423 |              0.6972 |           0.8124 |
| bright_leetcode            | BM25_RM3_Terrier_Default |       0.3755 |       0.5359 |        0.5946 |             0.2606 |             0.3944 |              0.4577 |           0.6257 |
| bright_leetcode            | BM25_Bo1_Terrier_Default |       0.4309 |       0.6439 |        0.7656 |             0.3099 |             0.5141 |              0.662  |           0.7891 |
| bright_leetcode            | DPH                      |       0.4549 |       0.7054 |        0.8164 |             0.3239 |             0.5915 |              0.7254 |           0.8351 |
| bright_pony                | BM25_Default             |       0.1439 |       0.4884 |        0.7426 |             0      |             0.0179 |              0.0625 |           0.9698 |
| bright_pony                | BM25_RM3_Terrier_Default |       0.1113 |       0.3983 |        0.6479 |             0.0089 |             0.0179 |              0.0357 |           0.9568 |
| bright_pony                | BM25_Bo1_Terrier_Default |       0.1491 |       0.5114 |        0.7464 |             0      |             0.0179 |              0.0625 |           0.9733 |
| bright_pony                | DPH                      |       0.1993 |       0.4841 |        0.6766 |             0.0089 |             0.0179 |              0.0268 |           0.9601 |
| bright_aops                | BM25_Default             |       0.1884 |       0.3431 |        0.4085 |             0.018  |             0.1081 |              0.1532 |           0.476  |
| bright_aops                | BM25_RM3_Terrier_Default |       0.1422 |       0.2878 |        0.3634 |             0.027  |             0.0721 |              0.1171 |           0.4255 |
| bright_aops                | BM25_Bo1_Terrier_Default |       0.1602 |       0.3386 |        0.4248 |             0.027  |             0.1171 |              0.1532 |           0.4899 |
| bright_aops                | DPH                      |       0.1916 |       0.3154 |        0.3776 |             0.009  |             0.0811 |              0.1171 |           0.4433 |
| bright_theoremqa_questions | BM25_Default             |       0.16   |       0.2956 |        0.3814 |             0.1237 |             0.2268 |              0.3144 |           0.4004 |
| bright_theoremqa_questions | BM25_RM3_Terrier_Default |       0.1149 |       0.17   |        0.2223 |             0.0876 |             0.1289 |              0.1804 |           0.2344 |
| bright_theoremqa_questions | BM25_Bo1_Terrier_Default |       0.1295 |       0.2621 |        0.3416 |             0.1031 |             0.2113 |              0.2629 |           0.3625 |
| bright_theoremqa_questions | DPH                      |       0.1659 |       0.2964 |        0.3698 |             0.1289 |             0.2268 |              0.2938 |           0.3921 |
| bright_theoremqa_theorems  | BM25_Default             |       0.114  |       0.2445 |        0.3289 |             0.0658 |             0.1974 |              0.25   |           0.3494 |
| bright_theoremqa_theorems  | BM25_RM3_Terrier_Default |       0.1096 |       0.2025 |        0.2845 |             0.0789 |             0.1447 |              0.2237 |           0.3026 |
| bright_theoremqa_theorems  | BM25_Bo1_Terrier_Default |       0.1601 |       0.2467 |        0.3443 |             0.1184 |             0.2105 |              0.2895 |           0.3586 |
| bright_theoremqa_theorems  | DPH                      |       0.0965 |       0.2047 |        0.3297 |             0.0526 |             0.1316 |              0.25   |           0.3518 |
| scifact                    | DPH_Bo1_Terrier_Default  |       0.9203 |       0.965  |        0.9767 |             0.9167 |             0.9633 |              0.9767 |           0.9767 |
| scifact                    | DPH_RM3_Terrier_Default  |       0.9117 |       0.9643 |        0.9867 |             0.9067 |             0.96   |              0.9867 |           0.9867 |
| nfcorpus                   | DPH_Bo1_Terrier_Default  |       0.307  |       0.4611 |        0.5469 |             0.0836 |             0.1269 |              0.1455 |           0.7987 |
| nfcorpus                   | DPH_RM3_Terrier_Default  |       0.317  |       0.4782 |        0.5641 |             0.0867 |             0.1269 |              0.1424 |           0.8147 |
| fiqa                       | DPH_Bo1_Terrier_Default  |       0.551  |       0.7216 |        0.7745 |             0.3719 |             0.5448 |              0.6049 |           0.8076 |
| fiqa                       | DPH_RM3_Terrier_Default  |       0.5276 |       0.7127 |        0.7797 |             0.358  |             0.5417 |              0.608  |           0.8125 |
| arguana                    | DPH_Bo1_Terrier_Default  |       0.9701 |       0.9893 |        0.9922 |             0.9701 |             0.9893 |              0.9922 |           0.9922 |
| arguana                    | DPH_RM3_Terrier_Default  |       0.9616 |       0.9893 |        0.9936 |             0.9616 |             0.9893 |              0.9936 |           0.9936 |
| scidocs                    | DPH_Bo1_Terrier_Default  |       0.372  |       0.5255 |        0.5949 |             0.057  |             0.136  |              0.181  |           0.6836 |
| scidocs                    | DPH_RM3_Terrier_Default  |       0.3615 |       0.5175 |        0.585  |             0.054  |             0.136  |              0.183  |           0.6736 |
| quora                      | DPH_Bo1_Terrier_Default  |       0.7635 |       0.87   |        0.9029 |             0.7269 |             0.8418 |              0.8797 |           0.9078 |
| quora                      | DPH_RM3_Terrier_Default  |       0.7551 |       0.8573 |        0.8886 |             0.7188 |             0.8284 |              0.8643 |           0.8936 |
| hotpotqa                   | DPH_Bo1_Terrier_Default  |       0.7597 |       0.8404 |        0.8682 |             0.5564 |             0.6941 |              0.7448 |           0.8961 |
| hotpotqa                   | DPH_RM3_Terrier_Default  |       0.7144 |       0.8014 |        0.8358 |             0.4878 |             0.6313 |              0.6907 |           0.8686 |
| trec_covid                 | DPH_Bo1_Terrier_Default  |       0.1294 |       0.3483 |        0.4606 |             0      |             0      |              0      |           0.9973 |
| trec_covid                 | DPH_RM3_Terrier_Default  |       0.1203 |       0.3383 |        0.4563 |             0      |             0      |              0      |           0.9947 |
| webis_touche2020           | DPH_Bo1_Terrier_Default  |       0.5782 |       0.7876 |        0.8721 |             0.0408 |             0.2449 |              0.3061 |           0.9945 |
| webis_touche2020           | DPH_RM3_Terrier_Default  |       0.5949 |       0.7918 |        0.8711 |             0.0204 |             0.2449 |              0.3061 |           0.9965 |
| dbpedia_entity             | DPH_Bo1_Terrier_Default  |       0.4681 |       0.6371 |        0.6892 |             0.105  |             0.1925 |              0.2275 |           0.8537 |
| dbpedia_entity             | DPH_RM3_Terrier_Default  |       0.4637 |       0.6241 |        0.6799 |             0.1025 |             0.1775 |              0.2225 |           0.8461 |
| nq                         | DPH_Bo1_Terrier_Default  |       0.7518 |       0.8717 |        0.9087 |             0.73   |             0.8569 |              0.896  |           0.9114 |
| nq                         | DPH_RM3_Terrier_Default  |       0.7506 |       0.8688 |        0.9026 |             0.7289 |             0.8552 |              0.8911 |           0.9051 |
| climate_fever              | DPH_Bo1_Terrier_Default  |       0.4863 |       0.6447 |        0.6997 |             0.2137 |             0.3733 |              0.4391 |           0.7516 |
| climate_fever              | DPH_RM3_Terrier_Default  |       0.4647 |       0.611  |        0.6638 |             0.2039 |             0.3362 |              0.4007 |           0.7161 |
| fever                      | DPH_Bo1_Terrier_Default  |       0.9077 |       0.9426 |        0.9545 |             0.8765 |             0.9154 |              0.9307 |           0.9599 |
| fever                      | DPH_RM3_Terrier_Default  |       0.8964 |       0.9342 |        0.9472 |             0.8647 |             0.9067 |              0.9227 |           0.9528 |
| bright_biology             | DPH_Bo1_Terrier_Default  |       0.5603 |       0.7358 |        0.8097 |             0.3786 |             0.5243 |              0.6699 |           0.832  |
| bright_biology             | DPH_RM3_Terrier_Default  |       0.5156 |       0.696  |        0.7515 |             0.3592 |             0.4854 |              0.5825 |           0.779  |
| bright_earth_science       | DPH_Bo1_Terrier_Default  |       0.6191 |       0.7462 |        0.7925 |             0.4224 |             0.5345 |              0.6207 |           0.8397 |
| bright_earth_science       | DPH_RM3_Terrier_Default  |       0.5808 |       0.7243 |        0.7715 |             0.4397 |             0.5345 |              0.5776 |           0.8212 |
| bright_economics           | DPH_Bo1_Terrier_Default  |       0.4737 |       0.6719 |        0.7418 |             0.3204 |             0.4563 |              0.5146 |           0.8075 |
| bright_economics           | DPH_RM3_Terrier_Default  |       0.4309 |       0.5961 |        0.7048 |             0.301  |             0.4175 |              0.5049 |           0.7691 |
| bright_psychology          | DPH_Bo1_Terrier_Default  |       0.4416 |       0.6466 |        0.7214 |             0.3168 |             0.4851 |              0.5248 |           0.7865 |
| bright_psychology          | DPH_RM3_Terrier_Default  |       0.3943 |       0.5535 |        0.6584 |             0.2871 |             0.4158 |              0.495  |           0.7283 |
| bright_robotics            | DPH_Bo1_Terrier_Default  |       0.473  |       0.6737 |        0.7802 |             0.3168 |             0.495  |              0.6238 |           0.8272 |
| bright_robotics            | DPH_RM3_Terrier_Default  |       0.3963 |       0.6108 |        0.707  |             0.2475 |             0.4158 |              0.5347 |           0.7538 |
| bright_stackoverflow       | DPH_Bo1_Terrier_Default  |       0.4414 |       0.6383 |        0.6955 |             0.3248 |             0.547  |              0.5983 |           0.723  |
| bright_stackoverflow       | DPH_RM3_Terrier_Default  |       0.4073 |       0.5707 |        0.6211 |             0.2821 |             0.453  |              0.5214 |           0.6485 |
| bright_sustainable_living  | DPH_Bo1_Terrier_Default  |       0.5215 |       0.7305 |        0.83   |             0.3796 |             0.5556 |              0.6296 |           0.8785 |
| bright_sustainable_living  | DPH_RM3_Terrier_Default  |       0.4547 |       0.6711 |        0.7577 |             0.3241 |             0.5093 |              0.5833 |           0.8116 |
| bright_leetcode            | DPH_Bo1_Terrier_Default  |       0.4485 |       0.6954 |        0.8241 |             0.331  |             0.5775 |              0.7254 |           0.8418 |
| bright_leetcode            | DPH_RM3_Terrier_Default  |       0.3892 |       0.5406 |        0.6741 |             0.2746 |             0.4085 |              0.5352 |           0.7033 |
| bright_pony                | DPH_Bo1_Terrier_Default  |       0.2198 |       0.5608 |        0.7572 |             0.0089 |             0.0179 |              0.0804 |           0.9719 |
| bright_pony                | DPH_RM3_Terrier_Default  |       0.1702 |       0.4376 |        0.6421 |             0.0089 |             0.0179 |              0.0268 |           0.955  |
| bright_aops                | DPH_Bo1_Terrier_Default  |       0.1596 |       0.327  |        0.3932 |             0.027  |             0.0811 |              0.0991 |           0.4603 |
| bright_aops                | DPH_RM3_Terrier_Default  |       0.148  |       0.289  |        0.3561 |             0.018  |             0.0541 |              0.0811 |           0.4273 |
| bright_theoremqa_questions | DPH_Bo1_Terrier_Default  |       0.1372 |       0.2461 |        0.3426 |             0.1082 |             0.1959 |              0.2526 |           0.3673 |
| bright_theoremqa_questions | DPH_RM3_Terrier_Default  |       0.1044 |       0.1653 |        0.2017 |             0.0773 |             0.1289 |              0.1598 |           0.2143 |
| bright_theoremqa_theorems  | DPH_Bo1_Terrier_Default  |       0.1601 |       0.2245 |        0.3506 |             0.1316 |             0.1842 |              0.2763 |           0.3714 |
| bright_theoremqa_theorems  | DPH_RM3_Terrier_Default  |       0.1042 |       0.196  |        0.302  |             0.0658 |             0.1711 |              0.2368 |           0.3174 |
| scifact                    | BGE_Small_Dense          |       0.96   |       0.99   |        0.9967 |             0.96   |             0.99   |              0.9967 |           0.9967 |
| scifact                    | SPLADE_v3_PISA           |       0.9043 |       0.9667 |        0.9833 |             0.9    |             0.9667 |              0.9833 |           0.9833 |
| nfcorpus                   | BGE_Small_Dense          |       0.3048 |       0.4905 |        0.6312 |             0.065  |             0.1115 |              0.161  |           0.8669 |
| nfcorpus                   | SPLADE_v3_PISA           |       0.2817 |       0.4386 |        0.5598 |             0.0619 |             0.0991 |              0.1269 |           0.8177 |
| arguana                    | BGE_Small_Dense          |       0.9829 |       0.995  |        0.9964 |             0.9829 |             0.995  |              0.9964 |           0.9964 |
| arguana                    | SPLADE_v3_PISA           |       0.9815 |       0.995  |        0.9957 |             0.9815 |             0.995  |              0.9957 |           0.9957 |
| bright_pony                | BGE_Small_Dense          |       0.1668 |       0.4796 |        0.6486 |             0.0089 |             0.0179 |              0.0179 |           0.9406 |
| bright_pony                | SPLADE_v3_PISA           |       0.2795 |       0.5864 |        0.7464 |             0.0089 |             0.0357 |              0.0625 |           0.9804 |
| bright_theoremqa_theorems  | BGE_Small_Dense          |       0.2289 |       0.382  |        0.4695 |             0.1579 |             0.2895 |              0.3684 |           0.4992 |
| bright_theoremqa_theorems  | SPLADE_v3_PISA           |       0.1985 |       0.3805 |        0.4588 |             0.1447 |             0.3026 |              0.3816 |           0.4803 |
| scidocs                    | BGE_Small_Dense          |       0.4619 |       0.6608 |        0.7524 |             0.083  |             0.235  |              0.353  |           0.8178 |
| scidocs                    | SPLADE_v3_PISA           |       0.3542 |       0.5114 |        0.5837 |             0.051  |             0.112  |              0.163  |           0.6747 |
| bright_economics           | BGE_Small_Dense          |       0.3825 |       0.5952 |        0.695  |             0.2427 |             0.3981 |              0.5049 |           0.7607 |
| bright_economics           | SPLADE_v3_PISA           |       0.4144 |       0.64   |        0.7262 |             0.2718 |             0.4078 |              0.5146 |           0.7945 |
| bright_psychology          | BGE_Small_Dense          |       0.3793 |       0.5929 |        0.6498 |             0.2574 |             0.3861 |              0.4257 |           0.711  |
| bright_psychology          | SPLADE_v3_PISA           |       0.356  |       0.6109 |        0.6982 |             0.2376 |             0.4356 |              0.495  |           0.7616 |
| bright_biology             | BGE_Small_Dense          |       0.3376 |       0.6349 |        0.7257 |             0.1068 |             0.3689 |              0.4854 |           0.7743 |
| bright_biology             | SPLADE_v3_PISA           |       0.4718 |       0.7176 |        0.8412 |             0.233  |             0.4854 |              0.6602 |           0.8721 |
| fiqa                       | BGE_Small_Dense          |       0.6824 |       0.8316 |        0.8846 |             0.4877 |             0.6852 |              0.7623 |           0.9026 |
| fiqa                       | SPLADE_v3_PISA           |       0.6204 |       0.765  |        0.825  |             0.4306 |             0.5972 |              0.6713 |           0.8528 |
| bright_sustainable_living  | BGE_Small_Dense          |       0.4028 |       0.5798 |        0.6841 |             0.2407 |             0.3981 |              0.5    |           0.738  |
| bright_sustainable_living  | SPLADE_v3_PISA           |       0.5076 |       0.7476 |        0.8346 |             0.2778 |             0.537  |              0.6574 |           0.8746 |
| bright_robotics            | BGE_Small_Dense          |       0.2983 |       0.4616 |        0.59   |             0.1584 |             0.297  |              0.396  |           0.6621 |
| bright_robotics            | SPLADE_v3_PISA           |       0.3857 |       0.6173 |        0.712  |             0.2376 |             0.3762 |              0.4752 |           0.7829 |
| bright_stackoverflow       | BGE_Small_Dense          |       0.3988 |       0.6619 |        0.74   |             0.265  |             0.4615 |              0.5726 |           0.7737 |
| bright_stackoverflow       | SPLADE_v3_PISA           |       0.3926 |       0.5643 |        0.6139 |             0.2222 |             0.3333 |              0.3846 |           0.6693 |
| bright_earth_science       | BGE_Small_Dense          |       0.4443 |       0.5952 |        0.6671 |             0.2845 |             0.3707 |              0.431  |           0.7357 |
| bright_earth_science       | SPLADE_v3_PISA           |       0.467  |       0.6838 |        0.7508 |             0.2328 |             0.4569 |              0.5172 |           0.8079 |
| trec_covid                 | BGE_Small_Dense          |       0.1191 |       0.3165 |        0.423  |             0      |             0      |              0      |           1      |
| trec_covid                 | SPLADE_v3_PISA           |       0.1223 |       0.31   |        0.411  |             0      |             0      |              0      |           0.9987 |
| bright_aops                | BGE_Small_Dense          |       0.204  |       0.3104 |        0.4062 |             0.036  |             0.0991 |              0.1712 |           0.4543 |
| bright_aops                | SPLADE_v3_PISA           |       0.2429 |       0.3895 |        0.4578 |             0      |             0.1171 |              0.2072 |           0.5117 |
| bright_theoremqa_questions | BGE_Small_Dense          |       0.2323 |       0.3401 |        0.4064 |             0.1649 |             0.2526 |              0.2938 |           0.4369 |
| bright_theoremqa_questions | SPLADE_v3_PISA           |       0.2153 |       0.3116 |        0.4133 |             0.1546 |             0.2268 |              0.3144 |           0.4408 |
| webis_touche2020           | BGE_Small_Dense          |       0.4385 |       0.6819 |        0.795  |             0.0408 |             0.102  |              0.1837 |           0.9621 |
| webis_touche2020           | SPLADE_v3_PISA           |       0.4897 |       0.7214 |        0.8228 |             0.0204 |             0.1224 |              0.2653 |           0.9658 |
| bright_leetcode            | BGE_Small_Dense          |       0.4229 |       0.5708 |        0.6714 |             0.2817 |             0.4507 |              0.5563 |           0.6976 |
| bright_leetcode            | SPLADE_v3_PISA           |       0.5171 |       0.7277 |        0.8207 |             0.3873 |             0.5915 |              0.7113 |           0.8426 |
| quora                      | BGE_Small_Dense          |       0.9949 |       0.9993 |        0.9997 |             0.9872 |             0.9973 |              0.9985 |           0.9998 |
| quora                      | SPLADE_v3_PISA           |       0.9813 |       0.9952 |        0.9971 |             0.9637 |             0.9878 |              0.9917 |           0.9978 |
| nq                         | BGE_Small_Dense          |       0.8699 |       0.9457 |        0.9631 |             0.8517 |             0.936  |              0.9563 |           0.9645 |
| nq                         | SPLADE_v3_PISA           |       0.9323 |       0.9745 |        0.982  |             0.92   |             0.9684 |              0.9771 |           0.983  |
| dbpedia_entity             | BGE_Small_Dense          |       0.4788 |       0.6516 |        0.7203 |             0.0875 |             0.175  |              0.215  |           0.8851 |
| dbpedia_entity             | SPLADE_v3_PISA           |       0.5568 |       0.7168 |        0.7771 |             0.13   |             0.2475 |              0.3075 |           0.906  |
| hotpotqa                   | BGE_Small_Dense          |       0.8208 |       0.8833 |        0.9084 |             0.6729 |             0.7814 |              0.8269 |           0.9269 |
| hotpotqa                   | SPLADE_v3_PISA           |       0.8052 |       0.8666 |        0.887  |             0.6273 |             0.7392 |              0.7789 |           0.9114 |
| climate_fever              | BGE_Small_Dense          |       0.5877 |       0.7261 |        0.7808 |             0.3049 |             0.4723 |              0.5596 |           0.8227 |
| climate_fever              | SPLADE_v3_PISA           |       0.4973 |       0.639  |        0.698  |             0.2182 |             0.3603 |              0.4332 |           0.7522 |
| fever                      | BGE_Small_Dense          |       0.9545 |       0.9699 |        0.9745 |             0.9244 |             0.9463 |              0.9533 |           0.9793 |
| fever                      | SPLADE_v3_PISA           |       0.9453 |       0.9667 |        0.9721 |             0.9118 |             0.9388 |              0.9469 |           0.978  |

### 3. Retrieval Efficiency & Resource Footprint

| dataset                    | pipeline                 |   retrieval_api_p50_ms |   retrieval_api_p99_ms |   batch_throughput_qps |   harness_per_query_ms |   index_disk_mb |   host_ram_peak_mb |   index_build_s |   cache_load_s |
|:---------------------------|:-------------------------|-----------------------:|-----------------------:|-----------------------:|-----------------------:|----------------:|-------------------:|----------------:|---------------:|
| scifact                    | BM25_Default             |                  21.88 |                  28.39 |                  44.1  |                  62.48 |           25.53 |             980.95 |            0    |           0.1  |
| scifact                    | BM25_RM3_Terrier_Default |                  30.27 |                  33.84 |                  39.47 |                  77.21 |           25.53 |             995.97 |            0    |           0.1  |
| scifact                    | BM25_Bo1_Terrier_Default |                  29.1  |                  33.29 |                  43.15 |                  72.75 |           25.53 |            1014.09 |            0    |           0.1  |
| scifact                    | DPH                      |                  21.79 |                  26.35 |                  51.21 |                  59.01 |           25.53 |            1028.41 |            0    |           0.1  |
| nfcorpus                   | BM25_Default             |                   6.51 |                  24.69 |                  90.15 |                  28.25 |           18.6  |             827.59 |            0    |           0.09 |
| nfcorpus                   | BM25_RM3_Terrier_Default |                  29.71 |                  34.28 |                  45.44 |                  67    |           18.6  |            1016.62 |            0    |           0.09 |
| nfcorpus                   | BM25_Bo1_Terrier_Default |                  28.79 |                  32.29 |                  52.13 |                  61.42 |           18.6  |            1054.41 |            0    |           0.09 |
| nfcorpus                   | DPH                      |                   6.55 |                  24.67 |                 126.73 |                  24.85 |           18.6  |            1030.44 |            0    |           0.09 |
| fiqa                       | BM25_Default             |                  24.15 |                  28.08 |                  41.2  |                  66.79 |           83.96 |            1008.04 |            0    |           0.1  |
| fiqa                       | BM25_RM3_Terrier_Default |                  33.69 |                  39.61 |                  35.83 |                  80.94 |           83.96 |            1041.79 |            0    |           0.1  |
| fiqa                       | BM25_Bo1_Terrier_Default |                  32.9  |                  38.19 |                  38.15 |                  78.29 |           83.96 |            1050.66 |            0    |           0.1  |
| fiqa                       | DPH                      |                  24.18 |                  28.7  |                  44.74 |                  64.74 |           83.96 |            1048.36 |            0    |           0.1  |
| arguana                    | BM25_Default             |                  27.67 |                  33.67 |                  37.19 |                  64.71 |           25.6  |            1375.22 |            0    |           0.01 |
| arguana                    | BM25_RM3_Terrier_Default |                  40.91 |                  57.33 |                  28.58 |                  83.32 |           25.6  |            1570.38 |            0    |           0.01 |
| arguana                    | BM25_Bo1_Terrier_Default |                  43.01 |                  63.85 |                  26.61 |                  87.88 |           25.6  |            1869.79 |            0    |           0.01 |
| arguana                    | DPH                      |                  28.06 |                  35.86 |                  37.74 |                  64.59 |           25.6  |            1861.12 |            0    |           0.01 |
| scidocs                    | BM25_Default             |                  23.02 |                  25.99 |                  43.78 |                  63.95 |           69.01 |            1847.34 |            0    |           0.03 |
| scidocs                    | BM25_RM3_Terrier_Default |                  32.11 |                  35.95 |                  38.82 |                  76.93 |           69.01 |            1862.29 |            0    |           0.03 |
| scidocs                    | BM25_Bo1_Terrier_Default |                  31.17 |                  35.58 |                  40.82 |                  74.79 |           69.01 |            1865.63 |            0    |           0.03 |
| scidocs                    | DPH                      |                  23.08 |                  26.64 |                  46.7  |                  62.61 |           69.01 |            1865.78 |            0    |           0.03 |
| quora                      | BM25_Default             |                  13.36 |                  17.32 |                  87.56 |                  29.06 |          212.62 |            1456.81 |            0    |           0.1  |
| quora                      | BM25_RM3_Terrier_Default |                  21.62 |                  28.63 |                  65.26 |                  34.74 |          212.62 |            1546.51 |            0    |           0.1  |
| quora                      | BM25_Bo1_Terrier_Default |                  22.15 |                  28.99 |                  64.12 |                  35.02 |          212.62 |            1561.54 |            0    |           0.1  |
| quora                      | DPH                      |                  14.18 |                  18.31 |                  82.02 |                  29.87 |          212.62 |            1531.55 |            0    |           0.1  |
| hotpotqa                   | BM25_Default             |                  47.62 |                 125.99 |                  17.24 |                  82.42 |         4178.75 |            1816.18 |            0    |           0.5  |
| hotpotqa                   | BM25_RM3_Terrier_Default |                 119.11 |                 272.77 |                   8.03 |                 160.03 |         4178.75 |            1759.12 |            0    |           0.5  |
| hotpotqa                   | BM25_Bo1_Terrier_Default |                 102.93 |                 289.37 |                   9.32 |                 140.55 |         4178.75 |            1758.39 |            0    |           0.5  |
| hotpotqa                   | DPH                      |                  51.8  |                 143.42 |                  17.66 |                  81.53 |         4178.75 |            1729.97 |            0    |           0.5  |
| trec_covid                 | BM25_Default             |                  20.48 |                  30.06 |                  39.6  |                  76.06 |          287    |            1726.51 |            0    |           0.14 |
| trec_covid                 | BM25_RM3_Terrier_Default |                  37.21 |                  55.41 |                  18.13 |                 132.76 |          287    |            1694.04 |            0    |           0.14 |
| trec_covid                 | BM25_Bo1_Terrier_Default |                  56    |                 104.92 |                  31.52 |                 135.77 |          287    |            2719.59 |            0    |           0.14 |
| trec_covid                 | DPH                      |                  21.94 |                  31.86 |                  50.09 |                  73.3  |          287    |            2728.74 |            0    |           0.14 |
| webis_touche2020           | BM25_Default             |                  16.97 |                  25    |                  39.17 |                  71.7  |          685.06 |            2087.85 |            0    |           0.47 |
| webis_touche2020           | BM25_RM3_Terrier_Default |                  43.59 |                  97.14 |                  15.98 |                 148.1  |          685.06 |            2917.49 |            0    |           0.47 |
| webis_touche2020           | BM25_Bo1_Terrier_Default |                  29.22 |                  47.81 |                  27.8  |                 103.09 |          685.06 |            3077.25 |            0    |           0.47 |
| webis_touche2020           | DPH                      |                  17.81 |                  27.21 |                  60.79 |                  63.53 |          685.06 |            3090.43 |            0    |           0.47 |
| dbpedia_entity             | BM25_Default             |                  41.38 |                 181.34 |                   8.8  |                 181.45 |         4041    |            3640.64 |            0    |           0.51 |
| dbpedia_entity             | BM25_RM3_Terrier_Default |                  69.76 |                 156.22 |                  11.11 |                 188.03 |         4041    |            3652.91 |            0    |           0.51 |
| dbpedia_entity             | BM25_Bo1_Terrier_Default |                  59.71 |                 139.92 |                  15.9  |                 150.72 |         4041    |            3656.23 |            0    |           0.51 |
| dbpedia_entity             | DPH                      |                  25.36 |                  71.24 |                  32.6  |                  78.97 |         4041    |            3641.86 |            0    |           0.51 |
| nq                         | BM25_Default             |                  27.5  |                  55.27 |                  28.71 |                  60.32 |         2304.87 |            3220.58 |            0    |           0.31 |
| nq                         | BM25_RM3_Terrier_Default |                  63.6  |                 122.47 |                  16.06 |                  99.45 |         2304.87 |            3260.18 |            0    |           0.31 |
| nq                         | BM25_Bo1_Terrier_Default |                  54.98 |                 114.1  |                  19.66 |                  85.18 |         2304.87 |            3268.09 |            0    |           0.31 |
| nq                         | DPH                      |                  29.44 |                  62.05 |                  34.27 |                  55.23 |         2304.87 |            3232.23 |            0    |           0.31 |
| climate_fever              | BM25_Default             |                  53.72 |                 163.63 |                  10.13 |                 156.66 |         5555.75 |            3238.53 |            0    |           0.66 |
| climate_fever              | BM25_RM3_Terrier_Default |                 121.6  |                 335.11 |                   7.37 |                 244.03 |         5555.75 |            3253.86 |            0    |           0.66 |
| climate_fever              | BM25_Bo1_Terrier_Default |                 106.19 |                 360.89 |                   8.59 |                 217.08 |         5555.75 |            3264.81 |            0    |           0.66 |
| climate_fever              | DPH                      |                  55.86 |                 179.75 |                  15.98 |                 122.49 |         5555.75 |            3245.43 |            0    |           0.66 |
| fever                      | BM25_Default             |                  36.64 |                 102.31 |                  19.92 |                  73.34 |         5555.71 |            3273.87 |            0    |           0.61 |
| fever                      | BM25_RM3_Terrier_Default |                 105.02 |                 236.52 |                   9.06 |                 145.09 |         5555.71 |            3812.55 |            0    |           0.61 |
| fever                      | BM25_Bo1_Terrier_Default |                  89.57 |                 229.52 |                  11.17 |                 121.43 |         5555.71 |            3817.24 |            0    |           0.61 |
| fever                      | DPH                      |                  37.68 |                 111.58 |                  23.45 |                  66.15 |         5555.71 |            3791.26 |            0    |           0.61 |
| bright_biology             | BM25_Default             |                  29.63 |                  38.51 |                  31.14 |                  85.38 |           75.34 |            3796.49 |            0    |           0.03 |
| bright_biology             | BM25_RM3_Terrier_Default |                  42.94 |                  58.16 |                  25.14 |                 113.2  |           75.34 |            3795.52 |            0    |           0.03 |
| bright_biology             | BM25_Bo1_Terrier_Default |                  41.41 |                  63.51 |                  27.38 |                 107.84 |           75.34 |            3793.62 |            0    |           0.03 |
| bright_biology             | DPH                      |                  27.85 |                  34.88 |                  39.61 |                  76.66 |           75.34 |            3775.67 |            0    |           0.03 |
| bright_earth_science       | BM25_Default             |                  29.74 |                  37.34 |                  25.83 |                  93.91 |          174.82 |            3783.02 |            0    |           0.05 |
| bright_earth_science       | BM25_RM3_Terrier_Default |                  43.49 |                  58.71 |                  22.45 |                 115.85 |          174.82 |            3783.83 |            0    |           0.05 |
| bright_earth_science       | BM25_Bo1_Terrier_Default |                  45.08 |                  64.25 |                  25.33 |                 114.67 |          174.82 |            3781.89 |            0    |           0.05 |
| bright_earth_science       | DPH                      |                  29.47 |                  36.88 |                  34.88 |                  82.89 |          174.82 |            3778.87 |            0    |           0.05 |
| bright_economics           | BM25_Default             |                  30.87 |                  36.59 |                  24.61 |                  97.54 |          168.75 |            3780.29 |            0    |           0.02 |
| bright_economics           | BM25_RM3_Terrier_Default |                  45.7  |                  61.94 |                  23.38 |                 118.7  |          168.75 |            3783.03 |            0    |           0.02 |
| bright_economics           | BM25_Bo1_Terrier_Default |                  46.01 |                  66.05 |                  23.23 |                 122.25 |          168.75 |            3787.92 |            0    |           0.02 |
| bright_economics           | DPH                      |                  32    |                  39.06 |                  33.86 |                  87.33 |          168.75 |            3776.89 |            0    |           0.02 |
| bright_psychology          | BM25_Default             |                  31.49 |                  37.79 |                  26.22 |                  94.63 |          191.01 |            3776.5  |            0    |           0.03 |
| bright_psychology          | BM25_RM3_Terrier_Default |                  42.75 |                  57.41 |                  22.22 |                 119.21 |          191.01 |            3782.21 |            0    |           0.03 |
| bright_psychology          | BM25_Bo1_Terrier_Default |                  47.08 |                  67.63 |                  23.66 |                 119.78 |          191.01 |            3784.02 |            0    |           0.03 |
| bright_psychology          | DPH                      |                  31.81 |                  38.63 |                  35.74 |                  84.53 |          191.01 |            3780.61 |            0    |           0.03 |
| bright_robotics            | BM25_Default             |                  31.2  |                  58.01 |                  24.2  |                 101.02 |           74.28 |            3819.26 |            0    |           0.02 |
| bright_robotics            | BM25_RM3_Terrier_Default |                  47.19 |                  91.02 |                  20.98 |                 130.34 |           74.28 |            3831    |            0    |           0.02 |
| bright_robotics            | BM25_Bo1_Terrier_Default |                  49.63 |                 110.88 |                  19.1  |                 139.69 |           74.28 |            4079.12 |            0    |           0.02 |
| bright_robotics            | DPH                      |                  31.62 |                  64.23 |                  28.73 |                  94.23 |           74.28 |            3785.48 |            0    |           0.02 |
| bright_stackoverflow       | BM25_Default             |                  40.4  |                  65.14 |                  19.36 |                 122.17 |          147.42 |            3800.96 |            0    |           0.07 |
| bright_stackoverflow       | BM25_RM3_Terrier_Default |                  64.44 |                 111.74 |                  14.89 |                 170.04 |          147.42 |            3807    |            0    |           0.07 |
| bright_stackoverflow       | BM25_Bo1_Terrier_Default |                  70.94 |                 152.85 |                  14.39 |                 181.29 |          147.42 |            3812.64 |            0    |           0.07 |
| bright_stackoverflow       | DPH                      |                  42.55 |                  73.29 |                  23.43 |                 114.42 |          147.42 |            3795.91 |            0    |           0.07 |
| bright_sustainable_living  | BM25_Default             |                  30.5  |                  40.78 |                  24.81 |                  96.47 |          193.46 |            4206.02 |            0    |           0.03 |
| bright_sustainable_living  | BM25_RM3_Terrier_Default |                  45.05 |                  66.76 |                  23    |                 118.69 |          193.46 |            4213.08 |            0    |           0.03 |
| bright_sustainable_living  | BM25_Bo1_Terrier_Default |                  45.33 |                  74.79 |                  23.88 |                 119.43 |          193.46 |            4219.62 |            0    |           0.03 |
| bright_sustainable_living  | DPH                      |                  29.29 |                  40.22 |                  36    |                  83.68 |          193.46 |            4207.52 |            0    |           0.03 |
| bright_leetcode            | BM25_Default             |                 123.51 |                 162.22 |                   7.37 |                 302.12 |          577.59 |            4232.58 |            0    |           0.24 |
| bright_leetcode            | BM25_RM3_Terrier_Default |                 228.86 |                 300.18 |                   4.22 |                 534.35 |          577.59 |            4572.3  |            0    |           0.24 |
| bright_leetcode            | BM25_Bo1_Terrier_Default |                 247.45 |                 364.22 |                   3.93 |                 575.46 |          577.59 |            4806.78 |            0    |           0.24 |
| bright_leetcode            | DPH                      |                 129.03 |                 172.87 |                   7.56 |                 307.47 |          577.59 |            4936.44 |            0    |           0.24 |
| bright_pony                | BM25_Default             |                  28.71 |                  30.63 |                  40.21 |                  76.82 |            8.32 |            4910.28 |            0    |           0.01 |
| bright_pony                | BM25_RM3_Terrier_Default |                  38.29 |                  43.29 |                  32.66 |                  96.93 |            8.32 |            4902.43 |            0    |           0.01 |
| bright_pony                | BM25_Bo1_Terrier_Default |                  39.06 |                  44.23 |                  30.3  |                  99.44 |            8.32 |            4894.14 |            0    |           0.01 |
| bright_pony                | DPH                      |                  26.69 |                  30.33 |                  38.71 |                  75.51 |            8.32 |            4900.38 |            0    |           0.01 |
| bright_aops                | BM25_Default             |                 314.96 |                 362.47 |                   2.95 |                 760.21 |          165.73 |             791.92 |            0    |           0.25 |
| bright_aops                | BM25_RM3_Terrier_Default |                 619.94 |                 705.25 |                   1.47 |                1491.86 |          165.73 |             885.42 |            0    |           0.25 |
| bright_aops                | BM25_Bo1_Terrier_Default |                 616.54 |                 723.74 |                   1.46 |                1495.44 |          165.73 |             856.23 |            0    |           0.25 |
| bright_aops                | DPH                      |                 313.51 |                 351.24 |                   3.05 |                 746.57 |          165.73 |             853.61 |            0    |           0.25 |
| bright_theoremqa_questions | BM25_Default             |                  51.98 |                  79.15 |                  17.31 |                 135.5  |          165.73 |             692.48 |            0    |           0.36 |
| bright_theoremqa_questions | BM25_RM3_Terrier_Default |                  80.34 |                 135.39 |                  12.11 |                 195.9  |          165.73 |             825.06 |            0    |           0.36 |
| bright_theoremqa_questions | BM25_Bo1_Terrier_Default |                  87.82 |                 151.98 |                  11.56 |                 208.12 |          165.73 |             907.87 |            0    |           0.36 |
| bright_theoremqa_questions | DPH                      |                  53.78 |                  84.06 |                  18.91 |                 133.06 |          165.73 |             836.34 |            0    |           0.36 |
| bright_theoremqa_theorems  | BM25_Default             |                  13.45 |                  16.74 |                  55.82 |                  53.01 |           23.79 |             463.84 |            0    |           0.11 |
| bright_theoremqa_theorems  | BM25_RM3_Terrier_Default |                  22.32 |                  29.03 |                  54.29 |                  65.96 |           23.79 |             511.77 |            0    |           0.11 |
| bright_theoremqa_theorems  | BM25_Bo1_Terrier_Default |                  24.14 |                  32.08 |                  52.96 |                  70.16 |           23.79 |             572.71 |            0    |           0.11 |
| bright_theoremqa_theorems  | DPH                      |                  13.84 |                  16.32 |                  76.17 |                  48.51 |           23.79 |             534.13 |            0    |           0.11 |
| scifact                    | DPH_Bo1_Terrier_Default  |                  27.08 |                  30.56 |                  37.94 |                  72.34 |           25.53 |             854.44 |            0    |           0.11 |
| scifact                    | DPH_RM3_Terrier_Default  |                  27.26 |                  30.94 |                  43.92 |                  69.22 |           25.53 |             890.55 |            0    |           0.11 |
| nfcorpus                   | DPH_Bo1_Terrier_Default  |                  26.09 |                  29.72 |                  46.59 |                  60.07 |           18.6  |             870.4  |            0    |           0.09 |
| nfcorpus                   | DPH_RM3_Terrier_Default  |                  26.81 |                  30.26 |                  52.46 |                  59.34 |           18.6  |             906.97 |            0    |           0.09 |
| fiqa                       | DPH_Bo1_Terrier_Default  |                  30.6  |                  36.24 |                  36.85 |                  75.26 |           83.96 |             885.19 |            0    |           0.14 |
| fiqa                       | DPH_RM3_Terrier_Default  |                  31.53 |                  36.33 |                  38.98 |                  74.56 |           83.96 |             916.26 |            0    |           0.14 |
| arguana                    | DPH_Bo1_Terrier_Default  |                  39.88 |                  56.6  |                  27.96 |                  82.44 |           25.6  |            1556.06 |            0    |           0.1  |
| arguana                    | DPH_RM3_Terrier_Default  |                  38.27 |                  54.71 |                  30.59 |                  77.45 |           25.6  |            1567.8  |            0    |           0.1  |
| scidocs                    | DPH_Bo1_Terrier_Default  |                  28.94 |                  32.83 |                  39.47 |                  71.68 |           69.01 |             995.02 |            0    |           0.13 |
| scidocs                    | DPH_RM3_Terrier_Default  |                  29.82 |                  33.96 |                  41.85 |                  71.39 |           69.01 |            1019.46 |            0    |           0.13 |
| quora                      | DPH_Bo1_Terrier_Default  |                  21.81 |                  27.45 |                  63.33 |                  34.01 |          212.62 |             816.73 |            0    |           0.24 |
| quora                      | DPH_RM3_Terrier_Default  |                  21.01 |                  28.08 |                  65.64 |                  33.38 |          212.62 |             832.85 |            0    |           0.24 |
| hotpotqa                   | DPH_Bo1_Terrier_Default  |                  94.83 |                 279.04 |                   9.32 |                 137.74 |         4178.75 |             704.91 |            0    |           0.83 |
| hotpotqa                   | DPH_RM3_Terrier_Default  |                 109.75 |                 255.67 |                   8.77 |                 146.24 |         4178.75 |             763.31 |            0    |           0.83 |
| trec_covid                 | DPH_Bo1_Terrier_Default  |                  33.93 |                  54.99 |                  18.04 |                 131.25 |          287    |             775.27 |            0    |           0.33 |
| trec_covid                 | DPH_RM3_Terrier_Default  |                  35.21 |                  52.43 |                  24.59 |                 114.11 |          287    |             767.01 |            0    |           0.33 |
| webis_touche2020           | DPH_Bo1_Terrier_Default  |                  47.19 |                  63.17 |                  16.42 |                 144.22 |          685.06 |            1624.97 |            0    |           0.87 |
| webis_touche2020           | DPH_RM3_Terrier_Default  |                  31.5  |                  45.51 |                  19.81 |                 120.57 |          685.06 |            1632.48 |            0    |           0.87 |
| dbpedia_entity             | DPH_Bo1_Terrier_Default  |                  54.37 |                 131.29 |                   6.83 |                 226.09 |         4041    |             539.79 |            0    |           0.64 |
| dbpedia_entity             | DPH_RM3_Terrier_Default  |                  67    |                 139.92 |                  13.26 |                 166.41 |         4041    |             558.28 |            0    |           0.64 |
| nq                         | DPH_Bo1_Terrier_Default  |                  52.27 |                 108.33 |                  17    |                  91.17 |         2304.87 |             583.18 |            0    |           0.44 |
| nq                         | DPH_RM3_Terrier_Default  |                  59.45 |                 114.96 |                  17.57 |                  91.19 |         2304.87 |             604.21 |            0    |           0.44 |
| climate_fever              | DPH_Bo1_Terrier_Default  |                 102.92 |                 338.4  |                   6.93 |                 241.26 |         5555.75 |             609.81 |            0    |           0.63 |
| climate_fever              | DPH_RM3_Terrier_Default  |                 117.97 |                 330.56 |                   7.65 |                 234.66 |         5555.75 |             607.8  |            0    |           0.63 |
| fever                      | DPH_Bo1_Terrier_Default  |                  84.57 |                 210.5  |                  10.51 |                 125.18 |         5555.71 |             682.18 |            0    |           0.6  |
| fever                      | DPH_RM3_Terrier_Default  |                 102.01 |                 229.75 |                   9.6  |                 136.93 |         5555.71 |             687.39 |            0    |           0.6  |
| bright_biology             | DPH_Bo1_Terrier_Default  |                  43.25 |                  61.06 |                  19.15 |                 126.48 |           75.34 |             609.71 |            0    |           0.14 |
| bright_biology             | DPH_RM3_Terrier_Default  |                  42.36 |                  56.5  |                  26.35 |                 110.16 |           75.34 |             599.08 |            0    |           0.14 |
| bright_earth_science       | DPH_Bo1_Terrier_Default  |                  47.6  |                  76.26 |                  15.71 |                 143.6  |          174.82 |             666.21 |            0    |           0.17 |
| bright_earth_science       | DPH_RM3_Terrier_Default  |                  49.63 |                  67.96 |                  20.11 |                 130.49 |          174.82 |             666.84 |            0    |           0.17 |
| bright_economics           | DPH_Bo1_Terrier_Default  |                  47.71 |                  67.49 |                  15.6  |                 146.43 |          168.75 |             691.4  |            0    |           0.13 |
| bright_economics           | DPH_RM3_Terrier_Default  |                  46.83 |                  61.54 |                  21.45 |                 126.05 |          168.75 |             635.58 |            0    |           0.13 |
| bright_psychology          | DPH_Bo1_Terrier_Default  |                  47.8  |                  70.52 |                  15.38 |                 146.34 |          191.01 |             675.97 |            0    |           0.13 |
| bright_psychology          | DPH_RM3_Terrier_Default  |                  46.42 |                  60.69 |                  20.46 |                 126.97 |          191.01 |             638.3  |            0    |           0.13 |
| bright_robotics            | DPH_Bo1_Terrier_Default  |                  51.86 |                 128.02 |                  13.56 |                 165.62 |           74.28 |             929.49 |            0    |           0.13 |
| bright_robotics            | DPH_RM3_Terrier_Default  |                  49.2  |                  95.89 |                  19.59 |                 136.38 |           74.28 |             837.91 |            0    |           0.13 |
| bright_stackoverflow       | DPH_Bo1_Terrier_Default  |                  71.22 |                 145.6  |                  10.63 |                 210.09 |          147.42 |             853.93 |            0    |           0.21 |
| bright_stackoverflow       | DPH_RM3_Terrier_Default  |                  65.66 |                 118.63 |                  14.89 |                 171.69 |          147.42 |             764.38 |            0    |           0.21 |
| bright_sustainable_living  | DPH_Bo1_Terrier_Default  |                  47.1  |                  76.36 |                  15.86 |                 143.42 |          193.46 |             718.62 |            0    |           0.14 |
| bright_sustainable_living  | DPH_RM3_Terrier_Default  |                  45.6  |                  69.61 |                  21.59 |                 123.68 |          193.46 |             637.46 |            0    |           0.14 |
| bright_leetcode            | DPH_Bo1_Terrier_Default  |                 232.5  |                 320.99 |                   3.79 |                 567.26 |          577.59 |            1404.47 |            0    |           0.49 |
| bright_leetcode            | DPH_RM3_Terrier_Default  |                 209.67 |                 275.23 |                   4.57 |                 495.22 |          577.59 |            1351.1  |            0    |           0.49 |
| bright_pony                | DPH_Bo1_Terrier_Default  |                  40.22 |                  45    |                  25.87 |                 106.78 |            8.32 |             618.04 |            0    |           0.09 |
| bright_pony                | DPH_RM3_Terrier_Default  |                  40.01 |                  44.35 |                  29.93 |                 100.71 |            8.32 |             611.69 |            0    |           0.09 |
| bright_aops                | DPH_Bo1_Terrier_Default  |                 130.32 |                 185.34 |                   6.95 |                 333.31 |          165.73 |             748.02 |            0    |           0.27 |
| bright_aops                | DPH_RM3_Terrier_Default  |                 124.99 |                 176.55 |                   7.75 |                 313.46 |          165.73 |             783.14 |            0    |           0.27 |
| bright_theoremqa_questions | DPH_Bo1_Terrier_Default  |                 127.78 |                 199.73 |                   6.86 |                 318.21 |          165.73 |             900.4  |            0    |           0.27 |
| bright_theoremqa_questions | DPH_RM3_Terrier_Default  |                 122.49 |                 182.64 |                   7.88 |                 289.4  |          165.73 |             898.48 |            0    |           0.27 |
| bright_theoremqa_theorems  | DPH_Bo1_Terrier_Default  |                  24.75 |                  31.43 |                  37    |                  78.77 |           23.79 |             547.36 |            0    |           0.11 |
| bright_theoremqa_theorems  | DPH_RM3_Terrier_Default  |                  23.63 |                  28.69 |                  54.47 |                  67.7  |           23.79 |             532.99 |            0    |           0.11 |
| scifact                    | BGE_Small_Dense          |                  10.87 |                  28.06 |                 756.58 |                  32.54 |            7.61 |            1886.56 |            0    |           0    |
| scifact                    | SPLADE_v3_PISA           |                   7.88 |                  23.41 |                 148.51 |                  33.88 |           12.28 |            1940.82 |           27.86 |           0    |
| nfcorpus                   | BGE_Small_Dense          |                  10.18 |                  18.59 |                 655.99 |                  29.97 |            5.36 |            1871.26 |            0    |           0    |
| nfcorpus                   | SPLADE_v3_PISA           |                   5.9  |                   7.66 |                 367.92 |                  25.87 |            8.31 |            1904.19 |           19.46 |           0    |
| arguana                    | BGE_Small_Dense          |                  12.61 |                  29.75 |                 215.61 |                  32.17 |           13.1  |            1874.05 |            0    |           0    |
| arguana                    | SPLADE_v3_PISA           |                  61.17 |                 111.7  |                  15.74 |                 126.02 |           22.63 |            1932.32 |           46.94 |           0    |
| bright_pony                | BGE_Small_Dense          |                  34.16 |                  56.92 |                  62.12 |                  67.93 |           12.04 |            1698.52 |            0    |           0    |
| bright_pony                | SPLADE_v3_PISA           |                  50.64 |                  70.76 |                  20.79 |                 112.22 |           19.18 |            1746.07 |           40.99 |           0    |
| bright_theoremqa_theorems  | BGE_Small_Dense          |                  13.59 |                  29.3  |                 401.36 |                  36.96 |           34.92 |            1944.52 |            0    |           0    |
| bright_theoremqa_theorems  | SPLADE_v3_PISA           |                  56.03 |                 111.43 |                  18.47 |                 131.87 |           38.43 |            1970.2  |          126.24 |           0    |
| scidocs                    | BGE_Small_Dense          |                  11.26 |                  20.94 |                 543.89 |                  19.93 |           38.3  |            2299.61 |            0    |           0    |
| scidocs                    | SPLADE_v3_PISA           |                   9.83 |                  24.17 |                 138.18 |                  24.93 |           59.32 |            2217.69 |          138.94 |           0    |
| bright_economics           | BGE_Small_Dense          |                  35.67 |                  54.52 |                  50.69 |                  73.7  |           79.55 |            2163.62 |            0    |           0    |
| bright_economics           | SPLADE_v3_PISA           |                  70.85 |                 137.62 |                  12.37 |                 176.28 |           70.12 |            2200.27 |          257.6  |           0    |
| bright_psychology          | BGE_Small_Dense          |                  35.8  |                  59.08 |                  49.66 |                  74.92 |           81.33 |            2313.82 |            0    |           0    |
| bright_psychology          | SPLADE_v3_PISA           |                  87.61 |                 200.7  |                  10.4  |                 212.81 |           80.09 |            2349.22 |          273.41 |           0    |
| bright_biology             | BGE_Small_Dense          |                  35.07 |                  58.41 |                  49.8  |                  73.55 |           88.45 |            2247.08 |            0    |           0    |
| bright_biology             | SPLADE_v3_PISA           |                  62.19 |                 154.41 |                  14.4  |                 156.15 |           93.61 |            2285.11 |          193.86 |           0    |
| fiqa                       | BGE_Small_Dense          |                  13.46 |                  32.22 |                 373.07 |                  21.85 |           85    |            2539.7  |            0    |           0    |
| fiqa                       | SPLADE_v3_PISA           |                  15.22 |                  30.54 |                  75.51 |                  32.23 |          120.29 |            2581.5  |          308.41 |           0    |
| bright_sustainable_living  | BGE_Small_Dense          |                  39.71 |                  82.93 |                  43.18 |                  81.02 |           93.18 |            2288.95 |            0    |           0    |
| bright_sustainable_living  | SPLADE_v3_PISA           |                  74.21 |                 168.86 |                  12.58 |                 172.81 |           83.53 |            2325.79 |          305.64 |           0    |
| bright_robotics            | BGE_Small_Dense          |                  36.27 |                  68.67 |                  43.83 |                  80.5  |           93.3  |            2228.12 |            0    |           0    |
| bright_robotics            | SPLADE_v3_PISA           |                 122.4  |                 195.82 |                   8.02 |                 272.94 |           79.35 |            2242.69 |          310.51 |           0    |
| bright_stackoverflow       | BGE_Small_Dense          |                  41.04 |                  75.54 |                  45.18 |                  78.96 |          168.2  |            3132.89 |            0    |           0    |
| bright_stackoverflow       | SPLADE_v3_PISA           |                 241.53 |                 466.09 |                   4.03 |                 492.06 |          198.86 |            3209.98 |          574.81 |           0    |
| bright_earth_science       | BGE_Small_Dense          |                  40.95 |                  83.4  |                  51.14 |                  76.02 |          190.81 |            3520.64 |            0    |           0    |
| bright_earth_science       | SPLADE_v3_PISA           |                 123.18 |                 287.97 |                   7.58 |                 272.37 |          285.26 |            3623.71 |          516.76 |           0    |
| trec_covid                 | BGE_Small_Dense          |                  23.74 |                  50.14 |                 158.66 |                  54.21 |          253.93 |            3801.94 |            0    |           0    |
| trec_covid                 | SPLADE_v3_PISA           |                  35.92 |                  80.17 |                  29.42 |                 103.42 |          317.54 |            3930.72 |          931.79 |           0    |
| bright_aops                | BGE_Small_Dense          |                  98.01 |                 172.7  |                  19.24 |                 166.48 |          288.4  |            3709.67 |            0    |           0    |
| bright_aops                | SPLADE_v3_PISA           |                 499.92 |                1021.37 |                   2.04 |                 988.33 |          315.16 |            3814.25 |         1004.35 |           0    |
| bright_theoremqa_questions | BGE_Small_Dense          |                  98.34 |                 156.91 |                  19.11 |                 125.42 |          288.4  |            3747.14 |            0    |           0    |
| bright_theoremqa_questions | SPLADE_v3_PISA           |                 363.06 |                 703.42 |                   2.6  |                 614.47 |          315.16 |            3866.75 |         1006.88 |           0    |
| webis_touche2020           | BGE_Small_Dense          |                  38.19 |                 255.68 |                  77.94 |                  93.59 |          578.06 |            4758.19 |            0    |           0    |
| webis_touche2020           | SPLADE_v3_PISA           |                  25.14 |                  58.53 |                  37.04 |                 104.52 |          761.59 |            5066.36 |         2096.62 |           0    |
| bright_leetcode            | BGE_Small_Dense          |                 106.89 |                 205.05 |                  18.88 |                 154.05 |          620.58 |            4636.66 |            0    |           0    |
| bright_leetcode            | SPLADE_v3_PISA           |                1508.96 |                2816.7  |                   0.63 |                2774.2  |          897.57 |            4936.65 |         2295.82 |           0    |
| quora                      | BGE_Small_Dense          |                  46.75 |                 137.31 |                  85.87 |                  28.82 |          771.41 |            2852.21 |            0    |           0    |
| quora                      | SPLADE_v3_PISA           |                  24.91 |                  64.95 |                  41.18 |                  41.17 |          124.83 |            2927.67 |          831.16 |           0    |
| nq                         | BGE_Small_Dense          |                 193.74 |                 248.9  |                  20.26 |                  71.03 |         3927.93 |            7565.98 |            0    |           0    |
| nq                         | SPLADE_v3_PISA           |                  84.64 |                 288.87 |                  10.33 |                 118.3  |         4464.61 |            9214.15 |        13692.9  |           0    |
| dbpedia_entity             | BGE_Small_Dense          |                 350.13 |                 479.21 |                   5.88 |                 280.71 |         7623.98 |            9364.37 |            0    |           0    |
| dbpedia_entity             | SPLADE_v3_PISA           |                 114.4  |                 446.38 |                   7.66 |                 228.65 |         5552.93 |            5342.04 |        14912.8  |           0    |
| hotpotqa                   | BGE_Small_Dense          |                 329.06 |                 514.83 |                  11.21 |                 110.87 |         7736.52 |           10176.8  |         7395.08 |           0    |
| hotpotqa                   | SPLADE_v3_PISA           |                 254.26 |                3721.51 |                   1.92 |                 550.16 |         5838.86 |            5839.92 |        21057.6  |           0    |
| climate_fever              | BGE_Small_Dense          |                 413.3  |                 684.32 |                   8.43 |                 165.34 |         9029.22 |           10902.6  |            0    |           0    |
| climate_fever              | SPLADE_v3_PISA           |                 755.86 |                5311.81 |                   0.92 |                1205.12 |         7101.39 |            5558.98 |        30443.5  |           0    |
| fever                      | BGE_Small_Dense          |                 478.55 |                 595.46 |                   9.06 |                 135.61 |         9029.18 |           11377.3  |        22266.3  |           0    |
| fever                      | SPLADE_v3_PISA           |                 168.5  |                 933.98 |                   4.1  |                 269.73 |         7101.33 |            5866.41 |        31763.2  |           0    |
