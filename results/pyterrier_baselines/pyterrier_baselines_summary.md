# PyTerrier Baseline Evaluation Summary

Generated on 2026-09-08 04:23:05

### Cross-Dataset Retrieval Performance (nDCG@10, MRR@10, Recall@10)

| dataset                    | pipeline                  |    ndcg_10 |     mrr_10 |   recall_10 |   strict_10 |       p_10 |   avg_latency_ms |
|:---------------------------|:--------------------------|-----------:|-----------:|------------:|------------:|-----------:|-----------------:|
| quora                      | BM25_Default              | 0.767574   | 0.758396   |  0.868396   |   0.9109    | 0.11703    |         27.8796  |
| quora                      | BM25_Analyzed             | 0.784682   | 0.778935   |  0.882278   |   0.9258    | 0.11855    |         35.2521  |
| quora                      | BM25_RM3_Terrier_Default  | 0.750776   | 0.737256   |  0.848913   |   0.8877    | 0.11508    |         43.7349  |
| quora                      | BM25_RM3_Unified_Default  | 0.577418   | 0.54892    |  0.731878   |   0.7831    | 0.09798    |         43.5081  |
| quora                      | BM25_RM3_Unified_Analyzed | 0.780327   | 0.774585   |  0.876455   |   0.9202    | 0.11789    |         66.0049  |
| bright_robotics            | BM25_Default              | 0.0996494  | 0.124419   |  0.142162   |   0.287129  | 0.0425743  |         42.059   |
| bright_robotics            | BM25_Analyzed             | 0.0541163  | 0.0714364  |  0.0814143  |   0.178218  | 0.0207921  |         40.9189  |
| bright_robotics            | BM25_RM3_Terrier_Default  | 0.0866382  | 0.108113   |  0.113265   |   0.207921  | 0.029703   |         59.2245  |
| bright_robotics            | BM25_RM3_Unified_Default  | 0.072327   | 0.0917217  |  0.109361   |   0.207921  | 0.0287129  |         63.4657  |
| bright_robotics            | BM25_RM3_Unified_Analyzed | 0.0296464  | 0.0435369  |  0.0417344  |   0.0990099 | 0.0128713  |         65.3464  |
| bright_pony                | BM25_Default              | 0.0251635  | 0.0635629  |  0.0150751  |   0.214286  | 0.0258929  |         28.8571  |
| bright_pony                | BM25_Analyzed             | 0.0306407  | 0.0727891  |  0.0167613  |   0.232143  | 0.0303571  |         30.3051  |
| bright_pony                | BM25_RM3_Terrier_Default  | 0.0192996  | 0.0419891  |  0.0114387  |   0.133929  | 0.0205357  |         49.7191  |
| bright_pony                | BM25_RM3_Unified_Default  | 0.00570081 | 0.0181548  |  0.00324802 |   0.0357143 | 0.00446429 |         48.7183  |
| bright_pony                | BM25_RM3_Unified_Analyzed | 0.0196585  | 0.0465632  |  0.011597   |   0.178571  | 0.0214286  |         49.6571  |
| trec_covid                 | BM25_Default              | 0.607264   | 0.8725     |  0.0170608  |   1         | 0.664      |         39.3151  |
| trec_covid                 | BM25_Analyzed             | 0.517197   | 0.827333   |  0.0139576  |   0.98      | 0.568      |         34.162   |
| trec_covid                 | BM25_RM3_Terrier_Default  | 0.62549    | 0.846667   |  0.0179088  |   0.96      | 0.706      |         60.1994  |
| trec_covid                 | BM25_RM3_Unified_Default  | 0.566767   | 0.767333   |  0.0160167  |   0.96      | 0.638      |         81.5045  |
| trec_covid                 | BM25_RM3_Unified_Analyzed | 0.581975   | 0.806857   |  0.0155818  |   0.98      | 0.648      |         88.7996  |
| nfcorpus                   | BM25_Default              | 0.329505   | 0.532077   |  0.153175   |   0.69969   | 0.239319   |          8.73861 |
| nfcorpus                   | BM25_Analyzed             | 0.314299   | 0.525683   |  0.141281   |   0.684211  | 0.227245   |          8.00837 |
| nfcorpus                   | BM25_RM3_Terrier_Default  | 0.351735   | 0.53586    |  0.174338   |   0.705882  | 0.259133   |         27.3221  |
| nfcorpus                   | BM25_RM3_Unified_Default  | 0.335892   | 0.530952   |  0.164497   |   0.696594  | 0.244892   |         42.3933  |
| nfcorpus                   | BM25_RM3_Unified_Analyzed | 0.317277   | 0.516901   |  0.151935   |   0.687307  | 0.229102   |         46.2224  |
| hotpotqa                   | BM25_Default              | 0.585781   | 0.750526   |  0.61607    |   0.886023  | 0.123214   |         75.2093  |
| hotpotqa                   | BM25_Analyzed             | 0.57926    | 0.744672   |  0.60817    |   0.878731  | 0.121634   |         88.4148  |
| hotpotqa                   | BM25_RM3_Terrier_Default  | 0.528567   | 0.716492   |  0.535449   |   0.83052   | 0.10709    |        154.881   |
| hotpotqa                   | BM25_RM3_Unified_Default  | 0.450053   | 0.585792   |  0.493113   |   0.766374  | 0.0986226  |        252.468   |
| hotpotqa                   | BM25_RM3_Unified_Analyzed | 0.537778   | 0.697148   |  0.568062   |   0.844024  | 0.113612   |        364.788   |
| fever                      | BM25_Default              | 0.507675   | 0.468202   |  0.679733   |   0.715572  | 0.0743024  |         67.0311  |
| fever                      | BM25_Analyzed             | 0.504253   | 0.465865   |  0.670465   |   0.705821  | 0.0731023  |         71.5601  |
| fever                      | BM25_RM3_Terrier_Default  | 0.474683   | 0.433122   |  0.651263   |   0.687969  | 0.0712571  |        161.734   |
| fever                      | BM25_RM3_Unified_Default  | 0.485215   | 0.444097   |  0.661804   |   0.69892   | 0.0722022  |        338.476   |
| fever                      | BM25_RM3_Unified_Analyzed | 0.53537    | 0.501241   |  0.693748   |   0.731173  | 0.0757576  |        472.096   |
| fiqa                       | BM25_Default              | 0.252634   | 0.3104     |  0.309708   |   0.490741  | 0.0703704  |         22.2547  |
| fiqa                       | BM25_Analyzed             | 0.247319   | 0.301615   |  0.311739   |   0.487654  | 0.0692901  |         23.4118  |
| fiqa                       | BM25_RM3_Terrier_Default  | 0.243154   | 0.292421   |  0.298587   |   0.462963  | 0.0665123  |         48.14    |
| fiqa                       | BM25_RM3_Unified_Default  | 0.199058   | 0.240656   |  0.264231   |   0.432099  | 0.0569444  |         62.4652  |
| fiqa                       | BM25_RM3_Unified_Analyzed | 0.231013   | 0.279394   |  0.296236   |   0.462963  | 0.0646605  |         72.2041  |
| scidocs                    | BM25_Default              | 0.158157   | 0.276956   |  0.164733   |   0.498     | 0.0813     |         22.3987  |
| scidocs                    | BM25_Analyzed             | 0.155768   | 0.275418   |  0.1609     |   0.49      | 0.0793     |         23.1816  |
| scidocs                    | BM25_RM3_Terrier_Default  | 0.157234   | 0.269939   |  0.16265    |   0.463     | 0.0801     |         51.0286  |
| scidocs                    | BM25_RM3_Unified_Default  | 0.149847   | 0.263308   |  0.157433   |   0.479     | 0.0778     |         66.5427  |
| scidocs                    | BM25_RM3_Unified_Analyzed | 0.151554   | 0.267833   |  0.15685    |   0.478     | 0.0773     |         94.6526  |
| webis_touche2020           | BM25_Default              | 0.293797   | 0.571259   |  0.18247    |   0.918367  | 0.277551   |         35.2435  |
| webis_touche2020           | BM25_Analyzed             | 0.357319   | 0.687302   |  0.211721   |   0.938776  | 0.320408   |         36.2602  |
| webis_touche2020           | BM25_RM3_Terrier_Default  | 0.335769   | 0.596599   |  0.20977    |   0.918367  | 0.310204   |         57.8088  |
| webis_touche2020           | BM25_RM3_Unified_Default  | 0.356838   | 0.684127   |  0.209722   |   0.959184  | 0.322449   |        107.769   |
| webis_touche2020           | BM25_RM3_Unified_Analyzed | 0.371316   | 0.6905     |  0.214383   |   0.959184  | 0.32449    |        141.25    |
| bright_economics           | BM25_Default              | 0.11774    | 0.135464   |  0.133012   |   0.262136  | 0.0563107  |         38.7731  |
| bright_economics           | BM25_Analyzed             | 0.110416   | 0.141501   |  0.108879   |   0.23301   | 0.0533981  |         36.9787  |
| bright_economics           | BM25_RM3_Terrier_Default  | 0.0870891  | 0.0999191  |  0.0805873  |   0.15534   | 0.0427184  |         56.5728  |
| bright_economics           | BM25_RM3_Unified_Default  | 0.099691   | 0.117352   |  0.112555   |   0.194175  | 0.0533981  |         65.6389  |
| bright_economics           | BM25_RM3_Unified_Analyzed | 0.105322   | 0.114667   |  0.10931    |   0.203883  | 0.0601942  |         68.2221  |
| bright_biology             | BM25_Default              | 0.0912447  | 0.129361   |  0.122691   |   0.271845  | 0.0378641  |         32.8408  |
| bright_biology             | BM25_Analyzed             | 0.0726457  | 0.10888    |  0.097187   |   0.213592  | 0.031068   |         35.568   |
| bright_biology             | BM25_RM3_Terrier_Default  | 0.0844188  | 0.120334   |  0.102564   |   0.23301   | 0.0368932  |         50.6803  |
| bright_biology             | BM25_RM3_Unified_Default  | 0.0630634  | 0.0951187  |  0.0831093  |   0.194175  | 0.0242718  |         56.648   |
| bright_biology             | BM25_RM3_Unified_Analyzed | 0.0633004  | 0.0912891  |  0.0815534  |   0.165049  | 0.023301   |         60.1078  |
| climate_fever              | BM25_Default              | 0.138601   | 0.188684   |  0.176004   |   0.368078  | 0.0441694  |        112.024   |
| climate_fever              | BM25_Analyzed             | 0.130432   | 0.180094   |  0.163279   |   0.347231  | 0.0409772  |        140.77    |
| climate_fever              | BM25_RM3_Terrier_Default  | 0.140767   | 0.184806   |  0.184267   |   0.380456  | 0.0467752  |        204.053   |
| climate_fever              | BM25_RM3_Unified_Default  | 0.123539   | 0.164915   |  0.159837   |   0.327687  | 0.0401303  |        360.406   |
| climate_fever              | BM25_RM3_Unified_Analyzed | 0.124781   | 0.169687   |  0.160836   |   0.341368  | 0.0405863  |        600.785   |
| scifact                    | BM25_Default              | 0.683904   | 0.644094   |  0.826278   |   0.843333  | 0.0916667  |         20.7182  |
| scifact                    | BM25_Analyzed             | 0.667464   | 0.629094   |  0.8        |   0.813333  | 0.0873333  |         19.0976  |
| scifact                    | BM25_RM3_Terrier_Default  | 0.673053   | 0.633279   |  0.805722   |   0.816667  | 0.0896667  |         43.2143  |
| scifact                    | BM25_RM3_Unified_Default  | 0.648071   | 0.609452   |  0.785111   |   0.796667  | 0.0866667  |         57.681   |
| scifact                    | BM25_RM3_Unified_Analyzed | 0.650232   | 0.611795   |  0.786667   |   0.803333  | 0.0856667  |         62.0055  |
| dbpedia_entity             | BM25_Default              | 0.287926   | 0.577668   |  0.194869   |   0.8       | 0.2735     |        127.652   |
| dbpedia_entity             | BM25_Analyzed             | 0.280269   | 0.564389   |  0.191446   |   0.7875    | 0.26475    |        113.036   |
| dbpedia_entity             | BM25_RM3_Terrier_Default  | 0.28079    | 0.56526    |  0.192291   |   0.77      | 0.27025    |        115.072   |
| dbpedia_entity             | BM25_RM3_Unified_Default  | 0.272737   | 0.541273   |  0.188484   |   0.7775    | 0.26575    |        209.105   |
| dbpedia_entity             | BM25_RM3_Unified_Analyzed | 0.268057   | 0.527309   |  0.183621   |   0.7675    | 0.25725    |        306.505   |
| bright_aops                | BM25_Default              | 0.0603602  | 0.124496   |  0.0589876  |   0.207207  | 0.0306306  |         48.0062  |
| bright_aops                | BM25_Analyzed             | 0.0513383  | 0.114489   |  0.049335   |   0.198198  | 0.0261261  |         43.042   |
| bright_aops                | BM25_RM3_Terrier_Default  | 0.0782365  | 0.161998   |  0.0760403  |   0.261261  | 0.0378378  |         81.5611  |
| bright_aops                | BM25_RM3_Unified_Default  | 0.0541774  | 0.106406   |  0.0548584  |   0.207207  | 0.0288288  |        109.713   |
| bright_aops                | BM25_RM3_Unified_Analyzed | 0.0446329  | 0.0948055  |  0.0499249  |   0.189189  | 0.0216216  |         97.9669  |
| arguana                    | BM25_Default              | 0.367459   | 0.241613   |  0.765882   |   0.765882  | 0.0765882  |         27.6267  |
| arguana                    | BM25_Analyzed             | 0.365958   | 0.24056    |  0.763026   |   0.763026  | 0.0763026  |         29.9918  |
| arguana                    | BM25_RM3_Terrier_Default  | 0.36448    | 0.241028   |  0.755175   |   0.755175  | 0.0755175  |         59.3384  |
| arguana                    | BM25_RM3_Unified_Default  | 0.249003   | 0.155484   |  0.551749   |   0.551749  | 0.0551749  |         70.0887  |
| arguana                    | BM25_RM3_Unified_Analyzed | 0.340637   | 0.220023   |  0.72591    |   0.72591   | 0.072591   |         74.1018  |
| bright_stackoverflow       | BM25_Default              | 0.156134   | 0.184517   |  0.194518   |   0.307692  | 0.0581197  |         45.1286  |
| bright_stackoverflow       | BM25_Analyzed             | 0.14372    | 0.186416   |  0.159975   |   0.282051  | 0.0529915  |         58.5725  |
| bright_stackoverflow       | BM25_RM3_Terrier_Default  | 0.147962   | 0.158544   |  0.186459   |   0.282051  | 0.0632479  |         78.4452  |
| bright_stackoverflow       | BM25_RM3_Unified_Default  | 0.111435   | 0.125451   |  0.150983   |   0.264957  | 0.0487179  |         95.7822  |
| bright_stackoverflow       | BM25_RM3_Unified_Analyzed | 0.103885   | 0.1283     |  0.117596   |   0.222222  | 0.0401709  |         98.1293  |
| bright_leetcode            | BM25_Default              | 0.249605   | 0.302767   |  0.289789   |   0.415493  | 0.0591549  |        116.342   |
| bright_leetcode            | BM25_Analyzed             | 0.239292   | 0.297966   |  0.260446   |   0.380282  | 0.0549296  |        144.385   |
| bright_leetcode            | BM25_RM3_Terrier_Default  | 0.20529    | 0.25484    |  0.241667   |   0.352113  | 0.0514085  |        189.44    |
| bright_leetcode            | BM25_RM3_Unified_Default  | 0.103865   | 0.12272    |  0.111385   |   0.161972  | 0.028169   |        246.448   |
| bright_leetcode            | BM25_RM3_Unified_Analyzed | 0.188689   | 0.234965   |  0.208216   |   0.316901  | 0.0457746  |        274.781   |
| bright_theoremqa_theorems  | BM25_Default              | 0.0192084  | 0.0177632  |  0.0394737  |   0.0657895 | 0.00657895 |         29.3452  |
| bright_theoremqa_theorems  | BM25_Analyzed             | 0.00509017 | 0.00657895 |  0.00657895 |   0.0131579 | 0.00131579 |         31.0678  |
| bright_theoremqa_theorems  | BM25_RM3_Terrier_Default  | 0.00818472 | 0.0060307  |  0.0197368  |   0.0263158 | 0.00263158 |         44.7266  |
| bright_theoremqa_theorems  | BM25_RM3_Unified_Default  | 0.0236439  | 0.0339912  |  0.0263158  |   0.0526316 | 0.00526316 |         55.1223  |
| bright_theoremqa_theorems  | BM25_RM3_Unified_Analyzed | 0.00268924 | 0.0018797  |  0.00657895 |   0.0131579 | 0.00131579 |         56.1328  |
| bright_earth_science       | BM25_Default              | 0.120131   | 0.158292   |  0.152444   |   0.318966  | 0.0560345  |         35.6928  |
| bright_earth_science       | BM25_Analyzed             | 0.0883679  | 0.116451   |  0.108574   |   0.258621  | 0.0422414  |         32.1121  |
| bright_earth_science       | BM25_RM3_Terrier_Default  | 0.0927712  | 0.112555   |  0.110768   |   0.215517  | 0.0413793  |         51.3914  |
| bright_earth_science       | BM25_RM3_Unified_Default  | 0.107954   | 0.141783   |  0.115719   |   0.241379  | 0.0491379  |         59.7735  |
| bright_earth_science       | BM25_RM3_Unified_Analyzed | 0.0874554  | 0.11093    |  0.102626   |   0.206897  | 0.0387931  |         63.3706  |
| bright_theoremqa_questions | BM25_Default              | 0.0700047  | 0.0803837  |  0.0852234  |   0.113402  | 0.0154639  |         45.5859  |
| bright_theoremqa_questions | BM25_Analyzed             | 0.072683   | 0.0815435  |  0.0869416  |   0.118557  | 0.0159794  |         50.9202  |
| bright_theoremqa_questions | BM25_RM3_Terrier_Default  | 0.111914   | 0.125221   |  0.126031   |   0.164948  | 0.0237113  |         78.1716  |
| bright_theoremqa_questions | BM25_RM3_Unified_Default  | 0.0382238  | 0.0442072  |  0.0468213  |   0.0670103 | 0.00876289 |        101.947   |
| bright_theoremqa_questions | BM25_RM3_Unified_Analyzed | 0.0651103  | 0.0712629  |  0.0781787  |   0.103093  | 0.0134021  |        109.093   |
| nq                         | BM25_Default              | 0.281401   | 0.243373   |  0.440542   |   0.475377  | 0.0516222  |         49.0671  |
| nq                         | BM25_Analyzed             | 0.281467   | 0.243628   |  0.43791    |   0.473638  | 0.0510718  |         58.4543  |
| nq                         | BM25_RM3_Terrier_Default  | 0.281793   | 0.240149   |  0.44508    |   0.475087  | 0.0522306  |         90.2207  |
| nq                         | BM25_RM3_Unified_Default  | 0.255282   | 0.21612    |  0.414204   |   0.448436  | 0.0488702  |        142.555   |
| nq                         | BM25_RM3_Unified_Analyzed | 0.278041   | 0.239241   |  0.437621   |   0.471611  | 0.0510139  |        272.475   |
| bright_psychology          | BM25_Default              | 0.0925743  | 0.0995992  |  0.130116   |   0.217822  | 0.0425743  |         38.8462  |
| bright_psychology          | BM25_Analyzed             | 0.0716814  | 0.0749568  |  0.108223   |   0.158416  | 0.0346535  |         36.7664  |
| bright_psychology          | BM25_RM3_Terrier_Default  | 0.0790437  | 0.0784103  |  0.112437   |   0.19802   | 0.039604   |         55.9356  |
| bright_psychology          | BM25_RM3_Unified_Default  | 0.0888636  | 0.10077    |  0.103108   |   0.168317  | 0.0455446  |         63.9126  |
| bright_psychology          | BM25_RM3_Unified_Analyzed | 0.0849255  | 0.0820957  |  0.10802    |   0.148515  | 0.039604   |         67.5207  |
| bright_sustainable_living  | BM25_Default              | 0.0981412  | 0.124949   |  0.139536   |   0.305556  | 0.0490741  |         32.9777  |
| bright_sustainable_living  | BM25_Analyzed             | 0.0898665  | 0.124067   |  0.124547   |   0.259259  | 0.0388889  |         37.2472  |
| bright_sustainable_living  | BM25_RM3_Terrier_Default  | 0.0863675  | 0.109189   |  0.120921   |   0.25      | 0.0425926  |         54.2753  |
| bright_sustainable_living  | BM25_RM3_Unified_Default  | 0.0734338  | 0.104512   |  0.102531   |   0.212963  | 0.0333333  |         60.6265  |
| bright_sustainable_living  | BM25_RM3_Unified_Analyzed | 0.0579429  | 0.0835501  |  0.0883488  |   0.166667  | 0.0277778  |         67.7886  |

### Indexing Efficiency (TTI in Seconds)

| dataset                    | pipeline                  |   default_index_time_s |   analyzed_pretokenize_s |   analyzed_index_s |   analyzed_total_tti_s |
|:---------------------------|:--------------------------|-----------------------:|-------------------------:|-------------------:|-----------------------:|
| quora                      | BM25_Default              |                   0    |                        0 |               0    |                   0    |
| quora                      | BM25_Analyzed             |                   0    |                        0 |               0    |                   0    |
| quora                      | BM25_RM3_Terrier_Default  |                   0    |                        0 |               0    |                   0    |
| quora                      | BM25_RM3_Unified_Default  |                   0    |                        0 |               0    |                   0    |
| quora                      | BM25_RM3_Unified_Analyzed |                   0    |                        0 |               0    |                   0    |
| bright_robotics            | BM25_Default              |                   2.66 |                        0 |               3.99 |                   3.99 |
| bright_robotics            | BM25_Analyzed             |                   2.66 |                        0 |               3.99 |                   3.99 |
| bright_robotics            | BM25_RM3_Terrier_Default  |                   2.66 |                        0 |               3.99 |                   3.99 |
| bright_robotics            | BM25_RM3_Unified_Default  |                   2.66 |                        0 |               3.99 |                   3.99 |
| bright_robotics            | BM25_RM3_Unified_Analyzed |                   2.66 |                        0 |               3.99 |                   3.99 |
| bright_pony                | BM25_Default              |                   0.4  |                        0 |               0.57 |                   0.57 |
| bright_pony                | BM25_Analyzed             |                   0.4  |                        0 |               0.57 |                   0.57 |
| bright_pony                | BM25_RM3_Terrier_Default  |                   0.4  |                        0 |               0.57 |                   0.57 |
| bright_pony                | BM25_RM3_Unified_Default  |                   0.4  |                        0 |               0.57 |                   0.57 |
| bright_pony                | BM25_RM3_Unified_Analyzed |                   0.4  |                        0 |               0.57 |                   0.57 |
| trec_covid                 | BM25_Default              |                  22.59 |                        0 |              46.3  |                  46.3  |
| trec_covid                 | BM25_Analyzed             |                  22.59 |                        0 |              46.3  |                  46.3  |
| trec_covid                 | BM25_RM3_Terrier_Default  |                  22.59 |                        0 |              46.3  |                  46.3  |
| trec_covid                 | BM25_RM3_Unified_Default  |                  22.59 |                        0 |              46.3  |                  46.3  |
| trec_covid                 | BM25_RM3_Unified_Analyzed |                  22.59 |                        0 |              46.3  |                  46.3  |
| nfcorpus                   | BM25_Default              |                   0.77 |                        0 |               1.53 |                   1.53 |
| nfcorpus                   | BM25_Analyzed             |                   0.77 |                        0 |               1.53 |                   1.53 |
| nfcorpus                   | BM25_RM3_Terrier_Default  |                   0.77 |                        0 |               1.53 |                   1.53 |
| nfcorpus                   | BM25_RM3_Unified_Default  |                   0.77 |                        0 |               1.53 |                   1.53 |
| nfcorpus                   | BM25_RM3_Unified_Analyzed |                   0.77 |                        0 |               1.53 |                   1.53 |
| hotpotqa                   | BM25_Default              |                 321.36 |                        0 |             505.84 |                 505.84 |
| hotpotqa                   | BM25_Analyzed             |                 321.36 |                        0 |             505.84 |                 505.84 |
| hotpotqa                   | BM25_RM3_Terrier_Default  |                 321.36 |                        0 |             505.84 |                 505.84 |
| hotpotqa                   | BM25_RM3_Unified_Default  |                 321.36 |                        0 |             505.84 |                 505.84 |
| hotpotqa                   | BM25_RM3_Unified_Analyzed |                 321.36 |                        0 |             505.84 |                 505.84 |
| fever                      | BM25_Default              |                 554.55 |                        0 |             975.16 |                 975.16 |
| fever                      | BM25_Analyzed             |                 554.55 |                        0 |             975.16 |                 975.16 |
| fever                      | BM25_RM3_Terrier_Default  |                 554.55 |                        0 |             975.16 |                 975.16 |
| fever                      | BM25_RM3_Unified_Default  |                 554.55 |                        0 |             975.16 |                 975.16 |
| fever                      | BM25_RM3_Unified_Analyzed |                 554.55 |                        0 |             975.16 |                 975.16 |
| fiqa                       | BM25_Default              |                   5.73 |                        0 |              10.74 |                  10.74 |
| fiqa                       | BM25_Analyzed             |                   5.73 |                        0 |              10.74 |                  10.74 |
| fiqa                       | BM25_RM3_Terrier_Default  |                   5.73 |                        0 |              10.74 |                  10.74 |
| fiqa                       | BM25_RM3_Unified_Default  |                   5.73 |                        0 |              10.74 |                  10.74 |
| fiqa                       | BM25_RM3_Unified_Analyzed |                   5.73 |                        0 |              10.74 |                  10.74 |
| scidocs                    | BM25_Default              |                   5.7  |                        0 |               7.56 |                   7.56 |
| scidocs                    | BM25_Analyzed             |                   5.7  |                        0 |               7.56 |                   7.56 |
| scidocs                    | BM25_RM3_Terrier_Default  |                   5.7  |                        0 |               7.56 |                   7.56 |
| scidocs                    | BM25_RM3_Unified_Default  |                   5.7  |                        0 |               7.56 |                   7.56 |
| scidocs                    | BM25_RM3_Unified_Analyzed |                   5.7  |                        0 |               7.56 |                   7.56 |
| webis_touche2020           | BM25_Default              |                  64.5  |                        0 |             156.12 |                 156.12 |
| webis_touche2020           | BM25_Analyzed             |                  64.5  |                        0 |             156.12 |                 156.12 |
| webis_touche2020           | BM25_RM3_Terrier_Default  |                  64.5  |                        0 |             156.12 |                 156.12 |
| webis_touche2020           | BM25_RM3_Unified_Default  |                  64.5  |                        0 |             156.12 |                 156.12 |
| webis_touche2020           | BM25_RM3_Unified_Analyzed |                  64.5  |                        0 |             156.12 |                 156.12 |
| bright_economics           | BM25_Default              |                   4.88 |                        0 |               5.73 |                   5.73 |
| bright_economics           | BM25_Analyzed             |                   4.88 |                        0 |               5.73 |                   5.73 |
| bright_economics           | BM25_RM3_Terrier_Default  |                   4.88 |                        0 |               5.73 |                   5.73 |
| bright_economics           | BM25_RM3_Unified_Default  |                   4.88 |                        0 |               5.73 |                   5.73 |
| bright_economics           | BM25_RM3_Unified_Analyzed |                   4.88 |                        0 |               5.73 |                   5.73 |
| bright_biology             | BM25_Default              |                   4.34 |                        0 |               5.18 |                   5.18 |
| bright_biology             | BM25_Analyzed             |                   4.34 |                        0 |               5.18 |                   5.18 |
| bright_biology             | BM25_RM3_Terrier_Default  |                   4.34 |                        0 |               5.18 |                   5.18 |
| bright_biology             | BM25_RM3_Unified_Default  |                   4.34 |                        0 |               5.18 |                   5.18 |
| bright_biology             | BM25_RM3_Unified_Analyzed |                   4.34 |                        0 |               5.18 |                   5.18 |
| climate_fever              | BM25_Default              |                 530.95 |                        0 |             926.18 |                 926.18 |
| climate_fever              | BM25_Analyzed             |                 530.95 |                        0 |             926.18 |                 926.18 |
| climate_fever              | BM25_RM3_Terrier_Default  |                 530.95 |                        0 |             926.18 |                 926.18 |
| climate_fever              | BM25_RM3_Unified_Default  |                 530.95 |                        0 |             926.18 |                 926.18 |
| climate_fever              | BM25_RM3_Unified_Analyzed |                 530.95 |                        0 |             926.18 |                 926.18 |
| scifact                    | BM25_Default              |                   2.11 |                        0 |               1.94 |                   1.94 |
| scifact                    | BM25_Analyzed             |                   2.11 |                        0 |               1.94 |                   1.94 |
| scifact                    | BM25_RM3_Terrier_Default  |                   2.11 |                        0 |               1.94 |                   1.94 |
| scifact                    | BM25_RM3_Unified_Default  |                   2.11 |                        0 |               1.94 |                   1.94 |
| scifact                    | BM25_RM3_Unified_Analyzed |                   2.11 |                        0 |               1.94 |                   1.94 |
| dbpedia_entity             | BM25_Default              |                 299.2  |                        0 |             496.77 |                 496.77 |
| dbpedia_entity             | BM25_Analyzed             |                 299.2  |                        0 |             496.77 |                 496.77 |
| dbpedia_entity             | BM25_RM3_Terrier_Default  |                 299.2  |                        0 |             496.77 |                 496.77 |
| dbpedia_entity             | BM25_RM3_Unified_Default  |                 299.2  |                        0 |             496.77 |                 496.77 |
| dbpedia_entity             | BM25_RM3_Unified_Analyzed |                 299.2  |                        0 |             496.77 |                 496.77 |
| bright_aops                | BM25_Default              |                  13.55 |                        0 |              30.37 |                  30.37 |
| bright_aops                | BM25_Analyzed             |                  13.55 |                        0 |              30.37 |                  30.37 |
| bright_aops                | BM25_RM3_Terrier_Default  |                  13.55 |                        0 |              30.37 |                  30.37 |
| bright_aops                | BM25_RM3_Unified_Default  |                  13.55 |                        0 |              30.37 |                  30.37 |
| bright_aops                | BM25_RM3_Unified_Analyzed |                  13.55 |                        0 |              30.37 |                  30.37 |
| arguana                    | BM25_Default              |                   0    |                        0 |               0    |                   0    |
| arguana                    | BM25_Analyzed             |                   0    |                        0 |               0    |                   0    |
| arguana                    | BM25_RM3_Terrier_Default  |                   0    |                        0 |               0    |                   0    |
| arguana                    | BM25_RM3_Unified_Default  |                   0    |                        0 |               0    |                   0    |
| arguana                    | BM25_RM3_Unified_Analyzed |                   0    |                        0 |               0    |                   0    |
| bright_stackoverflow       | BM25_Default              |                  12.82 |                        0 |              20.14 |                  20.14 |
| bright_stackoverflow       | BM25_Analyzed             |                  12.82 |                        0 |              20.14 |                  20.14 |
| bright_stackoverflow       | BM25_RM3_Terrier_Default  |                  12.82 |                        0 |              20.14 |                  20.14 |
| bright_stackoverflow       | BM25_RM3_Unified_Default  |                  12.82 |                        0 |              20.14 |                  20.14 |
| bright_stackoverflow       | BM25_RM3_Unified_Analyzed |                  12.82 |                        0 |              20.14 |                  20.14 |
| bright_leetcode            | BM25_Default              |                  39.33 |                        0 |              80.07 |                  80.07 |
| bright_leetcode            | BM25_Analyzed             |                  39.33 |                        0 |              80.07 |                  80.07 |
| bright_leetcode            | BM25_RM3_Terrier_Default  |                  39.33 |                        0 |              80.07 |                  80.07 |
| bright_leetcode            | BM25_RM3_Unified_Default  |                  39.33 |                        0 |              80.07 |                  80.07 |
| bright_leetcode            | BM25_RM3_Unified_Analyzed |                  39.33 |                        0 |              80.07 |                  80.07 |
| bright_theoremqa_theorems  | BM25_Default              |                   1.64 |                        0 |               3.95 |                   3.95 |
| bright_theoremqa_theorems  | BM25_Analyzed             |                   1.64 |                        0 |               3.95 |                   3.95 |
| bright_theoremqa_theorems  | BM25_RM3_Terrier_Default  |                   1.64 |                        0 |               3.95 |                   3.95 |
| bright_theoremqa_theorems  | BM25_RM3_Unified_Default  |                   1.64 |                        0 |               3.95 |                   3.95 |
| bright_theoremqa_theorems  | BM25_RM3_Unified_Analyzed |                   1.64 |                        0 |               3.95 |                   3.95 |
| bright_earth_science       | BM25_Default              |                   8.08 |                        0 |              12.01 |                  12.01 |
| bright_earth_science       | BM25_Analyzed             |                   8.08 |                        0 |              12.01 |                  12.01 |
| bright_earth_science       | BM25_RM3_Terrier_Default  |                   8.08 |                        0 |              12.01 |                  12.01 |
| bright_earth_science       | BM25_RM3_Unified_Default  |                   8.08 |                        0 |              12.01 |                  12.01 |
| bright_earth_science       | BM25_RM3_Unified_Analyzed |                   8.08 |                        0 |              12.01 |                  12.01 |
| bright_theoremqa_questions | BM25_Default              |                  13.46 |                        0 |              30.38 |                  30.38 |
| bright_theoremqa_questions | BM25_Analyzed             |                  13.46 |                        0 |              30.38 |                  30.38 |
| bright_theoremqa_questions | BM25_RM3_Terrier_Default  |                  13.46 |                        0 |              30.38 |                  30.38 |
| bright_theoremqa_questions | BM25_RM3_Unified_Default  |                  13.46 |                        0 |              30.38 |                  30.38 |
| bright_theoremqa_questions | BM25_RM3_Unified_Analyzed |                  13.46 |                        0 |              30.38 |                  30.38 |
| nq                         | BM25_Default              |                 190.24 |                        0 |             359.76 |                 359.76 |
| nq                         | BM25_Analyzed             |                 190.24 |                        0 |             359.76 |                 359.76 |
| nq                         | BM25_RM3_Terrier_Default  |                 190.24 |                        0 |             359.76 |                 359.76 |
| nq                         | BM25_RM3_Unified_Default  |                 190.24 |                        0 |             359.76 |                 359.76 |
| nq                         | BM25_RM3_Unified_Analyzed |                 190.24 |                        0 |             359.76 |                 359.76 |
| bright_psychology          | BM25_Default              |                   5.11 |                        0 |               6.28 |                   6.28 |
| bright_psychology          | BM25_Analyzed             |                   5.11 |                        0 |               6.28 |                   6.28 |
| bright_psychology          | BM25_RM3_Terrier_Default  |                   5.11 |                        0 |               6.28 |                   6.28 |
| bright_psychology          | BM25_RM3_Unified_Default  |                   5.11 |                        0 |               6.28 |                   6.28 |
| bright_psychology          | BM25_RM3_Unified_Analyzed |                   5.11 |                        0 |               6.28 |                   6.28 |
| bright_sustainable_living  | BM25_Default              |                   3.87 |                        0 |               6.2  |                   6.2  |
| bright_sustainable_living  | BM25_Analyzed             |                   3.87 |                        0 |               6.2  |                   6.2  |
| bright_sustainable_living  | BM25_RM3_Terrier_Default  |                   3.87 |                        0 |               6.2  |                   6.2  |
| bright_sustainable_living  | BM25_RM3_Unified_Default  |                   3.87 |                        0 |               6.2  |                   6.2  |
| bright_sustainable_living  | BM25_RM3_Unified_Analyzed |                   3.87 |                        0 |               6.2  |                   6.2  |
