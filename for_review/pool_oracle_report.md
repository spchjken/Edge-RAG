# 🔬 Stage 1 Report: Vocabulary Pool Oracle & Capacity Isolation (Final Recompiled & Audited)

**Status:** Completed, Audited & Methodologically Recompiled  
**Date:** September 16, 2026  
**Hardware Profile:** AMD Ryzen 7 / NVIDIA RTX (WSL2 Linux, 15 GiB RAM ceiling)  
**Evaluated Datasets:** `scifact` ($N=5,183$), `bright_aops` ($N=188,002$), `nfcorpus` ($N=3,633$), `trec_covid` ($N=171,332$) (50 sampled queries each, seed=42)  
**Master Execution Dataset:** [`results/pool_isolation/pool_candidate_audit.parquet`](file:///home/donghv/Projects/Edge-RAG/results/pool_isolation/pool_candidate_audit.parquet) (22.1 MB, 1,230,704 executed candidate-query-weight variants)  
**Analytical Outputs:**
- [`results/pool_isolation/pool_oracle_summary.csv`](file:///home/donghv/Projects/Edge-RAG/results/pool_isolation/pool_oracle_summary.csv) (Table 1: Independent Ceilings, Dual Joint-Safe Ceilings, 95% Bootstrap CIs)
- [`results/pool_isolation/pool_df_band_attribution.csv`](file:///home/donghv/Projects/Edge-RAG/results/pool_isolation/pool_df_band_attribution.csv) (Table 2: DF Band Attribution across Candidate Instances)
- [`results/pool_isolation/pool_df1_dominance.csv`](file:///home/donghv/Projects/Edge-RAG/results/pool_isolation/pool_df1_dominance.csv) (Dedicated 5-Bucket Mutually Exclusive DF=1 Dominance Audit)
- [`results/pool_isolation/pool_capacity_knee_curve.csv`](file:///home/donghv/Projects/Edge-RAG/results/pool_isolation/pool_capacity_knee_curve.csv) (Table 3: Fixed, Percentage, and Evaluated 15%/2.5k/10k Adaptive Capacities)
- [`results/pool_isolation/pool_weight_calibration.csv`](file:///home/donghv/Projects/Edge-RAG/results/pool_isolation/pool_weight_calibration.csv) (Unconditional & Conditional Optimal Weight Distributions across DF Bands & IDF Deciles)
- [`results/pool_isolation/metadata_manifest.json`](file:///home/donghv/Projects/Edge-RAG/results/pool_isolation/metadata_manifest.json) (Environment, true doc counts, eligible vocab sizes, unified tolerances)

---

## 1. Executive Summary & Core Verdicts

This report establishes the empirical findings for **Stage 1 (Vocabulary Pool Creation & Capacity Isolation)** of Edge-RAG. All metrics have been compiled with strict **per-query abstention** ($U_q^{\text{oracle}} = \max(U_q^{\text{base}}, \max_{t,\mu} U_q(t,\mu))$), candidate instance deduplication (`["dataset", "qid", "candidate_term"]`), unified numerical tolerances ($\tau = 10^{-5}, \epsilon = 0.001$), dual joint-safe oracles, index-derived percentage capacities, and bootstrap 95% confidence intervals.

### Key Empirical Takeaways:

1. **Large Latent Headroom Exists in the Corpus Vocabulary:**
   - Under an abstaining unigram oracle, retrieval effectiveness improves substantially across all four anchor corpora:
     - **SciFact:** nDCG@10 increases from $0.6444 \to 0.9069$ ($\Delta = +0.2625$, 95% CI $[0.1757, 0.3602]$).
     - **BRIGHT-AOPS ($N=188,002$):** nDCG@10 increases from $0.0372 \to 0.1245$ ($\Delta = +0.0872$, 95% CI $[0.0571, 0.1261]$), while Recall@1000 increases from $0.4589 \to 0.6179$ ($\Delta = +0.1589$, 95% CI $[0.1081, 0.2207]$). *(Note: while relative gain on AOPS is large, absolute final effectiveness remains low, reflecting the inherent difficulty of competition math).*
     - **NFCorpus:** nDCG@10 increases from $0.3098 \to 0.7384$ ($\Delta = +0.4286$, 95% CI $[0.3375, 0.5091]$), with Recall@1000 doubling from $0.3559 \to 0.7668$ ($\Delta = +0.4109$, 95% CI $[0.3192, 0.5138]$).
     - **TREC-COVID:** nDCG@10 increases from $0.6065 \to 0.9105$ ($\Delta = +0.3040$, 95% CI $[0.2502, 0.3596]$).
   - > [!IMPORTANT]
     > **Boundary of Stage 1:** Because oracle candidates are extracted from judged relevant documents, this finding proves that *at least one unigram from the corpus vocabulary can substantially improve retrieval when both term and weight are selected correctly*. It establishes the **opportunity available to a selector**, but does **not** prove that a practical, query-only selector can discover those terms or reject thousands of non-useful pool terms. The research challenge now shifts squarely to selector design.

2. **$DF=1$ Terms are Frequently Useful Individually, but Overwhelmingly Dominated:**
   - Direct query-level dominance analysis reveals that $DF=1$ terms are active and helpful in **86.0%** of TREC-COVID queries and **72.0%** of NFCorpus queries.
   - However, in **86.0%** of TREC-COVID queries and **60.0%** of NFCorpus queries, the best $DF=1$ candidate is **strictly dominated** by an eligible term with broader corpus support ($DF \ge 2$).
   - In BRIGHT-AOPS and SciFact, $DF=1$ terms **never** win outright against eligible candidates ($0.0\%$ wins).
   - In NFCorpus (biomedical clinical entities), $DF=1$ wins outright in only **8.0%** of queries (4/50).
   - *Policy Conclusion:* Excluding $DF=1$ terms removes over $60\%$ of noisy vocabulary entries while losing only ~6.4% of total potential gain on clinical corpora and 0% on math/scientific fact verification. An optional rare-entity rescue channel may be preserved for high-confidence contextual matches, but default exclusion from the principal pool remains sound.

3. **Deployable Pool Capacity: $B=10\text{k}$ Provisional Default & Evaluated 15% Adaptive Rule:**
   - **Fixed 10k Capacity:** Retains **100.0%** of eligible ceiling on SciFact and NFCorpus, **97.5%** on BRIGHT-AOPS ($+0.0851$ vs $+0.0872$), and **97.2%** on TREC-COVID ($+0.2953$ vs $+0.3040$). Fixed 10k serves as a strong provisional default for Stage 2 development.
   - **Evaluated 15% Adaptive Rule:** We explicitly evaluated the proposed rule:
     $$B = \min(10{,}000, \max(2{,}500, \lfloor 0.15 \cdot |V_{\text{eligible}}| \rfloor))$$
     yielding exact evaluated capacities of:
     - **SciFact ($|V|=5,595$):** $B = 2,500$ terms ($44.7\%$), retaining **$88.2\% - 95.8\%$** across policies.
     - **BRIGHT-AOPS ($|V|=33,656$):** $B = 5,048$ terms ($15.0\%$), retaining **$80.6\% - 87.8\%$**.
     - **NFCorpus ($|V|=11,811$):** $B = 2,500$ terms ($21.2\%$), retaining **$83.5\% - 96.9\%$**.
     - **TREC-COVID ($|V|=59,395$):** $B = 8,909$ terms ($15.0\%$), retaining **$92.0\% - 97.2\%$**.
     Macro retention across the four corpora reaches **90.5%** for `salience`, **93.1%** for `specificity`, **93.1%** for `hybrid`, and **88.1%** for `stratified`. The $B_{\min}=2,500$ floor prevents severe under-allocation on small vocabularies.
   - > [!WARNING]
     > **Corpus Scaling Caveat:** Percentage-based capacity appears promising across the four tested collections, but its behavior on million-document collections remains unverified (the largest tested collection has 188,002 documents). Fixed 10k is a provisional development default, not a universal optimum.

4. **Single Interventions Can Simultaneously Improve Both Precision and Recall:**
   - Under the same-intervention ranking-priority joint safe oracle (single best $(t, \mu)$ maximizing nDCG subject to $\Delta R@1000 \ge -\tau$):
     - On SciFact: $+0.2625$ nDCG@10 with $\Delta R@1000 = 0.0000$ (recall preserved).
     - On AOPS: $+0.0872$ nDCG@10 with $\Delta R@1000 = +0.0083$ (recall slightly increased).
     - On NFCorpus: $+0.4286$ nDCG@10 with $\Delta R@1000 = +0.2606$ (both precision and deep recall surge).
     - On TREC-COVID: $+0.2768$ nDCG@10 with $\Delta R@1000 = +0.0334$.
   - A single well-chosen unigram expansion does **not** inherently force an adverse precision-recall tradeoff.

5. **Weight Calibration: Dispersed Broad Weights vs. Score-Deficit Dynamics:**
   - Multi-weight exploration is essential: single small weight ($\mu=0.10$) captures only $+0.0015$ on AOPS (vs $+0.0872$ multi-weight).
   - **Denominator Clarification:** In `pool_weight_calibration.csv`, calibration metrics distinguish *percentage of all candidate instances* (unconditional) from *percentage of instances with a safe weight* (conditional). For example, in NFCorpus D10 (rare terms), $\mu=1.00$ accounts for $50.23\%$ of all candidate instances; conditional on having a safe weight (73.43% safe, 26.57% no safe weight), **68.40%** of safe instances require $\mu=1.00$.
   - **Score-Deficit Model:** Broad terms do not consistently favor low weights ($\mu \in [0.05, 0.10]$); rather, their safe weights are dispersed (often favoring $0.30 - 0.50$), and $70\% - 90\%$ have no safe weight at all. Rare terms frequently require large weights ($\mu=1.00$) because the relevant document begins far below the retrieval boundary:
     $$\mu(q, t) = c(q, t) \cdot \frac{\Delta_{\mathrm{needed}}(q, t)}{I(t) + \epsilon}$$
     IDF measures marginal term impact per unit weight $I(t)$, but does not dictate the score deficit $\Delta_{\mathrm{needed}}$ required to cross the retrieval threshold.

---

## 2. Experimental Design & Reconciled Policy Specifications

### 2.1 Reconciled Pool Construction Policies
The five deterministic nested candidate pool sequences generated in `src/evaluation/pool_generators.py` are defined as follows:

1. **`salience`:** Ranked descending by sublinear corpus salience:
   $$\text{Score}_{\text{sal}}(t) = \text{IDF}(t) \times \ln(1 + \text{DF}(t))$$
2. **`specificity`:** Ranked descending by sublinear collection frequency impact:
   $$\text{Score}_{\text{spec}}(t) = \text{IDF}(t) \times \ln(1 + \text{CF}(t))$$
3. **`hybrid`:** Ranked descending by frequency-damped collection impact:
   $$\text{Score}_{\text{hyb}}(t) = \text{IDF}(t) \times \ln(1 + \text{CF}(t)) \times \max\left(1.0 - \frac{\text{DF}(t)}{N}, 0.0\right)$$
4. **`stratified`:** Proportional round-robin interleaving across four disjoint strata:
   - $S_4$ (Index-Visible Compounds & Alphanumerics): Quota **10%** (digits or punctuation).
   - $S_1$ (High Specificity, $\text{spec} \ge 0.75$): Quota **30%**.
   - $S_2$ (Medium Specificity, $0.40 \le \text{spec} < 0.75$): Quota **40%**.
   - $S_3$ (Broad Context / Bridge Concepts, $\text{spec} < 0.40$): Quota **20%**.
   - Round-robin period: $[S_1, S_2, S_1, S_2, S_3, S_1, S_2, S_3, S_2, S_4]$ (ranked internally by $\text{Score}_{\text{sal}}$).
5. **`coverage`:** Cost-Effective Lazy Forward (CELF) submodular greedy selection maximizing IDF-weighted unique document coverage:
   $$\Delta_{\text{gain}}(t \mid C) = \text{IDF}(t) \times |\text{Docs}(t) \setminus C|$$

### 2.2 Mathematical Evaluation Contracts
- **Zero Tolerance:** $\tau = 10^{-5}$.
- **Recall Safety Margin:** $\epsilon = 0.001$.
- **Per-Query Abstention:** $U_q^{\text{oracle}} = \max(U_q^{\text{base}}, \max_{t,\mu} U_q(t,\mu))$.
- **Candidate Deduplication:** Grouped by `["dataset", "qid", "candidate_term"]` as **query-term candidate instances**.
- **Deployable Candidates:** Candidates with rank $\le 20,000$ in any evaluated policy received the full 5-weight grid $\mu \in \{0.05, 0.10, 0.30, 0.50, 1.00\}$.
- **Status of Full-Vocabulary Multi-Weight Ceilings:** Labeled strictly as **Screened Lower Bounds** (exact at $\mu=0.10$, multi-weight evaluated for terms active at 0.10 or deployable).

---

## 3. Table 1: Global Ceilings, Dual Joint-Safe Oracles & 10k Deployable Benchmarks

| Metric / Configuration | SciFact ($N=5,183$) | BRIGHT-AOPS ($N=188,002$) | NFCorpus ($N=3,633$) | TREC-COVID ($N=171,332$) |
|---|:---:|:---:|:---:|:---:|
| **Sampled Queries ($n$)** | 50 | 50 | 50 | 50 |
| **Eligible Vocab Size ($|V_{\text{elig}}|$)** | 5,595 | 33,656 | 11,811 | 59,395 |
| **Default BM25 nDCG@10** | 0.6444 | 0.0372 | 0.3098 | 0.6065 |
| **Default BM25 Recall@100** | 0.9300 | 0.1951 | 0.2540 | 0.1206 |
| **Default BM25 Recall@1000** | 0.9800 | 0.4589 | 0.3559 | 0.4486 |
| **Raw Lexical Ceiling (Screened Best-W)** | **0.9069** ($+0.2625$) | **0.1245** ($+0.0872$) | **0.7677** ($+0.4579$) | **0.9113** ($+0.3047$) |
| — 95% Bootstrap CI on $\Delta$ | $[0.1757, 0.3602]$ | $[0.0571, 0.1261]$ | $[0.3575, 0.5477]$ | $[0.2508, 0.3604]$ |
| — Raw Ceiling at $\mu=0.10$ (Exact) | 0.7226 ($+0.0782$) | 0.0389 ($+0.0017$) | 0.4719 ($+0.1621$) | 0.7010 ($+0.0945$) |
| **Eligible Ceiling (Screened Best-W)** | **0.9069** ($+0.2625$) | **0.1245** ($+0.0872$) | **0.7384** ($+0.4286$) | **0.9105** ($+0.3040$) |
| — 95% Bootstrap CI on $\Delta$ | $[0.1757, 0.3602]$ | $[0.0571, 0.1261]$ | $[0.3375, 0.5091]$ | $[0.2502, 0.3596]$ |
| — Eligible Ceiling at $\mu=0.10$ (Exact) | 0.7153 ($+0.0708$) | 0.0387 ($+0.0015$) | 0.4706 ($+0.1608$) | 0.6995 ($+0.0929$) |
| — Eligible Ceiling Recall@1000 | 1.0000 ($+0.0200$) | 0.6179 ($+0.1589$) | 0.7668 ($+0.4109$) | 0.5315 ($+0.0829$) |
| **Ranking-Priority Joint Safe Oracle** | **0.9069** ($+0.2625$) | **0.1245** ($+0.0872$) | **0.7384** ($+0.4286$) | **0.8833** ($+0.2768$) |
| — 95% Bootstrap CI on $\Delta$nDCG | $[0.1757, 0.3602]$ | $[0.0571, 0.1261]$ | $[0.3375, 0.5091]$ | $[0.2278, 0.3268]$ |
| — Resulting $\Delta R@1000$ (Paired) | $+0.0000$ | $+0.0083$ | $+0.2606$ | $+0.0334$ |
| **Recall-Priority Joint Safe Oracle** | **1.0000** ($+0.0200$) | **0.6179** ($+0.1589$) | **0.7665** ($+0.4106$) | **0.5263** ($+0.0776$) |
| — 95% Bootstrap CI on $\Delta$R@1000 | $[0.0000, 0.0600]$ | $[0.1081, 0.2207]$ | $[0.3192, 0.5138]$ | $[0.0554, 0.1064]$ |
| — Resulting $\Delta$nDCG@10 (Paired) | $+0.0000$ | $+0.0129$ | $+0.2159$ | $+0.1286$ |
| **Deployable: Salience ($B=10\text{k}$ Exact)** | **0.9069** ($+0.2625$) | **0.1223** ($+0.0851$) | **0.7384** ($+0.4286$) | **0.9005** ($+0.2939$) |
| **Deployable: Specificity ($B=10\text{k}$ Exact)** | **0.9069** ($+0.2625$) | **0.1223** ($+0.0851$) | **0.7384** ($+0.4286$) | 0.8949 ($+0.2883$) |
| **Deployable: Hybrid ($B=10\text{k}$ Exact)** | **0.9069** ($+0.2625$) | **0.1223** ($+0.0851$) | **0.7384** ($+0.4286$) | 0.8944 ($+0.2879$) |
| **Deployable: Stratified ($B=10\text{k}$ Exact)** | **0.9069** ($+0.2625$) | 0.1213 ($+0.0841$) | **0.7384** ($+0.4286$) | **0.9019** ($+0.2953$) |
| **Deployable: Coverage ($B=10\text{k}$ Exact)** | 0.9043 ($+0.2599$) | 0.1085 ($+0.0712$) | **0.7384** ($+0.4286$) | 0.8860 ($+0.2795$) |

---

## 4. Direct DF=1 vs. Eligible Dominance Audit

Using the mutually exclusive five-bucket decision tree, each query is categorized by comparing $G_q^{\text{DF1}} = \max_{t \in B_1, \mu} \Delta U_q(t,\mu)$ against $G_q^{\text{eligible}} = \max_{t \in V_{\text{eligible}}, \mu} \Delta U_q(t,\mu)$:

| Dataset | Scope | Total Queries | Neither Useful | DF=1 Wins | Eligible Wins, DF1 Inactive | Useful Tie | DF=1 Useful but Dominated |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`scifact`** | Reference $\mu=0.10$ | 50 | 35 (70.0%) | 1 (2.0%) | 8 (16.0%) | 4 (8.0%) | 2 (4.0%) |
| | Screened Multi-Weight | 50 | 27 (54.0%) | **0 (0.0%)** | 16 (32.0%) | 5 (10.0%) | 2 (4.0%) |
| **`bright_aops`** | Reference $\mu=0.10$ | 50 | 49 (98.0%) | 0 (0.0%) | 1 (2.0%) | 0 (0.0%) | 0 (0.0%) |
| | Screened Multi-Weight | 50 | 29 (58.0%) | **0 (0.0%)** | 21 (42.0%) | 0 (0.0%) | 0 (0.0%) |
| **`nfcorpus`** | Reference $\mu=0.10$ | 50 | 22 (44.0%) | 2 (4.0%) | 2 (4.0%) | 5 (10.0%) | 19 (38.0%) |
| | Screened Multi-Weight | 50 | 9 (18.0%) | **4 (8.0%)** | 5 (10.0%) | 2 (4.0%) | **30 (60.0%)** |
| **`trec_covid`** | Reference $\mu=0.10$ | 50 | 2 (4.0%) | 1 (2.0%) | 14 (28.0%) | 5 (10.0%) | 28 (56.0%) |
| | Screened Multi-Weight | 50 | 1 (2.0%) | **0 (0.0%)** | 6 (12.0%) | 0 (0.0%) | **43 (86.0%)** |

> [!IMPORTANT]
> **Dominance Findings:**
> 1. In TREC-COVID, $DF=1$ terms are active and helpful in 86.0% of queries, but in **every single one of those queries**, an eligible term with broader corpus support matches or exceeds their performance ($0.0\%$ DF=1 wins).
> 2. In BRIGHT-AOPS and SciFact, $DF=1$ terms never win outright against eligible candidates ($0.0\%$).
> 3. In NFCorpus (specialized clinical terminology), $DF=1$ terms win outright in only 4 of 50 queries ($8.0\%$).
> 4. *Conclusion:* Excluding $DF=1$ terms from the primary pool dramatically reduces vocabulary noise while preserving $>93.6\%$ of actionable headroom.

---

## 5. Table 2: DF Band Attribution across Candidate Instances

Candidate instances are grouped by `["dataset", "qid", "candidate_term"]`. We report exact reference-weight utility ($\mu=0.10$), screened any-weight utility (lower bound), and exact any-weight/robustness for deployable instances ($\ge 3$ weights safe):

| Dataset | DF Band | Candidate Instances | Unique Terms | Queries With Cands | Mean Max $\Delta\text{nDCG}$ | Mean Max $\Delta R@1000$ | % Queries Helped | Ref-W Useful % (Exact) | Any-W Useful % (Deployable) | Ranking Robust % ($\ge 3$ w) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`scifact`** | $B_1: DF=1$ | 97 | 97 | 68.0% | +0.0639 | +0.0200 | 14.0% | 14.4% | NA | NA |
| | $B_2: 2 \le DF \le 5$ | 190 | 190 | 88.0% | +0.2149 | +0.0200 | 40.0% | 23.2% | 54.0% | 31.4% |
| | $B_3: 6 \le DF \le 20$ | 322 | 321 | 98.0% | **+0.2506** | +0.0200 | **44.0%** | 22.0% | 45.2% | 26.5% |
| | $B_5: 0.1\% \le DF/N \le 1\%$ | 400 | 399 | 98.0% | +0.2135 | +0.0200 | 44.0% | 18.0% | 37.6% | 17.8% |
| | $B_6: DF/N > 1\%$ | 2,618 | 2,617 | 100.0% | +0.2113 | +0.0200 | 44.0% | 13.9% | 32.1% | 15.8% |
| **`bright_aops`**| $B_1: DF=1$ | 70 | 70 | 82.0% | **0.0000** | **0.0000** | **0.0%** | **0.0%** | NA | NA |
| | $B_2: 2 \le DF \le 5$ | 101 | 97 | 74.0% | +0.0195 | +0.0599 | 14.0% | 1.0% | 10.3% | 10.3% |
| | $B_3: 6 \le DF \le 20$ | 240 | 227 | 80.0% | +0.0350 | +0.0649 | 22.0% | 0.8% | 8.4% | 7.5% |
| | $B_4: DF/N < 0.1\%$ | 925 | 870 | 98.0% | **+0.0647** | +0.1389 | **34.0%** | 0.8% | 8.6% | 8.4% |
| | $B_5: 0.1\% \le DF/N \le 1\%$ | 1,826 | 1,698 | 100.0% | +0.0520 | **+0.1481** | 30.0% | 0.7% | 4.7% | 3.8% |
| | $B_6: DF/N > 1\%$ | 2,866 | 2,694 | 100.0% | +0.0266 | +0.1073 | 22.0% | 0.4% | 2.8% | 2.1% |
| **`nfcorpus`** | $B_1: DF=1$ | 3,365 | 3,275 | 94.0% | +0.2017 | +0.1515 | 72.0% | 15.6% | NA | NA |
| | $B_2: 2 \le DF \le 5$ | 4,660 | 4,578 | 98.0% | +0.3388 | +0.2291 | 82.0% | 16.7% | 55.5% | 34.5% |
| | $B_3: 6 \le DF \le 20$ | 8,984 | 8,506 | 100.0% | **+0.3814** | +0.2937 | **82.0%** | 12.8% | 36.5% | 17.0% |
| | $B_5: 0.1\% \le DF/N \le 1\%$ | 5,050 | 4,877 | 100.0% | +0.2449 | +0.2969 | 78.0% | 9.0% | 24.9% | 10.1% |
| | $B_6: DF/N > 1\%$ | 21,180 | 19,890 | 100.0% | +0.2681 | **+0.4413** | 78.0% | 7.9% | 21.0% | 8.9% |
| **`trec_covid`** | $B_1: DF=1$ | 3,747 | 3,669 | 100.0% | +0.0718 | +0.0033 | 86.0% | 4.4% | NA | NA |
| | $B_2: 2 \le DF \le 5$ | 1,974 | 1,939 | 100.0% | +0.1421 | +0.0060 | 98.0% | 8.6% | 30.6% | 15.9% |
| | $B_3: 6 \le DF \le 20$ | 7,540 | 7,294 | 100.0% | +0.1816 | +0.0090 | 98.0% | 5.5% | 17.9% | 7.6% |
| | $B_4: DF/N < 0.1\%$ | 37,037 | 35,468 | 100.0% | +0.2302 | +0.0234 | 98.0% | 3.3% | 10.5% | 2.9% |
| | $B_5: 0.1\% \le DF/N \le 1\%$ | 74,006 | 70,146 | 100.0% | **+0.2739** | +0.0584 | **98.0%** | 3.5% | 7.6% | 1.1% |
| | $B_6: DF/N > 1\%$ | 52,053 | 48,938 | 100.0% | +0.2723 | **+0.0711** | 98.0% | 7.9% | 14.4% | 2.0% |

---

## 6. Table 3: Capacity Knee Curves & Evaluated Adaptive Rule

We evaluate fixed capacities ($B \in [1\text{k} \dots 20\text{k}]$), unbounded relative capacities ($p \in [2\%, 5\%, 10\%, 15\%, 20\%]$), and the **evaluated bounded adaptive rule**:
$$B = \min(10{,}000, \max(2{,}500, \lfloor 0.15 \cdot |V_{\text{eligible}}| \rfloor))$$

### 6.1 Corpus-by-Corpus Capacity Evaluations

#### SciFact ($|V_{\text{eligible}}| = 5,595$)
| Capacity Setting | Effective $B$ | % of $|V_{\text{elig}}|$ | `salience` Retained % | `stratified` Retained % | `specificity` Retained % | `hybrid` Retained % | `coverage` Retained % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$p = 2\%$** | 111 | 2.0% | 44.6% | 39.3% | 29.1% | 29.1% | 40.7% |
| **$p = 5\%$** | 279 | 5.0% | 66.6% | 66.7% | 51.7% | 51.7% | 46.7% |
| **$p = 10\%$** | 559 | 10.0% | 75.4% | 73.1% | 72.7% | 72.7% | 51.0% |
| **$p = 15\%$** | 839 | 15.0% | 79.8% | 82.6% | 85.2% | 80.3% | 69.2% |
| **Fixed $1\text{k}$** | 1,000 | 17.9% | 81.1% | 84.1% | 89.0% | 89.0% | 69.7% |
| **$p = 20\%$** | 1,119 | 20.0% | 81.1% | 84.1% | 89.0% | 89.0% | 69.7% |
| **Fixed $2.5\text{k}$** | 2,500 | 44.7% | 88.2% | 91.1% | 95.8% | 95.8% | 85.7% |
| **Adaptive Rule ($15\%/2.5\text{k}/10\text{k}$)** | **2,500** | **44.7%** | **88.2%** | **91.1%** | **95.8%** | **95.8%** | **85.7%** |
| **Fixed $5\text{k}$** | 5,000 | 89.4% | 94.9% | 96.8% | 95.8% | 95.8% | 91.8% |
| **Fixed $10\text{k}$** | 5,595* | 100.0% | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 99.0% |

#### BRIGHT-AOPS ($|V_{\text{eligible}}| = 33,656, N=188,002$)
| Capacity Setting | Effective $B$ | % of $|V_{\text{elig}}|$ | `salience` Retained % | `stratified` Retained % | `specificity` Retained % | `hybrid` Retained % | `coverage` Retained % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fixed $1\text{k}$** | 1,000 | 3.0% | 58.5% | 49.6% | 61.3% | 61.3% | 31.8% |
| **$p = 2\%$** | 673 | 2.0% | 49.1% | 42.1% | 52.4% | 52.4% | 23.4% |
| **$p = 5\%$** | 1,682 | 5.0% | 59.6% | 56.4% | 71.3% | 71.3% | 38.6% |
| **Fixed $2.5\text{k}$** | 2,500 | 7.4% | 74.4% | 63.2% | 79.1% | 79.1% | 44.8% |
| **$p = 10\%$** | 3,365 | 10.0% | 80.3% | 71.9% | 84.5% | 84.5% | 49.8% |
| **Fixed $5\text{k}$** | 5,000 | 14.9% | 87.6% | 80.6% | 87.8% | 87.8% | 58.7% |
| **$p = 15\%$** | 5,048 | 15.0% | 87.6% | 80.6% | 87.8% | 87.8% | 58.7% |
| **Adaptive Rule ($15\%/2.5\text{k}/10\text{k}$)** | **5,048** | **15.0%** | **87.6%** | **80.6%** | **87.8%** | **87.8%** | **58.7%** |
| **$p = 20\%$** | 6,731 | 20.0% | 88.5% | 84.8% | 89.2% | 89.2% | 66.2% |
| **Fixed $10\text{k}$** | 10,000 | 29.7% | **97.5%** | **96.4%** | **97.5%** | **97.5%** | 81.7% |

#### NFCorpus ($|V_{\text{eligible}}| = 11,811$)
| Capacity Setting | Effective $B$ | % of $|V_{\text{elig}}|$ | `salience` Retained % | `stratified` Retained % | `specificity` Retained % | `hybrid` Retained % | `coverage` Retained % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$p = 2\%$** | 236 | 2.0% | 42.1% | 35.9% | 69.8% | 70.3% | 44.1% |
| **$p = 5\%$** | 590 | 5.0% | 54.7% | 51.5% | 82.5% | 83.1% | 59.8% |
| **$p = 10\%$** | 1,181 | 10.0% | 64.9% | 67.9% | 87.0% | 88.4% | 71.0% |
| **Fixed $1\text{k}$** | 1,000 | 8.5% | 60.1% | 64.8% | 86.0% | 87.3% | 69.0% |
| **$p = 15\%$** | 1,771 | 15.0% | 72.1% | 78.4% | 92.4% | 93.7% | 76.9% |
| **$p = 20\%$** | 2,362 | 20.0% | 88.4% | 82.5% | 96.6% | 96.6% | 82.0% |
| **Fixed $2.5\text{k}$** | 2,500 | 21.2% | 90.0% | 83.5% | 96.6% | 96.9% | 83.8% |
| **Adaptive Rule ($15\%/2.5\text{k}/10\text{k}$)** | **2,500** | **21.2%** | **90.0%** | **83.5%** | **96.6%** | **96.9%** | **83.8%** |
| **Fixed $5\text{k}$** | 5,000 | 42.3% | 95.8% | 92.1% | 98.6% | 98.4% | 92.0% |
| **Fixed $10\text{k}$** | 10,000 | 84.7% | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 98.7% |

#### TREC-COVID ($|V_{\text{eligible}}| = 59,395, N=171,332$)
| Capacity Setting | Effective $B$ | % of $|V_{\text{elig}}|$ | `salience` Retained % | `stratified` Retained % | `specificity` Retained % | `hybrid` Retained % | `coverage` Retained % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fixed $1\text{k}$** | 1,000 | 1.7% | 74.0% | 70.8% | 58.7% | 58.7% | 46.2% |
| **$p = 2\%$** | 1,187 | 2.0% | 76.6% | 74.3% | 60.1% | 60.1% | 46.7% |
| **$p = 5\%$** | 2,969 | 5.0% | 89.2% | 86.8% | 69.8% | 69.8% | 65.3% |
| **Fixed $5\text{k}$** | 5,000 | 8.4% | 92.8% | 92.7% | 84.5% | 84.5% | 81.3% |
| **$p = 10\%$** | 5,939 | 10.0% | 92.8% | 93.9% | 88.0% | 88.0% | 84.8% |
| **$p = 15\%$** | 8,909 | 15.0% | 96.0% | 97.2% | 92.0% | 92.0% | 92.0% |
| **Adaptive Rule ($15\%/2.5\text{k}/10\text{k}$)** | **8,909** | **15.0%** | **96.0%** | **97.2%** | **92.0%** | **92.0%** | **92.0%** |
| **Fixed $10\text{k}$** | 10,000 | 16.8% | **97.2%** | **97.2%** | 94.6% | 94.6% | 93.2% |
| **$p = 20\%$** | 11,879 | 20.0% | 97.2% | 97.2% | 95.8% | 95.8% | 93.7% |
| **Fixed $20\text{k}$** | 20,000 | 33.7% | **98.7%** | **100.0%** | 97.5% | 97.0% | 95.8% |

---

### 6.2 Macro-Average Retention across All 4 Anchor Corpora

| Capacity Configuration | `salience` | `stratified` | `specificity` | `hybrid` | `coverage` |
|---|:---:|:---:|:---:|:---:|:---:|
| **$p = 2\%$ of $|V_{\text{elig}}|$** | 54.2% | 48.9% | **58.6%** | 54.1% | 38.2% |
| **$p = 5\%$ of $|V_{\text{elig}}|$** | 68.8% | 66.4% | **73.8%** | 71.5% | 52.6% |
| **$p = 10\%$ of $|V_{\text{elig}}|$** | 79.6% | 76.8% | **83.1%** | 81.8% | 66.4% |
| **$p = 15\%$ of $|V_{\text{elig}}|$ (Unbounded)** | 83.9% | 84.7% | **89.3%** | 88.5% | 74.2% |
| **$p = 20\%$ of $|V_{\text{elig}}|$** | 88.4% | 86.8% | **90.4%** | 89.9% | 78.4% |
| **Fixed $1,000$** | 68.4% | 67.3% | **73.7%** | 71.8% | 62.2% |
| **Fixed $2,500$** | 85.7% | 80.6% | **89.6%** | 89.7% | 74.8% |
| **Fixed $5,000$** | 91.5% | 91.5% | **91.6%** | 91.4% | 84.0% |
| **Fixed $10,000$** | **98.56%** | **98.38%** | 98.10% | 98.06% | 93.15% |
| **Evaluated Adaptive Rule ($15\% / 2.5\text{k} / 10\text{k}$)** | **90.46%** | **88.11%** | **93.05%** | **93.12%** | **80.04%** |

> [!TIP]
> **Resolution on Capacity Sizing:**
> 1. **Why the Floor Matters:** A naive unbounded $15\%$ rule allocates only 839 terms on SciFact and 1,771 on NFCorpus, capturing only 79.8% and 72.1% headroom respectively. The evaluated bounded rule with $B_{\min}=2,500$ ensures that smaller corpora receive sufficient lexical diversity, lifting macro retention from **83.9% $\to$ 90.5%** for `salience` and **89.3% $\to$ 93.1%** for `specificity`.
> 2. **Scientific Claim Boundary:** Percentage-based capacity appears promising across the four tested collections, but its behavior on million-document collections remains unverified. Fixed 10k can currently be called a strong provisional default for Stage 2 development, not a universal optimum.

---

## 7. Weight Calibration Analysis: Specificity vs. Score Deficit & Injection Weight

To understand how expansion weight interacts with corpus statistics, we examine deployable candidate instances under two objectives:
1. **Ranking-Safe Optimum:** $\arg\max_\mu \Delta\text{nDCG@10}$ subject to $\Delta R@1000 \ge -\tau$.
2. **Recall-Safe Optimum:** $\arg\max_\mu \Delta R@1000$ subject to $\Delta\text{nDCG@10} \ge -\epsilon$.

### 7.1 Distribution across Within-Dataset IDF Deciles (D1=Broadest $\to$ D10=Rarest)

The table reports **both** unconditional percentages (percentage of all candidate instances, including those with no safe weight) and conditional percentages (percentage of instances conditional on having at least one safe weight):

| Dataset | Decile | Candidate Instances | Safe Instances | No Safe Weight % | Safe % ($\mu=0.05$) [Uncond / Cond] | Safe % ($\mu=0.10$) [Uncond / Cond] | Safe % ($\mu=0.30$) [Uncond / Cond] | Safe % ($\mu=0.50$) [Uncond / Cond] | Safe % ($\mu=1.00$) [Uncond / Cond] |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`trec_covid`** | D1 (Broad) | 72,451 | 9,137 | **87.39%** | 0.95% / 7.56% | 1.84% / 14.59% | **4.22% / 33.42%** | **3.90% / 30.91%** | 1.70% / 13.52% |
| | D5 (Mid) | 8,263 | 846 | **89.76%** | 0.27% / 2.60% | 0.65% / 6.38% | 2.60% / 25.41% | **3.93% / 38.42%** | 2.78% / 27.19% |
| | D10 (Rare) | 2,456 | 714 | 70.93% | 0.49% / 1.68% | 0.53% / 1.82% | 4.40% / 15.13% | 8.75% / 30.11% | **14.90% / 51.26%** |
| **`nfcorpus`** | D1 (Broad) | 14,073 | 2,841 | **79.81%** | 2.96% / 14.68% | 0.78% / 3.87% | 3.97% / 19.64% | 3.66% / 18.13% | **8.82% / 43.68%** |
| | D5 (Mid) | 2,715 | 1,000 | 63.17% | 10.20% / 27.70% | 1.10% / 3.00% | 3.46% / 9.40% | 4.68% / 12.70% | **17.38% / 47.20%** |
| | D10 (Rare) | 862 | 633 | 26.57% | 8.24% / 11.22% | 1.28% / 1.74% | 3.83% / 5.21% | 9.86% / 13.43% | **50.23% / 68.40%** |
| **`scifact`** | D1 (Broad) | 989 | 300 | **69.67%** | 0.20% / 0.67% | 0.40% / 1.33% | 5.56% / 18.33% | 9.20% / 30.33% | **14.96% / 49.33%** |
| | D5 (Mid) | 270 | 97 | 64.07% | 0.00% / 0.00% | 1.11% / 3.09% | 9.63% / 26.80% | 8.15% / 22.68% | **17.04% / 47.42%** |
| | D10 (Rare) | 170 | 90 | 47.06% | 2.35% / 4.44% | 1.18% / 2.22% | 10.00% / 18.89% | 7.65% / 14.44% | **31.76% / 60.00%** |
| **`bright_aops`**| D1 (Broad) | 1,496 | 45 | **96.99%** | 0.07% / 2.22% | 0.00% / 0.00% | 0.47% / 15.56% | 0.40% / 13.33% | **2.07% / 68.89%** |
| | D5 (Mid) | 438 | 21 | 95.21% | 0.00% / 0.00% | 0.00% / 0.00% | 0.91% / 19.05% | 0.23% / 4.76% | **3.65% / 76.19%** |
| | D10 (Rare) | 305 | 27 | 91.15% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.66% / 7.41% | **8.20% / 92.59%** |

---

### 7.2 Nuanced Interpretation: Score Deficit vs. Inverse Specificity

The empirical data reveal two critical insights:

1. **Broad Terms Do Not Uniformly Prefer Low Weights:**
   - On TREC-COVID D1, only 7.6% of safe instances prefer $\mu=0.05$ and 14.6% prefer $\mu=0.10$. Over **64.3%** of safe instances prefer $\mu \ge 0.30$.
   - On NFCorpus and SciFact D1, $\mu=1.00$ remains the plurality safe weight (43.7% and 49.3% of safe instances).
   - What distinguishes broad terms is not that they require $\mu \le 0.10$, but that **they have widely dispersed safe weights and frequently have no safe weight at all** ($70\% - 97\%$ no safe weight). When a broad term happens to be safe, it often still requires moderate-to-high mass to elevate the document.

2. **Rare Terms Heavily Favor Maximum Injection ($\mu=1.00$):**
   - In NFCorpus D10, conditional on safety, **68.40%** of candidate instances strictly require $\mu=1.00$.
   - In TREC-COVID D10, **51.26%** require $\mu=1.00$.
   - In SciFact D10, **60.00%** require $\mu=1.00$.
   - In BRIGHT-AOPS D10, **92.59%** require $\mu=1.00$.

3. **Theoretical Resolution: The Score-Deficit Model:**
   The naive inverse-specificity hypothesis $\mu(t) \propto \frac{A(q)}{I(t)}$ assumes that high-IDF terms need smaller weights because their marginal impact is high. The data decisively reject this simplification.
   A defensible formulation must account for the required intervention:
   $$\mu(q, t) = c(q, t) \cdot \frac{\Delta_{\mathrm{needed}}(q, t)}{I(t) + \epsilon}$$
   where:
   - $c(q, t)$ is contextual confidence that the term expresses the intended query concept;
   - $\Delta_{\mathrm{needed}}(q, t)$ is the score deficit between the relevant target document and the top-10 decision boundary;
   - $I(t)$ is the estimated marginal BM25 contribution per unit weight.
   
   > [!IMPORTANT]
   > **Core Mathematical Distinction:**
   > IDF estimates how strongly a term can act per unit weight ($I(t)$), but it does not dictate how much intervention is needed ($\Delta_{\mathrm{needed}}$). A rare diagnostic term may have high IDF, but if the relevant document begins far down the retrieval ranking, it still requires a large weight ($\mu = 0.50 - 1.00$) to generate enough total score mass to cross into the top 10. Conversely, broad terms frequently dilute ranking unless their score deficit is small.

---

## 8. Summary of Status & Stage 2 Transition

1. **Stage 1 Questions Fully Resolved:**
   - **Existence:** Latent unigram expansion headroom is confirmed across all four corpora (SciFact $+0.26$, AOPS $+0.09$, NFCorpus $+0.43$, TREC-COVID $+0.28$).
   - **Pool Design:** `DF>=2, CF>=3` is a defensible principal eligibility rule. $DF=1$ terms are dominated and safely excluded from the main pool.
   - **Capacity:** Fixed 10k is a strong provisional default retaining $>97\%$ headroom. Sizing via the evaluated adaptive rule $B = \min(10000, \max(2500, \lfloor 0.15 |V_{\text{elig}}| \rfloor))$ retains $90.5\% - 93.1\%$ macro headroom while preventing small-corpus starvation.
   - **Weighting:** Expansion weight cannot be determined from IDF alone; it requires modeling contextual confidence $c(q,t)$ and score deficit $\Delta_{\mathrm{needed}}$.

2. **The Hard Research Problem Transferred to Stage 2:**
   - Stage 1 proves that the required lexical unigrams exist inside a 10k deployable pool.
   - However, having a $+0.4$ oracle ceiling does not guarantee that a query-time model can find those terms without pulling in thousands of harmful distractors.
   - **Stage 2 Direction:** Design and evaluate the contextual selector (CRVE / BGE dense gating) to discover beneficial pool terms, calibrate expansion weights, and maintain high abstention precision.
