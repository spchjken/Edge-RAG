# Comprehensive Schema 6a & 6b Ablation & Diagnostics Report

Full observability benchmark evaluating **Query Centrality Scoring**, **Stem Deduplication**, **Fix B Validated Entity Boost**, **Multi-Level RAG Metrics**, and **Granular Query Traces** across the 3 largest stress benchmark corpora.

## Dataset: `fused_stress_500`

### 1. Multi-Level Retrieval Metrics (Strict Recall, Chunk Recall, Precision, MRR)

| Configuration | Parameters | Strict@10 | Strict@20 | Strict@50 | ChunkRec@10 | ChunkRec@50 | ChunkPrec@10 | MRR@10 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Schema 1 Baseline (3,3,3,-1)** | `(3, 3, 3, -1)` | **79.8%** | 91.6% | 96.2% | 66.6% | 91.3% | 11.6% | **0.456** | 150.6ms |
| **Schema 5b (-, 3, 5, -1)** | `(-, 3, 5, -1)` | **84.5%** | 92.3% | 95.8% | 70.9% | 91.1% | 12.4% | **0.528** | 210.1ms |
| **Schema 5b (-, 4, 5, -1)** | `(-, 4, 5, -1)` | **84.9%** | 92.5% | 95.8% | 71.5% | 91.1% | 12.5% | **0.537** | 223.7ms |
| **Schema 6a (3, 2, 5, -1)** 🌟 | `(3, 2, 5, -1)` | **72.9%** | 89.8% | 95.6% | 59.7% | 90.2% | 10.3% | **0.404** | 196.6ms |
| **Schema 6a (4, 2, 5, -1)** 🌟 | `(4, 2, 5, -1)` | **83.2%** | 92.5% | 96.0% | 69.6% | 91.1% | 12.2% | **0.513** | 206.6ms |
| **Schema 6b (-, 2, 4, -1)** 🌟 | `(-, 2, 4, -1)` | **82.6%** | 92.2% | 96.0% | 69.9% | 91.2% | 12.3% | **0.500** | 185.4ms |
| **Schema 6b (-, 3, 5, -1)** 🌟 | `(-, 3, 5, -1)` | **84.8%** | 93.0% | 96.1% | 71.7% | 91.5% | 12.6% | **0.536** | 220.1ms |
| **Schema 6b (-, 4, 5, -1)** 🌟 | `(-, 4, 5, -1)` | **85.5%** | 92.9% | 95.9% | 72.1% | 91.3% | 12.7% | **0.548** | 226.6ms |
| **Schema 6b (-, 2, 5, -1)** 🌟 | `(-, 2, 5, -1)` | **85.1%** | 92.6% | 96.0% | 71.9% | 91.1% | 12.6% | **0.535** | 204.1ms |

### 2. Query Expansion Internal Telemetry Dashboard (Exposing Bottlenecks)

| Configuration | Avg Anchors | Avg Cands $\ge \tau_{\text{sim}}$ | Avg Synonyms Injected | Starvation Rate (%) | Avg $Q_{\text{aug}}$ Length | Avg Anchor Rep ($R$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `v1_baseline` | 6.80 | 340.69 | **13.59** | **0.1%** | 38.9 tokens | 3.00x |
| `v5b_r3-5_c-1` | 6.80 | 340.69 | **23.17** | **0.2%** | 60.0 tokens | 4.43x |
| `v5b_r4-5_c-1` | 6.80 | 340.69 | **25.41** | **0.2%** | 64.8 tokens | 4.76x |
| `v6a_n3_r2-5_c-1` | 6.23 | 332.37 | **19.66** | **0.2%** | 44.5 tokens | 3.00x |
| `v6a_n4_r2-5_c-1` | 6.23 | 332.37 | **19.66** | **0.2%** | 50.7 tokens | 4.00x |
| `v6b_r2-4_c-1` | 6.23 | 332.37 | **15.25** | **0.2%** | 42.1 tokens | 3.46x |
| `v6b_r3-5_c-1` | 6.23 | 332.37 | **21.47** | **0.2%** | 55.7 tokens | 4.46x |
| `v6b_r4-5_c-1` | 6.23 | 332.37 | **23.52** | **0.2%** | 60.1 tokens | 4.79x |
| `v6b_r2-5_c-1` | 6.23 | 332.37 | **19.66** | **0.2%** | 51.7 tokens | 4.18x |

### 3. Qualitative Query Expansion Case Studies

#### Case 1: Query `q_8968b27d`
- **Raw Question:** *"How many test tasks are included in the EHR-Complex benchmark?"*
- **Gold Chunk IDs:** `['EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk1', 'EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk0', 'EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk3', 'EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk2']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 18** | 0% | `EHR EHR EHR erbb2 esr1 entry-filtered sca EHR-Complex EHR-Complex EHR-Complex entry-filter...` |
| `v5b_r3-5_c-1` | **Rank 12** | 0% | `EHR EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr EHR-Complex EHR-Complex EHR...` |
| `v5b_r4-5_c-1` | **Rank 11** | 0% | `EHR EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr EHR-Complex EHR-Complex EHR...` |
| `v6a_n3_r2-5_c-1` | **Rank 24** | 0% | `EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr EHR-Complex EHR-Complex EHR-Complex...` |
| `v6a_n4_r2-5_c-1` | **Rank 23** | 0% | `EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr EHR-Complex EHR-Complex EHR-Com...` |
| `v6b_r2-4_c-1` | **Rank 23** | 0% | `EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata EHR-Complex EHR-Complex EHR-Complex...` |
| `v6b_r3-5_c-1` | **Rank 18** | 0% | `EHR EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr EHR-Complex EHR-Complex EHR...` |
| `v6b_r4-5_c-1` | **Rank 16** | 0% | `EHR EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr EHR-Complex EHR-Complex EHR...` |
| `v6b_r2-5_c-1` | **Rank 18** | 0% | `EHR EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr EHR-Complex EHR-Complex EHR...` |

---

#### Case 2: Query `q_43cc4795`
- **Raw Question:** *"What challenge do population-level queries pose compared to patient-level queries in EHR reasoning?"*
- **Gold Chunk IDs:** `['EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk1']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 10** | 100% | `EHR EHR EHR erbb2 esr1 entry-filtered sca patient-level patient-level patient-level patien...` |
| `v5b_r3-5_c-1` | **Rank 10** | 100% | `EHR EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr patient-level patient-level...` |
| `v5b_r4-5_c-1` | **Rank 10** | 100% | `EHR EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr patient-level patient-level...` |
| `v6a_n3_r2-5_c-1` | **Rank 14** | 0% | `EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr patient-level patient-level patient...` |
| `v6a_n4_r2-5_c-1` | **Rank 7** | 100% | `EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr patient-level patient-level pat...` |
| `v6b_r2-4_c-1` | **Rank 7** | 100% | `EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata patient-level patient-level patient...` |
| `v6b_r3-5_c-1` | **Rank 7** | 100% | `EHR EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr patient-level patient-level...` |
| `v6b_r4-5_c-1` | **Rank 7** | 100% | `EHR EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr patient-level patient-level...` |
| `v6b_r2-5_c-1` | **Rank 7** | 100% | `EHR EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr patient-level patient-level...` |

---

#### Case 3: Query `q_e1f6a4fe`
- **Raw Question:** *"How do the dominant failure categories compare between Kimi-K2.5 and GPT-4.1 on EHR-Complex?"*
- **Gold Chunk IDs:** `['EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk2']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 15** | 0% | `EHR-Complex EHR-Complex EHR-Complex erbb2 esr1 entry-filtered sca GPT GPT GPT gata3 mki67 ...` |
| `v5b_r3-5_c-1` | **Rank 15** | 0% | `EHR-Complex EHR-Complex EHR-Complex EHR-Complex EHR-Complex erbb2 esr1 entry-filtered sca ...` |
| `v5b_r4-5_c-1` | **Rank 15** | 0% | `EHR-Complex EHR-Complex EHR-Complex EHR-Complex EHR-Complex erbb2 esr1 entry-filtered sca ...` |
| `v6a_n3_r2-5_c-1` | **Rank 17** | 0% | `EHR-Complex EHR-Complex EHR-Complex erbb2 esr1 entry-filtered sca gata3 mki67 standardized...` |
| `v6a_n4_r2-5_c-1` | **Rank 15** | 0% | `EHR-Complex EHR-Complex EHR-Complex EHR-Complex erbb2 esr1 entry-filtered sca gata3 mki67 ...` |
| `v6b_r2-4_c-1` | **Rank 14** | 0% | `EHR-Complex EHR-Complex EHR-Complex EHR-Complex erbb2 esr1 entry-filtered sca gata3 mki67 ...` |
| `v6b_r3-5_c-1` | **Rank 15** | 0% | `EHR-Complex EHR-Complex EHR-Complex EHR-Complex EHR-Complex erbb2 esr1 entry-filtered sca ...` |
| `v6b_r4-5_c-1` | **Rank 15** | 0% | `EHR-Complex EHR-Complex EHR-Complex EHR-Complex EHR-Complex erbb2 esr1 entry-filtered sca ...` |
| `v6b_r2-5_c-1` | **Rank 15** | 0% | `EHR-Complex EHR-Complex EHR-Complex EHR-Complex EHR-Complex erbb2 esr1 entry-filtered sca ...` |

---

#### Case 4: Query `q_a4117119`
- **Raw Question:** *"I want to boost the performance of a small open-weight model like Qwen3-32B for interactive EHR queries. What training approach does the EHR-Complex paper show to be effective?"*
- **Gold Chunk IDs:** `['EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk1', 'EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk0', 'EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk2', 'EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk3']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 13** | 0% | `Qwen3-32B Qwen3-32B Qwen3-32B gata3 mki67 mbagcn reverse-gnn EHR EHR EHR erbb2 esr1 entry-...` |
| `v5b_r3-5_c-1` | **Rank 15** | 0% | `Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B gata3 mki67 mbagcn reverse-gnn dropedge ...` |
| `v5b_r4-5_c-1` | **Rank 12** | 0% | `Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B gata3 mki67 mbagcn reverse-gnn dropedge ...` |
| `v6a_n3_r2-5_c-1` | **Rank 20** | 0% | `Qwen3-32B Qwen3-32B Qwen3-32B gata3 mki67 mbagcn reverse-gnn dropedge mbagcn foxa1 gata3 E...` |
| `v6a_n4_r2-5_c-1` | **Rank 12** | 0% | `Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B gata3 mki67 mbagcn reverse-gnn dropedge mbagcn fox...` |
| `v6b_r2-4_c-1` | **Rank 9** | 25% | `Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B gata3 mki67 mbagcn reverse-gnn dropedge mbagcn EHR...` |
| `v6b_r3-5_c-1` | **Rank 9** | 25% | `Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B gata3 mki67 mbagcn reverse-gnn dropedge ...` |
| `v6b_r4-5_c-1` | **Rank 8** | 25% | `Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B gata3 mki67 mbagcn reverse-gnn dropedge ...` |
| `v6b_r2-5_c-1` | **Rank 8** | 25% | `Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B gata3 mki67 mbagcn reverse-gnn dropedge ...` |

---

#### Case 5: Query `q_c7835beb`
- **Raw Question:** *"What is the average number of SQL structural components per query in the EHR-Complex benchmark?"*
- **Gold Chunk IDs:** `['EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block0_chunk0', 'EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block0_chunk1']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 5** | 100% | `SQL SQL SQL database entry-filtered sca EHR EHR EHR erbb2 esr1 entry-filtered sca EHR-Comp...` |
| `v5b_r3-5_c-1` | **Rank 3** | 100% | `SQL SQL SQL SQL SQL database entry-filtered sca mbagcn reverse-gnn erbb2 esr1 EHR EHR EHR ...` |
| `v5b_r4-5_c-1` | **Rank 3** | 100% | `SQL SQL SQL SQL SQL database entry-filtered sca mbagcn reverse-gnn erbb2 esr1 EHR EHR EHR ...` |
| `v6a_n3_r2-5_c-1` | **Rank 3** | 100% | `SQL SQL SQL database entry-filtered sca mbagcn reverse-gnn erbb2 esr1 EHR EHR EHR erbb2 es...` |
| `v6a_n4_r2-5_c-1` | **Rank 3** | 100% | `SQL SQL SQL SQL database entry-filtered sca mbagcn reverse-gnn erbb2 esr1 EHR EHR EHR EHR ...` |
| `v6b_r2-4_c-1` | **Rank 3** | 100% | `SQL SQL SQL SQL database entry-filtered sca mbagcn reverse-gnn EHR EHR EHR EHR erbb2 esr1 ...` |
| `v6b_r3-5_c-1` | **Rank 3** | 100% | `SQL SQL SQL SQL SQL database entry-filtered sca mbagcn reverse-gnn erbb2 esr1 EHR EHR EHR ...` |
| `v6b_r4-5_c-1` | **Rank 3** | 100% | `SQL SQL SQL SQL SQL database entry-filtered sca mbagcn reverse-gnn erbb2 esr1 EHR EHR EHR ...` |
| `v6b_r2-5_c-1` | **Rank 3** | 100% | `SQL SQL SQL SQL SQL database entry-filtered sca mbagcn reverse-gnn erbb2 esr1 EHR EHR EHR ...` |

---

## Dataset: `enterpriserag_stress_1000`

### 1. Multi-Level Retrieval Metrics (Strict Recall, Chunk Recall, Precision, MRR)

| Configuration | Parameters | Strict@10 | Strict@20 | Strict@50 | ChunkRec@10 | ChunkRec@50 | ChunkPrec@10 | MRR@10 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Schema 1 Baseline (3,3,3,-1)** | `(3, 3, 3, -1)` | **82.0%** | 85.6% | 89.8% | 58.3% | 75.7% | 15.7% | **0.555** | 66.6ms |
| **Schema 5b (-, 3, 5, -1)** | `(-, 3, 5, -1)` | **81.6%** | 85.0% | 91.0% | 58.9% | 76.8% | 15.9% | **0.592** | 80.6ms |
| **Schema 5b (-, 4, 5, -1)** | `(-, 4, 5, -1)` | **81.8%** | 85.4% | 90.4% | 59.0% | 75.9% | 16.1% | **0.599** | 84.5ms |
| **Schema 6a (3, 2, 5, -1)** 🌟 | `(3, 2, 5, -1)` | **79.4%** | 85.2% | 89.4% | 56.7% | 75.0% | 15.0% | **0.514** | 98.0ms |
| **Schema 6a (4, 2, 5, -1)** 🌟 | `(4, 2, 5, -1)` | **79.8%** | 85.8% | 89.4% | 57.8% | 75.5% | 15.6% | **0.576** | 101.7ms |
| **Schema 6b (-, 2, 4, -1)** 🌟 | `(-, 2, 4, -1)` | **80.4%** | 84.6% | 88.6% | 58.2% | 74.6% | 15.6% | **0.577** | 95.4ms |
| **Schema 6b (-, 3, 5, -1)** 🌟 | `(-, 3, 5, -1)` | **80.2%** | 85.0% | 89.6% | 57.8% | 75.2% | 15.6% | **0.595** | 105.6ms |
| **Schema 6b (-, 4, 5, -1)** 🌟 | `(-, 4, 5, -1)` | **80.4%** | 86.0% | 90.0% | 58.2% | 75.8% | 15.9% | **0.596** | 109.1ms |
| **Schema 6b (-, 2, 5, -1)** 🌟 | `(-, 2, 5, -1)` | **80.8%** | 85.0% | 89.8% | 58.5% | 75.0% | 15.8% | **0.585** | 102.1ms |

### 2. Query Expansion Internal Telemetry Dashboard (Exposing Bottlenecks)

| Configuration | Avg Anchors | Avg Cands $\ge \tau_{\text{sim}}$ | Avg Synonyms Injected | Starvation Rate (%) | Avg $Q_{\text{aug}}$ Length | Avg Anchor Rep ($R$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `v1_baseline` | 11.73 | 339.00 | **23.46** | **0.0%** | 61.6 tokens | 3.00x |
| `v5b_r3-5_c-1` | 11.73 | 339.00 | **39.22** | **0.0%** | 94.1 tokens | 4.35x |
| `v5b_r4-5_c-1` | 11.73 | 339.00 | **43.69** | **0.0%** | 103.3 tokens | 4.74x |
| `v6a_n3_r2-5_c-1` | 11.00 | 333.32 | **33.46** | **0.0%** | 69.9 tokens | 3.00x |
| `v6a_n4_r2-5_c-1` | 11.00 | 333.32 | **33.46** | **0.0%** | 80.9 tokens | 4.00x |
| `v6b_r2-4_c-1` | 11.00 | 333.32 | **25.95** | **0.0%** | 66.0 tokens | 3.37x |
| `v6b_r3-5_c-1` | 11.00 | 333.32 | **36.95** | **0.0%** | 88.7 tokens | 4.37x |
| `v6b_r4-5_c-1` | 11.00 | 333.32 | **41.22** | **0.0%** | 97.4 tokens | 4.75x |
| `v6b_r2-5_c-1` | 11.00 | 333.32 | **33.46** | **0.0%** | 81.5 tokens | 4.06x |

### 3. Qualitative Query Expansion Case Studies

#### Case 1: Query `qst_0001`
- **Raw Question:** *"What are the default size limits for file uploads and total request size for the new multipart upload support on the OpenAI-compatible API endpoints?"*
- **Gold Chunk IDs:** `['dsid_ae068ee4aa9640159427cd941bef0238_block0_chunk1', 'dsid_ae068ee4aa9640159427cd941bef0238_block0_chunk0']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 1** | 50% | `OpenAI-compatible OpenAI-compatible OpenAI-compatible kubernetes multi-region API API API ...` |
| `v5b_r3-5_c-1` | **Rank 1** | 50% | `OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible ...` |
| `v5b_r4-5_c-1` | **Rank 1** | 50% | `OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible ...` |
| `v6a_n3_r2-5_c-1` | **Rank 1** | 50% | `OpenAI-compatible OpenAI-compatible OpenAI-compatible kubernetes multi-region versioned hs...` |
| `v6a_n4_r2-5_c-1` | **Rank 1** | 50% | `OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible kubernetes multi-r...` |
| `v6b_r2-4_c-1` | **Rank 1** | 50% | `OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible kubernetes multi-r...` |
| `v6b_r3-5_c-1` | **Rank 1** | 50% | `OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible ...` |
| `v6b_r4-5_c-1` | **Rank 1** | 50% | `OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible ...` |
| `v6b_r2-5_c-1` | **Rank 1** | 50% | `OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible ...` |

---

#### Case 2: Query `qst_0002`
- **Raw Question:** *"What is the name of the new metric added so SRE can track when server-side streaming sessions get finalized due to hitting the time limit?"*
- **Gold Chunk IDs:** `['dsid_9550250a59e74f1bbd5612480b2e7100_block0_chunk0']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 2** | 100% | `SRE SRE SRE sse real-time server-side server-side server-side --org redwood-demo add-on fi...` |
| `v5b_r3-5_c-1` | **Rank 2** | 100% | `SRE SRE SRE SRE SRE sse real-time rps precision server-side server-side server-side server...` |
| `v5b_r4-5_c-1` | **Rank 2** | 100% | `SRE SRE SRE SRE SRE sse real-time rps precision server-side server-side server-side server...` |
| `v6a_n3_r2-5_c-1` | **Rank 5** | 100% | `server-side server-side server-side --org redwood-demo add-on kubernetes real-time finaliz...` |
| `v6a_n4_r2-5_c-1` | **Rank 3** | 100% | `server-side server-side server-side server-side --org redwood-demo add-on kubernetes real-...` |
| `v6b_r2-4_c-1` | **Rank 2** | 100% | `server-side server-side server-side server-side --org redwood-demo add-on kubernetes final...` |
| `v6b_r3-5_c-1` | **Rank 2** | 100% | `server-side server-side server-side server-side server-side --org redwood-demo add-on kube...` |
| `v6b_r4-5_c-1` | **Rank 2** | 100% | `server-side server-side server-side server-side server-side --org redwood-demo add-on kube...` |
| `v6b_r2-5_c-1` | **Rank 2** | 100% | `server-side server-side server-side server-side server-side --org redwood-demo add-on kube...` |

---

#### Case 3: Query `qst_0003`
- **Raw Question:** *"What are the acceptance criteria for the project introducing an algorithm to generate interactive UI color states and a Kappa-style elevation scale for dense table and grid components?"*
- **Gold Chunk IDs:** `['dsid_3fd6af404fae48e6b8ea5a57875ef78f_block0_chunk0']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 6** | 100% | `Kappa-style Kappa-style Kappa-style layout --org redwood-demo UI UI UI layout component gr...` |
| `v5b_r3-5_c-1` | **Rank 5** | 100% | `Kappa-style Kappa-style Kappa-style Kappa-style Kappa-style layout --org redwood-demo desi...` |
| `v5b_r4-5_c-1` | **Rank 2** | 100% | `Kappa-style Kappa-style Kappa-style Kappa-style Kappa-style layout --org redwood-demo desi...` |
| `v6a_n3_r2-5_c-1` | **Rank 7** | 100% | `Kappa-style Kappa-style Kappa-style layout --org redwood-demo designed component UI UI UI ...` |
| `v6a_n4_r2-5_c-1` | **Rank 6** | 100% | `Kappa-style Kappa-style Kappa-style Kappa-style layout --org redwood-demo designed compone...` |
| `v6b_r2-4_c-1` | **Rank 6** | 100% | `Kappa-style Kappa-style Kappa-style Kappa-style layout --org redwood-demo designed UI UI U...` |
| `v6b_r3-5_c-1` | **Rank 6** | 100% | `Kappa-style Kappa-style Kappa-style Kappa-style Kappa-style layout --org redwood-demo desi...` |
| `v6b_r4-5_c-1` | **Rank 5** | 100% | `Kappa-style Kappa-style Kappa-style Kappa-style Kappa-style layout --org redwood-demo desi...` |
| `v6b_r2-5_c-1` | **Rank 6** | 100% | `Kappa-style Kappa-style Kappa-style Kappa-style Kappa-style layout --org redwood-demo desi...` |

---

#### Case 4: Query `qst_0004`
- **Raw Question:** *"In the meeting about onboarding a SaaS product to Google Cloud Marketplace, what did the GCP team recommend for handling delays where a new subscription entitlement is not immediately available during the customer onboarding flow?"*
- **Gold Chunk IDs:** `['dsid_6c4c1c875e704f09b4d791d64d7bc7e5_block0_chunk3', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_block0_chunk2', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_block1_chunk3', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_block1_chunk2', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_block1_chunk0', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_bridge0-1', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_bridge1-2', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_block1_chunk1', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_block0_chunk0', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_block0_chunk1', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_block2_chunk0']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 3** | 18% | `GCP GCP GCP --org redwood-demo stakeholders google google google --org redwood-demo on-dem...` |
| `v5b_r3-5_c-1` | **Rank 1** | 36% | `GCP GCP GCP GCP GCP --org redwood-demo stakeholders rps roadmap google google google googl...` |
| `v5b_r4-5_c-1` | **Rank 1** | 36% | `GCP GCP GCP GCP GCP --org redwood-demo stakeholders rps roadmap google google google googl...` |
| `v6a_n3_r2-5_c-1` | **Rank 4** | 27% | `GCP GCP GCP --org redwood-demo stakeholders rps roadmap saas saas saas kubernetes slas on-...` |
| `v6a_n4_r2-5_c-1` | **Rank 1** | 27% | `GCP GCP GCP GCP --org redwood-demo stakeholders rps roadmap saas saas saas saas kubernetes...` |
| `v6b_r2-4_c-1` | **Rank 1** | 27% | `GCP GCP GCP GCP --org redwood-demo stakeholders rps saas saas saas saas kubernetes slas on...` |
| `v6b_r3-5_c-1` | **Rank 1** | 27% | `GCP GCP GCP GCP GCP --org redwood-demo stakeholders rps roadmap saas saas saas saas saas k...` |
| `v6b_r4-5_c-1` | **Rank 1** | 36% | `GCP GCP GCP GCP GCP --org redwood-demo stakeholders rps roadmap saas saas saas saas saas k...` |
| `v6b_r2-5_c-1` | **Rank 1** | 36% | `GCP GCP GCP GCP GCP --org redwood-demo stakeholders rps roadmap saas saas saas saas saas k...` |

---

#### Case 5: Query `qst_0005`
- **Raw Question:** *"What failover sequence and recovery targets did MedThink specify for handling an EU region outage, including any limits on how long traffic can shift to the US?"*
- **Gold Chunk IDs:** `['dsid_8e838ab6a98f4cbcb672d41f210ff89c_block0_chunk0', 'dsid_8e838ab6a98f4cbcb672d41f210ff89c_block0_chunk1', 'dsid_8e838ab6a98f4cbcb672d41f210ff89c_block0_chunk2']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 1** | 100% | `US US US us-west us-west-2 EU EU EU eu-west per-region medthink medthink medthink thinking...` |
| `v5b_r3-5_c-1` | **Rank 1** | 100% | `US US US US US us-west us-west-2 overage states EU EU EU EU EU eu-west per-region multi-re...` |
| `v5b_r4-5_c-1` | **Rank 1** | 100% | `US US US US US us-west us-west-2 overage states EU EU EU EU EU eu-west per-region multi-re...` |
| `v6a_n3_r2-5_c-1` | **Rank 1** | 100% | `US US US us-west us-west-2 overage states EU EU EU eu-west per-region multi-region migrati...` |
| `v6a_n4_r2-5_c-1` | **Rank 1** | 100% | `US US US US us-west us-west-2 overage states EU EU EU EU eu-west per-region multi-region m...` |
| `v6b_r2-4_c-1` | **Rank 1** | 100% | `US US US US us-west us-west-2 overage EU EU EU EU eu-west per-region multi-region medthink...` |
| `v6b_r3-5_c-1` | **Rank 1** | 100% | `US US US US US us-west us-west-2 overage states EU EU EU EU EU eu-west per-region multi-re...` |
| `v6b_r4-5_c-1` | **Rank 1** | 100% | `US US US US US us-west us-west-2 overage states EU EU EU EU EU eu-west per-region multi-re...` |
| `v6b_r2-5_c-1` | **Rank 1** | 100% | `US US US US US us-west us-west-2 overage states EU EU EU EU EU eu-west per-region multi-re...` |

---

## Dataset: `liverag_stress_full`

### 1. Multi-Level Retrieval Metrics (Strict Recall, Chunk Recall, Precision, MRR)

| Configuration | Parameters | Strict@10 | Strict@20 | Strict@50 | ChunkRec@10 | ChunkRec@50 | ChunkPrec@10 | MRR@10 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Schema 1 Baseline (3,3,3,-1)** | `(3, 3, 3, -1)` | **93.1%** | 95.0% | 97.2% | 78.9% | 87.4% | 18.4% | **0.800** | 37.8ms |
| **Schema 5b (-, 3, 5, -1)** | `(-, 3, 5, -1)` | **93.3%** | 95.8% | 97.1% | 79.6% | 87.5% | 18.7% | **0.834** | 43.1ms |
| **Schema 5b (-, 4, 5, -1)** | `(-, 4, 5, -1)` | **93.2%** | 95.3% | 97.0% | 79.1% | 87.5% | 18.5% | **0.832** | 44.2ms |
| **Schema 6a (3, 2, 5, -1)** 🌟 | `(3, 2, 5, -1)` | **92.6%** | 95.5% | 97.3% | 77.3% | 87.4% | 18.0% | **0.768** | 67.0ms |
| **Schema 6a (4, 2, 5, -1)** 🌟 | `(4, 2, 5, -1)` | **93.6%** | 96.1% | 97.4% | 78.9% | 87.6% | 18.4% | **0.819** | 68.3ms |
| **Schema 6b (-, 2, 4, -1)** 🌟 | `(-, 2, 4, -1)` | **93.4%** | 96.0% | 97.4% | 79.2% | 87.7% | 18.6% | **0.825** | 66.0ms |
| **Schema 6b (-, 3, 5, -1)** 🌟 | `(-, 3, 5, -1)` | **94.0%** | 96.1% | 97.4% | 80.1% | 87.7% | 18.7% | **0.836** | 69.4ms |
| **Schema 6b (-, 4, 5, -1)** 🌟 | `(-, 4, 5, -1)` | **93.7%** | 95.8% | 97.3% | 79.4% | 87.8% | 18.6% | **0.840** | 70.6ms |
| **Schema 6b (-, 2, 5, -1)** 🌟 | `(-, 2, 5, -1)` | **93.5%** | 96.0% | 97.4% | 79.7% | 87.8% | 18.7% | **0.833** | 68.6ms |

### 2. Query Expansion Internal Telemetry Dashboard (Exposing Bottlenecks)

| Configuration | Avg Anchors | Avg Cands $\ge \tau_{\text{sim}}$ | Avg Synonyms Injected | Starvation Rate (%) | Avg $Q_{\text{aug}}$ Length | Avg Anchor Rep ($R$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `v1_baseline` | 5.59 | 158.57 | **11.03** | **1.7%** | 31.1 tokens | 3.00x |
| `v5b_r3-5_c-1` | 5.59 | 158.57 | **19.15** | **2.8%** | 49.2 tokens | 4.54x |
| `v5b_r4-5_c-1` | 5.59 | 158.57 | **21.38** | **2.9%** | 54.0 tokens | 4.92x |
| `v6a_n3_r2-5_c-1` | 5.51 | 152.22 | **17.77** | **2.9%** | 39.0 tokens | 3.00x |
| `v6a_n4_r2-5_c-1` | 5.51 | 152.22 | **17.77** | **2.9%** | 44.5 tokens | 4.00x |
| `v6b_r2-4_c-1` | 5.51 | 152.22 | **13.49** | **2.3%** | 36.7 tokens | 3.54x |
| `v6b_r3-5_c-1` | 5.51 | 152.22 | **18.84** | **2.9%** | 48.5 tokens | 4.54x |
| `v6b_r4-5_c-1` | 5.51 | 152.22 | **20.99** | **3.0%** | 53.2 tokens | 4.91x |
| `v6b_r2-5_c-1` | 5.51 | 152.22 | **17.77** | **2.9%** | 46.2 tokens | 4.35x |

### 3. Qualitative Query Expansion Case Studies

#### Case 1: Query `q_live_0`
- **Raw Question:** *"How deep can fish survive in the ocean trenches?"*
- **Gold Chunk IDs:** `['<urn:uuid:a102a6cb-a608-493c-928f-d32a0da4dbf6>_block0_chunk1', '<urn:uuid:a102a6cb-a608-493c-928f-d32a0da4dbf6>_block0_chunk0']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 4** | 50% | `trenches trenches trenches armored division sand-fired pot survive survive survive lived h...` |
| `v5b_r3-5_c-1` | **Rank 3** | 100% | `trenches trenches trenches trenches trenches armored division sand-fired pot hydraulic fra...` |
| `v5b_r4-5_c-1` | **Rank 4** | 100% | `trenches trenches trenches trenches trenches armored division sand-fired pot hydraulic fra...` |
| `v6a_n3_r2-5_c-1` | **Rank 2** | 50% | `trenches trenches trenches armored division sand-fired pot hydraulic fracturing beneath su...` |
| `v6a_n4_r2-5_c-1` | **Rank 2** | 50% | `trenches trenches trenches trenches armored division sand-fired pot hydraulic fracturing b...` |
| `v6b_r2-4_c-1` | **Rank 2** | 50% | `trenches trenches trenches trenches armored division sand-fired pot hydraulic fracturing s...` |
| `v6b_r3-5_c-1` | **Rank 2** | 50% | `trenches trenches trenches trenches trenches armored division sand-fired pot hydraulic fra...` |
| `v6b_r4-5_c-1` | **Rank 2** | 50% | `trenches trenches trenches trenches trenches armored division sand-fired pot hydraulic fra...` |
| `v6b_r2-5_c-1` | **Rank 2** | 50% | `trenches trenches trenches trenches trenches armored division sand-fired pot hydraulic fra...` |

---

#### Case 2: Query `q_live_1`
- **Raw Question:** *"Based on temperature considerations alone, is March considered a suitable month to perform the final pruning of grape vines?"*
- **Gold Chunk IDs:** `['<urn:uuid:b5d19fcb-1711-4f9f-82cf-f81403382444>_block0_chunk1', '<urn:uuid:b5d19fcb-1711-4f9f-82cf-f81403382444>_block0_chunk0']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 25** | 0% | `grape grape grape peacock mantis mantis shrimp vines vines vines peacock mantis lab-grown ...` |
| `v5b_r3-5_c-1` | **Rank 19** | 0% | `grape grape grape grape grape peacock mantis mantis shrimp fruit valley vines vines vines ...` |
| `v5b_r4-5_c-1` | **Rank 33** | 0% | `grape grape grape grape grape peacock mantis mantis shrimp fruit valley vines vines vines ...` |
| `v6a_n3_r2-5_c-1` | **Rank 11** | 0% | `vines vines vines peacock mantis lab-grown diamonds fruits varieties grape grape grape pea...` |
| `v6a_n4_r2-5_c-1` | **Rank 12** | 0% | `vines vines vines vines peacock mantis lab-grown diamonds fruits varieties grape grape gra...` |
| `v6b_r2-4_c-1` | **Rank 10** | 50% | `vines vines vines vines peacock mantis lab-grown diamonds fruits grape grape grape grape p...` |
| `v6b_r3-5_c-1` | **Rank 9** | 50% | `vines vines vines vines vines peacock mantis lab-grown diamonds fruits varieties grape gra...` |
| `v6b_r4-5_c-1` | **Rank 11** | 0% | `vines vines vines vines vines peacock mantis lab-grown diamonds fruits varieties grape gra...` |
| `v6b_r2-5_c-1` | **Rank 12** | 0% | `vines vines vines vines vines peacock mantis lab-grown diamonds fruits varieties grape gra...` |

---

#### Case 3: Query `q_live_2`
- **Raw Question:** *"What major acts performed at the Brighton Hippodrome during its peak years?"*
- **Gold Chunk IDs:** `['<urn:uuid:95479dfb-3efd-4235-9bb8-4bfb98caab4f>_block0_chunk0']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 1** | 100% | `hippodrome hippodrome hippodrome pelvic girdle vertical brighton brighton brighton london ...` |
| `v5b_r3-5_c-1` | **Rank 1** | 100% | `hippodrome hippodrome hippodrome hippodrome hippodrome pelvic girdle vertical arm leg brig...` |
| `v5b_r4-5_c-1` | **Rank 1** | 100% | `hippodrome hippodrome hippodrome hippodrome hippodrome pelvic girdle vertical arm leg brig...` |
| `v6a_n3_r2-5_c-1` | **Rank 1** | 100% | `brighton brighton brighton london martin nearby hippodrome hippodrome hippodrome pelvic gi...` |
| `v6a_n4_r2-5_c-1` | **Rank 1** | 100% | `brighton brighton brighton brighton london martin nearby hippodrome hippodrome hippodrome ...` |
| `v6b_r2-4_c-1` | **Rank 1** | 100% | `brighton brighton brighton brighton london martin nearby hippodrome hippodrome hippodrome ...` |
| `v6b_r3-5_c-1` | **Rank 1** | 100% | `brighton brighton brighton brighton brighton london martin nearby hippodrome hippodrome hi...` |
| `v6b_r4-5_c-1` | **Rank 1** | 100% | `brighton brighton brighton brighton brighton london martin nearby hippodrome hippodrome hi...` |
| `v6b_r2-5_c-1` | **Rank 1** | 100% | `brighton brighton brighton brighton brighton london martin nearby hippodrome hippodrome hi...` |

---

#### Case 4: Query `q_live_3`
- **Raw Question:** *"I noticed some stucco houses in my neighborhood. What are the potential drawbacks and limitations of using a one-coat stucco system on exterior walls?"*
- **Gold Chunk IDs:** `['<urn:uuid:42ae24ce-dc04-4b1e-b0a8-ff18c900fae1>_block0_chunk1', '<urn:uuid:42ae24ce-dc04-4b1e-b0a8-ff18c900fae1>_block0_chunk2', '<urn:uuid:42ae24ce-dc04-4b1e-b0a8-ff18c900fae1>_block0_chunk0']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 1** | 67% | `one-coat one-coat one-coat covering layers drawbacks drawbacks drawbacks advantages risks ...` |
| `v5b_r3-5_c-1` | **Rank 1** | 67% | `one-coat one-coat one-coat one-coat one-coat covering layers designs walls drawbacks drawb...` |
| `v5b_r4-5_c-1` | **Rank 1** | 67% | `one-coat one-coat one-coat one-coat one-coat covering layers designs walls drawbacks drawb...` |
| `v6a_n3_r2-5_c-1` | **Rank 1** | 33% | `one-coat one-coat one-coat covering layers designs walls drawbacks drawbacks drawbacks adv...` |
| `v6a_n4_r2-5_c-1` | **Rank 1** | 33% | `one-coat one-coat one-coat one-coat covering layers designs walls drawbacks drawbacks draw...` |
| `v6b_r2-4_c-1` | **Rank 1** | 33% | `one-coat one-coat one-coat one-coat covering layers designs drawbacks drawbacks drawbacks ...` |
| `v6b_r3-5_c-1` | **Rank 1** | 33% | `one-coat one-coat one-coat one-coat one-coat covering layers designs walls drawbacks drawb...` |
| `v6b_r4-5_c-1` | **Rank 1** | 33% | `one-coat one-coat one-coat one-coat one-coat covering layers designs walls drawbacks drawb...` |
| `v6b_r2-5_c-1` | **Rank 1** | 33% | `one-coat one-coat one-coat one-coat one-coat covering layers designs walls drawbacks drawb...` |

---

#### Case 5: Query `q_live_4`
- **Raw Question:** *"I need help with a business case study - what format should I use for writing it up?"*
- **Gold Chunk IDs:** `['<urn:uuid:85d65922-709a-4f5b-9de7-0c2c19fe8ad3>_block0_chunk0']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 1** | 100% | `format format format utm bounding sheet writing writing writing wrote fountain pens busine...` |
| `v5b_r3-5_c-1` | **Rank 1** | 100% | `format format format format format utm bounding sheet pandas dataframe document writing wr...` |
| `v5b_r4-5_c-1` | **Rank 1** | 100% | `format format format format format utm bounding sheet pandas dataframe document writing wr...` |
| `v6a_n3_r2-5_c-1` | **Rank 2** | 100% | `format format format utm bounding sheet pandas dataframe document writing writing writing ...` |
| `v6a_n4_r2-5_c-1` | **Rank 1** | 100% | `format format format format utm bounding sheet pandas dataframe document writing writing w...` |
| `v6b_r2-4_c-1` | **Rank 2** | 100% | `format format format format utm bounding sheet pandas dataframe writing writing writing wr...` |
| `v6b_r3-5_c-1` | **Rank 1** | 100% | `format format format format format utm bounding sheet pandas dataframe document writing wr...` |
| `v6b_r4-5_c-1` | **Rank 1** | 100% | `format format format format format utm bounding sheet pandas dataframe document writing wr...` |
| `v6b_r2-5_c-1` | **Rank 1** | 100% | `format format format format format utm bounding sheet pandas dataframe document writing wr...` |

---

