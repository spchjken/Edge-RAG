# Comprehensive Schema 5a & 5b Ablation & Diagnostics Report

Full observability benchmark evaluating **Query Expansion Telemetry**, **Multi-Level RAG Metrics** (Strict Recall, Chunk Recall, Chunk Precision, MRR), and **Granular Query Traces** across the 3 largest stress benchmark corpora.

## Dataset: `fused_stress_500`

### 1. Multi-Level Retrieval Metrics (Strict Recall, Chunk Recall, Precision, MRR)

| Configuration | Parameters | Strict@10 | Strict@20 | Strict@50 | ChunkRec@10 | ChunkRec@50 | ChunkPrec@10 | MRR@10 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Schema 1 Baseline (3,3,3,-1)** 🏆 | `(3, 3, 3, -1)` | **79.8%** | 91.6% | 96.2% | 66.6% | 91.3% | 11.6% | **0.456** | 141.2ms |
| **Schema 5a (3, 2, 5, -1)** | `(3, 2, 5, -1)` | **71.4%** | 89.7% | 95.5% | 57.8% | 90.3% | 10.0% | **0.391** | 169.4ms |
| **Schema 5a (3, 2, 5, 0)** | `(3, 2, 5, 0)` | **65.2%** | 86.2% | 95.0% | 52.3% | 89.8% | 9.0% | **0.350** | 192.2ms |
| **Schema 5a (4, 2, 5, -1)** | `(4, 2, 5, -1)` | **82.6%** | 92.1% | 96.0% | 68.7% | 91.2% | 12.0% | **0.508** | 181.2ms |
| **Schema 5b (-, 2, 4, -1)** | `(-, 2, 4, -1)` | **81.7%** | 92.0% | 95.6% | 68.5% | 91.0% | 12.0% | **0.494** | 156.7ms |
| **Schema 5b (-, 3, 5, -1)** | `(-, 3, 5, -1)` | **84.5%** | 92.3% | 95.8% | 70.9% | 91.1% | 12.4% | **0.528** | 194.7ms |
| **Schema 5b (-, 4, 5, -1)** | `(-, 4, 5, -1)` | **84.9%** | 92.5% | 95.8% | 71.5% | 91.1% | 12.5% | **0.537** | 207.2ms |
| **Schema 5b (-, 2, 5, -1)** | `(-, 2, 5, -1)` | **84.5%** | 92.1% | 95.8% | 71.0% | 90.9% | 12.4% | **0.534** | 182.3ms |

### 2. Query Expansion Internal Telemetry Dashboard (Exposing Bottlenecks)

| Configuration | Avg Anchors | Avg Cands $\ge \tau_{\text{sim}}$ | Avg Synonyms Injected | Starvation Rate (%) | Avg $Q_{\text{aug}}$ Length | Avg Anchor Rep ($R$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `v1_baseline` | 6.80 | 340.69 | **13.59** | **0.1%** | 38.9 tokens | 3.00x |
| `v5a_n3_r2-5_c-1` | 6.80 | 340.69 | **20.99** | **0.2%** | 47.8 tokens | 3.00x |
| `v5a_n3_r2-5_c0` | 6.80 | 340.69 | **27.77** | **0.2%** | 55.5 tokens | 3.00x |
| `v5a_n4_r2-5_c-1` | 6.80 | 340.69 | **20.99** | **0.2%** | 54.6 tokens | 4.00x |
| `v5b_r2-4_c-1` | 6.80 | 340.69 | **16.38** | **0.2%** | 45.2 tokens | 3.43x |
| `v5b_r3-5_c-1` | 6.80 | 340.69 | **23.17** | **0.2%** | 60.0 tokens | 4.43x |
| `v5b_r4-5_c-1` | 6.80 | 340.69 | **25.41** | **0.2%** | 64.8 tokens | 4.76x |
| `v5b_r2-5_c-1` | 6.80 | 340.69 | **20.99** | **0.2%** | 55.2 tokens | 4.12x |

### 3. Qualitative Query Expansion Case Studies

#### Case 1: Query `q_8968b27d`
- **Raw Question:** *"How many test tasks are included in the EHR-Complex benchmark?"*
- **Gold Chunk IDs:** `['EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk1', 'EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk2', 'EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk0', 'EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk3']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 18** | 0% | `EHR-Complex EHR-Complex EHR-Complex entry-filtered sca erbb2 esr1 EHR EHR EHR erbb2 esr1 e...` |
| `v5a_n3_r2-5_c-1` | **Rank 18** | 0% | `EHR-Complex EHR-Complex EHR-Complex entry-filtered sca erbb2 esr1 standardized metadata EH...` |
| `v5a_n3_r2-5_c0` | **Rank 17** | 0% | `EHR-Complex EHR-Complex EHR-Complex entry-filtered sca erbb2 esr1 standardized metadata ta...` |
| `v5a_n4_r2-5_c-1` | **Rank 10** | 25% | `EHR-Complex EHR-Complex EHR-Complex EHR-Complex entry-filtered sca erbb2 esr1 standardized...` |
| `v5b_r2-4_c-1` | **Rank 13** | 0% | `EHR-Complex EHR-Complex EHR-Complex EHR-Complex entry-filtered sca erbb2 esr1 standardized...` |
| `v5b_r3-5_c-1` | **Rank 12** | 0% | `EHR-Complex EHR-Complex EHR-Complex EHR-Complex EHR-Complex entry-filtered sca erbb2 esr1 ...` |
| `v5b_r4-5_c-1` | **Rank 11** | 0% | `EHR-Complex EHR-Complex EHR-Complex EHR-Complex EHR-Complex entry-filtered sca erbb2 esr1 ...` |
| `v5b_r2-5_c-1` | **Rank 9** | 25% | `EHR-Complex EHR-Complex EHR-Complex EHR-Complex EHR-Complex entry-filtered sca erbb2 esr1 ...` |

---

#### Case 2: Query `q_43cc4795`
- **Raw Question:** *"What challenge do population-level queries pose compared to patient-level queries in EHR reasoning?"*
- **Gold Chunk IDs:** `['EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk1']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 10** | 100% | `patient-level patient-level patient-level patients patient population-level population-lev...` |
| `v5a_n3_r2-5_c-1` | **Rank 17** | 0% | `patient-level patient-level patient-level patients patient erbb2 esr1 entry-filtered sca p...` |
| `v5a_n3_r2-5_c0` | **Rank 14** | 0% | `patient-level patient-level patient-level patients patient erbb2 esr1 entry-filtered sca c...` |
| `v5a_n4_r2-5_c-1` | **Rank 10** | 100% | `patient-level patient-level patient-level patient-level patients patient erbb2 esr1 entry-...` |
| `v5b_r2-4_c-1` | **Rank 10** | 100% | `patient-level patient-level patient-level patient-level patients patient erbb2 esr1 popula...` |
| `v5b_r3-5_c-1` | **Rank 10** | 100% | `patient-level patient-level patient-level patient-level patient-level patients patient erb...` |
| `v5b_r4-5_c-1` | **Rank 10** | 100% | `patient-level patient-level patient-level patient-level patient-level patients patient erb...` |
| `v5b_r2-5_c-1` | **Rank 10** | 100% | `patient-level patient-level patient-level patient-level patient-level patients patient erb...` |

---

#### Case 3: Query `q_e1f6a4fe`
- **Raw Question:** *"How do the dominant failure categories compare between Kimi-K2.5 and GPT-4.1 on EHR-Complex?"*
- **Gold Chunk IDs:** `['EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk2']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 15** | 0% | `EHR-Complex EHR-Complex EHR-Complex erbb2 esr1 entry-filtered sca GPT GPT GPT gata3 mki67 ...` |
| `v5a_n3_r2-5_c-1` | **Rank 17** | 0% | `EHR-Complex EHR-Complex EHR-Complex erbb2 esr1 entry-filtered sca gata3 mki67 standardized...` |
| `v5a_n3_r2-5_c0` | **Rank 17** | 0% | `EHR-Complex EHR-Complex EHR-Complex erbb2 esr1 entry-filtered sca gata3 mki67 standardized...` |
| `v5a_n4_r2-5_c-1` | **Rank 15** | 0% | `EHR-Complex EHR-Complex EHR-Complex EHR-Complex erbb2 esr1 entry-filtered sca gata3 mki67 ...` |
| `v5b_r2-4_c-1` | **Rank 14** | 0% | `EHR-Complex EHR-Complex EHR-Complex EHR-Complex erbb2 esr1 entry-filtered sca gata3 mki67 ...` |
| `v5b_r3-5_c-1` | **Rank 15** | 0% | `EHR-Complex EHR-Complex EHR-Complex EHR-Complex EHR-Complex erbb2 esr1 entry-filtered sca ...` |
| `v5b_r4-5_c-1` | **Rank 15** | 0% | `EHR-Complex EHR-Complex EHR-Complex EHR-Complex EHR-Complex erbb2 esr1 entry-filtered sca ...` |
| `v5b_r2-5_c-1` | **Rank 15** | 0% | `EHR-Complex EHR-Complex EHR-Complex EHR-Complex EHR-Complex erbb2 esr1 entry-filtered sca ...` |

---

#### Case 4: Query `q_a4117119`
- **Raw Question:** *"I want to boost the performance of a small open-weight model like Qwen3-32B for interactive EHR queries. What training approach does the EHR-Complex paper show to be effective?"*
- **Gold Chunk IDs:** `['EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk1', 'EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk2', 'EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk3', 'EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block2_chunk0']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 13** | 0% | `Qwen3-32B Qwen3-32B Qwen3-32B gata3 mki67 mbagcn reverse-gnn EHR-Complex EHR-Complex EHR-C...` |
| `v5a_n3_r2-5_c-1` | **Rank 20** | 0% | `Qwen3-32B Qwen3-32B Qwen3-32B gata3 mki67 mbagcn reverse-gnn dropedge mbagcn foxa1 gata3 E...` |
| `v5a_n3_r2-5_c0` | **Rank 18** | 0% | `Qwen3-32B Qwen3-32B Qwen3-32B gata3 mki67 mbagcn reverse-gnn dropedge mbagcn foxa1 gata3 e...` |
| `v5a_n4_r2-5_c-1` | **Rank 14** | 0% | `Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B gata3 mki67 mbagcn reverse-gnn dropedge mbagcn fox...` |
| `v5b_r2-4_c-1` | **Rank 12** | 0% | `Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B gata3 mki67 mbagcn reverse-gnn dropedge mbagcn EHR...` |
| `v5b_r3-5_c-1` | **Rank 15** | 0% | `Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B gata3 mki67 mbagcn reverse-gnn dropedge ...` |
| `v5b_r4-5_c-1` | **Rank 12** | 0% | `Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B gata3 mki67 mbagcn reverse-gnn dropedge ...` |
| `v5b_r2-5_c-1` | **Rank 8** | 25% | `Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B Qwen3-32B gata3 mki67 mbagcn reverse-gnn dropedge ...` |

---

#### Case 5: Query `q_c7835beb`
- **Raw Question:** *"What is the average number of SQL structural components per query in the EHR-Complex benchmark?"*
- **Gold Chunk IDs:** `['EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block0_chunk0', 'EHR_Complex__Benchmarking_Medical_Agents_for_Compl_block0_chunk1']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 5** | 100% | `EHR EHR EHR erbb2 esr1 entry-filtered sca EHR-Complex EHR-Complex EHR-Complex entry-filter...` |
| `v5a_n3_r2-5_c-1` | **Rank 3** | 100% | `EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr EHR-Complex EHR-Complex EHR-Complex...` |
| `v5a_n3_r2-5_c0` | **Rank 8** | 100% | `EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr averaging EHR-Complex EHR-Complex E...` |
| `v5a_n4_r2-5_c-1` | **Rank 3** | 100% | `EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr EHR-Complex EHR-Complex EHR-Com...` |
| `v5b_r2-4_c-1` | **Rank 3** | 100% | `EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata EHR-Complex EHR-Complex EHR-Complex...` |
| `v5b_r3-5_c-1` | **Rank 3** | 100% | `EHR EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr EHR-Complex EHR-Complex EHR...` |
| `v5b_r4-5_c-1` | **Rank 3** | 100% | `EHR EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr EHR-Complex EHR-Complex EHR...` |
| `v5b_r2-5_c-1` | **Rank 3** | 100% | `EHR EHR EHR EHR EHR erbb2 esr1 entry-filtered sca metadata asr EHR-Complex EHR-Complex EHR...` |

---

## Dataset: `enterpriserag_stress_1000`

### 1. Multi-Level Retrieval Metrics (Strict Recall, Chunk Recall, Precision, MRR)

| Configuration | Parameters | Strict@10 | Strict@20 | Strict@50 | ChunkRec@10 | ChunkRec@50 | ChunkPrec@10 | MRR@10 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Schema 1 Baseline (3,3,3,-1)** 🏆 | `(3, 3, 3, -1)` | **82.0%** | 85.6% | 89.8% | 58.3% | 75.7% | 15.7% | **0.555** | 64.7ms |
| **Schema 5a (3, 2, 5, -1)** | `(3, 2, 5, -1)` | **80.0%** | 85.0% | 90.2% | 56.7% | 75.6% | 15.1% | **0.514** | 70.6ms |
| **Schema 5a (3, 2, 5, 0)** | `(3, 2, 5, 0)` | **79.2%** | 85.2% | 89.6% | 55.4% | 74.6% | 14.5% | **0.484** | 76.0ms |
| **Schema 5a (4, 2, 5, -1)** | `(4, 2, 5, -1)` | **81.8%** | 85.2% | 90.0% | 58.3% | 75.8% | 15.9% | **0.583** | 75.1ms |
| **Schema 5b (-, 2, 4, -1)** | `(-, 2, 4, -1)` | **81.0%** | 85.2% | 89.6% | 58.2% | 75.7% | 15.7% | **0.569** | 68.5ms |
| **Schema 5b (-, 3, 5, -1)** | `(-, 3, 5, -1)` | **81.6%** | 85.0% | 91.0% | 58.9% | 76.8% | 15.9% | **0.592** | 78.4ms |
| **Schema 5b (-, 4, 5, -1)** | `(-, 4, 5, -1)` | **81.8%** | 85.4% | 90.4% | 59.0% | 75.9% | 16.1% | **0.599** | 81.6ms |
| **Schema 5b (-, 2, 5, -1)** | `(-, 2, 5, -1)` | **81.0%** | 84.6% | 90.6% | 58.5% | 76.0% | 15.8% | **0.586** | 74.8ms |

### 2. Query Expansion Internal Telemetry Dashboard (Exposing Bottlenecks)

| Configuration | Avg Anchors | Avg Cands $\ge \tau_{\text{sim}}$ | Avg Synonyms Injected | Starvation Rate (%) | Avg $Q_{\text{aug}}$ Length | Avg Anchor Rep ($R$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `v1_baseline` | 11.73 | 339.00 | **23.46** | **0.0%** | 61.6 tokens | 3.00x |
| `v5a_n3_r2-5_c-1` | 11.73 | 339.00 | **35.30** | **0.0%** | 74.1 tokens | 3.00x |
| `v5a_n3_r2-5_c0` | 11.73 | 339.00 | **47.02** | **0.0%** | 86.3 tokens | 3.00x |
| `v5a_n4_r2-5_c-1` | 11.73 | 339.00 | **35.30** | **0.0%** | 85.9 tokens | 4.00x |
| `v5b_r2-4_c-1` | 11.73 | 339.00 | **27.49** | **0.0%** | 70.0 tokens | 3.35x |
| `v5b_r3-5_c-1` | 11.73 | 339.00 | **39.22** | **0.0%** | 94.1 tokens | 4.35x |
| `v5b_r4-5_c-1` | 11.73 | 339.00 | **43.69** | **0.0%** | 103.3 tokens | 4.74x |
| `v5b_r2-5_c-1` | 11.73 | 339.00 | **35.30** | **0.0%** | 86.1 tokens | 4.03x |

### 3. Qualitative Query Expansion Case Studies

#### Case 1: Query `qst_0001`
- **Raw Question:** *"What are the default size limits for file uploads and total request size for the new multipart upload support on the OpenAI-compatible API endpoints?"*
- **Gold Chunk IDs:** `['dsid_ae068ee4aa9640159427cd941bef0238_block0_chunk1', 'dsid_ae068ee4aa9640159427cd941bef0238_block0_chunk0']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 1** | 50% | `OpenAI-compatible OpenAI-compatible OpenAI-compatible kubernetes multi-region API API API ...` |
| `v5a_n3_r2-5_c-1` | **Rank 1** | 50% | `OpenAI-compatible OpenAI-compatible OpenAI-compatible kubernetes multi-region versioned hs...` |
| `v5a_n3_r2-5_c0` | **Rank 1** | 50% | `OpenAI-compatible OpenAI-compatible OpenAI-compatible kubernetes multi-region versioned hs...` |
| `v5a_n4_r2-5_c-1` | **Rank 1** | 50% | `OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible kubernetes multi-r...` |
| `v5b_r2-4_c-1` | **Rank 1** | 50% | `OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible kubernetes multi-r...` |
| `v5b_r3-5_c-1` | **Rank 1** | 50% | `OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible ...` |
| `v5b_r4-5_c-1` | **Rank 1** | 50% | `OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible ...` |
| `v5b_r2-5_c-1` | **Rank 1** | 50% | `OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible OpenAI-compatible ...` |

---

#### Case 2: Query `qst_0002`
- **Raw Question:** *"What is the name of the new metric added so SRE can track when server-side streaming sessions get finalized due to hitting the time limit?"*
- **Gold Chunk IDs:** `['dsid_9550250a59e74f1bbd5612480b2e7100_block0_chunk0']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 2** | 100% | `SRE SRE SRE sse real-time server-side server-side server-side --org redwood-demo add-on fi...` |
| `v5a_n3_r2-5_c-1` | **Rank 4** | 100% | `SRE SRE SRE sse real-time rps precision server-side server-side server-side --org redwood-...` |
| `v5a_n3_r2-5_c0` | **Rank 4** | 100% | `SRE SRE SRE sse real-time rps precision aug server-side server-side server-side --org redw...` |
| `v5a_n4_r2-5_c-1` | **Rank 2** | 100% | `SRE SRE SRE SRE sse real-time rps precision server-side server-side server-side server-sid...` |
| `v5b_r2-4_c-1` | **Rank 2** | 100% | `SRE SRE SRE SRE sse real-time rps server-side server-side server-side server-side --org re...` |
| `v5b_r3-5_c-1` | **Rank 2** | 100% | `SRE SRE SRE SRE SRE sse real-time rps precision server-side server-side server-side server...` |
| `v5b_r4-5_c-1` | **Rank 2** | 100% | `SRE SRE SRE SRE SRE sse real-time rps precision server-side server-side server-side server...` |
| `v5b_r2-5_c-1` | **Rank 2** | 100% | `SRE SRE SRE SRE SRE sse real-time rps precision server-side server-side server-side server...` |

---

#### Case 3: Query `qst_0003`
- **Raw Question:** *"What are the acceptance criteria for the project introducing an algorithm to generate interactive UI color states and a Kappa-style elevation scale for dense table and grid components?"*
- **Gold Chunk IDs:** `['dsid_3fd6af404fae48e6b8ea5a57875ef78f_block0_chunk0']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 6** | 100% | `UI UI UI layout component Kappa-style Kappa-style Kappa-style layout --org redwood-demo gr...` |
| `v5a_n3_r2-5_c-1` | **Rank 7** | 100% | `UI UI UI layout component screen --org redwood-demo Kappa-style Kappa-style Kappa-style la...` |
| `v5a_n3_r2-5_c0` | **Rank 7** | 100% | `UI UI UI layout component screen --org redwood-demo add-on Kappa-style Kappa-style Kappa-s...` |
| `v5a_n4_r2-5_c-1` | **Rank 5** | 100% | `UI UI UI UI layout component screen --org redwood-demo Kappa-style Kappa-style Kappa-style...` |
| `v5b_r2-4_c-1` | **Rank 5** | 100% | `UI UI UI UI layout component screen Kappa-style Kappa-style Kappa-style Kappa-style layout...` |
| `v5b_r3-5_c-1` | **Rank 5** | 100% | `UI UI UI UI UI layout component screen --org redwood-demo Kappa-style Kappa-style Kappa-st...` |
| `v5b_r4-5_c-1` | **Rank 2** | 100% | `UI UI UI UI UI layout component screen --org redwood-demo Kappa-style Kappa-style Kappa-st...` |
| `v5b_r2-5_c-1` | **Rank 5** | 100% | `UI UI UI UI UI layout component screen --org redwood-demo Kappa-style Kappa-style Kappa-st...` |

---

#### Case 4: Query `qst_0004`
- **Raw Question:** *"In the meeting about onboarding a SaaS product to Google Cloud Marketplace, what did the GCP team recommend for handling delays where a new subscription entitlement is not immediately available during the customer onboarding flow?"*
- **Gold Chunk IDs:** `['dsid_6c4c1c875e704f09b4d791d64d7bc7e5_block0_chunk2', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_block1_chunk2', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_block0_chunk3', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_bridge0-1', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_block1_chunk3', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_bridge1-2', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_block1_chunk0', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_block2_chunk0', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_block0_chunk0', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_block1_chunk1', 'dsid_6c4c1c875e704f09b4d791d64d7bc7e5_block0_chunk1']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 3** | 18% | `GCP GCP GCP --org redwood-demo stakeholders google google google --org redwood-demo on-dem...` |
| `v5a_n3_r2-5_c-1` | **Rank 2** | 27% | `GCP GCP GCP --org redwood-demo stakeholders rps roadmap google google google --org redwood...` |
| `v5a_n3_r2-5_c0` | **Rank 2** | 27% | `GCP GCP GCP --org redwood-demo stakeholders rps roadmap cert google google google --org re...` |
| `v5a_n4_r2-5_c-1` | **Rank 1** | 27% | `GCP GCP GCP GCP --org redwood-demo stakeholders rps roadmap google google google google --...` |
| `v5b_r2-4_c-1` | **Rank 1** | 27% | `GCP GCP GCP GCP --org redwood-demo stakeholders rps google google google google --org redw...` |
| `v5b_r3-5_c-1` | **Rank 1** | 36% | `GCP GCP GCP GCP GCP --org redwood-demo stakeholders rps roadmap google google google googl...` |
| `v5b_r4-5_c-1` | **Rank 1** | 36% | `GCP GCP GCP GCP GCP --org redwood-demo stakeholders rps roadmap google google google googl...` |
| `v5b_r2-5_c-1` | **Rank 1** | 36% | `GCP GCP GCP GCP GCP --org redwood-demo stakeholders rps roadmap google google google googl...` |

---

#### Case 5: Query `qst_0005`
- **Raw Question:** *"What failover sequence and recovery targets did MedThink specify for handling an EU region outage, including any limits on how long traffic can shift to the US?"*
- **Gold Chunk IDs:** `['dsid_8e838ab6a98f4cbcb672d41f210ff89c_block0_chunk2', 'dsid_8e838ab6a98f4cbcb672d41f210ff89c_block0_chunk0', 'dsid_8e838ab6a98f4cbcb672d41f210ff89c_block0_chunk1']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 1** | 100% | `EU EU EU eu-west per-region US US US us-west us-west-2 medthink medthink medthink thinking...` |
| `v5a_n3_r2-5_c-1` | **Rank 1** | 100% | `EU EU EU eu-west per-region multi-region migrations US US US us-west us-west-2 overage sta...` |
| `v5a_n3_r2-5_c0` | **Rank 1** | 100% | `EU EU EU eu-west per-region multi-region migrations infrastructure US US US us-west us-wes...` |
| `v5a_n4_r2-5_c-1` | **Rank 1** | 100% | `EU EU EU EU eu-west per-region multi-region migrations US US US US us-west us-west-2 overa...` |
| `v5b_r2-4_c-1` | **Rank 1** | 100% | `EU EU EU EU eu-west per-region multi-region US US US US us-west us-west-2 overage medthink...` |
| `v5b_r3-5_c-1` | **Rank 1** | 100% | `EU EU EU EU EU eu-west per-region multi-region migrations US US US US US us-west us-west-2...` |
| `v5b_r4-5_c-1` | **Rank 1** | 100% | `EU EU EU EU EU eu-west per-region multi-region migrations US US US US US us-west us-west-2...` |
| `v5b_r2-5_c-1` | **Rank 1** | 100% | `EU EU EU EU EU eu-west per-region multi-region migrations US US US US US us-west us-west-2...` |

---

## Dataset: `liverag_stress_full`

### 1. Multi-Level Retrieval Metrics (Strict Recall, Chunk Recall, Precision, MRR)

| Configuration | Parameters | Strict@10 | Strict@20 | Strict@50 | ChunkRec@10 | ChunkRec@50 | ChunkPrec@10 | MRR@10 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Schema 1 Baseline (3,3,3,-1)** 🏆 | `(3, 3, 3, -1)` | **93.1%** | 95.0% | 97.2% | 78.9% | 87.4% | 18.4% | **0.800** | 37.9ms |
| **Schema 5a (3, 2, 5, -1)** | `(3, 2, 5, -1)` | **92.0%** | 95.4% | 97.1% | 76.8% | 87.3% | 17.8% | **0.758** | 40.6ms |
| **Schema 5a (3, 2, 5, 0)** | `(3, 2, 5, 0)` | **91.2%** | 94.6% | 96.6% | 75.6% | 86.9% | 17.5% | **0.740** | 42.5ms |
| **Schema 5a (4, 2, 5, -1)** | `(4, 2, 5, -1)` | **92.8%** | 95.4% | 97.1% | 78.5% | 87.3% | 18.3% | **0.814** | 41.9ms |
| **Schema 5b (-, 2, 4, -1)** 🏆 | `(-, 2, 4, -1)` | **92.8%** | 95.9% | 97.2% | 79.0% | 87.6% | 18.6% | **0.817** | 39.9ms |
| **Schema 5b (-, 3, 5, -1)** | `(-, 3, 5, -1)` | **93.3%** | 95.8% | 97.1% | 79.6% | 87.5% | 18.7% | **0.834** | 43.0ms |
| **Schema 5b (-, 4, 5, -1)** | `(-, 4, 5, -1)` | **93.2%** | 95.3% | 97.0% | 79.1% | 87.5% | 18.5% | **0.832** | 44.2ms |
| **Schema 5b (-, 2, 5, -1)** | `(-, 2, 5, -1)` | **93.0%** | 95.4% | 97.1% | 79.3% | 87.6% | 18.6% | **0.830** | 42.4ms |

### 2. Query Expansion Internal Telemetry Dashboard (Exposing Bottlenecks)

| Configuration | Avg Anchors | Avg Cands $\ge \tau_{\text{sim}}$ | Avg Synonyms Injected | Starvation Rate (%) | Avg $Q_{\text{aug}}$ Length | Avg Anchor Rep ($R$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `v1_baseline` | 5.59 | 158.57 | **11.03** | **1.7%** | 31.1 tokens | 3.00x |
| `v5a_n3_r2-5_c-1` | 5.59 | 158.57 | **18.10** | **2.8%** | 39.6 tokens | 3.00x |
| `v5a_n3_r2-5_c0` | 5.59 | 158.57 | **23.50** | **3.3%** | 45.7 tokens | 3.00x |
| `v5a_n4_r2-5_c-1` | 5.59 | 158.57 | **18.10** | **2.8%** | 45.2 tokens | 4.00x |
| `v5b_r2-4_c-1` | 5.59 | 158.57 | **13.72** | **2.2%** | 37.2 tokens | 3.54x |
| `v5b_r3-5_c-1` | 5.59 | 158.57 | **19.15** | **2.8%** | 49.2 tokens | 4.54x |
| `v5b_r4-5_c-1` | 5.59 | 158.57 | **21.38** | **2.9%** | 54.0 tokens | 4.92x |
| `v5b_r2-5_c-1` | 5.59 | 158.57 | **18.10** | **2.8%** | 46.9 tokens | 4.36x |

### 3. Qualitative Query Expansion Case Studies

#### Case 1: Query `q_live_0`
- **Raw Question:** *"How deep can fish survive in the ocean trenches?"*
- **Gold Chunk IDs:** `['<urn:uuid:a102a6cb-a608-493c-928f-d32a0da4dbf6>_block0_chunk0', '<urn:uuid:a102a6cb-a608-493c-928f-d32a0da4dbf6>_block0_chunk1']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 4** | 50% | `trenches trenches trenches armored division sand-fired pot survive survive survive lived h...` |
| `v5a_n3_r2-5_c-1` | **Rank 5** | 50% | `trenches trenches trenches armored division sand-fired pot hydraulic fracturing beneath su...` |
| `v5a_n3_r2-5_c0` | **Rank 5** | 50% | `trenches trenches trenches armored division sand-fired pot hydraulic fracturing beneath un...` |
| `v5a_n4_r2-5_c-1` | **Rank 3** | 100% | `trenches trenches trenches trenches armored division sand-fired pot hydraulic fracturing b...` |
| `v5b_r2-4_c-1` | **Rank 3** | 100% | `trenches trenches trenches trenches armored division sand-fired pot hydraulic fracturing s...` |
| `v5b_r3-5_c-1` | **Rank 3** | 100% | `trenches trenches trenches trenches trenches armored division sand-fired pot hydraulic fra...` |
| `v5b_r4-5_c-1` | **Rank 4** | 100% | `trenches trenches trenches trenches trenches armored division sand-fired pot hydraulic fra...` |
| `v5b_r2-5_c-1` | **Rank 3** | 100% | `trenches trenches trenches trenches trenches armored division sand-fired pot hydraulic fra...` |

---

#### Case 2: Query `q_live_1`
- **Raw Question:** *"Based on temperature considerations alone, is March considered a suitable month to perform the final pruning of grape vines?"*
- **Gold Chunk IDs:** `['<urn:uuid:b5d19fcb-1711-4f9f-82cf-f81403382444>_block0_chunk0', '<urn:uuid:b5d19fcb-1711-4f9f-82cf-f81403382444>_block0_chunk1']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 25** | 0% | `grape grape grape peacock mantis mantis shrimp vines vines vines peacock mantis lab-grown ...` |
| `v5a_n3_r2-5_c-1` | **Rank 29** | 0% | `grape grape grape peacock mantis mantis shrimp fruit valley vines vines vines peacock mant...` |
| `v5a_n3_r2-5_c0` | **Rank 40** | 0% | `grape grape grape peacock mantis mantis shrimp fruit valley fruits vines vines vines peaco...` |
| `v5a_n4_r2-5_c-1` | **Rank 26** | 0% | `grape grape grape grape peacock mantis mantis shrimp fruit valley vines vines vines vines ...` |
| `v5b_r2-4_c-1` | **Rank 20** | 0% | `grape grape grape grape peacock mantis mantis shrimp fruit vines vines vines vines peacock...` |
| `v5b_r3-5_c-1` | **Rank 19** | 0% | `grape grape grape grape grape peacock mantis mantis shrimp fruit valley vines vines vines ...` |
| `v5b_r4-5_c-1` | **Rank 33** | 0% | `grape grape grape grape grape peacock mantis mantis shrimp fruit valley vines vines vines ...` |
| `v5b_r2-5_c-1` | **Rank 26** | 0% | `grape grape grape grape grape peacock mantis mantis shrimp fruit valley vines vines vines ...` |

---

#### Case 3: Query `q_live_2`
- **Raw Question:** *"What major acts performed at the Brighton Hippodrome during its peak years?"*
- **Gold Chunk IDs:** `['<urn:uuid:95479dfb-3efd-4235-9bb8-4bfb98caab4f>_block0_chunk0']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 1** | 100% | `hippodrome hippodrome hippodrome pelvic girdle vertical brighton brighton brighton london ...` |
| `v5a_n3_r2-5_c-1` | **Rank 1** | 100% | `hippodrome hippodrome hippodrome pelvic girdle vertical arm leg brighton brighton brighton...` |
| `v5a_n3_r2-5_c0` | **Rank 1** | 100% | `hippodrome hippodrome hippodrome pelvic girdle vertical arm leg vice brighton brighton bri...` |
| `v5a_n4_r2-5_c-1` | **Rank 1** | 100% | `hippodrome hippodrome hippodrome hippodrome pelvic girdle vertical arm leg brighton bright...` |
| `v5b_r2-4_c-1` | **Rank 1** | 100% | `hippodrome hippodrome hippodrome hippodrome pelvic girdle vertical arm brighton brighton b...` |
| `v5b_r3-5_c-1` | **Rank 1** | 100% | `hippodrome hippodrome hippodrome hippodrome hippodrome pelvic girdle vertical arm leg brig...` |
| `v5b_r4-5_c-1` | **Rank 1** | 100% | `hippodrome hippodrome hippodrome hippodrome hippodrome pelvic girdle vertical arm leg brig...` |
| `v5b_r2-5_c-1` | **Rank 1** | 100% | `hippodrome hippodrome hippodrome hippodrome hippodrome pelvic girdle vertical arm leg brig...` |

---

#### Case 4: Query `q_live_3`
- **Raw Question:** *"I noticed some stucco houses in my neighborhood. What are the potential drawbacks and limitations of using a one-coat stucco system on exterior walls?"*
- **Gold Chunk IDs:** `['<urn:uuid:42ae24ce-dc04-4b1e-b0a8-ff18c900fae1>_block0_chunk1', '<urn:uuid:42ae24ce-dc04-4b1e-b0a8-ff18c900fae1>_block0_chunk0', '<urn:uuid:42ae24ce-dc04-4b1e-b0a8-ff18c900fae1>_block0_chunk2']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 1** | 67% | `one-coat one-coat one-coat covering layers drawbacks drawbacks drawbacks advantages risks ...` |
| `v5a_n3_r2-5_c-1` | **Rank 1** | 33% | `one-coat one-coat one-coat covering layers designs walls drawbacks drawbacks drawbacks adv...` |
| `v5a_n3_r2-5_c0` | **Rank 1** | 33% | `one-coat one-coat one-coat covering layers designs walls protecting drawbacks drawbacks dr...` |
| `v5a_n4_r2-5_c-1` | **Rank 1** | 33% | `one-coat one-coat one-coat one-coat covering layers designs walls drawbacks drawbacks draw...` |
| `v5b_r2-4_c-1` | **Rank 1** | 67% | `one-coat one-coat one-coat one-coat covering layers designs drawbacks drawbacks drawbacks ...` |
| `v5b_r3-5_c-1` | **Rank 1** | 67% | `one-coat one-coat one-coat one-coat one-coat covering layers designs walls drawbacks drawb...` |
| `v5b_r4-5_c-1` | **Rank 1** | 67% | `one-coat one-coat one-coat one-coat one-coat covering layers designs walls drawbacks drawb...` |
| `v5b_r2-5_c-1` | **Rank 1** | 33% | `one-coat one-coat one-coat one-coat one-coat covering layers designs walls drawbacks drawb...` |

---

#### Case 5: Query `q_live_4`
- **Raw Question:** *"I need help with a business case study - what format should I use for writing it up?"*
- **Gold Chunk IDs:** `['<urn:uuid:85d65922-709a-4f5b-9de7-0c2c19fe8ad3>_block0_chunk0']`

| Configuration | Rank of 1st Gold Chunk | Chunk Recall@10 | Generated $Q_{\text{aug}}$ Token Sample |
| :--- | :---: | :---: | :--- |
| `v1_baseline` | **Rank 1** | 100% | `format format format utm bounding sheet writing writing writing wrote fountain pens busine...` |
| `v5a_n3_r2-5_c-1` | **Rank 2** | 100% | `format format format utm bounding sheet pandas dataframe document writing writing writing ...` |
| `v5a_n3_r2-5_c0` | **Rank 1** | 100% | `format format format utm bounding sheet pandas dataframe document vol pages writing writin...` |
| `v5a_n4_r2-5_c-1` | **Rank 1** | 100% | `format format format format utm bounding sheet pandas dataframe document writing writing w...` |
| `v5b_r2-4_c-1` | **Rank 2** | 100% | `format format format format utm bounding sheet pandas dataframe writing writing writing wr...` |
| `v5b_r3-5_c-1` | **Rank 1** | 100% | `format format format format format utm bounding sheet pandas dataframe document writing wr...` |
| `v5b_r4-5_c-1` | **Rank 1** | 100% | `format format format format format utm bounding sheet pandas dataframe document writing wr...` |
| `v5b_r2-5_c-1` | **Rank 1** | 100% | `format format format format format utm bounding sheet pandas dataframe document writing wr...` |

---

