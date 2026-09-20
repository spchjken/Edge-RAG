# 🗺️ Edge-RAG Results-to-Scripts Mapping

This file serves as the canonical registry linking every empirical result artifact (`.csv`, `.txt`, `.md`) under `results/` and `temp/` to its corresponding generator script or evaluation harness.

---

## 1. Active Evaluation Suite Registry

| Result File (`results/`) | Corresponding Script / Test | Description |
| :--- | :--- | :--- |
| `results/pyterrier_baselines/pyterrier_baselines_results.csv` | `scripts/run_pyterrier_baselines.py`, `scripts/run_benchmarks_neural_batch.py` | Canonical PyTerrier 8-baseline evaluation suite (6 classical + 2 neural: BGE_Small_Dense, SPLADE_v3_PISA) across 25 benchmark datasets (13 BEIR + 12 BRIGHT) |
| `results/pyterrier_baselines/pyterrier_baselines_summary.md` | `scripts/run_pyterrier_baselines.py`, `scripts/run_benchmarks_neural_batch.py` | Summary tables (Linear nDCG@10, Exp-nDCG, Recall@K, Completeness@K, Latency) |
| `results/pool_isolation/pool_candidate_audit.parquet` | `scripts/run_pool_oracle_isolation.py` | Master audit dataset (1,230,704 candidate evaluations across SciFact, BRIGHT-AOPS, NFCorpus, TREC-COVID) |
| `results/pool_isolation/pool_oracle_summary.csv` | `scripts/compile_pool_oracle_tables.py` | Table 1 summary comparing baseline, Raw Lexical Ceiling, Operational Eligible Ceiling, and 5 deployable policies at 10k capacity |
| `results/pool_isolation/pool_df_band_attribution.csv` | `scripts/compile_pool_oracle_tables.py` | Table 2 DF band attribution across 6 strictly disjoint corpus-aware DF bands |
| `results/pool_isolation/pool_capacity_knee_curve.csv` | `scripts/compile_pool_oracle_tables.py` | Table 3 capacity knee curves (1k, 2.5k, 5k, 10k, 15k, 20k) and retention across all 5 policies |
| `results/pool_isolation/metadata_manifest.json` | `scripts/compile_pool_oracle_tables.py` | Environment, commit, and metric definition manifest |
| `results/pool_isolation/pool_oracle_report.md` | `scripts/run_pool_oracle_isolation.py`, `scripts/compile_pool_oracle_tables.py` | Definitive Stage 1 evaluation and policy selection report |
| `results/gate1_selection/dev_run/gate1_candidate_audit.parquet` | `scripts/run_gate1_oracle_evaluation.py` | Phase 2 Gate 1 master candidate audit (parent table for counterfactual action evaluations) |
| `results/gate1_selection/dev_run/gate1_sparse_transitions.parquet` | `scripts/run_gate1_oracle_evaluation.py` | Phase 2 Gate 1 sparse document transitions (child table for judged relevant document rank movements and cutoff crossings) |
| `results/gate1_selection/dev_run/table1_reference_ceiling.csv` | `scripts/compile_gate1_research_tables.py` | Table 1 summary comparing baseline, Reference Opportunity Ceiling, and addressability |
| `results/gate1_selection/dev_run/table2_channel_comparison.csv` | `scripts/compile_gate1_research_tables.py` | Table 2 proposal channel comparison (WholeQuery, AnchorFiltered, AnchorAll, LexicalPPMI) |
| `results/gate1_selection/dev_run/table6_budget_sensitivity.csv` | `scripts/compile_gate1_research_tables.py` | Table 6 candidate budget retention knee curves (L in {10, 20, 50, 100, 200}) |
| `results/gate1_selection/dev_run/frozen_gate1_config.json` | `scripts/compile_gate1_research_tables.py` | Frozen winning Gate 1 deployable configuration and SHA-256 hash |
| `results/gate1_selection/dev_run/gate1_selection_report.md` | `scripts/compile_gate1_research_tables.py` | Comprehensive Gate 1 evaluation, centroid gating, and policy selection report |


---

## 2. Legacy Results & Scripts Registry (`results/legacy/` & `scripts/legacy/`)

### 2.1 Pipeline V1 Legacy Suite (`results/legacy/pipeline_v1_legacy/` & `scripts/legacy/pipeline_v1_legacy/`)

| Result File (`results/legacy/pipeline_v1_legacy/`) | Corresponding Script (`scripts/legacy/pipeline_v1_legacy/` or `src/`) |
| :--- | :--- |
| `results/legacy/pipeline_v1_legacy/doc_level_ablation/doc_level_sweep_summary.md` | `scripts/legacy/pipeline_v1_legacy/run_doc_level_ablation_sweep.py` |
| `results/legacy/pipeline_v1_legacy/doc_level_ablation/small tests/doc_level_sweep_results.csv` | `scripts/legacy/pipeline_v1_legacy/run_doc_level_ablation_sweep.py` |
| `results/legacy/pipeline_v1_legacy/doc_level_ablation/small tests/doc_level_sweep_summary.md` | `scripts/legacy/pipeline_v1_legacy/run_doc_level_ablation_sweep.py` |
| `results/legacy/pipeline_v1_legacy/v5_independent_ablation/v5_independent_ablation_report.md` | `scripts/legacy/pipeline_v1_legacy/evaluate_v5_independent_ablation.py` / `scripts/legacy/pipeline_v1_legacy/run_v5_independent_ablation.sh` |
| `results/legacy/pipeline_v1_legacy/v5_joint_ablation/v5_joint_ablation_report.md` | `scripts/legacy/pipeline_v1_legacy/evaluate_v5_joint_ablation.py` / `scripts/legacy/pipeline_v1_legacy/run_v5_joint_ablation.sh` |
| `results/legacy/pipeline_v1_legacy/baseline_comparison/baseline_comparison_fused_stress_*.md` (6 files) | `scripts/legacy/pipeline_v1_legacy/run_baseline_comparison.py` / `scripts/legacy/pipeline_v1_legacy/run_baseline_comparison.sh` |
| `results/legacy/pipeline_v1_legacy/baseline_comparison/existing baseline run/*.md` (17 files) | `scripts/legacy/pipeline_v1_legacy/run_baseline_comparison.py` / `scripts/legacy/pipeline_v1_legacy/run_baseline_comparison_rerank.sh` |
| `results/legacy/pipeline_v1_legacy/baseline_comparison/old comparison/*.md` (11 files) | `scripts/legacy/pipeline_v1_legacy/run_baseline_comparison.py` |
| `results/legacy/pipeline_v1_legacy/benchmark_final/external_benchmarks_eval_report.md` | `scripts/legacy/pipeline_v1_legacy/evaluate_external_runs.py` / `scripts/legacy/pipeline_v1_legacy/run_all_external_benchmarks.sh` / `scripts/legacy/pipeline_v1_legacy/run_eval_and_save_report.py` |
| `results/legacy/pipeline_v1_legacy/benchmark_final/dense_vocab_ablation/ablation_summary_report.md` | `scripts/legacy/pipeline_v1_legacy/generate_ablation_report.py` |
| `results/legacy/pipeline_v1_legacy/benchmark_final/new_v4_5_test/summary_report.md` | `scripts/legacy/pipeline_v1_legacy/evaluate_benchmark.py` |
| `results/legacy/pipeline_v1_legacy/benchmark_final/pre-refine-test/eval_ai.txt` | `scripts/legacy/pipeline_v1_legacy/run_all_evals.py` |
| `results/legacy/pipeline_v1_legacy/benchmark_final/pre-refine-test/eval_fintech.txt` | `scripts/legacy/pipeline_v1_legacy/run_all_evals.py` |
| `results/legacy/pipeline_v1_legacy/benchmark_final/pre-refine-test/eval_gemma4.txt` | `scripts/legacy/pipeline_v1_legacy/run_all_evals.py` |
| `results/legacy/pipeline_v1_legacy/benchmark_final/pre-refine-test/eval_gemma4_4b.txt` | `scripts/legacy/pipeline_v1_legacy/run_all_evals.py` |
| `results/legacy/pipeline_v1_legacy/benchmark_final/pre-refine-test/eval_granite4.txt` | `scripts/legacy/pipeline_v1_legacy/run_all_evals.py` |
| `results/legacy/pipeline_v1_legacy/benchmark_final/pre-refine-test/eval_multi.txt` | `scripts/legacy/pipeline_v1_legacy/run_all_evals.py` |
| `results/legacy/pipeline_v1_legacy/benchmark_final/pre-refine-test/eval_qwen4b.txt` | `scripts/legacy/pipeline_v1_legacy/run_all_evals.py` |
| `results/legacy/pipeline_v1_legacy/benchmark_final/pre-refine-test/eval_stress.txt` | `scripts/legacy/pipeline_v1_legacy/run_all_evals.py` |
| `results/legacy/pipeline_v1_legacy/benchmark_final/routing ablation/eval_gemma4_routing_ablation.txt` | `scripts/legacy/pipeline_v1_legacy/build_all_routing_reports.py` |
| `results/legacy/pipeline_v1_legacy/benchmark_final/routing ablation/eval_gemma4_routing_ablation_v4.txt` | `scripts/legacy/pipeline_v1_legacy/build_all_routing_reports.py` |
| `results/legacy/pipeline_v1_legacy/benchmark_final/routing ablation/routing_ablation_summary_report.md` | `scripts/legacy/pipeline_v1_legacy/generate_routing_report.py` |
| `results/legacy/pipeline_v1_legacy/benchmark_final/routing ablation/routing_ablation_summary_report_v4.md` | `scripts/legacy/pipeline_v1_legacy/generate_routing_report.py` |
| `results/legacy/pipeline_v1_legacy/pipeline_combinations/benchmark_eval_report.md` | `scripts/legacy/pipeline_v1_legacy/evaluate_benchmark.py` |
| `results/legacy/pipeline_v1_legacy/pipeline_combinations/eval_report.md` | Historical combination runner |
| `results/legacy/pipeline_v1_legacy/query_expansion/query_expansion_ablation.csv` | `src/evaluation/query_expansion_ablation.py` |
| `results/legacy/pipeline_v1_legacy/query_expansion/query_expansion_ablation_summary.md` | `src/evaluation/query_expansion_ablation.py` |
| `results/legacy/pipeline_v1_legacy/routing_test/cascade_router_metrics.csv` | `src/evaluation/evaluate_router.py` |
| `results/legacy/pipeline_v1_legacy/routing_test/cascade_router_sensitivity.md` | `src/evaluation/evaluate_router.py` |

### 2.2 Pipeline V2 Ablation Suite (`results/legacy/v2_ablation/` & `scripts/legacy/v2_ablation/`)

| Result File (`results/legacy/v2_ablation/`) | Corresponding Script (`scripts/legacy/v2_ablation/`) |
| :--- | :--- |
| `results/legacy/v2_ablation/p_sweep_ablation/p_sweep_results.csv` | `scripts/legacy/v2_ablation/run_p_sweep_ablation.py` / `scripts/legacy/v2_ablation/evaluate_v7_p_sweep.py` |
| `results/legacy/v2_ablation/p_sweep_ablation/p_sweep_summary.md` | `scripts/legacy/v2_ablation/run_p_sweep_ablation.py` / `scripts/legacy/v2_ablation/compute_macro_p_sweep.py` |
| `results/legacy/v2_ablation/p_sweep_ablation/small test/p_sweep_results.csv` | `scripts/legacy/v2_ablation/run_p_sweep_ablation.py` |
| `results/legacy/v2_ablation/p_sweep_ablation/small test/p_sweep_summary.md` | `scripts/legacy/v2_ablation/run_p_sweep_ablation.py` |
| `results/legacy/v2_ablation/v6_ablation/v6_sweep_results.csv` | `scripts/legacy/v2_ablation/run_v6_ablation_sweep.py` |
| `results/legacy/v2_ablation/v6_ablation/v6_sweep_summary.md` | `scripts/legacy/v2_ablation/run_v6_ablation_sweep.py` |
| `results/legacy/v2_ablation/v5_ablation/v5_sweep_results.csv` | `scripts/legacy/v2_ablation/run_v5_ablation_sweep.py` |
| `results/legacy/v2_ablation/v5_ablation/v5_sweep_summary.md` | `scripts/legacy/v2_ablation/run_v5_ablation_sweep.py` |
| `results/legacy/v2_ablation/v5_ablation/pre_vs_post_refine_comparison.md` | `scripts/legacy/v2_ablation/generate_v2_baseline_comparison_report.py` |
| `results/legacy/v2_ablation/schema/baseline_comparison_*.md` | `scripts/legacy/pipeline_v1_legacy/run_baseline_comparison.py` / `scripts/legacy/v2_ablation/run_all_fused_pipeline_v2.sh` |
| `results/legacy/v2_ablation/quick encoder test on technical pairs/multi_model_comparison_report.md` | `scripts/legacy/v2_ablation/eval_bge_technical_compounds.py` |
| `results/legacy/v2_ablation/quick encoder test on technical pairs/multi_model_comparison_summary.csv` | `scripts/legacy/v2_ablation/eval_bge_technical_compounds.py` |

### 2.3 V7 Historical & Calibration Suite (`results/legacy/v7_legacy/` & `scripts/legacy/v7_legacy/`)

| Result File (`results/legacy/v7_legacy/` or `results/temp/`) | Corresponding Script (`scripts/legacy/v7_legacy/`) |
| :--- | :--- |
| `results/legacy/v7_legacy/v7_vs_baselines/v7_vs_baselines_results.csv` | `scripts/legacy/v7_legacy/run_v7_vs_baselines_comparison.py` |
| `results/legacy/v7_legacy/v7_vs_baselines/v7_vs_baselines_summary.md` | `scripts/legacy/v7_legacy/run_v7_vs_baselines_comparison.py` |
| `results/legacy/v7_legacy/v7_vs_baselines_archive/v7_vs_baselines_results.csv` | `scripts/legacy/v7_legacy/run_v7_vs_baselines_comparison.py` |
| `results/legacy/v7_legacy/v7_vs_baselines_archive/v7_vs_baselines_summary.md` | `scripts/legacy/v7_legacy/run_v7_vs_baselines_comparison.py` |
| `results/legacy/v7_legacy/v7_calibration/coverage_vs_salience_pool_results.csv` | `scripts/legacy/v7_legacy/run_v7_pool_sweep.py` |
| `results/legacy/v7_legacy/v7_calibration/coverage_vs_salience_pool_summary.md` | `scripts/legacy/v7_legacy/run_v7_pool_sweep.py` / `scripts/legacy/v7_legacy/make_summary.py` |
| `results/legacy/v7_legacy/v7_calibration/stage1_pool_results.csv` | `scripts/legacy/v7_legacy/run_v7_calibration_suite.py` |
| `results/legacy/v7_legacy/v7_calibration/v7_bailout_grid_results.csv` | `scripts/legacy/v7_legacy/run_v7_bailout_grid_sweep.py` |
| `results/legacy/v7_legacy/v7_calibration/v7_bailout_grid_summary.md` | `scripts/legacy/v7_legacy/run_v7_bailout_grid_sweep.py` |
| `results/legacy/v7_legacy/v7_calibration/v7_calibration_decision_log.md` | `scripts/legacy/v7_legacy/run_v7_calibration_suite.py` |
| `results/legacy/v7_legacy/v7_calibration/old_calibration/*.csv` | `scripts/legacy/v7_legacy/run_v7_calibration_suite.py` |
| `results/legacy/v7_legacy/weighting_ablation/weighting_ablation_summary.md` | `scripts/legacy/v7_legacy/run_weighting_expansion_ablation.py` |
| `results/legacy/v7_legacy/pos_weight_ablation/pos_ratio_grid_summary.md` | `scripts/legacy/v7_legacy/run_pos_ratio_grid_ablation.py` |
| `results/legacy/v7_legacy/v7_large_scale/profile_*.md` | `scripts/legacy/v7_legacy/profile_v7_10_benchmarks.py` / `scripts/legacy/v7_legacy/profile_v7_vram_forensics.py` / `scripts/legacy/v7_legacy/run_v7_streaming_large_benchmarks.py` |
| `results/temp/v7_smoke_test/smoke_test_summary.md` | `scripts/legacy/v7_legacy/smoke_test_v7.py` |
| `results/temp/anchor_encoding_diagnostic_*.csv` | `scripts/legacy/v7_legacy/run_v7_anchor_diagnostic.py` |
| `results/temp/bailout_experiment_*.csv` | `scripts/legacy/v7_legacy/run_v7_anchor_diagnostic.py` / `scripts/legacy/v7_legacy/test_bailout_speed.py` |
| `results/temp/bge_encode_bench_*.csv` | `scripts/legacy/v7_legacy/bench_bge_encode.py` |
| `results/temp/v7_pool_verification_*.csv` | `scripts/legacy/v7_legacy/run_v7_pool_verification.py` |

---

---

## 2. Infrastructure, Core Unit Tests & Data Tools Registry

The following files do not write result tables into `results/` or `temp/` directly, but are integral components of the test harness, dataset generation, or environment setup:

### 2.1 Core Regression & Parity Tests (`tests/`)
Run via `pytest` to assert mathematical invariants and module contracts:
- `tests/test_aho_corasick.py` — Exact string matching engine verification
- `tests/test_interval_merging.py` — 1D interval merging logic
- `tests/test_lucene_parity.py` — Exact Lucene BM25 score identity test
- `tests/test_vectorized_bm25_parity.py` — In-memory vectorized posting score parity
- `tests/test_v7_bailout_parity.py` — Boundary prefix lookup parity
- `tests/test_v7_bge_parity.py` — BGE dual-embedding parity
- `tests/test_metrics.py` — Metric definitions (Strict@K, MRR, Recalls)
- `tests/test_router.py` — BM25 cascade triage routing logic
- `tests/test_models.py` — Local LLM / Ollama server connectivity
- `tests/test_baselines.py` — Isolation testing of Lucene BM25, Dense BGE, SPLADE baselines
- `tests/test_pipeline_v2.py` — End-to-end integration test of Pipeline V2
- `tests/test_pipeline_v2_e2e_weighted.py` — End-to-end weighted aspect expansion test

### 2.2 Benchmark Dataset Generation (`scripts/benchmark_creation/`)
5-step synthetic dataset generation pipeline:
- `step1_chunking.py` — Hierarchical chunking
- `step2_seed_generation.py` — Seed QA generation
- `step3_query_paraphrasing.py` — Lexical gap injection
- `step4_global_recall.py` — Hybrid global recall
- `step5_oracle_filtering.py` — LLM-as-a-Judge annotation
- `build_fused_stress_benchmark.py`, `build_stress_benchmarks.py`, `check_multi_block_core.py`, `reinject_core_chunks.py`, `run_pipeline.py`, `verify_paper_inclusion.py`

### 2.3 External Data Adapters (`scripts/data_adapters/`)
Converts BEIR, BRIGHT, and FinanceBench datasets to standard benchmark formats:
- `convert_doc_level_benchmarks.py`, `convert_external_benchmarks.py`, `convert_retriever_doc_level_benchmarks.py`, `count_raw_queries.py`, `inspect_columns.py`

### 2.4 Asset & Environment Setup
- `scripts/setup_zaya.sh` — Vendor tool build script
- `scripts/download_arxiv.py`, `scripts/download_datasets.py`, `scripts/download_retriever_benchmarks.py` — Asset fetchers
- `scripts/prepare_global_assets.py`, `scripts/prepare_test_assets.py` — Vocabulary and index compilers
- `scripts/compute_p_optimality.py` — Mathematical optimality derivation utility
