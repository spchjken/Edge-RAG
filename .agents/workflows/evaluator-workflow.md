---
description: Edge-RAG evaluator workflow for baselines and benchmarks.
---

# Edge-RAG Evaluator Workflow

## Pre-Work
1. Read all rules in `.agents/rules/`.
2. Read `docs/EVALUATION_METRICS.md`.
3. Read `draft.md` §V for benchmark design (note: draft may lag behind code).

## Evaluation Modules (`src/evaluation/`)
- `metrics.py` — EM, F1, TTFT, Peak VRAM, Factual Consistency
- `device_simulator.py` — VRAM fraction limiter for simulated edge profiles
- `benchmark_runner.py` — Orchestrate systems × profiles × datasets
- `evaluate_router.py` — Cascade Router threshold sensitivity (Strict/Balanced/Permissive)
- `query_expansion_ablation.py` — Cross-method QE comparison (aspect coverage, keyword Jaccard, acronym retention)

## Baselines (`src/baselines/`)
- `bm25.py` — BM25 via rank_bm25
- `dense_rag.py` — BGE-m3 via FlagEmbedding + FAISS
- `llm_lingua.py` — LLMLingua-2 wrapper

## Benchmark Creation (`scripts/benchmark_creation/`)
5-step synthetic dataset generation pipeline:
1. `step1_chunking.py` — Hierarchical parent blocks + child chunks
2. `step2_seed_generation.py` — LLM query + golden answer generation
3. `step3_query_paraphrasing.py` — Vocabulary gap injection
4. `step4_global_recall.py` — Offline brute-force recall pass
5. `step5_oracle_filtering.py` — LLM-as-a-Judge annotation

See `benchmark_generation_pipeline.md` for full methodology.

## Model Backends
- Qwen3.5-2B: `ollama run qwen3.5:2b`
- Qwen3.5-4B: `ollama run qwen3.5:4b`
- Gemma-4-E2B: `ollama run gemma4:e2b`
- Gemma-4-E4B: `ollama run gemma4:e4b`
- ZAYA1-8B: `./vendor/llama.cpp/build/bin/llama-server` (from setup_zaya.sh)

## Primary Evaluation Approach
- **Combination Matrix Testing:** All QE methods × all Lexical Search variants × all Routers are tested as a full Cartesian product. See `tests/test_whole_pipeline.py` and `tests/test_benchmark_final.py`.
- Results output to `results/pipeline_combinations/` as timestamped JSON.

## CLI Scripts
- `scripts/run_benchmarks.py` — CLI for Table 1 & 2
- `scripts/run_ablations.py` — CLI for §5.3 ablations
- `scripts/download_datasets.py` — Fetch datasets, prep data dirs
- `scripts/setup_zaya.sh` — Build llama.cpp, download GGUF

## Post-Work
1. Update `docs/EVALUATION_METRICS.md` if metrics changed.
2. Update `docs/manuscript_evidence_map.md` to link claims → new evidence.
3. Output CSV/JSON to `results/`. Never overwrite previous results.
