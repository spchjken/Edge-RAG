---
description: Edge-RAG evaluator workflow for baselines and benchmarks.
---

# Edge-RAG Evaluator Workflow
**Role prefix:** `[Evaluator]`

## Pre-Work
1. Read all rules. Adopt [Evaluator] role.
2. Read docs/EVALUATION_METRICS.md.
3. Read draft.md §V for benchmark design.

## Implementation Order
1. `scripts/download_datasets.py` — fetch MS MARCO, prep data dirs.
2. `scripts/setup_zaya.sh` — build llama.cpp from PR #23112, download GGUF.
3. `src/baselines/bm25.py` — BM25 via rank_bm25.
4. `src/baselines/dense_rag.py` — BGE-m3 via FlagEmbedding + FAISS.
5. `src/baselines/llm_lingua.py` — LLMLingua-2 wrapper.
6. `src/evaluation/device_simulator.py` — VRAM fraction limiter.
7. `src/evaluation/metrics.py` — EM, F1, TTFT, Peak VRAM, Factual Consistency.
8. `src/evaluation/benchmark_runner.py` — orchestrate systems × profiles × datasets.
9. `scripts/run_benchmarks.py` — CLI for Table 1 & 2.
10. `scripts/run_ablations.py` — CLI for §5.3 ablations.

## Model Backends
- Qwen3.5-4B: `ollama run qwen3.5:4b`
- Gemma-4-E4B: `ollama run gemma4:e4b`
- ZAYA1-8B: `./vendor/llama.cpp/build/bin/llama-server` (from setup_zaya.sh)

## Post-Work
1. Update docs/EVALUATION_METRICS.md.
2. Output CSV to results/. Pass to [QA] via Orchestrator.
