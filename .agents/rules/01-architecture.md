---
trigger: always_on
---

# 🏗️ MODULE BOUNDARIES (Edge-RAG)

## 1. Pipeline (`src/pipeline/`)
- Pure algorithmic code. Dependencies: pyahocorasick, standard library.
- ONLY query_expansion.py and late_expansion.py may call the LLM client.
- No torch imports except in late_expansion.py (VRAM guard).

## 2. Baselines (`src/baselines/`)
- Fully isolated. Each baseline is self-contained.
- May import torch, transformers, FlagEmbedding, rank_bm25.
- MUST NOT import from src/pipeline/.

## 3. Evaluation (`src/evaluation/`)
- Orchestrates pipeline + baselines. Reads configs from configs/.
- Outputs results to results/ as CSV + Markdown.

## 4. Utils (`src/utils/`)
- Shared: llm_client.py (OpenAI-compatible wrapper), helpers.py.
- llm_client.py auto-detects backend (Ollama vs llama-cpp) from configs/models.yaml.
- No domain-specific logic in utils.

## 5. Configs (`configs/`)
- All hyperparameters in YAML. No magic numbers in source code.
- models.yaml: per-model backend, endpoint, tags, context windows.
- thresholds.yaml: τ_cont, τ_scat, window L, N_max.
- hardware_profiles.yaml: VRAM fractions for simulated profiles.

## 6. Vendor (`vendor/`)
- Custom-built external tools (llama.cpp for ZAYA1-8B).
- Built via scripts/setup_zaya.sh. Never committed to git.
- Add vendor/ to .gitignore.
