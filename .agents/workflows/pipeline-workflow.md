---
description: Edge-RAG pipeline implementation workflow.
---

# Edge-RAG Pipeline Workflow
**Role prefix:** `[Pipeline]`

## Pre-Work
1. Read all rules. Adopt [Pipeline] role.
2. Read docs/ARCHITECTURE.md.
3. Read draft.md §IV for mathematical specifications.

## Implementation Order
1. `src/utils/llm_client.py` — Generic OpenAI-compatible wrapper (Ollama + llama-cpp).
2. `src/pipeline/query_expansion.py` — §4.2: zero-shot expansion → weighted anchors K.
3. `src/pipeline/aho_corasick.py` — §3.3: build automaton from K, scan chunks.
4. `src/pipeline/interval_merging.py` — §4.3 Algorithm 1: sort + merge overlapping windows.
5. `src/pipeline/router.py` — §4.3: ρ_cont, ρ_scat, apply τ thresholds from configs.
6. `src/pipeline/late_expansion.py` — §4.4: index mapping, VRAM overflow protection.

## Code Standards
- Read thresholds from configs/thresholds.yaml, never hardcode.
- Type hints on all public functions.
- Docstrings referencing paper sections (e.g., "See §4.3 Eq. 5").

## Post-Work
1. Update docs/ARCHITECTURE.md.
2. Pass to [QA] via Orchestrator.
