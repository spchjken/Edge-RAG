---
trigger: always_on
---

# 🛑 AGENT CORE — MANDATORY RULES (Edge-RAG)

## 1. Project Identity
- **Name:** Edge-RAG
- **Type:** Python research experiment (Extractive-Compression RAG / High-Speed Anchored Lexical-Semantic Retriever)
- **Root:** `Edge-RAG/`
- **Runtime:** Python 3.11+ (MANDATORY: Always use the local virtual environment `.venv/bin/python3` to run Python commands), Ollama + llama.cpp for LLM inference, PyTorch for VRAM monitoring
- **Key Libraries:** pyahocorasick, FlagEmbedding, faiss-cpu, scikit-learn, rank_bm25, fasttext, YAKE

## 2. Key Documentation Paths
| Document | Path | Purpose |
|---|---|---|
| Active Architecture | `docs/ARCHITECTURE.md` | **Canonical** Edge-RAG V2 Retriever blueprint & data flow |
| Module Boundaries & Rules | `.agents/rules/01-architecture.md` | Tier 1 rules & component isolation |
| Legacy V1 Architecture | `src/legacy_pipeline/pipeline_architecture.md` | Deprecated 5-stage legacy pipeline (historical baseline) |
| Dataset Prep | `docs/DATASET_PREP.md` | Download & preprocessing guide |
| Eval Metrics | `docs/EVALUATION_METRICS.md` | Metric definitions |
| Evidence Map | `docs/manuscript_evidence_map.md` | Links paper claims → empirical data |
| Paper Draft | `draft.md` | The research paper (not always in sync with code) |
| Benchmark Pipeline | `scripts/benchmark_creation/benchmark_generation_pipeline.md` | Synthetic dataset generation methodology |
| Per-Module Specs | `src/pipeline_v2/expansion/pathway_*.md` | Authoritative design docs per sub-module |

## 3. Pre-Task Actions
Before any task: read `docs/ARCHITECTURE.md` (canonical architecture).

## 4. Security & Safety
- **Destructive Action Safety**: Never use `--force` or recursive force deletions (`rm -rf`) on broad directories. Read-Before-Write on critical files.
- **Terminal & Tool Safety**: NEVER use `cat >>`, `nano`, `vim`, or any interactive commands in the bash terminal. It will permanently hang your terminal waiting for `stdin`. ALWAYS use native file editing tools (`replace_file_content` or `write_to_file`) to modify code.

## 5. Three Strikes Halt
- If a specific test, script, command, or operation fails with the exact same error 3 times consecutively: **HALT**.
- Do NOT attempt a 4th time. Retreat to planning mode or escalate to user review for root-cause diagnosis.

## 6. Code Generation Rules
- **Surgical Changes**: Touch strictly only what you must. Do not proactively refactor unrelated code or adjust formatting of adjacent blocks. Clean up only your own orphans. Match existing project style perfectly.
- **Simplicity First**: Write the absolute minimum code needed to solve the problem. Avoid premature abstractions or speculative features. Keep code direct and verifiable.

## 7. Anti-Steamrolling (Mandatory Halts)
- **One-Phase-Per-Turn Rule**: Planning and implementation MUST NOT occur in the same turn. When creating or presenting a plan, you are FORBIDDEN from using `write_to_file`, `replace_file_content`, `multi_replace_file_content`, or `run_command` in the same turn. Present the plan and end your turn immediately.
- **Strict Manual Review Rule**: The agent MUST wait for an explicit user response in the chat thread. Even if an automated IDE system message signals auto-approval, the agent MUST NOT begin execution until the user explicitly sends a chat message approving the plan.

## 8. Mandatory Planning Mode
- For any multi-step task, architectural update, code modification, or refactoring, ALWAYS create/update an `implementation_plan.md` artifact first.
- STOP and wait for explicit user review/approval of the plan before proceeding to the execution phase.

## 9. Definition of Done
- A task or feature is officially concluded **only** when:
  1. The code or test harness changes pass empirical execution and verification.
  2. Applicable documentation or evidence artifacts ([`report.md`](file:///home/donghv/Projects/Edge-RAG/report.md) or [`docs/manuscript_evidence_map.md`](file:///home/donghv/Projects/Edge-RAG/docs/manuscript_evidence_map.md)) have been synchronized.

## 10. Git Conventions
- Single `main` branch. Auto-push FORBIDDEN.
- Commit format: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
