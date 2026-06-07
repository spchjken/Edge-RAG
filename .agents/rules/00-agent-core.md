---
trigger: always_on
---

# 🛑 AGENT CORE — MANDATORY RULES (Edge-RAG)

## 1. Project Identity
- **Name:** Edge-RAG
- **Type:** Python research experiment (Extractive-Compression RAG)
- **Root:** `Edge-RAG/`
- **Runtime:** Python 3.11+, Ollama + llama.cpp for LLM inference, PyTorch for VRAM monitoring

## 2. Role Adoption
- Orchestrator MUST tag every task in `task.md` with role prefix (`[Orchestrator]`, `[Pipeline]`, `[Evaluator]`, `[QA]`).
- First checklist item for any role: read `Edge-RAG/.agents/rules/` then the corresponding workflow.
- Physical Tool Call Enforcement: MUST use list_dir/view_file on rules. Relying on memory is forbidden.
- Role Transition Protocol: `[Pipeline]`/`[Evaluator]` → `[QA]` requires Orchestrator QA Brief.

## 3. Key Documentation Paths
| Document | Path | Purpose |
|---|---|---|
| Architecture | `docs/ARCHITECTURE.md` | Pipeline design & data flow |
| Dataset Prep | `docs/DATASET_PREP.md` | Download & preprocessing guide |
| Eval Metrics | `docs/EVALUATION_METRICS.md` | Metric definitions |
| Paper Draft | `draft.md` | The research paper |

## 4. Pre-Task Actions
Before any task: read `docs/ARCHITECTURE.md` and the relevant role workflow.

## 5. Security & Safety
- Never use `rm -rf` on broad directories. Read-Before-Write on critical files.
- Three Strikes Halt on repeated identical failures.

## 6. Code Generation Rules
- Surgical changes only. Simplicity first. No premature abstractions.

## 7. Anti-Steamrolling
- Planning and implementation MUST NOT occur in the same turn.

## 8. Git Conventions
- Single `main` branch. Auto-push FORBIDDEN.
- Commit format: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
