# 🤖 AGENTS.md — AI Agent Operating Instructions (CRVE / Edge-RAG)

Welcome to the **CRVE** (Context-Reranked Vocabulary Expansion) codebase. This document is the primary entry point for AI agents (OpenAI Codex, Claude Code, Cursor, Aider, etc.) to understand the project architecture, operational constraints, and mandatory rules.

---

## 1. Project Overview & Active Codebase Invariant

- **Project Focus:** **CRVE 1st-Stage Retrieval** (Context-Reranked Vocabulary Expansion). The project focuses strictly on 1st-stage lexical-semantic retrieval, decoupling and deprecating legacy downstream E2E RAG modules (cascade routers, listwise LLM rerankers, late expansion).
- **SOLE ACTIVE CODEBASE:** All active development, tests, scripts, and documentation reside in the **[`CRVE/`](file:///home/donghv/Projects/Edge-RAG/CRVE)** directory.
- **Python Runtime & Execution Rule:**
  - **MANDATORY:** Always execute Python scripts and tests using the local virtual environment and `CRVE/` on `PYTHONPATH`:
    ```bash
    PYTHONPATH=CRVE .venv/bin/python3 <script_or_command>
    ```
  - **CRVE-Only Execution:** NEVER run scripts, harnesses, or tests against the legacy root `src/` or obsolete scripts. All executions must target `CRVE/src/`, `CRVE/scripts/`, and `CRVE/tests/`.
- **Hardware Profile & Budgets:** Tested under consumer edge constraints (WSL2 Linux, 15 GiB RAM ceiling, NVIDIA GPU). All code must be strictly memory-safe and respect memory caps.

---

## 2. Mandatory Rules & Documentation Hierarchy

Before making changes, all agents MUST read and strictly adhere to the rules defined in `.agents/rules/`:

| Rule File | Scope & Mandatory Constraints |
|---|---|
| [`.agents/rules/00-agent-core.md`](file:///home/donghv/Projects/Edge-RAG/.agents/rules/00-agent-core.md) | **Core Project Invariants:** `CRVE/` active root enforcement, local venv with `PYTHONPATH=CRVE`, terminal safety (NEVER use interactive commands like `nano`, `vim`, `cat >>`), process & terminal hang prevention (ephemeral subshells, detached background FDs, unbuffered I/O, watchdog liveness protocol via `schedule`, probing command timeouts), Three-Strikes Halt rule, surgical edits, anti-steamrolling, and definition of done. |
| [`.agents/rules/01-architecture.md`](file:///home/donghv/Projects/Edge-RAG/.agents/rules/01-architecture.md) | **Module Boundaries & Tiered Architecture:** Tier 0 ([`CRVE/docs/ARCHITECTURE.md`](file:///home/donghv/Projects/Edge-RAG/CRVE/docs/ARCHITECTURE.md)), Tier 1 (module boundaries: `crve/` and `evaluation/baselines/`), Tier 2 (co-located `pathway_*.md`), PyTerrier baseline harness contracts, 15 GiB RAM limits. |
| [`.agents/rules/02-reproducibility.md`](file:///home/donghv/Projects/Edge-RAG/.agents/rules/02-reproducibility.md) | **Scientific Reproducibility:** Deterministic RNG seed locking (`--seed`), PyTorch VRAM measurement protocols (`torch.cuda.reset_peak_memory_stats()`), and results directory structure (`results/`). |

---

## 3. Workflows & Execution Procedures

Specialized agent execution playbooks are located in `.agents/workflows/`:
- [`.agents/workflows/evaluator-workflow.md`](file:///home/donghv/Projects/Edge-RAG/.agents/workflows/evaluator-workflow.md): Baseline benchmarking, PyTerrier harness execution, and Table 1/2 reproduction.
- [`.agents/workflows/pipeline-workflow.md`](file:///home/donghv/Projects/Edge-RAG/.agents/workflows/pipeline-workflow.md): CRVE implementation patterns, vocabulary builder, analyzer, and Gate 1 candidate selection.
- [`.agents/workflows/qa-workflow.md`](file:///home/donghv/Projects/Edge-RAG/.agents/workflows/qa-workflow.md): Unit and integration testing standards, pytest regression suites (`PYTHONPATH=CRVE .venv/bin/python3 -m pytest CRVE/tests/...`).
- [`.agents/workflows/orchestrator-workflow.md`](file:///home/donghv/Projects/Edge-RAG/.agents/workflows/orchestrator-workflow.md): End-to-end plan execution and verification.

---

## 4. Canonical Reference Documents

When modifying core components or checking evidence, consult:
1. **System Architecture:** [`CRVE/docs/ARCHITECTURE.md`](file:///home/donghv/Projects/Edge-RAG/CRVE/docs/ARCHITECTURE.md) — Canonical blueprint for CRVE 1st-Stage Retrieval and PyTerrier Baseline Suite.
2. **Configuration Contract:** [`CRVE/configs/crve.yaml`](file:///home/donghv/Projects/Edge-RAG/CRVE/configs/crve.yaml) — Authoritative single source of truth for all CRVE hyperparameters.
3. **Evaluation Metrics:** [`CRVE/docs/EVALUATION_METRICS.md`](file:///home/donghv/Projects/Edge-RAG/CRVE/docs/EVALUATION_METRICS.md) — Formal definitions of `Strict@K`, official linear `nDCG@K`, supplemental Table 2 exponential gains `BEIR_EXP_GAINS`, `MRR@K`, and query chunking invariance.
4. **Selection Under Uncertainty:** [`CRVE/docs/phase2_selection_under_uncertainty.md`](file:///home/donghv/Projects/Edge-RAG/CRVE/docs/phase2_selection_under_uncertainty.md) — Formal mathematical foundation for Gate 1 selection and candidate proposers.
5. **Results-to-Scripts Mapping:** [`CRVE/scripts/results_scripts_mapping.md`](file:///home/donghv/Projects/Edge-RAG/CRVE/scripts/results_scripts_mapping.md) — Authoritative mapping linking every result file under `results/` to its generator script.

---

## 5. Critical Engineering Principles for CRVE

When developing or evaluating in this repository, always observe these rules:

1. **Large Corpora Scaling (5M+ Documents):**
   - NEVER load full raw corpora into in-memory Python lists or dictionaries (`BenchmarkLoader.load()` is only for small datasets).
   - ALWAYS stream documents line-by-line via `BenchmarkLoader.stream_corpus()` directly into `IterDictIndexer`.
   - Hard-cap JVM heap at 4 GiB (`pt.java.set_memory_limit(4096)`).
   - Configure Terrier's indexing buffer (`indexing.max.memory = 1073741824`) for disk flushes.
2. **Query Chunking vs. Candidate Truncation:**
   - NEVER truncate candidate document rankings (preserve full depth $K=1,000$).
   - Cap RAM and JNI overhead by chunking the query list (`chunk_size=200`). This is mathematically invariant for all evaluation metrics.
3. **Unified Baselines Namespace (`CRVE/src/evaluation/baselines/`):**
   - All 8 baselines live together under `evaluation.baselines`:
     - 6 Classical PyTerrier baselines (`pyterrier_harness.py`, `pyterrier_qe.py`): BM25, BM25+RM3, BM25+Bo1, DPH, DPH+Bo1, DPH+RM3.
     - 2 Neural baselines: Dense BGE-small-en-v1.5 (`dense_rag.py`), SPLADE-v3 (`splade.py`).
4. **No Steamrolling / Anti-Hallucination:**
   - Multi-step tasks require an upfront plan approved by the user.
   - If a command fails 3 consecutive times with the same error, **HALT** and diagnose root cause.
5. **Anti-Sycophancy & Reviewer Critique Defense (`handle-reviewer-critique`):**
   - NEVER reflexively concede to external reviewer opinions or speculative suggestions.
   - When receiving critiques, reviews, or comments, the agent MUST immediately load and execute the [`.agents/skills/handle-reviewer-critique/SKILL.md`](file:///home/donghv/Projects/Edge-RAG/.agents/skills/handle-reviewer-critique/SKILL.md) protocol.
   - For every claim, perform a rigorous 3-bucket triage (Valid vs. Flawed vs. Ambiguous).
   - Actively defend sound designs by mounting structured counter-arguments backed by mathematical logic and concrete repository evidence (e.g., WSL2 15 GiB RAM limits, pool audit Parquets, IR theorems).
