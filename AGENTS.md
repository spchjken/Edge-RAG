# 🤖 AGENTS.md — AI Agent Operating Instructions (Edge-RAG)

Welcome to the **Edge-RAG** codebase. This document is the primary entry point for AI agents (OpenAI Codex, Claude Code, Cursor, Aider, etc.) to understand the project architecture, operational constraints, and mandatory rules.

---

## 1. Project Overview & Environment
- **Project Name:** Edge-RAG
- **Domain:** Extractive-Compression RAG / High-Speed Anchored Lexical-Semantic Retriever
- **Python Runtime:** **MANDATORY:** Always use the local virtual environment `.venv/bin/python3` to execute Python scripts or tests. Never use global/system Python.
- **Hardware Profile & Budgets:** Tested under consumer edge constraints (WSL2 Linux, 15 GiB RAM ceiling, NVIDIA GPU). All code must be strictly memory-safe and respect memory caps.

---

## 2. Mandatory Rules & Documentation Hierarchy

Before making changes, all agents MUST read and strictly adhere to the rules defined in `.agents/rules/`:

| Rule File | Scope & Mandatory Constraints |
|---|---|
| [`.agents/rules/00-agent-core.md`](file:///home/donghv/Projects/Edge-RAG/.agents/rules/00-agent-core.md) | **Core Project Invariants:** Local venv enforcement, terminal safety (NEVER use interactive commands like `nano`, `vim`, `cat >>`), process & terminal hang prevention (ephemeral subshells, detached background FDs, unbuffered I/O), Three-Strikes Halt rule, surgical edits, anti-steamrolling, and definition of done. |
| [`.agents/rules/01-architecture.md`](file:///home/donghv/Projects/Edge-RAG/.agents/rules/01-architecture.md) | **Module Boundaries & Tiered Architecture:** Tier 0 (`docs/ARCHITECTURE.md`), Tier 1 (module boundaries), Tier 2 (co-located `pathway_*.md`), PyTerrier baseline harness contracts, 15 GiB RAM limits. |
| [`.agents/rules/02-reproducibility.md`](file:///home/donghv/Projects/Edge-RAG/.agents/rules/02-reproducibility.md) | **Scientific Reproducibility:** Deterministic RNG seed locking (`--seed`), PyTorch VRAM measurement protocols (`torch.cuda.reset_peak_memory_stats()`), and results directory structure (`results/`). |

---

## 3. Workflows & Execution Procedures

Specialized agent execution playbooks are located in `.agents/workflows/`:
- [`.agents/workflows/evaluator-workflow.md`](file:///home/donghv/Projects/Edge-RAG/.agents/workflows/evaluator-workflow.md): Baseline benchmarking, PyTerrier harness execution, and Table 1/2 reproduction.
- [`.agents/workflows/pipeline-workflow.md`](file:///home/donghv/Projects/Edge-RAG/.agents/workflows/pipeline-workflow.md): Pipeline V2 implementation patterns, vocabulary builder, analyzer, and V7 aspect extractor.
- [`.agents/workflows/qa-workflow.md`](file:///home/donghv/Projects/Edge-RAG/.agents/workflows/qa-workflow.md): Unit and integration testing standards, pytest regression suites.
- [`.agents/workflows/orchestrator-workflow.md`](file:///home/donghv/Projects/Edge-RAG/.agents/workflows/orchestrator-workflow.md): End-to-end plan execution and verification.

---

## 4. Canonical Reference Documents

When modifying core components or checking evidence, consult:
1. **System Architecture:** [`docs/ARCHITECTURE.md`](file:///home/donghv/Projects/Edge-RAG/docs/ARCHITECTURE.md) — Canonical blueprint for V2 Retriever and PyTerrier Baseline Suite.
2. **Evaluation Metrics:** [`docs/EVALUATION_METRICS.md`](file:///home/donghv/Projects/Edge-RAG/docs/EVALUATION_METRICS.md) — Formal definitions of `Strict@K`, `nDCG@K` (with Table 2 exponential gains `BEIR_EXP_GAINS`), `MRR@K`, and query chunking invariance.
3. **PyTerrier Adoption & Large-Corpus Plan:** [`docs/pyterrier_adoption_plan.md`](file:///home/donghv/Projects/Edge-RAG/docs/pyterrier_adoption_plan.md) — Strategic integration design for PyTerrier and `ir_measures`.
4. **Results-to-Scripts Mapping:** [`scripts/results_scripts_mapping.md`](file:///home/donghv/Projects/Edge-RAG/scripts/results_scripts_mapping.md) — Authoritative mapping linking every result file under `results/` to its generator script.

---

## 5. Critical Engineering Principles for Edge-RAG

When developing or evaluating in this repository, always observe these rules:

1. **Large Corpora Scaling (5M+ Documents):**
   - NEVER load full raw corpora into in-memory Python lists or dictionaries (`BenchmarkLoader.load()` is only for small datasets).
   - ALWAYS stream documents line-by-line via `BenchmarkLoader.stream_corpus()` directly into `IterDictIndexer`.
   - Hard-cap JVM heap at 4 GiB (`pt.java.set_memory_limit(4096)`).
   - Configure Terrier's indexing buffer (`indexing.max.memory = 1073741824`) for disk flushes.
2. **Query Chunking vs. Candidate Truncation:**
   - NEVER truncate candidate document rankings (preserve full depth $K=1,000$).
   - Cap RAM and JNI overhead by chunking the query list (`chunk_size=200`). This is mathematically invariant for all evaluation metrics.
3. **No Steamrolling / Anti-Hallucination:**
   - Multi-step tasks require an upfront plan approved by the user.
   - If a command fails 3 consecutive times with the same error, **HALT** and diagnose root cause.
