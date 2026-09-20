---
trigger: always_on
---

# 🔬 REPRODUCIBILITY RULES (Edge-RAG)

## 1. Deterministic Execution
- Every run MUST accept a --seed argument.
- Lock random, numpy, torch seeds at entry point.

## 2. Environment Logging
- Log: CPU model, GPU name, driver version, CUDA version,
  PyTorch version, Ollama version, Python version.
- Store as JSON header in every results CSV.

## 3. VRAM Measurement
- Use torch.cuda.max_memory_allocated() after torch.cuda.reset_peak_memory_stats().
- Clear GPU cache between benchmark iterations.
- Record peak VRAM per-run, not averaged.

## 4. Timing
- TTFT: from query submission to first generated token.
- Use torch.cuda.synchronize() before timing GPU operations.
- Report median over 3+ runs with std deviation.

## 5. Results Integrity
- Raw results → results/ as timestamped CSVs / JSONs.
- Never overwrite previous results.
- **Validation Status**: Record each material validation as `Pass`, `Fail`, or `Not verified`. `Pass` requires directly inspectable evidence. `Fail` means the required evidence was checked and did not meet the condition. `Not verified` means evidence or access is unavailable; record what is missing, who or what must provide it, and the precise re-verification scope. Both `Fail` and `Not verified` block a claim of successful validation.
- **Claim-to-Evidence Traceability**: Every material benchmark, result, or manuscript claim MUST identify its generating script or command, exact result artifact, configuration and environment, and verification scope. Keep these links synchronized with `CRVE/scripts/results_scripts_mapping.md`.
- Results directory structure:
  - `results/pyterrier_baselines/` — 6 classical baselines across 25 BEIR & BRIGHT datasets
  - `results/pyterrier_qe_baselines/` — Sparse lexical QE baselines (BGE_Vocab_QE, LLM_Q2E_ZS)
  - `results/gate1_selection/` — Phase 2 Gate 1 candidate selection under uncertainty results
  - `results/pool_oracle_isolation/` — Stage 1 vocabulary pool oracle isolation results
  - `results/legacy/` — Archived legacy runs

## 6. Reporting
- `report.md` in project root serves as the weekly progress artifact.
- `for_review/` contains frozen artifacts and reports for reviewer verification.

## 7. Currency of External Facts
- Claims about external APIs, models, dependencies, pricing, hardware, policies, or tools that can change over time MUST cite a primary source and record the checked date and relevant version where available.
- Separate durable technical principles from version-specific instructions so volatile material can be updated independently.
- If current information cannot be verified, state the limitation and mark the claim `Not verified`; do not present an inference as a fact.
