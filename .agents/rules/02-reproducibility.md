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
- Raw results → results/ as timestamped CSVs.
- Never overwrite previous results.
