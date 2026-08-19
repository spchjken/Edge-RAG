---
description: Edge-RAG QA workflow for testing and validation.
---

# Edge-RAG QA Workflow

## Pre-Test Gate
1. Read all rules in `.agents/rules/`.
2. Verify implementation work is complete.

## Test Framework
- **Framework:** pytest
- **Location:** `tests/`
- **Command:** `python -m pytest tests/ -v`

## Test Inventory

### Unit Tests
- `test_aho_corasick.py` — Aho-Corasick pattern matching (empty input, zero matches, overlapping patterns, Unicode)
- `test_interval_merging.py` — Interval merge algorithm (adjacent, fully overlapping, single, empty list)
- `test_router.py` — Cascade routing decisions (boundary thresholds, all-bypass, all-discard, three-way triage)
- `test_metrics.py` — Known-answer EM/F1 calculations
- `test_models.py` — LLM client connectivity and response parsing
- `test_baselines.py` — BM25/Dense RAG/LLMLingua smoke tests

### Integration Tests
- `test_pipeline_query_expansion.py` — All 6 QE methods against fintech dataset (aspect coverage, keyword Jaccard, acronym retention, latency)
- `test_whole_pipeline.py` — Full combination matrix: 12 pipeline combinations (3 Aspect-Only + 3 Similar-KW extractors) × 2 routers, tested with LLM reranking
- `test_benchmark_final.py` — Hierarchical chunk benchmark using `benchmark_dataset_final.json` format (parent blocks + child chunks)

### Device Simulation
- `DeviceSimulator` (from `src/evaluation/device_simulator.py`) is integrated into pipeline tests for VRAM monitoring.
- GPU cache cleared between benchmark iterations via `simulator.clear_gpu_state()`.

## Error Handling
- Three Strikes Halt on same error.
- Trace failures to source before fixing.

## Post-Test
- Record test results and verify all assertions pass.
