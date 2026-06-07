---
description: Edge-RAG QA workflow for testing and validation.
---

# Edge-RAG QA Workflow
**Role prefix:** `[QA]`

## Pre-Test Gate
1. Read all rules. Adopt [QA] role.
2. Verify [Pipeline] or [Evaluator] work is complete.
3. Read QA Brief from Orchestrator. HALT if missing.
4. Write `### QA Checklist` in task.md.

## Test Framework
- **Framework:** pytest
- **Location:** `tests/`
- **Command:** `python -m pytest tests/ -v`

## Test Categories
### Unit Tests (Pipeline)
- Aho-Corasick: empty input, zero matches, overlapping patterns, Unicode.
- Interval Merging: adjacent, fully overlapping, single, empty list.
- Router: boundary thresholds, all-bypass, all-rerank edge cases.

### Integration Tests (Evaluation)
- Metrics: known-answer EM/F1 calculations.
- Device simulator: verify VRAM cap enforced.

### Stress Tests
- Max-length document → pipeline, verify no OOM under simulated 8GB.
- VRAM overflow protection triggers at N_max boundary.

## Error Handling
- Three Strikes Halt on same error.
- Trace failures to source before fixing.

## Post-Test
- Pass control back to [Orchestrator].
