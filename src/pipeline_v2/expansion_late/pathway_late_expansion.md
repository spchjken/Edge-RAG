# Pathway Specification: LateExpansionV2 (`pathway_late_expansion.md`)

## Overview
Restores full uncompressed target chunk texts for final LLM generation while enforcing the VRAM budget limit ($N_{\text{max}} \le 10$).

## Rules
- Enforces $N_{\text{max}} \le 10$ chunk budget.
- Formats context blocks with clear document separators.
- Passes prompt to `LLMClient` for final answer generation.
