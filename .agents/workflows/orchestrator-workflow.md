---
description: Edge-RAG orchestrator for planning and verification.
trigger: always_on
---

# Edge-RAG Orchestrator Workflow

## Pre-Planning
1. Read all rules in `.agents/rules/`.
2. Read `docs/ARCHITECTURE.md` (canonical Edge-RAG V2 architecture).
3. Read `configs/pipeline_v2.yaml` for authoritative active parameters.

## Planning
1. Understand requirements.
2. Generate plan in `implementation_plan.md`.
3. Break into atomic, verifiable checklist items.

## Validation Gate
- `docs/ARCHITECTURE.md` matches implemented pipeline.
- `src/pipeline_v2/**/pathway_*.md` docs match their corresponding module implementations.
- `configs/pipeline_v2.yaml` matches actual code defaults.
- `docs/manuscript_evidence_map.md` artifact paths are valid.
- `results/` outputs complete before reporting.
