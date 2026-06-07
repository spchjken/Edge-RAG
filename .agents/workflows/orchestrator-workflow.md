---
description: Edge-RAG orchestrator for planning, delegation, and verification.
trigger: always_on
---

# Edge-RAG Orchestrator Workflow
**Role prefix:** `[Orchestrator]`

## Pre-Planning
1. Read all rules in `Edge-RAG/.agents/rules/`.
2. Read `docs/ARCHITECTURE.md`.
3. Read `draft.md` §IV–V for methodology & evaluation.

## Planning
1. Understand requirements.
2. Generate plan in `task.md` and `implementation_plan.md`.
3. Delegate with atomic, verifiable checklist items.
4. First item for every role: read rules + workflow.

## Implementation → QA Handoff
Before [QA]:
1. Verify all [Pipeline]/[Evaluator] items marked [x].
2. Write `### QA Brief` with: files changed, what to break (≥2), blast radius.

## Validation Gate
- docs/ARCHITECTURE.md matches implemented pipeline.
- configs/ YAML matches actual code defaults.
- results/ outputs complete before reporting.
