# Implementation Plan — Preserve CRVE Design Refinement Notes

## Status

Approved by the user on 2026-09-09; executed and verified.

## Scope

Create a standalone design-decision note for the current discussion of
Context-Reranked Vocabulary Expansion (CRVE). Do not alter the active
architecture or the broader research plan.

## Steps

1. Record the reasoning path from archived V7-style vocabulary expansion to a
   bounded term-context memory.
2. State the ambiguity problem and the joint nDCG/recall objective.
3. Preserve the candidate scoring, sampling, retrieval-injection, and
   experiment ideas discussed in this session.
4. Check the new note for internal links and working-tree scope.

## Intended Artifact

- `docs/crve_design_refinement_notes.md`

---

# Implementation Plan — Agent Rule Adoption

## Status

Completed and verified on 2026-09-10.

## Scope

Adopt the ten transferable governance, evidence, safety, reproducibility, and
documentation-consistency rules proposed from `.agents/template rules/`.
Do not import curriculum-specific requirements or modify production code,
architecture, benchmarks, or results.

## Steps

1. Add core rules for evidence-based conflict handling, external effects and
   secrets, risk-proportional rollback, authority and documentation
   consistency, rule-change decision records, and evidence-bearing completion.
2. Add the Tier 0/1/2 canonical-document reconciliation rule to the
   architecture rules.
3. Add `Pass` / `Fail` / `Not verified` semantics, claim-to-evidence
   traceability, and currency requirements to the reproducibility rules.
4. Verify that the final rules are internally consistent, preserve existing
   Edge-RAG safety constraints, and contain no learning-specific policy.

## Intended Artifacts

- `.agents/rules/00-agent-core.md`
- `.agents/rules/01-architecture.md`
- `.agents/rules/02-reproducibility.md`

## Verification

- Confirmed all ten adopted rule topics appear in the three target rule files.
- Confirmed the rule-change decision record exists under `.agents/decisions/`.
- Ran `git diff --check` successfully; no whitespace errors were reported.
