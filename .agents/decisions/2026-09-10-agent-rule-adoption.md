# Decision Record: Adopt Transferable Agent Governance Rules

## Effective date

2026-09-10

## Problem

The existing Edge-RAG agent rules defined strong operational safety,
architecture, and reproducibility requirements, but did not explicitly cover
evidence-based conflict handling, external effects, validation uncertainty,
or documentation reconciliation.

## Alternatives considered

1. Retain the existing rules unchanged.
2. Copy the curriculum-oriented template rule set in full.
3. Adopt only the transferable engineering and research-governance rules.

## Decision

Adopt alternative 3. Add ten focused rules covering authority and conflicts,
secrets and external effects, rollback, evidence-bearing completion,
rule-change records, canonical-document reconciliation, validation status,
claim traceability, and time-sensitive external facts.

## Evidence and rationale

- `.agents/template rules/README.md` supplies the conflict-resolution and
  rule-governance patterns.
- `.agents/template rules/quality-gates.md` supplies the explicit validation
  status semantics and evidence requirement.
- `.agents/template rules/safety-and-currency.md` supplies external-effects,
  rollback, secret-handling, and currency patterns.
- Course-specific requirements were excluded because they do not apply to an
  Edge-RAG research codebase.

## Affected scope

- `.agents/rules/00-agent-core.md`
- `.agents/rules/01-architecture.md`
- `.agents/rules/02-reproducibility.md`
- Future agent execution, architecture documentation, and empirical-result
  reporting.

## Migration impact

The rules apply prospectively. Existing artifacts are not retroactively
invalidated; update them only when they are next modified or when a separate
decision record identifies a required correction.

## Verification

Review the amended rules for the required ten adoptions, preservation of the
existing Edge-RAG constraints, and absence of curriculum-specific policy.
