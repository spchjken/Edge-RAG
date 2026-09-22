# Decision Record: Brand-New Planning Artifacts per Review Round

**Date:** 2026-09-22  
**Status:** ACCEPTED  
**Decision Owner:** Edge-RAG Core Architecture & Governance  

---

## 1. Problem Statement
During iterative multi-round reviews (e.g. peer review, external referee critique, or multi-stage evaluation planning), overwriting an existing planning artifact (such as `implementation_plan.md`) causes loss of earlier review history, historical rationale, and sequential provenance. Reviewers and users cannot easily trace how a plan evolved across rounds or compare earlier proposals against later amendments.

---

## 2. Alternatives Considered
1. **Single Overwritten Artifact (`implementation_plan.md`):**
   - *Pros:* Keeps only one plan file in the artifact directory.
   - *Cons:* Destroys historical context; makes it impossible to inspect earlier review consensus or verify whether a critique was previously addressed.
2. **Single Monolithic Document with Appended Sections:**
   - *Pros:* Retains history in one file.
   - *Cons:* Files become excessively large, increasing token overhead and making it difficult for reviewers to identify the active, authoritative scope.
3. **Brand-New Planning Artifact per Review Round (Chosen):**
   - *Pros:* Each review round has an immutable, dedicated record with a clear version/round identifier. Historical context is 100% preserved.
   - *Cons:* Requires generating unique artifact filenames.

---

## 3. Decision
In [`.agents/rules/00-agent-core.md`](file:///home/donghv/Projects/Edge-RAG/.agents/rules/00-agent-core.md) (Section 8: Mandatory Planning Mode), mandate that in each review or feedback round, whenever creating or revising an implementation plan, the agent MUST create a brand-new artifact with a unique filename (e.g., `..._plan_round2.md`, `..._plan_round3.md`) instead of overwriting an existing planning artifact.

---

## 4. Affected Scope & Workflows
- [`.agents/rules/00-agent-core.md`](file:///home/donghv/Projects/Edge-RAG/.agents/rules/00-agent-core.md) (Section 8)
- All agent planning workflows in `.agents/workflows/`
- Effective immediately on 2026-09-22.
