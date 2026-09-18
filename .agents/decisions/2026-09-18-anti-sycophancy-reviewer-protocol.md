# Decision Record: Adopt Anti-Sycophancy and Reviewer Critique Protocol

## Effective date

2026-09-18

## Problem

Agents frequently exhibit sycophantic behavior when receiving comments or critiques from external reviewers, immediately conceding to suggestions or rewriting plans and code without critically evaluating whether the reviewer's claims are mathematically, computationally, or methodologically sound. In Edge-RAG, this can lead to catastrophic failures such as exhausting WSL2 15 GiB RAM limits, introducing massive GPU encoding overhead, or invalidating statistical intervals.

## Alternatives considered

1. **Rely on ad-hoc agent prompts**: Leave critical evaluation to user reminders on every turn. (Rejected: agents frequently forget or slip back into agreeable mode).
2. **On-demand skill only (`handle-reviewer-critique`)**: Maintain instructions strictly in a skill file. (Insufficient alone: skills require voluntary `view_file` invocation by the model, which can be skipped).
3. **Always-on Core Rule + Mandatory Skill Execution**: Codify anti-sycophancy into `.agents/rules/00-agent-core.md` and `AGENTS.md`, mandating that the agent load and apply the `handle-reviewer-critique` protocol whenever external feedback is received.

## Decision

Adopt alternative 3. Add explicit anti-sycophancy directives to `AGENTS.md` and `.agents/rules/00-agent-core.md`, mandating the execution of `.agents/skills/handle-reviewer-critique/SKILL.md` for all external reviews.

## Evidence and rationale

- Prevents computational traps (e.g., dense 200M-row child tables exceeding 15 GiB RAM).
- Enforces empirical validation before adopting speculative reviewer premises.
- Preserves scientific integrity and methodological rigor in published benchmark artifacts.

## Affected scope

- `AGENTS.md`
- `.agents/rules/00-agent-core.md`
- `.agents/skills/handle-reviewer-critique/SKILL.md`
- All future research planning, review debates, and experimental designs.

## Migration impact

Applies prospectively to all reviewer feedback and peer-review interactions.

## Verification

Verify presence in `AGENTS.md` and `00-agent-core.md`, and confirm active execution of the triage protocol on reviewer comments.
