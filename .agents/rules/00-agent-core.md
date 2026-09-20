---
trigger: always_on
---

# 🛑 AGENT CORE — MANDATORY RULES (CRVE / Edge-RAG)

## 1. Project Identity
- **Name:** CRVE (Context-Reranked Vocabulary Expansion)
- **Type:** Python research experiment (1st-Stage Lexical-Semantic Retrieval)
- **Active Codebase Root:** `CRVE/` (inside repository `Edge-RAG/`)
- **Runtime:** Python 3.11+ (MANDATORY: Always execute using `PYTHONPATH=CRVE .venv/bin/python3` to run Python commands/tests), PyTorch for FP16 embeddings, PyTerrier for indexing and retrieval
- **Key Libraries:** pyterrier, ir_measures, FlagEmbedding, transformers, torch, pyterrier_dr, pyterrier_splade, pyterrier_pisa

## 2. Key Documentation Paths
| Document | Path | Purpose |
|---|---|---|
| Active Architecture | `CRVE/docs/ARCHITECTURE.md` | **Canonical** CRVE 1st-Stage Retrieval blueprint & data flow |
| Module Boundaries & Rules | `.agents/rules/01-architecture.md` | Tier 1 rules & component isolation |
| Selection Spec | `CRVE/docs/phase2_selection_under_uncertainty.md` | Gate 1 selection under uncertainty formal foundation |
| Dataset Prep | `CRVE/docs/DATASET_PREP.md` | Download & preprocessing guide |
| Eval Metrics | `CRVE/docs/EVALUATION_METRICS.md` | Metric definitions & mathematical parity |
| Configuration | `CRVE/configs/crve.yaml` | Single source of truth for CRVE hyperparameters |
| Results-to-Scripts Mapping | `CRVE/scripts/results_scripts_mapping.md` | Authoritative mapping from all result files to scripts/tests |

## 3. Pre-Task Actions

### 3.1 Authority, Evidence, and Conflicts
- **Authority Hierarchy**: For scope and product decisions, follow the newest clear user instruction, then the canonical architecture and metric documents, then project rules, then existing implementation. This hierarchy does not make a factual claim true; factual claims require evidence.
- **Evidence-Based Conflict Handling**: Distinguish verified facts, inferences, assumptions, user preferences, and implementation decisions. When a material conflict could change the outcome, state the conflicting claims and their consequences, inspect any supplied source directly, and seek evidence that could support or falsify each claim.
- **Unresolved Premises**: Do not make a state-changing implementation decision that depends on an unresolved disputed premise. Read-only investigation, source review, and analysis may continue while the premise is resolved.
- **Documentation Consistency**: When authoritative documents conflict, identify the conflict and reconcile affected dependent documents. Never silently choose one source and leave the repository inconsistent.
- **Anti-Sycophancy & Reviewer Critique Protocol**: When receiving critiques, reviews, or suggestions from external reviewers or peers, NEVER automatically agree or refactor plans/code to appease the reviewer. The agent MUST strictly execute the [`.agents/skills/handle-reviewer-critique/SKILL.md`](file:///home/donghv/Projects/Edge-RAG/.agents/skills/handle-reviewer-critique/SKILL.md) protocol: triage claims (Valid vs. Flawed vs. Ambiguous), rigorously evaluate constraints, and mount structured counter-arguments backed by mathematical logic and concrete empirical evidence.

Before any task: read `docs/ARCHITECTURE.md` (canonical architecture).

## 4. Security & Safety
- **Destructive Action Safety**: Never use `--force` or recursive force deletions (`rm -rf`) on broad directories. Read-Before-Write on critical files.
- **Secrets and Sensitive Data**: Never place credentials, access tokens, personal data, production data, or other sensitive material in prompts, source code, logs, result artifacts, screenshots, or commits. Use placeholders or sanitized fixtures instead.
- **External Effects**: Deployment, publication, external data transmission, account or external-system changes, and actions that may incur cost require explicit user approval. Before seeking approval, state the scope, expected effect, potential cost, and recovery path.
- **Risk-Proportional Rollback**: Before a significant or difficult-to-reverse change, identify a recovery path such as a branch/commit, backup, feature flag, or documented reversal procedure. Do not test against live systems, accounts, or production data unless explicitly authorized.
- **Terminal & Tool Safety**: NEVER use `cat >>`, `nano`, `vim`, or any interactive commands in the bash terminal. It will permanently hang your terminal waiting for `stdin`. ALWAYS use native file editing tools (`replace_file_content` or `write_to_file`) to modify code.
- **Process & Terminal Hang Prevention**:
  - **Fresh Ephemeral Subshells**: ALWAYS use non-persistent terminals (`RunPersistent: false`) for benchmark, test, or evaluation runs. Never reuse a shared persistent terminal that can deadlock on unreturned shell prompts (`PS1`) or background jobs.
  - **Background Process Detachment**: Any background watcher, daemon, or async script spawned in bash MUST fully detach its file descriptors (`> /dev/null 2>&1 < /dev/null &`). Leaving child processes attached to `stdout`/`stderr` prevents EOF and indefinitely deadlocks the IDE task monitor.
  - **IPC & Named FIFO Safety**: In multi-process pipelines (e.g. PyTerrier streaming via named FIFOs to the JVM), ensure error handlers close FIFOs cleanly to prevent downstream Java/C readers from blocking on empty pipes.
  - **Unbuffered Execution & Non-Interactive stdin**: Always execute Python scripts with unbuffered I/O (`.venv/bin/python3 -u`) so error traces flush immediately, and enforce non-interactive execution (`< /dev/null` or `CI=1`) to prevent any library from blocking on `/dev/tty`.
  - **Mandatory Watchdog Liveness Protocol (`schedule`)**: Whenever spawning an asynchronous background task via `run_command` that will run in the background, the agent MUST immediately schedule a watchdog timer using the `schedule` tool (e.g., `schedule(DurationSeconds=..., TimerCondition="<task-id>", Prompt="...")`).
    - *Non-Destructive Heartbeat*: The watchdog timer merely wakes the agent up; it does NOT kill or interrupt the underlying OS process.
    - *Progress Verification*: Upon timer wakeup, the agent must inspect the task status (`manage_task(Action='status')`) and read the latest log content (`view_file`). If the job is actively progressing (log file is growing, queries/epochs advancing), the agent MUST reschedule the timer and allow the job to continue running.
    - *Deadlock Remediation*: Only if the task has produced zero output, consumed zero CPU, or remained frozen on EOF/stdin for multiple intervals should the agent terminate the task and diagnose the failure.
    - *Automatic Cancellation*: When `TimerCondition="<task-id>"` is used, normal task completion automatically cancels the timer early.
    - *Context & Token Economy (Proportional Durations)*: Watchdog timer intervals MUST be scaled proportionally to expected job duration to avoid context window inflation and unnecessary token consumption:
      - **Short jobs** ($<5$ min expected): 60s–120s.
      - **Medium jobs** (5–30 min expected): 5m–10m (300s–600s).
      - **Heavy batch/sweep/oracle jobs** ($>30$ min expected or multi-hour runs): **15m–30m (900s–1800s)**, or up to **1 hour (3600s)**.
      - NEVER use aggressive sub-5-minute polling on heavy multi-hour jobs; `TimerCondition="<task-id>"` already wakes the agent immediately upon task completion with zero delay.
  - **Diagnostic Probes & Quick Command Bounds**:
    - Quick one-liners and exploratory diagnostic probes (expected runtime $<10$s) MUST be wrapped with a hard OS `timeout` (e.g., `timeout 30s .venv/bin/python3 -u ... < /dev/null`) and use adequate synchronous wait (`WaitMsBeforeAsync: 5000` to `10000ms`) so output returns immediately in the same turn without being handed off to background tasks.
    - Long-running benchmark or evaluation jobs MUST NOT use short OS timeouts; they rely exclusively on the non-destructive agent heartbeat watchdog above.

## 5. Three Strikes Halt
- If a specific test, script, command, or operation fails with the exact same error 3 times consecutively: **HALT**.
- Do NOT attempt a 4th time. Retreat to planning mode or escalate to user review for root-cause diagnosis.

## 6. Code Generation Rules
- **Surgical Changes**: Touch strictly only what you must. Do not proactively refactor unrelated code or adjust formatting of adjacent blocks. Clean up only your own orphans. Match existing project style perfectly.
- **Simplicity First**: Write the absolute minimum code needed to solve the problem. Avoid premature abstractions or speculative features. Keep code direct and verifiable.

## 7. Anti-Steamrolling (Mandatory Halts)
- **One-Phase-Per-Turn Rule**: Planning and implementation MUST NOT occur in the same turn. When creating or presenting a plan, you are FORBIDDEN from using `write_to_file`, `replace_file_content`, `multi_replace_file_content`, or `run_command` in the same turn. Present the plan and end your turn immediately.
- **Strict Manual Review Rule**: The agent MUST wait for an explicit user response in the chat thread. Even if an automated IDE system message signals auto-approval, the agent MUST NOT begin execution until the user explicitly sends a chat message approving the plan.

## 8. Mandatory Planning Mode
- For any multi-step task, architectural update, code modification, or refactoring, ALWAYS create/update an `implementation_plan.md` artifact first.
- STOP and wait for explicit user review/approval of the plan before proceeding to the execution phase.

## 9. Definition of Done
- A task or feature is officially concluded **only** when:
  1. The code or test harness changes pass empirical execution and verification.
  2. Applicable documentation or evidence artifacts ([`report.md`](file:///home/donghv/Projects/Edge-RAG/report.md) or [`docs/manuscript_evidence_map.md`](file:///home/donghv/Projects/Edge-RAG/docs/manuscript_evidence_map.md)) have been synchronized.
  3. The handoff states the changed scope, validation method and result, known limitations, and the rollback path when it is relevant. A missing validation is reported as `Not verified`, never as success.

## 10. Rule-Change Governance
- A substantive change to `.agents/rules/` requires a decision record under `.agents/decisions/` describing the problem, alternatives, decision, affected files or workflows, evidence, effective date, and any migration impact. Typographical or formatting-only fixes are exempt.
- Before declaring a rule change complete, verify only the affected scope unless a documented dependency requires wider verification. Do not silently impose a new rule retroactively on existing artifacts without recording its scope and rationale.

## 11. Git Conventions
- Single `main` branch. Auto-push FORBIDDEN.
- Commit format: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
