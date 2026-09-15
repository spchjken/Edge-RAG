# Decision Record: Agent Watchdog Liveness Protocol & Terminal Hang Prevention

## Effective Date

2026-09-15

## Problem

During interactive and diagnostic evaluation, background commands (`run_command`) can deadlock or fail to signal EOF to the subshell (e.g. library imports, JVM JNI bindings, or stdin waits on `/dev/tty`). Because the agent architecture relies on passive reactive wakeups upon process completion, a stalled background process leaves the agent indefinitely suspended in the event loop (evidenced by a >5-hour hang on a quick probe). Conversely, hard kill timers could prematurely terminate legitimate long-running benchmark and evaluation jobs.

## Alternatives Considered

1. **Rely solely on user intervention**: Reject automated monitoring and rely on the user to manually detect and cancel stalled jobs in chat. (Rejected: wastes hours of idle compute and session time).
2. **Apply blanket OS-level timeouts (`timeout 10m`) to all commands**: Impose hard process kills across all commands. (Rejected: violently interrupts legitimate multi-hour evaluation and indexing runs across large datasets).
3. **Dual-Tiered Architecture (Watchdog Liveness Heartbeat + Bounded Diagnostic Probes)**:
   - Quick one-liners and exploratory diagnostic probes ($<10$s expected runtime) enforce hard Linux `timeout` and adequate synchronous wait (`WaitMsBeforeAsync`).
   - Long-running jobs enforce mandatory one-shot/recurring watchdog timers via `schedule` (`schedule(DurationSeconds=..., TimerCondition=...)`). When fired, the agent inspects process status and log growth; if the job is actively progressing, the timer is extended without interrupting the job. Stalled/deadlocked processes are detected, diagnosed, and terminated.

## Decision

Adopt Alternative 3. Amend `.agents/rules/00-agent-core.md` Section 4 to formally define:
1. Mandatory Watchdog Liveness Heartbeats via `schedule` for all asynchronous background tasks.
2. Non-destructive progress inspection upon timer wakeups (inspect logs/CPU before intervening; never blindly kill progressing jobs).
3. Hard `timeout` wrapping and adequate synchronous wait bounds for quick diagnostic and inspection commands.
4. Prohibition of shared stale persistent terminals for benchmark and test execution.

## Evidence and Rationale

- Empirical occurrence: `task-7799` stalled for 5h 55m without reaching EOF or waking the agent, because no watchdog timer was registered.
- In contrast, long-running jobs (e.g., PyTerrier 20-corpus sweeps) require hours of continuous execution and must not be aborted mid-run by hard timeouts.
- The `schedule` tool provides safe agent wakeup without affecting the target OS process, and cancels early automatically upon task completion when `TimerCondition` is bound to the task ID.

## Affected Scope

- `.agents/rules/00-agent-core.md` (Section 4: Security & Safety)
- `AGENTS.md` (Section 2 summary table)
- All future agent background task and terminal executions across the repository.

## Migration Impact

Applies prospectively to all background command executions. Existing completed results and scripts are not retroactively modified.

## Verification

Verify amended rule text in `.agents/rules/00-agent-core.md` and `AGENTS.md` for exact coverage of the dual-tier monitoring pattern and watchdog non-interruption semantics.
