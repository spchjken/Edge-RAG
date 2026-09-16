# Decision Record: Proportional Watchdog Timer Intervals & Context Token Economy

## Effective Date

2026-09-16

## Problem

Under the initial Watchdog Liveness Protocol (`00-agent-core.md`), background evaluation and oracle harnesses (such as multi-corpus evaluations taking 4+ hours) were polled aggressively at 60s, 120s, and 240s intervals. While this guaranteed active monitoring and hang detection, each turn triggered full context re-ingestion, assistant turn generation, and log inspection, leading to unnecessary token burning and rapid context window inflation.

## Alternatives Considered

1. **Keep sub-5-minute polling for all tasks**: Maximizes real-time progress visibility at the expense of heavy token consumption. (Rejected: burns thousands of context tokens per hour during normal, uninterrupted execution).
2. **Remove watchdogs for long runs**: Allow long runs to run without timers. (Rejected: reintroduces the silent deadlock / unreturned EOF failure mode where jobs hang for 5+ hours without agent or user awareness).
3. **Proportional Timer Durations & Early Cancellation**:
   - Calibrate watchdog intervals strictly to expected execution duration.
   - For multi-hour batch runs and large corpus sweeps, set intervals to **15–30 minutes (900s–1800s)** or up to **1 hour (3600s)**.
   - For medium tasks (5–30 min), use **5–10 minutes (300s–600s)**.
   - For short tasks (<5 min), use **1–2 minutes (60s–120s)**.
   - Rely on `TimerCondition="<task-id>"`, which guarantees immediate early wake-up upon task completion without waiting for the timer to expire.

## Decision

Adopt Alternative 3. Amend `.agents/rules/00-agent-core.md` Section 4 to formally define proportional watchdog durations and prohibit token-burning micro-polling on long-running jobs.

## Evidence and Rationale

- In Phase B (`task-8543`), the harness ran smoothly for 5+ hours evaluating 1.2M variants across 4 datasets. Micro-polling every 2–4 minutes generated over 15 redundant turns with zero intervention required.
- `schedule(DurationSeconds=1800, TimerCondition="<task-id>")` automatically wakes up the instant the task exits (code 0 or error), so a 30-minute timer introduces zero latency on task completion, while reducing idle turn overhead by ~85%.

## Affected Scope

- `.agents/rules/00-agent-core.md` (Section 4: Mandatory Watchdog Liveness Protocol)
- Future execution of long-running evaluations and benchmarks across all agents.

## Migration Impact

Applies immediately to all subsequent background task invocations.
