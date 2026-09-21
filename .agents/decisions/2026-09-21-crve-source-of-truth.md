# CRVE experiment configuration authority

- **Date/effective scope:** 2026-09-21; current CRVE first-stage retrieval documentation and future claims. Historical result artifacts are unchanged.
- **Problem:** Agent rules and `AGENTS.md` called generic `CRVE/configs/crve.yaml` the source of truth for *all* hyperparameters, although the tested pool/Gate 1 protocol is frozen separately and the generic orchestrator still exposes a live-PPMI route.
- **Alternatives:** Force generic runtime defaults onto frozen experiments; replace all runtime code and configs now; or distinguish per-run frozen evidence from generic runtime defaults. The first falsifies provenance, and the second is outside this documentation-only task.
- **Decision:** `CRVE/docs/ARCHITECTURE.md` owns scope and stage status. Each experiment's frozen config and run manifest own its tested settings and results. `crve.yaml` owns only generic orchestrator defaults. For current Phase 2.1a Gate 1, use `CRVE/configs/gate1_phase2_1a.yaml` and the immutable `for_review/selection_phase/gate_1/run_2/` evidence.
- **Affected references:** `.agents/rules/01-architecture.md`, `AGENTS.md`, `CRVE/docs/ARCHITECTURE.md`, `CRVE/configs/crve.yaml`, and dependent CRVE research docs.
- **Evidence:** Frozen Gate 1 config records DF >= 2, CF >= 3, `RRF_Core3` with `PPMISidecar`, and deployable budget 200. The generic config records `min_df: 1` and a live lexical-PPMI channel; the current orchestrator does not enforce `min_df`.
- **Migration:** Future result claims must cite their actual run config and manifest. This decision does not retroactively reinterpret earlier runs or implement Gate 2/3.
