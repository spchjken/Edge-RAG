# CRVE first-stage retrieval architecture

This is the canonical description of CRVE's scope and stage boundaries. CRVE produces a first-stage lexical ranking; downstream reranking and answer generation are outside its active scope. The [research roadmap](CRVE_RESEARCH_ROADMAP.md) describes experiments and longer-term deployment questions, not completed components.

## System flow and implementation status

```text
Corpus -> bounded eligible term pool -> reusable static evidence
       -> Gate 1: recall-focused term proposal, at most 200 candidates
       -> Gate 2: precision-focused context verification and harmful-candidate rejection
       -> Gate 3: choose lexical weights, possibly no expansion
       -> one full first-stage BM25/DPH retrieval -> ranked documents
```

Pool construction, Gate 1 proposers, and a direct expanded-query retrieval path exist in `CRVE/src/crve/`. Gates 2 and 3 are **short-term research stages, not implemented production gates**. The current orchestrator's simple normalized candidate-score weights are an experimental bridge to retrieval, not a calibrated Gate 3 policy or evidence that the full cascade exists.

The gates make distinct decisions: pool membership establishes availability; Gate 1 preserves potentially useful terms under a hard candidate budget; Gate 2 uses query-conditioned evidence to reject unsupported, ambiguous, redundant, or potentially harmful terms while tracking false rejection; Gate 3 determines each survivor's lexical influence. Query-conditioned evidence comes from corpus information prepared without the future query. The full document retrieval occurs after these decisions, not as pseudo-relevance feedback inside a gate. Returning no expansion is valid.

## Phase 1: pool and static evidence

For the **frozen pool/Gate 1 experiment**, eligible terms satisfy DF >= 2 and CF >= 3, and the pool contains `min(10,000, |V_eligible|)` terms. The exact frozen pool sizes, hashes, exclusions, and settings are recorded in [`gate1_phase2_1a.yaml`](../configs/gate1_phase2_1a.yaml) and the [review copy](../../for_review/selection_phase/gate_1/run_2/frozen_gate1_config_phase2_1a.yaml). `for_review/` artifacts are immutable evidence for their specific runs; older run reports retain their historical configurations. Generic [`crve.yaml`](../configs/crve.yaml) is a runtime configuration, not a substitute for a frozen experiment manifest.

Static evidence can include analyzed identities and surface forms, DF/CF/IDF, embeddings, co-occurrence sidecars, posting-cost summaries, aliases, sketches, and sampled corpus usage. CRVE is not defined by the presence of samples. Each artifact must have provenance, a capacity bound, build cost, and a corpus/index version.

For the proposed Gate 2 context sidecar, partition documents into small **canonical chunks**, store each retained chunk once, and maintain term-to-chunk references. Do not store a separate `+/-50`-token centered window for every occurrence: adjacent terms such as `JVM` and `garbage` could produce nearly identical copies, potentially making the saved sample text larger than the corpus. Chunk boundaries and deduplication policy require empirical validation; no particular chunk length is frozen yet. Sweep **5, 10, 15, 20, and 30 retained chunks per term**, with fewer when fewer distinct eligible chunks exist. Measure unique chunks, term-to-chunk references, duplicate/reuse rates, storage, and preparation cost. Representative and diversity-oriented sampling are alternative or complementary policies to evaluate, not an established winner.

## Gate 1: bounded, recall-focused proposal

Gate 1 emits a ranked candidate set `C1(q)` with `|C1(q)| <= 200`; it does not decide final admission or lexical weights. The [co-located pathway](../src/crve/selection/pathway_gate1_selection.md) owns detailed proposer and audit semantics. Compare individual channels and fusions at matched candidate budgets. `L=500` is diagnostic only, beyond the deployable cap.

The **current frozen Gate 1 core RRF** combines WholeQueryBGE, AnchorBGEFiltered, and **PPMISidecar** (`k=60`, input depth 500). PPMISidecar is prebuilt from index evidence during preparation and bounds query-time work by storing a limited neighbor list per anchor. **LivePPMI** computes PPMI from live index postings at query time and is a higher-fidelity diagnostic comparator, not the low-latency core RRF member. Precomputation can truncate neighbors; it should not be described as mathematically identical to unrestricted live computation. The [review report](../../for_review/selection_phase/gate_1/run_2/gate1_halftime_review_report.md) records their measured latency for that run. Other evaluated channels and historical run variants remain valid as labeled experiments, not current defaults.

## Gate 2: precision-focused verification

Gate 2 is proposed. For every Gate 1 candidate, fetch its retained static evidence and score all available evidence **in one batch**; deduplicate shared chunk IDs across candidates before scoring. There is no adaptive `2 -> 4 -> 8` or `8 -> 16 -> 30` evidence-fetch loop in the current design. Context compatibility, representative support, diversity/ambiguity, lexical statistics, redundancy, and optional calibrated harm estimates are candidate signals. Static statistics can describe context around a term without being literal text samples.

Gate 2 aims to improve admitted-term precision and harmful-candidate rejection without destroying useful-opportunity recall. A raw similarity score is not a probability of helpfulness. Any harm label or calibrated probability must declare its retrieval action, metric, depth, corpus/index version, and development/test split. Term-level proxy evidence cannot by itself prove that a term will improve the final ranking.

## Gate 3: weighting, then full retrieval

Gate 3 is proposed and its weighting **method is open**. It must balance potential gain, loss, redundancy, and execution cost for Gate 2 survivors; a survivor can receive zero weight, and the query can abstain from expansion. Equal weights, fixed weight grids, or mass bounds may be studied as baselines, but no historical weighting theorem or particular formula is the selected CRVE policy. Evaluation requires running the complete first-stage retrieval and comparing paired query-level ranking/recall outcomes, including negative outcomes.

## Source-of-truth order

1. This page owns active scope, stage boundaries, and implemented-versus-proposed status.
2. A frozen `for_review/` config and its run manifest own **that experiment's** numeric settings and results. The matching `CRVE/configs/gate1_phase2_1a.yaml` is its working config. The generic `CRVE/configs/crve.yaml` owns only its own runtime defaults; it does not retroactively define frozen runs.
3. The [Gate 1 pathway](../src/crve/selection/pathway_gate1_selection.md) owns Gate 1 algorithm details; [evaluation metrics](EVALUATION_METRICS.md) owns metric definitions; the [selection design](phase2_selection_under_uncertainty.md) owns proposed labels and research hypotheses where consistent with this architecture.
4. The [roadmap](CRVE_RESEARCH_ROADMAP.md) owns the short-/medium-term study sequence. The [older corpus-informed plan](corpus_informed_query_expansion_plan.md), [design notes](crve_design_refinement_notes.md), and [IT-MPE theory note](theoretical_foundations_anchored_expansion.md) preserve historical proposals, not current defaults.

Where a generic runtime config and a frozen experiment differ, report the config actually used for the claimed result. Do not silently change historical artifacts or infer that an unimplemented stage ran in a frozen Gate 1 test.
