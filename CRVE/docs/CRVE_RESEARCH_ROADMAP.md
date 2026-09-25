# CRVE Study Roadmap: From V7/V8 to a Risk-Aware Selection Cascade

> **Status:** short- and medium-term research map, updated 2026-09-25 after Gate 1 Run 2 and Round 4 post-hoc analysis. This document organizes the study;
> it does not describe a fully implemented pipeline or supersede the
> [canonical architecture](ARCHITECTURE.md). V7 and V8 are archived experimental controls. The
> detailed selection theory remains in
> [Phase 2 — Query-Conditioned Expansion-Term Selection Under Uncertainty](phase2_selection_under_uncertainty.md).

The current Gate 1 proposer comparison is concluded as exploratory research without a frozen-protocol
pass. [Gate 1 results and handoff](GATE1_RESULTS_AND_HANDOFF.md) records the evidence for proceeding
to Gate 2 research. This transition does not require another Gate 1 run or a retroactive threshold change.

## 1. Research objective

CRVE studies whether a bounded, query-independent representation of a target corpus can support safe
lexical query expansion before first-stage retrieval. The intended deployment path is:

```text
Phase 1: build the pool and static evidence
    -> Phase 2 / Gate 1: recover useful candidates under a hard budget
    -> Gate 2: reject unsupported or harmful candidates
    -> Gate 3: assign risk-aware lexical influence, or abstain
    -> one BM25/DPH retrieval
```

The study is deliberately decomposed because four claims that were previously easy to conflate are
not equivalent:

1. a term is available in a corpus-derived vocabulary;
2. a term is semantically plausible for a query;
3. admitting the term is unlikely to damage the ranking;
4. a particular lexical weight produces a favorable gain/loss trade-off.

The new schema assigns these questions to separate stages and gives each stage its own objective,
budget, evidence, and exit criterion.

## 2. Why the study moves beyond V7 and V8

### 2.1 What is retained from V7

The archived V7 family established the useful systems premise: prepare a bounded, corpus-derived
vocabulary before queries arrive, embed its terms once, and use it to bridge vocabulary mismatch
without a dense document index or query-dependent feedback retrieval. Its 1,000-term salience pool was
competitive with a much larger bailout store in the archived development evidence, supporting a hard
capacity bound rather than a corpus-sized semantic sidecar.

CRVE retains:

- corpus-grounded vocabulary construction;
- precomputed term representations and statistics;
- strict memory, latency, and candidate budgets;
- a single lexical retrieval after expansion;
- V7 as a reproducible historical control.

### 2.2 What changes after V7/V8

V7 coupled static pool construction, semantic selection, and lexical injection too closely for their
failure modes to be studied independently. The V8 root-cause interpretation adds a sharper warning:
context-free semantic similarity is a useful proposal signal, but it is not an estimate of retrieval
utility. A high-similarity term may be polysemous, have an indiscriminate posting list, displace
relevant documents, or require a much smaller weight than its semantic score suggests.

CRVE therefore changes the unit of study from **“select similar terms and inject them”** to a staged
decision:

| Earlier assumption | Current CRVE research position |
|---|---|
| Pool membership indicates usefulness. | Pool membership only defines availability. |
| Semantic similarity can finalize selection. | Similarity proposes candidates; later evidence decides admission. |
| A selected term is safe to inject. | Compatibility, ambiguity, lexical influence, and harm risk must be tested. |
| Selection score can serve as lexical weight. | Admission and weighting are separate decisions. |
| A fixed number of terms must be emitted. | Returning fewer terms, zero weight, or no expansion is valid. |

The V7/V8 results remain controls and diagnostic evidence. They are not active alternatives to be
extended in parallel with the staged CRVE study.

## 3. Grand scheme of the study

| Stage | Primary question | Optimization emphasis | Output | Current status |
|---|---|---|---|---|
| Phase 1 | What bounded corpus vocabulary and reusable evidence should be available? | Coverage, evidence quality, preparation and memory cost | Pool \(\mathcal P\) and static evidence store | Core pool/statistics implemented; richer context evidence is proposed |
| Gate 1 | Can useful terms be retained within the deployable candidate budget? | Recall of useful opportunities | Candidate set \(C_1(q)\), \(|C_1|\le L\) | Current proposer study concluded; exploratory, frozen criteria unmet |
| Gate 2 | Which candidates lack support or present unacceptable harm risk? | Precision and harmful-candidate rejection | Safer set \(C_2(q)\subseteq C_1(q)\) | Next research focus; downstream effectiveness and cost unverified |
| Gate 3 | How much influence should each survivor receive? | Expected gain versus lower-tail loss and cost | Weighted set \(S(q)=\{(t,\mu_t)\}\), possibly empty | Short-term research design |

“Static” means collected without knowing the future query. Gate 2 and Gate 3 may derive
query-conditioned features from those static artifacts, but they must not perform a first retrieval
over corpus documents. This keeps CRVE distinct from pseudo-relevance feedback.

## 4. Phase 1 — Pool construction and static evidence

### Purpose

Build a capacity-bounded pool that contains useful lexical opportunities and gather enough reusable
corpus information for later gates to reason about meaning, ambiguity, influence, and cost.

### Candidate evidence

The minimum artifact is not just a list of words. Depending on the ablation, Phase 1 may retain:

- analyzed term identity and canonical surface form;
- DF, CF, IDF, salience, and posting-cost summaries;
- frozen term embeddings and optional semantic-coverage assignments;
- lexical or morphological relations and corpus co-occurrence statistics;
- a bounded set of representative term-use contexts;
- a bounded diversity sample for rare but distinct senses;
- a deduplicated context store plus term-to-context references;
- provenance, sampling policy, capacity, and preparation-cost metadata.

Context samples are one candidate representation, not a foregone conclusion. Aggregate or sketch-based
static evidence should remain eligible when it provides a better evidence-to-memory trade-off.
Where text samples are retained, use shared canonical chunks with term-to-chunk references, not
duplicated term-centered windows. Sweep retained chunks per term at 5, 10, 15, 20, and 30; the chunk
segmentation and representative/diversity selection policies remain research choices.

### Research questions

- Which pool policy best preserves downstream useful-term opportunity at fixed capacity?
- Which static features add evidence beyond term similarity?
- How many representative and diversity contexts are worth storing?
- Can contexts be shared across terms enough to keep the sidecar bounded?
- How do pool preparation time, updates, RAM, disk, and VRAM scale with corpus size?

### Exit criterion

Freeze a versioned Phase-1 artifact contract only after the pool is audited for opportunity coverage
and every additional evidence field has a measured downstream purpose. Pool quality is necessary but
does not establish end-to-end retrieval benefit.

## 5. Phase 2 — Query-conditioned decision cascade

### 5.1 Gate 1 — recall-focused candidate collection

Gate 1 asks:

> Under a limited query-time budget, can the system retain the terms that later gates might use?

Its role is intentionally permissive: false negatives here cannot be recovered later, whereas false
positives can still be rejected by Gate 2.

#### Gate-1 candidate methods

Run 2 and its offline post-hoc analysis compared the following single-channel methods and fusions.
The table distinguishes frozen evaluation roles from later offline alternatives; availability does
not establish effectiveness or deployment selection.

| Method | Candidate evidence | Research role and status |
|---|---|---|
| **WholeQueryBGE** | Cosine similarity between the complete query and frozen pool-term embeddings | Available semantic global-match baseline; core RRF channel |
| **AnchorBGEFiltered** | Maximum anchor-to-term similarity after DF-ratio and specificity filtering | Available query-aspect proposer; core RRF channel |
| **AnchorBGEAll** | Maximum anchor-to-term similarity without the anchor filters | Available ablation for measuring the value and false-negative cost of filtering |
| **PPMISidecar** | Precomputed, bounded top co-occurrence neighbors for each query anchor, weighted by anchor IDF | Available low-latency lexical-association proposer; core RRF channel |
| **LivePPMI** | PPMI computed from the live Terrier index | Available diagnostic/fidelity comparator; not the preferred deployment path because it performs query-time posting intersections |
| **SparseLexicalContextProfiles** | BM25 retrieval over an auxiliary index of candidate terms' corpus-context profiles | Available experimental lexical-context proposer |
| **AcronymDefinitionRescue** | Bidirectional acronym/full-form mappings restricted to pool terms | Available experimental high-precision rescue channel |
| **RRF_Core3** | Reciprocal Rank Fusion of WholeQueryBGE, AnchorBGEFiltered, and PPMISidecar | Available primary heterogeneous fusion baseline |
| **RRF_Extended** | RRF_Core3 plus sparse lexical context profiles and acronym rescue | Available extended fusion; must justify its additional sidecars and latency |
| **RRF_Lexical2** | RRF over PPMISidecar and SparseLexicalContextProfiles | Evaluated offline in Round 4; strong material-term coverage alternative |
| **LexicalUnion** | Deduplicated union of equal-size lexical channel prefixes | Evaluated offline in Round 4; preserves both lists with fewer actual candidates than the nominal cap |

The following additional proposer ideas are deferred. They are not prerequisites for the current
Gate 2 handoff and have not been ruled out by the completed method comparison:

- **Query-to-context-centroid proposal:** retrieve terms through one or more precomputed centroids of
  their retained corpus contexts. This tests whether corpus usage improves recall before full Gate-2
  verification.
- **Compound, alias, and morphological rescue:** propose protected technical compounds, aliases, and
  controlled morphological variants that dense similarity or co-occurrence misses.
- **Pool-prior controls:** salience-only, IDF-only, and seeded-random rankings. These are deliberately
  weak controls that reveal how much query conditioning each method actually contributes.
- **Adaptive union or calibrated rank fusion:** allocate the hard candidate budget across channels
  from development evidence, while preserving per-channel provenance and an abstaining cutoff.

Context-rich proposal must not collapse Gate 1 and Gate 2 into one opaque model. Gate 1 may use a cheap
centroid or sparse profile to improve recall; Gate 2 still owns detailed compatibility, ambiguity, and
harm rejection.

Constraints and outputs:

- exclude analyzed terms already present in the query;
- retain the frozen Run 2 deployment cap \(L\le 200\); larger research caps require separate cost evidence;
- measure performance across the full candidate-budget curve rather than only one top-k;
- allow an adaptive stopping policy below the cap;
- return ranked candidates with channel scores and provenance, but no final lexical weights.

The completed comparisons use the same eligible pool and exclusion rules and reuse raw top-500 lists
for audit. Round 4 characterizes caps 50, 100, 200, 300, 400 and 500, plus an Extended RRF set matched
to Union400's actual candidate count on every query. Caps above 200 are exploratory; matching output
counts does not establish equal proposer or context-scoring cost.

Primary evidence includes TermRecall, query-level helpful-opportunity hit, near-best opportunity
retention, deep-recall opportunity coverage, latency, and candidate count. TermPrecision remains a
cost diagnostic, not Gate 1's sole objective. Ranking-safe and deep-recall-safe opportunities must be
reported separately.

The current proposer study is closed. Extended@400 retains 81.69% of reference best-gain opportunity
and 81.27% of safe document opportunity, with 66.85% NearBestHit. The original 85% TermRecall,
90% NearBestHit and 92% BOR floors remain unmet even in the L=500 curves. The later exploratory
70/80/85/70 milestones are met at L=500 by Extended and Lexical2, without changing the failed
Checkpoint B status. The decision is to test how well Gate 2 uses these candidates, not to keep
expanding the proposer budget to satisfy an exit label. See the [results record](GATE1_RESULTS_AND_HANDOFF.md)
for exact sources, metric denominators and the walkthrough/CSV discrepancy. The frozen algorithm
contract remains in the [co-located pathway](../src/crve/selection/pathway_gate1_selection.md).

### 5.2 Gate 2 — precision-focused harmful rejection

Gate 2 asks:

> Given a plausible candidate, does the available corpus evidence justify admitting it, or is the
> candidate unsupported, ambiguous, redundant, dormant, or too risky?

Gate 2 should reuse the Phase-1 evidence and add query-conditioned comparisons. Candidate signals may
include:

- maximum or top-k query-to-context compatibility;
- representative support and diversity support;
- incompatibility or ambiguity across retained contexts;
- DF, CF, IDF, posting cost, and lexical-influence bounds;
- agreement among Gate-1 proposal channels;
- overlap or redundancy with other candidates;
- calibrated helpfulness and harm estimates learned only from declared development data.

Reuse Run 2 ranks and action labels for development. Keep Extended RRF as a balanced comparison and
the lexical policies as alternatives, with a small declared policy set rather than a new Gate 1 search.
Retain L=200 as the frozen control; measure the downstream costs of an L=400 research condition before
selecting a larger operational budget. Evaluate precision and retained opportunity together, both
relative to the incoming candidates and to the evaluated reference universe. A rejection-only Gate 2
cannot recover omitted terms, and oracle gain across tested weights is not a safe deployed action.

For a candidate, fetch its retained evidence together and score it in one batch. Deduplicate shared
chunk IDs across candidates. Adaptive multi-round context fetching is not part of this short-term map.

Probabilistic properties are optional evidence, not assumed truth. Raw cosine, a sigmoid of cosine, or
an uncalibrated classifier score must not be labeled as a probability. Any \(p_{\mathrm{help}}\) or
\(p_{\mathrm{harm}}\) must specify its fixed term-weight action or frozen weighting policy, calibration
split, reliability error, and domain-transfer limitations. Harm is weight- and metric-dependent; a term
is not intrinsically “harmful” without a declared action and evaluation target.

Possible Gate-2 decisions are accept, reject, defer to a constrained representation, or mark as too
uncertain. The gate may also reject candidates whose maximum possible lexical influence is too small
to affect the target depth, although such a bound proves dormancy rather than positive utility.

Primary evidence includes accepted-term precision, harmful accepted-term rate, useful-term false
rejection, ambiguity strata, pruning rate, calibration quality, and added latency. Gate 2 advances only
if it materially improves precision or harm rejection without erasing Gate 1's useful-opportunity
coverage.

### 5.3 Gate 3 — risk-aware weighting and abstention

Gate 3 asks:

> For each admitted candidate, how much lexical influence is justified by its potential gain, loss,
> redundancy, and cost?

Selection and weighting remain separate, and no Gate-3 weighting method has been selected. A Gate-2 survivor may still receive a small weight or
\(\mu_t=0\). A useful research objective is a constrained portfolio rather than independent semantic
weights:

\[
\max_{S,\mu}\quad
\widehat{G}(q,S,\mu)
- \lambda\,\widehat{R}_{\mathrm{harm}}(q,S,\mu)
- \eta\,\widehat{C}(q,S)
\]

subject to declared limits on expansion count, total expansion mass, postings touched, and latency.
Here \(\widehat{G}\) is estimated retrieval gain, \(\widehat{R}_{\mathrm{harm}}\) is a calibrated
lower-tail or loss-risk estimate, and \(\widehat{C}\) is execution cost. These are research quantities,
not current guarantees.

Gate 3 should study:

- expected gain and harm probability or lower-tail loss;
- context support and ambiguity;
- term specificity and posting cost;
- redundancy and complementary new reach across the selected set;
- separate policies for shallow ranking quality and deep candidate recall;
- safe lexical forms such as phrases, proximity constraints, or a restricted expansion branch;
- query-level abstention and baseline preservation.

Primary evidence includes paired nDCG and Recall deltas, improved/degraded/tied query counts, lower-tail
loss, baseline-preservation rate, expansion mass, postings touched, and latency. Equal weighting and
direct reuse of semantic scores are baselines, not default CRVE policies.

## 6. Short-term map

The short-term horizon establishes evidence and validates each decision boundary before optimizing the
full cascade.

1. **Freeze the Phase-1 audit contract.** Record pool membership, static term evidence, context-sample
   provenance where applicable, and resource costs.
2. **Gate-1 proposer study concluded.** Preserve Run 2 and Round 4 as exploratory evidence, with the
   frozen protocol unmet. Carry the measured candidates into Gate 2; no new Gate 1 pass or rerun is
   required for this research transition. Final policy and budget remain subject to downstream cost
   and quality evidence.
3. **Build the smallest useful Gate-2 sidecar.** Start with term statistics plus a bounded,
   deduplicated context representation rather than a high-dimensional learned predictor.
4. **Run fixed-action Gate-2 ablations.** Compare maximum compatibility, top-2 support,
   representative support, ambiguity evidence, and one declared probabilistic rule if calibration data
   are sufficient.
5. **Study Gate-3 weighting.** Compare declared candidate policies, including a fixed-weight grid and
   zero weight, against gain, lower-tail loss, and execution cost. Choose a weighting method only after
   Gate-2 evidence and paired full-retrieval results justify it. Gate 3 remains in the short-term map.
6. **Establish the abstention baseline.** Compare every safety and weighting method with leaving the original query
   unchanged.
7. **Run a bounded cascade feasibility test.** Connect the frozen Phase-1 artifact, Gate 1, the
   smallest viable Gate 2, and the Gate-3 prototype. Treat this as an integration diagnostic, not the
   final end-to-end claim.

The short-term deliverable is not a claim that CRVE wins end to end. It is a reproducible answer to
whether useful opportunities can be retained cheaply and whether harmful candidates can then be
rejected with acceptable false-rejection cost, followed by an initial answer to whether admitted terms
can be weighted without unacceptable lower-tail harm.

## 7. Medium-term map: deployment under real system constraints

The medium-term horizon turns a feasible short-term cascade into a family of deployable systems. It
should not assume that one execution layout fits every corpus. The same decision semantics can be
tested in four modes:

1. **Fully on-device:** the index, static evidence, all gates, and retrieval stay on one constrained
   device.
2. **Edge-assisted:** the device preserves the query and local index path while optionally requesting
   bounded evidence or computation from a nearby trusted node.
3. **Distributed or federated:** vocabulary and corpus evidence remain partitioned across devices,
   collections, organizations, or geographic regions and are merged through a declared protocol.
4. **Streaming-data RAG:** documents or events arrive continuously, so the lexical index, vocabulary,
   static evidence, and calibration state must evolve without stopping query service.

### 7.1 Fully on-device CRVE

The on-device track asks how much of the cascade can be made genuinely local rather than merely tested
on an edge-class GPU. Candidate implementation directions include:

- quantize term and context representations to FP16, INT8, or product-quantized forms and measure the
  resulting Gate-1 recall and Gate-2 rejection loss, not only cosine reconstruction error;
- memory-map cold evidence and keep only query anchors, candidate rows, and active context blocks in
  RAM;
- use progressive evidence loading: cheap term statistics first, representative contexts second, and
  diversity contexts only for unresolved candidates;
- schedule work across CPU, GPU, and available NPU/DSP resources according to model residency,
  transfer cost, current thermal state, and deadline;
- fuse candidate scoring and top-k maintenance to avoid materializing full score vectors;
- support incremental corpus updates with append-only evidence segments, background compaction, and
  versioned snapshots rather than rebuilding the complete sidecar;
- develop a resource controller that jointly allocates candidate count, contexts per candidate,
  expansion mass, and postings touched under per-query latency and energy budgets;
- preserve a complete offline mode with no network dependency and a deterministic lexical-only
  fallback when accelerators or evidence blocks are unavailable.

Evaluation should add cold-start time, joules per query, sustained throughput under thermal throttling,
flash reads and writes, model-residency transitions, update amplification, and peak RAM/VRAM to the
existing effectiveness and latency measures. A fast first query that later throttles is not an
edge-ready result.

### 7.2 Edge-assisted split execution

The edge-assisted track should test several split points instead of assuming that semantic inference
belongs entirely on a server:

- **Local Gate 1, remote Gate 2:** transmit candidate identifiers plus the minimum query representation
  required to request richer context evidence.
- **Local Gates 1-2, remote Gate 3:** keep semantic evidence local and ask a trusted coordinator only
  for a calibrated policy or global cost signal.
- **Remote proposal, local admission:** receive a broad external candidate list but let the device's
  own corpus evidence and risk policy decide what enters its lexical query.
- **Uncertainty-triggered assistance:** remain fully local for confident queries and escalate only the
  unresolved candidate subset, making network cost follow uncertainty rather than every query.

Each split must define what leaves the device. Raw queries, raw contexts, embeddings, candidate IDs,
and score vectors have different privacy and reconstruction risks. Hashing or quantization alone must
not be described as privacy. Experiments may compare redaction, local feature projection, differential
privacy, secure aggregation, or a trusted execution boundary, but only with measured utility,
bandwidth, and leakage assumptions.

The system must also specify timeout behavior, cache validity, stale-evidence handling, and a local
fallback. Report end-to-end p50/p95/p99 latency with realistic network delay and failure injection,
rather than reporting remote compute time alone.

### 7.3 Distributed and federated CRVE

The distributed track treats the static evidence store as a federation of bounded local memories. A
node may represent one personal device, one domain corpus, one organization, or one shard of a large
collection. Research directions include:

- hierarchical proposal: each node emits a small local Gate-1 list and a coordinator performs a
  deterministic, budgeted merge;
- globally comparable scoring without assuming that local DF, IDF, score calibration, or corpus size
  are identical across nodes;
- domain-specialist routing, where only nodes with plausible vocabulary coverage participate in the
  later gates;
- two-level evidence: retain rare local senses at the node while sharing only bounded aggregate
  support or risk summaries;
- federated calibration of harm models without centralizing raw queries, relevance judgments, or
  contexts, with explicit tests for client drift and non-IID corpora;
- distributed set selection that rewards complementary document reach across shards instead of
  choosing many synonymous terms from the largest node;
- versioned evidence manifests, deterministic merge order, replayable decisions, and graceful
  operation under slow, stale, partitioned, or missing nodes;
- optional peer-to-peer execution for disconnected environments, using a strict message and energy
  budget and no assumption of a permanently available coordinator.

This setting creates a new scientific question: whether a term rejected as broad globally can remain
highly discriminative inside a local domain. The study should compare global normalization, local
normalization, and hierarchical priors rather than forcing one universal IDF or harm threshold.

### 7.4 CRVE for RAG over streaming data

This track studies RAG over a continuously changing document or event stream. It does not mean
token-by-token answer generation. CRVE remains responsible for first-stage retrieval; its additional
responsibility is to produce a traceable result from a declared view of an evolving corpus.

A candidate architecture is:

```text
document/event stream
    -> analyzed index delta + statistics delta
    -> vocabulary/context-evidence update queues
    -> atomic evidence snapshot publication
    -> query pinned to snapshot version
    -> Gates 1-3 -> lexical retrieval from the compatible index view
```

The index and CRVE sidecar do not have to become current at exactly the same instant, but their version
relationship must be explicit. A query must never silently combine IDF from one corpus state, context
evidence from another, and postings from a third.

#### 7.4.1 Incremental evidence maintenance

- Maintain append-only index and evidence deltas, with bounded background compaction, instead of
  rebuilding the complete pool after every event.
- Update DF, CF, IDF, co-occurrence summaries, and posting-cost estimates by epoch. Keep weights stable
  inside an active snapshot even while the next epoch is being assembled.
- Admit a newly observed term first to a lexical-only probation state. Promote it into semantic
  proposal only after its canonical form, minimum support, and embedding are ready.
- Refresh representative contexts with streaming reservoir sampling and diversity contexts with a
  bounded replacement policy. Record the time horizon and sampling bias of both memories.
- Support sliding windows, exponential decay, or explicit temporal strata when old evidence should
  lose authority. These policies must not be mixed without declaring which corpus history a query is
  intended to search.
- Handle corrections, expiry, and deletion through tombstones and compensating statistics; test when
  accumulated approximation error requires a full reconciliation pass.

#### 7.4.2 Query service under concurrent updates

- Pin each query to a versioned index/evidence snapshot so Gate decisions and lexical retrieval are
  reproducible.
- Separate freshness service-level objectives from query latency objectives. An update may be indexed
  immediately while richer Gate-2 context evidence becomes available within a bounded lag.
- Use micro-batching for embeddings and evidence refresh only when it improves throughput without
  violating maximum time-to-searchable.
- Apply backpressure independently to ingestion, embedding, compaction, and query queues. Query
  overload must not cause unbounded evidence growth, and ingestion bursts must not evict the active
  retrieval state from memory.
- Preserve a degraded but correct path: if semantic evidence is late, use the newest compatible
  lexical snapshot with conservative expansion or no expansion.
- Revalidate Gate-2 calibration as vocabulary, term senses, and document sources drift. Detected drift
  should increase abstention before it triggers automatic retraining or aggressive reweighting.

#### 7.4.3 Distributed streaming state

For sharded or federated streams, each update needs a stable source ID, partition, offset, event time,
and evidence version. Research prototypes should test:

- idempotent replay after failure and deterministic recovery from a checkpoint plus an update log;
- watermarks for late or out-of-order events and declared behavior for evidence arriving after a
  query's snapshot boundary;
- local delta summaries followed by bounded global merges, without requiring every node to pause at
  one barrier;
- source-aware trust and temporal priors so a burst from one noisy producer cannot immediately
  dominate the global vocabulary or harm model;
- node churn, network partitions, stale replicas, and resynchronization without losing the local
  lexical-only fallback;
- deletion propagation and retention guarantees across cached, replicated, and sampled contexts.

#### 7.4.4 Streaming-data evaluation

A static benchmark replay is insufficient. Evaluation should interleave timestamped updates and
queries, then report:

- time-to-searchable and time-to-CRVE-ready for new evidence;
- update throughput, backlog size, compaction cost, and peak state during bursts;
- query p50/p95/p99 latency under mixed ingestion and retrieval load;
- snapshot staleness and the fraction of queries served from degraded or lexical-only paths;
- retrieval quality over time, including temporal slices, emerging terms, changed senses, and expired
  facts;
- calibration drift, abstention rate, and harmful-expansion rate by evidence age;
- crash recovery time, replay volume, duplicate-update tolerance, and result reproducibility;
- memory, storage writes, energy, and network traffic per ingested item and per query.

The core streaming hypothesis is that CRVE's bounded pool and evidence store can absorb corpus change
through small, versioned deltas while preserving most of the benefit of a periodically rebuilt oracle.
The required control is periodic full rebuilding from the same stream history. If incremental CRVE
diverges materially in effectiveness, calibration, or memory, the streaming design has not succeeded
even if its update latency is low.

### 7.5 Cross-cutting research bets

Several higher-risk ideas are worth testing after the basic deployment tracks are measurable:

- **Compute follows uncertainty:** dynamically stop *between* gates when the decision margin is large;
  when Gate 2 evaluates a term, fetch all its retained evidence together rather than using adaptive
  multi-round context fetching. Spend network and weighting budget only on ambiguous cases.
- **Anytime CRVE:** emit a safe preliminary query under a tight deadline, then improve the expansion
  set if more local or distributed evidence arrives before retrieval begins.
- **Evidence distillation:** compress an expensive distributed teacher cascade into a small local
  policy while retaining the full cascade as an offline audit oracle.
- **Multi-resolution context memory:** keep term statistics permanently resident, short context
  sketches in warm storage, and exact context blocks in cold storage or on trusted peers.
- **Drift-aware abstention:** detect when corpus updates or a new device population invalidate Gate-2
  calibration and automatically fall back to conservative weights or no expansion.
- **Personal and shared vocabularies:** combine a private on-device pool with a public domain pool while
  preventing the shared layer from revealing which private terms or contexts triggered a decision.
- **Retrieval-budget exchange:** allow Gate 3 to trade fewer high-cost expansion postings for a deeper
  retrieval cutoff, or more low-cost terms for a shallower cutoff, under one declared latency budget.

These are hypotheses, not architectural commitments. Each survives only if it improves a declared
effectiveness/resource/privacy objective over the simpler local cascade.

### 7.6 Medium-term milestones

1. **Hardware envelope matrix:** reproduce the cascade on at least three declared profiles, such as
   CPU-only low memory, integrated accelerator, and discrete edge GPU.
2. **Compressed sidecar prototype:** compare representation precision, memory mapping, evidence tiers,
   and incremental updates under the same held-out queries.
3. **Adaptive resource controller:** demonstrate deadline-aware allocation with hard caps and
   deterministic fallbacks.
4. **Split-execution prototype:** benchmark at least two trust boundaries with realistic bandwidth,
   latency, outage, and privacy accounting.
5. **Federated/sharded prototype:** validate distributed top-k merging, local/global calibration, node
   failures, and non-IID corpora.
6. **Streaming-data prototype:** replay interleaved updates and queries; validate versioned snapshots,
   bounded freshness lag, drift-aware abstention, deletion, backpressure, and failure recovery against
   periodic full-rebuild controls.
7. **Complete system evaluation:** compare untouched BM25/DPH, V7/V8 controls, RM3/Bo1, and relevant
   neural baselines under matched preparation, memory, energy, latency, and retrieval-depth budgets.
8. **Scope decision:** determine whether CRVE is best supported as a fully local expansion method, an
   uncertainty-routed edge service, a distributed vocabulary layer, a selective candidate-recall
   tool, a streaming-data retriever, or a diagnostic framework.

## 8. Stage-transition rules

| Transition | Evidence required | If the evidence fails |
|---|---|---|
| Phase 1 -> Gate 1 | Useful-term opportunity exists in the bounded pool at acceptable preparation cost. | Revisit eligibility, pool construction, or capacity before selector work. |
| Gate 1 -> Gate 2 research | Completed Run 2 / Round 4 evidence shows useful retained opportunities; preserve the failed frozen-protocol status and measure downstream cost. | Revisit proposal only if retained opportunity or measured cost makes the downstream study unviable; a missed historical floor alone does not require another proposer search. |
| Gate 2 -> Gate 3 | Precision or harmful rejection improves without unacceptable useful-term loss. | Simplify or retire the context/risk mechanism; preserve abstention. |
| Gate 3 -> end-to-end claim | Weighted expansion improves a declared effectiveness/resource objective on held-out data. | Narrow the claim or retain the cascade as a diagnostic result. |

No stage may borrow final-test relevance labels to justify advancement. nDCG-oriented and
Recall-oriented conclusions remain separate unless an explicit multi-objective policy is declared.
Advancement to research is distinct from a confirmatory or deployment pass. The current transition
accepts measured Gate 1 false negatives as a documented limitation; Gate 2 must quantify any further
loss, and final retrieval claims still require held-out evaluation.

## 9. Study-wide guardrails

- Keep the active work inside `CRVE/` and under the 15 GiB host-memory ceiling.
- Stream large corpora; never materialize complete raw corpora in Python collections.
- Preserve full retrieval depth and control memory through query chunking, not ranking truncation.
- Report preparation, sidecar size, startup, query latency, postings touched, RAM, and VRAM alongside
  effectiveness.
- Treat counterfactual term utility as offline development evidence, not an observable query-time
  signal.
- Use held-out evaluation for final claims and keep V7/V8 frozen as historical controls.
- State whether each result applies to nDCG, Recall, or both.
- Treat “no expansion” as a successful safety decision when evidence is insufficient.

## 10. Relationship to other documents

- [ARCHITECTURE.md](ARCHITECTURE.md) is the canonical description of the active implementation.
- [Gate 1 results and handoff](GATE1_RESULTS_AND_HANDOFF.md) records the completed exploratory study,
  its unmet frozen criteria and the decision to proceed to Gate 2 research.
- [Capacity-Bounded Corpus-Informed Query Expansion](corpus_informed_query_expansion_plan.md) preserves
  an earlier, broader context-memory proposal; its old windows and budgets are not current defaults.
- [Phase 2 — Selection Under Uncertainty](phase2_selection_under_uncertainty.md) defines the formal
  labels, metrics, uncertainty limits, and detailed gate hypotheses.
- [CRVE Design Refinement Notes](crve_design_refinement_notes.md) records the context-memory and
  ambiguity discussion that motivates Gate 2.
- [Gate 1 Pathway Specification](../src/crve/selection/pathway_gate1_selection.md) defines the
  implemented Gate-1 pathway.
- [Evaluation Metrics](EVALUATION_METRICS.md) owns retrieval metric definitions.

In short, the research program moves from **bounded corpus knowledge**, to **high-recall opportunity
collection**, to **precision and harm control**, and only then to **risk-aware lexical weighting**.
