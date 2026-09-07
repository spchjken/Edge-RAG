# Edge-RAG: Corpus-Informed Query Expansion — Research and Refactor Plan

> Status: revised research proposal, 2026-09-07. Formerly `expansion_platform_plan.md`.
> V7 in Pipeline V2 is the existing implementation and experimental control.
> This proposal broadens the research direction; it does not describe a completed refactor or
> supersede [the current implementation architecture](ARCHITECTURE.md).

## 1. Research objective and positioning

Investigate how compact corpus representations can improve query expansion for BM25 without
additional task-specific model training, under explicit indexing, memory, and query-latency budgets.

The name **Corpus-Informed Query Expansion** describes the research focus without committing to
a general-purpose software platform or a particular vocabulary representation. V7 is one instance:
a corpus vocabulary, frozen dense term representations, anchor-to-term similarity, and bounded
weighted expansion.

The primary comparison family is expansion over BM25. Learned sparse and dense retrievers remain
valid reference systems for effectiveness and resource tradeoffs. Different training regimes must
be reported; they do not make comparisons invalid. Describe the method as requiring **no additional
task-specific training**, rather than claiming no learned components or zero total training cost.
Any use of target-dataset relevance labels for parameter tuning must also be disclosed.

Separate two deployment settings:

- **Ephemeral collections:** measure full time to usable retrieval, including representation
  construction and embedding. Expensive offline structures may be unsuitable.
- **Persistent large collections:** measure amortized construction, index size, updates, and
  query latency under the same memory ceiling.

A technique may succeed in one setting and fail in the other. A broad source interface alone is
not the research contribution; the experiments must show which corpus evidence helps, when it
helps, and what it costs.

## 2. Current implementation and evidence

V7 analyzes the query into distinct content anchors, assigns POS-based weights, probes a corpus
vocabulary using frozen BGE-small embeddings, and compiles a sparse weighted query for BM25.
The default vocabulary design has a 1,000-term probing pool and up to 50,000 stored terms for
out-of-pool rescue. In-memory and streaming retrieval currently have separate construction paths.

Recorded results motivate improvement but do not establish universal expansion gains:

| Recorded evaluation | Analyzed BM25 | V7 | Interpretation |
|---|---|---|---|
| Ten-corpus macro Strict@10 | 61.17% | 63.05% | Modest average improvement, with dataset regressions |
| Ten-corpus macro DocRec@10 | 48.21% | 49.54% | Improvement needs attribution to weighting versus expansion |
| Ten-corpus mean query latency | 0.96 ms | 14.41 ms | Expansion adds material overhead |
| Five large corpora, mean query latency | 27.63–84.89 ms | 1,191.29–8,112.08 ms | Severe scaling problem |

Sources: [V7 comparison, 2026-09-04](../results/legacy/v7_vs_baselines/v7_vs_baselines_summary.md)
and [streaming comparison, 2026-09-06](../results/legacy/v7_large_scale/v7_streaming_large_summary.md).
These are archived observations, not fresh measurements of the current checkout. The large-scale
quality changes are mixed; for example, DBpedia nDCG@10 falls from 0.2850 to 0.2793, while
Climate-FEVER improves from 0.1303 to 0.1429.

The [PyTerrier results](../results/pyterrier_baselines/pyterrier_baselines_summary.md) provide BM25
and RM3 controls. They are a separate evaluation artifact: dataset versions, query subsets,
scoring, retrieval depth, and timing boundaries must be aligned before combining comparisons.

## 3. Organizing the design space

Expansion can obtain evidence in three ways. These categories describe the source of evidence,
not mutually exclusive algorithm families.

| Evidence source | Query-time behavior | Examples and role |
|---|---|---|
| Query and external knowledge | Propose alternatives without consulting the target corpus | Lexical resources or generated terms/pseudo-documents |
| Query-dependent corpus documents | Retrieve documents and extract feedback | PRF, with RM3 as the primary classical comparator |
| Reusable corpus representation | Select relevant parts of a structure built before the query | V7 vocabulary, graphs, phrases, context prototypes, topic summaries |

Hybrid designs are allowed: query-only generation can be validated using corpus structures, and
PRF can use a static representation to interpret its feedback.

The research has three independent axes:

1. **Representation:** what reusable corpus knowledge is stored?
2. **Relation:** how is that knowledge related to an anchor, phrase, or whole query?
3. **Action:** how does selected evidence propose, validate, or weight expansion terms?

Selecting a centroid is not itself query expansion. Any centroid-based proposal here must define
a subsequent mapping into lexical query terms. A resemblance to an indexing component of another
retriever is not sufficient to identify the resulting method with that retriever.

## 4. Candidate corpus representations

The following are proposed designs and hypotheses, not established performance claims.

| Representation | Query relation and lexical output | Hypothesis | Main cost or risk |
|---|---|---|---|
| Vocabulary vectors: V7 control | Anchor-to-term similarity produces weighted terms | Isolated semantic neighbors bridge vocabulary gaps | Ambiguity and generic neighbors |
| Phrase/entity inventory | Match query spans or the whole query to corpus phrases and aliases | Better lexical units preserve technical meaning | Extraction errors, phrase matching and tokenization |
| Multiple context prototypes per term | Select a term sense using query context; emit vocabulary associated with that sense | Corpus usage resolves ambiguity better than isolated term embeddings | Context collection, embedding and storage |
| Term association graph | Find supported neighbors or connections among query concepts | Joint evidence rejects unrelated single-anchor expansions | Graph size, noisy edges and frequent-term bias |
| Topic/cluster summaries | Select topic prototypes, then terms characteristic of their clusters | Topic context identifies useful vocabulary missed by global term similarity | Construction cost and wrong-topic selection |
| Compressed passage representatives | Match query context to representative sentences/passages; select terms from them | Small contextual units capture relations absent from word vectors | Coverage, redundancy and setup time |
| Relation/event inventory | Match patterns such as cause, treatment or comparison; emit lexical realizations | Relation evidence preserves what the query asks about | Extraction quality and substantial complexity |

For context prototypes, begin with bounded context samples and a small number of prototypes for
selected ambiguous terms or technical phrases. For topics, store a term distribution or explicit
representative terms alongside the prototype: a vector alone cannot define the expansion output.

PMI/co-occurrence can supply graph edges or direct candidate scores. Alias dictionaries can supply
phrase/entity entries. Their preprocessing and update costs must be measured rather than labeled
universally cheap.

## 5. Query-to-representation relations and expansion actions

### 5.1 Relation experiments

Start with the existing vocabulary matrix so relation changes can be isolated from representation
changes:

- **Independent anchors:** frozen V7 relation.
- **Whole query:** query-to-vocabulary similarity, as a separate experimental condition.
- **Agreement across anchors:** support from one anchor plus compatibility with other query concepts,
  or support from multiple anchors.
- **Query spans/aspects:** match coherent phrases or subquestions independently.
- **Contextual matching:** query context selects corpus contexts, senses or topics.

For example, an expansion for “battery degradation in cold weather” should have support from the
battery/cold context, rather than merely being associated with weather. Requiring every anchor to
agree would wrongly suppress useful terms in queries containing independent subquestions.
Agreement policies should permit support from coherent subsets and account for redundant anchors.

Compare these policies at matched candidate and weight budgets. Otherwise, apparent gains may
come from adding fewer terms rather than using better evidence.

### 5.2 Actions beyond candidate generation

**Validate proposals.** An external generator may propose terms while corpus contexts or a graph
assess their compatibility with the query. Corpus-vocabulary projection guarantees only that a term
is in the indexed vocabulary; it does not guarantee semantic correctness or eliminate hallucination.

**Allocate effort to lexical gaps.** Test whether expansion is more useful for concepts with weak
lexical coverage: absent terms, missing phrases, or low co-occurrence among constituent terms.
These are hypotheses, not relevance guarantees. Rare exact terms may still need aliases.

**Preserve the original query.** Store original weights separately from expansion increments.
An expansion candidate that coincides with another original anchor must have an explicit collision
policy and remain charged to the expansion budget.

## 6. Shared architecture and simplification

Proposed flow:

`query analysis → evidence source → candidate validation → budget allocation → weighted BM25`

Keep the shared policy understandable: evidence selection, admission, and budget allocation.
This is not a claim that every source has only three parameters. Freeze and report source-local
presets such as PRF feedback depth, graph neighborhoods, prototype count, or generator settings.

A source adapter should provide:

| Field | Contract |
|---|---|
| Indexed term or explicit phrase mapping | Final scoring terms use the retrieval analyzer; phrase splitting must account for every emitted term |
| Raw score and score semantics | Identify cosine, association statistic, feedback weight, etc. |
| Admission value/policy | Source-specific gate, or a documented calibrated common scale |
| Support attribution | Anchor, span, or whole-query support and any responsibilities used for allocation |
| Provenance | Source, representation version and supporting evidence |
| Cost metadata | Candidate DF/posting count where available, plus source construction/query cost |

Cosine, PMI and feedback weights are not interchangeable. Merely mapping values into [0,1] does
not make them calibrated confidence probabilities. Initially use documented source-specific gates
and nonnegative normalized allocation weights; require evidence before adopting a common threshold.

Query-level sources may use a direct global allocator. If an anchor-based guarantee is claimed,
their attribution to anchors must be explicitly defined. Keep native RM3 as a baseline even if
an RM3-derived candidate source is also evaluated through the shared allocator.

Shelve legacy gate/allocation variants from the primary method while retaining named ablation
controls. Treat uniform anchor weights as a simplification control against frozen POS priors.
Renaming POS weights “salience” does not remove their parameters or establish a better estimator.

Introduce a minimal retrieval interface for both existing backends: analyze, expose corpus
statistics, and retrieve weighted terms. PRF additionally needs access to feedback document terms;
do not assume that the streaming backend's document IDs and scores supply this capability.
Any source requiring embeddings should declare that dependency rather than forcing all sources
to construct a dense vocabulary matrix.

## 7. Expansion budgets and the scope of the guarantee

### 7.1 What needs testing

V7 applies a query-dependent multiplier separately to each anchor. With one normalized candidate
channel, total added weight is bounded by μ(Q) times the sum of anchor weights. More anchors thus
allow more absolute expansion, but the relative expansion budget does not automatically grow.
Whether long queries benefit from less expansion is an empirical question, not a proven flaw.

Activating the existing η specificity multiplier rescales per-anchor allocations; it does not
implement a fixed or saturating global budget. Maximum query IDF is also sensitive to an isolated
rare term or typo. Compare it with direct lexical-coverage signals and a constant-budget control.

### 7.2 Explicit global budget proposal

Let original anchor weights be w_a ≥ 0 and A(Q) = Σ_a w_a. Define a candidate global budget in
**query-weight units**, before IDF damping:

`B(Q) = μ · f(Q) · min(A(Q), A_cap)`

Here 0 ≤ μ ≤ 1, 0 ≤ f(Q) ≤ 1, and A_cap is a declared saturation scale in the same units as
A(Q). Fix the anchor-weight convention so this scale is meaningful.

- A_cap = infinity and f(Q) = 1 recover proportional total budgeting.
- A finite A_cap makes total expansion saturate as anchor mass grows.
- Test f(Q) = 1 before introducing specificity or coverage damping.

Allocate anchor budgets b_a with Σ_a b_a ≤ B(Q) and b_a ≤ μ f(Q) w_a. Uniform/proportional shares
and salience-based shares are separate ablations; cap and redistribute shares when necessary.
If an anchor has no admitted candidates, allow its budget to remain unused. Redistribution is
optional and must respect all caps.

For candidate s supported by anchor a, use a normalized nonnegative allocation p(s|a):

`Δw(s|a) = b_a · p(s|a) · min(1, IDF(a)/IDF(s))`

This defines both the units and the per-anchor limits. Lower expansion per anchor on long queries
is a predicted behavior of the saturating policy, not the metric used to select a winner.

### 7.3 Precise mathematical claim

With positive, identical IDF values in allocation and retrieval, and Σ_s p(s|a) ≤ 1:

`Σ_s Δw(s|a) ≤ b_a`

`Σ_s Δw(s|a) · IDF(s) ≤ b_a · IDF(a)`

These bound query weights and IDF-weighted coefficients for any support size. They do **not**
unconditionally bound actual expansion BM25 scores by the original anchor's score in a document:
term frequencies and length normalization affect contributions, and the anchor may be absent.
For the implemented BM25 TF factor, an absolute bound can additionally use its upper bound k1+1.
Any stronger document-score or ranking claim must state and justify further assumptions.

Normalized association scores are allocation weights; they need not be calibrated probabilities.
The bound alone does not establish semantic correctness, ranking safety, or information-theoretic
optimality. Reconcile the existing theory documentation before promoting a stronger claim.

### 7.4 Multiple channels, aliases and numerical integrity

The current V7 code normalizes bailout and main-pool candidates separately and gives each channel
a μ-scaled allocation before adding their weights. When both channels contribute, their combined
budget can exceed the intended single-channel limit. This is a code-level issue independent of
the long-query hypothesis.

All channels must share one budget or receive explicit shares whose sum respects the cap.
Deduplicate indexed terms and retain attribution before final collision summation. Verify the
bound after phrase decomposition, source merging and numerical rounding; rounding upward can
violate a strict mathematical inequality.

Alias expansion may use a reserved share exempt from specificity damping, but it still consumes
a declared total budget and posting-cost allowance. Never provide an unlimited alias exemption.
Exact canonicalization is a separate analyzer decision requiring retrieval-parity validation.

## 8. Large-corpus latency: evidence, diagnosis and repair

The current streaming index already memory-maps postings and document lengths. Adding mmap is
therefore not a new fix. The implementation also allocates a float64 score vector of length N and,
for each active term, adds `np.bincount(..., minlength=N)`. This creates repeated corpus-sized
allocation and memory traffic even for short posting lists.

At roughly 5.4M documents, one such float64 vector is about 43 MB. The scorer consequently has
O(TN) dense accumulation work in addition to posting traversal, where T is the active term count.
This is a concrete candidate bottleneck, not proof of the measured latency breakdown.
Page faults, common-term postings, array conversions, top-K selection and ID lookup also need
measurement. See [streaming scorer](../src/pipeline_v2/indexer/streaming_posting_index.py).

Required diagnostic:

1. Replay identical original and expanded term vectors on the same index; separate source,
   encoding, admission, allocation, scoring, top-K and result-materialization time.
2. Record active terms, summed DF, postings touched, logical bytes accessed, temporary allocations,
   CPU time, page faults and physical I/O where measurable. Logical bytes are not physical reads.
3. Compare repeated-query warm-cache behavior and documented cold/cache-constrained conditions,
   under the same RAM ceiling. Record disk and filesystem placement.
4. Include original-query, V7-anchor-only and full-V7 controls; do not infer fan-out from a guessed
   30–80 terms or use the in-memory orchestrator as a proxy for streaming measurements.

Repair candidates, selected by profiling:

- Remove corpus-sized per-term temporaries using touched-document accumulation or an equivalent
  strategy that preserves weighted scores and exact top-K.
- Evaluate an optimized retrieval backend where justified, checking analyzer, IDF, weighting and
  ranking parity rather than assuming identical BM25 implementations.
- Add bounded caching only when measured page faults or repeated I/O justify it. Common posting
  lists are large; caching them must compete with other memory needs.
- Evaluate explicit posting-cost limits after establishing the unmodified quality control.

A small expansion weight still triggers posting traversal in the current scorer. Weight budgets
and computational budgets are therefore separate constraints. A budget change reduces cost only
when terms are actually pruned or the retrieval algorithm can safely skip their work.

Do not promise restoration to small-corpus latency. Report corpus-specific p50/p95, throughput and
quality tradeoffs after repair.

## 9. Evaluation design

### 9.1 Initial suite and later extensions

Initial controls:

- Standard BM25 and analyzed BM25 on pinned corpora and queries.
- Analyzed BM25 with V7's anchor weighting but no expansion.
- Frozen V7, including its effective configuration.
- Native RM3 and the existing analyzed/unified RM3 condition, clearly distinguished.
- Proposed variants changing one representation, relation or allocation policy at a time.

Later extensions include lexical-resource candidates, corpus association candidates, other PRF
variants, topic representations and generated expansion. Each is a separate named condition.
A generated pseudo-document projected into vocabulary terms must be labeled as that adaptation,
rather than assumed equivalent to the original generative retrieval method.

Report feasible frozen dense/learned-sparse references for system context. Keep published numbers
separate from local measurements; do not compute a “gap closed” percentage across incompatible
corpora, splits, analyzers, metrics or protocols.

### 9.2 Attribution and acceptance

Predeclare development and held-out evaluation partitions, tuning budgets, model versions and
resource limits. Do not tune on all benchmark queries and then describe the same results as
unseen evaluation. Existing calibration results are development evidence where labels influenced
selection.

Measure:

- nDCG@10 as the primary general ranking metric, Recall@10/50/1000, MRR@10, and Strict@10 as a
  hit metric rather than a substitute for precision.
- Per-dataset and macro results, per-query paired differences and uncertainty intervals.
- Query buckets: short/verbose, ambiguous/precise, exact entity, compound/phrase, lexical gap,
  and multiple subquestions. Length and specificity should be measured separately.
- Expansion count, support attribution, realized weight/IDF mass, unused budget and postings touched.
- Construction time, cold/warm startup, disk size, process and system memory, GPU peak memory,
  and p50/p95 end-to-end and component latency.

Preserve retrieval depth K=1,000 when using the existing evaluation harness. Chunk queries to
control memory rather than truncating rankings. Source cost includes the first retrieval round
for PRF and generation/projection for generated candidates.

For each source, compare the same candidate set under native/simple normalized weighting and
the proposed allocator. For each relation, match candidate and computational budgets where
possible. This separates candidate quality, original-query weighting and mass control.

Promote a variant only after a preregistered held-out effectiveness/resource criterion is met.
Choose the numerical improvement, regression tolerance and latency ceilings before running its
selection sweep, using the repaired baseline and intended deployment setting. Evidence of reduced
drift or better efficiency at comparable quality is also valid; universal wins are not required.

## 10. Implementation sequence and decision gates

This document authorizes a research direction for review; the stages below describe future work.

| Stage | Work | Required result before proceeding |
|---|---|---|
| 0. Establish the control | Audit effective config, entry points, analyzer/index parity and channel budgets; preserve archived runs | Reproducible frozen V7 plus separately labeled correctness repairs |
| 1. Resolve scaling | Profile and repair streaming scoring using replayed fixed query vectors | Score/top-K parity and measured component latency under the memory cap |
| 2. Minimal shared contracts | Separate evidence, allocation and retrieval; support V7 and PRF without mandatory dense initialization | Same frozen behavior through the new interfaces and explicit source dependencies |
| 3. Relation experiments | Anchor, whole-query and anchor-agreement policies over the existing vocabulary | Controlled evidence about query use before changing stored representations |
| 4. Better lexical units | Phrase/entity inventory with explicit mapping into indexed terms | Measured gain attributable to representation, including construction cost |
| 5. Context representations | Bounded sense/context prototypes; then topic or passage prototypes if justified | Effectiveness/setup/memory tradeoff supports further complexity |
| 6. Budget and integration | Proportional versus saturating budgets, coverage signals, bounded aliases and selected source validation | Held-out query-bucket results; final joint confirmation with frozen choices |

Fix demonstrable budget violations in Stage 0; keep heuristic budget-policy tuning distinct until
source/relation controls are established. Staged experiments may interact, so confirm the selected
combination jointly before freezing it.

Specific prerequisites already visible in the checkout:

- YAML sets `mass_floor: 0.005`, but the normal orchestrator does not forward it to the extractor.
- Bailout thresholds, allocation/gate options and effective defaults differ across configuration,
  constructor paths and documentation. Log the fully resolved runtime configuration.
- The normal orchestrator consumes an in-memory corpus; streaming retrieval has a separate path.
- Bailout and pool allocation need combined accounting.
- Archived result generators and dataset manifests must be located and pinned; references to a
  results-to-scripts mapping do not establish reproducibility when that file is absent.

Each implemented variant must include its co-located pathway specification, resolved configuration,
representation/cache identity, deterministic seeds, invariant checks and timestamped evidence.
Synchronize architecture and result documentation after verified implementation changes.
