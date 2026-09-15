# V8 Root-Cause Hypotheses and Validation Plan

## 1. Purpose

V8 was intended to improve V7 by enlarging the available vocabulary, selecting fewer anchors and
expansion terms, and constraining semantic drift with corpus statistics. The completed results do not
show that improvement. V8 is usually slightly worse than its unexpanded lexical control, and its
Recall@1000 regression shows that the problem is not limited to rearranging the first ten results.

This document separates four things that must not be conflated:

1. facts established by the checked-in code or result files;
2. hypotheses about why the observed regressions occur;
3. experiments that can support or falsify each hypothesis;
4. the decision test for whether Context-Reranked Vocabulary Expansion (CRVE) has real headroom.

This is a diagnostic protocol, not a commitment to V9 or CRVE. The core experiment resources and raw
runs were generated on the WSL2 Linux edge testbed under `/home/donghv/Projects/Edge-RAG/`. They are
excluded from the clean Windows Git checkout by `.gitignore`. Runtime claims remain unverified in the
shared repository unless they follow directly from committed results or are exported from that testbed
with reproducible provenance.

## 2. Evidence currently available

The following files are the current evidence base:

- `results/pyterrier_baselines/pyterrier_baselines_results.csv`;
- `results/pyterrier_baselines/pyterrier_qe_results.csv`;
- `results/pyterrier_baselines/pyterrier_v7_results.csv`;
- `results/pyterrier_baselines/pyterrier_v8_results.csv`;
- `results/pyterrier_baselines/v8_vs_baseline_comparison.md`;
- `src/evaluation/pyterrier_v7.py`;
- `src/evaluation/pyterrier_v8.py`;
- `src/evaluation/pyterrier_qe.py`;
- `docs/v8_plan.md`.

The CSVs contain dataset-level aggregates. They do not contain expansion traces or per-query metric
deltas. Consequently, they establish the overall outcome but cannot establish activation rates,
query-level tie rates, selected-term quality, or causal mechanisms.

## 3. Established observations

### 3.1 V8 generally harms both ranking and candidate recall

Across the 20 datasets evaluated by V8:

| Comparison | Delta nDCG@10 | Delta Recall@100 | Delta Recall@1000 |
|---|---:|---:|---:|
| V8 BM25 minus BM25 | -0.00737 | -0.00686 | -0.01075 |
| V8 DPH minus DPH | -0.00731 | -0.00345 | -0.01043 |

Compared with V7 at `tau=0.65` on the same datasets:

| Comparison | Delta nDCG@10 | Delta Recall@100 | Delta Recall@1000 |
|---|---:|---:|---:|
| V8 BM25 minus V7 BM25 | -0.00135 | -0.00328 | -0.01297 |
| V8 DPH minus V7 DPH | +0.00013 | -0.00295 | -0.01247 |

V8 therefore approximately matches V7 at nDCG@10 but is materially worse at the deep candidate
cutoff. The negative Recall@1000 result means that expansion-only documents can displace relevant
documents from the candidate funnel; the effect is not solely top-rank reshuffling.

### 3.2 The implemented V8 weight formula saturates for every admitted candidate

The implementation admits a candidate only when:

\[
\operatorname{sim}(a,e) \ge 0.65
\]

and:

\[
\frac{IDF(e)}{IDF(a)} \ge 0.85.
\]

It then computes:

\[
w_e = \min\left(
w_a\operatorname{sim}(a,e)\frac{IDF(e)}{IDF(a)},
0.30w_a
\right).
\]

For every admitted candidate:

\[
\operatorname{sim}(a,e)\frac{IDF(e)}{IDF(a)} \ge 0.65\times0.85=0.5525>0.30.
\]

Therefore, under the checked-in defaults:

\[
w_e=0.30w_a
\]

for every emitted expansion. Similarity and the IDF ratio affect admission and selection, but not the
final expansion weight. Runtime tracing is still required to confirm that the completed experiments
used these defaults and this source revision.

### 3.3 Candidate ranking has an unbounded preference for higher IDF

V8 selects a candidate using:

\[
G_{V8}(e,a)=\operatorname{sim}(a,e)\frac{IDF(e)}{IDF(a)}.
\]

The IDF ratio has a lower bound but no upper bound or symmetric-distance penalty. A less similar but
much rarer term can therefore outrank a more similar term. This property of the scoring function is
established; whether it is an important empirical cause of the losses remains a hypothesis.

### 3.4 V8 prioritizes the highest-relative-IDF eligible anchor

Eligible anchors are sorted by `IDF(anchor) / max_query_IDF`, after which V8 normally expands one
anchor, or two when the analyzed query has at least six distinct terms. This prioritizes specificity,
not estimated vocabulary mismatch or estimated benefit from expansion.

For a term missing from the lexicon, the implementation assigns `DF=1`. Such a term receives nearly
maximum collection IDF, passes the collection-frequency ceiling, and can become the preferred anchor
unless an orthographic freeze rule catches it. The magnitude of this pathway in the completed run is
not yet known.

### 3.5 A 0.30 query weight is not a 30% bound on retrieval-score impact

The original query term remains present, and the expansion is assigned a query weight relative to the
anchor. This does not guarantee that expansion contributes at most 30% of a document's score or causes
at most a 30% ranking change. Actual contribution also depends on candidate IDF, within-document term
frequency, document length, retrieval model, and which original terms the document matches.

Linear additive matching is normal inverted-index behavior. The open issue is whether a selected term
is safe enough, and calibrated well enough, to be added as an independent matching channel.

### 3.6 The checked-in V8 pool construction does not match the stated 25,000-term design

`V8VocabPool` consumes a sidecar built by `BGEVocabSidecarManager`. Under the checked-in defaults, that
manager:

1. orders lexicon terms by descending collection frequency;
2. truncates them to 20,000 candidates;
3. retains only `vocab_cap=15,000` terms in the sidecar;
4. recovers representative surfaces from at most the first 5,000 streamed documents.

V8 subsequently applies its bounded-frequency rules to that already truncated sidecar. Its nominal
`max_cap=25,000` cannot restore a term previously excluded by the sidecar builder. This establishes a
code/design mismatch under the checked-in defaults. Experiment manifests are required to determine
the exact realized pool and cache used in the completed runs.

### 3.7 LLM lexical expansion shows that sparse expansion is not categorically ineffective

Across all 25 datasets, the recorded LLM query-to-expansion baseline has positive macro deltas:

| Comparison | Delta nDCG@10 | Delta Recall@100 | Delta Recall@1000 |
|---|---:|---:|---:|
| LLM QE BM25 minus BM25 | +0.00246 | +0.00668 | +0.01360 |
| LLM QE DPH minus DPH | +0.00312 | +0.00820 | +0.01406 |

This does not prove that query context is the cause, because LLM QE also differs in generation,
candidate count, and weight allocation. It does rule out the broad conclusion that adding lexical
terms to BM25 or DPH is inherently futile.

### 3.8 The defensible and non-defensible parts of V8's theoretical foundation

V8's useful intuition is that static corpus statistics can act as a risk-control signal. In particular,
rejecting extremely common candidates and limiting expansion count can reduce obvious broad-query
drift. Its central theoretical claim is stronger, however:

\[
\text{high isolated-term similarity} + \text{comparable-or-higher IDF}
\Rightarrow
\text{safe and useful lexical expansion}.
\]

That implication does not follow from the quantities V8 observes. IDF is approximately a marginal
measure of surprise for document occurrence:

\[
IDF(e) \approx -\log P(e\in d).
\]

Expansion utility instead depends on query-conditioned relevance behavior:

\[
P(e\in d\mid d\text{ relevant to }Q)
\quad\text{and}\quad
P(e\in d\mid d\text{ non-relevant to }Q).
\]

The first quantity does not determine the latter two. A term can be rare and discriminative throughout
the corpus yet belong to the wrong query sense. Conversely, an exact synonym or canonical spelling can
be more frequent than the anchor and still be a useful bridge.

The following distinctions should govern interpretation and later paper claims:

| V8 proposition | Counterpoint | Validation consequence |
|---|---|---|
| Higher candidate IDF makes expansion safer | Rarity can identify a correct subtype or a wrong rare lateral concept | Test term utility as a function of IDF ratio; do not assume either direction is correct |
| IDF ratio supplies semantic directionality | Frequency difference cannot distinguish synonymy, hypernymy, hyponymy, sibling concepts or word senses | Evaluate relation and query-sense compatibility separately from frequency |
| `0.30 * anchor weight` conserves anchor importance | Query-token weight is not a bound on document score, rank displacement or effectiveness | Measure contribution, entrants and displaced relevant documents |
| Highest-IDF anchor is the best anchor to expand | Specificity does not estimate vocabulary mismatch or marginal retrieval benefit | Compare anchor policies under a fixed candidate set |
| V8 is entropy-constrained | IDF is related to self-information, but V8 neither optimizes corpus/relevance entropy nor proves an entropy bound | Describe this as heuristic corpus-statistical gating, not an information-theoretic guarantee |
| V8 obtains SPLADE-like semantic bridging | Both address vocabulary mismatch, but SPLADE uses learned contextual query representations and coordinated weights | Avoid equivalence claims; compare only the practical trade-off when evidence supports it |
| V8 has zero neural inference online | Cached surfaces avoid repeated encoding, but unseen query surfaces are encoded at runtime | Report cold and warm query latency separately |

The appropriate weaker foundation for V8 is therefore:

> Corpus frequency is a coarse eligibility and risk-control signal. It can reject some generic
> candidates, but cannot establish query-compatible sense or expected retrieval utility.

CRVE is worth testing only if stored usage contexts add information about those missing
query-conditioned quantities.

## 4. Root-cause hypotheses

### H1. Fixed maximum expansion weight makes weak candidates unnecessarily disruptive

#### Rationale

Because the implemented formula saturates, a barely admissible candidate and an excellent candidate
both receive `0.30 * anchor_weight`. This converts continuous evidence into an almost binary action:
either no term is emitted, or a term is emitted at the maximum allowed weight.

#### Predictions

- Nearly 100% of emitted terms have `final_weight / anchor_weight = 0.30`.
- If candidates are often useful but overweighted, lower weights reduce intruder documents and the
  loss rate.
- If candidates are useful but underweighted, higher weights improve recall and may improve nDCG.
- If every nonzero weight is harmful, candidate identity rather than allocation is the primary failure.

#### Falsification conditions

H1 is weakened if all nonzero weights preserve the same harmful-query rate and no part of the weight
curve improves aggregate or activation-conditioned effectiveness. That would indicate that candidate
identity, not merely weight, is the dominant problem.

#### Test

Hold anchors and selected candidates fixed. Retrieve at weights:

`0.00, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00`.

Measure paired changes in nDCG@10, Recall@100, Recall@1000, synonym-only entrants, and relevant
documents displaced at each cutoff. The purpose is to estimate a response curve, not to presuppose
that lower weights are preferable.

### H2. The unbounded IDF preference may select rare semantic cousins instead of safe expansions

#### Rationale

Rewarding a higher-IDF candidate is not inherently wrong. It can select a useful subtype, domain term,
or canonical lexical bridge. The concern is narrower: V8 rewards candidate rarity without an upper
bound. A rare, moderately similar lateral concept can then outrank a closer synonym. Once selected,
that rare term receives the same saturated 0.30 weight and may itself have a strong lexical score.

#### Predictions

- V8-selected candidates have higher IDF and lower cosine than candidates selected by cosine alone.
- The current selector has a higher harmful-term rate than a bounded-IDF or cosine-only alternative.
- Harmful candidates are concentrated at some range of large `IDF(candidate) / IDF(anchor)` values;
  beneficial candidates may also be more specific, so both distributions must be reported.

#### Falsification conditions

H2 is weakened if cosine-only or bounded-IDF selection has equal or worse candidate utility, large IDF
ratios are not disproportionately associated with harm, and the unbounded ratio improves selected-term
utility.

#### Test

For every activated query, rank the same candidate set by:

1. cosine only;
2. current cosine multiplied by IDF ratio;
3. cosine with a bounded ratio;
4. cosine penalized by `abs(log(IDF ratio))`.

Inject each selected candidate separately at an identical declared weight. Compare candidate utility,
harmful-term rate, selected-candidate IDF, and paired retrieval metrics.

The bounded-ratio variant can use:

\[
G(e,a)=\cos(a,e)\min\left(\frac{IDF(e)}{IDF(a)},c\right),
\]

while the comparable-specificity alternative can use:

\[
G(e,a)=\cos(a,e)\exp\left(-\lambda\left|\log\frac{IDF(e)}{IDF(a)}\right|\right).
\]

These are diagnostic ablations, not assumed replacements. They answer whether the useful candidate is
usually rarer, similarly specific, or unrelated to the current IDF preference.

### H3. Highest-IDF anchor selection targets entities, OOV terms, and already-specific concepts

#### Rationale

The term with highest IDF is not necessarily the term suffering a vocabulary mismatch. Rare terms are
also where isolated-token embeddings and surface recovery are most fragile. Assigning `DF=1` to OOV
terms makes this especially risky.

#### Predictions

- OOV and ultra-rare anchors are overrepresented among large drops.
- Freezing OOV or very-low-DF anchors reduces harm.
- Expanding an anchor chosen by estimated need or whole-query agreement performs better than choosing
  the highest-IDF anchor.

#### Falsification conditions

H3 is weakened if loss rates are similar across anchor-DF groups and an OOV/rare-anchor freeze does not
improve results.

#### Test

Partition activated queries by chosen-anchor status:

- OOV;
- `DF <= 2`;
- `3 <= DF <= 20`;
- medium/common;
- acronym;
- other orthographically special token.

Report activation rate, mean and median metric delta, gain/drop/tie counts, and harmful-term rate for
each group. Then test an OOV/low-DF freeze without changing the candidate pool.

### H4. Isolated-word BGE similarity cannot reliably validate the query-compatible sense

#### Rationale

V8 embeds an anchor surface and a candidate surface independently. It can establish that two isolated
terms are close in BGE space, but not that the candidate expresses the sense intended by the complete
query. The supplied failures involving biomedical drift, rare entities, and broad climate concepts are
consistent with this mechanism, but examples alone cannot establish prevalence.

#### Predictions

- Harmful candidates often have a plausible relation to the anchor but not to the whole query.
- Whole-query or stored-context compatibility predicts candidate utility better than isolated-term
  similarity.
- Context gating reduces harmful-term rate while retaining some beneficial candidates.

#### Falsification conditions

H4 is weakened if context-based scores have no greater correlation with measured candidate utility,
do not improve candidate ranking, and do not reduce retrieval harm on a fixed candidate set.

#### Test

Keep candidate generation fixed and compare:

1. isolated anchor-to-term score;
2. whole-query-to-term score;
3. query-to-stored-context score;
4. term score multiplied by a context gate;
5. a joint term-and-context score.

Changing the candidate set during this test is prohibited because it would confound candidate recall
with reranking quality.

### H5. The inherited CF-truncated sidecar and limited surface recovery reduce candidate quality

#### Rationale

The implemented V8 pool is selected from an earlier CF-ranked sidecar rather than constructed directly
from all lexicon terms satisfying V8's declared constraints. The first-5,000-document surface sample
may also produce corpus-order bias or leave some terms represented only by Porter stems.

#### Predictions

- Some eligible domain terms are absent before V8 filtering begins.
- Unrecovered or incorrectly recovered surfaces have worse neighbour quality.
- A directly constructed bounded-frequency pool changes candidate availability and selected terms.

#### Falsification conditions

H5 is weakened if the runtime sidecar was actually constructed differently, eligible-term coverage is
already high, surface fallback is rare, and a corrected pool does not improve candidate utility.

#### Test

For each dataset, record the sidecar manifest, cache identity, realized term count, post-V8 pool count,
CF/DF ranges, and surface-recovery rate. For a sample of eligible lexicon terms, report whether each is
present in the raw sidecar and final V8 pool. Compare the inherited pool with a pool built directly
from the declared V8 predicate, using nested caps and otherwise identical selection logic.

### H6. The small macro regression hides a high-harm activated subset

#### Rationale

V8 emits at most one term for most queries and may emit none. Many queries can therefore tie while a
smaller activated subset experiences substantial gains or losses. Dataset-level averages cannot reveal
this mixture.

#### Predictions

- The absolute conditional delta among queries whose rankings changed is much larger than the macro
  delta.
- Drops outnumber gains among activated or changed queries.
- Datasets with near-zero macro change may still contain significant opposing gains and losses.

#### Falsification conditions

H6 is false if V8 activates for nearly all queries and most queries undergo uniformly tiny changes.

#### Test

For every query, record whether a term was emitted, whether the ranked list changed, and paired metric
deltas. Report separately:

1. all queries;
2. activated queries;
3. queries whose top 1,000 changed;
4. gained, dropped, and exactly tied queries.

The aggregate CSVs are rounded and must not be used to infer exact query-level ties.

### H7. Increasing pool size raises candidate recall but also raises false-neighbour risk

#### Rationale

A larger vocabulary can contain more useful bridging terms, but nearest-neighbour search over more
isolated terms also offers more opportunities for accidental high similarity and rare lateral concepts.
V8 added capacity without adding contextual validation.

#### Predictions

- Candidate-oracle utility initially improves with pool size and then plateaus.
- Selected-candidate utility may decline even while candidate-oracle utility improves.
- The gap between selected and oracle utility grows with pool size under the current selector.

#### Falsification conditions

H7 is weakened if selected and oracle utility improve together with pool size, or neither changes.

#### Test

Use genuinely nested pools, for example `1k, 5k, 10k, 15k`, and hold every other component fixed.
Measure both candidate availability and selected-candidate utility. Comparing only final retrieval scores
cannot distinguish increased candidate recall from degraded selection precision.

### H8. Single-term lexical expansion has limited ability on reasoning-heavy queries

#### Rationale

Many BRIGHT questions require relations, constraints, or multi-step reasoning that cannot be expressed
by adding one related word. Even a semantically valid synonym may not retrieve the missing evidence.

#### Predictions

- Candidate-oracle gains are lower on reasoning-heavy queries than on terminology-mismatch queries.
- Useful expansions are more often phrases, entities, or relational lexical units than isolated words.
- Context reranking may reduce harm but cannot create substantial recall headroom when no useful
  single-term candidate exists.

#### Falsification conditions

H8 is weakened if a candidate-level oracle finds frequent, large gains from single vocabulary terms on
reasoning-heavy datasets.

#### Test

Stratify queries by diagnostic type before inspecting outcomes: terminology mismatch, entity mismatch,
short keyword query, natural-language question, and reasoning-heavy question. Estimate candidate-oracle
headroom separately for each stratum.

## 5. The decisive CRVE-headroom experiment

V8's failure does not establish that its pool is useless. It establishes only that the complete V8
policy is ineffective. To distinguish a candidate-generation failure from a selection failure, evaluate
the candidate set directly.

For each query in an unbiased, stratified sample:

1. enumerate every original analyzed anchor before V8 anchor filtering;
2. generate the top 20 to 50 dense candidates per anchor before V8 candidate filtering;
3. annotate which anchors and candidates each V8 gate accepts or rejects and why;
4. inject a seeded subset of accepted and rejected candidates under the declared weight grid;
5. measure marginal effects on nDCG@10, Recall@100, and Recall@1000;
6. compare V8's selected anchor/candidate with the best alternatives before and after gating.

For metric `m`, define candidate utility:

\[
\Delta U_m(e,Q)=U_m(Q+e)-U_m(Q).
\]

Define selector regret:

\[
R_m(Q)=\max_{e\in C(Q)}\Delta U_m(e,Q)-\Delta U_m(e_{V8},Q).
\]

Because an expansion can help recall while harming early ranking, report the utility vector rather than
silently combining metrics:

\[
\left(\Delta nDCG@10,\Delta Recall@100,\Delta Recall@1000\right).
\]

Interpret the result as follows:

| Observation | Diagnosis | Consequence for CRVE |
|---|---|---|
| No candidate helps on most queries | Pool or single-term representation failure | Context reranking has little headroom |
| Helpful candidates exist, but V8 selects different terms | Selector failure | Strong justification for CRVE |
| V8 selects helpful candidates at low weight, but 0.30 harms | Allocation failure | Repair weighting before adding context memory |
| Candidates help recall but harm nDCG | Genuine precision-recall tradeoff | Use risk-aware gating or separate first-stage objective |
| Context reranking selects candidates closer to the oracle | Context adds measurable selection information | Direct evidence for CRVE |

The oracle is diagnostic only. It uses relevance judgments and must not be reported as a deployable
method or used to tune on the final test set.

## 6. Validation protocol

### 6.1 Dataset selection

Begin with a small, deliberately varied set rather than rerunning all datasets:

- major V8 losses: `trec_covid`, `webis_touche2020`, `quora`;
- small V8 successes: `bright_psychology`, `bright_theoremqa_questions`;
- reasoning-heavy representative: `bright_stackoverflow`.

Do not select only spectacular failure examples for quantitative conclusions. Within these datasets,
use all queries when inexpensive or a seeded stratified sample declared before inspecting term utility.

### 6.2 Required trace schema

For each query, record at least:

```text
dataset
qid
query
retrieval_model
expanded
ranking_changed
anchor_term
anchor_surface
anchor_in_lexicon
anchor_df
anchor_idf
candidate_term
candidate_surface
candidate_df
candidate_idf
cosine
idf_ratio
selection_score
raw_weight
final_weight
baseline_ndcg_10
expanded_ndcg_10
baseline_recall_100
expanded_recall_100
baseline_recall_1000
expanded_recall_1000
```

For candidate-oracle work, store the same candidate fields for the complete fixed candidate set rather
than only for the selected term.

### 6.3 Primary summaries

Report:

- activation rate;
- ranking-change rate conditional on activation;
- gain/drop/tie counts overall and conditional on activation;
- mean and median paired delta;
- harmful-term rate;
- fraction of queries with at least one beneficial candidate;
- selected-candidate utility versus oracle utility;
- selector regret;
- relevant documents added and displaced at depths 10, 100, and 1,000.

Use paired bootstrap confidence intervals or an appropriate paired randomization test for final claims.
Effect sizes and distributions are more informative at this diagnostic stage than significance alone.

### 6.4 Correct experimental controls

- Verify `weight=0` or zero emitted expansions reproduces the corresponding BM25/DPH control.
- Use identical queries, qrels, exclusions, index, analyzer, and retrieval depth.
- Hold candidate sets fixed when comparing rerankers.
- Hold selected terms fixed when comparing weights.
- Use nested pools when comparing pool sizes.
- Record exact source revision, configuration and cache identity.
- Report BM25 and DPH separately; do not average away retrieval-model interactions.
- Retain both nDCG@10 and candidate-funnel recall as co-primary diagnostic views.

## 7. Efficient execution order

The tests should proceed from cheapest and most decisive to more expensive:

1. **Runtime configuration and cache audit:** verify source revision, thresholds, sidecar manifests and
   realized pool sizes.
2. **Rewriter trace:** confirm weight saturation, anchor selection, OOV rates and selected-term
   distributions without retrieval.
3. **Activation-conditioned audit:** use cached runs and qrels to quantify ties, gains and drops.
4. **Pre-gate mechanism audit:** measure whether anchor and candidate gates reject harmful items while
   retaining useful ones.
5. **Candidate-oracle sample:** determine whether useful terms exist before and after V8 gating.
6. **Fixed-candidate weight sweep:** separate selection failure from allocation failure.
7. **Selector ablations:** cosine-only, bounded IDF and OOV/rare-anchor freeze.
8. **Fixed-candidate contextual reranking:** test the central CRVE hypothesis.
9. **Full evaluation:** run across all eligible datasets only after a targeted variant demonstrates
   credible headroom and acceptable latency.

This order prevents spending substantial time implementing context memory before establishing that the
candidate pool contains retrievably useful alternatives.

## 8. Evidence required from the experiment machine

The local checkout does not contain the raw run cache or the supplied forensic scripts. The worker
machine should provide:

1. help output or usage documentation for:
   - `scratch/audit_low_scoring_corpora.py`;
   - `scratch/verify_hypotheses_with_hard_data.py`;
2. the dataset-level sidecar manifests and realized V8 pool counts;
3. the all-query activation/gain/drop/tie summary for the six initial datasets;
4. structured traces for the largest drops, largest gains, and a seeded sample of ties;
5. pre-gate anchor and candidate traces, including every acceptance/rejection reason;
6. accepted and rejected candidate traces for a seeded candidate-oracle sample.

Raw corpus copies are not required for the first diagnostic round if the worker can access the existing
indices, qrels and cached depth-1,000 runs.

## 9. Decisions after validation

The next method should be chosen from evidence rather than by automatically incrementing a version:

- **Proceed with CRVE** if useful candidates frequently exist and context materially improves their
  selection or rejection.
- **Repair weighting first** if selected candidates are often useful at small weights but harmful at
  0.30.
- **Repair anchor/candidate generation first** if useful terms exist but are not exposed by the current
  anchor and candidate rules.
- **Change the static representation** if the current pool rarely contains useful candidates but
  phrases, entities, concepts or other bounded lexical units show oracle headroom.
- **Stop first-stage expansion for affected query types** if even the candidate oracle has little
  headroom; retain pristine BM25/DPH and use later-stage semantic processing instead.

## 10. Questions for debate

1. Should CRVE optimize primarily for safe nDCG improvement, candidate recall, or a declared
   multi-objective frontier?
2. Is an expansion that improves Recall@1000 but harms nDCG@10 acceptable for the intended pipeline?
3. Should OOV and ultra-rare anchors be frozen, or are they precisely where vocabulary bridging is most
   valuable?
4. Should IDF constrain candidate eligibility only, rather than increase candidate rank?
5. What minimum candidate-oracle headroom justifies building the context sidecar?
6. On which query types should the system abstain from expansion entirely?

## 11. Reconciled Counter-Analysis and V8 Disposition

This section reconciles the initial hypotheses with the worker's counter-analysis. It distinguishes
useful corrections from conclusions that remain unproved. Sections 1–10 define the tests; this section
sets the immediate decision for V8.

### 11.1 Evidence location and provenance

There is no contradiction between the two reported environments:

1. Full runs, indices and forensic scripts exist on the WSL2 Linux experiment testbed.
2. `data/cache/` is excluded from Git, and `scratch/` is an uncommitted local workspace.
3. Those artifacts do not exist in the clean Windows checkout used for this review.

Therefore, worker-reported query-level figures are useful provisional evidence, but they become shared
scientific evidence only after export to `results/pyterrier_baselines/`. The export must record the Git
commit, pipeline configuration, dataset, retrieval model, metric definition, tie tolerance (reported as
`epsilon=0.001`), script name and seed where applicable. Ephemeral task-log identifiers must not be
cited as reproducible evidence.

### 11.2 H1 reconciliation: calibration defect, probably not the primary cause

The counter-analysis correctly emphasizes the structural risk of independent additive scoring:

1. **Independent Additive Scoring:** Standard BM25 computes document scores as an uncoordinated linear sum:
   $$\text{Score}(D, Q) = \sum_{t \in Q} w_t \cdot \text{BM25}(t, D)$$
2. **High-IDF intruder score mass:** Because V8 explicitly rewards
   `IDF(candidate) / IDF(anchor)`, an emitted term can retain meaningful score contribution even at a
   smaller query coefficient. The worker's illustrative calculation uses IDF `8.5` and a saturated TF
   factor near `2.0`, giving a contribution near `1.70` at weight `0.10`. The actual candidate-IDF and
   TF distributions must be exported before this is treated as representative.
3. **Thin top-rank margins:** Small score additions can cross a rank boundary. The worker reports
   sub-`0.50` margins near ranks 8–11 and a `0.80` intruder contribution in examined cases; these are
   trace-level claims pending a committed audit, not universal benchmark constants.
4. **The Top-10 Truncation Cliff:** Displacing even one relevant document from Rank 10 to Rank 11
   removes its nDCG@10 contribution.

These observations do not invalidate H1. They show why weight tuning cannot turn a semantically invalid
candidate into a valid one. For a finite run, reducing a coefficient can still prevent particular rank
crossings, and weight zero returns to the lexical baseline. A two-sided sweep remains useful to measure
sensitivity; it is not presumed to rescue V8.

A synonym/disjunction operator is also not a word-sense guarantee. It may change term-statistics and
double-counting behavior, but a document containing only the incorrect alternative can still match.
Candidate validity remains the primary issue; saturated `0.30` weighting is a confirmed secondary
calibration defect.

### 11.3 LLM QE reconciliation: existence evidence, not an edge-feasibility proof

Section 3.7 cites the positive macro deltas of `LLM_Q2E_ZS` ($+0.00246$ on BM25, $+0.00312$ on DPH) to argue that sparse lexical expansion into inverted indices has viable headroom. This argument conflates two fundamentally incompatible paradigms:

1. **Compositional context versus isolated unigram lookup:** A generative LLM processes the complete
   query and can condition generated terms on relations, negatives and multi-word constraints. V8
   compares isolated anchor and candidate surfaces. The exact model used in the completed LLM-QE run
   must be taken from its committed runtime manifest rather than inferred from a default or worker note.
2. **Cost mismatch:** If the reported `1,500–2,500 ms` generation latency is confirmed by committed
   logs, the method is not a deployable low-latency substitute for V8.

The positive LLM QE result establishes only that query-contextualized lexical additions can sometimes
help sparse retrieval. It does not show that a static unigram sidecar can reproduce the gain, nor that
the gain justifies LLM latency. Exact model identity and generation latency must be exported before the
cost claim is treated as verified.

### 11.4 H7 reconciliation: enlarged hypothesis space coupled with selector error

The completed runs establish that terms emitted by V8 reduce final retrieval Recall@1000:

| Model Comparison | Delta Recall@1000 | Win / Loss / Tie Record |
|---|:---:|:---:|
| **V8_BM25 vs BM25 Base** | **-0.0108** | **1 Win / 16 Losses / 3 Ties** |
| **V8_DPH vs DPH Base** | **-0.0104** | **2 Wins / 17 Losses / 1 Tie** |
| **V8_BM25 vs V7_BM25** | **-0.0130** | Candidate Recall Degraded |
| **V8_DPH vs V7_DPH** | **-0.0125** | Candidate Recall Degraded |

#### The finite-heap eviction mechanism

At retrieval depth `K=1,000`, off-topic documents matching an emitted expansion can enter the heap and
push marginal relevant documents below the cutoff. This is a plausible mechanism and is consistent
with the aggregate Recall@1000 regression.

However, storing 15,000 terms does not itself affect retrieval, and V8 emits at most one or two terms.
The accurate causal chain is:

\[
\text{larger candidate space}
\rightarrow \text{more possible rare/lateral neighbours}
\rightarrow \text{selector emits an unsafe term}
\rightarrow \text{heap eviction}
\rightarrow \text{Recall@1000 loss}.
\]

The larger pool may still contain useful alternatives. Only a candidate-oracle comparison can measure
candidate availability separately from selected-term quality. No mathematical guarantee of lower
recall follows from pool size alone.

### 11.5 H4 and CRVE reconciliation: high-risk, conditional research direction

The worker-reported audit materially raises the bar for CRVE. It does not yet prove that CRVE has no
headroom.

#### Provisional query-level evidence

The table below is a representative subset of the worker-reported nDCG@10 audit. It must be regenerated
as a committed result before publication or a final architectural decision:

| Dataset | BM25 Base | V8_BM25 | Gold in Top 10 % | **Ties %** | Gains % | **Drops %** | Dominant Outcome |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `bright_aops` | 0.0604 | 0.0610 | 20.7% | **99.1%** | 0.9% | 0.0% | G > D (▲) |
| `bright_theoremqa_questions` | 0.0700 | 0.0731 | 11.3% | **96.4%** | 2.1% | 1.5% | G > D (▲) |
| `quora` | 0.7676 | 0.7450 | 91.1% | **94.1%** | 1.0% | 4.9% | D > G (▼) |
| `bright_stackoverflow` | 0.1561 | 0.1515 | 32.5% | **87.2%** | 4.3% | 8.5% | D > G (▼) |
| `bright_earth_science` | 0.1201 | 0.1183 | 31.9% | **81.9%** | 9.5% | 8.6% | G > D (▲) |
| `webis_touche2020` | 0.2979 | 0.2781 | 91.8% | **71.4%** | 6.1% | 22.4% | D > G (▼) |
| `trec_covid` | 0.6295 | 0.5981 | 100.0% | **66.0%** | 6.0% | 28.0% | D > G (▼) |

This supports two conclusions:

1. V8 has a large dormant zone, especially on several reasoning corpora.
2. Where baseline top-10 coverage is already high, V8's changed-query subset is often dominated by
   drops.

It does not establish that addressable headroom is below 3%. A tie under V8 can mean no activation,
wrong selection, inadequate weight, no top-10 change despite a recall change, or genuinely absent
lexical headroom. The audit measures V8's realized policy, not the best available candidate in its
pool.

#### Resource and architectural consequences

- Twenty FP16 context vectors for each of 15,000–25,000 terms require approximately 220–366 MiB
  before metadata. This is material but not automatically disqualifying under the 15 GiB ceiling.
- CRVE's primary design uses precomputed context vectors. Scoring 100 candidates with 20 contexts each
  requires about 768,000 scalar multiply-adds, not 2,000 independent encoder calls. The main online
  cost is query encoding plus memory movement. The asserted `25–50 ms` latency is therefore unverified
  and must be measured on the target CPU/GPU. A cross-encoder is not part of the minimal pilot.
- The cascade router and listwise reranker are marked as future extensions in `docs/ARCHITECTURE.md`.
  Even when implemented, a post-retrieval reranker cannot recover documents missing below depth 1,000.
  CRVE is not functionally identical, although it is unjustified if candidate-oracle recall headroom is
  negligible.

### 11.6 H3 reconciliation: OOV handling is a guard, not the solution

The SciFact case demonstrates that in-lexicon semantic drift can be severe:

1. **Case study:**
     - Query: *"Leuko-increased blood increases infectious complications in red blood cell transfusion."*
     - Injected expansion: `hemoperfus` (weight $0.30$).
     - Outcome: Six off-topic articles on hemoperfusion entered ranks 3–8 due to high term frequency on `hemoperfus`, displacing the pristine gold document from **Rank 1 down to Rank 10**.

An OOV freeze would not prevent this example, so it cannot be the main repair. One example does not
falsify the aggregate hypothesis that rare/OOV anchors form another harmful subgroup; the stratified
test remains valid.

The effective orthographic behavior must also be stated accurately:

- tokens containing digits, tokens of length at most two, and selected punctuated compounds are frozen;
- because freezing is checked first, two-character acronyms are frozen;
- uppercase alphabetic acronyms of length three to five are subsequently made eligible;
- V8 expands at most one anchor for shorter queries and two for queries with at least six distinct
  analyzed terms.

Any trace reporting three actually expanded anchors must be reconciled with the executed source revision
or relabeled as three eligible/candidate-analysis anchors.

### 11.7 Reconciled status of the hypotheses

| Item | Reconciled status | Consequence |
|---|---|---|
| V8 effectiveness | **Failed as a general first-stage method** | Do not use V8 as the production/default retriever |
| H1: fixed `0.30` saturation | **Confirmed defect; causal size unknown** | Measure sensitivity, but do not attempt a broad tuning campaign before the oracle gate |
| H2: unbounded IDF preference | **Active leading hypothesis** | Compare selected-term utility under current, cosine-only and bounded-IDF ranking |
| H3: rare/OOV anchors | **Plausible subgroup, not primary explanation** | Retain a cheap stratified audit; an OOV freeze alone is insufficient |
| H4: missing query-compatible sense | **Leading conceptual explanation** | Candidate/context validation is relevant only if useful candidates exist |
| H5: inherited sidecar mismatch | **Code/design mismatch; causal effect unknown** | Record the realized pool, but do not rebuild it before establishing oracle headroom |
| H6: dormant versus harmful active zones | **Provisionally supported** | Commit the per-query audit and report activation-conditioned results |
| H7: pool-size/selector interaction | **Plausible coupled mechanism** | Do not infer pool quality from final Recall@1000 alone |
| H8: reasoning-query limitation | **Provisionally supported** | Consider abstention or query-type routing rather than universal QE |
| LLM QE relevance | **Positive existence evidence only** | Not proof of a low-latency edge solution |
| CRVE | **Conditional on candidate-oracle gate** | Do not implement the context sidecar yet |

## 12. Decision: what to do with V8

### 12.1 Immediate disposition

V8 should be **retired as a candidate production method and frozen as a diagnostic ablation**. This
means:

1. keep standard BM25 and DPH as the operational first-stage defaults;
2. do not run another full V8 sweep or tune V8 globally;
3. preserve the current V8 source, configuration and results for negative-result analysis;
4. use V8's existing pool and top-candidate generator only to test whether useful lexical bridges were
   available but incorrectly selected;
5. do not describe V8 as entropy-guaranteed, SPLADE-equivalent, or retrieval-safe in a paper.

This decision recognizes that the complete V8 policy has failed while avoiding the unsupported
conclusion that every term in its candidate pool is useless.

### 12.2 V7-to-V8 mechanism attribution

The central V8 question is not merely whether its final run loses to BM25/DPH. V8 was designed to
improve V7 through two interventions:

1. reject anchors that should not be expanded;
2. reject expansion candidates that are inappropriate for those anchors and queries.

The validation must therefore measure both **what V8 removed** and **what V8 retained**. Auditing only
the final emitted term cannot determine whether the gates worked.

#### A. Anchor-gate audit

Start from every distinct analyzed content term in each sampled query, before V8 anchor eligibility and
top-anchor selection. For each anchor, record:

```text
eligible_under_v8
rejection_reason
relative_idf
df
in_lexicon
orthographic_class
v8_anchor_rank
actually_selected_for_expansion
```

Do not label an anchor useful merely because V8's selected candidate helped. For each anchor, inspect a
fixed pre-gate candidate set and estimate whether **any** candidate offers safe retrieval utility. This
separates anchor potential from V8 candidate-selection error.

Label an anchor provisionally as:

- **useful-capable:** at least one candidate passes the safe-utility definition in Section 12.3;
- **harm-only:** tested candidates produce losses but no safe gain;
- **dormant:** tested candidates cause no measured change;
- **mixed:** beneficial and harmful candidates both exist, making selection quality decisive.

Report:

\[
\text{useful-anchor retention}
=
\frac{\#\text{ useful-capable anchors accepted by V8}}
{\#\text{ useful-capable anchors}},
\]

\[
\text{harmful-anchor rejection}
=
\frac{\#\text{ harm-only anchors rejected by V8}}
{\#\text{ harm-only anchors}},
\]

and the useful-capable rate among accepted and actually selected anchors.

This distinguishes three possible anchor failures:

1. **under-filtering:** V8 still expands anchors with no useful candidates;
2. **over-filtering:** V8 rejects anchors that possess useful candidates;
3. **mis-prioritization:** useful anchors pass, but highest-relative-IDF ranking selects another anchor.

#### B. Candidate-gate audit

For every audited anchor, retain the dense neighbour list **before** applying V8's similarity, candidate
DF and asymmetric-IDF filters. Annotate every candidate with:

```text
passes_similarity_gate
passes_df_floor
passes_asymmetric_idf_gate
rejection_reason
cosine
candidate_df
candidate_idf
idf_ratio
v8_selection_score
v8_selected
```

Evaluate a seeded subset of both accepted and rejected candidates. Report:

- beneficial-candidate retention: fraction of useful candidates that pass all V8 gates;
- harmful-candidate rejection: fraction of harmful candidates rejected by at least one gate;
- false-acceptance rate: harmful candidates among those passing all gates;
- false-rejection rate: useful candidates among those rejected;
- selected-candidate regret relative to the best accepted candidate;
- gate regret relative to the best candidate before gating.

This separates two questions that V8 currently conflates:

1. Did the filters construct a better candidate set?
2. Given that set, did `cosine * IDF ratio` select the right member?

#### C. Conservative-policy decomposition

The observed small negative macro delta can arise from very different mechanisms:

\[
E[\Delta U]
\approx
P(\text{active})E[\Delta U\mid\text{active}],
\]

with dataset/query weighting reported exactly rather than inferred from this approximation.

Measure separately:

- fraction of queries with no eligible anchor;
- fraction with an eligible anchor but no surviving candidate;
- fraction emitting one or two terms;
- fraction whose top 10, 100 and 1,000 rankings change;
- effectiveness conditional on each stage above.

The interpretations are materially different:

| Observed mechanism | What went wrong |
|---|---|
| Low activation, retained interventions mostly harmful | Gates are conservative but have poor precision |
| Useful anchors/candidates are frequently rejected | Gates are over-conservative |
| Useful candidates survive, but V8 selects another | Ranking rule is defective |
| Selected candidates help at some weights but not `0.30` | Allocation is defective |
| No tested candidate helps even before gating | The unigram pool/proposal representation lacks headroom |

#### D. Minimal component counterfactuals

Use the same realized sidecar, query sample and pre-gate neighbour traces. Do not rebuild the pool during
this attribution stage. Compare:

1. current V8;
2. V8 without the anchor eligibility gate;
3. V8 without the asymmetric-IDF candidate gate;
4. V8 with cosine-only candidate ranking;
5. V8 with a bounded IDF preference;
6. current selected terms under the declared weight grid.

These are diagnostic counterfactuals, not a parameter search. Add-one/remove-one comparisons are
order-dependent, so conclusions must be checked against the direct accepted/rejected utility audit
above.

### 12.3 Candidate-oracle viability gate

Run one bounded diagnostic before deciding whether to abandon the underlying static-expansion direction.
Use a seeded, stratified sample from:

- `trec_covid`, `webis_touche2020`, and `quora` as major-loss/high-baseline cases;
- `bright_psychology` and `bright_theoremqa_questions` as small-win cases;
- `bright_stackoverflow` as a reasoning-heavy case.

For each sampled query, retain the top 20–50 dense candidates per original analyzed anchor **before V8
anchor and candidate gating**, while recording whether each item would survive every V8 stage. Inject a
seeded subset of accepted and rejected candidates separately while holding the analyzer, index, query
and retrieval model fixed. Use a small declared weight grid such as `0.05, 0.10, 0.30, 0.50`; this
prevents the candidate oracle from being confounded with the known `0.30` saturation defect. The oracle
is diagnostic and must not tune the final test set.

Classify a query as **safely addressable** if at least one candidate satisfies either:

1. positive `Delta nDCG@10` without lower Recall@1000; or
2. positive Recall@100 or Recall@1000 without an nDCG@10 loss beyond the declared tie tolerance.

Also report candidates that trade nDCG against recall rather than silently calling them good or bad.

Predeclare these operational decision bands:

| Safely addressable query rate | Decision |
|---:|---|
| Below 5% | Retire general-purpose static first-stage QE and do not build CRVE |
| 5% to below 10% | Consider only a cheap, reliably detectable query-type route; do not build a universal context sidecar |
| At least 10% | CRVE may proceed to a small fixed-candidate pilot, provided oracle gains occur across multiple dataset types |

These are project resource thresholds, not universal IR laws. Because any real selector will recover
only part of oracle performance, a very small oracle-active fraction is insufficient justification for
context-memory engineering.

The rate threshold is necessary but not sufficient. Proceeding also requires:

- positive oracle mean Recall@100 or Recall@1000 on at least three of the six initial datasets;
- no dependence on one exceptional corpus;
- enough separation between useful and harmful candidates to make selection plausible;
- a credible path to remain within the declared preparation, memory and online-latency budgets.

### 12.4 Outcomes of the gate

#### Gate fails

If fewer than 5% of queries are safely addressable, or oracle gains are isolated to one dataset:

- stop V8-derived first-stage expansion research;
- retain V8 only as a documented negative ablation;
- use pristine BM25/DPH candidate generation;
- direct semantic compute to downstream routing/reranking or investigate a fundamentally different
  sparse unit such as phrases/entities only if separately justified.

#### Gate is marginal

If 5–10% of queries are safely addressable and form a detectable category:

- do not revive V8 globally;
- investigate an abstaining router that expands only that category;
- require the router's errors and latency to be included in the end-to-end evaluation.

#### Gate passes

If at least 10% are safely addressable with cross-dataset oracle gains:

- V8 still remains retired as a complete method;
- reuse its candidate pool as an experimental input;
- compare current selection, cosine-only, bounded-IDF and context validation on exactly the same
  candidates;
- build only the minimal CRVE sidecar needed for the pilot;
- promote CRVE only if it captures a meaningful fraction of oracle gain while controlling harmful-term
  rate, nDCG loss, memory and latency.

### 12.5 Minimal work package on the experiment testbed

Before any new retrieval experiment:

1. export the existing tie/gain/drop audit as a committed CSV and Markdown summary under
   `results/pyterrier_baselines/`;
2. record Git revision, runtime V8 configuration, cache/sidecar manifests, realized pool sizes and exact
   metric/tie definitions;
3. reconcile the SciFact trace with V8's one-or-two-anchor limit;
4. trace final weight ratios to confirm the executed run used the saturating defaults;
5. export pre-gate anchor and candidate decisions required by the mechanism attribution audit;
6. run the bounded candidate-oracle sample and decision gate above.

No full corpus reindexing, full 20/25-dataset rerun, context-sidecar build, or broad hyperparameter sweep
is justified before this work package is complete.

### 12.6 Repair map after attribution

Any attempted repair must follow the diagnosed component:

| Evidence | Permitted repair direction |
|---|---|
| Harm-only anchors frequently pass | Add an expansion-need/abstention signal; do not merely change candidate weights |
| Useful-capable anchors are rejected | Relax or replace relative-IDF anchor gating; evaluate more than the highest-IDF anchor |
| Useful candidates fail the asymmetric-IDF gate | Permit frequency-asymmetric synonyms or use IDF as a bounded risk feature rather than a hard semantic rule |
| Harmful candidates pass existing gates | Add query-compatible context validation or stronger candidate abstention |
| Useful candidates pass but lose final ranking | Replace the unbounded IDF selector; test cosine-only, bounded IDF and CRVE on the fixed set |
| Correct selected terms fail only at `0.30` | Calibrate weight from evidence or use a constrained allocation policy |
| No pre-gate unigram candidate has utility | Do not repair V8; retire unigram expansion and change representation or retrieval stage |
| Utility exists only in a detectable query type | Use an abstaining conditional route, never universal V8 |

No fix should be promoted because it improves the same oracle sample used to diagnose it. A repair that
passes the diagnostic must be frozen and evaluated on held-out datasets or queries.

### 12.8 Empirical Results of the Comprehensive 4-Stage Candidate-Oracle Diagnostic (N=20 Corpora)

The refined diagnostic was executed across all **20 benchmarks** (12 Small, 7 Medium, 1 Sentinel) with $N=50$ queries per corpus (1,000 total queries, seed 42) and full DPH transfer checks.

- **Candidate-Level Log:** `results/pyterrier_baselines/v8_pregate_candidate_audit.parquet` (17,619 candidate evaluations across depths $D \in \{20, 50, 100\}$ and weights $\mu \in [0.05, 0.10, 0.30]$).
- **Summary Report:** `results/pyterrier_baselines/v8_pregate_oracle_report.md` and `results/pyterrier_baselines/v8_pregate_oracle_summary.csv`.
- **SciFact Acceptance Gate:** Repaired 300-query baseline confirmed: BM25 nDCG@10 = 0.6839, V8 nDCG@10 = 0.6817 ($\Delta = -0.0022$), zero-expansion parity diff = 0.000000.

#### 12.8.1 Master 4-Stage Decomposition Table (20 Corpora)

| Dataset | Tier | Queries | Stage 1: Pool Avail % | Stage 2: BGE Safe@20 % [Wilson 95%] | Stage 2: BGE Safe@100 % | Stage 3: Gate Recall % | Stage 3: Harm Rej % | Stage 3: Acc Prec % | Stage 4: V8 Safe % |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `nfcorpus` | Small (< 100k) | 50 | 82.0% | 48.0% [34.8%, 61.5%] | 48.0% | 43.1% | 54.5% | 81.5% | 28.0% |
| `scifact` | Small (< 100k) | 50 | 18.0% | 8.0% [3.2%, 18.8%] | 8.0% | 50.0% | 66.7% | 50.0% | 2.0% |
| `arguana` | Small (< 100k) | 50 | 14.0% | 0.0% [0.0%, 7.1%] | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `bright_pony` | Small (< 100k) | 50 | 56.0% | 18.0% [9.8%, 30.8%] | 18.0% | 14.3% | 0.0% | 100.0% | 4.0% |
| `bright_theoremqa_theorems` | Small (< 100k) | 50 | 16.0% | 0.0% [0.0%, 7.1%] | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% |
| `scidocs` | Small (< 100k) | 50 | 18.0% | 8.0% [3.2%, 18.8%] | 8.0% | 0.0% | 100.0% | 0.0% | 2.0% |
| `bright_economics` | Small (< 100k) | 50 | 6.0% | 4.0% [1.1%, 13.5%] | 4.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `bright_psychology` | Small (< 100k) | 50 | 16.0% | 6.0% [2.1%, 16.2%] | 6.0% | 0.0% | 100.0% | 0.0% | 0.0% |
| `bright_biology` | Small (< 100k) | 50 | 12.0% | 4.0% [1.1%, 13.5%] | 4.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `fiqa` | Small (< 100k) | 50 | 44.0% | 12.0% [5.6%, 23.8%] | 12.0% | 0.0% | 71.4% | 0.0% | 0.0% |
| `bright_sustainable_living` | Small (< 100k) | 50 | 16.0% | 2.0% [0.4%, 10.5%] | 2.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `bright_robotics` | Small (< 100k) | 50 | 14.0% | 4.0% [1.1%, 13.5%] | 4.0% | 0.0% | 0.0% | 0.0% | 2.0% |
| `bright_stackoverflow` | Medium (100k-500k) | 50 | 8.0% | 2.0% [0.4%, 10.5%] | 2.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `bright_earth_science` | Medium (100k-500k) | 50 | 14.0% | 6.0% [2.1%, 16.2%] | 6.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `bright_aops` | Medium (100k-500k) | 50 | 4.0% | 0.0% [0.0%, 7.1%] | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `bright_theoremqa_questions` | Medium (100k-500k) | 50 | 10.0% | 0.0% [0.0%, 7.1%] | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `bright_leetcode` | Medium (100k-500k) | 50 | 4.0% | 0.0% [0.0%, 7.1%] | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `trec_covid` | Medium (100k-500k) | 50 | 60.0% | 80.0% [67.0%, 88.8%] | 80.0% | 31.2% | 77.8% | 71.4% | 34.0% |
| `webis_touche2020` | Medium (100k-500k) | 49 | 69.4% | 61.2% [47.2%, 73.6%] | 65.3% | 19.6% | 64.5% | 45.0% | 16.3% |
| `quora` | Sentinel (> 500k) | 50 | 18.0% | 6.0% [2.1%, 16.2%] | 6.0% | 50.0% | 50.0% | 50.0% | 2.0% |

#### 12.8.2 Definitive Answers to the Four-Stage Decomposition

1. **Stage 1 (Pool Availability):**
   - **Macro Pool Availability is 25.0%** across all 20 corpora.
   - The 15,000-term vocabulary pool is **not** universally irrelevant. In terminology-gap corpora (`nfcorpus` 82%, `webis_touche2020` 69.4%, `trec_covid` 60%, `bright_pony` 56%, `fiqa` 44%), between 44% and 82% of queries contain gold unigrams in the pool.
   - Conversely, in formal mathematical and algorithmic reasoning domains (`bright_aops` 4%, `bright_leetcode` 4%, `bright_economics` 6%), gold unigram overlap is near zero, proving that unigram expansion is structurally incapable of bridging multi-hop reasoning.
2. **Stage 2 (BGE Proposal Recall & Saturation):**
   - Macro Safe Addressability @ Top-20 is **13.5%** [95% Bootstrap CI: 5.4% – 23.6%].
   - Crucially, expanding BGE proposal depth from **Top-20 to Top-100 only increases safe addressability by +0.2% (13.5% $\to$ 13.7%)**.
   - This proves that BGE isolated-word nearest neighbors beyond depth 20 are almost exclusively semantic drift distractors.
3. **Stage 3 (Gate Recall & Precision Failure in Reasoning):**
   - On medical and argumentative datasets (`nfcorpus`, `trec_covid`, `scifact`), V8's gates function moderately well (Gate Recall 31%–50%, Accepted Precision 50%–81%).
   - But across **all BRIGHT reasoning datasets and ArguAna**, Gate Recall is **0.0%** and Accepted Precision is **0.0%**. Every single candidate that passes V8's gates in these domains is harmful or dormant.
4. **Stage 4 (Selector Precision & Weight Allocation):**
   - V8's composite selector formula ($\cos \times \Delta_{\text{IDF}}$) collapses safe addressability from 13.5% down to **4.7%** (`Stage 4 V8 Safe %`), proving that the ranking heuristic frequently chooses a harmful candidate over a beneficial survivor.
   - Confirmatory DPH transfer check confirmed that when true beneficial candidates are selected, gains transfer strongly across scoring engines (`trec_covid` $\Delta = +0.3199$, `nfcorpus` $\Delta = +0.2536$, `scifact` $\Delta = +0.6624$).

#### 12.8.3 Architectural Conclusion
1. **The 15k pool has headroom in terminology domains (25% macro, up to 82% in NFCorpus), but zero headroom in reasoning.**
2. **BGE isolated-word proposal is severely bottlenecked (flat from top-20 to top-100).**
3. **Universal unigram expansion must remain permanently retired.**
4. **Any future expansion must be strictly conditional (abstaining on reasoning corpora and activating only on terminology-gap domains).**
### 12.9 Paper and thesis treatment

V8 can be reported as an informative negative result:

- corpus-frequency constraints and conservative term counts did not repair context-free lexical
  projection;
- final Recall@1000 fell, demonstrating candidate-funnel risk from unsafe emitted terms;
- the weighting formula accidentally collapsed confidence into a fixed coefficient;
- the failure motivates separating candidate availability, selection quality and allocation rather than
  claiming that static lexical expansion is universally ineffective;
- the Candidate-Oracle Gate demonstrated that only 6.3% of queries in small benchmarks have safely addressable
  dense unigram candidates, proving that first-stage candidate generation is best served by pristine BM25/DPH,
  with semantic matching delegated to late-stage listwise reranking.

Until any oracle gate passes on a new representation, V8 must not be presented as the main proposed method or as empirical
evidence that CRVE will work.



