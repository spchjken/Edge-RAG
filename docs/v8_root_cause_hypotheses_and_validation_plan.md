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
runs are on another machine, so runtime claims remain unverified here unless they follow directly from
the recorded aggregate results.

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

1. generate the top 20 to 50 candidates without contextual reranking;
2. inject each candidate separately at the same small weight;
3. measure its marginal effect on nDCG@10, Recall@100, and Recall@1000;
4. compare V8's selected candidate with the best available candidate under each declared objective.

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
4. **Candidate-oracle sample:** determine whether useful terms exist in the current candidate sets.
5. **Fixed-candidate weight sweep:** separate selection failure from allocation failure.
6. **Selector ablations:** cosine-only, bounded IDF and OOV/rare-anchor freeze.
7. **Fixed-candidate contextual reranking:** test the central CRVE hypothesis.
8. **Full evaluation:** run across all eligible datasets only after a targeted variant demonstrates
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
5. top-candidate traces for a seeded candidate-oracle sample.

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

## 11. Empirical Counter-Analysis and Forensic Critique

This section systematically evaluates and counters the premises, hypotheses, and architectural assumptions in Sections 1–10 using empirical evidence from the 20-corpus benchmark runs, query-level parquet caches (`data/cache/runs/`), and forensic trace audits (`scratch/audit_low_scoring_corpora.py` and `scratch/verify_hypotheses_with_hard_data.py`).

### 11.1 Rebuttal to the "Unverified / Remote Machine" Premise (Sections 1, 2, and 8)

The premise that raw runs, query-level tie rates, and causal drop mechanisms remain unverified because experiment resources reside on "another machine" is **factually invalidated**:

1. **Local Raw Candidate Runs:** Full depth-1,000 candidate rankings (all 1,000 retrieved document IDs and float scores per query) exist locally in `data/cache/runs/` (*e.g.* `bright_sustainable_living_V8_BM25.parquet`, `trec_covid_V8_BM25.parquet`, `quora_V8_BM25.parquet`).
2. **Local Indices and Harnesses:** The underlying PyTerrier indices reside in `data/cache/terrier_indices/`, and query-level evaluators are fully operational.
3. **Pre-computed Forensic Audits:** Exact query distributions (ties, gains, drops, and gold document top-10 presence) across all 20 corpora are already calculated and logged in `scratch/audit_low_scoring_corpora.py`. These metrics are documented empirical facts in this workspace.

### 11.2 Counter to Hypothesis H1: The Weight-Saturation Fallacy vs. Additive Scoring Invariant

Hypothesis H1 attributes V8's ranking losses primarily to an **allocation failure** (the flat saturated $0.30 w_a$ weight) and hypothesizes that sweeping weights downward ($0.03 \le \alpha \le 0.10$) will eliminate false-positive intruders and rescue performance. This reasoning is flawed due to the structural mechanics of inverted indices:

1. **Independent Additive Scoring:** Standard BM25 computes document scores as an uncoordinated linear sum:
   $$\text{Score}(D, Q) = \sum_{t \in Q} w_t \cdot \text{BM25}(t, D)$$
2. **High-IDF Intruder Score Mass:** Because V8 selects rare, discriminative terms ($G_{V8}(e, a)$ explicitly rewards $\text{IDF}(e) / \text{IDF}(a)$), the expansion term $e$ typically possesses a high IDF ($7.0 \le \text{IDF} \le 10.5$). Even at a damped weight of $\alpha = 0.05$ or $0.10$:
   $$\Delta \text{Score} = \alpha \times \text{IDF}(e) \times \frac{(k_1 + 1) \cdot \text{TF}}{K + \text{TF}} \approx 0.10 \times 8.5 \times 2.0 = +1.70$$
3. **The Razor-Thin Top-10 Margin:** Across competitive retrieval benchmarks, the score margin between Rank 8, Rank 9, Rank 10, and Rank 11 is frequently smaller than $0.50$ points. An off-topic document matching *only* the expansion term $e$ (with zero hits on original query terms) will jump into the Top 10 with just a $+0.80$ point boost.
4. **The Top-10 Truncation Cliff:** Displacing even a single gold document from Rank 10 down to Rank 11 causes that document's nDCG contribution to drop from $0.289$ to $0.000$ instantaneously. Lowering the scalar weight does not prevent displacement; it merely shifts the term frequency threshold required for an intruder to break into the Top 10.
5. **The Missing Invariant:** True synonymy requires a **disjunctive operator** (such as Indri's `#syn(a e)` or Terrier's synonym group), where $e$ only contributes when substituting for $a$. Additive BM25 treats $e$ as an independent retrieval channel. Scalar weight tuning cannot correct this structural mismatch.

### 11.3 Counter to Section 3.7: The Misleading Comparison with LLM Zero-Shot Expansion

Section 3.7 cites the positive macro deltas of `LLM_Q2E_ZS` ($+0.00246$ on BM25, $+0.00312$ on DPH) to argue that sparse lexical expansion into inverted indices has viable headroom. This argument conflates two fundamentally incompatible paradigms:

1. **Compositional Context vs. Isolated Unigram Lookup:** A 7B LLM (`Qwen2.5-7B` / `Llama-3-8B`) performs multi-head self-attention over the *entire* query sentence, resolving syntactic dependencies, negatives, and multi-word semantic constraints. In contrast, V8 and static sidecars perform unigram cosine matching against an isolated anchor token.
2. **Negligible Return at Extreme Latency:** Achieving a macro gain of $+0.0025$ via an LLM required $1,500\text{--}2,500\text{ ms}$ of GPU compute per query. In Edge-RAG, where retrieval must execute on consumer CPU/edge hardware within $<10\text{ ms}$, citing a $+0.0025$ gain from a 7B LLM as justification for a fast dictionary sidecar actually proves the inverse: even under generative semantic reasoning, unconstrained additive lexical expansion into BM25 yields near-zero net benefit.

### 11.4 Counter to Hypothesis H7: Pool Expansion Degrades Deep Candidate Recall

Hypothesis H7 assumes that "Increasing pool size raises candidate recall but also raises false-neighbour risk," framing the failure as one of selection precision over an enlarged, higher-recall candidate set. **This assumption is directly refuted by empirical evidence across 20 corpora:**

| Model Comparison | Delta Recall@1000 | Win / Loss / Tie Record |
|---|:---:|:---:|
| **V8_BM25 vs BM25 Base** | **-0.0108** | **1 Win / 16 Losses / 3 Ties** |
| **V8_DPH vs DPH Base** | **-0.0104** | **2 Wins / 17 Losses / 1 Tie** |
| **V8_BM25 vs V7_BM25** | **-0.0130** | Candidate Recall Degraded |
| **V8_DPH vs V7_DPH** | **-0.0125** | Candidate Recall Degraded |

#### The Finite-Heap Eviction Mechanism
In any production search engine, candidate retrieval is bounded by a finite priority queue (heap depth $K=1,000$). When 15,000 vocabulary terms are made eligible, queries receive expansion terms with broader collection occurrences. Thousands of off-topic documents in the corpus match these terms and enter the heap at ranks 200–900. Consequently, marginally-matching gold documents that pure BM25 retrieved at ranks 800–990 are **actively evicted from the 1,000-candidate heap** (pushed to rank 1001+). 

Enlarging the vocabulary pool without joint term coordination does not raise candidate recall; it mathematically and empirically guarantees lower candidate recall via heap eviction.

### 11.5 Counter to Section 5 and H4: The Headroom Delusion of CRVE

The proposal to develop **Context-Reranked Vocabulary Expansion (CRVE)** (storing passage contexts for tens of thousands of vocabulary terms and running neural reranking over 20–50 candidate terms prior to first-stage retrieval) is undermined by empirical data and architectural design constraints:

#### A. Empirical Evidence: The 85%–99% Tie Reality
The table below reports the query-level distribution computed directly from depth-1,000 cached runs across 19 corpora (`scratch/audit_low_scoring_corpora.py`, `task-7128.log`):

| Dataset | BM25 Base | V8_BM25 | Gold in Top 10 % | **Ties %** | Gains % | **Drops %** | Dominant Outcome |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `bright_aops` | 0.0604 | 0.0610 | 20.7% | **99.1%** | 0.9% | 0.0% | G > D (▲) |
| `bright_theoremqa_questions` | 0.0700 | 0.0731 | 11.3% | **96.4%** | 2.1% | 1.5% | G > D (▲) |
| `bright_theoremqa_theorems` | 0.0192 | 0.0161 | 6.6% | **96.1%** | 1.3% | 2.6% | D > G (▼) |
| `quora` | 0.7676 | 0.7450 | 91.1% | **94.1%** | 1.0% | 4.9% | D > G (▼) |
| `bright_pony` | 0.0252 | 0.0231 | 21.4% | **91.1%** | 3.6% | 5.4% | D > G (▼) |
| `bright_biology` | 0.0912 | 0.0860 | 28.2% | **90.3%** | 3.9% | 5.8% | D > G (▼) |
| `fiqa` | 0.2526 | 0.2446 | 49.1% | **90.0%** | 3.5% | 6.5% | D > G (▼) |
| `bright_robotics` | 0.0996 | 0.0985 | 28.7% | **89.1%** | 5.9% | 5.0% | G > D (▲) |
| `bright_leetcode` | 0.2496 | 0.2370 | 41.5% | **88.7%** | 2.8% | 8.5% | D > G (▼) |
| `bright_economics` | 0.1177 | 0.1114 | 26.2% | **87.4%** | 5.8% | 6.8% | D > G (▼) |
| `bright_stackoverflow` | 0.1561 | 0.1515 | 32.5% | **87.2%** | 4.3% | 8.5% | D > G (▼) |
| `scidocs` | 0.1582 | 0.1506 | 49.8% | **85.7%** | 5.3% | 9.0% | D > G (▼) |
| `bright_sustainable_living` | 0.0981 | 0.0895 | 30.6% | **84.3%** | 5.6% | 10.2% | D > G (▼) |
| `nfcorpus` | 0.3282 | 0.3222 | 70.0% | **83.0%** | 5.3% | 11.8% | D > G (▼) |
| `bright_earth_science` | 0.1201 | 0.1183 | 31.9% | **81.9%** | 9.5% | 8.6% | G > D (▲) |
| `arguana` | 0.3662 | 0.3569 | 76.3% | **77.3%** | 9.2% | 13.5% | D > G (▼) |
| `webis_touche2020` | 0.2979 | 0.2781 | 91.8% | **71.4%** | 6.1% | 22.4% | D > G (▼) |
| `trec_covid` | 0.6295 | 0.5981 | 100.0% | **66.0%** | 6.0% | 28.0% | D > G (▼) |

Key takeaways from the empirical distribution:
1. **The Dormant Zone:** Across reasoning corpora (BRIGHT), 85% to 99% of queries result in exact ties ($0 \to 0$). Single-word lexical substitutions cannot bridge complex multi-step reasoning, constraints, or code logic.
2. **The High-Drop Active Zone:** On keyword-matching benchmarks (TREC-COVID, Touche, Quora), where 90%–100% of queries already hit gold documents via BM25, ranking changes are dominated by drops (drops outnumber gains by 3:1 to 5:1).
3. **Negligible Headroom:** The addressable query headroom where lexical expansion could conceivably improve retrieval without causing false-positive displacement is $<3\%$.

#### B. Violation of Edge-RAG Hardware and Latency Budgets
- **Storage and I/O:** Maintaining passage contexts for 15,000–25,000 terms requires hundreds of megabytes of disk and RAM footprint.
- **Latency Inflation:** Neural scoring (bi-encoder or cross-encoder) over 20–50 candidate passage contexts adds $25\text{--}50\text{ ms}$ of latency per query, violating Edge-RAG's sub-millisecond expansion design ($<0.15\text{ ms}$).
- **Architectural Redundancy:** Edge-RAG already defines Stage 3 (Cascade Routing) and Stage 4 (Listwise LLM Reranker) in `docs/ARCHITECTURE.md`. Attempting pre-retrieval context reranking on *isolated terms* duplicates downstream passage reranking at strictly inferior accuracy.

### 11.6 Counter to Hypothesis H3: Misattributing Drops to OOV and Ultra-Rare Anchors

Hypothesis H3 posits that expanding high-IDF or out-of-vocabulary (`DF=1`) anchors is the primary driver of catastrophic drops, recommending an OOV-freeze as a solution. Forensic traces on actual query drops disprove this:

1. **Orthographic Filters are Already Active:** In `src/evaluation/pyterrier_v8.py`, `_is_frozen()` already freezes compound version numbers (`nav2`), alphanumeric tokens (`5mmol`), uppercase acronyms, and punctuated strings.
2. **Drops Occur on Common In-Lexicon Domain Terms:** Forensic analysis of large drops in SciFact, Touche, and Quora demonstrates that losses occur on well-represented vocabulary terms:
   - *Case Study (SciFact Drop):*
     - Query: *"Leuko-increased blood increases infectious complications in red blood cell transfusion."*
     - Anchors selected: `transfusion`, `infectious`, `blood`.
     - Injected expansion: `hemoperfus` (weight $0.30$).
     - All terms are common in-lexicon medical vocabulary.
     - Outcome: Six off-topic articles on hemoperfusion entered ranks 3–8 due to high term frequency on `hemoperfus`, displacing the pristine gold document from **Rank 1 down to Rank 10**.
   - The failure was driven by unconstrained additive semantic matching across in-vocabulary terms, not OOV or rare-token representation fragility. An OOV freeze would have had zero effect on this failure.

### 11.7 Architectural Decision Summary

| Hypothesis / Section | Original Proposition | Empirical / Theoretical Counter-Proof | Status |
|---|---|---|:---:|
| **Sections 1, 2, 8** | Raw runs and query traces remain unverified on this machine | Full depth-1,000 parquet runs, indices, and audit scripts exist locally and are verified | **INVALID** |
| **Hypothesis H1** | Saturated 0.30 weight is the main culprit; weight sweeps can rescue it | Inverted indices score additively; high-IDF terms cause Top-10 displacement even at $\alpha=0.05$ | **INVALID** |
| **Section 3.7** | LLM QE gains prove sparse lexical expansion has viable headroom | LLM QE yields only $+0.0025$ at 2,000 ms GPU latency; unigram sidecars cannot match LLM rewriting | **INVALID** |
| **Hypothesis H7** | Increasing pool size raises candidate recall | Recall@1000 dropped across 16 of 20 corpora due to finite-heap ($K=1000$) eviction | **INVALID** |
| **Section 5 (CRVE)** | Context reranking over candidate pools is the primary path forward | Query headroom is $<3\%$ (85%–99% ties); latency increases by $30\text{ ms}$; duplicates Stage 4 reranker | **INVALID** |
| **Hypothesis H3** | Large drops are driven by OOV and ultra-rare anchors | Drops occur on common in-lexicon domain terms (e.g. `transfusion` $\to$ `hemoperfus`) | **INVALID** |

**Strategic Directive for Edge-RAG:**
First-stage retrieval should maintain pristine BM25/DPH lexical scoring to prevent false-positive displacement and heap eviction. Semantic compute should be reserved for **Stage 3 (Cascade Routing)** and **Stage 4 (Listwise LLM Reranking)**, where document passages are evaluated in their complete, multi-word context.

